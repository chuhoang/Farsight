"""plan_csci C: how much face (KP-RPE) and body correct each other, on the current stores (body = whatever was
extracted: AIM before eval.regait --mod body, CSCI after). Top-1 under the CAL rule (gallery entries with the
query's pid AND camera are ignored); queries with no valid positive are dropped. A probe without a face counts as
a face miss.

Reported per set (CCVID test clean / degraded, MEVID test):
  face_fixes_body = % of queries body gets wrong and face gets right   (threshold: >= 10%)
  body_fixes_face = % of queries face gets wrong and body gets right   (threshold: >= 10%)
  (also as a share of the other modality's misses), oracle = either right, and the Pearson correlation of face and
  body genuine-pair scores (lower = more complementary; recorded, no threshold).

    bash run.sh eval.csci_complement [--tag aim]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.main_protocol import DEGRADED, Set, load

F = Path.home() / "datasets/feats"


def top1_right(S, qid, gid, qcam, gcam):
    """S (P, G) with NaN = missing -> bool (P,) top-1 correct, bool (P,) query has a valid positive."""
    junk = (qid[:, None] == gid[None]) & (qcam[:, None] == gcam[None])
    s = np.where(junk | np.isnan(S), -np.inf, S)
    valid = ((qid[:, None] == gid[None]) & ~junk).any(1)
    right = (gid[s.argmax(1)] == qid) & np.isfinite(s.max(1))
    return right, valid


def complement(st):
    qcam = np.array([m["camid"] for m in st.pm])
    gcam = np.array([m["camid"] for m in st.gm])
    S = st.S["full"]
    face, valid = top1_right(S[..., 0], st.qid, st.gid, qcam, gcam)
    body, _ = top1_right(S[..., 2], st.qid, st.gid, qcam, gcam)
    face, body = face[valid], body[valid]
    gen = (st.qid[:, None] == st.gid[None]) & ~np.isnan(S[..., 0]) & ~np.isnan(S[..., 2])
    pct = lambda x: round(100 * float(np.mean(x)), 2)  # noqa: E731
    return {"n_query": int(valid.sum()), "dropped_no_positive": int((~valid).sum()),
            "face_R1": pct(face), "body_R1": pct(body), "oracle_R1": pct(face | body),
            "face_fixes_body": pct(~body & face), "body_fixes_face": pct(~face & body),
            "face_fixes_body_of_body_misses": pct(face[~body]) if (~body).any() else None,
            "body_fixes_face_of_face_misses": pct(body[~face]) if (~face).any() else None,
            "genuine_score_corr": round(float(np.corrcoef(S[..., 0][gen], S[..., 2][gen])[0, 1]), 3),
            "n_genuine_pairs": int(gen.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="csci", help="label of the body encoder in the stores (output file suffix)")
    a = ap.parse_args()
    test = set(json.loads(Path("splits/ccvid_qme.json").read_text())["test"])
    sets = {"ccvid_test": ("ccvid_query", test, "ccvid_gallery"),
            "ccvid_test_degraded": ("ccvid_query_degraded", test, "ccvid_gallery"),
            "mevid_test": ("mevid_query", None, "mevid_gallery")}
    if not DEGRADED:
        del sets["ccvid_test_degraded"]
    res = {}
    for name, (pq, ids, pg) in sets.items():
        res[name] = complement(Set(*load(F, pq, ids), *load(F, pg, ids)))
        r = res[name]
        print(f"{name:20s} face {r['face_R1']:5.1f} body {r['body_R1']:5.1f} oracle {r['oracle_R1']:5.1f} | "
              f"face fixes body {r['face_fixes_body']:5.1f}%  body fixes face {r['body_fixes_face']:5.1f}%  "
              f"corr {r['genuine_score_corr']:.3f}", flush=True)
    Path(f"eval/results/csci_complement_{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

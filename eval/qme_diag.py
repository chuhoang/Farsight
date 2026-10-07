"""Per-sample diagnosis of QME's face-quality decisions (analysis only: uses test labels, never selects anything).

For each probe of a test set (CAL rule: same-identity same-camera gallery entries ignored; top-1 correctness):
face / gait / body / QME right or wrong, the QE face weight W, and where QME loses to / beats body-only.
Oracle runs (upper bounds of a perfect face-quality decision, same trained QME heads):
  W oracle        W = 1 if the face is top-1 right else 0
  face gate oracle face score removed (missing) for probes whose face is top-1 wrong
  W = 0 / W = 1   constant QE weight
  no face / no gait / body only   modality removed for every probe
If a perfect W barely helps but the face-gate oracle does, the bottleneck is how the experts use the face score,
not the QE estimate.

    FARSIGHT_DEGRADED=0 bash run.sh eval.qme_diag [--tag ""]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.main_protocol import Set, cal_metrics, load
from eval.prcc import aim_metrics
from eval.train_2b import DEV, qe_weights
from farsight.modules.m4_fusion.qme.model import load_head
from farsight.modules.m4_fusion.qme.train import fuse_np

F = Path.home() / "datasets/feats"
W_DIR = Path("weights/m4_fusion/qme")
MODS = ("face", "gait", "body")


def top1(S, st, qcam, gcam):
    """CAL-rule top-1 correctness per probe; valid = probe has an other-camera positive."""
    same = st.qid[:, None] == st.gid[None]
    junk = same & (qcam[:, None] == gcam[None])
    s = np.where(junk | np.isnan(S), -np.inf, S)
    valid = (same & ~junk).any(1)
    return (st.gid[s.argmax(1)] == st.qid) & np.isfinite(s.max(1)), valid


def heads(d, tag):
    out = []
    for s in range(3):
        p = W_DIR / f"qme_kprpe_2b_csci_{d}{tag}_s{s}.pth"
        out.append(load_head(p)[0].to(DEV))
    return out


def run(d, pq, ids, pg, tag, m):
    P, Pm = load(F, pq, ids)
    G, Gm = load(F, pg, ids)
    st = Set(P, Pm, G, Gm)
    S = st.S["full"]
    qcam = np.array([x["camid"] for x in Pm])
    gcam = np.array([x["camid"] for x in Gm])
    hf = ~np.isnan(S[..., 0]).all(1)
    right = {mod: top1(S[..., k], st, qcam, gcam)[0] for k, mod in enumerate(MODS)}
    valid = top1(S[..., 2], st, qcam, gcam)[1]
    hs = heads(d, tag)
    Ws = [qe_weights(h.qe, P) for h in hs]

    def qme(S_, W_=None):
        """mean over seeds of (per-probe top-1 right, GR_R1 CAL)"""
        r, g = [], []
        for h, w in zip(hs, Ws):
            fz = fuse_np(h, S_, w if W_ is None else W_, DEV)
            r.append(top1(fz, st, qcam, gcam)[0])
            g.append(cal_metrics(m, fz, Pm, Gm)["GR_R1"])
        return np.mean(r, 0) >= 0.5, float(np.mean(g))

    q_right, q_r1 = qme(S)
    W = np.mean(Ws, 0)
    v = valid
    pct = lambda x: round(100 * float(np.mean(x)), 1) if np.size(x) else None  # noqa: E731
    res = {"n_probe": int(len(P)), "n_valid": int(v.sum()), "face_probes": int((hf & v).sum())}
    res["R1_CAL"] = {"face": pct(right["face"][v]), "gait": pct(right["gait"][v]), "body": pct(right["body"][v]),
                     "QME": round(100 * q_r1, 1)}
    fv = hf & v
    res["face_probes_only"] = {k: pct(right[k][fv]) for k in MODS} | {"QME": pct(q_right[fv])}
    res["noface_probes_only"] = {k: pct(right[k][~hf & v]) for k in ("gait", "body")} | {"QME": pct(q_right[~hf & v])}
    # QME vs body, per probe
    lost, won = v & right["body"] & ~q_right, v & ~right["body"] & q_right
    res["QME_vs_body"] = {"lost (body right, QME wrong)": int(lost.sum()), "won (body wrong, QME right)": int(won.sum())}
    res["lost_cases"] = {"with face": int((lost & hf).sum()), "face wrong": int((lost & hf & ~right["face"]).sum()),
                         "face right": int((lost & hf & right["face"]).sum()), "no face": int((lost & ~hf).sum()),
                         "gait wrong": int((lost & ~right["gait"]).sum()),
                         "mean W (face lost cases)": round(float(W[lost & hf].mean()), 3) if (lost & hf).any() else None}
    res["won_cases"] = {"with face": int((won & hf).sum()), "face right": int((won & hf & right["face"]).sum()),
                        "gait right": int((won & right["gait"]).sum())}
    # QE quality of the face decision on face probes
    fr = right["face"][fv]
    w = W[fv]
    auc = float((w[fr][:, None] > w[~fr][None]).mean() + 0.5 * (w[fr][:, None] == w[~fr][None]).mean()) \
        if fr.any() and (~fr).any() else None
    res["QE"] = {"W_mean_face_right": round(float(w[fr].mean()), 3) if fr.any() else None,
                 "W_mean_face_wrong": round(float(w[~fr].mean()), 3) if (~fr).any() else None,
                 "AUC_W_vs_face_right": round(auc, 3) if auc is not None else None,
                 "face_wrong_with_W>0.5_%": pct(w[~fr] > 0.5) if (~fr).any() else None,
                 "face_right_with_W<0.5_%": pct(w[fr] < 0.5) if fr.any() else None}
    # oracles (same heads)
    o = {}
    Wor = np.where(hf, right["face"].astype(np.float32), 0.0)
    o["W oracle"] = qme(S, Wor)[1]
    Sg = S.copy()
    Sg[hf & ~right["face"], :, 0] = np.nan
    o["face gate oracle"] = qme(Sg)[1]
    o["W = 0"] = qme(S, np.zeros(len(P), np.float32))[1]
    o["W = 1"] = qme(S, np.where(hf, 1.0, 0.0).astype(np.float32))[1]
    for name, drop in (("no face", (0,)), ("no gait", (1,)), ("body only", (0, 1))):
        S2 = S.copy()
        S2[..., list(drop)] = np.nan
        o[name] = qme(S2)[1]
    res["oracle_GR_R1"] = {k: round(100 * x, 1) for k, x in o.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default="all", choices=["mevid", "ccvid", "all"])
    a = ap.parse_args()
    m = aim_metrics()
    ct = set(json.loads(Path("splits/ccvid_qme.json").read_text())["test"])
    out = {}
    for d, (pq, ids, pg) in {"mevid": ("mevid_query", None, "mevid_gallery"),
                              "ccvid": ("ccvid_query", ct, "ccvid_gallery")}.items():
        if a.dataset not in ("all", d):
            continue
        out[d] = run(d, pq, ids, pg, a.tag, m)
        print(f"\n### {d} test (QME-{d}{a.tag})")
        for k, v in out[d].items():
            print(f"  {k}: {v}")
    Path(f"eval/results/qme_diag_{a.dataset}{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

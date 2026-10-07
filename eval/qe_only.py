"""QE without QME: fusion v1 (z-score weighted mean) whose face weight is multiplied by the QE weight W per probe.

  fused = (W*w_f*z_face + w_g*z_gait + w_b*z_body) / (W*w_f + w_g + w_b)      (missing modality -> dropped)

Per dataset: z-score impostor stats + (w_f, w_g, w_b) grid fitted on its val (camera/clothes-split held-out train IDs,
selection score rank-1 + TAR@1%FAR - FNIR@1%FPIR over full + noface stacked, as eval.main_protocol.fit_v1), tau @1%
FPIR on val; test = whole official test. QE = the QE carried by a trained QME checkpoint (no QME expert used).
Rows: fusion v1 (no W), v1 x QE W (QE of each listed checkpoint), v1 x oracle W (analysis only: W = 1 if the face is
CAL top-1 right else 0, uses test labels), body only.

    FARSIGHT_DEGRADED=0 bash run.sh eval.qe_only
"""
import itertools
import json
from pathlib import Path

import numpy as np

from eval.main_protocol import GRID, VAL_SPLIT, Set, condition, load, score_all, val_pairs
from eval.metrics import evaluate, fnir_at_fpir
from eval.prcc import aim_metrics
from eval.qme_diag import top1
from eval.train_2b import qe_weights
from farsight.modules.m4_fusion.qme.model import load_head
from farsight.modules.m4_fusion.zscore.model import ZScoreFusion

F = Path.home() / "datasets/feats"
W_DIR = Path("weights/m4_fusion/qme")
CONDS = ("full", "noface")
QES = {"mevid": {"QE (orig)": "qme_kprpe_2b_csci_mevid_s0.pth", "QE (CAL labels, 100 steps)": "qme_kprpe_2b_csci_mevid_cam_cal_qe100_s0.pth"},
       "ccvid": {"QE (orig)": "qme_kprpe_2b_csci_ccvid_s0.pth"}}


def face_w(qe, probes, cond):
    return qe_weights(qe, condition(probes, cond)) if qe is not None else None


FIXED_W = None   # set by main(): (w_f, w_g, w_b) fixed beforehand -> no grid (val too easy: grid picks face = 0)


def fit(val, Wv):
    """z stats + weight grid on val (full + noface stacked). Wv: {cond: (P,) face weight} or None (plain v1)."""
    S = np.concatenate([val.S[c] for c in CONDS])
    q = None if Wv is None else np.concatenate([np.c_[Wv[c], np.ones((len(Wv[c]), 2))] for c in CONDS])
    qid = np.concatenate([val.qid] * len(CONDS))
    z = ZScoreFusion(stats="__none__.json").fit(val.S["full"], val.qid, val.gid)
    z.use_quality = Wv is not None
    best = None
    for w in ([FIXED_W] if FIXED_W else itertools.product(GRID, repeat=3)):
        if any(w):
            z.w = np.array(w, np.float32)
            r = evaluate(z.fuse(S, q), qid, val.gid)
            v = r["rank1"] + r["tar@0.01far"] - r["fnir@0.01fpir"]
            if best is None or v > best[0]:
                best = (v, z.w)
    z.w = best[1]
    keep = val.open_masks(seed0=1000)[0]
    z.tau = fnir_at_fpir(z.fuse(S, q)[:, keep], qid, val.gid[keep])[1e-2][1]
    return z, round(float(best[0]), 4)


def main():
    global FIXED_W
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed_w", type=float, nargs=3, help="face gait body weights fixed beforehand (no val grid)")
    a = ap.parse_args()
    FIXED_W = tuple(a.fixed_w) if a.fixed_w else None
    m = aim_metrics()
    sp = {d: json.loads(Path(f"splits/{d}_qme.json").read_text()) for d in ("ccvid", "mevid")}
    tests = {"mevid": ("mevid_query", None, "mevid_gallery"), "ccvid": ("ccvid_query", set(sp["ccvid"]["test"]), "ccvid_gallery")}
    out = {}
    for d, (pq, ids, pg) in tests.items():
        vP, vPm, vG, vGm = val_pairs(*load(F, f"{d}_train", set(sp[d]["val"])), VAL_SPLIT[d])
        val = Set(vP, vPm, vG, vGm)
        P, Pm = load(F, pq, ids)
        G, Gm = load(F, pg, ids)
        st = Set(P, Pm, G, Gm)
        qcam, gcam = np.array([x["camid"] for x in Pm]), np.array([x["camid"] for x in Gm])
        right = top1(st.S["full"][..., 0], st, qcam, gcam)[0]
        rows, fits = {}, {}
        variants = {"fusion v1 (no W)": None, **{f"v1 x {k}": v for k, v in QES[d].items()}, "v1 x oracle W": "oracle"}
        for name, src in variants.items():
            if src is None:
                Wv = Wt = None
            elif src == "oracle":
                Wv = None   # weights fitted as plain v1; oracle W applied on test only
                Wt = {c: np.where(right, 1.0, 0.0).astype(np.float32) * (np.array([t["face"] is not None for t in condition(P, c)])) for c in CONDS}
            else:
                qe = load_head(W_DIR / src)[0].qe.to("cuda")
                Wv = {c: face_w(qe, vP, c) for c in CONDS}
                Wt = {c: face_w(qe, P, c) for c in CONDS}
            z, sel = fit(val, Wv)
            if src == "oracle":
                z.use_quality = True
            fits[name] = {"w": dict(zip(("face", "gait", "body"), z.w.tolist())), "val_select": sel}
            rows[name] = {}
            for c in CONDS:
                q = None if Wt is None else np.c_[Wt[c], np.ones((len(P), 2))]
                rows[name][c] = score_all(st, z.fuse(st.S[c], q), m, tau=z.tau)
        rows["body only"] = {c: score_all(st, st.S[c][..., 2], m) for c in CONDS}
        out[d] = {"fits": fits, "test": rows}
        print(f"\n## {d} test   General R1/mAP | CC R1/mAP | SC R1/mAP   (noface General R1)   fitted w (face, gait, body)")
        for name, r in rows.items():
            f, nf = r["full"], r["noface"]
            w = fits.get(name, {}).get("w")
            print(f"{name:34s} {100*f['GR_R1']:5.1f} {100*f['GR_mAP']:5.1f} | {100*f['CC_R1']:5.1f} {100*f['CC_mAP']:5.1f} | "
                  f"{100*f['SC_R1']:5.1f} {100*f['SC_mAP']:5.1f}   ({100*nf['GR_R1']:5.1f})   {w if w else ''}")
    Path(f"eval/results/qe_only{'_w' + '_'.join(f'{x:g}' for x in FIXED_W) if FIXED_W else ''}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

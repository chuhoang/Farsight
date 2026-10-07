"""Route 2B (plan M4 2B.2-2B.4): QE on KP-RPE inter_feat, then QME on score matrices (QE frozen), vs fusion v1.

One QME per dataset (--dataset ccvid | mevid), as in the QME paper:
  train = the dataset's train IDs (eval.main_protocol.train_pairs; CCVID probes mixed with degraded copies p=0.3 when
          FARSIGHT_DEGRADED=1), val = IDs held out of its train IDs (val_pairs: camera-split, so val rank-1 follows the
          CAL rule of test) -> QE, QME model selection and tau; test = the dataset's whole official test, conditions
          full / noface / short exactly as eval.main_protocol. --dataset both = one QME on both (old D4 setting).

    bash run.sh eval.train_2b --dataset ccvid --v1 eval/results/main_protocol_csci.json --tag _csci_ccvid_s0
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from eval.main_protocol import CONDS, DEGRADED, Set, condition, export_qme, load, score_all, train_pairs, val_pairs, VAL_SPLIT
from eval.metrics import fnir_at_fpir
from eval.prcc import aim_metrics
from farsight.modules.m4_fusion.qe.train import labelled, train as train_qe
from farsight.modules.m4_fusion.qme.train import fuse_np, train as train_qme

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = Path("weights/m4_fusion")


def mix(ts, meta, deg, p, seed):
    """Replace each probe by its degraded copy (same tracklet id) with probability p."""
    rng = np.random.default_rng(seed)
    out = [deg[t["subject_or_track_id"]] if t["subject_or_track_id"] in deg and rng.random() < p else t for t in ts]
    return out, meta, sum(o is not t for o, t in zip(out, ts))


def qe_weights(net, templates):
    """Probe face weight W = sigmoid(QE(face inter_feat)); 0 when the probe has no face."""
    X = [t["face"]["inter_feat"] if t.get("face") is not None else None for t in templates]
    w = np.zeros(len(X), np.float32)
    ok = [i for i, x in enumerate(X) if x is not None]
    if ok:
        with torch.no_grad():
            w[ok] = torch.sigmoid(net(torch.tensor(np.stack([X[i] for i in ok]), device=DEV))).cpu().numpy().ravel()
    return w


def auc(pos, neg):
    """P(score of a clean face > score of a degraded face)."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def scores_file(F, name):
    with h5py.File(Path(F, f"scores_{name}.h5"), "r") as f:
        S = np.stack([f[f"{m}/score_mat"][()] for m in ("face", "gait", "body")], -1).astype(np.float32)
        return S, f["q_pids"][()].astype(np.int64), f["g_pids"][()].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=str(Path.home() / "datasets/feats"))
    ap.add_argument("--p_degraded", type=float, default=0.3)
    ap.add_argument("--qe_steps", type=int, default=6000)
    ap.add_argument("--qme_steps", type=int, default=1500)
    ap.add_argument("--qe_wd", type=float, default=0.01)
    ap.add_argument("--out", default="eval/results/qme_2b.json")
    ap.add_argument("--dataset", default="ccvid", choices=["ccvid", "mevid", "both"],
                    help="train / select / test QME on this dataset only (both = one QME for CCVID + MEVID)")
    ap.add_argument("--mevid", action="store_true", help="legacy: same as --dataset both")
    ap.add_argument("--tag", default="", help="suffix for weight files, e.g. _cm")
    ap.add_argument("--v1", default="eval/results/main_protocol.json", help="eval.main_protocol output to compare with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--body_noise", type=float, default=0.0,
                    help="k_max of the body-score noise augmentation in QME training (qme.train body_noise)")
    ap.add_argument("--face_gate", type=float, default=0.0, help="QME: QE weight < t -> face treated as missing")
    ap.add_argument("--face_scale", action="store_true", help="QME: normalised face score x QE weight in the experts")
    ap.add_argument("--pairs", default=None, choices=["alt", "cam"],
                    help="train probe/gallery pairs: alt = train_pairs, cam = camera-split (default: cam iff --qe_label cal)")
    ap.add_argument("--qe_label", default="plain", choices=["plain", "cal"],
                    help="cal: QE pseudo-labels ignore same-identity same-camera gallery entries (CAL rule of test) and "
                         "train probe/gallery pairs are camera-split (val_pairs) so every probe has such a positive")
    a = ap.parse_args()
    F, m = a.feats, aim_metrics()
    names = ["ccvid", "mevid"] if a.mevid or a.dataset == "both" else [a.dataset]
    sp = {d: json.loads(Path(f"splits/{d}_qme.json").read_text()) for d in ("ccvid", "mevid")}
    ct = set(sp["ccvid"]["test"])
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    res = {"datasets": names, "data": {}}

    # ---- train / val pairs per dataset: train = its train IDs, val = IDs held out of them (camera-split pairs)
    tr, va = {}, {}
    for d in names:
        cam = a.pairs == "cam" if a.pairs else a.qe_label == "cal"
        pair = (lambda t, mm: val_pairs(t, mm, "camid")) if cam else train_pairs
        P, Pm, G, Gm = pair(*load(F, f"{d}_train", set(sp[d]["train"])))
        n_deg = 0
        if d == "ccvid" and DEGRADED:   # FARSIGHT_DEGRADED=0: degraded stores not re-extracted
            deg_tr = {t["subject_or_track_id"]: t for t in load(F, "ccvid_train_degraded")[0]}
            P, Pm, n_deg = mix(P, Pm, deg_tr, a.p_degraded, 0)
        tr[d], va[d] = (P, Pm, G, Gm), val_pairs(*load(F, f"{d}_train", set(sp[d]["val"])), VAL_SPLIT[d])
        export_qme(F, F, {f"{d}_qme_train": tr[d], f"{d}_qme_val": va[d]})
        res["data"][d] = {"train_probes": len(P), "train_degraded": n_deg, "val_probes": len(va[d][0])}

    # ---- stage 1: QE on face inter_feat with pseudo-quality labels (encoders frozen), faces of these datasets only
    parts = []
    for d in names:
        with h5py.File(Path(F, f"scores_{d}_qme_train.h5"), "r") as f:
            cams = (f["q_camids"][()], f["g_camids"][()]) if a.qe_label == "cal" else (None, None)
            parts.append((f["face/inter_feat"][()], f["face/score_mat"][()], f["q_pids"][()], f["g_pids"][()], 3, *cams))
    qe = train_qe(*parts[0][:4], rank_threshold=3, q_cams=parts[0][5], g_cams=parts[0][6], steps=a.qe_steps, wd=a.qe_wd, device=DEV,
                  log_every=1000, extra=[labelled(*x) for x in parts[1:]])
    res["qe_label"] = a.qe_label
    W.joinpath("qe").mkdir(parents=True, exist_ok=True)
    qe_path = W / f"qe/qe_kprpe_2b{a.tag}.pth"
    torch.save({"model_state_dict": qe.cpu().state_dict()}, qe_path)
    qe.to(DEV)

    # W distribution check (plan 2B.3): clean vs degraded faces of the same CCVID test probes
    both = []
    if "ccvid" in names and DEGRADED:
        clean = {t["subject_or_track_id"]: t for t in load(F, "ccvid_query", ct)[0]}
        degr = {t["subject_or_track_id"]: t for t in load(F, "ccvid_query_degraded", ct)[0]}
        both = [k for k in degr if clean.get(k, {}).get("face") is not None and degr[k].get("face") is not None]
    if both:
        wc, wd = qe_weights(qe, [clean[k] for k in both]), qe_weights(qe, [degr[k] for k in both])
        res["qe_W_check"] = {"pairs": len(both), "clean_mean": float(wc.mean()), "degraded_mean": float(wd.mean()),
                             "auc_clean_gt_degraded": auc(wc, wd)}
        print("QE W check:", json.dumps(res["qe_W_check"]), flush=True)

    # ---- stage 2: QME on score matrices, QE frozen; model selection on the val set(s)
    T, V = [], []
    for d in names:
        S, q, g = scores_file(F, f"{d}_qme_train")
        T.append((S, qe_weights(qe, tr[d][0]), q, g))
        Sv, qv, gv = scores_file(F, f"{d}_qme_val")
        V.append((Sv, qe_weights(qe, va[d][0]), qv, gv))
    head, best = train_qme(*map(list, zip(*T)), val=V, steps=a.qme_steps, device=DEV, qe_state=str(qe_path), seed=a.seed,
                           body_noise=a.body_noise, face_gate=a.face_gate, face_scale=a.face_scale)
    Sv, wv, qv, gv = V[0]   # operating point (tau @ 1% FPIR) on the first dataset's val
    keep = Set(*va[names[0]]).open_masks(seed0=1000)[0]
    tau = fnir_at_fpir(fuse_np(head, Sv, wv, DEV)[:, keep], qv, gv[keep])[1e-2][1]
    W.joinpath("qme").mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": head.cpu().state_dict(), "tau": float(tau),
                "head_cfg": {"face_gate": a.face_gate, "face_scale": a.face_scale}}, W / f"qme/qme_kprpe_2b{a.tag}.pth")
    head.to(DEV)
    res["qme_val_best"] = {k: float(v) for k, v in best.items()} if best else None
    res["config"] = {"qe_steps": a.qe_steps, "qe_wd": a.qe_wd, "body_noise": a.body_noise, "face_gate": a.face_gate, "face_scale": a.face_scale,
                     "pairs": a.pairs or ("cam" if a.qe_label == "cal" else "alt"), "qe_label": a.qe_label}

    # ---- test: the whole official test of each dataset, conditions as eval.main_protocol
    v1 = json.loads(Path(a.v1).read_text())
    sets = {}
    if "ccvid" in names:
        sets["ccvid_test"] = ("ccvid_query", ct, "ccvid_gallery")
        if DEGRADED:
            sets["ccvid_test_degraded"] = ("ccvid_query_degraded", ct, "ccvid_gallery")
    if "mevid" in names:
        sets["mevid_test"] = ("mevid_query", None, "mevid_gallery")
    for name, (pq, ids, pg) in sets.items():
        probes, pm = load(F, pq, ids)
        st = Set(probes, pm, *load(F, pg, ids))
        res[name] = {}
        for c in CONDS:
            wt = qe_weights(qe, condition(probes, c))
            r = score_all(st, fuse_np(head, st.S[c], wt, DEV), m, tau=tau)
            res[name][c] = {"QME 2B": r, "fusion v1": v1[name][c]["fusion v1"],
                            "best single": max((v1[name][c][f"{mm} only"] for mm in ("face", "gait", "body")),
                                               key=lambda x: x["GR_R1"])}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\nGR-R1 (full / noface / short)")
    for name in sets:
        for row in ("best single", "fusion v1", "QME 2B"):
            print(f"{name:20s} {row:12s}", " / ".join(f"{100 * res[name][c][row]['GR_R1']:5.1f}" for c in CONDS))


if __name__ == "__main__":
    main()

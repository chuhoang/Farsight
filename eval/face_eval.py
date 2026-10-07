"""Face-only evaluation from the face crop cache (train_m3 A1-A4, C4 validation, E1-E2), any KP-RPE checkpoint.

Templates = KP-RPE on the cached DFA-aligned crops + the same aggregation as the encoder (kprpe.model.aggregate),
so the original checkpoint reproduces the stores' "face only" numbers. Metrics: CAL rule (eval.main_protocol
cal_metrics, probes without a face count as misses) + TAR@FAR on all pairs; per track-IPD bin (median eye distance of
the kept frames, native pixels; degraded sets: / down factor).

    bash run.sh eval.face_eval [--ckpt weights/.../model.pt] [--out eval/results/face_eval.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.face_cache import read
from eval.main_protocol import cal_metrics
from eval.metrics import tar_at_far
from eval.prcc import aim_metrics
from farsight.modules.m3_encode.face.kprpe.model import aggregate

BINS = ((0, 10), (10, 20), (20, 40), (40, 1e9))
SPLITS = {d: json.loads(Path(f"splits/{d}_qme.json").read_text()) for d in ("ccvid", "mevid")}
# name -> (probe cache, gallery cache, ids (split key) or None, IPD divisor, val split key of eval.main_protocol)
TESTS = {"ccvid_test": ("ccvid_query", "ccvid_gallery", None, 1),
         "ccvid_test_degraded": ("ccvid_query_degraded_full", "ccvid_gallery", None, 4),
         "mevid_test": ("mevid_query", "mevid_gallery", None, 1)}
VALS = {"ccvid_val": ("ccvid_train", "clothes", 1), "ccvid_val_degraded": ("ccvid_train_degraded_val", "clothes", 4),
        "mevid_val": ("mevid_train", "camid", 1)}


def load_net(ckpt=None, device="cuda"):
    """KP-RPE net (+ its config) without the aligner."""
    from farsight.modules.m3_encode.face.kprpe.model import KPRPE
    k = KPRPE.__new__(KPRPE)
    KPRPE.__init__(k, device=device, aligner=object(), **({"checkpoint": str(ckpt)} if ckpt else {}))
    return k.net, k.cfg


@torch.no_grad()
def embed(net, img, ldmk, bs=32):   # pure-torch KP-RPE (no rpe_ops ext): ~3.5 GB at 256
    dev = next(net.parameters()).device
    out = []
    for i in range(0, len(img), bs):
        x = torch.from_numpy(img[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        out.append(net(x, torch.from_numpy(ldmk[i:i + bs]).to(dev)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 512), np.float32)


def templates(net, cfg, cache, prefix=""):
    """{tid: (feat (512,) or None, track IPD or nan)} from cache arrays <prefix>img / ldmk / score / ipd."""
    tids = list(cache)
    n = [len(cache[t][prefix + "img"]) if prefix + "img" in cache[t] else 0 for t in tids]
    cat = lambda k: np.concatenate([cache[t][prefix + k] for t, c in zip(tids, n) if c])  # noqa: E731
    f = embed(net, cat("img"), cat("ldmk")) if sum(n) else np.zeros((0, 512), np.float32)
    sc = cat("score") if sum(n) else np.zeros(0)
    ipd = np.concatenate([cache[t]["ab_ipd" if prefix else "ipd"] for t, c in zip(tids, n) if c]) if sum(n) else np.zeros(0)
    out, s = {}, 0
    for t, c in zip(tids, n):
        a = aggregate(f[s:s + c], None, sc[s:s + c], cfg) if c else None
        out[t] = (None, np.nan) if a is None else (a["feat"], float(np.median(ipd[s:s + c][a["keep"]])))
        s += c
    return out


def sim(Tq, qt, Tg, gt):
    G = np.stack([Tg[t][0] if Tg[t][0] is not None else np.full(512, np.nan, np.float32) for t in gt])
    Q = np.stack([Tq[t][0] if Tq[t][0] is not None else np.full(512, np.nan, np.float32) for t in qt])
    return Q @ G.T


def metrics(m, S, qm, gm):
    r = {k: round(100 * float(v), 2) for k, v in cal_metrics(m, S, qm, gm).items()}
    qid, gid = np.array([x["pid"] for x in qm]), np.array([x["pid"] for x in gm])
    ok = ~np.isnan(S).all(1)
    tar = tar_at_far(np.nan_to_num(S[ok], nan=-1), qid[ok], gid, fars=(1e-3, 1e-2)) if ok.any() else {}
    return r | {f"TAR@{f:g}FAR (face probes)": round(100 * v, 2) for f, v in tar.items()} | \
        {"n_probe": int(len(S)), "face_coverage": round(100 * float(ok.mean()), 1)}


def by_ipd(m, S, Tq, qt, qm, gm, div):
    ipd = np.array([Tq[t][1] for t in qt]) / div
    out = {"no face": int(np.isnan(ipd).sum())}
    for lo, hi in BINS:
        k = (ipd >= lo) & (ipd < hi)
        if k.sum():
            out[f"{lo:g}-{hi:g}px" if hi < 1e9 else f">{lo:g}px"] = metrics(m, S[k], [qm[i] for i in np.where(k)[0]], gm)
    return out


def meta(cache):
    return [{k: cache[t][k] for k in ("pid", "camid", "clothes")} for t in cache]


def run_test(net, cfg, m, name, a3=False):
    pq, pg, _, div = TESTS[name]
    keys = ("img", "ldmk", "ipd", "score") + (("a_img", "a_ldmk", "a_score", "b_img", "b_ldmk", "b_score", "ab_ipd") if a3 else ())
    Cq, Cg = read(pq, keys=keys), read(pg)
    qt, gt = list(Cq), list(Cg)
    Tq, Tg = templates(net, cfg, Cq), templates(net, cfg, Cg)
    S = sim(Tq, qt, Tg, gt)
    res = {"all": metrics(m, S, meta(Cq), meta(Cg)), "by_ipd": by_ipd(m, S, Tq, qt, meta(Cq), meta(Cg), div)}
    if a3:   # train_m3 A3: same (clean) face boxes; a = DFA on degraded crop, b = clean crop's alignment
        for p in ("a_", "b_"):
            T = templates(net, cfg, {t: Cq[t] for t in qt}, prefix=p)
            Sp = sim(T, qt, Tg, gt)
            res[f"A3_{p[0]}"] = {"all": metrics(m, Sp, meta(Cq), meta(Cg)), "by_ipd": by_ipd(m, Sp, T, qt, meta(Cq), meta(Cg), div)}
    return res


def val_sets(net, cfg, m, names=VALS):
    """C4 model selection: held-out train IDs (splits/*_qme.json val), probe/gallery split as eval.main_protocol
    val_pairs (CCVID by clothes, MEVID by camera)."""
    from eval.main_protocol import val_pairs
    out = {}
    for name, (cache, key, _) in names.items():
        d = name.split("_")[0]
        C = read(cache, tids=None)
        C = {t: v for t, v in C.items() if v["pid"] in set(SPLITS[d]["val"])}
        T = templates(net, cfg, C)
        ts = [{"subject_or_track_id": t} for t in C]
        P, Pm, G, Gm = val_pairs(ts, meta(C), key)
        qt, gt = [x["subject_or_track_id"] for x in P], [x["subject_or_track_id"] for x in G]
        out[name] = metrics(m, sim(T, qt, T, gt), Pm, Gm)
    return out


def distribution(name, div=1):
    """A1 + A4: frame / track IPD percentiles, face detection rate per person-height bin."""
    C = read(name, keys=("ipd", "score", "body_h", "pos"))
    ipd = np.concatenate([c["ipd"][c["score"] >= 0.5] for c in C.values()]) / div
    trk = np.array([np.median(c["ipd"][c["score"] >= 0.5]) / div if (c["score"] >= 0.5).any() else np.nan for c in C.values()])
    hs = np.concatenate([c["body_h"] for c in C.values()])
    det = np.concatenate([np.isin(np.arange(len(c["body_h"])), c["pos"]) for c in C.values()])
    pct = lambda x: {f"p{q}": round(float(np.percentile(x, q)), 1) for q in (10, 25, 50, 75, 90)} if len(x) else {}  # noqa: E731
    hb = {}
    for lo, hi in ((0, 100), (100, 150), (150, 200), (200, 300), (300, 1e9)):
        k = (hs >= lo) & (hs < hi)
        if k.any():
            hb[f"{lo}-{hi:g}" if hi < 1e9 else f">{lo}"] = {"frames": int(k.sum()), "face_det_%": round(100 * float(det[k].mean()), 1)}
    return {"frame_ipd": pct(ipd), "track_ipd": pct(trk[~np.isnan(trk)]), "tracks": len(trk),
            "tracks_without_face": int(np.isnan(trk).sum()),
            "track_ipd_bins": {f"{lo:g}-{hi:g}": int(((trk >= lo) & (trk < hi)).sum()) for lo, hi in BINS},
            "face_detection_by_person_height": hb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="fine-tuned KP-RPE state_dict (default: original WebFace12M checkpoint)")
    ap.add_argument("--tests", nargs="*", default=list(TESTS))
    ap.add_argument("--val", action="store_true", help="also the C4 val sets")
    ap.add_argument("--a3", action="store_true", help="A3 on ccvid_test_degraded")
    ap.add_argument("--dist", action="store_true", help="A1/A4 distributions (no model)")
    ap.add_argument("--out", default="eval/results/face_eval.json")
    a = ap.parse_args()
    res = {"ckpt": a.ckpt or "kprpe WebFace12M"}
    if a.dist:
        res["distribution"] = {n: distribution(TESTS[n][0], TESTS[n][3]) for n in a.tests} | \
                              {"ccvid_gallery": distribution("ccvid_gallery"), "mevid_gallery": distribution("mevid_gallery")}
    net, cfg = load_net(a.ckpt)
    m = aim_metrics()
    for n in a.tests:
        res[n] = run_test(net, cfg, m, n, a3=a.a3 and n == "ccvid_test_degraded")
        print(n, json.dumps(res[n]["all"]), flush=True)
    if a.val:
        res["val"] = val_sets(net, cfg, m)
        print("val", json.dumps(res["val"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

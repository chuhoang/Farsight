"""BigGait behaviour probes with the pretrained CCPG checkpoint: one input factor changed at a time.

Subsets: CCVID val IDs and MEVID test IDs (2 query + up to 4 gallery tracklets per ID, gallery spread over cameras).
Per tracklet the gait window (eval.extract.plan_frames) is loaded once and encoded under each variant.
Reports cross-camera GR-R1 (CAL/AIM protocol), Rank-1 incl. same-camera gallery, genuine/impostor cosine,
and how much each variant moves the embedding (cos to the base embedding).

    bash run.sh eval.gait_probe
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from eval.extract import ccvid_tracklets, mevid_tracklets, plan_frames
from eval.main_protocol import cal_metrics
from eval.metrics import evaluate
from eval.prcc import aim_metrics
from farsight.core import registry
from farsight.modules.m3_encode.gait.biggait.preprocess import preprocess

WINDOW = 60   # frames per tracklet (~2 gait cycles); same window for every variant
VARIANTS = ("base_clips30", "whole_seq", "shuffled", "reversed", "static_frame", "grayscale", "hflip", "lowres_x3")


def variant(name, crops):
    if name in ("base_clips30", "whole_seq"):
        return crops
    if name == "shuffled":
        return [crops[i] for i in np.random.default_rng(0).permutation(len(crops))]
    if name == "reversed":
        return crops[::-1]
    if name == "static_frame":
        return [crops[len(crops) // 2]] * len(crops)
    if name == "grayscale":
        return [cv2.cvtColor(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for c in crops]
    if name == "hflip":
        return [c[:, ::-1].copy() for c in crops]
    if name == "lowres_x3":
        return [cv2.resize(cv2.resize(c, (max(1, c.shape[1] // 3), max(1, c.shape[0] // 3))), (c.shape[1], c.shape[0]))
                for c in crops]
    raise ValueError(name)


@torch.no_grad()
def embed(enc, name, crops):
    """base = the pipeline (30-frame clips, mean of L2-normed); others = one pass over the whole window
    (OpenGait all_ordered style) so only the named factor differs from whole_seq."""
    if name == "base_clips30":
        f = enc.embed_clips(crops)[0].mean(0)
        return f / np.linalg.norm(f)
    x, r = preprocess(variant(name, crops), enc.cfg["input_size"])
    with torch.autocast("cuda", dtype=torch.float16):
        f, _ = enc.forward(x[None].to(enc.device), r[None].to(enc.device))
    return F.normalize(f.float(), dim=1)[0].cpu().numpy()


def subset(tr_q, tr_g, ids, nq=2, ng=4):
    by_q, by_g = defaultdict(list), defaultdict(list)
    for t, v in sorted(tr_q.items()):
        if ids is None or v["pid"] in ids:
            by_q[v["pid"]].append(t)
    for t, v in sorted(tr_g.items()):
        if v["pid"] in by_q:
            by_g[v["pid"]].append(t)
    q = [t for p in by_q for t in by_q[p][:nq]]
    g = []
    for p, ts in by_g.items():   # round-robin over cameras so the gallery has cross-camera matches
        cams = defaultdict(list)
        for t in ts:
            cams[tr_g[t]["camid"]].append(t)
        g += [c[k] for k in range(max(map(len, cams.values()))) for c in cams.values() if k < len(c)][:ng]
    return q, g


def run(name, tr_q, tr_g, ids, enc, m):
    q, g = subset(tr_q, tr_g, ids)
    E = {v: {} for v in VARIANTS}
    for k, t in enumerate(q + g):
        info = tr_q.get(t) or tr_g[t]
        fr = info["frames"]
        _, gi = plan_frames(len(fr))
        mid = len(gi) // 2   # ponytail: middle WINDOW frames only (120 frames in one pass nearly fills 8 GB VRAM)
        gi = gi[max(0, mid - WINDOW // 2): mid + WINDOW // 2]
        crops = [cv2.imread(fr[i]) for i in gi]
        for v in VARIANTS:
            E[v][t] = embed(enc, v, crops)
        if k % 50 == 0:
            print(f"  {name}: {k}/{len(q) + len(g)}", flush=True)
    meta = lambda ts, tr: [{"pid": tr[t]["pid"], "camid": tr[t]["camid"], "clothes": tr[t]["clothes"]} for t in ts]  # noqa
    qm, gm = meta(q, tr_q), meta(g, tr_g)
    qid, gid = np.array([x["pid"] for x in qm]), np.array([x["pid"] for x in gm])
    same = qid[:, None] == gid[None]
    res = {"n_query": len(q), "n_gallery": len(g)}
    for v in VARIANTS:
        Q, G = np.stack([E[v][t] for t in q]), np.stack([E[v][t] for t in g])
        S = Q @ G.T
        res[v] = {"GR_R1": float(cal_metrics(m, S, qm, gm)["GR_R1"]), "rank1_incl_same_cam": float(evaluate(S, qid, gid)["rank1"]),
                  "cos_genuine": float(S[same].mean()), "cos_impostor": float(S[~same].mean()),
                  "cos_to_whole_seq": float(np.mean([E[v][t] @ E["whole_seq"][t] for t in q + g]))}
        print(f"{name:6s} {v:13s}", json.dumps({k: round(x, 3) for k, x in res[v].items()}), flush=True)
    return res


if __name__ == "__main__":
    enc, m = registry.build("biggait"), aim_metrics()
    val = set(json.loads(Path("splits/ccvid_qme_halftest.json").read_text())["val"])
    out = {"ccvid_val": run("ccvid", ccvid_tracklets("query"), ccvid_tracklets("gallery"), val, enc, m),
           "mevid_test": run("mevid", mevid_tracklets("query"), mevid_tracklets("gallery"), None, enc, m)}
    Path("eval/results/gait_probe.json").write_text(json.dumps(out, indent=1))

"""BigGait diagnostics (stored templates, no GPU): does the matching rule explain weak gait scores?

Stored gait feat = L2-normalised flattened BigGait embeddings (256 ch x 16 parts, channel-major), mean over
30-frame clips. OpenGait evaluates the checkpoint with per-part Euclidean distance averaged over parts.

    bash run.sh eval.gait_diag
"""
import json
from pathlib import Path

import numpy as np

from eval.main_protocol import cal_metrics, load
from eval.metrics import evaluate
from eval.prcc import aim_metrics

C, P = 256, 16


def feats(ts):
    keep = [i for i, t in enumerate(ts) if t["gait"] is not None]
    return np.stack([ts[i]["gait"]["feat"] for i in keep]).astype(np.float32), keep


def sims(Q, G, rule, mu=None):
    if rule == "global_cos":
        return Q @ G.T
    if rule == "global_cos_centered":
        q, g = Q - mu, G - mu
        q /= np.linalg.norm(q, axis=1, keepdims=True); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return q @ g.T
    q, g = Q.reshape(-1, C, P), G.reshape(-1, C, P)
    if rule == "part_cos":
        q = q / np.linalg.norm(q, axis=1, keepdims=True); g = g / np.linalg.norm(g, axis=1, keepdims=True)
        return np.einsum("acp,bcp->ab", q, g) / P
    if rule == "part_euc":   # OpenGait cuda_dist(metric='euc'), as similarity = -distance
        d = sum(np.sqrt(np.maximum((q[:, :, i] ** 2).sum(1)[:, None] + (g[:, :, i] ** 2).sum(1)[None]
                                   - 2 * q[:, :, i] @ g[:, :, i].T, 0)) for i in range(P))
        return -d / P
    raise ValueError(rule)


if __name__ == "__main__":
    F, m = Path.home() / "datasets/feats", aim_metrics()
    sp = {d: json.loads(Path(f"splits/{d}_qme.json").read_text()) for d in ("ccvid", "mevid")}
    mus = {d: feats(load(F, f"{d}_train")[0])[0].mean(0) for d in ("ccvid", "mevid")}
    sets = {"ccvid_val": ("ccvid", set(sp["ccvid"]["val"])), "ccvid_test": ("ccvid", set(sp["ccvid"]["test"])),
            "mevid_test": ("mevid", None)}
    out = {}
    for name, (d, ids) in sets.items():
        qt, qm = load(F, f"{d}_query", ids)
        gt, gm = load(F, f"{d}_gallery", ids)
        (Q, qi), (G, gi) = feats(qt), feats(gt)
        qm, gm = [qm[i] for i in qi], [gm[i] for i in gi]
        qid, gid = np.array([x["pid"] for x in qm]), np.array([x["pid"] for x in gm])
        same = qid[:, None] == gid[None]
        g0 = Q @ G.T
        out[name] = {"n": [len(Q), len(G)], "cos_genuine_mean": float(g0[same].mean()),
                     "cos_impostor_mean": float(g0[~same].mean()), "cos_impostor_std": float(g0[~same].std()),
                     "chance_R1": float(np.mean([same[i].mean() for i in range(len(Q))]))}
        for rule in ("global_cos", "global_cos_centered", "part_cos", "part_euc"):
            S = sims(Q, G, rule, mus[d])
            r = cal_metrics(m, S, qm, gm)
            out[name][rule] = {"GR_R1": r["GR_R1"], "GR_mAP": r["GR_mAP"], "rank1_all": evaluate(S, qid, gid)["rank1"]}
        print(name, json.dumps(out[name]), flush=True)
    Path("eval/results/gait_diag.json").write_text(json.dumps(out, indent=1))

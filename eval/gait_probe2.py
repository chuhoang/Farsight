"""BigGait probes, part 2 (pretrained CCPG checkpoint):
(a) branch ablation at inference: zero the Appearance or the Denoising branch output (the model saw exactly this
    in training: "Black DA" zeroes one branch for 20% of sequences), to see which branch carries identity vs camera;
(b) the foreground mask the model actually uses (Mask_Branch -> get_body -> get_edge, 64x32): area fraction and
    whether it reaches the image border; overlays saved for visual inspection.
Same subsets / window as eval.gait_probe.

    bash run.sh eval.gait_probe2
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from eval.extract import ccvid_tracklets, mevid_tracklets, plan_frames
from eval.gait_probe import WINDOW, subset
from eval.main_protocol import cal_metrics
from eval.metrics import evaluate
from eval.prcc import aim_metrics
from farsight.core import registry
from farsight.modules.m3_encode.gait.biggait.preprocess import preprocess

VARIANTS = ("whole_seq", "no_appearance", "no_denoising")
OUT_DIR = [Path("eval/results/gait_masks")]


class Probe:
    def __init__(self, mask_fix=True):
        self.enc = registry.build("biggait", mask_fix=mask_fix)
        net, self.mode, self.masks = self.enc.net, None, []
        orig_edge = net.get_edge
        net.get_edge = lambda s: self.masks.append(orig_edge(s)) or self.masks[-1]          # capture foreground
        for name, br in (("no_appearance", net.Appearance_Branch), ("no_denoising", net.Denoising_Branch)):
            br.register_forward_hook(lambda m, i, o, name=name: (torch.zeros_like(o[0]), *o[1:]) if self.mode == name else o)

    @torch.no_grad()
    def __call__(self, mode, crops):
        self.mode, self.masks = mode, []
        x, r = preprocess(crops, self.enc.cfg["input_size"])
        with torch.autocast("cuda", dtype=torch.float16):
            f, _ = self.enc.forward(x[None].to(self.enc.device), r[None].to(self.enc.device))
        m = self.masks[0].float().view(-1, 64, 32).cpu().numpy()
        return F.normalize(f.float(), dim=1)[0].cpu().numpy(), m, x


def overlay(x, m, path):
    """x (S,3,256,128) normalised RGB, m (S,64,32) -> strip of 6 frames with the mask in red."""
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    tiles = []
    for i in np.linspace(0, len(m) - 1, 6).astype(int):
        img = ((x[i].permute(1, 2, 0).numpy() * std + mean) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()
        mk = cv2.resize(m[i], (128, 256), interpolation=cv2.INTER_NEAREST) > 0.5
        img[mk] = (0.5 * img[mk] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        tiles.append(img)
    cv2.imwrite(str(path), np.concatenate(tiles, 1))


def run(name, tr_q, tr_g, ids, pr, m):
    q, g = subset(tr_q, tr_g, ids)
    E, area, border = {v: {} for v in VARIANTS}, [], []
    for k, t in enumerate(q + g):
        fr = (tr_q.get(t) or tr_g[t])["frames"]
        _, gi = plan_frames(len(fr))
        mid = len(gi) // 2
        crops = [cv2.imread(fr[i]) for i in gi[max(0, mid - WINDOW // 2): mid + WINDOW // 2]]
        for v in VARIANTS:
            E[v][t], mk, x = pr(v, crops)
        area.append(mk.mean())
        border.append(np.mean([(f[:, :2].any() or f[:, -2:].any()) for f in mk > 0.5]))   # mask touches left/right edge
        if k < 6:
            overlay(x, mk, OUT_DIR[0] / f"{name}_{k}.jpg")
        if k % 50 == 0:
            print(f"  {name}: {k}/{len(q) + len(g)}", flush=True)
    meta = lambda ts, tr: [{"pid": tr[t]["pid"], "camid": tr[t]["camid"], "clothes": tr[t]["clothes"]} for t in ts]  # noqa
    qm, gm = meta(q, tr_q), meta(g, tr_g)
    qid, gid = np.array([x["pid"] for x in qm]), np.array([x["pid"] for x in gm])
    same = qid[:, None] == gid[None]
    res = {"mask_area_mean": float(np.mean(area)), "mask_touches_side_border_frac": float(np.mean(border))}
    for v in VARIANTS:
        S = np.stack([E[v][t] for t in q]) @ np.stack([E[v][t] for t in g]).T
        res[v] = {"GR_R1": float(cal_metrics(m, S, qm, gm)["GR_R1"]), "rank1_incl_same_cam": float(evaluate(S, qid, gid)["rank1"]),
                  "cos_genuine": float(S[same].mean()), "cos_impostor": float(S[~same].mean())}
        print(f"{name:6s} {v:13s}", json.dumps({k: round(x, 3) for k, x in res[v].items()}), flush=True)
    print(f"{name:6s} mask", json.dumps({k: round(res[k], 3) for k in ("mask_area_mean", "mask_touches_side_border_frac")}))
    return res


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else ""    # e.g. "_fixed" (mask_fix on) / "_orig" (mask_fix off)
    OUT_DIR[0] = Path(f"eval/results/gait_masks{tag}")
    OUT_DIR[0].mkdir(parents=True, exist_ok=True)
    pr, m = Probe(mask_fix=tag != "_orig"), aim_metrics()
    val = set(json.loads(Path("splits/ccvid_qme_halftest.json").read_text())["val"])
    out = {"ccvid_val": run("ccvid", ccvid_tracklets("query"), ccvid_tracklets("gallery"), val, pr, m),
           "mevid_test": run("mevid", mevid_tracklets("query"), mevid_tracklets("gallery"), None, pr, m)}
    Path(f"eval/results/gait_probe2{tag}.json").write_text(json.dumps(out, indent=1))

"""CCVID body-only video protocol (CAL / Simple-CCReID data/datasets/ccvid.py + test.py), plan M3c steps 3 and 6.
query.txt vs gallery.txt tracklets; camid = cam index (+12 for session3); clothes = pid_clotheslabel.
Tracklet feature = AIMEncoder.encode (mean of L2-normed feats of <= --frames uniformly sampled frames).
Scoring = AIM/CAL tools/eval_metrics: general `evaluate` (junk = same pid & cam) and `evaluate_with_clothes(mode='CC')`.
Each checkpoint is run on clean crops and on "face-blurred" crops (heavy Gaussian blur over the top 1/5).

    bash run.sh eval.ccvid_body --root ~/datasets/ccvid/CCVID [--frames 32]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from eval.prcc import aim_metrics
from farsight.modules.m3_encode.body.aim.model import AIMEncoder

CKPTS = ["ltcc-checkpoint.pth.tar", "prcc-checkpoint.pth.tar"]


def blur_head(img, frac=0.2):
    img = img.copy()
    h = max(1, int(round(img.shape[0] * frac)))
    k = (img.shape[1] // 2) | 1  # kernel ~ half the crop width: face unrecognisable
    img[:h] = cv2.GaussianBlur(img[:h], (k, k), 0)
    return img


def tracklets(root, name):
    out = []
    for line in Path(root, name).read_text().split("\n"):
        if not line.strip():
            continue
        path, pid, clothes = line.split()
        cam = int(path.split("_")[1]) + (12 if path.split("/")[0] == "session3" else 0)
        out.append((path, pid, cam, f"{pid}_{clothes}"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "datasets/ccvid/CCVID"))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--out", default="eval/results/ccvid_body.json")
    a = ap.parse_args()
    m = aim_metrics()
    encs = {c: AIMEncoder(checkpoint=c) for c in CKPTS}
    q, g = tracklets(a.root, "query.txt"), tracklets(a.root, "gallery.txt")
    print(f"query {len(q)} gallery {len(g)} tracklets, frames/tracklet <= {a.frames}")
    feats = {(c, v): [] for c in CKPTS for v in ("clean", "face_blur")}
    t0, n_img = time.time(), 0
    for i, (path, *_) in enumerate(q + g):
        frames = sorted(Path(a.root, path).glob("*.jpg"))
        idx = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), a.frames)).round().astype(int))
        crops = [cv2.imread(str(frames[j])) for j in idx]  # same uniform sampling as AIMEncoder.encode
        blurred = [blur_head(c) for c in crops]
        n_img += len(crops)
        for c, enc in encs.items():
            feats[c, "clean"].append(enc.encode(crops, None)["feat"])
            feats[c, "face_blur"].append(enc.encode(blurred, None)["feat"])
        if i % 100 == 0:
            print(f"  {i}/{len(q) + len(g)} {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    ids = lambda ts, k: np.array([t[k] for t in ts])
    args = lambda: (ids(q, 1), ids(g, 1), ids(q, 2), ids(g, 2))
    res = {"frames_per_tracklet": a.frames, "n_query": len(q), "n_gallery": len(g), "n_images": n_img,
           "seconds": round(secs, 1), "forward_passes_per_image": 2 * len(CKPTS)}
    for (c, v), f in feats.items():
        f = np.stack(f)
        D = -f[:len(q)] @ f[len(q):].T
        cmc, mAP = m.evaluate(D, *args())
        cc_cmc, cc_mAP = m.evaluate_with_clothes(D, *args(), ids(q, 3), ids(g, 3), mode="CC")
        res.setdefault(c, {})[v] = {"general": {"rank1": round(100 * cmc[0], 2), "mAP": round(100 * mAP, 2)},
                                    "CC": {"rank1": round(100 * cc_cmc[0], 2), "mAP": round(100 * cc_mAP, 2)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

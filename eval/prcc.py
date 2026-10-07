"""PRCC image protocol = AIM-CCReID test_prcc (plan M3c acceptance: within 1 point of the AIM repo numbers).
gallery = test/A, query same-clothes = test/B, query cloth-changing = test/C; cosine; AIM tools/eval_metrics.evaluate.
AIM test.py always sums features of the image and its horizontal flip; both flip=False and flip=True are reported.

    bash run.sh eval.prcc --root ~/datasets/prcc/prcc [--ckpt prcc-checkpoint.pth.tar]
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np

from farsight.modules.m3_encode.body.aim.model import REPO, AIMEncoder

# AIM repo table, row "Repo": (R1, mAP) in %
PUBLISHED = {"SC": (100.0, 99.8), "CC": (58.2, 58.0)}


def aim_metrics():
    """AIM's own tools/eval_metrics.py, loaded by path (its `tools` package name clashes with ours)."""
    spec = importlib.util.spec_from_file_location("aim_eval_metrics", REPO / "tools" / "eval_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def embed_paths(enc, paths, flips=(False, True), crop_fn=None, bs=32):
    """Stream images in batches of `bs`; returns {flip: (N, D) L2-normed feats}."""
    out = {f: [] for f in flips}
    for i in range(0, len(paths), bs):
        imgs = [cv2.imread(str(p)) for p in paths[i:i + bs]]
        if crop_fn:
            imgs = [crop_fn(x) for x in imgs]
        for f in flips:
            enc.cfg["flip"] = f
            out[f].append(enc.embed(imgs)[0])
        if i % (bs * 50) == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return {f: np.concatenate(v) for f, v in out.items()}


def split(root, cam):
    paths = sorted(Path(root, "rgb", "test", cam).glob("*/*.jpg"))
    return paths, np.array([int(p.parent.name) for p in paths]), np.full(len(paths), "ABC".index(cam))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "datasets/prcc/prcc"))
    ap.add_argument("--ckpt", default="prcc-checkpoint.pth.tar")
    ap.add_argument("--out", default="eval/results/prcc.json")
    a = ap.parse_args()
    ev = aim_metrics().evaluate
    enc = AIMEncoder(checkpoint=a.ckpt)
    t0 = time.time()
    data = {c: split(a.root, c) for c in "ABC"}
    print({c: len(d[0]) for c, d in data.items()})
    feats = {c: embed_paths(enc, d[0]) for c, d in data.items()}
    secs = time.time() - t0
    res = {"checkpoint": a.ckpt, "n": {c: len(d[0]) for c, d in data.items()}, "published_repo": PUBLISHED,
           "extract_seconds": round(secs, 1)}
    g_ids, g_cams = data["A"][1:]
    for flip in (False, True):
        r = {}
        for name, cam in (("SC", "B"), ("CC", "C")):
            q_ids, q_cams = data[cam][1:]
            cmc, mAP = ev(-feats[cam][flip] @ feats["A"][flip].T, q_ids, g_ids, q_cams, g_cams)
            r[name] = {"rank1": round(100 * cmc[0], 2), "rank5": round(100 * cmc[4], 2), "mAP": round(100 * mAP, 2)}
        res[f"flip={flip}"] = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

"""TinyFace 1:N identification (plan M3a acceptance: within 1% of the checkpoint's published numbers).
Protocol = CVLface evaluations/tinyface: probes vs Gallery_Match + Gallery_Distractor (label -100), cosine.

    bash run.sh eval.tinyface --root ~/datasets/tinyface/tinyface [--limit-distractors N]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import first_match_rank
from farsight.core import registry

PUBLISHED = {"kprpe": {1: 76.10, 5: 78.92}}  # CVLface README, ViT-B KP-RPE WebFace12M


def label(p):
    return int(p.stem.split("_")[0])


def embed_dir(enc, paths, bs=256):
    out = []
    for i in range(0, len(paths), bs):
        out.append(enc.embed([cv2.imread(str(p)) for p in paths[i:i + bs]]))
        if i % (bs * 40) == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "datasets/tinyface/tinyface"))
    ap.add_argument("--model", default="kprpe")
    ap.add_argument("--limit-distractors", type=int)
    ap.add_argument("--out", default="eval/results/tinyface.json")
    a = ap.parse_args()
    ts = Path(a.root, "Testing_Set")
    probe, match = sorted((ts / "Probe").glob("*.*")), sorted((ts / "Gallery_Match").glob("*.*"))
    distr = sorted((ts / "Gallery_Distractor").glob("*.*"))[:a.limit_distractors]
    print(f"probe {len(probe)} match {len(match)} distractor {len(distr)}")
    cache = Path.home() / f"datasets/cache/tinyface_{a.model}_{len(distr)}.npz"
    t0 = time.time()
    if cache.exists():
        P, M, D = (np.load(cache)[k] for k in "PMD")
    else:
        enc = registry.build(a.model)
        P, M, D = (embed_dir(enc, x) for x in (probe, match, distr))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, P=P, M=M, D=D)
    # ponytail: probes vs 153k gallery is ~6 GB as one float64 matrix -> rank per probe chunk instead
    G = np.concatenate([M, D])
    g = np.array([label(p) for p in match] + [-100] * len(distr))
    q = np.array([label(p) for p in probe])
    rank = np.concatenate([first_match_rank(P[i:i + 256] @ G.T, q[i:i + 256], g) for i in range(0, len(P), 256)])
    rank = rank[np.isfinite(rank)]
    r = {k: 100 * float((rank <= k).mean()) for k in (1, 5, 20)}
    res = {"model": a.model, "n_distractors": len(distr), "rank": r, "published": PUBLISHED.get(a.model),
           "images_per_s": (len(probe) + len(match) + len(distr)) / (time.time() - t0)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))

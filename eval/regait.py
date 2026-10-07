"""Re-extract one modality of existing template stores, other modalities untouched.

gait: current BigGait wrapper (mask_fix + mask-based gait quality), replaying exactly the frames (and degradation
      RNG) of eval.extract.
body: the body encoder named by --body (plan_csci D2: csci_video). It reads frames lazily, only the frames its
      clips use. Degraded stores get a per-frame RNG (crc32(tid) + frame index), so results do not depend on order.
Resumable: new templates go to <name>.<mod>2.h5 in batches (atomic), gait quality parts to <name>.gait2.json; at
the end the store is rewritten (old file kept as <name>.h5.<mod>1.h5).

    bash run.sh eval.regait [names...]                       # gait
    bash run.sh eval.regait --mod body --body csci_video [names...]
"""
import argparse
import json
import os
import shutil
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

from eval.extract import LazyFrames, ccvid_tracklets, mevid_tracklets, plan_frames, save_batch
from farsight.core import registry
from farsight.io.store import read_templates, write_templates
from tools.degrade_dataset import degrade

F = Path.home() / "datasets/feats"
D = "down:4 turb:2 jpeg:30".split()
FILES = {  # store -> (dataset, split, degradation ops) as created by eval/run_extract.sh + later runs
    "mevid_query": ("mevid", "query", None), "mevid_gallery": ("mevid", "gallery", None),
    "mevid_train": ("mevid", "train", None),
    "ccvid_query": ("ccvid", "query", None), "ccvid_gallery": ("ccvid", "gallery", None),
    "ccvid_train": ("ccvid", "train", None), "ccvid_query_degraded": ("ccvid", "query", D),
    "ccvid_train_degraded": ("ccvid", "train", D), "ccvid_query_degraded_val": ("ccvid", "query", D),
}


def gait_frames(frames, tid, ops):
    """Same frames as eval.extract.Extractor feeds to the gait encoder (degradation RNG replayed in order)."""
    uni, gi = plan_frames(len(frames))
    need = sorted(set(uni.tolist()) | set(gi.tolist())) if ops else gi.tolist()
    rng = np.random.default_rng(zlib.crc32(tid.encode()))
    imgs = {}
    for i in need:
        im = cv2.imread(frames[i])
        imgs[i] = degrade(im, ops, rng) if ops else im
    return [imgs[i] for i in gi]


def run(name, enc, mod="gait"):
    d, split, ops = FILES[name]
    store, part, side = F / f"{name}.h5", F / f"{name}.{mod}2.h5", F / f"{name}.gait2.json"
    frames_of = gait_frames if mod == "gait" else LazyFrames
    if Path(f"{store}.{mod}1.h5").exists() and not part.exists():
        print(f"{name}: already done"); return
    tr = (ccvid_tracklets if d == "ccvid" else mevid_tracklets)(split)
    ts = read_templates(store)
    done = {t["subject_or_track_id"] for t in read_templates(part)} if part.exists() else set()
    qp = json.loads(side.read_text()) if side.exists() else {}
    todo = [t["subject_or_track_id"] for t in ts if t["subject_or_track_id"] not in done]
    print(f"{name}: {len(ts)} templates, {len(done)} done, {len(todo)} to go", flush=True)
    t0, buf = time.time(), []
    for k, tid in enumerate(todo):
        g = enc.encode(frames_of(tr[tid]["frames"], tid, ops), None)
        if mod == "gait":
            qp[tid] = {"quality": g["quality"], **g["quality_parts"]} if g else None
        buf.append({"subject_or_track_id": tid, "face": None, "body": None, "gait": None, "qe_weight": None, mod: g})
        if len(buf) == 20 or k == len(todo) - 1:
            save_batch(part, buf)
            if mod == "gait":
                side.write_text(json.dumps(qp))
            buf = []
            el = time.time() - t0
            print(f"  {k + 1}/{len(todo)} {el:.0f}s ({el / (k + 1):.2f} s/trk)", flush=True)
    new = {t["subject_or_track_id"]: t[mod] for t in read_templates(part)}
    for t in ts:
        t[mod] = new.get(t["subject_or_track_id"])
    shutil.copyfile(store, f"{store}.{mod}1.h5")                      # keep the old modality for before/after
    tmp = Path(f"{store}.tmp")
    write_templates(tmp, ts, mode="w")
    os.replace(tmp, store)
    part.unlink()
    print(f"{name}: rewritten ({sum(t[mod] is not None for t in ts)}/{len(ts)} with {mod})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--mod", default="gait", choices=["gait", "body"])
    ap.add_argument("--body", default="csci_video")
    ap.add_argument("--checkpoint", help="body encoder checkpoint override, e.g. csci_v_ccvid.pth")
    a = ap.parse_args()
    kw = {"checkpoint": a.checkpoint} if a.checkpoint else {}
    enc = registry.build("biggait") if a.mod == "gait" else registry.build(a.body, **kw)   # biggait: min_quality 0
    for name in a.names or FILES:
        run(name, enc, a.mod)

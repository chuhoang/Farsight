"""Sanity check of eval.extract.mevid_tracklets against the image names (PPPP O ooo C ccc T ttt F fffff):
every tracklet must be one (pid, outfit, camera, track) with consecutive frames, labels must match the
annotation row, files must exist, query/gallery must be disjoint. Also flags blank crops (near-constant image).

    bash run.sh eval.check_mevid_tracks
"""
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from eval.extract import mevid_tracklets

PAT = re.compile(r"(\d{4})O(\d{3})C(\d{3})T(\d{3})F(\d{5})\.jpg$")


def check(split, rng):
    tr = mevid_tracklets(split)
    bad = Counter()
    blank_tracks, nblank_frames, nsampled = [], 0, 0
    for tid, v in tr.items():
        parts = [PAT.search(f) for f in v["frames"]]
        if any(p is None for p in parts):
            bad["unparsable_name"] += 1
            continue
        key = {p.groups()[:4] for p in parts}
        if len(key) != 1:
            bad["mixed_pid_outfit_cam_track"] += 1
            continue
        pid, outfit, cam, _ = next(iter(key))
        if pid != v["pid"]:
            bad["pid_mismatch"] += 1
        if f"{pid}_{int(outfit)}" != v["clothes"]:
            bad["outfit_mismatch"] += 1
        if int(cam) != v["camid"]:
            bad["camid_mismatch"] += 1
        fr = np.array([int(p.group(5)) for p in parts])
        if (np.diff(fr) != 1).any():
            bad["non_consecutive_frames"] += 1
        for f in rng.choice(v["frames"], min(3, len(v["frames"])), replace=False):   # existence + blankness sample
            im = cv2.imread(f)
            nsampled += 1
            if im is None:
                bad["missing_file"] += 1
            elif im.std() < 8:
                nblank_frames += 1
                blank_tracks.append(tid)
    return tr, {"tracklets": len(tr), "problems": dict(bad), "blank_frame_frac": nblank_frames / max(nsampled, 1),
                "tracklets_with_blank_sample": len(set(blank_tracks)), "examples_blank": sorted(set(blank_tracks))[:5]}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = {}
    trs = {}
    for split in ("train", "query", "gallery"):
        trs[split], out[split] = check(split, rng)
        print(split, json.dumps(out[split]), flush=True)
    q, g = set(trs["query"]), set(trs["gallery"])
    out["query_gallery_overlap"] = len(q & g)
    out["query_ids_missing_in_gallery"] = len({trs["query"][t]["pid"] for t in q} - {trs["gallery"][t]["pid"] for t in g})
    print("overlap", out["query_gallery_overlap"], "query IDs absent from gallery", out["query_ids_missing_in_gallery"])
    Path("eval/results/check_mevid_tracks.json").write_text(json.dumps(out, indent=1))

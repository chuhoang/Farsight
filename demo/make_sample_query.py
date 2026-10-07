"""Make a sample query for demo.infer from a video: crops of the longest track in its first N frames.

    bash run.sh demo.make_sample_query --video weights/m1_detect_track/test_data/vtest.avi --frames 100
"""
import argparse
from pathlib import Path

import cv2

from farsight.io.video import crop, read_frames
from farsight.modules.m1_detect_track.module import M1DetectTrack

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--n", type=int, default=30, help="images to save (>= 15 enables gait)")
    ap.add_argument("--out", default="demo/sample_query")
    a = ap.parse_args()
    t = max(M1DetectTrack().run(a.video, max_frames=a.frames), key=lambda t: len(t["frames"]))
    obs = {o["frame_idx"]: o for o in t["frames"][-a.n:]}   # last n consecutive observations
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for start, frames in read_frames(a.video):
        for k, f in enumerate(frames):
            if start + k in obs:
                cv2.imwrite(str(out / f"{start + k:06d}.jpg"), crop(f, obs[start + k]["body_box"], pad=0.1))
        if start > max(obs):
            break
    print(f"track {t['track_id']}: saved {len(obs)} crops to {out}")

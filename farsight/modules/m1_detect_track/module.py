"""M1 step: frames -> detector -> verifier -> ByteTrack (+PSR) -> filtered Tracklets (plan 3.1 / 4.M1)."""
import time
from collections import defaultdict

import numpy as np

from farsight.core import registry
from farsight.core.interfaces import BaseTracker
from farsight.io.video import read_frames, video_fps

DEFAULT = {"detector": "bpjdet", "verifier": "yolov8_verifier", "tracker": "bytetrack",
           "appearance": "psr_appearance", "min_track_len": 15, "min_body_h": 32, "batch_size": 8}


class M1DetectTrack(BaseTracker):
    def __init__(self, cfg=None, fps=30, device="cuda"):
        self.cfg = c = {**DEFAULT, **(cfg or {})}
        self.fps, self.device = fps, device
        self.det = registry.build(c["detector"], device=device)
        self.ver = registry.build(c["verifier"], device=device) if c.get("verifier") else None
        self.app = registry.build(c["appearance"], device=device, fps=fps) if c.get("appearance") else None
        self.reset()

    def reset(self, fps=None):
        self.fps = fps or self.fps
        self.trk = registry.build(self.cfg["tracker"], frame_rate=self.fps)
        if self.app:
            self.app.reset()
            self.app.window = self.app.cfg["window_s"] * self.fps
        self.obs = defaultdict(list)

    def detect(self, frames):
        dets = self.det(frames)
        return self.ver.verify(frames, dets) if self.ver else dets

    def update(self, frame_idx, frame, det):
        tr = self.trk.update(det["body"])
        if self.app:
            tr = self.app(frame_idx, frame, tr)
        for tid, i in tr[:, [4, 6]].astype(int):
            b, f = det["body"][i], det["face"][i]      # raw detection box (not Kalman-smoothed) for cropping
            has_face = not np.isnan(f[0])
            self.obs[tid].append({"frame_idx": int(frame_idx), "body_box": b[:4].tolist(), "body_score": float(b[4]),
                                  "face_box": f[:4].tolist() if has_face else None,
                                  "face_score": float(f[4]) if has_face else None})

    def tracklets(self, video_id):
        out = []
        for tid, frames in sorted(self.obs.items()):
            frames = [o for o in frames if o["body_box"][3] - o["body_box"][1] >= self.cfg["min_body_h"]]
            if len(frames) >= self.cfg["min_track_len"]:
                out.append({"video_id": video_id, "track_id": int(tid), "frames": frames})
        return out

    def run(self, video, video_id=None, fps=None, max_frames=None):
        """Whole video -> tracklets. Sets self.stats = {frames, seconds, fps}."""
        self.reset(fps or video_fps(video))   # real fps: PSR window T and track times depend on it
        t0, n = time.perf_counter(), 0
        for start, frames in read_frames(video, batch=self.cfg["batch_size"]):
            if max_frames is not None:
                frames = frames[:max_frames - n]
            for k, (f, d) in enumerate(zip(frames, self.detect(frames))):
                self.update(start + k, f, d)
            n += len(frames)
            if max_frames is not None and n >= max_frames:
                break
        dt = time.perf_counter() - t0
        self.stats = {"frames": n, "seconds": dt, "fps": n / dt}
        return self.tracklets(video_id or str(video))

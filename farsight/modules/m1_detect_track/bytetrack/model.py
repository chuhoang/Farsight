"""ByteTrack = ultralytics' BYTETracker (same algorithm as ifzhang/ByteTrack), fed with our (N,5) boxes."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from ultralytics.trackers.byte_tracker import BYTETracker

HERE = Path(__file__).parent


class _Dets:
    """Minimal ultralytics Boxes-like view over (N,5) x1y1x2y2s."""

    def __init__(self, d):
        self.d = d

    conf = property(lambda s: s.d[:, 4])
    cls = property(lambda s: np.zeros(len(s.d), np.float32))
    xyxy = property(lambda s: s.d[:, :4])
    xywh = property(lambda s: np.concatenate([(s.d[:, :2] + s.d[:, 2:4]) / 2, s.d[:, 2:4] - s.d[:, :2]], 1))

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        return _Dets(self.d[i])


class ByteTrack:
    def __init__(self, frame_rate=30, **over):
        cfg = {**yaml.safe_load((HERE / "config.yaml").read_text()), **over}
        cfg["track_buffer"] = int(round(cfg["track_buffer"] * frame_rate / 30))  # as in ultralytics' track mode
        self.t = BYTETracker(SimpleNamespace(**cfg))

    def update(self, det):
        """det (N,5) -> (M,7) rows [x1, y1, x2, y2, track_id, score, det_idx] for confirmed active tracks."""
        det = np.asarray(det, np.float32).reshape(-1, 5)
        out = self.t.update(_Dets(det))
        return out[:, [0, 1, 2, 3, 4, 5, 7]] if len(out) else np.zeros((0, 7), np.float32)

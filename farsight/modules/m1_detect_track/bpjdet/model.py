"""BPJDet (YOLOv5 fork) body+face joint detector. Uses the repo's own NMS + post_process_batch decode."""
import contextlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

from farsight.core.interfaces import BaseDetector
from farsight.core.weights import ROOT, fetch

from .preprocess import letterbox

HERE = Path(__file__).parent
REPO = ROOT / "third_party" / "BPJDet"


@contextlib.contextmanager
def _repo_imports(repo, names=("models", "utils", "val")):
    """BPJDet uses top-level `models`/`utils` packages; import them in isolation, then unregister
    so other third_party repos with the same package names don't collide. Loaded objects keep working."""
    def ours(k):
        return k.split(".")[0] in names
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if ours(k)}
    sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        sys.path.remove(str(repo))
        for k in [k for k in sys.modules if ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


class BPJDet(BaseDetector):
    def __init__(self, device="cuda", **over):
        cfg = {**yaml.safe_load((HERE / "config.yaml").read_text()), **over}
        self.cfg, self.device = cfg, torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = cfg["half"] and self.device.type == "cuda"
        with _repo_imports(REPO):
            from utils.general import non_max_suppression
            # val.py imports eval-only code (fvcore); stub it, we only need post_process_batch
            sys.modules["utils.bp_eval"] = types.SimpleNamespace(body_part_association_evaluation=None)
            from val import post_process_batch
            ck = torch.load(fetch(HERE), map_location="cpu", weights_only=False)  # pickled nn.Module
        self._nms, self._post = non_max_suppression, post_process_batch
        m = (ck.get("ema") or ck["model"]).float().fuse().eval()
        self.model = (m.half() if self.half else m).to(self.device)
        self.stride = int(self.model.stride.max())
        self.data = yaml.safe_load((REPO / cfg["data_yaml"]).read_text())
        self.data.update(conf_thres_part=cfg["conf_face"], iou_thres_part=cfg["iou_thres"],
                         match_iou_thres=cfg["match_iou"])

    @torch.no_grad()
    def __call__(self, frames):
        out, buf = [], []
        for f in frames:  # batch runs of same-shape frames (video frames all share one shape)
            if buf and (len(buf) == self.cfg["batch_size"] or f.shape != buf[0].shape):
                out += self._batch(buf)
                buf = []
            buf.append(f)
        return out + (self._batch(buf) if buf else [])

    def _batch(self, frames):
        c, no = self.cfg, self.data["num_offsets"]
        img = letterbox(frames, c["imgsz"], self.stride, self.device, self.half)
        pred = self.model(img)[0].float()
        body = self._nms(pred, c["conf_body"], c["iou_thres"], classes=[0], num_offsets=no)
        part = self._nms(pred, c["conf_face"], c["iou_thres"], classes=[1], num_offsets=no)
        res = []
        for b, p, f in zip(body, part, frames):
            # post_process_batch works per image; call it with a batch of one so outputs stay per-frame
            boxes, points, scores, *_ = self._post(self.data, img[:1], [], [[f.shape[:2]]], [b], [p])
            n = len(boxes)
            bd = np.zeros((n, 5), np.float32)
            fc = np.full((n, 5), np.nan, np.float32)
            if n:
                bd[:, :4], bd[:, 4] = np.asarray(boxes), np.asarray(scores)
                pts = np.asarray(points)[:, 0]                    # (n, 7): xc, yc, conf, x1, y1, x2, y2
                hit = pts[:, 2] > 0
                fc[hit, :4], fc[hit, 4] = pts[hit, 3:7], pts[hit, 2]
            res.append({"body": bd, "face": fc})
        return res

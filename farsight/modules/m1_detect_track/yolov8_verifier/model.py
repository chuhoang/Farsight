"""YOLOv8 person verifier (CrowdHuman YOLOv8n by default, COCO YOLOv8x optional): keep a detector body box only if a confident YOLO person overlaps it."""
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_ioa

from farsight.core.interfaces import BaseDetector
from farsight.core.weights import fetch

HERE = Path(__file__).parent


class YOLOv8Verifier(BaseDetector):
    def __init__(self, device="cuda", **over):
        self.cfg = {**yaml.safe_load((HERE / "config.yaml").read_text()), **over}
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(fetch(HERE, self.cfg["checkpoint"])))

    def __call__(self, frames):
        """BGR frames -> YOLO person boxes as {"body": (N,5), "face": NaN rows} (BaseDetector format)."""
        c = self.cfg
        rs = self.model.predict(list(frames), imgsz=c["imgsz"], conf=c["conf"], classes=[0], batch=c["batch_size"],
                                quantize=16 if c["half"] and self.device != "cpu" else None, device=self.device, verbose=False)
        out = []
        for r in rs:
            b = np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]], 1).astype(np.float32)
            out.append({"body": b, "face": np.full_like(b, np.nan)})
        return out

    def verify(self, frames, dets):
        """Filter detector outputs (body rows + their face rows) by YOLO agreement."""
        out = []
        for d, y in zip(dets, self(frames)):
            if len(d["body"]) and len(y["body"]):
                keep = bbox_ioa(d["body"][:, :4], y["body"][:, :4], iou=True).max(1) >= self.cfg["iou_keep"]
            else:
                keep = np.zeros(len(d["body"]), bool)
            out.append({"body": d["body"][keep], "face": d["face"][keep]})
        return out

"""PSR (patch-memory re-identification) on top of ByteTrack, ResNet-18 ImageNet features of body crops.

Per final ID: FIFO memory of K features, one added every N frames while the box is not overlapping others.
- ByteTrack spawns a new raw ID  -> compare with memories of IDs absent in this frame and seen within T s;
  closest under `thr` takes over (raw ID is aliased to it), else the new ID is kept.
- Two boxes stop overlapping       -> if the swapped assignment matches memories better, swap the two IDs.
"""
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from ultralytics.utils.metrics import bbox_ioa

from farsight.core.weights import fetch

from .preprocess import crops_to_tensor

HERE = Path(__file__).parent


class PSRAppearance:
    def __init__(self, device="cuda", fps=30, **over):
        self.cfg = c = {**yaml.safe_load((HERE / "config.yaml").read_text()), **over}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = c["half"] and self.device.type == "cuda"
        net = torchvision.models.resnet18()
        net.load_state_dict(torch.load(fetch(HERE), map_location="cpu"))
        net.fc = torch.nn.Identity()
        self.net = (net.half() if self.half else net).eval().to(self.device)
        self.window = c["window_s"] * fps
        self.reset()

    def reset(self):
        self.alias, self.mem, self.last_seen, self.last_add, self.pairs = {}, {}, {}, {}, set()

    @torch.no_grad()
    def embed(self, frame, boxes):
        """(N,4) boxes -> (N,512) L2-normalised float32 features."""
        if len(boxes) == 0:
            return np.zeros((0, 512), np.float32)
        x = crops_to_tensor(frame, boxes, self.cfg["input_hw"]).to(self.device)
        f = self.net(x.half() if self.half else x).float()
        return torch.nn.functional.normalize(f, dim=1).cpu().numpy()

    def dist(self, f, tid):
        m = np.stack(self.mem[tid])
        if self.cfg["metric"] == "mse":
            return float(((m - f) ** 2).mean(1).min())
        return float(1 - (m @ f).max())

    def __call__(self, frame_idx, frame, tracks):
        """tracks (M,7) [x1,y1,x2,y2,raw_id,score,det_idx] from ByteTrack -> same rows with remapped IDs."""
        c, tracks = self.cfg, tracks.copy()
        if not len(tracks):
            self._forget(frame_idx)
            return tracks
        raw = tracks[:, 4].astype(int)
        iou = bbox_ioa(tracks[:, :4], tracks[:, :4], iou=True)
        np.fill_diagonal(iou, 0)
        overl = iou.max(1) > c["overlap_iou"]
        new = np.array([r not in self.alias for r in raw])
        ids = np.array([self.alias.get(r, r) for r in raw])
        # a raw track aliased to identity X while ByteTrack revives X's own raw track -> two boxes with ID X;
        # ByteTrack's own continuity wins, the alias falls back to its raw ID (never used as a final ID before)
        for t in {t for t in ids[~new] if (ids[~new] == t).sum() > 1}:
            dup = np.flatnonzero((ids == t) & ~new)
            keep = next((i for i in dup if raw[i] == t), dup[0])
            for i in dup[dup != keep]:
                self.alias[raw[i]] = ids[i] = raw[i]

        # pairs (by final ID) that were crossing last frame and are separated now
        cur = {tuple(sorted((ids[i], ids[j]))) for i, j in zip(*np.nonzero(np.triu(iou > c["overlap_iou"])))}
        pos = {t: i for i, t in enumerate(ids)}
        ended = [p for p in self.pairs - cur if p[0] in pos and p[1] in pos and p[0] in self.mem and p[1] in self.mem]
        self.pairs = cur
        due = ~overl & np.array([frame_idx - self.last_add.get(t, -1e9) >= c["mem_every"] for t in ids])
        need = new | due
        for p in ended:
            need[[pos[p[0]], pos[p[1]]]] = True
        feats = np.zeros((len(tracks), 512), np.float32)
        if need.any():
            feats[need] = self.embed(frame, tracks[need, :4])

        present = set(ids[~new])
        for i in np.flatnonzero(new):                     # new raw ID -> maybe an old identity
            cand = [t for t in self.mem if t not in present and frame_idx - self.last_seen[t] <= self.window]
            d = [self.dist(feats[i], t) for t in cand]
            if d and min(d) < c["thr"]:
                ids[i] = cand[int(np.argmin(d))]
            self.alias[raw[i]] = ids[i]
            present.add(ids[i])

        for a, b in ended:                                  # crossing finished -> undo an ID swap
            ia, ib = pos[a], pos[b]
            keep = self.dist(feats[ia], a) + self.dist(feats[ib], b)
            swap = self.dist(feats[ia], b) + self.dist(feats[ib], a)
            if swap < keep - c["swap_margin"]:
                self.alias[raw[ia]], self.alias[raw[ib]] = b, a
                ids[ia], ids[ib] = b, a

        for i, t in enumerate(ids):
            self.last_seen[t] = frame_idx
            if due[i] or (new[i] and not overl[i]):
                self.mem.setdefault(t, deque(maxlen=c["mem_size"])).append(feats[i])
                self.last_add[t] = frame_idx
        self._forget(frame_idx)
        tracks[:, 4] = ids
        return tracks

    def _forget(self, frame_idx):
        for t in [t for t, s in self.last_seen.items() if frame_idx - s > self.window]:
            self.mem.pop(t, None), self.last_seen.pop(t), self.last_add.pop(t, None)

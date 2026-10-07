"""CSCI (ICCV'25, Colors See Colors Ignore) body encoder: EVA02-L/14, 224x224, 1024-d ReID token.

csci_video = EZ-CLIP video model (ez_eva02_vid_hybrid_extra, T=4), author's test protocol: 8-frame clips
(CAL recombination, stride 4), every 2nd frame -> 4-frame model input, track feature = L2(mean of clip features).
csci_image = image model (eva02_l_cloth) on uniformly sampled frames, L2(mean). The colour token / colour head only
matter for training; inference uses the ReID (cls) token after fc_norm, as Script/test.sh.
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

from farsight.core.hooks import LayerCapture, pool_tokens
from farsight.core.interfaces import BaseEncoder
from farsight.core.weights import fetch

from .clips import model_clips
from .preprocess import preprocess

HERE = Path(__file__).parent
REPO = HERE.parents[4] / "third_party" / "CSCI"
_PKGS = ("model", "loss", "tools", "config", "torchinfo")


def _csci():
    """Import CSCI's model code. Its top-level packages (model, loss, tools, config) are loaded as bare namespaces
    (their __init__ pull training-only deps: mmengine, ...) and removed from sys.modules afterwards so they cannot
    clash with FarSight's own `tools` or other repos' `model` packages."""
    own = lambda k: k.split(".")[0] in _PKGS  # noqa: E731
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if own(k)}
    for p in _PKGS[:-1]:
        sys.modules[p] = types.ModuleType(p)
        sys.modules[p].__path__ = [str(REPO / p)]
    sys.modules["torchinfo"] = types.SimpleNamespace(summary=None)   # imported, never called at inference
    sys.path.insert(0, str(REPO))
    try:
        from config.defaults import _C
        from model.eva_cloth_embed import eva02_large_patch14_clip_224_cloth
        from model.ez_eva_custom import ez_eva02_vid_hybrid_extra
    finally:
        sys.path.remove(str(REPO))
        for k in [k for k in sys.modules if own(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
    return _C.clone(), ez_eva02_vid_hybrid_extra, eva02_large_patch14_clip_224_cloth


def _state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd.get("state_dict", sd))
    return {k.removeprefix("module."): v for k, v in sd.items()}


def _build(video, sd=None):
    """Model sized from the checkpoint (classifier / colour heads differ per dataset and colour profile)."""
    cfg, vid, img = _csci()
    rows = lambda k, d: sd[k].shape[0] if sd is not None and k in sd else d  # noqa: E731
    cfg.MODEL.EXTRA_DIM = rows("mlp.fc2.weight", 1024)
    cfg.TRAIN.TEACH1_NUMCLASSES = rows("head_image.weight", 0) or None
    kw = dict(config=cfg, pretrained=False, num_classes=rows("head.weight", 1), cloth=rows("cloth_embed", 1),
              cloth_xishu=cfg.MODEL.CLOTH_XISHU, spatial_avg=None)
    net = vid(tim_dim=4, e2e_train=True, joint=None, temporal_avg=None, **kw) if video else img(**kw)
    if sd is not None:
        net.load_state_dict(sd, strict=True)
    return net


class CSCIEncoder(BaseEncoder):
    video = True

    def __init__(self, pretrained=True, **overrides):
        self.cfg = {**yaml.safe_load((HERE / "config.yaml").read_text()), **overrides}
        dev = self.cfg["device"]
        self.device = torch.device(dev if dev != "cuda" or torch.cuda.is_available() else "cpu")
        self.pretrained = pretrained
        ck = self.cfg["checkpoint" if self.video else "image_checkpoint"]
        self.net = _build(self.video, _state_dict(fetch(HERE, ck)) if pretrained else None).eval().to(self.device)
        self.hook = self.net.blocks[self.cfg["hook_block"]]
        self._image = None

    def _run(self, x):
        """x (B, 3, H, W) images or (B, 3, T, H, W) clips -> (feat (B, 1024), inter (B, 1024)) float32."""
        amp = self.cfg["fp16"] and self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=amp), \
                LayerCapture(self.hook, lambda o: pool_tokens(o, "mean")) as cap:
            f = self.net(x.to(self.device), None)
        inter = cap.out.float().view(x.shape[0], -1, cap.out.shape[-1]).mean(1)   # video: (B*T, C) -> per clip
        return f.float(), inter

    @torch.no_grad()
    def embed(self, crops):
        """BGR crops of one track -> (per-clip / per-image feat (C, 1024) unnormalised, inter (C, 1024)) numpy."""
        bs = self.cfg["batch_size"]
        if self.video:
            clips = model_clips(len(crops), self.cfg["seq_len"], self.cfg["stride"], self.cfg["max_clips"])
            used = sorted({i for c in clips for i in c})
            pos = {i: k for k, i in enumerate(used)}
            x = preprocess([crops[i] for i in used], self.cfg["input_size"])
            batches = [x[[pos[i] for c in clips[j:j + bs] for i in c]].view(-1, 4, *x.shape[1:]).transpose(1, 2)
                       for j in range(0, len(clips), bs)]
        else:
            x = preprocess(crops, self.cfg["input_size"])
            batches = list(x.split(4 * bs))
        feats, inters = zip(*(self._run(b) for b in batches))
        return torch.cat(feats).cpu().numpy(), torch.cat(inters).cpu().numpy()

    def frames(self, n):
        """Indices of the n crops this encoder reads."""
        if self.video and n >= self.cfg["min_video_frames"]:
            return sorted({i for c in model_clips(n, self.cfg["seq_len"], self.cfg["stride"], self.cfg["max_clips"]) for i in c})
        return np.unique(np.linspace(0, n - 1, min(n, self.cfg["image_max_frames"])).round().astype(int)).tolist()

    def encode(self, crops, track):
        if not crops:
            return None
        n = len(crops)
        if self.video and n < self.cfg["min_video_frames"]:   # too short for a 4-frame clip -> image model (plan B3)
            if self._image is None:
                self._image = CSCIImageEncoder(self.pretrained, **{**self.cfg, "device": str(self.device)})
            return self._image.encode(crops, track)
        idx = self.frames(n)
        feats, inters = self.embed(crops if self.video else [crops[i] for i in idx])
        feat = feats.mean(0)
        feat /= max(np.linalg.norm(feat), 1e-12)
        # quality = mean resolution factor (crop height vs model input) x mean detector confidence (as AIM)
        res = np.mean([min(1.0, crops[i].shape[0] / self.cfg["input_size"][0]) for i in idx])
        frames = (track or {}).get("frames") or []
        scores = [frames[i].get("body_score") for i in idx] if len(frames) == n else [f.get("body_score") for f in frames]
        scores = [s for s in scores if s is not None]
        conf = float(np.clip(np.mean(scores), 0, 1)) if scores else 1.0
        return {"feat": feat.astype(np.float32), "quality": float(res * conf),
                "inter_feat": inters.mean(0).astype(np.float32), "n_frames": len(idx)}


class CSCIImageEncoder(CSCIEncoder):
    video = False

"""AIM (CVPR'23) identity branch: ResNet-50, last stride 1, maxavg pooling + BN -> 4096-d."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml

from farsight.core.hooks import LayerCapture, pool_tokens
from farsight.core.interfaces import BaseEncoder
from farsight.core.weights import fetch

HERE = Path(__file__).parent
REPO = HERE.parents[4] / "third_party" / "AIM-CCReID"


def _aim_resnet50():
    # AIM's code lives in a top-level package called `models`; import it with its repo on sys.path, then
    # restore sys.modules so it cannot clash with other repos' `models` packages.
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "models" or k.startswith("models.")}
    sys.path.insert(0, str(REPO))
    try:
        from models.img_resnet import ResNet50
    finally:
        sys.path.remove(str(REPO))
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    cfg = NS(MODEL=NS(RES4_STRIDE=1, FEATURE_DIM=4096, POOLING=NS(NAME="maxavg", P=3)))
    rn50 = torchvision.models.resnet50
    with mock.patch("torchvision.models.resnet50", lambda **k: rn50(weights=None)):  # skip ImageNet download
        return ResNet50(cfg)


class AIMEncoder(BaseEncoder):
    def __init__(self, pretrained=True, **overrides):
        self.cfg = {**yaml.safe_load((HERE / "config.yaml").read_text()), **overrides}
        dev = self.cfg["device"]
        self.device = torch.device(dev if dev != "cuda" or torch.cuda.is_available() else "cpu")
        self.net = _aim_resnet50()
        if pretrained:
            sd = torch.load(fetch(HERE, self.cfg["checkpoint"]), map_location="cpu", weights_only=False)
            self.net.load_state_dict(sd["model_state_dict"])  # identity branch only; model2/classifiers not saved/used
        self.net.eval().to(self.device)
        self.hook = self.net.get_submodule(self.cfg["hook_layer"])

    @torch.no_grad()
    def embed(self, crops):
        """BGR crops -> (feat (N, D) L2-normed, inter (N, 1024)) numpy float32."""
        from .preprocess import preprocess
        feats, inters = [], []
        for i in range(0, len(crops), self.cfg["batch_size"]):
            x = preprocess(crops[i:i + self.cfg["batch_size"]], self.cfg["input_size"]).to(self.device)
            with LayerCapture(self.hook, pool_tokens) as cap:
                f = self.net(x)[1]
            inter = cap.out
            if self.cfg["flip"]:
                f = f + self.net(torch.flip(x, [3]))[1]
            if self.cfg["half_dim"]:
                f = f[:, f.shape[1] // 2:]  # avg-pool half
            feats.append(F.normalize(f, dim=1).float().cpu())
            inters.append(inter.float().cpu())
        return torch.cat(feats).numpy(), torch.cat(inters).numpy()

    def encode(self, crops, track):
        if not crops:
            return None
        n = len(crops)
        idx = np.unique(np.linspace(0, n - 1, min(n, self.cfg["max_frames"])).round().astype(int))
        sel = [crops[i] for i in idx]
        feats, inters = self.embed(sel)
        feat = feats.mean(0)
        feat /= max(np.linalg.norm(feat), 1e-12)
        # quality = mean resolution factor (crop height vs model input) x mean detector confidence, both in [0, 1]
        res = np.mean([min(1.0, c.shape[0] / self.cfg["input_size"][0]) for c in sel])
        frames = (track or {}).get("frames") or []
        scores = [frames[i].get("body_score") for i in idx if i < len(frames)] if len(frames) == n else \
            [f.get("body_score") for f in frames]
        scores = [s for s in scores if s is not None]
        conf = float(np.clip(np.mean(scores), 0, 1)) if scores else 1.0
        return {"feat": feat.astype(np.float32), "quality": float(res * conf),
                "inter_feat": inters.mean(0).astype(np.float32), "n_frames": len(idx)}

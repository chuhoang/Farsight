"""Quality Estimator (QE) of QME (Zhu et al., ICCV 2025, MIT licence).
FaceQE/Mlp are ported from third_party/QME_ICCV25/model.py (Face_Quality_Estimator, Mlp) with identical
parameter names so the released checkpoints load. QME model.py itself is not importable here: it pulls
every backbone (WBModules, transformers, sklearn)."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from farsight.core.registry import register
from farsight.core.weights import fetch
from farsight.modules.m4_fusion.module import load_cfg

HERE = Path(__file__).parent


class Mlp(nn.Module):
    def __init__(self, i, h, o, act=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1, self.act, self.fc2, self.drop = nn.Linear(i, h), act(), nn.Linear(h, o), nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)


class FaceQE(nn.Module):
    """style (..., 4*C) -> per-frame logit. QME: sigmoid per frame, then mean over frames."""

    def __init__(self, patch_dim=512, drop=0.0):
        super().__init__()
        self.f_style_mlp = Mlp(patch_dim * 4, patch_dim * 2, patch_dim, nn.ReLU, drop)
        self.weight_assigner = nn.Sequential(nn.Linear(patch_dim, 1))
        init_weights(self)

    def forward(self, style):
        return self.weight_assigner(self.f_style_mlp(style)).squeeze(-1)


def style_from_blocks(blocks):
    """QME face style. blocks: list (one per hooked block) of (B, P, C) tokens -> (B, n_blk*2*C).
    This is what M3 face should store as inter_feat (averaged over frames)."""
    x = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in blocks])        # (blk, B, P, C)
    x = nn.functional.layer_norm(x, x.shape[-1:])
    s = torch.stack([x.mean(2), x.std(2)], dim=2)                                     # (blk, B, 2, C)
    return s.permute(1, 0, 2, 3).flatten(1)


def load_qe_state(net, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    if any(k.startswith("qe.") for k in sd):          # a full QME checkpoint also carries its QE
        sd = {k[3:]: v for k, v in sd.items() if k.startswith("qe.")}
    net.load_state_dict(sd)
    return net


@register("qe")
class QualityEstimator:
    def __init__(self, **overrides):
        c = load_cfg(HERE, **overrides)
        if "checkpoint" in overrides and overrides["checkpoint"] is None:
            c["checkpoint"] = None
        self.device = c["device"]
        self.net = FaceQE(c["patch_dim"]).eval().to(self.device)
        ck = c.get("checkpoint")
        if ck:
            load_qe_state(self.net, ck if Path(str(ck)).is_file() else fetch(HERE, ck))
            self.net.to(self.device)

    @torch.no_grad()
    def __call__(self, inter_feat):
        """inter_feat: (4C,) track style, (N, 4C) per-frame styles, or (N, blk, P, C) raw tokens -> W in (0,1)."""
        x = torch.as_tensor(np.asarray(inter_feat), dtype=torch.float32, device=self.device)
        if x.dim() == 4:
            x = style_from_blocks(list(x.transpose(0, 1)))
        return float(torch.sigmoid(self.net(x.reshape(-1, x.shape[-1]))).mean())

    @torch.no_grad()
    def batch(self, X):
        """(B, 4C) styles -> (B,) weights; NaN rows (no face) -> 0."""
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        bad = torch.isnan(X).any(-1)
        w = torch.sigmoid(self.net(torch.nan_to_num(X)))
        return w.masked_fill(bad, 0.0).cpu().numpy()

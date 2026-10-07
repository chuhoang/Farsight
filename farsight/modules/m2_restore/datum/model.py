"""DATUM (CVPR'24) turbulence mitigation, dynamic-scene checkpoint, run on face-crop sequences.

The repo's guided deformable attention is a JIT-compiled CUDA extension (needs nvcc/gcc, absent here);
we swap in `deform_attn_torch`, a grid_sample re-implementation of its forward pass (inference only).
"""
import sys
from argparse import Namespace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from farsight.core.interfaces import BaseRestorer
from farsight.core.weights import ROOT, fetch

HERE = Path(__file__).parent
REPO = ROOT / "third_party" / "DATUM" / "code"


def deform_attn_torch(q, kv, offset, kh, kw, stride, pad, dil, heads, dg, clip):
    """Forward of TM_model/op deform_attn (modulated im2col, mask=1, then per-pixel softmax attention).
    q (B,1,C,H,W), kv (B,1,2C,H,W), offset (B, dg*kh*kw*2, H, W) laid out (dg, k, [dy, dx])."""
    assert clip == 1 and stride == 1 and dil == 1
    B, _, C2, H, W = kv.shape
    C, A, d = C2 // 2, kh * kw, C2 // 2 // heads
    off = offset.view(B, dg, A, 2, H, W)
    ki = torch.arange(kh, device=q.device).repeat_interleave(kw) - pad
    kj = torch.arange(kw, device=q.device).repeat(kh) - pad
    ys = torch.arange(H, device=q.device).view(1, 1, 1, H, 1) + ki.view(1, 1, A, 1, 1) + off[:, :, :, 0]
    xs = torch.arange(W, device=q.device).view(1, 1, 1, 1, W) + kj.view(1, 1, A, 1, 1) + off[:, :, :, 1]
    grid = torch.stack([xs * 2 / max(W - 1, 1) - 1, ys * 2 / max(H - 1, 1) - 1], -1)  # B,dg,A,H,W,2
    s = F.grid_sample(kv[:, 0].reshape(B * dg, C2 // dg, H, W), grid.reshape(B * dg, A * H, W, 2),
                      mode="bilinear", padding_mode="zeros", align_corners=True)  # B*dg, C2/dg, A*H, W
    s = s.view(B, 2, heads, d, A, H * W)
    qh = q.reshape(B, heads, d, H * W) * d ** -0.5
    attn = torch.einsum("bhdn,bhdan->bhan", qh, s[:, 0]).softmax(2)
    return torch.einsum("bhan,bhdan->bhdn", attn, s[:, 1]).reshape(B, 1, C, H, W)


def _load_repo_model(para):
    sys.path.insert(0, str(REPO))
    try:
        with mock.patch("torch.utils.cpp_extension.load", lambda *a, **k: None):
            from TM_model import Model
            import TM_model.DATUM as D
            import TM_model.op.deform_attn as op
        D.deform_attn = op.deform_attn = deform_attn_torch
        return Model(para)
    finally:
        sys.path.remove(str(REPO))


def windows(n, clip, margin):
    """Window starts covering [0, n) with `clip`-long windows overlapping by 2*margin."""
    if n <= clip:
        return [0]
    step = clip - 2 * margin
    starts = list(range(0, n - clip + 1, step))
    if starts[-1] + clip < n:
        starts.append(n - clip)
    return starts


class DATUM(BaseRestorer):
    def __init__(self, device="cuda", **over):
        cfg = {**yaml.safe_load((HERE / "config.yaml").read_text()), **over}
        self.cfg = cfg
        assert torch.cuda.is_available(), "DATUM code hardcodes cuda"
        para = Namespace(model="DATUM", spynet_path=None, output_full=True, **{
            k: cfg[k] for k in ("n_features", "n_blocks", "past_frames", "future_frames", "activation")})
        self.model = _load_repo_model(para)
        ck = torch.load(fetch(HERE), map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck.get("state_dict", ck))
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, crops):
        if not crops:
            return []
        n, s, clip, m = len(crops), self.cfg["size"], self.cfg["clip"], self.cfg["margin"]
        x = np.stack([cv2.resize(c, (s, s), interpolation=cv2.INTER_CUBIC)[..., ::-1] for c in crops])
        x = torch.from_numpy(x.copy()).to(self.device).permute(0, 3, 1, 2).float() / 255  # n,3,s,s RGB
        L = max(n, self.cfg["past_frames"] + self.cfg["future_frames"] + 1)  # model needs >= window frames
        x = torch.cat([x, x[-1:].expand(L - n, -1, -1, -1)]) if L > n else x
        out = torch.empty_like(x)
        for st in windows(L, clip, m):
            seq = x[st:st + clip][None]
            y, _ = self.model((seq, seq))
            k = m if st else 0
            out[st + k:st + seq.shape[1]] = y[0, k:].clamp(0, 1)
        out = (out[:n] * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
        return [cv2.resize(np.ascontiguousarray(o), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_AREA)
                for o, c in zip(out, crops)]

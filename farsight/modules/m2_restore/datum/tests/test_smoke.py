import glob

import cv2
import numpy as np
import pytest
import torch

from farsight.core.weights import ROOT
from farsight.modules.m2_restore.datum.model import deform_attn_torch, windows
from tools.degrade_dataset import degrade


def _ref_deform_attn(q, kv, off, k, heads, dg):
    """Literal loop port of the CUDA kernel (modulated im2col w/ DCN bilinear + softmax attention)."""
    B, _, C2, H, W = kv.shape
    C, A, pad = C2 // 2, k * k, k // 2
    d, cpg = C // heads, C2 // dg
    out = np.zeros((B, C, H, W))
    o = off.reshape(B, dg, A, 2, H, W)

    def bil(im, y, x):
        if not (y > -1 and x > -1 and y < H and x < W):
            return 0.0
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        v = 0.0
        for yy, wy in ((y0, 1 - (y - y0)), (y0 + 1, y - y0)):
            for xx, wx in ((x0, 1 - (x - x0)), (x0 + 1, x - x0)):
                if 0 <= yy < H and 0 <= xx < W:
                    v += wy * wx * im[yy, xx]
        return v

    for b in range(B):
        for i in range(H):
            for j in range(W):
                col = np.zeros((C2, A))
                for c in range(C2):
                    g = c // cpg
                    for a in range(A):
                        y = i - pad + a // k + o[b, g, a, 0, i, j]
                        x = j - pad + a % k + o[b, g, a, 1, i, j]
                        col[c, a] = bil(kv[b, 0, c], y, x)
                for h in range(heads):
                    qv = q[b, 0, h * d:(h + 1) * d, i, j] * d ** -0.5
                    s = qv @ col[h * d:(h + 1) * d]
                    s = np.exp(s - s.max()); s /= s.sum()
                    out[b, h * d:(h + 1) * d, i, j] = col[C + h * d:C + (h + 1) * d] @ s
    return out


def test_deform_attn_matches_reference():
    g = torch.Generator().manual_seed(0)
    B, C, H, W, k, heads, dg = 1, 8, 5, 6, 3, 2, 4
    q, kv = torch.randn(B, 1, C, H, W, generator=g), torch.randn(B, 1, 2 * C, H, W, generator=g)
    off = torch.randn(B, dg * k * k * 2, H, W, generator=g) * 2
    got = deform_attn_torch(q, kv, off, k, k, 1, k // 2, 1, heads, dg, 1)[:, 0].double().numpy()
    want = _ref_deform_attn(q.double().numpy(), kv.double().numpy(), off.double().numpy(), k, heads, dg)
    assert np.allclose(got, want, atol=1e-4)


def test_windows_cover():
    for n in (1, 5, 16, 17, 40, 41):
        st = windows(n, 16, 2)
        assert st[0] == 0 and st[-1] + min(16, n) >= n and all(b - a <= 12 for a, b in zip(st, st[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DATUM needs CUDA")
def test_restores_synthetic_turbulence():
    from farsight.core import registry
    r = registry.build("datum")
    im = cv2.imread(sorted(glob.glob(str(ROOT / "third_party/BPJDet/test_imgs/COCO/*.jpg")))[0])
    h, w = im.shape[:2]
    clean = cv2.resize(im[h // 2 - 150:h // 2 + 150, w // 2 - 150:w // 2 + 150], (256, 256), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(0)
    seq = [degrade(clean, ["turb:1.5", "gblur:1.0", "noise:3"], rng) for _ in range(20)]
    seq[3] = cv2.resize(seq[3], (240, 250))  # crops of varying size come back at their own size
    out = r(seq)
    assert len(out) == 20 and all(o.shape == s.shape and o.dtype == np.uint8 for o, s in zip(out, seq))
    psnr = lambda xs: np.mean([cv2.PSNR(clean, x) for i, x in enumerate(xs) if i != 3])
    print(f"PSNR degraded {psnr(seq):.2f} -> restored {psnr(out):.2f}")
    assert psnr(out) > psnr(seq) + 1.0
    assert r([seq[0]])[0].shape == seq[0].shape  # single-frame track

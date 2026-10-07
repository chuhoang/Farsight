import gc
import time

import cv2
import numpy as np
import pytest
import torch

from farsight.core import registry
from farsight.core.weights import load_manifest, weight_dir
from farsight.modules.m3_encode.gait import biggait
from farsight.modules.m3_encode.gait.biggait.preprocess import clip_starts, quality

HAVE_CKPT = (weight_dir(load_manifest(biggait.model.HERE)) / "BigGait__Dinov2_Gaitbase_Frame30-40000.pt").exists()


def walker(n, shirt, pants, leg=1.0, phase=0.0, h=220, w=100):
    """Synthetic BGR body-crop sequence: head, torso, swinging legs/arms on grey background."""
    out = []
    for t in range(n):
        a = 0.5 * np.sin(2 * np.pi * t / 24 + phase)
        im = np.full((h, w, 3), 120, np.uint8)
        cx = w // 2
        cv2.circle(im, (cx, 22), 14, (140, 170, 210), -1)
        cv2.rectangle(im, (cx - 18, 38), (cx + 18, 115), shirt, -1)
        hip = (cx, 115)
        for s in (a, -a):
            cv2.line(im, hip, (int(cx + 90 * leg * np.sin(s)), int(115 + 90 * leg * np.cos(s))), pants, 12)
            cv2.line(im, (cx, 45), (int(cx + 60 * np.sin(-s)), int(45 + 60 * np.cos(s))), shirt, 8)
        out.append(im)
    return out


A = walker(75, (40, 40, 200), (90, 50, 20))
B = walker(75, (30, 160, 30), (20, 20, 20), leg=0.8, phase=1.0)


def test_clip_and_quality_logic():
    assert clip_starts(20) == [0] and clip_starts(30) == [0]
    assert clip_starts(75) == [0, 30, 45] and clip_starts(61) == [0, 30, 31]
    crops = [np.zeros((h, 10, 3), np.uint8) for h in [200] * 30 + [50] * 30]
    assert quality(crops, 128, 60) == 0.5 and quality(crops[:30], 128, 60) == 0.5


@pytest.fixture(scope="module")
def enc():
    if not HAVE_CKPT:
        pytest.skip("BigGait checkpoint missing: fetch via farsight.core.weights.fetch(biggait dir)")
    e = registry.build("biggait")
    yield e
    del e
    gc.collect(); torch.cuda.empty_cache()


def test_encode(enc):
    out = enc.encode(A, None)
    assert out["feat"].shape == (4096,) and out["feat"].dtype == np.float32
    assert out["inter_feat"].shape == (768,) and out["n_frames"] == 75
    assert np.isfinite(out["feat"]).all() and np.isfinite(out["inter_feat"]).all()
    assert abs(np.linalg.norm(out["feat"]) - 1) < 1e-5 and 0 < out["quality"] <= 1
    np.testing.assert_allclose(enc.encode(A, None)["feat"], out["feat"], atol=2e-3)  # deterministic
    assert enc.encode(A[:14], None) is None
    assert enc.encode(A[:15], None)["feat"].shape == (4096,)


def test_discriminates(enc):
    fa, _ = enc.embed_clips(A)       # 3 clips of walker A
    fb, _ = enc.embed_clips(B)
    assert fa[0] @ fa[1] > fa[0] @ fb[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="speed only measured on GPU")
def test_speed(enc):
    crops = A[:30] * 8   # 240 frames -> 8 clips of 30
    enc.embed_clips(crops)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(3):
        enc.embed_clips(crops)
    torch.cuda.synchronize()
    cps = 3 * 8 / (time.perf_counter() - t)
    print(f"\nbiggait: {cps:.1f} track-clips/s (30 frames, batch {enc.cfg['clip_batch']})")
    assert cps > 1

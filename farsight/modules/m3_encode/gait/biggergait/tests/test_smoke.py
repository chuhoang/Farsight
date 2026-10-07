import gc
import time

import numpy as np
import pytest
import torch

from farsight.core import registry
from farsight.core.weights import load_manifest, weight_dir
from farsight.modules.m3_encode.gait import biggergait
from farsight.modules.m3_encode.gait.biggait.tests.test_smoke import A, B

CKPT = "BiggerGait__SmallDINOv2_Gaitbase_84Frame30_448224_6432HPP32_NoAlign_Sep12B_WiMask-30000.pt"
HAVE_CKPT = (weight_dir(load_manifest(biggergait.model.HERE)) / CKPT).exists()


@pytest.fixture(scope="module")
def enc():
    if not HAVE_CKPT:
        pytest.skip("BiggerGait checkpoint missing: fetch via farsight.core.weights.fetch(biggergait dir)")
    e = registry.build("biggergait")
    yield e
    del e
    gc.collect(); torch.cuda.empty_cache()


def test_encode(enc):
    out = enc.encode(A, None)
    assert out["feat"].shape == (98304,) and out["feat"].dtype == np.float32
    assert out["inter_feat"].shape == (768,) and out["n_frames"] == 75
    assert np.isfinite(out["feat"]).all() and np.isfinite(out["inter_feat"]).all()
    assert abs(np.linalg.norm(out["feat"]) - 1) < 1e-5 and 0 < out["quality"] <= 1
    np.testing.assert_allclose(enc.encode(A, None)["feat"], out["feat"], atol=2e-3)  # deterministic
    assert enc.encode(A[:14], None) is None


def test_discriminates(enc):
    fa, _ = enc.embed_clips(A)
    fb, _ = enc.embed_clips(B)
    assert fa[0] @ fa[1] > fa[0] @ fb[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="speed only measured on GPU")
def test_speed(enc):
    crops = A[:30] * 4
    enc.embed_clips(crops)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(3):
        enc.embed_clips(crops)
    torch.cuda.synchronize()
    cps = 3 * 4 / (time.perf_counter() - t)
    print(f"\nbiggergait: {cps:.1f} track-clips/s (30 frames, batch {enc.cfg['clip_batch']})")
    assert cps > 0.5

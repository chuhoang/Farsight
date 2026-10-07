import numpy as np
import pytest
import torch

from farsight.core import registry
from farsight.core.weights import load_manifest, weight_dir
from farsight.modules.m3_encode.body import aim

HAVE_CKPT = (weight_dir(load_manifest(aim.model.HERE)) / "ltcc-checkpoint.pth.tar").exists()
rng = np.random.default_rng(0)
CROPS = [rng.integers(0, 255, (int(rng.integers(80, 400)), int(rng.integers(40, 200)), 3), np.uint8) for _ in range(40)]
TRACK = {"video_id": "v", "track_id": 1, "frames": [{"frame_idx": i, "body_score": 0.8} for i in range(40)]}


@pytest.fixture(scope="module")
def enc():
    if not HAVE_CKPT:
        pytest.skip("AIM checkpoint missing: see README (weights/m3_encode/body/aim/ltcc-checkpoint.pth.tar)")
    return registry.build("aim")


def test_random_init_shapes():
    e = aim.AIMEncoder(pretrained=False, device="cpu", batch_size=4)
    f, inter = e.embed(CROPS[:3])
    assert f.shape == (3, 4096) and inter.shape == (3, 1024)
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5)


def test_encode_pretrained(enc):
    out = enc.encode(CROPS, TRACK)
    assert out["feat"].shape == (4096,) and out["feat"].dtype == np.float32
    assert np.isfinite(out["feat"]).all() and np.isfinite(out["inter_feat"]).all()
    assert abs(np.linalg.norm(out["feat"]) - 1) < 1e-5
    assert out["inter_feat"].shape == (1024,) and out["n_frames"] == 32
    assert 0 < out["quality"] <= 0.8
    np.testing.assert_allclose(enc.encode(CROPS, TRACK)["feat"], out["feat"], atol=1e-5)  # deterministic
    assert enc.encode([], TRACK) is None


def test_pretrained_discriminates(enc):
    f, _ = enc.embed(CROPS[:2] + [np.ascontiguousarray(CROPS[0][:, ::-1])])
    assert f[0] @ f[2] > f[0] @ f[1]  # mirrored crop closer than a different crop


def test_half_dim(enc):
    enc.cfg["half_dim"] = True
    try:
        f, _ = enc.embed(CROPS[:2])
    finally:
        enc.cfg["half_dim"] = False
    assert f.shape == (2, 2048)
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="speed only measured on GPU")
def test_speed(enc):
    import time
    crops = [np.full((256, 128, 3), 127, np.uint8)] * 64
    enc.embed(crops[:8])
    torch.cuda.synchronize(); t = time.perf_counter()
    enc.embed(crops)
    torch.cuda.synchronize()
    print(f"\nAIM: {64 / (time.perf_counter() - t):.1f} crops/s on {torch.cuda.get_device_name()}")

import numpy as np
import pytest
import torch

from farsight.core import registry
from farsight.core.weights import load_manifest, weight_dir
from farsight.modules.m3_encode.body import csci
from farsight.modules.m3_encode.body.csci.clips import recombine, model_clips

HAVE_CKPT = (weight_dir(load_manifest(csci.model.HERE)) / "csci_v_mevid.pth").exists()
rng = np.random.default_rng(0)
CROPS = [rng.integers(0, 255, (int(rng.integers(80, 400)), int(rng.integers(40, 200)), 3), np.uint8) for _ in range(40)]
TRACK = {"video_id": "v", "track_id": 1, "frames": [{"frame_idx": i, "body_score": 0.8} for i in range(40)]}


@pytest.mark.parametrize("n", [1, 3, 7, 8, 9, 31, 32, 33, 40, 77, 300])
def test_recombine_matches_cal(n):
    clips = recombine(n)
    assert len(clips) == -(-n // 8) and all(len(c) == 8 for c in clips)
    assert sorted({i for c in clips for i in c}) == list(range(n))           # every frame used
    if n >= 32:
        assert clips[0] == list(range(0, 32, 4)) and clips[3] == list(range(3, 32, 4))


def test_test_clips_budget():
    c = model_clips(300, max_clips=8)
    assert len(c) == 8 and all(len(x) == 4 for x in c)
    assert c[0] == [0, 8, 16, 24] and max(max(x) for x in c) >= 290      # spread over the whole track
    assert len(model_clips(300, max_clips=0)) == 38


def test_random_init_shapes():
    e = csci.CSCIEncoder(pretrained=False, device="cpu", batch_size=2, max_clips=2)
    f, inter = e.embed(CROPS[:20])
    assert f.shape == (2, 1024) and inter.shape == (2, 1024)
    out = e.encode(CROPS[:2], TRACK)                                        # < 4 frames -> image model
    assert out["feat"].shape == (1024,) and out["n_frames"] == 2 and e._image is not None


@pytest.fixture(scope="module")
def enc():
    if not HAVE_CKPT:
        pytest.skip("CSCI checkpoint missing: see csci/README.md")
    return registry.build("csci_video")


def test_encode_pretrained(enc):
    out = enc.encode(CROPS, TRACK)
    assert out["feat"].shape == (1024,) and out["feat"].dtype == np.float32
    assert np.isfinite(out["feat"]).all() and np.isfinite(out["inter_feat"]).all()
    assert abs(np.linalg.norm(out["feat"]) - 1) < 1e-5 and out["n_frames"] == 20   # 5 clips x 4 frames fed
    assert 0 < out["quality"] <= 0.8
    np.testing.assert_allclose(enc.encode(CROPS, TRACK)["feat"], out["feat"], atol=1e-4)   # deterministic
    assert enc.encode([], TRACK) is None


def test_pretrained_discriminates(enc):
    a, b = CROPS[:16], CROPS[16:32]
    fa, fb = (enc.encode(x, None)["feat"] for x in (a, b))
    fm = enc.encode([np.ascontiguousarray(c[:, ::-1]) for c in a], None)["feat"]
    assert fa @ fm > fa @ fb   # mirrored track closer than a different track


@pytest.mark.skipif(not torch.cuda.is_available(), reason="speed only measured on GPU")
def test_speed(enc):
    import time
    crops = [np.full((256, 128, 3), 127, np.uint8)] * 64          # 8 clips x 4 frames = 32 frames through EVA02-L
    enc.encode(crops, None)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.perf_counter()
    enc.encode(crops, None)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    print(f"\nCSCI video: {dt:.2f} s/track (8 clips), {32 / dt:.1f} frames/s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB on {torch.cuda.get_device_name()}")

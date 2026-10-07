import cv2
import numpy as np
import pytest

from farsight.core.weights import ROOT

# CVLface verification example: (0,1) same person, (2,3) and (4,5) different people
IMGS = ROOT / "third_party/CVLface/cvlface/apps/verification/example/images"
pytestmark = pytest.mark.skipif(not IMGS.exists(), reason="third_party/CVLface submodule missing")


@pytest.fixture(scope="module")
def model():
    from farsight.core.registry import build
    return build("kprpe", batch_size=4)  # small batches to exercise batching + keep GPU memory low


@pytest.fixture(scope="module")
def ims():
    return [cv2.imread(str(p)) for p in sorted(IMGS.iterdir())]


def test_forward_shapes_and_identity(model, ims):
    f, inter, score = model.forward(ims)
    assert f.shape == (6, 512) and inter.shape == (6, 2048) and score.shape == (6,)
    assert np.isfinite(f).all() and np.isfinite(inter).all()
    e = model.embed(ims)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(model.quality(ims), np.linalg.norm(f, axis=1), rtol=1e-4)
    same, diff = e[0] @ e[1], max(e[2] @ e[3], e[4] @ e[5])
    assert same > 0.4 and diff < 0.2 and same > diff + 0.3, (same, diff)  # CVLface README: 0.67 / -0.02 / 0.05


def test_deterministic_and_batch_invariant(model, ims):
    a = model.embed(ims)
    b = np.concatenate([model.embed(ims[i:i + 1]) for i in range(len(ims))])
    np.testing.assert_allclose(a, b, atol=1e-3)  # cuBLAS kernels vary with batch size
    np.testing.assert_allclose(a, model.embed(ims), atol=1e-6)


def test_encode_template(model, ims):
    track = {"video_id": "v", "track_id": 0, "frames": []}
    noise = np.random.default_rng(0).integers(0, 255, (112, 112, 3), np.uint8)
    t = model.encode([ims[0], ims[1], noise], track)
    assert t["n_frames"] == 2  # noise rejected by DFA face score
    assert t["feat"].shape == (512,) and t["feat"].dtype == np.float32
    assert abs(np.linalg.norm(t["feat"]) - 1) < 1e-5
    assert t["inter_feat"].shape == (2048,) and t["quality"] > 0
    e = model.embed(ims)
    assert t["feat"] @ e[0] > 0.7 and t["feat"] @ e[2] < 0.3  # template stays on identity 0/1
    assert model.encode([noise], track) is None
    assert model.encode([], track) is None

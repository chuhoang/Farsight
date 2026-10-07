import cv2
import numpy as np
import pytest

from tools import degrade_dataset as D


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    x = cv2.resize(rng.integers(0, 255, (12, 10, 3), dtype=np.uint8), (80, 96), interpolation=cv2.INTER_CUBIC)
    return np.clip(x.astype(int) + rng.integers(-20, 20, x.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("op", ["down:2", "down:4", "down:8", "gblur:1.5", "mblur:9:30", "turb:2", "noise:5", "jpeg:20"])
def test_op_keeps_shape_changes_image(img, op):
    out = D.degrade(img, [op], rng=0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img).mean() > 0.5


def test_chain_and_seed(img):
    ops = ["down:4", "turb:3", "jpeg:30"]
    assert np.array_equal(D.degrade(img, ops, rng=1), D.degrade(img, ops, rng=1))
    assert not np.array_equal(D.degrade(img, ops, rng=1), D.degrade(img, ops, rng=2))


def test_turb_strength_monotone(img):
    e = [np.abs(D.turb(img, s, rng=0).astype(int) - img).mean() for s in (0.5, 4)]
    assert e[0] < e[1]

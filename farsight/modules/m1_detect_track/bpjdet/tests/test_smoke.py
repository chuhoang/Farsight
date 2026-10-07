import glob

import cv2
import numpy as np
import pytest

from farsight.core import registry
from farsight.core.weights import ROOT

IMGS = sorted(glob.glob(str(ROOT / "third_party/BPJDet/test_imgs/CrowdHuman/*.jpg")))


@pytest.fixture(scope="module")
def det():
    return registry.build("bpjdet")


def test_crowdhuman_images(det):
    frames = [cv2.imread(p) for p in IMGS]
    out = det(frames)
    assert len(out) == len(frames)
    total_faces = 0
    for f, o in zip(frames, out):
        b, fc = o["body"], o["face"]
        assert b.shape[1] == 5 and fc.shape == b.shape and len(b) > 0
        h, w = f.shape[:2]
        assert (b[:, [0, 2]] >= 0).all() and (b[:, [0, 2]] <= w).all() and (b[:, [1, 3]] <= h).all()
        assert ((b[:, 4] > 0) & (b[:, 4] <= 1)).all()
        ok = ~np.isnan(fc[:, 0])
        total_faces += ok.sum()
        # matched face lies inside its body (inner-IoU > 0.6 enforced by decode)
        assert (fc[ok, 1] >= b[ok, 1] - 2).all() and (fc[ok, 3] <= b[ok, 3] + 2).all()
    assert total_faces > 0


def test_batch_matches_single_and_deterministic(det):
    # same-size frames batch together; batched result == one-by-one
    f = cv2.imread(IMGS[0])
    batch = det([f, f[:, ::-1].copy(), f])
    single = det([f])[0]
    np.testing.assert_allclose(batch[0]["body"], single["body"], atol=1.0)
    np.testing.assert_allclose(batch[0]["body"], batch[2]["body"], atol=1e-3)


def test_empty_frame(det):
    o = det([np.zeros((720, 1280, 3), np.uint8)])[0]
    assert o["body"].shape == (0, 5) and o["face"].shape == (0, 5)

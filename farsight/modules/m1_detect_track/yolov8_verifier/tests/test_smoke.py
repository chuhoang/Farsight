import glob

import cv2
import numpy as np
import pytest

from farsight.core import registry
from farsight.core.weights import ROOT

IMG = sorted(glob.glob(str(ROOT / "third_party/BPJDet/test_imgs/CrowdHuman/*.jpg")))[1]


@pytest.fixture(scope="module")
def ver():
    return registry.build("yolov8_verifier")


def test_persons_detected_deterministic(ver):
    f = cv2.imread(IMG)
    a, b = ver([f, f])
    assert a["body"].shape[1] == 5 and len(a["body"]) >= 3
    assert (a["body"][:, 4] >= ver.cfg["conf"]).all()
    np.testing.assert_allclose(a["body"], b["body"], atol=1e-3)


def test_verify_logic(ver):
    f = cv2.imread(IMG)
    y = ver([f])[0]["body"]
    fake = np.array([[0, 0, 5, 5, 0.9]], np.float32)       # nowhere near a person
    shifted = y[:1].copy()
    shifted[0, :4] += [3, 3, 3, 3]                          # IoU ~0.9 with a real person
    det = {"body": np.concatenate([shifted, fake]), "face": np.array([[1, 2, 3, 4, .8], [np.nan] * 5], np.float32)}
    out = ver.verify([f], [det])[0]
    assert len(out["body"]) == 1 and np.allclose(out["body"], shifted) and np.allclose(out["face"][0], [1, 2, 3, 4, .8])
    empty = ver.verify([np.zeros((480, 640, 3), np.uint8)], [det])[0]
    assert empty["body"].shape == (0, 5) and empty["face"].shape == (0, 5)

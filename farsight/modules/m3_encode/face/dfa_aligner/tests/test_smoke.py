from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from farsight.core.weights import ROOT

IMGS = ROOT / "third_party/CVLface/cvlface/apps/verification/example/images"
pytestmark = pytest.mark.skipif(not IMGS.exists(), reason="third_party/CVLface submodule missing")


def faces():
    from skimage.data import astronaut
    ims = [cv2.imread(str(p)) for p in sorted(IMGS.iterdir())]
    ims.append(np.ascontiguousarray(astronaut()[20:230, 120:300, ::-1]))  # non-square RGB->BGR face crop
    return ims


@pytest.fixture(scope="module")
def aligner():
    from farsight.core.registry import build
    return build("dfa_aligner")


def test_align_real_faces(aligner):
    ims = faces()
    out = aligner(ims)
    n = len(ims)
    assert out["aligned"].shape == (n, 3, 112, 112) and out["ldmk_aligned"].shape == (n, 5, 2)
    assert out["ldmk"].shape == (n, 5, 2) and out["score"].shape == (n,)
    assert (out["score"] > 0.9).all(), out["score"]
    for im, l in zip(ims, out["ldmk"]):  # landmarks: L/R eye, nose, L/R mouth, inside the crop
        h, w = im.shape[:2]
        assert (l[:, 0] > 0).all() and (l[:, 0] < w).all() and (l[:, 1] > 0).all() and (l[:, 1] < h).all()
        assert l[0, 0] < l[1, 0] and l[3, 0] < l[4, 0]
        assert max(l[0, 1], l[1, 1]) < l[2, 1] < min(l[3, 1], l[4, 1])
    # aligned landmarks land near the canonical 112x112 template (ArcFace 5-point reference)
    ref = np.array([[38.29, 51.70], [73.53, 51.50], [56.03, 71.74], [41.55, 92.37], [70.73, 92.20]]) / 112
    assert np.abs(out["ldmk_aligned"].cpu().numpy() - ref).max() < 0.08


def test_deterministic_and_batch_invariant(aligner):
    ims = faces()
    a, b = aligner(ims), aligner(ims[:1])
    assert torch.allclose(a["aligned"][:1], b["aligned"], atol=1e-4)
    np.testing.assert_allclose(a["ldmk"][:1], b["ldmk"], atol=1e-3)


def test_warp_with_own_theta_reproduces_align(aligner):
    ims = faces()
    a = aligner(ims)
    torch.testing.assert_close(aligner.warp(ims, a["theta"]), a["aligned"], atol=1e-4, rtol=0)

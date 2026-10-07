import numpy as np
import pytest
import torch

from farsight.core import registry
from farsight.core.weights import load_manifest, weight_dir
from farsight.modules.m4_fusion.module import cosine_scores, stack
from farsight.modules.m4_fusion.qe.model import QualityEstimator, style_from_blocks
from farsight.modules.m4_fusion.qe.train import pseudo_labels, train
from farsight.modules.m4_fusion.zscore.tests.synth import centers, label, templates

QE_DIR = "farsight/modules/m4_fusion/qe"
_M = load_manifest(QE_DIR)
HAVE_CKPT = (weight_dir(_M) / _M["checkpoints"][0]["file"]).exists()


def test_pseudo_labels():
    S = np.array([[.9, .1, .2], [.1, .9, .2], [.3, .2, .1]])
    # true-id ranks: 1, 3, inf (non-mated)
    y = pseudo_labels(S, ["a", "a", "z"], ["a", "b", "c"], rank_threshold=3)
    assert np.allclose(y, [1.0, 0.0, 0.0])


def test_pseudo_labels_cal_rule():
    # gallery: a@cam0 (same-camera near-duplicate), a@cam1, b@cam1
    S = np.array([[.95, .50, .60],     # duplicate ignored -> genuine .50 ranks 2 behind b -> 0.5
                  [.95, .80, .60],     # other-camera genuine on top -> 1
                  [.95, .10, .60]])    # duplicate ignored -> genuine .10 behind b -> rank 2 -> 0.5
    q, g = ["a", "a", "a"], ["a", "a", "b"]
    assert np.allclose(pseudo_labels(S, q, g), [1, 1, 1])                       # plain rule: all "good"
    assert np.allclose(pseudo_labels(S, q, g, 3, [0, 0, 0], [0, 1, 1]), [0.5, 1, 0.5])
    assert np.allclose(pseudo_labels(S[:1], ["a"], g, 3, [0], [0, 0, 1]), [0])   # no other-camera genuine


def test_train_qe_learns_quality():
    rng = np.random.default_rng(0)
    C = centers(80, rng)
    gal = templates(C, range(80), rng, noise={m: (0.3, 0.3) for m in C}, prefix="g")
    prb = templates(C, list(range(80)) * 4, rng, noise={"face": (0.5, 6.0), "gait": (1, 1), "body": (1, 1)}, prefix="p")
    P, G = stack(prb), stack(gal)
    S = cosine_scores(P, G)[..., 0]
    X = np.stack([t["face"]["inter_feat"] for t in prb])  # synthetic inter_feat encodes the true noise level
    net = train(X, S, [label(i) for i in P["ids"]], [label(i) for i in G["ids"]], rank_threshold=3, steps=600,
                lr=1e-3, log_every=0)
    with torch.no_grad():
        w = torch.sigmoid(net(torch.tensor(X))).numpy()
    assert np.corrcoef(w, X.mean(1))[0, 1] < -0.6     # noisier face -> lower quality weight


def test_style_shapes_and_frame_mean():
    toks = [torch.randn(3, 196, 512) for _ in range(2)]
    s = style_from_blocks(toks)
    assert s.shape == (3, 2048)
    qe = QualityEstimator(checkpoint=None)
    raw = torch.stack(toks, 1).numpy()                        # (N, blk, P, C)
    assert np.isclose(qe(raw), qe(s.numpy()), atol=1e-6)      # raw-token path == style path
    w = qe.batch(np.vstack([s[:1].numpy(), np.full((1, 2048), np.nan)]))
    assert 0 < w[0] < 1 and w[1] == 0


@pytest.mark.skipif(not HAVE_CKPT, reason="QE checkpoint not downloaded: farsight.core.weights.fetch(qe dir)")
def test_released_checkpoint_loads():
    qe = registry.build("qe")
    assert 0 < qe(np.random.randn(2048)) < 1
    qe_from_qme = registry.build("qe", checkpoint=str(weight_dir(_M).parent / "qme/lsf-ccvid-bs8-seq8-245.92-630.pth"))
    assert 0 < qe_from_qme(np.random.randn(2048)) < 1

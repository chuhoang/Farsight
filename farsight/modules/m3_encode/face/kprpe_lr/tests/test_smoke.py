import numpy as np
import torch

from farsight.modules.m3_encode.face.kprpe_lr.degrade import ALIGNED_IPD, degrade
from farsight.modules.m3_encode.face.kprpe_lr.train import Pool, Samples


def test_degrade_scale_and_determinism():
    img = np.random.default_rng(0).integers(0, 255, (112, 112, 3), np.uint8)
    a, p = degrade(img, 30.0, np.random.default_rng(1))
    b, _ = degrade(img, 30.0, np.random.default_rng(1))
    assert a.shape == img.shape and a.dtype == np.uint8 and np.array_equal(a, b)
    assert 2.0 <= p["ipd"] <= 30.0
    _, p = degrade(img, 1.0, np.random.default_rng(2))          # crop already below ipd_min: stays at ipd_min
    assert p["ipd"] == 2.0 and ALIGNED_IPD > 30


def test_samples_flip_and_plan():
    pool = Pool.__new__(Pool)
    pool.names, pool.ids = ["a", "b"], ["a:0", "a:1", "b:0"]
    pool.img = np.zeros((6, 112, 112, 3), np.uint8)
    pool.img[:, :, :56] = 255                                    # left half white
    pool.ldmk = np.tile(np.array([[.3, .4], [.7, .4], [.5, .6], [.35, .8], [.65, .8]], np.float32), (6, 1, 1))
    pool.ipd = np.full(6, 30, np.float32)
    pool.label, pool.src = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 0, 0, 1, 1])
    pool.by_label = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    pool.labels_of = [[0, 1], [2]]
    idx = pool.plan(2000, {"a": 1, "b": 0}, np.random.default_rng(0))
    assert set(pool.src[idx]) == {0}                              # weight 0 -> never sampled
    s = Samples(pool, idx, [0.0, 0.0], {"ldmk_noise": 0}, 0)
    flipped = 0
    for i in range(40):
        clean, deg, ldmk, ldmk_s, y, d = s[i]
        assert not d and torch.equal(clean, deg) and torch.equal(ldmk, ldmk_s)
        if clean[0, 0, 0] < 0:                                    # image mirrored -> landmarks mirrored too
            flipped += 1
            assert torch.allclose(ldmk[0], torch.tensor([.3, .4])) and torch.allclose(ldmk[3], torch.tensor([.35, .8]))
    assert 5 < flipped < 35

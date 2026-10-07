"""Decision logic with fakes: crops are 1x1 'images' whose pixel value encodes identity signal."""
import numpy as np

from farsight.modules.m2_restore.module import M2Restore
from farsight.modules.m2_restore.quality_gate.model import QualityGate

ID = np.array([1.0, 0, 0])


def embed(crops):  # identity direction + noise that grows as crop value (quality) drops
    rng = np.random.default_rng(0)
    e = np.stack([ID + (1 - c.mean() / 255) * rng.normal(size=3) for c in crops])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def quality(crops):
    return np.array([c.mean() / 255 * 40 for c in crops])  # pseudo feature norm


def crops(v, n=8):
    return [np.full((4, 4, 3), v, np.uint8) for _ in range(n)]


def build(restorer, enabled=True, q0=20):
    calls = []
    def r(cs):
        calls.append(len(cs))
        return restorer(cs)
    m = M2Restore(quality, embed, cfg={"enabled": enabled, "gate": "quality_gate", "restorer": "datum"},
                  gate=QualityGate(q0=q0), restorer=r)
    return m, calls


helps = lambda cs: [np.full_like(c, 250) for c in cs]
hurts = lambda cs: [np.full_like(c, 10) for c in cs]


def test_disabled_passthrough():
    m, calls = build(helps, enabled=False)
    c = crops(50)
    out, info = m(c)
    assert out is c and not info["restored"] and calls == []


def test_good_quality_skips_restore():
    m, calls = build(helps)
    out, info = m(crops(200))  # quality 31 >= 20
    assert not info["restored"] and info["reason"] == "gate_pass" and calls == []


def test_low_quality_restorer_helps():
    m, calls = build(helps)
    c = crops(50)  # quality ~7.8 < 20
    out, info = m(c)
    assert info["restored"] and calls == [8] and out[0].mean() == 250
    assert info["sim_restored"] > info["sim_orig"]


def test_low_quality_restorer_hurts_keeps_originals():
    m, _ = build(hurts)
    c = crops(50)
    out, info = m(c)
    assert out is c and not info["restored"] and info["reason"] == "safety_reject"


def test_empty_track():
    m, calls = build(helps)
    assert m([])[0] == [] and calls == []

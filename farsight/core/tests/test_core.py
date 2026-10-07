import torch
import yaml

from farsight.core import registry, weights
from farsight.core.hooks import LayerCapture, pool_tokens


def test_registry_roundtrip():
    @registry.register("_dummy")
    class D:
        def __init__(self, x=1):
            self.x = x
    assert registry.build("_dummy", x=3).x == 3
    try:
        registry.get("nope")
        assert False
    except KeyError:
        pass


def test_layer_capture_single_forward():
    m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    x = torch.randn(3, 4)
    with LayerCapture(m[1]) as cap:
        y = m(x)
    assert cap.out.shape == (3, 8) and y.shape == (3, 2)
    assert torch.allclose(cap.out, torch.relu(m[0](x)))
    assert len(m[1]._forward_hooks) == 0  # hook removed


def test_pool_tokens():
    t = torch.arange(24.).view(1, 3, 8)
    assert torch.equal(pool_tokens(t, "cls"), t[:, 0])
    assert torch.equal(pool_tokens(t, "mean"), t[:, 1:].mean(1))
    assert pool_tokens(torch.ones(2, 5, 3, 3)).shape == (2, 5)


def test_weights_fetch_and_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "WEIGHTS", tmp_path / "w")
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    good = weights.sha256(src)
    mdir = tmp_path / "model"
    mdir.mkdir()
    man = {"name": "x", "module": "m", "checkpoints": [{"file": "a.bin", "url": src.as_uri(), "sha256": good}]}
    (mdir / "manifest.yaml").write_text(yaml.safe_dump(man))
    p = weights.fetch(mdir)
    assert p.read_bytes() == b"abc"
    man["checkpoints"][0]["sha256"] = "0" * 64
    (mdir / "manifest.yaml").write_text(yaml.safe_dump(man))
    try:
        weights.fetch(mdir)
        assert False
    except ValueError:
        pass

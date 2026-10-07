import h5py
import numpy as np
import pytest

from farsight.io.store import write_templates
from farsight.modules.m4_fusion.zscore.tests.synth import centers, templates
from tools.export_scores import export
from tools.make_splits import check_disjoint, make_split


def test_export_scores(tmp_path):
    rng = np.random.default_rng(0)
    C = centers(6, rng)
    gal = templates(C, [0, 0, 1, 2, 3], rng, noise={m: (0.1, 0.1) for m in C}, prefix="g")
    prb = templates(C, [0, 1, 5], rng, noise={m: (0.1, 0.1) for m in C}, prefix="p")
    prb[1]["face"] = None
    write_templates(tmp_path / "g.h5", gal, mode="w")
    write_templates(tmp_path / "p.h5", prb, mode="w")
    export(tmp_path / "p.h5", tmp_path / "g.h5", tmp_path / "s.h5")
    with h5py.File(tmp_path / "s.h5", "r") as f:
        S = f["face/score_mat"][()]
        assert S.shape == (3, 5) and f["face/merge_score_mat"].shape == (3, 4)
        names = [x.decode() for x in f["pid_names"][()]]
        q = [names[i] for i in f["q_pids"][()]]
        g = [names[i] for i in f["g_pids"][()]]
        assert q == ["0000", "0001", "0005"] and g[:2] == ["0000", "0000"]
        assert np.isnan(S[1]).all() and np.isnan(f["face/inter_feat"][1]).all()
        assert S[0, :2].min() > S[0, 2:].max()          # genuine above impostors
        assert f["body/inter_feat"].shape == (3, 16)
        assert (f["q_camids"][()] != f["g_camids"][()][0]).all()


def test_make_split():
    s = make_split("ccvid", range(75), range(100, 251), seed=0)
    assert len(s["train"]) == 60 and len(s["val"]) == 15 and len(s["test"]) == 151   # whole test kept for test
    assert make_split("ccvid", range(75), range(100, 251), seed=0) == s   # deterministic
    m = make_split("mevid", range(104), range(200, 254))
    assert len(m["val"]) == 20 and len(m["train"]) == 84 and len(m["test"]) == 54   # val held out of train IDs
    assert set(m["val"]) | set(m["train"]) == set(map(str, range(104)))
    assert make_split("mevid", range(104), range(200, 254)) == m                     # deterministic
    with pytest.raises(AssertionError):
        make_split("mevid", ["a", "b"], ["b", "c"])
    with pytest.raises(AssertionError):
        check_disjoint({"dataset": "x", "train": ["1"], "val": [], "test": ["1"]})

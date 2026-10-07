import time

import numpy as np

from eval.metrics import evaluate
from farsight.core import registry
from farsight.modules.m4_fusion.module import cosine_scores, stack
from farsight.modules.m4_fusion.zscore.tests.synth import centers, label, templates


def _setup(seed=0, n_ids=60, p_missing=None):
    rng = np.random.default_rng(seed)
    C = centers(n_ids, rng)
    gal = templates(C, range(n_ids // 2), rng, noise={m: (0.3, 0.3) for m in C}, prefix="g")  # clean gallery
    prb = templates(C, list(range(n_ids)) * 2, rng, p_missing=p_missing, prefix="p")  # half non-mated
    return gal, prb


def _mats(gal, prb):
    G, P = stack(gal), stack(prb)
    return (cosine_scores(P, G), P["quality"], [label(i) for i in P["ids"]], [label(i) for i in G["ids"]])


def test_fit_grid_tau_and_missing(tmp_path):
    f = registry.build("zscore", stats=str(tmp_path / "s.json"))
    gal, val = _setup(0, p_missing={"face": 0.3, "gait": 0.2})
    S, q, qi, gi = _mats(gal, val)
    f.fit(S, qi, gi)
    imp = np.array(qi)[:, None] != np.array(gi)[None]
    z = (S - f.mu) / f.sd
    assert abs(np.nanmean(z[..., 0][imp])) < 1e-3 and abs(np.nanstd(z[..., 0][imp]) - 1) < 1e-3
    w, best = f.grid_search(S, q, qi, gi, metric="rank1")
    singles = [evaluate(np.where(np.isnan(S[..., k]), np.nan, S[..., k]), qi, gi)["rank1"] for k in range(3)]
    assert best >= max(singles), (best, singles)
    tau = f.set_tau(S, q, qi, gi, fpir=0.01)
    top1 = np.nan_to_num(f.fuse(S, q), nan=-np.inf).max(1)
    nonmated = ~np.isin(qi, gi)
    assert (top1[nonmated] > tau).mean() <= 0.01
    f.save()
    g = registry.build("zscore", stats=str(tmp_path / "s.json"))
    assert np.allclose(g.w, f.w) and g.tau == f.tau

    # every missing-modality combination still yields a result; all-missing -> nothing known
    p = val[0]
    for drop in [(), ("face",), ("face", "gait"), ("gait", "body"), ("face", "gait", "body")]:
        r = f.search({**p, **{m: None for m in drop}}, gal)
        assert len(r["ranked"]) == min(f.top_k, len(gal))
        if len(drop) == 3:
            assert not r["is_known"] and np.isnan(r["ranked"][0]["score"])
        else:
            s = [x["score"] for x in r["ranked"]]
            assert s == sorted(s, reverse=True)
            assert all(np.isnan(x["per_modality"][["face", "gait", "body"].index(m)]) for m in drop for x in r["ranked"])


def test_gallery_10k_under_100ms():
    rng = np.random.default_rng(1)
    C = {m: rng.standard_normal((10000, d)).astype(np.float32) for m, d in {"face": 512, "gait": 256, "body": 2048}.items()}
    gal = [{"subject_or_track_id": str(i), "qe_weight": None,
            **{m: {"feat": C[m][i], "quality": 1.0, "n_frames": 1, "inter_feat": None} for m in C}} for i in range(10000)]
    gal[5]["face"] = None
    f = registry.build("zscore", stats="__none__.json")
    probe = {**gal[123], "subject_or_track_id": "p"}
    f.search(probe, gal)  # warm gallery cache (one-off stacking, done at enrolment time)
    t = time.perf_counter()
    for _ in range(10):
        r = f.search(probe, gal)
    dt = (time.perf_counter() - t) / 10
    print(f"10k gallery search: {dt * 1e3:.1f} ms")
    assert r["ranked"][0]["gallery_id"] == "123" and dt < 0.1


def test_m4_module_builds_by_name():
    from farsight.modules.m4_fusion.module import M4
    gal, prb = _setup(2, n_ids=20)
    for cfg in ({"method": "zscore", "stats": "__none__.json"}, {"method": "qme"}):
        r = M4(cfg).search(prb[0], gal)
        assert r["probe_id"] == prb[0]["subject_or_track_id"] and len(r["ranked"]) == len(gal)

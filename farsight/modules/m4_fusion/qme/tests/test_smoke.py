import numpy as np
import pytest
import torch

from eval.metrics import evaluate
from farsight.core import registry
from farsight.core.weights import load_manifest, weight_dir
from farsight.modules.m4_fusion.module import cosine_scores, stack
from farsight.modules.m4_fusion.qme.model import QMEHead
from farsight.modules.m4_fusion.qme.train import train
from farsight.modules.m4_fusion.zscore.tests.synth import centers, label, templates

_M = load_manifest("farsight/modules/m4_fusion/qme")
CKPT = weight_dir(_M) / _M["checkpoints"][0]["file"]
NOISE = {"face": (0.3, 4.0), "gait": (2.5, 2.5), "body": (1.5, 1.5)}   # face: strong but variable; gait noisy
MISS = {"face": 0.4, "gait": 0.2}


def _split(rng, C, n_ids, reps, prefix):
    t = templates(C, list(range(n_ids)) * reps, rng, noise=NOISE, p_missing=MISS, prefix=prefix)
    P = stack(t)
    face_w = np.nan_to_num(np.clip((P["quality"][:, 0] - 0.2) / 0.6, 0, 1))   # stand-in for QE output
    return P, face_w


def test_gates_reduce_to_qme():
    h = QMEHead(n_experts=2)
    w = torch.tensor([0.0, 0.3, 1.0])
    assert torch.allclose(h.gates(w), torch.stack([w, 1 - w], 1))
    g = QMEHead(n_experts=4).gates(torch.rand(10))
    assert torch.allclose(g.sum(1), torch.ones(10))


def test_train_synthetic_beats_single_modalities():
    rng = np.random.default_rng(0)
    n_ids = 60
    C = centers(n_ids + 20, rng)
    gal = stack(templates(C, range(n_ids), rng, noise={m: (0.3, 0.3) for m in C}, prefix="g"))
    gi = [label(i) for i in gal["ids"]]
    tr, wtr = _split(rng, C, n_ids, 4, "tr")
    te, wte = _split(rng, C, n_ids + 20, 3, "te")           # 20 identities not enrolled -> open-set distractors
    S_tr, S_te = cosine_scores(tr, gal), cosine_scores(te, gal)
    q_tr = np.array([int(label(i)) for i in tr["ids"]])
    q_te = [label(i) for i in te["ids"]]
    head, _ = train(S_tr, wtr, q_tr, np.array([int(x) for x in gi]), steps=400, lr=1e-3, eval_every=0)
    fq = registry.build("qme")
    fq.head = head
    fused = fq.fuse(S_te, wte)
    res = {"qme": evaluate(fused, q_te, gi)}
    for k, m in enumerate(("face", "gait", "body")):
        res[m] = evaluate(S_te[..., k], q_te, gi)
    z = registry.build("zscore", stats="__none__.json").fit(S_tr, q_tr.astype(str).tolist(), gi)
    res["zscore"] = evaluate(z.fuse(S_te, te["quality"]), q_te, gi)
    for k, v in res.items():
        print(f"{k:7s} " + " ".join(f"{m}={x:.3f}" for m, x in v.items() if not m.startswith("tau")))
    best = max(res[m]["rank1"] for m in ("face", "gait", "body"))
    assert res["qme"]["rank1"] >= best
    assert res["qme"]["tar@0.01far"] >= max(res[m]["tar@0.01far"] for m in ("face", "gait", "body"))

    # search(): every missing combination works
    fq.set_tau(S_te, wte, q_te, gi)
    tmpl = templates(C, [0], rng, prefix="s")[0]
    glist = templates(C, range(n_ids), np.random.default_rng(5), noise={m: (0.3, 0.3) for m in C}, prefix="g")
    for drop in [(), ("face",), ("face", "gait"), ("face", "gait", "body")]:
        r = fq.search({**tmpl, **{m: None for m in drop}}, glist)
        assert len(r["ranked"]) == min(fq.top_k, n_ids)
        assert r["is_known"] == (len(drop) < 3 and r["ranked"][0]["score"] >= fq.tau)


def test_face_gate_and_scale():
    torch.manual_seed(0)
    s = torch.rand(4, 3, 6)
    w = torch.tensor([0.2, 0.9, 0.2, 0.9])
    no_face = s.clone()
    no_face[:, 0] = float("nan")
    h = QMEHead(face_gate=0.5).eval()
    out, ref = h(s, w), h(no_face, torch.zeros(4))
    torch.testing.assert_close(out[[0, 2]], ref[[0, 2]])           # W < t -> exactly the face-missing output
    assert not torch.allclose(out[[1, 3]], ref[[1, 3]])            # W >= t -> face used
    h = QMEHead(face_scale=True).eval()
    s2 = s.clone()
    s2[:, 0] = torch.rand(4, 6)
    w0 = torch.tensor([0.0, 0.0, 1.0, 1.0])
    a, b = h(s, w0), h(s2, w0)
    torch.testing.assert_close(a[:2], b[:2])                       # W = 0 -> face scores have no effect
    assert not torch.allclose(a[2:], b[2:])


def test_body_noise_trains_and_leaves_val_untouched():
    rng = np.random.default_rng(1)
    C = centers(30, rng)
    gal = stack(templates(C, range(30), rng, noise={m: (0.3, 0.3) for m in C}, prefix="g"))
    gi = np.array([int(label(i)) for i in gal["ids"]])
    tr, wtr = _split(rng, C, 30, 2, "tr")
    va, wva = _split(rng, C, 30, 1, "va")
    S_tr, S_va = cosine_scores(tr, gal), cosine_scores(va, gal)
    q = lambda P: np.array([int(label(i)) for i in P["ids"]])  # noqa: E731
    keep = S_va.copy()
    head, best = train(S_tr, wtr, q(tr), gi, val=(S_va, wva, q(va), gi), steps=60, lr=1e-3, eval_every=30,
                       body_noise=1.0)
    np.testing.assert_array_equal(np.nan_to_num(S_va, nan=-9), np.nan_to_num(keep, nan=-9))   # noisy copy only
    assert best and np.isfinite(best["val0_rank1"])


@pytest.mark.skipif(not CKPT.exists(), reason="QME 2A checkpoint not downloaded")
def test_released_2a_checkpoint_loads():
    f = registry.build("qme", checkpoint=_M["checkpoints"][0]["file"], use_mask=False)
    S = np.random.rand(2, 5, 3).astype(np.float32)
    S[1, :, 0] = np.nan                                       # probe 1 has no face
    out = f.fuse(S, [0.8, 0.8])
    assert out.shape == (2, 5) and np.isfinite(out).all()

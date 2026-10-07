import numpy as np

from eval.metrics import cmc, evaluate, fnir_at_fpir, tar_at_far, threshold_at_rate

# gallery ids: A A B C ; probes: A, B, C, D(non-mated)
G = ["A", "A", "B", "C"]
Q = ["A", "B", "C", "D"]
S = np.array([
    [0.2, 0.9, 0.5, 0.1],   # A: best genuine 0.9 -> rank 1
    [0.8, 0.3, 0.6, 0.2],   # B: genuine 0.6, one impostor 0.8 above -> rank 2
    [0.7, 0.6, 0.5, np.nan],  # C: genuine missing -> -inf -> rank 4 (3 impostors above)
    [0.4, 0.3, 0.35, 0.1],  # D: non-mated, top1 0.4
])


def test_threshold():
    assert threshold_at_rate([1, 2, 3, 4], 0.0) == 4      # nothing strictly above max
    assert threshold_at_rate([1, 2, 3, 4], 0.25) == 3     # only 4 above
    assert threshold_at_rate([1, 2, 3, 4], 1.0) == -np.inf


def test_cmc():
    r = cmc(S, Q, G, ranks=(1, 2, 4))
    assert r == {1: 1 / 3, 2: 2 / 3, 4: 1.0}


def test_tar_at_far():
    # genuine: .2 .9 .6 nan(-inf) ; 12 impostors -> FAR 1/12 threshold = 2nd highest impostor
    imp = sorted([.5, .1, .8, .3, .2, .7, .6, .5, .4, .3, .35, .1], reverse=True)
    r = tar_at_far(S, Q, G, fars=(0.0, 1 / 12))
    assert imp[0] == 0.8 and imp[1] == 0.7
    assert r[0.0] == 1 / 4            # only 0.9 > 0.8
    assert r[1 / 12] == 1 / 4         # 0.9 > 0.7, 0.6 not


def test_fnir():
    # one non-mated probe, FPIR 0 -> tau = 0.4. Mated: A hit1 & .9>.4 ok ; B miss rank ; C miss
    fnir, tau = fnir_at_fpir(S, Q, G, fpirs=(0.0,))[0.0]
    assert tau == 0.4 and np.isclose(fnir, 2 / 3)
    r = evaluate(S, Q, G)
    assert {"tar@0.001far", "tar@0.01far", "rank1", "rank5", "rank20", "fnir@0.01fpir"} <= set(r)

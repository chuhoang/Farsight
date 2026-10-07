"""Biometric metrics from a probe x gallery score matrix (plan 5.3). numpy only.

S: (P, G) similarity, higher = more similar, NaN = no score (treated as -inf).
q_ids: (P,) probe identities, g_ids: (G,) gallery identities (a gallery ID may have several templates).
Probes whose ID is absent from the gallery are non-mated (open-set distractors).
"""
import numpy as np


def _prep(S, q_ids, g_ids):
    S = np.nan_to_num(np.asarray(S, np.float64), nan=-np.inf)
    q, g = np.asarray(q_ids), np.asarray(g_ids)
    return S, q[:, None] == g[None, :]


def threshold_at_rate(neg, rate):
    """Smallest threshold t such that the fraction of `neg` scores strictly above t is <= rate.
    Accept rule everywhere: score > t."""
    neg = np.sort(np.asarray(neg, np.float64))[::-1]
    if neg.size == 0:
        return -np.inf
    k = int(np.floor(rate * neg.size))
    return neg[k] if k < neg.size else -np.inf


def tar_at_far(S, q_ids, g_ids, fars=(1e-3, 1e-2)):
    """Verification on all probe-gallery pairs: {far: TAR}."""
    S, mask = _prep(S, q_ids, g_ids)
    gen, imp = S[mask], S[~mask]
    return {f: float((gen > threshold_at_rate(imp, f)).mean()) if gen.size else float("nan") for f in fars}


def first_match_rank(S, q_ids, g_ids):
    """1-based rank of the best-scoring mated gallery entry per probe (inf for non-mated probes).
    Ties are counted pessimistically against the probe."""
    S, mask = _prep(S, q_ids, g_ids)
    best = np.where(mask, S, -np.inf).max(1)
    rank = ((S >= best[:, None]) & ~mask).sum(1) + 1.0
    rank[~mask.any(1)] = np.inf
    return rank


def cmc(S, q_ids, g_ids, ranks=(1, 5, 20)):
    """Closed-set identification over mated probes: {k: Rank-k accuracy}."""
    r = first_match_rank(S, q_ids, g_ids)
    r = r[np.isfinite(r)]
    return {k: float((r <= k).mean()) if r.size else float("nan") for k in ranks}


def fnir_at_fpir(S, q_ids, g_ids, fpirs=(1e-2,)):
    """Open-set identification: tau from non-mated probes' top-1 scores so FPIR <= fpir;
    a mated probe is missed unless its rank-1 is correct AND top-1 > tau. Returns {fpir: (fnir, tau)}."""
    S0, mask = _prep(S, q_ids, g_ids)
    mated = mask.any(1)
    top1 = S0.max(1)
    hit1 = first_match_rank(S, q_ids, g_ids) == 1
    out = {}
    for f in fpirs:
        tau = threshold_at_rate(top1[~mated], f)
        fnir = 1.0 - (hit1[mated] & (top1[mated] > tau)).mean() if mated.any() else float("nan")
        out[f] = (float(fnir), float(tau))
    return out


def evaluate(S, q_ids, g_ids):
    """All plan-5.3 numbers in one flat dict."""
    r = {f"tar@{f:g}far": v for f, v in tar_at_far(S, q_ids, g_ids).items()}
    r.update({f"rank{k}": v for k, v in cmc(S, q_ids, g_ids).items()})
    fnir, tau = fnir_at_fpir(S, q_ids, g_ids)[1e-2]
    r.update({"fnir@0.01fpir": fnir, "tau@0.01fpir": tau})
    return r

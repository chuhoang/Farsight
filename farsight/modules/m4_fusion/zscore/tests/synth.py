"""Synthetic templates with known structure, shared by the M4 tests."""
import numpy as np

from farsight.core.types import MODALITIES

DIMS = {"face": 64, "gait": 32, "body": 48}


def centers(n_ids, rng, dims=DIMS):
    return {m: rng.standard_normal((n_ids, d)).astype(np.float32) for m, d in dims.items()}


def templates(C, id_idx, rng, noise=None, p_missing=None, prefix="t", inter_dim=16):
    """One template per entry of id_idx. noise[m] = (lo, hi) per-template noise std (quality = 1/(1+std));
    p_missing[m] = probability the modality is None. inter_feat = noise level encoded (for QE tests)."""
    noise = noise or {"face": (0.3, 3.0), "gait": (2.0, 2.0), "body": (1.2, 1.2)}
    p_missing = p_missing or {}
    out = []
    for j, i in enumerate(id_idx):
        t = {"subject_or_track_id": f"{i:04d}_{prefix}{j}", "qe_weight": None}
        for m in MODALITIES:
            if rng.random() < p_missing.get(m, 0.0):
                t[m] = None
                continue
            sd = rng.uniform(*noise[m])
            f = C[m][i] + sd * rng.standard_normal(C[m].shape[1]).astype(np.float32)
            inter = np.full(inter_dim, sd, np.float32) + 0.05 * rng.standard_normal(inter_dim).astype(np.float32)
            t[m] = {"feat": f / np.linalg.norm(f), "quality": float(1 / (1 + sd)), "n_frames": 30,
                    "inter_feat": inter}
        out.append(t)
    return out


def label(tid):
    return tid.split("_")[0]

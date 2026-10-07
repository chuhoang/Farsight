"""M4 step: build the fusion named in configs/pipeline (m4_fusion: {method: zscore|qme, ...overrides})
plus the scoring helpers every fusion shares (stack templates -> per-modality cosine score tensors)."""
from pathlib import Path

import numpy as np
import yaml

from farsight.core import registry
from farsight.core.types import MODALITIES, SearchResult


def load_cfg(model_dir, **overrides):
    cfg = yaml.safe_load(Path(model_dir, "config.yaml").read_text(encoding="utf-8")) or {}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def stack(templates):
    """List[Template] -> {"ids", m: (feats (N,D) L2-normed float32, present (N,) bool), "quality": (N,3) NaN=missing}."""
    out = {"ids": [t["subject_or_track_id"] for t in templates]}
    qual = np.full((len(templates), len(MODALITIES)), np.nan, np.float32)
    for k, m in enumerate(MODALITIES):
        mos = [t.get(m) for t in templates]
        present = np.array([mo is not None for mo in mos], bool)
        dim = next((len(mo["feat"]) for mo in mos if mo is not None), 0)
        F = np.zeros((len(mos), dim), np.float32)
        for i, mo in enumerate(mos):
            if mo is not None:
                F[i] = mo["feat"]
                qual[i, k] = mo["quality"]
        F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
        out[m] = (F, present)
    out["quality"] = qual
    return out


def cosine_scores(P, G):
    """Stacked probes x stacked gallery -> (n_p, n_g, 3) cosine; NaN where either side lacks the modality."""
    S = np.full((len(P["ids"]), len(G["ids"]), len(MODALITIES)), np.nan, np.float32)
    for k, m in enumerate(MODALITIES):
        (pf, pp), (gf, gp) = P[m], G[m]
        if pp.any() and gp.any():
            s = pf @ gf.T
            s[~pp] = np.nan
            s[:, ~gp] = np.nan
            S[:, :, k] = s
    return S


_GALLERY_CACHE = [None, None]


def stack_gallery(gallery):
    """stack() with a one-entry cache keyed on the list object (re-stacking 10k templates costs ~10 ms).
    ponytail: identity+len key; mutating the list in place without changing its length serves stale scores."""
    key = (id(gallery), len(gallery))
    if _GALLERY_CACHE[0] != key or _GALLERY_CACHE[1][0] is not gallery:
        _GALLERY_CACHE[:] = [key, (gallery, stack(gallery))]
    return _GALLERY_CACHE[1][1]


def to_result(probe_id, fused, per_mod, gallery_ids, tau, top_k=None) -> SearchResult:
    """fused (G,), per_mod (G,3) -> SearchResult ranked by fused score (NaN last)."""
    f = np.where(np.isnan(fused), -np.inf, fused)
    idx = np.arange(len(f))
    if top_k and top_k < len(f):
        idx = np.argpartition(-f, top_k - 1)[:top_k]
    idx = idx[np.argsort(-f[idx], kind="stable")]
    ranked = [{"gallery_id": gallery_ids[i], "score": float(fused[i]), "per_modality": per_mod[i].tolist()}
              for i in idx]
    return {"probe_id": probe_id, "ranked": ranked, "is_known": bool(len(f) and f[idx[0]] >= tau)}


class M4:
    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.fusion = registry.build(cfg.pop("method", "zscore"), **cfg)

    def search(self, probe, gallery) -> SearchResult:
        return self.fusion.search(probe, gallery)

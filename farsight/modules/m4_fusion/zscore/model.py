"""Fusion v1 (no training): per-modality cosine -> impostor z-score -> quality-weighted mean; open-set tau."""
import itertools
import json
from pathlib import Path

import numpy as np

from eval.metrics import evaluate, fnir_at_fpir
from farsight.core.interfaces import BaseFusion
from farsight.core.registry import register
from farsight.core.types import MODALITIES
from farsight.core.weights import WEIGHTS
from farsight.modules.m4_fusion.module import cosine_scores, load_cfg, stack, stack_gallery, to_result

HERE = Path(__file__).parent


@register("zscore")
class ZScoreFusion(BaseFusion):
    def __init__(self, **overrides):
        c = load_cfg(HERE, **overrides)
        self.w = np.array([c["weights"][m] for m in MODALITIES], np.float32)
        self.use_quality, self.top_k = c["use_quality"], c["top_k"]
        self.stats_path = WEIGHTS / c["stats"]
        self.mu, self.sd, self.tau = np.zeros(3, np.float32), np.ones(3, np.float32), -np.inf
        if self.stats_path.exists():
            self.load(self.stats_path)

    # ---- core math: S (P,G,3) raw cosine (NaN = missing), q (P,3) probe quality -> (P,G) fused
    def fuse(self, S, q=None):
        z = (S - self.mu) / self.sd
        a = np.broadcast_to(self.w, (S.shape[0], 3)).astype(np.float32)
        if self.use_quality and q is not None:
            a = a * np.clip(np.nan_to_num(q, nan=0.0), 1e-3, None)
        a = np.where(np.isnan(S), 0.0, a[:, None, :])
        den = a.sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, (a * np.nan_to_num(z)).sum(-1) / den, np.nan)

    # ---- calibration on validation
    def fit(self, S, q_ids, g_ids):
        """Impostor mean/std per modality from (P,G,3) validation scores."""
        imp = np.asarray(q_ids)[:, None] != np.asarray(g_ids)[None, :]
        for k in range(3):
            v = S[..., k][imp & ~np.isnan(S[..., k])]
            if v.size > 1:
                self.mu[k], self.sd[k] = v.mean(), max(v.std(), 1e-6)
        return self

    def grid_search(self, S, q, q_ids, g_ids, metric="rank1", grid=(0.0, 0.5, 1.0, 2.0)):
        """Pick w maximizing evaluate()[metric] (minimizing for fnir*). Sets self.w; returns (w, value)."""
        sign = -1 if metric.startswith("fnir") else 1
        best = None
        for w in itertools.product(grid, repeat=3):
            if not any(w):
                continue
            self.w = np.array(w, np.float32)
            v = evaluate(self.fuse(S, q), q_ids, g_ids)[metric]
            if best is None or sign * v > sign * best[1]:
                best = (self.w, v)
        self.w = best[0]
        return best

    def set_tau(self, S, q, q_ids, g_ids, fpir=0.01):
        """tau on fused top-1 so that FPIR <= fpir on validation non-mated probes (distractors)."""
        self.tau = fnir_at_fpir(self.fuse(S, q), q_ids, g_ids, (fpir,))[fpir][1]
        return self.tau

    def save(self, path=None):
        p = Path(path or self.stats_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(),
                                 "tau": float(self.tau)}, indent=1))

    def load(self, path):
        d = json.loads(Path(path).read_text())
        self.mu, self.sd, self.w = (np.array(d[k], np.float32) for k in ("mu", "sd", "w"))
        self.tau = d["tau"]
        return self

    # ---- BaseFusion
    def search(self, probe, gallery):
        G = stack_gallery(gallery)
        P = stack([probe])
        S = cosine_scores(P, G)
        fused = self.fuse(S, P["quality"])[0]
        return to_result(probe["subject_or_track_id"], fused, S[0], G["ids"], self.tau, self.top_k)

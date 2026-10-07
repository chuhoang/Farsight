"""M2 step for one track: quality gate -> (maybe) restore -> identity safety check.

quality_fn(crops) -> (N,) per-crop quality;  embed_fn(crops) -> (N, D) L2-normed per-crop feats.
Both come from M3's face encoder (KP-RPE), injected so M2 never imports it.
"""
import numpy as np
import yaml

from farsight.core import registry
from farsight.core.weights import ROOT


def _sim_to_template(feats, template):
    return float(np.mean(np.asarray(feats, np.float64) @ template))


class M2Restore:
    def __init__(self, quality_fn, embed_fn, cfg=None, gate=None, restorer=None):
        if cfg is None:
            cfg = yaml.safe_load((ROOT / "configs/pipeline/v1.yaml").read_text())["m2_restore"]
        self.cfg, self.quality_fn, self.embed_fn = cfg, quality_fn, embed_fn
        self.enabled = bool(cfg.get("enabled", True))
        self.gate = gate or registry.build(cfg["gate"])
        self._restorer = restorer  # built lazily: most tracks never need the GPU model

    @property
    def restorer(self):
        if self._restorer is None:
            self._restorer = registry.build(self.cfg["restorer"])
        return self._restorer

    def __call__(self, crops):
        """BGR face crops of one track -> (crops to use downstream, info dict)."""
        info = {"restored": False, "reason": "disabled"}
        if not self.enabled or not crops:
            return crops, info
        q = np.asarray(self.quality_fn(crops), np.float64)
        info.update(quality_median=float(np.median(q)), reason="gate_pass")
        if not self.gate(q):
            return crops, info
        restored = self.restorer(crops)
        e0, e1 = self.embed_fn(crops), self.embed_fn(restored)
        t = np.asarray(e0, np.float64).mean(0)
        t /= np.linalg.norm(t) + 1e-12  # track template from ORIGINAL crops (plan M2 step 4)
        s0, s1 = _sim_to_template(e0, t), _sim_to_template(e1, t)
        info.update(sim_orig=s0, sim_restored=s1)
        if s1 < s0:
            info["reason"] = "safety_reject"
            return crops, info
        info.update(restored=True, reason="restored")
        return restored, info

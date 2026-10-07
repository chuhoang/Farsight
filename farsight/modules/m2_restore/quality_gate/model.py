"""Per-track restore decision: median per-crop quality below q0 -> restore."""
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent


class QualityGate:
    def __init__(self, **over):
        self.q0 = float({**yaml.safe_load((HERE / "config.yaml").read_text()), **over}["q0"])

    def __call__(self, qualities) -> bool:
        q = np.asarray(qualities, np.float64).ravel()
        q = q[np.isfinite(q)]
        return bool(q.size) and float(np.median(q)) < self.q0

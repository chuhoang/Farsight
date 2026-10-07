"""Open-set threshold of fusion v1 per available-modality combination (FPIR 1% on CCVID val).
One global tau is dominated by face (w=4): a face-less probe can never reach it. demo.infer picks the tau of the
modalities that both the query and the track have.

    bash run.sh eval.fit_tau_modalities   ->  eval/results/zscore_v1_tau_by_modalities.json
"""
import itertools
import json
from pathlib import Path

import numpy as np

from eval.main_protocol import Set, load
from eval.metrics import fnir_at_fpir
from farsight.core.types import MODALITIES
from farsight.modules.m4_fusion.zscore.model import ZScoreFusion

if __name__ == "__main__":
    F = Path.home() / "datasets/feats"
    val = set(json.loads(Path("splits/ccvid_qme_halftest.json").read_text())["val"])
    st = Set(*load(F, "ccvid_query", val), *load(F, "ccvid_gallery", val))
    z = ZScoreFusion(stats=str(Path("eval/results/zscore_v1_ccvid_val.json").resolve()), use_quality=False)
    out = {}
    for r in range(1, 4):
        for mods in itertools.combinations(MODALITIES, r):
            S = st.S["full"].copy()
            S[..., [k for k, m in enumerate(MODALITIES) if m not in mods]] = np.nan
            taus, fnirs = [], []
            for keep in st.open_masks(seed0=1000):
                fn, t = fnir_at_fpir(z.fuse(S)[:, keep], st.qid, st.gid[keep])[1e-2]
                taus.append(t), fnirs.append(fn)
            out["+".join(mods)] = {"tau": float(np.median(taus)), "fnir@1%fpir_val": float(np.mean(fnirs))}
            print("+".join(mods).ljust(15), json.dumps(out["+".join(mods)]))
    Path("eval/results/zscore_v1_tau_by_modalities.json").write_text(json.dumps(out, indent=1))

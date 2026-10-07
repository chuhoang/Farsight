"""Route 2A check (plan M4 v2): QME's released heads + QE weights on QME's released score matrices
(CCVID: adaface/biggait/cal-ccvid, MEVID: adaface/cal-mevid/agrl), scored with QME's own test_score so numbers
are comparable to the QME paper. Also single modalities and our z-score v1 (label-free stats: no val split here).

    bash run.sh eval.qme_2a --data ~/datasets/qme/test_feats
"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np

from farsight.core.weights import ROOT
from farsight.modules.m4_fusion.qme.model import QMEFusion
from farsight.modules.m4_fusion.zscore.model import ZScoreFusion

SETS = {  # dataset: (model order used by the released QME head, QME ckpt, QE weights file)
    "ccvid": (["adaface", "biggait", "cal-ccvid"], "lsf-ccvid-bs8-seq8-245.92-630.pth", "mod_qe_adaface-t1r3_ccvid_3.h5"),
    "mevid": (["adaface", "cal-mevid", "agrl"], "lsf-mevid-bs8-seq1-52.23-300.pth", "mod_qe_adaface_t1r3_mevid_6000.h5"),
}
KEYS = ["GR_top1", "GR_mAP", "CC_top1", "CC_mAP", "TAR@1.00%FAR", "FNIR@1.00%FPIR"]


def qme_test_score():
    sys.modules.setdefault("transformers", types.SimpleNamespace(set_seed=lambda s: np.random.seed(s)))
    spec = importlib.util.spec_from_file_location("qme_eval", ROOT / "third_party/QME_ICCV25/tools/eval_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.test_score


def run(data, name):
    mods, ck, qe = SETS[name]
    with h5py.File(Path(data, f"scoremats_{name}.h5"), "r") as f:
        S = np.stack([f[f"{m}/score_mat"][()] for m in mods], -1)
        M = np.stack([f[f"{m}/merge_score_mat"][()] for m in mods], -1)
        meta = [f[k][()] for k in ("q_pids", "q_camids", "q_clothes_ids", "g_pids", "g_camids", "g_clothes_ids",
                                   "unique_g_pids")]
    with h5py.File(Path(data, qe), "r") as f:
        w = f["face_weights"][()].reshape(-1)
    test_score = qme_test_score()
    score = lambda s, m: {k: float(v) for k, v in test_score(s, m, *meta, dataset=name, seed=0).items() if k in KEYS}

    rows = {f"{m} only": score(S[..., i], M[..., i]) for i, m in enumerate(mods)}
    z = ZScoreFusion(stats="__none__.json")
    z.mu, z.sd = np.nanmean(M, (0, 1)), np.nanstd(M, (0, 1))  # label-free normalisation
    rows["zscore v1 (equal w)"] = score(z.fuse(S), z.fuse(M))
    q = QMEFusion(checkpoint=ck, use_mask=False)
    rows["QME 2A (released)"] = score(q.fuse(S, w), q.fuse(M, w))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "datasets/qme/test_feats"))
    ap.add_argument("--out", default="eval/results/qme_2a.json")
    a = ap.parse_args()
    res = {n: run(a.data, n) for n in SETS}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for n, rows in res.items():
        print(f"\n## {n}\n| config | " + " | ".join(KEYS) + " |\n|---|" + "---|" * len(KEYS))
        for r, v in rows.items():
            print(f"| {r} | " + " | ".join(f"{100 * v.get(k, float('nan')):.1f}" for k in KEYS) + " |")

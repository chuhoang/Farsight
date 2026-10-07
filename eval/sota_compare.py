"""Our configs scored with QME's own test_score (third_party/QME_ICCV25/tools/eval_metrics.py), i.e. the protocol
behind published CCVID / MEVID numbers (QME, z-score, FarSight fusion ...): GR Rank-1 / mAP on the tracklet
gallery, TAR@1%FAR and FNIR@1%FPIR on the per-identity merged gallery.

Sets: ccvid_test (our 76 test IDs, never used for selection), ccvid_full (all 151 official test IDs = the papers'
set; half of them are our val IDs -> slightly optimistic), mevid_test (official, eval-only).

    bash run.sh eval.sota_compare
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from eval.main_protocol import export_qme, load
from eval.qme_2a import KEYS, qme_test_score
from farsight.modules.m4_fusion.qe.model import FaceQE, load_qe_state
from farsight.modules.m4_fusion.qme.model import load_head
from farsight.modules.m4_fusion.qme.train import fuse_np
from farsight.modules.m4_fusion.zscore.model import ZScoreFusion

F = Path.home() / "datasets/feats"
MODS = ("face", "gait", "body")


def scores(name):
    with h5py.File(F / f"scores_{name}.h5", "r") as f:
        S = np.stack([f[f"{m}/score_mat"][()] for m in MODS], -1).astype(np.float32)
        M = np.stack([f[f"{m}/merge_score_mat"][()] for m in MODS], -1).astype(np.float32)
        meta = [f[k][()] for k in ("q_pids", "q_camids", "q_clothes_ids", "g_pids", "g_camids", "g_clothes_ids",
                                   "unique_g_pids")]
    return S, M, meta


if __name__ == "__main__":
    sp = json.loads(Path("splits/ccvid_qme_halftest.json").read_text())
    data = {"ccvid_test": ("ccvid", set(sp["test"])), "ccvid_full": ("ccvid", None), "mevid_test": ("mevid", None)}
    export_qme(F, F, {"ccvid_full": (*load(F, "ccvid_query"), *load(F, "ccvid_gallery"))})   # others exist already
    z = ZScoreFusion(stats=str(Path("eval/results/zscore_v1_ccvid_val.json").resolve()), use_quality=False)
    heads = {}
    for tag, label in (("", "QME 2B (CCVID)"), ("_cm", "QME 2B (CCVID+MEVID)")):
        qe = FaceQE(512)
        load_qe_state(qe, f"weights/m4_fusion/qe/qe_kprpe_2b{tag}.pth")
        heads[label] = (load_head(f"weights/m4_fusion/qme/qme_kprpe_2b{tag}.pth", use_mask=True)[0], qe.eval())
    test_score = qme_test_score()
    out = {}
    for name, (d, ids) in data.items():
        S, M, meta = scores(name)
        probes = load(F, f"{d}_query", ids)[0]
        sc = lambda s, mm: {k: float(v) for k, v in test_score(np.nan_to_num(s, nan=-1e3), np.nan_to_num(mm, nan=-1e3),  # noqa
                                                                *meta, dataset=d, seed=0).items() if k in KEYS}
        rows = {f"{m} only": sc(S[..., k], M[..., k]) for k, m in enumerate(MODS)}
        rows["fusion v1 (z-score)"] = sc(z.fuse(S), z.fuse(M))
        for label, (head, qe) in heads.items():
            with torch.no_grad():   # same weights as eval.train_2b: sigmoid(QE(face inter_feat)), 0 without a face
                X = [t["face"]["inter_feat"] if t.get("face") else None for t in probes]
                w = np.zeros(len(X), np.float32)
                ok = [i for i, x in enumerate(X) if x is not None]
                w[ok] = torch.sigmoid(qe(torch.tensor(np.stack([X[i] for i in ok])))).numpy().ravel()
            rows[label] = sc(fuse_np(head, S, w, "cpu"), fuse_np(head, M, w, "cpu"))
        out[name] = {"n_query": len(meta[0]), "n_gallery": len(meta[3]), "n_ids": len(meta[6]), "rows": rows}
        print(f"\n## {name} ({len(meta[0])} probes, {len(meta[3])} gallery, {len(meta[6])} IDs)")
        print("| config | " + " | ".join(KEYS) + " |")
        for r, v in rows.items():
            print(f"| {r} | " + " | ".join(f"{100 * v.get(k, float('nan')):.1f}" for k in KEYS) + " |")
    Path("eval/results/sota_compare.json").write_text(json.dumps(out, indent=1))

"""plan_csci E1: General / Same-Clothes / Clothes-Changing (CAL rule, R1 + mAP) plus the 5-metric line, CCVID test and
MEVID test (whole official tests), full / noface conditions. No training. Per dataset: QME = its own D4 checkpoints
(qme_kprpe_2b_csci_<dataset>_s{0,1,2}.pth, each self-contained with its QE), averaged over seeds; fusion v1 = its own
z-score fit (eval/results/main_protocol_csci_zscore_v1_<dataset>.json); AIM body = the pre-CSCI stores kept by
eval.regait (<store>.h5.body1.h5).

    FARSIGHT_DEGRADED=0 bash run.sh eval.main_protocol --out eval/results/main_protocol_csci.json --no-export
    FARSIGHT_DEGRADED=0 bash run.sh eval.csci_report
"""
import json
from pathlib import Path

import numpy as np
import torch

from eval.main_protocol import Set, condition, load, score_all, table
from eval.prcc import aim_metrics
from eval.train_2b import DEV, qe_weights
from farsight.io.store import read_templates
from farsight.modules.m4_fusion.qme.model import load_head
from farsight.modules.m4_fusion.qme.train import fuse_np
from farsight.modules.m4_fusion.zscore.model import ZScoreFusion

F = Path.home() / "datasets/feats"
W = Path("weights/m4_fusion/qme")
SEEDS = (0, 1, 2)
CONDS = ("full", "noface")
SPLITS = ("GR", "SC", "CC")
FIVE = ("tar@0.001far", "rank20", "fnir@0.01fpir", "GR_R1", "CC_mAP")


def aim_body(ts, name):
    """Same templates with the body of the pre-CSCI store (AIM)."""
    old = {t["subject_or_track_id"]: t["body"] for t in read_templates(F / f"{name}.h5.body1.h5")}
    return [{**t, "body": old[t["subject_or_track_id"]]} for t in ts]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="QME checkpoint tag after the dataset, e.g. _cal -> qme_kprpe_2b_csci_<d>_cal_s<k>")
    tag = ap.parse_args().tag
    m = aim_metrics()
    ct = set(json.loads(Path("splits/ccvid_qme.json").read_text())["test"])
    sets = {"ccvid_test": ("ccvid_query", ct, "ccvid_gallery"),     # whole official test (151 IDs)
            "mevid_test": ("mevid_query", None, "mevid_gallery")}   # whole official test (54 IDs)
    mp = json.loads(Path("eval/results/main_protocol_csci.json").read_text())
    res = {}
    sets = {k: v for k, v in sets.items() if (W / f"qme_kprpe_2b_csci_{k.split('_')[0]}{tag}_s0.pth").exists()}
    for name, (pq, ids, pg) in sets.items():
        d = name.split("_")[0]
        z = ZScoreFusion(stats="__none__.json", use_quality=mp[f"fit_v1_on_{d}_val"]["use_quality"]).load(
            f"eval/results/main_protocol_csci_zscore_v1_{d}.json")
        heads = []
        for s in SEEDS:
            h, tau = load_head(W / f"qme_kprpe_2b_csci_{d}{tag}_s{s}.pth")   # carries face_gate / face_scale
            heads.append((h.to(DEV).eval(), tau))
        P, Pm = load(F, pq, ids)
        G, Gm = load(F, pg, ids)
        st = Set(P, Pm, G, Gm)
        sa = Set(aim_body(P, pq), Pm, aim_body(G, pg), Gm)
        res[name] = {}
        print(name, len(P), "query", len(G), "gallery", flush=True)
        v1 = table(st, z, m)
        for c in CONDS:
            rows = {"AIM body": score_all(sa, sa.S[c][..., 2], m)}
            for k in ("face only", "gait only", "body only", "fusion v1"):
                rows["CSCI body" if k == "body only" else k] = v1[c][k]
            runs = []
            for h, tau in heads:
                runs.append(score_all(st, fuse_np(h, st.S[c], qe_weights(h.qe, condition(P, c)), DEV), m, tau=tau))
            rows["QME (3 seeds)"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            rows["QME (3 seeds)"]["GR_R1_std"] = float(np.std([r["GR_R1"] for r in runs]))
            res[name][c] = rows
    Path(f"eval/results/csci_report{tag}.json").write_text(json.dumps(res, indent=1))
    for name in sets:
        for c in CONDS:
            print(f"\n## {name} / {c}   General R1 mAP | Same-clothes R1 mAP | Clothes-changing R1 mAP"
                  f" || TAR@0.1% R20 FNIR@1%")
            for row, r in res[name][c].items():
                print(f"{row:14s}", " | ".join(f"{100 * r[f'{s}_R1']:5.1f} {100 * r[f'{s}_mAP']:5.1f}" for s in SPLITS),
                      " || " + " ".join(f"{100 * r[k]:5.1f}" for k in FIVE[:3]))


if __name__ == "__main__":
    main()

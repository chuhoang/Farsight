"""Is the BigGait mask quality a useful gait-reliability signal, and which gate q_min to use? (val IDs only)

Val sets: CCVID val IDs (query vs gallery) and the 20 MEVID val IDs (held out of MEVID train; eval.main_protocol
.train_pairs). (1) AUC of gait quality for "gait alone ranks the true ID first" (probe side). (2) q_min sweep: gait
removed from every template with quality < q_min; fusion v1 (fixed CCVID-val weights/stats) GR-R1 per set.

    bash run.sh eval.gait_quality_calib
"""
import json
from pathlib import Path

import numpy as np

from eval.main_protocol import Set, cal_metrics, load, train_pairs
from eval.metrics import first_match_rank
from eval.prcc import aim_metrics
from farsight.modules.m4_fusion.zscore.model import ZScoreFusion

F = Path.home() / "datasets/feats"
QS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def gate(ts, q):
    return [{**t, "gait": t["gait"] if t["gait"] is not None and t["gait"]["quality"] >= q else None} for t in ts]


def auc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean()) if pos.size and neg.size else float("nan")


if __name__ == "__main__":
    cv = set(json.loads(Path("splits/ccvid_qme_halftest.json").read_text())["val"])
    mv = set(json.loads(Path("splits/mevid_qme.json").read_text())["val"])
    sets = {"ccvid_val": (*load(F, "ccvid_query", cv), *load(F, "ccvid_gallery", cv)),
            "mevid_val": train_pairs(*load(F, "mevid_train", mv))}
    z = ZScoreFusion(stats=str(Path("eval/results/zscore_v1_ccvid_val.json").resolve()), use_quality=False)
    m, out = aim_metrics(), {}
    for name, (P, Pm, G, Gm) in sets.items():
        st = Set(P, Pm, G, Gm)
        r1 = first_match_rank(st.S["full"][..., 1], st.qid, st.gid) == 1
        q = np.array([t["gait"]["quality"] if t["gait"] else np.nan for t in P])
        ok = ~np.isnan(q)
        res = {"probe_gait_quality_q10_q50_q90": np.nanquantile(q, [.1, .5, .9]).tolist(),
               "auc_quality_gait_correct": auc(q[ok & r1], q[ok & ~r1]), "gait_r1_all_probes": float(r1[ok].mean())}
        for qm in QS:
            s = Set(gate(P, qm), Pm, gate(G, qm), Gm)
            keep = np.array([t["gait"] is not None and t["gait"]["quality"] >= qm for t in P])
            res[f"qmin_{qm}"] = {"gait_kept_probe": float(keep.mean()),
                                 "gait_R1_kept": float(r1[keep].mean()) if keep.any() else float("nan"),
                                 "fusion_v1_GR_R1": float(cal_metrics(m, z.fuse(s.S["full"]), Pm, Gm)["GR_R1"]),
                                 "fusion_v1_GR_R1_noface": float(cal_metrics(m, z.fuse(s.S["noface"]), Pm, Gm)["GR_R1"])}
        out[name] = res
        print(name, json.dumps({k: v for k, v in res.items() if not k.startswith("qmin")}))
        for qm in QS:
            print(f"  q_min {qm}: " + json.dumps({k: round(v, 3) for k, v in res[f'qmin_{qm}'].items()}))
    Path("eval/results/gait_quality_calib.json").write_text(json.dumps(out, indent=1))

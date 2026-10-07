"""Automatic search over QME training ideas, one dataset at a time, seed 0; selection on VAL only.

Grid (MEVID): train pairs {alt, cam} x QE labels {plain, cal} x face_gate {0, .3, .5, .7} x face_scale {off, on}.
Grid (CCVID): train pairs {alt, cam} x face_gate {0, .5} x face_scale {off, on} (QE labels plain: CCVID faces are good).
Each config = eval.train_2b run (QE + QME, val = IDs held out of train), results in eval/results/search/<tag>.json;
reruns skip finished configs. Ranking key = val selection score (rank-1 + TAR@1%FAR - FNIR@1%FPIR, camera-split val);
test GR-R1 / CC-R1 are recorded for analysis but never used to choose. The chosen config is then run with 3 seeds.

    FARSIGHT_DEGRADED=0 bash run.sh eval.qme_search [--dataset mevid|ccvid|all]
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

OUT = Path("eval/results/search")
V1 = "eval/results/main_protocol_csci.json"
GRID = {"mevid": dict(pairs=("alt", "cam"), qe_label=("plain", "cal"), face_gate=(0.0, 0.3, 0.5, 0.7), face_scale=(0, 1)),
        "ccvid": dict(pairs=("alt", "cam"), qe_label=("plain",), face_gate=(0.0, 0.5), face_scale=(0, 1))}


def tag(d, c, seed=0):
    return f"_csci_{d}_{c['pairs']}_{c['qe_label']}_g{c['face_gate']:g}{'_fs' if c['face_scale'] else ''}_s{seed}"


def run(d, c, seed=0):
    t = tag(d, c, seed)
    out = OUT / f"{t[1:]}.json"
    if not out.exists():
        cmd = [sys.executable, "-m", "eval.train_2b", "--dataset", d, "--seed", str(seed), "--v1", V1, "--tag", t,
               "--out", str(out), "--pairs", c["pairs"], "--qe_label", c["qe_label"], "--face_gate", str(c["face_gate"])]
        cmd += ["--face_scale"] if c["face_scale"] else []
        print(" ".join(cmd[2:]), flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    r = json.loads(out.read_text())
    b = r["qme_val_best"]
    val = b["val0_rank1"] + b["val0_tar@0.01far"] - b["val0_fnir@0.01fpir"]
    te = r[f"{d}_test"]
    return {**c, "seed": seed, "val_score": round(val, 4), "val_R1": round(100 * b["val0_rank1"], 1),
            "test_GR_R1": round(100 * te["full"]["QME 2B"]["GR_R1"], 1), "test_CC_R1": round(100 * te["full"]["QME 2B"]["CC_R1"], 1),
            "test_noface_GR_R1": round(100 * te["noface"]["QME 2B"]["GR_R1"], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["mevid", "ccvid", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for d in (("mevid", "ccvid") if a.dataset == "all" else (a.dataset,)):
        g = GRID[d]
        rows = [run(d, dict(zip(g, v))) for v in itertools.product(*g.values())]
        rows.sort(key=lambda r: -r["val_score"])
        best = rows[0]
        chosen = {k: best[k] for k in g}
        seeds = [run(d, chosen, s) for s in (0, 1, 2)]
        summary[d] = {"ranked_by_val": rows, "chosen": chosen, "chosen_3_seeds": seeds,
                      "chosen_mean_test_GR_R1": round(sum(x["test_GR_R1"] for x in seeds) / 3, 1)}
        print(f"\n## {d}: {len(rows)} configs ranked by VAL score (test shown for analysis only)")
        print(f"{'pairs':5s} {'QE':5s} {'gate':>4s} {'scale':>5s} | {'val':>6s} {'valR1':>5s} | {'testR1':>6s} {'CC':>5s} {'noface':>6s}")
        for r in rows:
            print(f"{r['pairs']:5s} {r['qe_label']:5s} {r['face_gate']:4g} {r['face_scale']:5d} | {r['val_score']:6.3f} "
                  f"{r['val_R1']:5.1f} | {r['test_GR_R1']:6.1f} {r['test_CC_R1']:5.1f} {r['test_noface_GR_R1']:6.1f}")
        print(f"chosen by val: {chosen} -> 3 seeds test GR-R1 {[x['test_GR_R1'] for x in seeds]}")
    Path(OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

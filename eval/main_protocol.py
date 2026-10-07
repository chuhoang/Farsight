"""Main protocol (plan 5.3 / 5.4) on CCVID test-half and MEVID test, from eval/extract.py template stores.

probe = query templates, gallery = gallery templates, restricted to the split's identities (splits/*_qme.json).
Rows: face / gait / body only (cosine), fusion v1 (z-score; impostor stats, weight + use_quality grid search and
tau fitted on CCVID *val* IDs only, applied unchanged to CCVID test and MEVID test).
Conditions (probe side, template level): full / noface (probe face=None) / short (probe gait=None).
Metrics: (a) CAL/AIM eval_metrics GR + CC Rank-1/mAP; (b) eval/metrics.py TAR@0.1/1%FAR, Rank-1/5/20 on the
full gallery, FNIR@1%FPIR averaged over 10 open-set galleries keeping a random 80% of gallery IDs (QME test_score
style). Fusion also reports FPIR/FNIR at the val-fitted tau. Also exports QME-format score files (tools/export_scores).

    bash run.sh eval.main_protocol [--feats ~/datasets/feats]
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import h5py
import numpy as np

from eval.metrics import evaluate, fnir_at_fpir
from eval.prcc import aim_metrics
from farsight.core.types import MODALITIES
from farsight.io.store import read_templates, write_templates
from farsight.modules.m4_fusion.module import cosine_scores, stack
from farsight.modules.m4_fusion.zscore.model import ZScoreFusion
from tools.export_scores import export

CONDS = ("full", "noface", "short")
ROWS = ("face only", "gait only", "body only", "fusion v1")
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
OPEN_KEEP, OPEN_REPS = 0.8, 10


GAIT_QMIN = float(os.environ.get("FARSIGHT_GAIT_QMIN", "0"))   # gait quality gate (BigGait mask quality)
DEGRADED = os.environ.get("FARSIGHT_DEGRADED", "1") != "0"     # 0 = skip the degraded CCVID stores (CSCI run)
TESTS = ("ccvid_test", "ccvid_test_degraded", "mevid_test") if DEGRADED else ("ccvid_test", "mevid_test")


def load(feats, name, ids=None):
    meta = json.loads(Path(feats, f"{name}.h5.meta.json").read_text())
    ts = [t for t in read_templates(Path(feats, f"{name}.h5")) if ids is None or meta[t["subject_or_track_id"]]["pid"] in ids]
    for t in ts:   # low-quality gait -> missing modality (plan: unreliable gait must not drag fusion down)
        if t["gait"] is not None and t["gait"]["quality"] < GAIT_QMIN:
            t["gait"] = None
    return ts, [meta[t["subject_or_track_id"]] for t in ts]


def condition(ts, cond):
    drop = {"full": None, "noface": "face", "short": "gait"}[cond]
    return [{**t, drop: None} if drop else t for t in ts]


class Set:
    """One probe set x gallery: raw (P,G,3) cosine + ids, per condition."""
    def __init__(self, probes, pmeta, gallery, gmeta):
        self.G, self.gm = stack(gallery), gmeta
        self.pm = pmeta
        Ps = {c: stack(condition(probes, c)) for c in CONDS}
        self.S = {c: cosine_scores(P, self.G) for c, P in Ps.items()}
        self.q = {c: P["quality"] for c, P in Ps.items()}
        self.qid = np.array([m["pid"] for m in pmeta])
        self.gid = np.array([m["pid"] for m in gmeta])

    def open_masks(self, seed0=0):
        u = np.unique(self.gid)
        return [np.isin(self.gid, np.random.default_rng(seed0 + r).choice(u, int(len(u) * OPEN_KEEP), replace=False))
                for r in range(OPEN_REPS)]


def cal_metrics(m, sim, qm, gm, seed=0):
    """CAL/AIM general + same-clothes (SC) + clothes-changing (CC); distance = -sim, missing scores -> random far
    distances (a guess). SC/CC keep only same-/different-outfit positives (the others become junk)."""
    D = np.where(np.isnan(sim), 1e6 + np.random.default_rng(seed).random(sim.shape), -sim)
    f = lambda ms, k: np.array([x[k] for x in ms])  # noqa: E731
    a = (f(qm, "pid"), f(gm, "pid"), f(qm, "camid"), f(gm, "camid"))
    cmc, mAP = m.evaluate(D, *a)
    ccmc, cmAP = m.evaluate_with_clothes(D, *a, f(qm, "clothes"), f(gm, "clothes"), mode="CC")
    scmc, smAP = m.evaluate_with_clothes(D, *a, f(qm, "clothes"), f(gm, "clothes"), mode="SC")
    return {"GR_R1": cmc[0], "GR_mAP": mAP, "SC_R1": scmc[0], "SC_mAP": smAP, "CC_R1": ccmc[0], "CC_mAP": cmAP}


def score_all(st, sim, m, tau=None):
    r = cal_metrics(m, sim, st.pm, st.gm)
    r.update(evaluate(sim, st.qid, st.gid))
    fn, fp_tau, fn_tau = [], [], []
    for keep in st.open_masks():
        s = sim[:, keep]
        fn.append(fnir_at_fpir(s, st.qid, st.gid[keep])[1e-2][0])
        if tau is not None:  # operating point fixed on val
            mated = np.isin(st.qid, st.gid[keep])
            s0 = np.nan_to_num(s, nan=-np.inf)
            top = s0.max(1)
            hit = st.gid[keep][s0.argmax(1)] == st.qid
            fp_tau.append(float((top[~mated] > tau).mean()))
            fn_tau.append(float(1 - (hit[mated] & (top[mated] > tau)).mean()))
    r["fnir@0.01fpir"] = float(np.mean(fn))
    if tau is not None:
        r.update({"fpir@tau_val": float(np.mean(fp_tau)), "fnir@tau_val": float(np.mean(fn_tau))})
    r.pop("tau@0.01fpir", None)
    r["probe_coverage"] = float((~np.isnan(sim).all(1)).mean())
    return {k: float(v) for k, v in r.items()}


def fit_v1(val):
    """z-score stats (impostors), then (use_quality, w) grid search over the three conditions stacked (plan 2B.2:
    val carries the missing-modality cases), then tau @1% FPIR on val open-set. Selection score = QME's model
    selection score, rank-1 + TAR@1%FAR - FNIR@1%FPIR: rank-1 alone saturates at 1.0 on an easy val (CCVID train =
    session 1 only, no real outfit change) and the first weight combo reaching it wins (e.g. body only)."""
    S = np.concatenate([val.S[c] for c in CONDS])
    q = np.concatenate([val.q[c] for c in CONDS])
    qid = np.concatenate([val.qid] * len(CONDS))
    z = ZScoreFusion(stats="__none__.json").fit(val.S["full"], val.qid, val.gid)
    best = None
    for uq in (False, True):
        z.use_quality = uq
        for w in itertools.product(GRID, repeat=3):
            if any(w):
                z.w = np.array(w, np.float32)
                r = evaluate(z.fuse(S, q), qid, val.gid)
                v = r["rank1"] + r["tar@0.01far"] - r["fnir@0.01fpir"]
                if best is None or v > best[0]:
                    best = (v, uq, z.w)
    _, z.use_quality, z.w = best
    keep = val.open_masks(seed0=1000)[0]
    z.tau = fnir_at_fpir(z.fuse(S, q)[:, keep], qid, val.gid[keep])[1e-2][1]
    return z, {"val_select_score": best[0], "use_quality": bool(z.use_quality), "w": dict(zip(MODALITIES, z.w.tolist())),
               "mu": z.mu.tolist(), "sd": z.sd.tolist(), "tau": float(z.tau)}


def table(st, z, m):
    out = {}
    for c in CONDS:
        S = st.S[c]
        rows = {f"{mod} only": score_all(st, S[..., k], m) for k, mod in enumerate(MODALITIES)}
        rows["fusion v1"] = score_all(st, z.fuse(S, st.q[c]), m, tau=z.tau)
        out[c] = rows
    return out


def export_qme(feats, out_dir, sets):
    """QME-format score files; real camids/clothes patched in over export_scores' dummies."""
    for name, (probe, pmeta, gal, gmeta) in sets.items():
        tmp = [Path(out_dir, f"_{name}_{s}.h5") for s in ("probe", "gallery")]
        for p, ts in zip(tmp, (probe, gal)):
            p.unlink(missing_ok=True)
            write_templates(p, ts)
        lab = {t["subject_or_track_id"]: mm["pid"] for t, mm in zip(probe + gal, pmeta + gmeta)}
        dst = Path(out_dir, f"scores_{name}.h5")
        export(tmp[0], tmp[1], dst, lab.__getitem__)
        cl = {c: i for i, c in enumerate(sorted({mm["clothes"] for mm in pmeta + gmeta}))}
        with h5py.File(dst, "a") as f:
            for side, mm in (("q", pmeta), ("g", gmeta)):
                for k, v in (("camids", [x["camid"] for x in mm]), ("clothes_ids", [cl[x["clothes"]] for x in mm])):
                    del f[f"{side}_{k}"]
                    f[f"{side}_{k}"] = np.array(v)
        for p in tmp:
            p.unlink()
        print("wrote", dst)


def train_pairs(ts, meta):
    """CCVID train has no query/gallery lists: per identity, sorted tracklets alternate gallery / probe."""
    seen, P, G = {}, ([], []), ([], [])
    for t, mm in sorted(zip(ts, meta), key=lambda x: x[0]["subject_or_track_id"]):
        k = seen[mm["pid"]] = seen.get(mm["pid"], -1) + 1
        side = G if k % 2 == 0 else P
        side[0].append(t), side[1].append(mm)
    return P[0], P[1], G[0], G[1]


VAL_SPLIT = {"ccvid": "clothes", "mevid": "camid"}   # what val_pairs splits per identity (makes val look like test)


def val_pairs(ts, meta, key="camid"):
    """Held-out train IDs -> val probe / gallery. Per identity the values of `key` are split alternately between
    gallery and probe (train_pairs gives same-camera near-duplicates -> a trivially easy val):
      camid   -> no same-camera positive, plain rank-1 = the CAL general rule of test (MEVID: test is ~90%
                 same-clothes, and so is this split)
      clothes -> gallery and probe clothes labels differ (CCVID: 68% of test queries have only different-clothes
                 positives). CCVID train is session 1 only, where labels differ by one attribute, not a real outfit
                 change, so this val stays easy (body / face ~100% R1); fit_v1 therefore selects on a score that
                 does not saturate.
    Identities with a single value go to the gallery only (distractors)."""
    by = {}
    for t, mm in sorted(zip(ts, meta), key=lambda x: x[0]["subject_or_track_id"]):
        by.setdefault(mm["pid"], []).append((t, mm))
    P, G = ([], []), ([], [])
    for items in by.values():
        vals = sorted({mm[key] for _, mm in items})
        gal = set(vals[0::2]) if len(vals) > 1 else set(vals)
        for t, mm in items:
            side = G if mm[key] in gal else P
            side[0].append(t), side[1].append(mm)
    return P[0], P[1], G[0], G[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=str(Path.home() / "datasets/feats"))
    ap.add_argument("--out", default="eval/results/main_protocol.json")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    F, m = a.feats, aim_metrics()
    sp = {d: json.loads(Path(f"splits/{d}_qme.json").read_text()) for d in ("ccvid", "mevid")}
    cv, ct = set(sp["ccvid"]["val"]), set(sp["ccvid"]["test"])
    data = {   # per dataset: val = IDs held out of its train IDs (fits fusion v1), test = its whole official test
        "ccvid_val": val_pairs(*load(F, "ccvid_train", cv), VAL_SPLIT["ccvid"]),
        "mevid_val": val_pairs(*load(F, "mevid_train", set(sp["mevid"]["val"])), VAL_SPLIT["mevid"]),
        "ccvid_test": (*load(F, "ccvid_query", ct), *load(F, "ccvid_gallery", ct)),
        "mevid_test": (*load(F, "mevid_query"), *load(F, "mevid_gallery")),
    }
    if DEGRADED:
        data["ccvid_test_degraded"] = (*load(F, "ccvid_query_degraded", ct), *load(F, "ccvid_gallery", ct))
    sets = {k: Set(*v) for k, v in data.items()}
    default = a.out == ap.get_default("out")   # other outputs keep their own v1 stats (the demo reads the default)
    res, zs = {"n": {k: [len(v[0]), len(v[2])] for k, v in data.items()}}, {}
    for d in ("ccvid", "mevid"):              # fusion v1 fitted per dataset on its own val
        zs[d], res[f"fit_v1_on_{d}_val"] = fit_v1(sets[f"{d}_val"])
        zs[d].save(Path(a.out).with_name(f"zscore_v1_{d}_val.json" if default else f"{Path(a.out).stem}_zscore_v1_{d}.json"))
    for k in TESTS + ("ccvid_val", "mevid_val"):   # val rows for reference only (fitted there)
        res[k] = table(sets[k], zs[k.split("_")[0]], m)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({d: res[f"fit_v1_on_{d}_val"] for d in zs}))
    for k in TESTS:
        print(f"\n## {k}  (TAR@0.1% / R20 / FNIR@1% / GR-R1 / CC-mAP)")
        for r in ROWS:
            print(r.ljust(10), " | ".join(" ".join(f"{100 * res[k][c][r][x]:5.1f}" for x in
                  ("tar@0.001far", "rank20", "fnir@0.01fpir", "GR_R1", "CC_mAP")) for c in CONDS))
    if not a.no_export:
        tr = load(F, "ccvid_train", set(sp["ccvid"]["train"])) if Path(F, "ccvid_train.h5").exists() else None
        ex = {k: data[k] for k in ("ccvid_val", "mevid_val") + TESTS}
        if tr:
            ex["ccvid_train"] = train_pairs(*tr)
        export_qme(F, F, ex)


if __name__ == "__main__":
    main()

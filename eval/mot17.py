"""M1 tracker eval on MOT17 train (plan 4/M1 acceptance: PSR-ByteTrack has fewer ID switches than plain ByteTrack).

Stage 1 (GPU, once): BPJDet bodies (conf >= 0.1) + YOLOv8x persons (conf >= 0.25) per frame -> ~/datasets/mot17_cache/<seq>.npz.
  Lower confs than M1's configs so threshold variants are offline filters; NMS keeps the same boxes above the
  configured thresholds (conf filter happens before NMS in both repos), so defaults are reproduced exactly.
Stage 2: tracker variants through M1DetectTrack.update (frames re-read from img1/ only when PSR needs crops).
Stage 3: TrackEval MotChallenge2DBox (official MOT17 preprocessing: distractor removal, class 1 + consider flag)
  -> HOTA / CLEAR (MOTA, IDSW) / Identity (IDF1), on the raw per-frame M1 observations (before tracklet filtering).

    bash run.sh eval.mot17 [--grid] [--seqs MOT17-02-FRCNN ...]
"""
import argparse
import configparser
import itertools
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

for _n, _t in (("float", float), ("int", int), ("bool", bool)):  # TrackEval still uses removed numpy aliases
    if not hasattr(np, _n):
        setattr(np, _n, _t)
import trackeval  # noqa: E402
from ultralytics.utils.metrics import bbox_ioa  # noqa: E402

from farsight.core import registry  # noqa: E402
from farsight.modules.m1_detect_track.module import DEFAULT, M1DetectTrack  # noqa: E402

HOME = Path.home()
SEQS = [f"MOT17-{s:02d}-FRCNN" for s in (2, 4, 5, 9, 10, 11, 13)]
BODY_CONF, YOLO_CONF = 0.1, 0.25  # cache thresholds


def seqinfo(gt_root, seq):
    c = configparser.ConfigParser()
    c.read(gt_root / seq / "seqinfo.ini")
    s = c["Sequence"]
    return gt_root / seq / s["imDir"], int(s["seqLength"]), int(round(float(s["frameRate"])))


def frame_paths(img_dir, n):
    return [img_dir / f"{i + 1:06d}.jpg" for i in range(n)]


def detect(gt_root, cache, seq, bs=8):
    out = cache / f"{seq}.npz"
    if out.exists():
        return
    img_dir, n, _ = seqinfo(gt_root, seq)
    det = registry.build("bpjdet", conf_body=BODY_CONF)
    ver = registry.build("yolov8_verifier", conf=YOLO_CONF)
    B, F, Y, bfr, yfr = [], [], [], [], []
    t0 = time.perf_counter()
    paths = frame_paths(img_dir, n)
    for s in range(0, n, bs):
        frames = [cv2.imread(str(p)) for p in paths[s:s + bs]]
        for k, (d, y) in enumerate(zip(det(frames), ver(frames))):
            B.append(d["body"]), F.append(d["face"]), Y.append(y["body"])
            bfr += [s + k] * len(d["body"])
            yfr += [s + k] * len(y["body"])
    dt = time.perf_counter() - t0
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, body=np.concatenate(B), face=np.concatenate(F), yolo=np.concatenate(Y),
                        bfr=np.array(bfr), yfr=np.array(yfr), seconds=dt, frames=n)
    print(f"{seq}: {n} frames detected in {dt:.0f}s ({n / dt:.1f} FPS, BPJDet+YOLO, incl. JPEG decode)", flush=True)
    del det, ver


VERIFIER = {"boxes": None, "conf": 0.7}  # --verifier: cached verifier boxes (eval.verifier_eval) instead of COCO


def load_dets(cache, seq, n, body_conf=0.3, verifier=True, yolo_conf=None, iou_keep=0.5):
    """Per-frame {"body", "face"} as M1.detect would return them with the given thresholds."""
    z = np.load(cache / f"{seq}.npz")
    yolo_conf = VERIFIER["conf"] if yolo_conf is None else yolo_conf
    if VERIFIER["boxes"]:
        z = {**z, **np.load(cache / f"ver_{VERIFIER['boxes']}_{seq}.npz")}
    out = []
    for i in range(n):
        m = (z["bfr"] == i) & (z["body"][:, 4] >= body_conf)
        b, f = z["body"][m], z["face"][m]
        if verifier:
            y = z["yolo"][(z["yfr"] == i) & (z["yolo"][:, 4] >= yolo_conf)]
            keep = (bbox_ioa(b[:, :4], y[:, :4], iou=True).max(1) >= iou_keep) if len(b) and len(y) \
                else np.zeros(len(b), bool)
            b, f = b[keep], f[keep]
        out.append({"body": b, "face": f})
    return out


def write_mot(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for fr, tid, (x1, y1, x2, y2), sc in rows:
            fh.write(f"{fr + 1},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{sc:.3f},-1,-1,-1\n")


def track(m1, gt_root, seq, dets, psr, bt):
    """Run M1's tracker part; returns (raw rows, filtered rows, seconds)."""
    img_dir, n, fps = seqinfo(gt_root, seq)
    m1.app = None
    if psr is not None:
        m1.app = m1._psr
        m1.app.cfg = {**m1._psr_cfg, **psr}
    m1.reset(fps)
    m1.trk = registry.build("bytetrack", frame_rate=fps, **bt)
    t0 = time.perf_counter()
    for i, (p, d) in enumerate(zip(frame_paths(img_dir, n), dets)):
        m1.update(i, cv2.imread(str(p)) if m1.app else None, d)
    dt = time.perf_counter() - t0
    raw = [(o["frame_idx"], t, o["body_box"], o["body_score"]) for t, obs in m1.obs.items() for o in obs]
    filt = [(o["frame_idx"], t["track_id"], o["body_box"], o["body_score"])
            for t in m1.tracklets(seq) for o in t["frames"]]
    return dedup(raw), dedup(filt), dt


DUP_OFFSET = 1_000_000


def dedup(rows):
    """PSR can emit one final ID twice in a frame (a new raw ID is remapped to a lost identity while ByteTrack
    later revives that identity's own raw track). TrackEval rejects that; the 2nd box gets ID + DUP_OFFSET
    (acts like a separate track, so it is penalised as an ID error, not hidden). Returns (rows, n_dup)."""
    seen, out, n = set(), [], 0
    for fr, tid, b, sc in sorted(rows, key=lambda r: (r[0], r[1])):
        while (fr, tid) in seen:
            tid, n = tid + DUP_OFFSET, n + 1
        seen.add((fr, tid))
        out.append((fr, tid, b, sc))
    return out, n


def trackeval_run(gt_root, trk_root, names, seqs):
    seqmap = trk_root / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
              PLOT_CURVES=False, USE_PARALLEL=True, NUM_PARALLEL_CORES=2, TIME_PROGRESS=False,
              DISPLAY_LESS_PROGRESS=True, PRINT_ONLY_COMBINED=True)
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update(GT_FOLDER=str(gt_root), TRACKERS_FOLDER=str(trk_root), TRACKERS_TO_EVAL=names, BENCHMARK="MOT17",
              SPLIT_TO_EVAL="train", SEQMAP_FILE=str(seqmap), SKIP_SPLIT_FOL=True, TRACKER_SUB_FOLDER="data",
              OUTPUT_FOLDER=str(trk_root / "_out"), PRINT_CONFIG=False)
    res, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)],
                                              [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
                                               trackeval.metrics.Identity()])
    out = {}
    for name in names:
        out[name] = {}
        for seq, r in res["MotChallenge2DBox"][name].items():
            r = r["pedestrian"]
            c, i = r["CLEAR"], r["Identity"]
            out[name][seq] = {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "MOTA": float(c["MOTA"]),
                              "IDF1": float(i["IDF1"]), "IDSW": int(c["IDSW"]), "Recall": float(c["CLR_Re"]),
                              "Precision": float(c["CLR_Pr"]),
                              **{k: int(c[k]) for k in ("CLR_TP", "CLR_FN", "CLR_FP", "Frag")},
                              **{k: int(i[k]) for k in ("IDTP", "IDFN", "IDFP")}}
    return out


def combine(per_seq, seqs):
    """Recompute MOTA / IDF1 / IDSW over a subset of sequences from per-sequence counts."""
    s = {k: sum(per_seq[q][k] for q in seqs) for k in ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "IDTP", "IDFN", "IDFP")}
    return {"MOTA": 1 - (s["CLR_FN"] + s["CLR_FP"] + s["IDSW"]) / (s["CLR_TP"] + s["CLR_FN"]),
            "IDF1": 2 * s["IDTP"] / (2 * s["IDTP"] + s["IDFN"] + s["IDFP"]), "IDSW": s["IDSW"]}


def variants(grid):
    v = {"bytetrack": dict(psr=None), "bytetrack+psr": dict(psr={}),
         "bytetrack_noverif": dict(psr=None, verifier=False), "bytetrack+psr_noverif": dict(psr={}, verifier=False)}
    if grid:
        for thr, win in itertools.product((0.05, 0.1, 0.15, 0.2, 0.3), (5, 10, 20)):
            v[f"psr_cos_thr{thr}_win{win}"] = dict(psr={"thr": thr, "window_s": win})
        # mse on L2-normed 512-d feats == (2 - 2cos)/512 -> same ranking as cosine; thr 0.15 cos == 0.15/256 mse
        v["psr_mse_thr0.000586"] = dict(psr={"metric": "mse", "thr": 0.15 / 256})
        v["psr_mse_thr0.002"] = dict(psr={"metric": "mse", "thr": 0.002})
        v["psr_noswap"] = dict(psr={"swap_margin": 1e9})
        v["psr_every5"] = dict(psr={"mem_every": 5})
        v["psr_overlap0.5"] = dict(psr={"overlap_iou": 0.5})
        for mt, buf in itertools.product((0.7, 0.8, 0.9), (30, 60)):
            v[f"bt_match{mt}_buf{buf}"] = dict(psr=None, bt={"match_thresh": mt, "track_buffer": buf})
            v[f"bt_match{mt}_buf{buf}+psr"] = dict(psr={}, bt={"match_thresh": mt, "track_buffer": buf})
        v["bt_lowconf+psr"] = dict(psr={}, body_conf=0.1)        # feeds ByteTrack's low-score stage
        v["bt_lowconf"] = dict(psr=None, body_conf=0.1)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HOME / "datasets/mot17/MOT17/train"))
    ap.add_argument("--cache", default=str(HOME / "datasets/mot17_cache"))
    ap.add_argument("--seqs", nargs="+", default=SEQS)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="re-run variants already in the cache's trackers/ dir")
    ap.add_argument("--out", default="eval/results/mot17.json")
    ap.add_argument("--verifier", help="cached verifier boxes name, e.g. crowdhuman_yolov8n (default: COCO YOLOv8x)")
    ap.add_argument("--verifier-conf", type=float, default=0.7)
    a = ap.parse_args()
    gt_root, cache = Path(a.root), Path(a.cache)
    VERIFIER.update(boxes=a.verifier, conf=a.verifier_conf)
    trk_root = cache / ("trackers" + (f"_{a.verifier}_c{a.verifier_conf}" if a.verifier else ""))
    if a.fresh:
        shutil.rmtree(trk_root, ignore_errors=True)

    for seq in a.seqs:
        detect(gt_root, cache, seq)
    det_sec = {s: float(np.load(cache / f"{s}.npz")["seconds"]) for s in a.seqs}
    nfr = {s: seqinfo(gt_root, s)[1] for s in a.seqs}

    # detector-only "trackers" (one ID per box): recall / precision under the MOT17 protocol
    det_names = {"det_bpjdet_c0.3": dict(verifier=False), "det_bpjdet_c0.1": dict(verifier=False, body_conf=0.1),
                 "det_bpjdet_c0.3+yolo": {}}
    for name, kw in det_names.items():
        for seq in a.seqs:
            dets = load_dets(cache, seq, nfr[seq], **kw)
            # ID = box index within its frame: a unique ID per box needs >15 GB RAM in TrackEval's HOTA/Identity
            # (OOM-killed). IDs are meaningless for a detector either way; Recall/Precision don't depend on them.
            rows = [(i, k + 1, b[:4], b[4]) for i, d in enumerate(dets) for k, b in enumerate(d["body"])]
            write_mot(trk_root / name / "data" / f"{seq}.txt", rows)

    m1 = M1DetectTrack.__new__(M1DetectTrack)          # tracker half of M1 only; detections come from the cache
    m1.cfg, m1.fps = dict(DEFAULT), 30
    m1._psr = registry.build("psr_appearance", device="cuda", fps=30)
    m1._psr_cfg = dict(m1._psr.cfg)
    runtime = {}
    V = variants(a.grid)
    for name, v in V.items():
        meta = trk_root / name / "runtime.json"
        if meta.exists():                                   # resume: variant finished in an earlier run
            runtime[name] = json.loads(meta.read_text())
            continue
        t_trk, ndup = 0.0, 0
        for seq in a.seqs:
            dets = load_dets(cache, seq, nfr[seq], body_conf=v.get("body_conf", 0.3), verifier=v.get("verifier", True))
            (raw, nd), (filt, _), dt = track(m1, gt_root, seq, dets, v["psr"], v.get("bt", {}))
            t_trk += dt
            ndup += nd
            write_mot(trk_root / name / "data" / f"{seq}.txt", raw)
            if name in ("bytetrack", "bytetrack+psr"):
                write_mot(trk_root / f"{name}_filtered" / "data" / f"{seq}.txt", filt)
        runtime[name] = {"track_seconds": t_trk, "track_fps": sum(nfr.values()) / t_trk, "dup_id_boxes": ndup}
        meta.write_text(json.dumps(runtime[name]))
        print(f"{name}: tracked in {t_trk:.0f}s, {ndup} duplicate-ID boxes", flush=True)

    names = sorted(set(det_names) | set(V) | {f"{n}_filtered" for n in ("bytetrack", "bytetrack+psr") if n in V})
    res = trackeval_run(gt_root, trk_root, names, a.seqs)

    # leave-one-sequence-out selection of the PSR config by IDF1 (ties -> fewer IDSW)
    psr_names = [n for n in V if V[n]["psr"] is not None and V[n].get("verifier", True)]
    loso = {}
    for held in a.seqs if len(a.seqs) > 1 else []:
        rest = [s for s in a.seqs if s != held]
        best = max(psr_names, key=lambda n: (combine(res[n], rest)["IDF1"], -combine(res[n], rest)["IDSW"]))
        loso[held] = best
    loso_tot = {k: sum(res[loso[s]][s][k] for s in loso) for k in res["bytetrack"][a.seqs[0]] if k[0] in "CI"}
    loso_res = combine({"all": loso_tot}, ["all"]) if loso else None

    out = {"protocol": "MOT17 train FRCNN copies (images identical across detector variants), TrackEval "
                       "MotChallenge2DBox; raw M1 observations (raw det boxes per tracked frame) unless *_filtered",
           "sequences": a.seqs, "frames": nfr,
           "detect_runtime": {"seconds": sum(det_sec.values()), "fps": sum(nfr.values()) / sum(det_sec.values()),
                              "per_seq": det_sec},
           "tracker_runtime": runtime,
           "variants": {n: {"cfg": V.get(n, det_names.get(n, {})), "combined": res[n]["COMBINED_SEQ"],
                            "per_seq": {s: res[n][s] for s in a.seqs}} for n in names},
           "loso": {"choice": loso, "combined": loso_res}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"{'variant':32s} {'HOTA':>6s} {'MOTA':>6s} {'IDF1':>6s} {'IDSW':>5s} {'Rec':>6s} {'Prec':>6s}")
    for n in names:
        r = res[n]["COMBINED_SEQ"]
        print(f"{n:32s} {r['HOTA']*100:6.1f} {r['MOTA']*100:6.1f} {r['IDF1']*100:6.1f} {r['IDSW']:5d} "
              f"{r['Recall']*100:6.1f} {r['Precision']*100:6.1f}")
    print("LOSO:", loso, loso_res)

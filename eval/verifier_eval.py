"""Which person detector should verify BPJDet bodies? (M1 step 2; COCO YOLOv8x drops recall 71% -> 29% on MOT17.)
Box-level Recall / Precision of BPJDet bodies (conf >= 0.3) kept by each verifier at several (conf, IoU) settings,
vs MOT17 train GT (class 1, consider=1; predictions matching distractor classes 2/7/8/12 are ignored, as TrackEval).

    bash run.sh eval.verifier_eval
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics.utils.metrics import bbox_ioa

from eval.mot17 import HOME, SEQS, frame_paths, seqinfo

VER = HOME / "datasets/cache/verifiers"
CANDIDATES = {  # name -> (weights, loader)
    "coco_yolov8x": (None, None),  # boxes already in the MOT17 cache (conf >= 0.25)
    "crowdhuman_yolov8n": (VER / "crowdhuman_yolov8n_best.pt", "ultralytics"),
    "crowdhuman_yolov5m": (VER / "crowdhuman_yolov5m.pt", "yolov5"),
}


def run_detector(name, gt_root, cache, seq, bs=16):
    out = cache / f"ver_{name}_{seq}.npz"
    if out.exists():
        return np.load(out)
    w, kind = CANDIDATES[name]
    img_dir, n, _ = seqinfo(gt_root, seq)
    if kind == "ultralytics":
        from ultralytics import YOLO
        m = YOLO(str(w))
        pred = lambda fr: [np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]
                           for r in m.predict(fr, imgsz=640, conf=0.1, classes=[0], verbose=False, half=True)]
    else:
        import functools
        import torch
        import yolov5
        # full unpickle needed (old YOLOv5 format); pickle globals checked statically first with
        # tools/_scan_pickle.py: only torch/numpy/collections/models + set/_codecs.encode, nothing executable
        load, torch.load = torch.load, functools.partial(torch.load, weights_only=False)
        try:
            m = yolov5.load(str(w))
        finally:
            torch.load = load
        m.conf, m.classes = 0.1, None
        pred = lambda fr: [p[p[:, 5] == 0, :5].cpu().numpy() for p in m([f[:, :, ::-1] for f in fr], size=640).pred]
    boxes, fr_idx = [], []
    paths = frame_paths(img_dir, n)
    for s in range(0, n, bs):
        for k, b in enumerate(pred([cv2.imread(str(p)) for p in paths[s:s + bs]])):
            boxes.append(b.astype(np.float32))
            fr_idx += [s + k] * len(b)
    np.savez_compressed(out, yolo=np.concatenate(boxes), yfr=np.array(fr_idx))
    return np.load(out)


def gt_boxes(gt_root, seq):
    g = np.loadtxt(gt_root / seq / "gt/gt.txt", delimiter=",")
    xyxy = np.c_[g[:, 2:4], g[:, 2:4] + g[:, 4:6]]
    keep = (g[:, 7] == 1) & (g[:, 6] == 1)
    distr = np.isin(g[:, 7], [2, 7, 8, 12])
    return g[:, 0].astype(int) - 1, xyxy, keep, distr


def match(pred, gt, thr=0.5):
    if not len(pred) or not len(gt):
        return np.zeros(len(pred), bool)
    iou = bbox_ioa(pred, gt, iou=True)
    r, c = linear_sum_assignment(-iou)
    hit = np.zeros(len(pred), bool)
    hit[r[iou[r, c] >= thr]] = True
    return hit


def evaluate(gt_root, cache, name, settings, body_conf=0.3):
    stats = {s: [0, 0, 0] for s in settings}   # tp, fp, n_gt
    for seq in SEQS:
        z = np.load(cache / f"{seq}.npz")
        y = z if name == "coco_yolov8x" else run_detector(name, gt_root, cache, seq)
        gfr, gxy, gkeep, gdis = gt_boxes(gt_root, seq)
        for i in range(seqinfo(gt_root, seq)[1]):
            b = z["body"][(z["bfr"] == i) & (z["body"][:, 4] >= body_conf), :4]
            yy = y["yolo"][y["yfr"] == i]
            g, d = gxy[(gfr == i) & gkeep], gxy[(gfr == i) & gdis]
            for (yc, iou_keep) in settings:
                if yc is None:
                    kept = b
                else:
                    v = yy[yy[:, 4] >= yc, :4]
                    kept = b[bbox_ioa(b, v, iou=True).max(1) >= iou_keep] if len(b) and len(v) else b[:0]
                tp = match(kept, g)
                ign = ~tp & match(kept, d)                 # matched a distractor -> not counted
                st = stats[(yc, iou_keep)]
                st[0] += tp.sum(); st[1] += (~tp & ~ign).sum(); st[2] += len(g)
    return {f"conf{s[0]}_iou{s[1]}" if s[0] is not None else "no_verifier":
            {"recall": 100 * tp / ng, "precision": 100 * tp / max(tp + fp, 1)} for s, (tp, fp, ng) in stats.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HOME / "datasets/mot17/MOT17/train"))
    ap.add_argument("--cache", default=str(HOME / "datasets/mot17_cache"))
    ap.add_argument("--only", nargs="+", default=list(CANDIDATES))
    a = ap.parse_args()
    gt_root, cache = Path(a.root), Path(a.cache)
    settings = [(None, None)] + [(c, i) for c in (0.25, 0.4, 0.5, 0.7) for i in (0.3, 0.5)]
    res = {}
    for name in a.only:
        if CANDIDATES[name][0] and not CANDIDATES[name][0].exists():
            print(f"skip {name}: weights missing"); continue
        res[name] = evaluate(gt_root, cache, name, settings)
        print(f"\n## {name}\n| setting | Recall | Precision |\n|---|---|---|")
        for k, v in res[name].items():
            print(f"| {k} | {v['recall']:.1f} | {v['precision']:.1f} |", flush=True)
    out = Path("eval/results/verifier.json")
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**prev, **res}, indent=1))

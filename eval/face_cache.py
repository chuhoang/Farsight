"""Face crop cache (train_m3 A1-A3, B3; eval / training of a fine-tuned KP-RPE without re-running detector + aligner).

The exact face crops of eval.extract (BPJDet on the <= 32 body frames -> best face box -> crop pad 0.2 -> DFA), for
the tracklets of an existing store (its .meta.json), stored DFA-aligned. Per tracklet group <tid>
(attrs pid, camid, clothes):
  img (n,112,112,3) uint8 RGB aligned, ldmk (n,5,2) aligned landmarks in [0,1] (KP-RPE input),
  ipd (n,) eye distance in native crop pixels (degraded frames keep the clean size: effective IPD = ipd / down factor),
  score (n,) DFA face score, frame (n,) index into the tracklet's frame list, pos (n,) index into the body frames
  searched; body_h (<=32,) heights of the body
  frames searched for a face (A4: face detection rate vs person size).
With degradation (the store's ops, or --degrade): frames degraded exactly as eval.extract (same RNG order), plus A3
groups on the CLEAN frame's face box: a_img/a_ldmk/a_score = DFA on the degraded crop, b_img/b_ldmk = the clean
crop's DFA alignment applied to the degraded crop.
--folder DIR (face datasets, train_m3 B1): one sub-folder per identity of face images (whole image = face, e.g.
TinyFace Training_Set) -> DFA only, one group per identity (pid = <out>_<folder>).
Written as <out>/part_XXXX.h5 of 50 tracklets / identities (atomic rename); reruns skip finished parts.

    bash run.sh eval.face_cache --store mevid_query [--degrade down:4 turb:2 jpeg:30 --out NAME]
    bash run.sh eval.face_cache --folder ~/datasets/tinyface/tinyface/Training_Set --out tinyface_train
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import cv2
import h5py
import numpy as np

from eval.extract import FACE_IMGSZ, FACE_PAD, ccvid_tracklets, face_boxes, mevid_tracklets, plan_frames
from eval.regait import FILES
from farsight.core.registry import build
from farsight.io.video import crop
from tools.degrade_dataset import degrade

FEATS = Path.home() / "datasets/feats"
ROOT = Path.home() / "datasets/face_cache"
PART = 50


def u8(x):
    """(B,3,112,112) RGB [-1,1] tensor -> (B,112,112,3) uint8."""
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def ipd(ldmk_px):
    return np.linalg.norm(ldmk_px[:, 0] - ldmk_px[:, 1], axis=1).astype(np.float32)


class Cacher:
    def __init__(self, device="cuda"):
        self.det = build("bpjdet", device=device, imgsz=FACE_IMGSZ)
        self.al = build("dfa_aligner", device=device)

    def frames(self, frames, tid, ops):
        """Body frames given to the face path by eval.extract.Extractor (degradation RNG replayed in order)."""
        uni, gi = plan_frames(len(frames))
        need = sorted(set(uni.tolist()) | set(gi.tolist())) if ops else uni.tolist()
        rng = np.random.default_rng(zlib.crc32(tid.encode()))
        imgs = {}
        for i in need:
            im = cv2.imread(frames[i])
            imgs[i] = degrade(im, ops, rng) if ops else im
        return uni, [imgs[i] for i in uni]

    def __call__(self, frames, tid, ops):
        uni, body = self.frames(frames, tid, ops)
        out = self.aligned(body, face_boxes(self.det, body), uni) | {"body_h": np.array([b.shape[0] for b in body])}
        if ops:   # A3: clean boxes + clean alignment on the degraded frames
            clean = [cv2.imread(frames[i]) for i in uni]
            boxes = face_boxes(self.det, clean)
            if boxes:
                c = [crop(clean[i], b, pad=FACE_PAD) for i, b in boxes]
                d = [crop(body[i], b, pad=FACE_PAD) for i, b in boxes]
                ac, ad = self.al(c), self.al(d)
                out |= {"a_img": u8(ad["aligned"]), "a_ldmk": ad["ldmk_aligned"].cpu().numpy(),
                        "a_score": ad["score"].cpu().numpy(), "b_img": u8(self.al.warp(d, ac["theta"])),
                        "b_ldmk": ac["ldmk_aligned"].cpu().numpy(), "b_score": ac["score"].cpu().numpy(),
                        "ab_frame": uni[[i for i, _ in boxes]], "ab_ipd": ipd(ac["ldmk"])}
        return out

    def aligned(self, body, boxes, uni):
        if not boxes:
            return {"img": np.zeros((0, 112, 112, 3), np.uint8), "ldmk": np.zeros((0, 5, 2), np.float32),
                    "ipd": np.zeros(0, np.float32), "score": np.zeros(0, np.float32), "frame": np.zeros(0, int), "pos": np.zeros(0, int)}
        a = self.al([crop(body[i], b, pad=FACE_PAD) for i, b in boxes])
        return {"img": u8(a["aligned"]), "ldmk": a["ldmk_aligned"].cpu().numpy(), "ipd": ipd(a["ldmk"]),
                "score": a["score"].cpu().numpy(), "frame": uni[[i for i, _ in boxes]],
                "pos": np.array([i for i, _ in boxes])}


def read(name, tids=None, keys=("img", "ldmk", "ipd", "score")):
    """Cache -> {tid: {key: array} | attrs}; tids None = all. Arrays of other prefixes (a_, b_) via keys."""
    out = {}
    for p in sorted((ROOT / name).glob("part_*.h5")):
        with h5py.File(p, "r") as f:
            for t in f:
                if tids is None or t in tids:
                    out[t] = {k: f[t][k][()] for k in keys if k in f[t]} | dict(f[t].attrs)
    return out


def folder(root, name, limit=None):
    ids = sorted(p for p in root.iterdir() if p.is_dir())[:limit]
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    parts = [ids[i:i + PART] for i in range(0, len(ids), PART)]
    todo = [k for k in range(len(parts)) if not (out / f"part_{k:04d}.h5").exists()]
    print(f"{root} -> {name}: {len(ids)} identities, {len(parts) - len(todo)}/{len(parts)} parts done", flush=True)
    al = build("dfa_aligner") if todo else None
    for k in todo:
        tmp = out / f"part_{k:04d}.h5.tmp"
        with h5py.File(tmp, "w") as f:
            for d in parts[k]:
                ims = [cv2.imread(str(p)) for p in sorted(d.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
                a = al(ims)
                g = f.create_group(f"{name}_{d.name}")
                g.create_dataset("img", data=u8(a["aligned"]), compression="gzip", compression_opts=4)
                for key, v in (("ldmk", a["ldmk_aligned"].cpu().numpy()), ("score", a["score"].cpu().numpy()),
                               ("ipd", ipd(a["ldmk"]))):
                    g.create_dataset(key, data=v)
                g.attrs.update(pid=f"{name}_{d.name}", camid=0, clothes="")
        os.replace(tmp, out / f"part_{k:04d}.h5")
        print(f"  part {k}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", help="store name in ~/datasets/feats (tracklet list + dataset/split/ops)")
    ap.add_argument("--folder", help="face dataset root, one sub-folder per identity (instead of --store)")
    ap.add_argument("--degrade", nargs="*", help="override the store's degradation ops")
    ap.add_argument("--out", help="cache name (default = store name)")
    ap.add_argument("--ids", help="splits/x.json:key -> keep only these identities")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.folder:
        return folder(Path(a.folder).expanduser(), a.out, a.limit)
    d, split, ops = FILES[a.store]
    ops = a.degrade if a.degrade is not None else ops
    meta = json.loads(Path(FEATS, f"{a.store}.h5.meta.json").read_text())
    tr = (ccvid_tracklets if d == "ccvid" else mevid_tracklets)(split)
    keep = set(json.loads(Path(a.ids.rsplit(":", 1)[0]).read_text())[a.ids.rsplit(":", 1)[1]]) if a.ids else None
    tids = sorted(t for t in meta if keep is None or meta[t]["pid"] in keep)[:a.limit]
    out = ROOT / (a.out or a.store)
    out.mkdir(parents=True, exist_ok=True)
    parts = [tids[i:i + PART] for i in range(0, len(tids), PART)]
    todo = [k for k in range(len(parts)) if not (out / f"part_{k:04d}.h5").exists()]
    print(f"{a.store} -> {out.name}: {len(tids)} tracklets, {len(parts) - len(todo)}/{len(parts)} parts done, ops={ops}",
          flush=True)
    if not todo:
        return
    c, t0 = Cacher(), time.time()
    for n, k in enumerate(todo):
        tmp = out / f"part_{k:04d}.h5.tmp"
        with h5py.File(tmp, "w") as f:
            for tid in parts[k]:
                g = f.create_group(tid)
                for key, v in c(tr[tid]["frames"], tid, ops).items():
                    g.create_dataset(key, data=v, **({"compression": "gzip", "compression_opts": 4} if "img" in key else {}))
                for key in ("pid", "camid", "clothes"):
                    g.attrs[key] = meta[tid][key]
        os.replace(tmp, out / f"part_{k:04d}.h5")
        el = time.time() - t0
        print(f"  part {k} ({n + 1}/{len(todo)}) {el:.0f}s, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()

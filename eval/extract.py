"""Dataset tracklets (pre-cropped body frames) -> Templates via the real M3 encoders, streamed + resumable.

Per tracklet (n frames):
  body : <= 32 uniformly sampled frames -> AIM (LTCC); CSCI video picks its own clips over the whole tracklet
  face : the same <= 32 frames -> BPJDet (body crop as image, imgsz 320) face box -> crop pad 0.2 -> DFA + KP-RPE
         (KP-RPE drops frames with DFA score < 0.5; face = None if none pass)
  gait : middle window of <= 120 consecutive frames (4 x 30 clips) -> BigGait; < 15 frames -> None
Output: --out h5 (farsight.io.store, inter_feat kept) + <out>.meta.json {tid: pid/camid/clothes/session}
        + <out>.stats.jsonl (per tracklet: frames read, faces detected / kept, seconds).
Already-written tracklets are skipped, so a killed run just restarts.

    bash run.sh eval.extract --dataset ccvid --split query --out ~/datasets/feats/ccvid_query.h5
      [--ids splits/ccvid_qme.json:test] [--degrade down:4 turb:2 jpeg:30] [--limit 20]
"""
import argparse
import json
import os
import shutil
import time
import zlib
from collections.abc import Sequence
from pathlib import Path

import cv2
import h5py
import numpy as np

from farsight.core.registry import build
from farsight.io.store import write_templates


def save_batch(out, templates):
    """Append via copy -> write -> atomic rename: a crash mid-write (WSL killed) leaves `out` intact,
    only the last batch is lost and redone on resume (an in-place HDF5 append can truncate the whole file)."""
    tmp = Path(f"{out}.tmp")
    if Path(out).exists():
        shutil.copyfile(out, tmp)
    write_templates(tmp, templates)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, out)
from farsight.io.video import crop
from tools.degrade_dataset import degrade

HOME = Path.home() / "datasets"
BODY_N, FACE_N, GAIT_N = 32, 32, 120
FACE_IMGSZ, FACE_PAD, DET_HW = 320, 0.2, (256, 128)


def ccvid_tracklets(split, root=HOME / "ccvid/CCVID"):
    out = {}
    for line in Path(root, f"{split}.txt").read_text().split("\n"):
        if not line.strip():
            continue
        path, pid, clothes = line.split()
        sess, d = path.split("/")
        cam = int(d.split("_")[1]) + (12 if sess == "session3" else 0)  # = eval/ccvid_body.py (CAL loader)
        out[f"{sess}_{d}"] = {"pid": pid, "camid": cam, "clothes": f"{pid}_{clothes}", "session": sess,
                              "frames": [str(p) for p in sorted(Path(root, path).glob("*.jpg"))]}
    return out


def mevid_tracklets(split, root=HOME / "mevid"):
    """Kitware MEVID / QME loader: track_<train|test>_info rows = start, end, pid, outfit, camid;
    query = test rows in query_IDX, gallery = the other test rows."""
    ann = root / "mevid-v1-annotation-data/mevid-v1-annotation-data"
    part = "train" if split == "train" else "test"
    names = (ann / f"{part}_name.txt").read_text().split()
    info = np.loadtxt(ann / f"track_{part}_info.txt").astype(int)
    rows = range(len(info))
    if split != "train":
        q = set(np.loadtxt(ann / "query_IDX.txt").astype(int).tolist())
        rows = [i for i in rows if (i in q) == (split == "query")]
    img = root / f"mevid-v1-bbox-{part}/bbox_{part}"
    out = {}
    for i in rows:
        s, e, pid, outfit, cam = info[i]
        fr = names[s:e + 1]
        if fr:
            out[f"mevid{part}_{i:05d}"] = {"pid": f"{pid:04d}", "camid": int(cam), "clothes": f"{pid:04d}_{outfit}",
                                           "session": part, "frames": [str(img / n[:4] / n) for n in fr]}
    return out


def sample_per_id(tr, n):
    """Deterministic <= n tracklets per identity, round-robin over (outfit, camera) groups for diversity."""
    by = {}
    for t in sorted(tr):
        by.setdefault(tr[t]["pid"], {}).setdefault((tr[t]["clothes"], tr[t]["camid"]), []).append(t)
    keep = []
    for groups in by.values():
        queues, picked = [list(g) for _, g in sorted(groups.items())], []
        while len(picked) < n and any(queues):
            for q in queues:
                if q and len(picked) < n:
                    picked.append(q.pop(0))
        keep += picked
    return keep


def plan_frames(n):
    """-> (body/face indices, gait indices) into the tracklet's frame list."""
    uni = np.unique(np.linspace(0, n - 1, min(n, BODY_N)).round().astype(int))
    s = max(0, (n - GAIT_N) // 2)
    return uni, np.arange(s, min(n, s + GAIT_N))


class LazyFrames(Sequence):
    """Frame list read (and degraded) on first access; len() = full tracklet length."""

    def __init__(self, frames, tid, ops):
        self.frames, self.tid, self.ops, self.cache = frames, tid, ops, {}

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i not in self.cache:
            im = cv2.imread(self.frames[i])
            self.cache[i] = degrade(im, self.ops, np.random.default_rng(zlib.crc32(self.tid.encode()) + i)) if self.ops else im
        return self.cache[i]



def face_boxes(det, crops):
    """BPJDet on body crops (resized to DET_HW so they batch) -> [(crop index, best face box in crop pixels)]."""
    h, w = DET_HW
    res = det([cv2.resize(c, (w, h)) for c in crops])
    out = []
    for i, (c, r) in enumerate(zip(crops, res)):
        ok = ~np.isnan(r["face"][:, 0])
        if ok.any():
            j = np.argmax(np.where(ok, r["face"][:, 4], -1))
            out.append((i, r["face"][j, :4] * np.array([c.shape[1] / w, c.shape[0] / h] * 2)))
    return out


def detect_faces(det, crops):
    """-> face crops (pad FACE_PAD) of the frames with a face."""
    return [crop(crops[i], b, pad=FACE_PAD) for i, b in face_boxes(det, crops)]


class Extractor:
    def __init__(self, device="cuda"):
        from farsight.modules.m3_encode.module import M3Encode
        self.det = build("bpjdet", device=device, imgsz=FACE_IMGSZ)
        self.m3 = M3Encode(device=device)

    def __call__(self, tid, frames, ops=None):
        uni, gi = plan_frames(len(frames))
        need = sorted(set(uni.tolist()) | set(gi.tolist()))
        rng = np.random.default_rng(zlib.crc32(tid.encode()))  # deterministic per tracklet
        imgs = {}
        for i in need:
            im = cv2.imread(frames[i])
            imgs[i] = degrade(im, ops, rng) if ops else im
        body = [imgs[i] for i in uni]
        faces = detect_faces(self.det, body)
        if hasattr(self.m3.body, "frames"):     # clip-based body encoder (CSCI video): picks its own frames
            body = LazyFrames(frames, tid, ops)
        t = {"subject_or_track_id": tid, "qe_weight": None,
             "face": self.m3.face.encode(faces, None) if faces else None,
             "gait": self.m3.gait.encode([imgs[i] for i in gi], None),
             "body": self.m3.body.encode(body, None)}
        return t, {"n_read": len(need), "n_face_det": len(faces),
                   "n_face_kept": t["face"]["n_frames"] if t["face"] else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ccvid", "mevid"])
    ap.add_argument("--split", required=True, choices=["train", "query", "gallery"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", help="splits/x.json:key -> keep only these identities")
    ap.add_argument("--degrade", nargs="*", help="tools.degrade_dataset ops, e.g. down:4 turb:2 jpeg:30")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per_id", type=int, help="keep <= N tracklets per identity (round-robin over outfit x camera)")
    a = ap.parse_args()
    tr = (ccvid_tracklets if a.dataset == "ccvid" else mevid_tracklets)(a.split)
    if a.ids:
        f, k = a.ids.rsplit(":", 1)
        keep = set(json.loads(Path(f).read_text())[k])
        tr = {t: v for t, v in tr.items() if v["pid"] in keep}
    if a.per_id:
        tr = {t: tr[t] for t in sample_per_id(tr, a.per_id)}
    tids = sorted(tr)[:a.limit]
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {t: {k: v for k, v in tr[t].items() if k != "frames"} | {"n_frames_total": len(tr[t]["frames"])}
            for t in tids}
    Path(f"{out}.meta.json").write_text(json.dumps(meta, indent=0))
    done = set()
    if out.exists():
        with h5py.File(out, "r") as f:
            done = set(f.keys())
    todo = [t for t in tids if t not in done]
    print(f"{a.dataset}/{a.split}: {len(tids)} tracklets, {len(done)} done, {len(todo)} to go, degrade={a.degrade}",
          flush=True)
    if not todo:
        return
    ex = Extractor()
    t0, buf = time.time(), []
    with open(f"{out}.stats.jsonl", "a") as log:
        for k, tid in enumerate(todo):
            s = time.time()
            t, st = ex(tid, tr[tid]["frames"], a.degrade)
            buf.append(t)
            log.write(json.dumps({"tid": tid, **st, "sec": round(time.time() - s, 3)}) + "\n")
            if len(buf) == 20 or k == len(todo) - 1:  # flush to disk every 20 tracklets
                save_batch(out, buf)
                log.flush()
                buf = []
                el = time.time() - t0
                print(f"  {k + 1}/{len(todo)} {el:.0f}s ({el / (k + 1):.2f} s/trk, eta {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min)",
                      flush=True)


if __name__ == "__main__":
    main()

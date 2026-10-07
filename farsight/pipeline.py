"""enroll(video, subject_id) / search(video): M1 -> (M2) -> M3 -> (M4). Only calls each module.py.

python -m farsight.pipeline enroll VIDEO SUBJECT_ID [--gallery g.h5] [--config configs/pipeline/v1.yaml]
python -m farsight.pipeline search VIDEO [--gallery g.h5]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from farsight.core.weights import ROOT
from farsight.io.store import read_templates, write_templates
from farsight.io.video import crop, read_frames

FACE_PAD = 0.2       # plan M2.1: faces cropped with 20% margin
MAX_GAIT = 120       # ponytail: keep at most 4x30 frames of body crops per track in RAM; raise if tracks are long


class FarSight:
    def __init__(self, config="configs/pipeline/v1.yaml", device="cuda"):
        self.cfg = yaml.safe_load(Path(ROOT, config).read_text())
        self.device = device
        self._m1 = self._m2 = self._m3 = self._m4 = None

    # modules are built on first use so e.g. enrolment never loads fusion
    @property
    def m1(self):
        if self._m1 is None:
            from farsight.modules.m1_detect_track.module import M1DetectTrack
            self._m1 = M1DetectTrack(self.cfg["m1_detect_track"], device=self.device)
        return self._m1

    @property
    def m3(self):
        if self._m3 is None:
            from farsight.modules.m3_encode.module import M3Encode
            self._m3 = M3Encode(self.cfg["m3_encode"], device=self.device)
        return self._m3

    @property
    def m2(self):
        if self._m2 is None:
            from farsight.modules.m2_restore.module import M2Restore
            f = self.m3.face
            self._m2 = M2Restore(f.quality, f.embed, cfg=self.cfg["m2_restore"])
        return self._m2

    @property
    def m4(self):
        if self._m4 is None:
            from farsight.modules.m4_fusion.module import M4
            self._m4 = M4(self.cfg["m4_fusion"])
        return self._m4

    def templates(self, video, video_id=None, max_frames=None):
        """Video -> one Template per tracklet (id = "<video_id>:<track_id>")."""
        vid = video_id or Path(video).stem
        tracks = self.last_tracks = self.m1.run(video, video_id=vid, max_frames=max_frames)
        if not tracks:
            return []
        # second decode pass: collect crops only for frames the tracklets kept
        want = defaultdict(list)
        for t in tracks:
            for o in _gait_window(t["frames"]):
                want[o["frame_idx"]].append((t["track_id"], o))
        faces, bodies = defaultdict(list), defaultdict(list)
        last = max(want)
        for start, frames in read_frames(video, batch=8):
            for k, fr in enumerate(frames):
                for tid, o in want.get(start + k, ()):
                    bodies[tid].append(crop(fr, o["body_box"]))
                    if o["face_box"] is not None:
                        faces[tid].append(crop(fr, o["face_box"], pad=FACE_PAD))
            if start + len(frames) > last:
                break
        out = []
        for t in tracks:
            tid = t["track_id"]
            fc = faces[tid]
            if fc and self.cfg["m2_restore"].get("enabled"):
                fc, _ = self.m2(fc)
            out.append(self.m3(f"{vid}:{tid}", t, fc, bodies[tid]))
        return out

    def enroll(self, video, subject_id, gallery, **kw):
        """Enrolment video of one subject: keep the longest track's template under subject_id."""
        ts = self.templates(video, **kw)
        if not ts:
            raise ValueError(f"no usable track in {video}")
        t = max(ts, key=lambda t: max((t[m] or {}).get("n_frames", 0) for m in ("face", "gait", "body")))
        t["subject_or_track_id"] = subject_id
        write_templates(gallery, [t])
        return t

    def search(self, video, gallery, **kw):
        g = read_templates(gallery) if isinstance(gallery, (str, Path)) else gallery
        return [self.m4.search(p, g) for p in self.templates(video, **kw)]


def _gait_window(frames):
    """Contiguous middle window of at most MAX_GAIT observations (gait wants consecutive frames)."""
    s = max(0, (len(frames) - MAX_GAIT) // 2)
    return frames[s:s + MAX_GAIT]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["enroll", "search"])
    ap.add_argument("video")
    ap.add_argument("subject_id", nargs="?")
    ap.add_argument("--gallery", default="gallery.h5")
    ap.add_argument("--config", default="configs/pipeline/v1.yaml")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    fs = FarSight(a.config)
    if a.cmd == "enroll":
        t = fs.enroll(a.video, a.subject_id, a.gallery, max_frames=a.max_frames)
        print(a.subject_id, {m: (t[m] or {}).get("n_frames") for m in ("face", "gait", "body")})
    else:
        for r in fs.search(a.video, a.gallery, max_frames=a.max_frames):
            top = r["ranked"][:5]
            print(json.dumps({"probe": r["probe_id"], "known": r["is_known"],
                              "top": [(x["gallery_id"], round(x["score"], 3)) for x in top]}))

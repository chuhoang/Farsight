"""Demo: find a person, given by image(s), among the people walking in a video.

    bash run.sh demo.infer --query person.jpg --video street.mp4 --out demo/out.mp4
    bash run.sh demo.infer --query frames_dir/ --video street.mp4           # >= 15 frames -> gait is used too

Query image(s): BPJDet crops the most confident person in each image (an image without a detection is used whole,
i.e. already-cropped person photos work). One image -> face + body; >= 15 consecutive frames -> face + gait + body.
Video: M1 (detect + track) -> M3 templates per track -> fusion v1 score of each track against the query.
Tracks scoring >= tau are the target; tau = fusion v1 threshold at 1% FPIR on CCVID val for the modalities that both
the query and the track have (eval/results/zscore_v1_tau_by_modalities.json); --threshold overrides it:
green box "TARGET"; others red (if nobody passes, the best track is drawn yellow "CANDIDATE"). Prints when/where the target appears and writes the annotated video.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from farsight.core.weights import ROOT
from farsight.io.video import crop, read_frames
from farsight.modules.m4_fusion.zscore.model import ZScoreFusion
from farsight.pipeline import FACE_PAD, FarSight

GREEN, YELLOW, RED = (0, 200, 0), (0, 220, 255), (0, 0, 255)   # BGR
STATS = ROOT / "eval/results/zscore_v1_ccvid_val.json"   # fusion v1 fitted on CCVID val (eval.main_protocol)
TAUS = ROOT / "eval/results/zscore_v1_tau_by_modalities.json"   # FPIR 1% per modality combo (eval.fit_tau_modalities)
MODS = ("face", "gait", "body")


def query_template(fs, paths):
    """Query images -> one Template (face / gait / body as available)."""
    frames = [cv2.imread(str(p)) for p in paths]
    if any(f is None for f in frames):
        raise ValueError(f"unreadable image in {paths}")
    bodies, faces, obs = [], [], []
    for i, (f, d) in enumerate(zip(frames, fs.m1.det(frames))):
        h, w = f.shape[:2]
        b, fc = (d["body"][d["body"][:, 4].argmax()], None) if len(d["body"]) else (np.array([0, 0, w, h, 1.0]), None)
        if len(d["body"]):
            fc = d["face"][d["body"][:, 4].argmax()]
        bodies.append(crop(f, b[:4]))
        has_face = fc is not None and not np.isnan(fc[0])
        if has_face:
            faces.append(crop(f, fc[:4], pad=FACE_PAD))
        obs.append({"frame_idx": i, "body_box": b[:4].tolist(), "body_score": float(b[4]),
                    "face_box": fc[:4].tolist() if has_face else None, "face_score": float(fc[4]) if has_face else None})
    track = {"video_id": "query", "track_id": 0, "frames": obs}
    return fs.m3("query", track, faces, bodies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", nargs="+", required=True, help="image file(s) or one directory of frames")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="demo/out.mp4")
    ap.add_argument("--threshold", type=float, help="fused-score threshold (default: tau fitted on CCVID val)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-restore", action="store_true", help="skip M2 face restoration (DATUM)")
    a = ap.parse_args()

    q = [Path(p) for p in a.query]
    if len(q) == 1 and q[0].is_dir():
        q = sorted(p for p in q[0].iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    fs = FarSight()
    if a.no_restore:
        fs.cfg["m2_restore"]["enabled"] = False
    qt = query_template(fs, q)
    have = [m for m in ("face", "gait", "body") if qt[m] is not None]
    print(f"query: {len(q)} image(s), modalities: {', '.join(have) or 'none'}")
    if not have:
        raise SystemExit("no usable person in the query image(s)")

    fusion = ZScoreFusion(stats=str(STATS), use_quality=False)   # v1 was fitted without quality weighting
    taus = json.loads(TAUS.read_text())
    temps = fs.templates(a.video, max_frames=a.max_frames)
    tracks = {t["track_id"]: t for t in fs.last_tracks}
    fps = fs.m1.fps   # set from the video by M1.run
    scores = {}
    for t in temps:
        r = fusion.search(t, [qt])["ranked"][0]          # probe = video track, gallery = the query person
        tid = int(t["subject_or_track_id"].rsplit(":", 1)[1])
        # threshold of the modalities both sides have (a face-less match can't reach the face-dominated global tau)
        common = "+".join(m for m, v in zip(MODS, r["per_modality"]) if v == v)
        tau = a.threshold if a.threshold is not None else taus[common]["tau"] if common else float("inf")
        scores[tid] = (r["score"], r["per_modality"], tau)
    targets = {tid for tid, (s, _, tau) in scores.items() if s == s and s >= tau}

    print(f"\n{len(tracks)} people tracked")
    for tid, (s, pm, tau) in sorted(scores.items(), key=lambda x: -np.nan_to_num(x[1][0], nan=-1e9)):
        fr = [o["frame_idx"] for o in tracks[tid]["frames"]]
        mark = "TARGET" if tid in targets else "      "
        print(f"  {mark} track {tid:3d}  score {s:6.2f}  (face/gait/body cos: "
              + "/".join("-" if v != v else f"{v:.2f}" for v in pm) + f")  tau {tau:.2f}  t={fr[0] / fps:.1f}-{fr[-1] / fps:.1f}s")
    if targets:
        print(f"\n>>> Person from the query FOUND in the video: track(s) {sorted(targets)}")
    else:
        best = max(scores.items(), key=lambda x: np.nan_to_num(x[1][0], nan=-1e9)) if scores else None
        print("\n>>> Person from the query NOT found" + (f" (best candidate: track {best[0]}, score {best[1][0]:.2f} "
                                                          f"< tau {best[1][2]:.2f}; drawn yellow)" if best else ""))
    candidate = None if targets or not scores else best[0]

    # draw: one more pass over the video
    boxes = {}
    for tid, t in tracks.items():
        for o in t["frames"]:
            boxes.setdefault(o["frame_idx"], []).append((tid, o["body_box"]))
    writer = None
    n = 0
    for start, frames in read_frames(a.video, batch=8):
        for k, f in enumerate(frames):
            i = start + k
            if a.max_frames is not None and i >= a.max_frames:
                break
            if writer is None:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
            for tid, (x1, y1, x2, y2) in boxes.get(i, []):
                hit, cand = tid in targets, tid == candidate
                c, label = (GREEN, "TARGET") if hit else (YELLOW, "CANDIDATE") if cand else (RED, f"id {tid}")
                s = scores.get(tid, (float("nan"),))[0]
                cv2.rectangle(f, (int(x1), int(y1)), (int(x2), int(y2)), c, 3 if hit or cand else 1)
                cv2.putText(f, f"{label} {s:.1f}", (int(x1), max(12, int(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2 if hit or cand else 1)
            writer.write(f)
            n += 1
    if writer is not None:
        writer.release()
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"query": [str(p) for p in q], "query_modalities": have, "targets": sorted(targets),
         "tracks": {str(k): {"score": v[0], "per_modality": v[1], "tau": v[2]} for k, v in scores.items()}}, indent=1, default=float))
    print(f"wrote {a.out} ({n} frames) and {Path(a.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()

"""End-to-end M1 on OpenCV's pedestrian sample vtest.avi (768x576, 10 fps)."""
import urllib.request

import pytest

from farsight.core.weights import WEIGHTS

VIDEO = WEIGHTS / "m1_detect_track/test_data/vtest.avi"
URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
N = 200


@pytest.fixture(scope="module")
def video():
    if not VIDEO.exists():
        VIDEO.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, VIDEO)
    return VIDEO


@pytest.fixture(scope="module")
def m1():
    from farsight.modules.m1_detect_track.module import M1DetectTrack
    return M1DetectTrack({"detector": "bpjdet", "verifier": "yolov8_verifier", "tracker": "bytetrack",
                          "appearance": "psr_appearance"}, fps=10)


def test_end_to_end(m1, video):
    tl = m1.run(video, "vtest", max_frames=N)
    assert m1.stats["frames"] == N
    assert len(tl) >= 3
    ids = [t["track_id"] for t in tl]
    assert len(ids) == len(set(ids))
    for t in tl:
        fr = t["frames"]
        assert t["video_id"] == "vtest" and len(fr) >= 15
        idx = [o["frame_idx"] for o in fr]
        assert idx == sorted(set(idx)) and 0 <= idx[0] and idx[-1] < N     # one box per frame per track
        for o in fr:
            x1, y1, x2, y2 = o["body_box"]
            assert y2 - y1 >= 32 and x2 > x1 and 0 < o["body_score"] <= 1
            assert (o["face_box"] is None) == (o["face_score"] is None)
    # stable IDs: long tracks exist (people stay tracked under one ID for most of their visible time)
    assert max(len(t["frames"]) for t in tl) >= 0.5 * N
    print(f"\nM1 vtest {N} frames: {len(tl)} tracklets, {m1.stats['fps']:.1f} FPS")


def test_deterministic(m1, video):
    a = m1.run(video, "v", max_frames=40)
    b = m1.run(video, "v", max_frames=40)
    assert [(t["track_id"], len(t["frames"])) for t in a] == [(t["track_id"], len(t["frames"])) for t in b]

import numpy as np

from farsight.io.store import read_templates, write_templates
from farsight.io.video import crop


def _mo(d, inter=True):
    f = np.random.randn(d).astype(np.float32)
    return {"feat": f / np.linalg.norm(f), "quality": 0.7, "n_frames": 12,
            "inter_feat": np.ones(4, np.float32) if inter else None}


def test_store_roundtrip_with_missing(tmp_path):
    ts = [
        {"subject_or_track_id": "a", "face": _mo(512), "gait": None, "body": _mo(4096, False),
         "qe_weight": {"face": 0.9, "gait": 0.0, "body": 0.4}},
        {"subject_or_track_id": "b", "face": None, "gait": _mo(256), "body": None, "qe_weight": None},
    ]
    p = tmp_path / "g.h5"
    write_templates(p, ts, mode="w")
    back = {t["subject_or_track_id"]: t for t in read_templates(p)}
    assert back["a"]["gait"] is None and back["b"]["face"] is None
    assert np.allclose(back["a"]["face"]["feat"], ts[0]["face"]["feat"])
    assert back["a"]["body"]["inter_feat"] is None
    assert back["a"]["qe_weight"]["face"] == 0.9 and back["b"]["qe_weight"] is None
    write_templates(p, [ts[1]], keep_inter=False)  # overwrite existing id
    assert len(read_templates(p)) == 2


def test_crop_clips():
    fr = np.zeros((100, 200, 3), np.uint8)
    assert crop(fr, [10, 10, 50, 90]).shape == (80, 40, 3)
    assert crop(fr, [-20, -20, 300, 300]).shape == (100, 200, 3)
    assert crop(fr, [10, 10, 50, 90], pad=0.2).shape == (100, 56, 3)  # y clipped at 0 and 100


def test_store_rejects_slash_id(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        write_templates(tmp_path / "g.h5", [{"subject_or_track_id": "vid/3", "face": None, "gait": None, "body": None}])


def test_video_fps():
    from farsight.core.weights import WEIGHTS
    from farsight.io.video import video_fps
    v = WEIGHTS / "m1_detect_track/test_data/vtest.avi"
    if v.exists():
        assert video_fps(v) == 10.0

import numpy as np

from farsight.core import registry


def _run(n=40, crossing=False):
    tr = registry.build("bytetrack")
    ids = []
    for t in range(n):
        a = [100 + 5 * t, 100, 160 + 5 * t, 260, 0.9]
        b = [600 - (5 * t if crossing else 0), 120, 660 - (5 * t if crossing else 0), 280, 0.85]
        out = tr.update(np.array([a, b], np.float32))
        ids.append({int(r[6]): int(r[4]) for r in out})   # det_idx -> track_id
    return ids


def test_two_walkers_stable_ids():
    ids = _run()
    assert ids[5] and len(set(ids[5].values())) == 2
    assert all(i == ids[5] for i in ids[5:])               # same detection keeps same ID


def test_deterministic_and_empty():
    assert _run() == _run()
    tr = registry.build("bytetrack")
    assert tr.update(np.zeros((0, 5))).shape == (0, 7)


def test_low_score_box_keeps_track():
    tr = registry.build("bytetrack")
    for t in range(10):
        out = tr.update(np.array([[100 + 2 * t, 100, 160 + 2 * t, 260, 0.9]]))
    tid = out[0, 4]
    out = tr.update(np.array([[120, 100, 180, 260, 0.15]]))  # ByteTrack second-stage (low-score) association
    assert len(out) == 1 and out[0, 4] == tid

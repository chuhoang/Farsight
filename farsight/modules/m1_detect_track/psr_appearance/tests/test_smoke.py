import glob

import cv2
import numpy as np
import pytest

from farsight.core import registry
from farsight.core.weights import ROOT

IMG = sorted(glob.glob(str(ROOT / "third_party/BPJDet/test_imgs/CrowdHuman/*.jpg")))[1]  # 1200x800 boat photo


@pytest.fixture(scope="module")
def psr():
    return registry.build("psr_appearance")


def test_embed_real(psr):
    f = cv2.imread(IMG)
    woman, woman2, man = [760, 335, 1170, 800], [770, 345, 1165, 795], [340, 465, 830, 800]
    e = psr.embed(f, np.array([woman, woman2, man], np.float32))
    assert e.shape == (3, 512) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-3)
    assert 1 - e[0] @ e[1] < 1 - e[0] @ e[2]                  # same person closer than another person
    np.testing.assert_allclose(e, psr.embed(f, np.array([woman, woman2, man], np.float32)), atol=1e-3)
    assert psr.embed(f, np.zeros((0, 4))).shape == (0, 512)


class FakePSR:
    """Logic test without the CNN: feature = one-hot of the 'true person', read from box x2 (unused by tracker)."""
    @staticmethod
    def build():
        p = registry.get("psr_appearance").__new__(registry.get("psr_appearance"))
        p.cfg = dict(mem_size=10, mem_every=10, metric="cosine", thr=0.15, overlap_iou=0.3, swap_margin=0.05)
        p.window, p.alias, p.mem, p.last_seen, p.last_add, p.pairs = 300, {}, {}, {}, {}, set()
        p.embed = lambda frame, boxes: np.eye(512, dtype=np.float32)[(np.asarray(boxes)[:, 3] % 7).astype(int)]
        return p


def row(x, raw, person):
    return [x, 100, x + 60, 300 + person, raw, 0.9, 0]   # y2 % 7 = person identity


def test_new_id_remapped_to_lost_identity():
    p = FakePSR.build()
    for t in range(30):
        p(t, None, np.array([row(100, 1, 1), row(400, 2, 2)], np.float32))
    # person 1 lost then re-appears as ByteTrack raw ID 3 -> should get ID 1 back; unrelated person 4 keeps new ID
    out = p(60, None, np.array([row(120, 3, 1), row(400, 2, 2), row(700, 4, 4)], np.float32))
    assert out[:, 4].tolist() == [1, 2, 4]
    out = p(61, None, np.array([row(122, 3, 1), row(400, 2, 2)], np.float32))
    assert out[:, 4].tolist() == [1, 2]                     # alias persists


def test_revived_raw_track_keeps_its_id_no_duplicates():
    p = FakePSR.build()
    for t in range(30):
        p(t, None, np.array([row(100, 1, 1)], np.float32))
    p(40, None, np.array([row(120, 3, 1)], np.float32))          # person 1 back as raw 3 -> aliased to ID 1
    # ByteTrack now revives raw track 1 too (e.g. look-alike re-matched): IDs must stay unique, raw 1 keeps ID 1
    out = p(41, None, np.array([row(100, 1, 1), row(120, 3, 1)], np.float32))
    assert sorted(out[:, 4].tolist()) == [1, 3]
    assert out[0, 4] == 1


def test_no_remap_onto_present_or_expired_id():
    p = FakePSR.build()
    for t in range(20):
        p(t, None, np.array([row(100, 1, 1)], np.float32))
    out = p(20, None, np.array([row(100, 1, 1), row(400, 5, 1)], np.float32))  # look-alike while ID 1 is present
    assert out[:, 4].tolist() == [1, 5]
    p2 = FakePSR.build()
    p2(0, None, np.array([row(100, 1, 1)], np.float32))
    out = p2(1000, None, np.array([row(100, 7, 1)], np.float32))            # beyond window T
    assert out[:, 4].tolist() == [7]


def test_swap_after_crossing_is_undone():
    p = FakePSR.build()
    for t in range(20):
        p(t, None, np.array([row(100, 1, 1), row(400, 2, 2)], np.float32))
    p(20, None, np.array([row(250, 1, 1), row(255, 2, 2)], np.float32))    # overlapping
    # ByteTrack swapped them during the crossing: raw 1 now on person 2, raw 2 on person 1
    out = p(21, None, np.array([row(400, 1, 2), row(100, 2, 1)], np.float32))
    assert out[:, 4].tolist() == [2, 1]
    out = p(22, None, np.array([row(402, 1, 2), row(98, 2, 1)], np.float32))
    assert out[:, 4].tolist() == [2, 1]

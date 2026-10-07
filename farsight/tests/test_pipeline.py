"""End-to-end: vtest.avi (OpenCV sample, fetched by M1 tests) -> templates -> gallery HDF5 -> search."""
import numpy as np
import pytest
import torch

from farsight.core.weights import WEIGHTS
from farsight.pipeline import FarSight

VIDEO = WEIGHTS / "m1_detect_track" / "test_data" / "vtest.avi"


@pytest.mark.skipif(not VIDEO.exists() or not torch.cuda.is_available(), reason="needs vtest.avi + GPU")
def test_end_to_end(tmp_path):
    fs = FarSight()
    ts = fs.templates(VIDEO, max_frames=120)
    assert ts, "no tracklets"
    for t in ts:
        for m in ("gait", "body"):
            if t[m] is not None:
                assert np.isclose(np.linalg.norm(t[m]["feat"]), 1, atol=1e-3)
    assert any(t["body"] is not None for t in ts)

    g = tmp_path / "gallery.h5"
    from farsight.io.store import write_templates
    write_templates(g, ts, mode="w")
    res = fs.search(VIDEO, g, max_frames=120)
    assert len(res) == len(ts)
    # same video as probe and gallery: every probe must retrieve its own template first
    hits = [r["ranked"][0]["gallery_id"] == r["probe_id"] for r in res]
    assert all(hits), hits
    print(f"\n{len(ts)} tracks; faces={sum(t['face'] is not None for t in ts)} "
          f"gait={sum(t['gait'] is not None for t in ts)}; M1 {fs.m1.stats['fps']:.1f} FPS; "
          f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

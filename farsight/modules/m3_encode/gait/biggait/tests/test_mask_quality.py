import numpy as np

from farsight.modules.m3_encode.gait.biggait.preprocess import mask_quality


def _person(S=60, x0=10, x1=22):
    fg = np.zeros((S, 64, 32), bool)
    fg[:, 8:60, 13:19] = True                       # body inside the valid band, away from its borders
    valid = np.zeros((S, 64, 32), bool)
    valid[:, :, x0:x1] = True
    return fg, valid


def test_clean_track_scores_high():
    fg, valid = _person()
    q, p = mask_quality(fg, valid, [200] * 60)
    assert q > 0.9 and min(p.values()) > 0.9


def test_each_failure_mode_lowers_quality():
    fg, valid = _person()
    good = mask_quality(fg, valid, [200] * 60)[0]
    bg = valid.copy()                                # mask = whole valid region (background chosen)
    assert mask_quality(bg, valid, [200] * 60)[0] < 0.5 * good
    flicker = fg.copy()
    flicker[::2] = False                            # mask on/off every other frame
    assert mask_quality(flicker, valid, [200] * 60)[0] < 0.5 * good
    cut = np.zeros_like(fg)
    cut[:, 8:60, 10:16] = True                      # body glued to the left valid border
    assert mask_quality(cut, valid, [200] * 60)[1]["unoccl"] < 0.1
    assert mask_quality(fg[:10], valid[:10], [200] * 10)[1]["length"] < 0.2
    assert mask_quality(fg, valid, [50] * 60)[0] < 0.5 * good           # tiny person

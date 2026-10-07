"""train_m3 B2: random "long-range" degradation of a DFA-aligned 112x112 face, at the face's native scale.

The aligned face is a resample of a native crop whose eye distance is `ipd` px (the canonical aligned IPD is ~35 px).
A target native IPD t <= ipd is drawn (log-uniform in [ipd_min, ipd]); the face is resized to that scale, blur /
turbulence / sensor noise / JPEG are applied there (same ops and pixel units as tools.degrade_dataset, i.e. as the
degraded CCVID eval set), then it is upsampled back to 112x112. Landmarks do not move (no geometric op).
"""
import cv2
import numpy as np

from tools.degrade_dataset import gblur, jpeg, mblur, noise, turb

ALIGNED_IPD = 35.2   # ArcFace 5-point template: eyes at x = 38.29, 73.53 of 112

DEFAULT = {"ipd_min": 2.0, "p_gblur": 0.3, "gblur": (0.3, 1.5), "p_mblur": 0.2, "mblur": (3, 7), "p_turb": 0.5,
           "turb": (0.5, 3.0), "p_noise": 0.3, "noise": (1.0, 8.0), "p_jpeg": 0.7, "jpeg": (10, 60),
           "p_gamma": 0.3, "gamma": (0.6, 1.6)}


def degrade(img, ipd, rng, cfg=DEFAULT):
    """img (112,112,3) uint8, ipd = native eye distance of the crop -> (degraded uint8, params dict)."""
    c = {**DEFAULT, **(cfg or {})}
    hi = max(float(ipd), c["ipd_min"])
    t = float(np.exp(rng.uniform(np.log(c["ipd_min"]), np.log(hi))))
    s = min(1.0, t / ALIGNED_IPD)
    side = max(4, round(112 * s))
    x = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA) if side < 112 else img.copy()
    p = {"ipd": round(t, 2)}
    if rng.random() < c["p_gblur"]:
        p["gblur"] = rng.uniform(*c["gblur"])
        x = gblur(x, p["gblur"])
    if rng.random() < c["p_mblur"]:
        p["mblur"] = int(rng.integers(c["mblur"][0], c["mblur"][1] + 1)) | 1
        x = mblur(x, p["mblur"], float(rng.uniform(0, 180)))
    if rng.random() < c["p_turb"]:
        p["turb"] = rng.uniform(*c["turb"])
        x = turb(x, p["turb"], rng=rng)
    if rng.random() < c["p_noise"]:
        p["noise"] = rng.uniform(*c["noise"])
        x = noise(x, p["noise"], rng=rng)
    if rng.random() < c["p_gamma"]:
        p["gamma"] = rng.uniform(*c["gamma"])
        x = np.clip(255 * (x / 255.0) ** p["gamma"], 0, 255).astype(np.uint8)
    if rng.random() < c["p_jpeg"]:
        p["jpeg"] = int(rng.integers(*c["jpeg"]))
        x = jpeg(x, p["jpeg"])
    if side < 112:
        x = cv2.resize(x, (112, 112), interpolation=cv2.INTER_CUBIC)
    return x, p

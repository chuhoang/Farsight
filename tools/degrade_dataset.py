"""Synthetic long-range degradations (plan 5.2) for probes / QME missing-modality data.
All ops: BGR uint8 HxWx3 -> same shape/dtype.

python -m tools.degrade_dataset SRC_DIR DST_DIR down:4 gblur:1.5 turb:2 jpeg:30 [--seed 0]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def down(img, factor=4):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, round(w / factor)), max(1, round(h / factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def gblur(img, sigma=1.5):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def mblur(img, length=9, angle=0.0):
    k = np.zeros((length, length), np.float32)
    k[length // 2] = 1
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle, 1), (length, length))
    return cv2.filter2D(img, -1, k / k.sum())


def turb(img, strength=2.0, corr=0.1, rng=None):
    """Turbulence-like tilt: smooth random displacement field, ~`strength` px RMS, correlation
    length `corr` * image size. ponytail: tilt only, no Zernike PSF; use DATUM's simulator for D/r0."""
    rng = np.random.default_rng(rng)
    h, w = img.shape[:2]
    g = max(2, round(1 / corr))
    f = [cv2.resize(rng.standard_normal((g, g)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) for _ in "xy"]
    f = [d * strength / (d.std() + 1e-6) for d in f]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + f[0], gy + f[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def noise(img, sigma=5.0, rng=None):
    n = np.random.default_rng(rng).normal(0, sigma, img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)


def jpeg(img, quality=30):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1], cv2.IMREAD_COLOR)


OPS = {"down": down, "gblur": gblur, "mblur": mblur, "turb": turb, "noise": noise, "jpeg": jpeg}


def degrade(img, ops, rng=None):
    """ops: ["down:4", "mblur:9:30", "turb:2", ...] = name:positional args."""
    rng = np.random.default_rng(rng)
    for op in ops:
        name, *args = op.split(":")
        args = [int(a) if name == "mblur" and i == 0 else float(a) for i, a in enumerate(args)]
        img = OPS[name](img, *args, **({"rng": rng} if name in ("turb", "noise") else {}))
    return img


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src"), p.add_argument("dst"), p.add_argument("ops", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    src = Path(a.src)
    for f in sorted(x for x in src.rglob("*") if x.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")):
        out = Path(a.dst) / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), degrade(cv2.imread(str(f)), a.ops, rng))

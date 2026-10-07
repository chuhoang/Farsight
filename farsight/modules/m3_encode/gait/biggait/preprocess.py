"""OpenGait RGB preprocessing (datasets/pretreatment_rgb.py + BaseRgbTransform) and 30-frame clip splitting."""
import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def resize_with_padding(img, target_size):
    """Copied from OpenGait datasets/pretreatment_rgb.py: keep aspect, height=target_h, center pad/crop width."""
    h, w, _ = img.shape
    target_h, target_w = target_size
    resized_img = cv2.resize(img, (int(w * target_h / h), target_h))
    padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - resized_img.shape[1]) // 2
    if x_offset < 0:
        x_offset = abs(x_offset)
        padded_img = resized_img[:, x_offset:x_offset + target_w, :]
    else:
        padded_img[:, x_offset:x_offset + resized_img.shape[1]] = resized_img
    return padded_img


def preprocess(crops, size=(256, 128)):
    """BGR crops -> (frames (S, 3, H, W) float32 normalised RGB, ratios (S,) = crop w/h)."""
    x = np.stack([cv2.cvtColor(resize_with_padding(c, size), cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32)
    x = (x - MEAN) / STD
    ratios = np.array([c.shape[1] / c.shape[0] for c in crops], np.float32)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(ratios)


def clip_starts(n, clip_len=30):
    """Start indices of clip_len-frame clips covering n frames; last clip is right-aligned (overlaps) so
    all clips have equal length and batch together. n < clip_len -> one clip of n frames."""
    if n <= clip_len:
        return [0]
    return list(range(0, n - clip_len, clip_len)) + [n - clip_len]


def quality(crops, min_h, full_len):
    """Fraction of frames whose body box height >= min_h  x  min(1, n / full_len), in [0, 1]."""
    big = np.mean([c.shape[0] >= min_h for c in crops])
    return float(big * min(1.0, len(crops) / full_len))


def mask_quality(fg, valid, heights, area_range=(0.2, 0.75), full_len=60, h_lo=64, h_hi=160):
    """Gait quality from BigGait's own foreground masks (no extra forward pass).
    fg, valid: (S, 64, 32) bool = body mask and non-padded region per frame; heights: original crop heights (px).
    Components in [0, 1] (geometric mean -> one bad factor pulls the score down):
      area   frames whose body area / valid area lies in area_range (too small: person lost; too big: background)
      stable mean IoU of consecutive masks, mapped 0.5 -> 0, 0.9 -> 1 (flicker = occlusion / wrong channel)
      unoccl 1 - fraction of body rows that touch the left/right valid border (cut off or another person attached)
      length valid frames / full_len
      height median crop height mapped h_lo -> 0, h_hi -> 1
    """
    fg, valid = np.asarray(fg, bool), np.asarray(valid, bool)
    S = len(fg)
    area = fg.sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1)
    ok = (area >= area_range[0]) & (area <= area_range[1])
    inter = (fg[1:] & fg[:-1]).sum((1, 2))
    union = np.maximum((fg[1:] | fg[:-1]).sum((1, 2)), 1)
    iou = float(np.mean(inter / union)) if S > 1 else 0.0
    cols = valid.any(1)                                                     # (S, 32) valid columns per frame
    left = np.array([c.argmax() for c in cols])
    right = np.array([len(c) - 1 - c[::-1].argmax() for c in cols])
    rows = fg.any(2)                                                        # (S, 64) rows with body
    touch = np.array([(fg[i][:, [left[i], right[i]]].any(1) & rows[i]).sum() / max(rows[i].sum(), 1) for i in range(S)])
    parts = {"area": float(ok.mean()),
             "stable": float(np.clip((iou - 0.5) / 0.4, 0, 1)),
             "unoccl": float(1 - touch.mean()),
             "length": float(min(1.0, ok.sum() / full_len)),
             "height": float(np.clip((np.median(heights) - h_lo) / (h_hi - h_lo), 0, 1))}
    q = float(np.prod([max(v, 1e-3) for v in parts.values()]) ** (1 / len(parts)))
    return q, parts

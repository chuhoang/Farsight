"""BGR crop (any size) -> RGB [-1, 1] square tensor, same as CVLface's DFA Preprocessor (zero-pad to square,
bilinear resize) but done per crop so crops of different sizes can be batched."""
import cv2
import numpy as np


def to_square(crop, size=160):
    """Returns (CHW float32 in [-1,1] RGB, (side, pad_left, pad_top)) for mapping landmarks back to the crop."""
    h, w = crop.shape[:2]
    s = max(h, w)
    l, t = (s - w) // 2, (s - h) // 2
    sq = np.zeros((s, s, 3), np.uint8)
    sq[t:t + h, l:l + w] = crop
    sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_LINEAR)
    x = sq[:, :, ::-1].astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    return x, (s, l, t)

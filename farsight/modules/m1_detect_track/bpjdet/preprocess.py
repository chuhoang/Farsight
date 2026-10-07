"""GPU letterbox, same geometry as BPJDet utils.augmentations.letterbox(auto=True): resize long side
to imgsz, pad to a stride multiple, centred, grey 114. Returns (B,3,H,W) RGB float in [0,1]."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def letterbox(frames, imgsz, stride, device, half=False):
    x = torch.from_numpy(np.stack(frames)).to(device, non_blocking=True)        # B,H,W,3 BGR uint8
    x = x.permute(0, 3, 1, 2).flip(1).float().div_(255)                           # B,3,H,W RGB
    h, w = x.shape[2:]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    dh, dw = (-nh) % stride / 2, (-nw) % stride / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    bottom, right = int(round(dh + 0.1)), int(round(dw + 0.1))
    x = F.pad(x, (left, right, top, bottom), value=114 / 255)
    return x.half() if half else x

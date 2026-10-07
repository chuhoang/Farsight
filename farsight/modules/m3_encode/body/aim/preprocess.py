"""Same as AIM data/__init__.py transform_test: T.Resize((384,192)) (PIL bilinear) + ToTensor + ImageNet Normalize."""
import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(crops_bgr, size=(384, 192)):
    """List of BGR uint8 crops -> float tensor (N, 3, H, W)."""
    h, w = size
    out = [(np.asarray(Image.fromarray(c[:, :, ::-1]).resize((w, h), Image.BILINEAR), np.float32) / 255 - MEAN) / STD
           for c in crops_bgr]
    return torch.from_numpy(np.stack(out).transpose(0, 3, 1, 2).copy())

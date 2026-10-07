"""Same as CSCI build_vid_transforms test: ST.Scale((224,224), interpolation=3 = PIL bicubic) + ToTensor + ImageNet Normalize."""
import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(crops_bgr, size=(224, 224)):
    """List of BGR uint8 crops -> float tensor (N, 3, H, W)."""
    h, w = size
    out = [(np.asarray(Image.fromarray(np.ascontiguousarray(c[:, :, ::-1])).resize((w, h), Image.BICUBIC), np.float32) / 255
            - MEAN) / STD for c in crops_bgr]
    return torch.from_numpy(np.stack(out).transpose(0, 3, 1, 2).copy())

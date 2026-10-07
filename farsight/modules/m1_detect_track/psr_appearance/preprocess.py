"""Body crops -> ImageNet-normalised (B,3,H,W) tensor."""
import cv2
import numpy as np
import torch

from farsight.io.video import crop

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crops_to_tensor(frame, boxes, hw):
    x = [cv2.resize(crop(frame, b), (hw[1], hw[0]))[:, :, ::-1] for b in boxes]       # BGR->RGB
    x = (np.stack(x).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)

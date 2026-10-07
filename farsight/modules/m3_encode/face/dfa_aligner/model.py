"""DFA face aligner (CVLface): BGR face crop -> 5 landmarks + aligned 112x112 face for KP-RPE."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from farsight.core.weights import fetch

from .._cvlface import load_pkg
from .preprocess import to_square

HERE = Path(__file__).parent


class DFAAligner:
    def __init__(self, device="cuda", **over):
        self.cfg = {**yaml.safe_load((HERE / "config.yaml").read_text()), **over}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        net = load_pkg("aligners").get_aligner(OmegaConf.create(self.cfg))
        net.load_state_dict_from_path(str(fetch(HERE)))
        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def align(self, crops):
        """List of BGR crops -> dict of torch tensors on device:
        aligned (B,3,112,112) RGB in [-1,1]; ldmk_aligned (B,5,2) in [0,1] of the aligned image (KP-RPE input);
        score (B,) face confidence; theta (B,2,3) alignment (see warp). Plus ldmk (B,5,2) float32 numpy in crop
        pixel coords."""
        out = {k: [] for k in ("aligned", "ldmk_aligned", "score", "ldmk", "theta")}
        bs = self.cfg["batch_size"]
        for i in range(0, len(crops), bs):
            xs, geo = zip(*(to_square(c, self.cfg["input_size"]) for c in crops[i:i + bs]))
            x = torch.from_numpy(np.stack(xs)).to(self.device)
            aligned, ldmk, ldmk_al, score, theta, _ = self.net(x)
            g = np.array(geo, np.float32)  # (b, 3): side, pad_left, pad_top
            ldmk_px = ldmk.cpu().numpy() * g[:, None, :1] - g[:, None, 1:]
            for k, v in zip(out, (aligned, ldmk_al, score.view(-1), ldmk_px, theta)):
                out[k].append(v)
        return {k: (np.concatenate(v) if k == "ldmk" else torch.cat(v)) for k, v in out.items()}

    @torch.no_grad()
    def warp(self, crops, theta):
        """Align crops with given thetas (align()["theta"] of other crops of the same size, e.g. the clean version of
        degraded crops: train_m3 A3). -> (B,3,112,112) RGB in [-1,1], as align()["aligned"]."""
        x = torch.from_numpy(np.stack([to_square(c, self.cfg["input_size"])[0] for c in crops])).to(self.device)
        x = self.net.preprocessor(x)
        o = self.cfg["output_size"]
        grid = F.affine_grid(theta.to(self.device), (len(crops), 3, o, o), align_corners=True)
        return F.grid_sample(x + 1, grid, align_corners=True) - 1

    __call__ = align

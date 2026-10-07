"""KP-RPE ViT-B (CVLface, WebFace12M): face crops -> DFA -> feat (512, L2), quality (norm), inter_feat (QE input)."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from farsight.core.hooks import LayerCapture
from farsight.core.interfaces import BaseEncoder
from farsight.core.registry import build
from farsight.core.weights import fetch

from .._cvlface import load_pkg

HERE = Path(__file__).parent


def _pool(tokens, mode):
    """(B, 196, C) patch tokens (no CLS) -> (B, D). 'style' = QME FaceQE input: LayerNorm, then mean+std."""
    if mode == "mean":
        return tokens.mean(1)
    t = F.layer_norm(tokens.float(), tokens.shape[-1:], eps=1e-6)
    return torch.cat([t.mean(1), t.std(1)], -1)


class KPRPE(BaseEncoder):
    def __init__(self, device="cuda", aligner=None, **over):
        self.cfg = {**yaml.safe_load((HERE / "config.yaml").read_text()), **over}
        self.aligner = aligner or build(self.cfg["aligner"], device=device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        net = load_pkg("models").get_model(OmegaConf.create(self.cfg))
        if self.cfg.get("checkpoint"):   # fine-tuned weights (train_m3: kprpe_lr), full state_dict of this net
            net.load_state_dict(torch.load(self.cfg["checkpoint"], map_location="cpu"), strict=True)
        else:
            net.load_state_dict_from_path(str(fetch(HERE)))
        self.net = net.to(self.device).eval()
        self.blocks = [self.net.net.blocks[i] for i in self.cfg["inter_blocks"]]

    @torch.no_grad()
    def forward(self, crops):
        """BGR crops -> raw feat (N,512) pre-L2, inter_feat (N,D), DFA face score (N,); float32 numpy, one pass."""
        if len(crops) == 0:
            return np.zeros((0, self.cfg["output_dim"]), np.float32), None, np.zeros(0, np.float32)
        feats, inters, scores, bs = [], [], [], self.cfg["batch_size"]
        for i in range(0, len(crops), bs):
            a = self.aligner(crops[i:i + bs])
            with ExitStack() as st, torch.autocast(self.device.type, enabled=self.cfg["fp16"]):
                caps = [st.enter_context(LayerCapture(b, lambda o: _pool(o, self.cfg["inter_pool"])))
                        for b in self.blocks]
                feats.append(self.net(a["aligned"], a["ldmk_aligned"]).float().cpu())
            inters.append(torch.cat([c.out.float() for c in caps], -1).cpu())
            scores.append(a["score"].float().cpu())
        return torch.cat(feats).numpy(), torch.cat(inters).numpy(), torch.cat(scores).numpy()

    def embed(self, crops):
        """Per-crop L2-normalised 512-d features (M2 safety check)."""
        f = self.forward(crops)[0]
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)

    def quality(self, crops):
        """Per-crop quality = feature norm before L2 (M2 quality gate)."""
        return np.linalg.norm(self.forward(crops)[0], axis=1)

    def encode(self, crops, track=None):
        return aggregate(*self.forward(crops), self.cfg)


def aggregate(f, inter, score, cfg):
    """Per-frame raw feats (N,512), inter_feat (N,D) or None, DFA scores (N,) -> face template or None."""
    q = np.linalg.norm(f, axis=1)
    # norm alone does not reject non-faces (noise crops get face-like norms), so also gate on DFA score
    keep = (q >= cfg["min_quality"]) & (score >= cfg["min_face_score"])
    if not keep.any():
        return None
    f, q = f[keep], q[keep]
    w = np.exp((q - q.max()) / cfg["softmax_temp"])
    w /= w.sum()
    feat = (w[:, None] * (f / q[:, None])).sum(0)
    return {
        "feat": (feat / np.linalg.norm(feat)).astype(np.float32),
        "quality": float((w * q).sum()),
        "inter_feat": None if inter is None else (w[:, None] * inter[keep]).sum(0).astype(np.float32),
        "n_frames": int(keep.sum()),
        "keep": keep,
    }

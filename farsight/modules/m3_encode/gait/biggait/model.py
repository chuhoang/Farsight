"""BigGait (CVPR'24, OpenGait BigGait__Dinov2_Gaitbase), DINOv2 ViT-S/14 + GaitBase, CCPG Frame30 checkpoint."""
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from farsight.core.hooks import LayerCapture
from farsight.core.interfaces import BaseEncoder
from farsight.core.weights import fetch

from .. import _opengait
from .preprocess import clip_starts, mask_quality, preprocess, quality

HERE = Path(__file__).parent
_SILENT = types.SimpleNamespace(log_info=lambda *a, **k: None)


def get_body_valid(mask, edge=5):
    """OpenGait BigGait.get_body with the edge test normalised by the valid (non-padded) width.
    Original: channel 0 is taken as background (-> use channel 1) when it covers > w*edge pixels of the top+bottom
    `edge` rows, i.e. > 50% of them counting padded columns. A crop narrower than w/h 0.5 is padded, so a
    background channel can never pass that threshold and the background becomes the "body" (69% of MEVID crops;
    0% of CCVID/CCPG, which is why it never showed there). Identical to the original when nothing is padded."""
    ch0 = torch.round(mask[..., 0]) - mask[..., 0].detach() + mask[..., 0]
    valid = (mask.sum(-1) > 0).to(mask.dtype)          # padded pixels are all-zero in every channel
    count = ch0[:, :edge].sum((1, 2)) + ch0[:, -edge:].sum((1, 2))
    n_valid = valid[:, :edge].sum((1, 2)) + valid[:, -edge:].sum((1, 2))
    cond = count > 0.5 * n_valid
    mask[cond, :, :, 0] = mask[cond, :, :, 1]
    return mask[..., 0]


class BigGaitEncoder(BaseEncoder):
    here = HERE

    def __init__(self, pretrained=True, **overrides):
        self.cfg = {**yaml.safe_load((self.here / "config.yaml").read_text()), **overrides}
        dev = self.cfg["device"]
        self.device = torch.device(dev if dev != "cuda" or torch.cuda.is_available() else "cpu")
        self.model_cfg = yaml.safe_load((_opengait.OPENGAIT.parent / self.cfg["opengait_config"]).read_text())["model_cfg"]
        self.net = self.build()
        if pretrained:
            sd = torch.load(fetch(self.here, self.cfg["checkpoint"]), map_location="cpu", weights_only=False)["model"]
            self.load(sd)
        self.net.eval().to(self.device)
        choose = get_body_valid if self.cfg.get("mask_fix", True) else self.net.get_body
        self._masks = []

        def get_body(m):   # same choice as the model, plus a record of (body mask, valid region) for gait_quality
            valid = m.sum(-1) > 0
            fg = choose(m)
            self._masks.append((((fg > 0.5) & valid).cpu().numpy(), valid.cpu().numpy()))
            return fg
        self.net.get_body = get_body

    def build(self):
        mod = _opengait.load("BigGait")
        net = mod.BigGait__Dinov2_Gaitbase(self.model_cfg)
        net.backbone = _opengait.use_sdpa(mod.vit_small(logger=_SILENT))  # weights come from the checkpoint, not dinov2_vits14_pretrain.pth
        return net

    def load(self, sd):
        self.net.load_state_dict(sd, strict=True)

    def forward(self, x, ratios):
        """x (n, s, 3, 256, 128), ratios (n, s) -> (embeddings (n, D), inter (n, D_mid))."""
        a, b = self.cfg["inter_slice"]
        n, s = x.shape[:2]
        with LayerCapture(self.net.backbone, lambda o: o["x_norm_patchtokens_mid4"][..., a:b].float().mean(1)) as cap:
            emb = self.net(([x, ratios], None, None, None, None))["inference_feat"]["embeddings"]
        return emb.flatten(1), cap.out.view(n, s, -1).mean(1)

    @torch.no_grad()
    def embed_clips(self, crops):
        """BGR crops of one track -> per-clip (feat (C, D) L2-normed, inter (C, D_mid)) numpy float32."""
        L = self.cfg["clip_len"]
        x, r = preprocess(crops, self.cfg["input_size"])
        starts = clip_starts(len(crops), L)
        clips = [(x[i:i + L], r[i:i + L]) for i in starts]
        feats, inters, self._masks = [], [], []
        for j in range(0, len(clips), self.cfg["clip_batch"]):
            xb = torch.stack([c[0] for c in clips[j:j + self.cfg["clip_batch"]]]).to(self.device)
            rb = torch.stack([c[1] for c in clips[j:j + self.cfg["clip_batch"]]]).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.cfg["fp16"] and self.device.type == "cuda"):
                f, inter = self.forward(xb, rb)
            feats.append(F.normalize(f.float(), dim=1).cpu())
            inters.append(inter.float().cpu())
        return torch.cat(feats).numpy(), torch.cat(inters).numpy()

    def encode(self, crops, track):
        if len(crops) < self.cfg["min_frames"]:
            return None
        feats, inters = self.embed_clips(crops)
        if self._masks:
            fg = np.concatenate([f.reshape(-1, 64, 32) for f, _ in self._masks])
            valid = np.concatenate([v.reshape(-1, 64, 32) for _, v in self._masks])
            q, parts = mask_quality(fg, valid, [c.shape[0] for c in crops])
        else:   # networks without BigGait's mask step (BiggerGait): size/length heuristic
            q, parts = quality(crops, 128, 60), {}
        if q < self.cfg.get("min_quality", 0.0):      # unreliable gait (mask lost / occluded / tiny / flickering) -> missing
            return None
        feat = feats.mean(0)
        feat /= max(np.linalg.norm(feat), 1e-12)
        return {"feat": feat.astype(np.float32), "quality": q, "quality_parts": parts,
                "inter_feat": inters.mean(0).astype(np.float32), "n_frames": len(crops)}

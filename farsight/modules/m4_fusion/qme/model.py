"""QME score fusion (Zhu et al., ICCV 2025, MIT licence): learnable score normalisation (BatchNorm) +
mixture of Z score-fusion experts steered by the face QE weight; trained with QME's score triplet loss.

Ported from third_party/QME_ICCV25/model.py (LSN, MoNormQE_dev; parameter names kept so the released
lsf-*.pth checkpoints load with use_mask=False). Additions (plan 2B.2) - QME has no masking, it only
zeroes raw scores of dropped modalities during training:
  * missing modality (probe or gallery side, NaN score) -> neutral 0 after normalisation,
  * optional binary `missing` flags concatenated to each expert's MLP input (use_mask=True),
  * face QE weight forced to 0 when the probe has no face,
  * Z > 2 experts: piecewise-linear "hat" gates on the QE weight (Z=2 reduces to QME's w*E0 + (1-w)*E1),
  * face_gate t > 0: probes with QE weight < t are treated as face-missing (hard gate),
  * face_scale: the normalised face score entering each expert is multiplied by the QE weight.
  (QME's W only blends the experts and both experts read the face score, so a W of 0 does not stop a wrong face
  from steering the result: eval/qme_diag.py, MEVID perfect-W oracle 74.8 vs face-gate oracle 80.0 R1.)
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval.metrics import fnir_at_fpir
from farsight.core.interfaces import BaseFusion
from farsight.core.registry import register
from farsight.core.weights import fetch
from farsight.modules.m4_fusion.module import cosine_scores, load_cfg, stack, stack_gallery, to_result
from farsight.modules.m4_fusion.qe.model import FaceQE, Mlp, init_weights

HERE = Path(__file__).parent


class Expert(nn.Module):
    """QME LSN: BatchNorm over modalities -> per-pair MLP -> fused score."""

    def __init__(self, n_mod=3, mlp_ratio=3, use_mask=True, drop=0.1):
        super().__init__()
        self.use_mask = use_mask
        self.bn = nn.BatchNorm1d(n_mod)
        self.score_mlp = Mlp(n_mod * (2 if use_mask else 1), n_mod * mlp_ratio, 1, nn.SELU, drop)

    def forward(self, scores, miss, face_scale=None):
        """scores (B, M, G) raw (any value where miss), miss (B, M, G) bool, face_scale (B,) or None -> (B, G)."""
        fill = self.bn.running_mean.detach()[None, :, None].expand_as(scores)
        x = self.bn(torch.where(miss, fill, scores)).masked_fill(miss, 0.0).transpose(1, 2)  # (B, G, M)
        if face_scale is not None:
            x = torch.cat([x[..., :1] * face_scale[:, None, None], x[..., 1:]], -1)
        if self.use_mask:
            x = torch.cat([x, miss.transpose(1, 2).float()], -1)
        return self.score_mlp(x).mean(-1)


class QMEHead(nn.Module):
    def __init__(self, n_mod=3, n_experts=2, mlp_ratio=3, use_mask=True, drop=0.1, patch_dim=512, face_gate=0.0,
                 face_scale=False):
        super().__init__()
        self.face_gate, self.face_scale = float(face_gate), bool(face_scale)
        self.experts = nn.ModuleList(Expert(n_mod, mlp_ratio, use_mask, drop) for _ in range(n_experts))
        init_weights(self.experts)
        self.qe = FaceQE(patch_dim)          # frozen; carried so a checkpoint is self-contained (as in QME)
        self.qe.requires_grad_(False)

    def gates(self, w):
        """(B,) QE weight -> (B, Z) hat gates over expert centres linspace(1, 0, Z)."""
        Z = len(self.experts)
        if Z == 1:
            return torch.ones_like(w)[:, None]
        c = torch.linspace(1, 0, Z, device=w.device)
        return torch.relu(1 - (w[:, None] - c).abs() * (Z - 1))

    def forward(self, scores, face_w, missing=None):
        """scores (B, M, G) raw, NaN = missing pair; face_w (B,); missing (B, M) probe-level flags -> (B, G)."""
        miss = torch.isnan(scores)
        if missing is not None:
            miss = miss | missing[:, :, None]
        scores = torch.nan_to_num(scores)
        face_w = face_w * (~miss[:, 0].all(-1)).float()   # no face for this probe -> weight 0
        if self.face_gate > 0:                             # low-quality face -> handled as a missing face
            off = face_w < self.face_gate
            miss = miss.clone()
            miss[:, 0] |= off[:, None]
            face_w = face_w * (~off).float()
        g = self.gates(face_w)
        fs = face_w if self.face_scale else None
        return sum(g[:, z, None] * e(scores, miss, fs) for z, e in enumerate(self.experts))

    def qe_weight(self, style):
        with torch.no_grad():
            return torch.sigmoid(self.qe(style))


def load_head(path, **kw):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    head = QMEHead(**{**kw, **sd.get("head_cfg", {})})
    head.load_state_dict(sd.get("model_state_dict", sd))
    return head.eval(), sd.get("tau")


@register("qme")
class QMEFusion(BaseFusion):
    def __init__(self, **overrides):
        c = load_cfg(HERE, **overrides)
        self.top_k, self.device, self.default_w = c["top_k"], c["device"], c["default_face_weight"]
        kw = {k: c[k] for k in ("n_experts", "mlp_ratio", "use_mask", "patch_dim")}
        ck, tau = c.get("checkpoint"), None
        if ck:
            self.head, tau = load_head(ck if Path(str(ck)).is_file() else fetch(HERE, ck), **kw)
        else:
            self.head = QMEHead(**kw).eval()
        self.head.to(self.device)
        self.tau = c["tau"] if c.get("tau") is not None else (tau if tau is not None else -np.inf)

    def face_weight(self, probe):
        if probe.get("qe_weight") and probe["qe_weight"].get("face") is not None:
            return float(probe["qe_weight"]["face"])
        f = probe.get("face")
        if f is None:
            return 0.0
        inter = f.get("inter_feat")
        if inter is not None and np.size(inter) == self.head.qe.f_style_mlp.fc1.in_features:
            x = torch.as_tensor(np.asarray(inter, np.float32), device=self.device)
            return float(self.head.qe_weight(x[None]))
        return self.default_w

    @torch.no_grad()
    def fuse(self, S, face_w):
        """S (P, G, 3) raw cosine (NaN = missing), face_w (P,) -> (P, G) numpy."""
        s = torch.as_tensor(S, dtype=torch.float32, device=self.device).permute(0, 2, 1)
        w = torch.as_tensor(np.asarray(face_w, np.float32), device=self.device)
        out = self.head(s, w).cpu().numpy()
        return np.where(np.isnan(S).all(-1), np.nan, out)

    def set_tau(self, S, face_w, q_ids, g_ids, fpir=0.01):
        self.tau = fnir_at_fpir(self.fuse(S, face_w), q_ids, g_ids, (fpir,))[fpir][1]
        return self.tau

    def search(self, probe, gallery):
        G = stack_gallery(gallery)
        S = cosine_scores(stack([probe]), G)
        fused = self.fuse(S, [self.face_weight(probe)])[0]
        return to_result(probe["subject_or_track_id"], fused, S[0], G["ids"], self.tau, self.top_k)

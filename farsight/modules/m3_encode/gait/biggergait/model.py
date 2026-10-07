"""BiggerGait (OpenGait BiggerGait__DINOv2): all 12 DINOv2 ViT-S/14 layers -> 12 GaitBase heads, CCPG checkpoint.
Same clip/quality/inter_feat logic as biggait; only the network differs."""
import types
from pathlib import Path

import torch

from farsight.core.hooks import LayerCapture

from .. import _opengait
from ..biggait.model import BigGaitEncoder

HERE = Path(__file__).parent
_SILENT = types.SimpleNamespace(log_info=lambda *a, **k: None)


def _hidden_states(self, x, output_hidden_states=True):
    """HF Dinov2Model-style output on the original DINOv2 ViT: hidden_states = (embeddings, block_1..block_12),
    un-normalised, CLS first. The checkpoint stores the backbone in original-DINOv2 key format, so we keep that
    module and adapt its output instead of converting weights to HF."""
    x = self.prepare_tokens_with_masks(x)
    hs = [x]
    for blk in self.blocks:
        x = blk(x)
        hs.append(x)
    return types.SimpleNamespace(hidden_states=hs)


def _mid4(o):
    hs = o.hidden_states
    x = torch.cat([hs[3], hs[6], hs[9], hs[12]], -1)[:, 1:]       # blocks 2,5,8,11 like BigGait x_mid4
    return torch.nn.functional.layer_norm(x.float(), x.shape[-1:], eps=1e-6)


class BiggerGaitEncoder(BigGaitEncoder):
    here = HERE

    def build(self):
        mod = _opengait.load("BiggerGait_DINOv2")
        net = mod.BiggerGait__DINOv2(self.model_cfg)
        vit = _opengait.use_sdpa(_opengait.load("BigGait_utils.DINOv2").vit_small(logger=_SILENT))
        vit.forward = types.MethodType(_hidden_states, vit)
        net.Backbone = vit
        return net

    def load(self, sd):
        r = self.net.load_state_dict(sd, strict=False)
        # Gait_List[i] is the same module object as real_gait[j]; checkpoint only has the Gait_List names
        bad = [k for k in r.missing_keys if not k.startswith("Gait_Net.real_gait.")]
        assert not bad and not r.unexpected_keys, (bad[:5], r.unexpected_keys[:5])

    def forward(self, x, ratios):
        a, b = self.cfg["inter_slice"]
        n, s = x.shape[:2]
        with LayerCapture(self.net.Backbone, lambda o: _mid4(o)[..., a:b].mean(1)) as cap:
            emb = self.net(([x], None, None, None, None))["inference_feat"]["embeddings"]
        return emb.flatten(1), cap.out.view(n, s, -1).mean(1)

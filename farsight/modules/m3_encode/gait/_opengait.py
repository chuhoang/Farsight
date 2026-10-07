"""Import OpenGait model classes without its training stack (tensorboard, datasets, evaluators, every model).

OpenGait uses generic top-level package names (utils, modeling, data); other third_party repos do too, so
they are imported in isolation and the global names are restored afterwards.
"""
import importlib
import sys
import types
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

OPENGAIT = Path(__file__).resolve().parents[4] / "third_party" / "OpenGait" / "opengait"
_TOP = ("utils", "modeling", "data", "evaluation")
_cache = {}


class _BaseModel(nn.Module):
    """Stand-in for opengait BaseModel: just builds the network (no loaders/optimizers/ckpt logic)."""

    def __init__(self, model_cfg):
        super().__init__()
        self.build_network(model_cfg)


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def load(module):
    """load("BigGait") -> the imported opengait/modeling/models/<module>.py module."""
    if module in _cache:
        return _cache[module]
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in _TOP}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(OPENGAIT))
    try:
        mdir = OPENGAIT / "modeling" / "models"
        stubs = {
            # namespace pkgs so their __init__ (which imports every model / loss) never runs
            "modeling": _stub("modeling", __path__=[str(OPENGAIT / "modeling")]),
            "modeling.models": _stub("modeling.models", __path__=[str(mdir)]),
            "modeling.models.BigGait_utils": _stub("modeling.models.BigGait_utils",
                                                   __path__=[str(mdir / "BigGait_utils")]),
            "modeling.base_model": _stub("modeling.base_model", BaseModel=_BaseModel),
            "modeling.models.BigGait_utils.save_img": _stub("save_img", save_image=None, pca_image=None),
            "utils.msg_manager": _stub("utils.msg_manager", get_msg_mgr=None),  # avoids tensorboard
        }
        sys.modules.update(stubs)
        mod = importlib.import_module(f"modeling.models.{module}")
    finally:
        sys.path.remove(str(OPENGAIT))
        for k in [k for k in sys.modules if k.split(".")[0] in _TOP]:
            del sys.modules[k]
        sys.modules.update(saved)
    _cache[module] = mod
    return mod


def _sdpa_forward(self, x, attn_bias=None):
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


def use_sdpa(vit):
    """Same math as dino_layers Attention (xFormers missing -> naive q@k^T), via torch SDPA: ~2-3x faster."""
    for blk in vit.blocks:
        blk.attn.forward = types.MethodType(_sdpa_forward, blk.attn)
    return vit

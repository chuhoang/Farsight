"""Capture an intermediate layer output (inter_feat) during the normal forward pass."""
import torch


class LayerCapture:
    """with LayerCapture(model.blocks[8]) as cap: y = model(x); mid = cap.out"""

    def __init__(self, module, transform=None):
        self.module, self.transform, self.out, self._h = module, transform, None, None

    def _hook(self, m, i, o):
        o = o[0] if isinstance(o, (tuple, list)) else o
        self.out = self.transform(o) if self.transform else o

    def __enter__(self):
        self._h = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self._h.remove()


def pool_tokens(x, mode="cls"):
    """(B, N, D) ViT tokens -> (B, D). (B, C, H, W) maps -> GAP. 'cls' = token 0, 'mean' = patch mean."""
    if x.dim() == 4:
        return x.mean(dim=(2, 3))
    if x.dim() == 3:
        return x[:, 0] if mode == "cls" else x[:, 1:].mean(1)
    return x


def get_submodule(model, dotted):
    return model.get_submodule(dotted) if hasattr(model, "get_submodule") else eval("model." + dotted)


__all__ = ["LayerCapture", "pool_tokens", "get_submodule", "torch"]

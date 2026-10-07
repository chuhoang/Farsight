from pathlib import Path

import yaml

from farsight.core.registry import register
from farsight.core.weights import ROOT

from ..kprpe.model import KPRPE


@register("kprpe_lr")
class KPRPELongRange(KPRPE):
    """KP-RPE with the fine-tuned checkpoint of kprpe_lr/config.yaml (same config, aggregation and inter_feat)."""

    def __init__(self, device="cuda", aligner=None, **over):
        ck = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())["checkpoint"]
        super().__init__(device=device, aligner=aligner, **{"checkpoint": str(ROOT / ck), **over})

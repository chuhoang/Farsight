from farsight.core.registry import register

from .model import QualityGate

register("quality_gate")(QualityGate)

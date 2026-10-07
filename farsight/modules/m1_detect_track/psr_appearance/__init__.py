from farsight.core.registry import register

from .model import PSRAppearance

register("psr_appearance")(PSRAppearance)

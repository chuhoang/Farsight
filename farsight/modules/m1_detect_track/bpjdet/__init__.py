from farsight.core.registry import register

from .model import BPJDet

register("bpjdet")(BPJDet)

from farsight.core.registry import register

from .model import DFAAligner

register("dfa_aligner")(DFAAligner)

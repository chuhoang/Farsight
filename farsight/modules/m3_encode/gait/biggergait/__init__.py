from farsight.core import registry

from .model import BiggerGaitEncoder

registry.register("biggergait")(BiggerGaitEncoder)

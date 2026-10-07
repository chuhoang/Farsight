from farsight.core import registry

from .model import BigGaitEncoder

registry.register("biggait")(BigGaitEncoder)

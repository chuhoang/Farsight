from farsight.core import registry

from .model import AIMEncoder

registry.register("aim")(AIMEncoder)

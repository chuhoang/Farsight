from farsight.core import registry

from .model import CSCIEncoder, CSCIImageEncoder

registry.register("csci_video")(CSCIEncoder)
registry.register("csci_image")(CSCIImageEncoder)

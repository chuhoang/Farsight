from farsight.core.registry import register

from .model import ByteTrack

register("bytetrack")(ByteTrack)

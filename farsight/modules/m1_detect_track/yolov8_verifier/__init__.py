from farsight.core.registry import register

from .model import YOLOv8Verifier

register("yolov8_verifier")(YOLOv8Verifier)

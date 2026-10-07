"""Common interfaces. Pipeline/module.py only talk to these, never to third-party code."""
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

from .types import ModalityOut, SearchResult, Template, Tracklet


class BaseDetector(ABC):
    @abstractmethod
    def __call__(self, frames: Sequence[np.ndarray]) -> List[dict]:
        """BGR uint8 HxWx3 frames -> per frame {"body": (N,5) x1y1x2y2s, "face": (N,5) or NaN rows}.
        Row i of "face" is the face matched to body i (NaN row = no face)."""


class BaseTracker(ABC):
    @abstractmethod
    def update(self, frame_idx: int, frame: np.ndarray, det: dict) -> None: ...

    @abstractmethod
    def tracklets(self, video_id: str) -> List[Tracklet]: ...


class BaseRestorer(ABC):
    @abstractmethod
    def __call__(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Sequence of BGR crops -> restored crops (same length/size)."""


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, crops: List[np.ndarray], track: Tracklet) -> Optional[ModalityOut]:
        """Crops of one track (BGR) -> one aggregated template, or None if modality unusable."""


class BaseFusion(ABC):
    @abstractmethod
    def search(self, probe: Template, gallery: List[Template]) -> SearchResult: ...

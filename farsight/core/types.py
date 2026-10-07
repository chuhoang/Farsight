"""Data schemas exchanged between modules (plan section 3). Plain dicts, typed for docs."""
from typing import List, Optional, TypedDict

import numpy as np

MODALITIES = ("face", "gait", "body")


class FrameObs(TypedDict):
    frame_idx: int
    body_box: List[float]            # x1, y1, x2, y2
    body_score: float
    face_box: Optional[List[float]]
    face_score: Optional[float]


class Tracklet(TypedDict):
    video_id: str
    track_id: int
    frames: List[FrameObs]


class ModalityOut(TypedDict):
    feat: np.ndarray                 # float32[D], L2-normalised
    quality: float
    inter_feat: Optional[np.ndarray]  # float32[D_mid], QE input (fusion v2)
    n_frames: int


class Template(TypedDict):
    subject_or_track_id: str
    face: Optional[ModalityOut]
    gait: Optional[ModalityOut]
    body: Optional[ModalityOut]
    qe_weight: Optional[dict]        # {"face": w, "gait": w, "body": w}


class RankedItem(TypedDict):
    gallery_id: str
    score: float
    per_modality: List[float]        # [s_face, s_gait, s_body], NaN = missing


class SearchResult(TypedDict):
    probe_id: str
    ranked: List[RankedItem]
    is_known: bool

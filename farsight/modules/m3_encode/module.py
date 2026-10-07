"""M3 step: one track's crops -> Template (face / gait / body, any may be None)."""
from farsight.core import registry

DEFAULT = {"face": ["dfa_aligner", "kprpe"], "gait": "biggait", "body": "aim"}


class M3Encode:
    def __init__(self, cfg=None, device="cuda"):
        c = {**DEFAULT, **(cfg or {})}
        aligner, face = c["face"]
        self.face = registry.build(face, device=device, aligner=registry.build(aligner, device=device))
        self.gait = registry.build(c["gait"], device=device)
        self.body = registry.build(c["body"], device=device)

    def __call__(self, template_id, track, face_crops, body_crops):
        """face_crops: BGR crops of frames that have a face (possibly restored by M2); body_crops: one per frame."""
        return {
            "subject_or_track_id": template_id,
            "face": self.face.encode(face_crops, track) if face_crops else None,
            "gait": self.gait.encode(body_crops, track),
            "body": self.body.encode(body_crops, track),
            "qe_weight": None,  # filled by QE in fusion v2
        }

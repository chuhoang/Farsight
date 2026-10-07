"""Name -> class registry. Each model folder's __init__ registers itself; configs pick by name."""
import importlib

_REG = {}


def register(name):
    def deco(cls):
        _REG[name] = cls
        return cls
    return deco


# name -> package to import lazily (keeps heavy deps out until a model is actually used)
_PATHS = {
    "bpjdet": "farsight.modules.m1_detect_track.bpjdet",
    "yolov8_verifier": "farsight.modules.m1_detect_track.yolov8_verifier",
    "bytetrack": "farsight.modules.m1_detect_track.bytetrack",
    "psr_appearance": "farsight.modules.m1_detect_track.psr_appearance",
    "quality_gate": "farsight.modules.m2_restore.quality_gate",
    "datum": "farsight.modules.m2_restore.datum",
    "dfa_aligner": "farsight.modules.m3_encode.face.dfa_aligner",
    "kprpe": "farsight.modules.m3_encode.face.kprpe",
    "kprpe_lr": "farsight.modules.m3_encode.face.kprpe_lr",
    "biggait": "farsight.modules.m3_encode.gait.biggait",
    "biggergait": "farsight.modules.m3_encode.gait.biggergait",
    "aim": "farsight.modules.m3_encode.body.aim",
    "csci_video": "farsight.modules.m3_encode.body.csci",
    "csci_image": "farsight.modules.m3_encode.body.csci",
    "zscore": "farsight.modules.m4_fusion.zscore",
    "qe": "farsight.modules.m4_fusion.qe",
    "qme": "farsight.modules.m4_fusion.qme",
}


def get(name):
    if name not in _REG and name in _PATHS:
        importlib.import_module(_PATHS[name])
    if name not in _REG:
        raise KeyError(f"model '{name}' not registered; known: {sorted(set(_REG) | set(_PATHS))}")
    return _REG[name]


def build(name, **kw):
    return get(name)(**kw)

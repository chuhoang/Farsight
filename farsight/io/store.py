"""HDF5 template store: /<template_id>/<modality>/{feat, inter_feat} + attrs (quality, n_frames).
qe_weight stored as attrs on the template group."""
import json

import h5py
import numpy as np

from farsight.core.types import MODALITIES


def write_templates(path, templates, mode="a", keep_inter=True):
    with h5py.File(path, mode) as f:
        for t in templates:
            tid = t["subject_or_track_id"]
            if "/" in tid:  # HDF5 would silently nest groups
                raise ValueError(f"template id must not contain '/': {tid!r}")
            if tid in f:
                del f[tid]
            g = f.create_group(tid)
            if t.get("qe_weight") is not None:
                g.attrs["qe_weight"] = json.dumps(t["qe_weight"])
            for m in MODALITIES:
                mo = t.get(m)
                if mo is None:
                    continue
                mg = g.create_group(m)
                mg.create_dataset("feat", data=np.asarray(mo["feat"], np.float32))
                if keep_inter and mo.get("inter_feat") is not None:
                    mg.create_dataset("inter_feat", data=np.asarray(mo["inter_feat"], np.float32))
                mg.attrs["quality"] = float(mo["quality"])
                mg.attrs["n_frames"] = int(mo["n_frames"])


def read_templates(path):
    out = []
    with h5py.File(path, "r") as f:
        for tid, g in f.items():
            t = {"subject_or_track_id": tid, "qe_weight": None}
            if "qe_weight" in g.attrs:
                t["qe_weight"] = json.loads(g.attrs["qe_weight"])
            for m in MODALITIES:
                if m not in g:
                    t[m] = None
                    continue
                mg = g[m]
                t[m] = {
                    "feat": mg["feat"][()],
                    "inter_feat": mg["inter_feat"][()] if "inter_feat" in mg else None,
                    "quality": float(mg.attrs["quality"]),
                    "n_frames": int(mg.attrs["n_frames"]),
                }
            out.append(t)
    return out

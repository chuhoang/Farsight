"""Template stores (farsight/io/store.py HDF5) -> per-modality score matrices in QME test_feats format.

Output (matches third_party/QME_ICCV25 scoremats_<ds>.h5 + our extras):
  <m>/score_mat (P, G)          probe vs every gallery template, cosine, NaN = modality missing
  <m>/merge_score_mat (P, U)    probe vs per-identity mean template (QME "merge" gallery)
  <m>/inter_feat (P, D_mid)     probe inter_feat (NaN rows when absent) - QE input
  q_pids, g_pids, unique_g_pids int ids; pid_names (str) maps int -> identity
  q_camids=0 / g_camids=1, q_clothes_ids=0 / g_clothes_ids=1 (dummy, so QME's same-camera junk filter keeps all)
  q_ids, g_ids template ids; quality (P, 3) probe quality

    bash run.sh tools.export_scores --probe probe.h5 --gallery gallery.h5 --out scores.h5 [--sep _]
Identity of a template = subject_or_track_id.split(sep)[0] unless --labels json {template_id: identity}.
"""
import argparse
import json

import h5py
import numpy as np

from farsight.core.types import MODALITIES
from farsight.io.store import read_templates
from farsight.modules.m4_fusion.module import cosine_scores, stack


def merge_by_identity(templates, labels):
    """One template per identity: mean of the available feats per modality."""
    out = []
    for u in sorted(set(labels)):
        ts = [t for t, l in zip(templates, labels) if l == u]
        t = {"subject_or_track_id": u, "qe_weight": None}
        for m in MODALITIES:
            fs = [x[m]["feat"] for x in ts if x[m] is not None]
            t[m] = {"feat": np.mean(fs, 0), "quality": 1.0, "n_frames": 0, "inter_feat": None} if fs else None
        out.append(t)
    return out


def export(probe_path, gallery_path, out_path, label_fn=lambda tid: tid.split("_")[0]):
    probes, gallery = read_templates(probe_path), read_templates(gallery_path)
    ql = [label_fn(t["subject_or_track_id"]) for t in probes]
    gl = [label_fn(t["subject_or_track_id"]) for t in gallery]
    names = sorted(set(ql) | set(gl))
    pid = {n: i for i, n in enumerate(names)}
    P, G = stack(probes), stack(gallery)
    U = merge_by_identity(gallery, gl)
    S, SM = cosine_scores(P, G), cosine_scores(P, stack(U))
    with h5py.File(out_path, "w") as f:
        for k, m in enumerate(MODALITIES):
            f[f"{m}/score_mat"] = S[..., k]
            f[f"{m}/merge_score_mat"] = SM[..., k]
            inter = [t[m]["inter_feat"] if t[m] is not None else None for t in probes]
            dim = next((np.size(x) for x in inter if x is not None), 0)
            if dim:
                f[f"{m}/inter_feat"] = np.stack([np.ravel(x) if x is not None else np.full(dim, np.nan, np.float32)
                                                 for x in inter]).astype(np.float32)
        f["q_pids"] = np.array([pid[x] for x in ql])
        f["g_pids"] = np.array([pid[x] for x in gl])
        f["unique_g_pids"] = np.array([pid[t["subject_or_track_id"]] for t in U])
        f["q_camids"], f["g_camids"] = np.zeros(len(ql), int), np.ones(len(gl), int)
        f["q_clothes_ids"], f["g_clothes_ids"] = np.zeros(len(ql), int), np.ones(len(gl), int)
        f["pid_names"] = np.array(names, dtype=h5py.string_dtype())
        f["q_ids"] = np.array(P["ids"], dtype=h5py.string_dtype())
        f["g_ids"] = np.array(G["ids"], dtype=h5py.string_dtype())
        f["quality"] = P["quality"]
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sep", default="_")
    ap.add_argument("--labels", help="json {template_id: identity}")
    a = ap.parse_args()
    if a.labels:
        lab = json.load(open(a.labels, encoding="utf-8"))
        fn = lab.__getitem__
    else:
        fn = lambda tid: tid.split(a.sep)[0]  # noqa: E731
    print("wrote", export(a.probe, a.gallery, a.out, fn))


if __name__ == "__main__":
    main()

"""Stage 2 of QME (plan 2B.3): train the MoE score-fusion head on precomputed score matrices, QE frozen.

Input HDF5 (QME test_feats format, as written by tools/export_scores.py):
  <m>/score_mat (P, G) for m in --modalities (NaN = missing), q_pids (P,), g_pids (G,)
  face weights: --qe_weights h5 with face_weights (P,1) (QME mod_qe output), else `face_weights` in the
  scores file, else QE(face/inter_feat) with --qe checkpoint, else 0.5.
Loss = QME ScoreTripletLoss.cal_score_loss_cc (imported from third_party/QME_ICCV25/loss.py).
Missing-modality augmentation (plan 2B.2) at score level: drop face / gait of a probe with p_drop.

    bash run.sh farsight.modules.m4_fusion.qme.train --train tr.h5 --val va.h5 --out weights/m4_fusion/qme/qme_2b.pth
"""
import argparse
import copy
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch

from eval.metrics import evaluate, fnir_at_fpir
from farsight.core.weights import ROOT
from farsight.modules.m4_fusion.qe.model import load_qe_state
from farsight.modules.m4_fusion.qme.model import QMEHead

MODS = ("face", "gait", "body")


def qme_loss(margin=1.0):
    spec = importlib.util.spec_from_file_location("qme_loss", ROOT / "third_party/QME_ICCV25/loss.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ScoreTripletLoss(margin=margin, s=1.0, rank_threshold=3)


def load_scores(path, mods=MODS, qe_weights=None, qe_ckpt=None):
    """-> S (P, G, M) float32, face_w (P,), q_pids, g_pids."""
    with h5py.File(path, "r") as f:
        S = np.stack([f[f"{m}/score_mat"][()] for m in mods], -1).astype(np.float32)
        q, g = f["q_pids"][()].astype(np.int64), f["g_pids"][()].astype(np.int64)
        if qe_weights:
            with h5py.File(qe_weights, "r") as fw:
                w = fw["face_weights"][()].reshape(-1)
        elif "face_weights" in f:
            w = f["face_weights"][()].reshape(-1)
        elif qe_ckpt and "face/inter_feat" in f:
            from farsight.modules.m4_fusion.qe.model import QualityEstimator
            qe = QualityEstimator(checkpoint=qe_ckpt)
            w = qe.batch(f["face/inter_feat"][()])
        else:
            w = np.full(len(q), 0.5, np.float32)
    return S, np.nan_to_num(w.astype(np.float32)), q, g


def overall(S_fused, q, g):
    """QME model-selection score (GR_overall without mAP): rank1 + TAR@1%FAR - FNIR@1%FPIR."""
    r = evaluate(S_fused, q, g)
    return r["rank1"] + r["tar@0.01far"] - r["fnir@0.01fpir"], r


def fuse_np(head, S, w, device):
    with torch.no_grad():
        out = head.eval()(torch.tensor(S, device=device).permute(0, 2, 1), torch.tensor(w, device=device)).cpu().numpy()
    return np.where(np.isnan(S).all(-1), np.nan, out)


def train(S, w, q, g, val=None, steps=600, bs=8, lr=1e-4, wd=0.01, margin=1.0, p_drop=(0.2, 0.15, 0.0),
          n_experts=2, mlp_ratio=3, use_mask=True, drop=0.1, eval_every=100, device="cpu", seed=0, qe_state=None,
          body_noise=0.0, body_mod=2, face_gate=0.0, face_scale=False):
    """S, w, q, g: one training set, or lists of sets with their own galleries (e.g. CCVID + MEVID); each step
    draws its batch from one set (prob. ~ #probes). val: (S, w, q, g) or a list; model selection = mean overall.
    body_noise = k_max > 0: body scores (modality body_mod) get Gaussian noise of std k * sd(body scores of that set),
    k ~ U(0, k_max) per probe, in training batches and (fixed draw) on the val sets used for selection. Counters a
    body backbone that saw the training identities: its train-ID scores separate genuine / impostor far better
    than on unseen test IDs, and QME trained on them learns that noise from other modalities is harmless."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sets = list(zip(S, w, q, g)) if isinstance(S, (list, tuple)) else [(S, w, q, g)]
    vals = [] if val is None else list(val) if isinstance(val, list) else [val]
    head = QMEHead(sets[0][0].shape[-1], n_experts, mlp_ratio, use_mask, drop, face_gate=face_gate,
                   face_scale=face_scale).to(device)
    if qe_state:
        load_qe_state(head.qe, qe_state)
    crit = qme_loss(margin)
    opt = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=300, T_mult=2)
    T = [(torch.tensor(s_, device=device).permute(0, 2, 1), torch.tensor(w_, device=device),
          torch.tensor(q_, device=device), torch.tensor(g_, device=device)) for s_, w_, q_, g_ in sets]
    sd_body = [float(np.nanstd(s_[..., body_mod])) for s_, *_ in sets]
    if body_noise > 0:   # fixed noisy copy of each val set, so selection also happens in the weaker-body regime
        vr = np.random.default_rng(10_000 + seed)
        vals = [(v[0] + np.zeros_like(v[0]), *v[1:]) for v in vals]
        for v in vals:
            k = vr.uniform(0, body_noise, (len(v[0]), 1))
            v[0][..., body_mod] += (k * np.nanstd(v[0][..., body_mod]) * vr.standard_normal(v[0].shape[:2])).astype(v[0].dtype)
    p_set = np.array([len(x[2]) for x in sets], float)
    p_set /= p_set.sum()
    best = (-np.inf, None, None)
    for step in range(1, steps + 1):
        head.train()
        j = rng.choice(len(T), p=p_set)
        St, wt, qt, gt = T[j]
        i = torch.as_tensor(rng.choice(len(qt), bs, replace=False), device=device)
        missing = torch.as_tensor(rng.random((bs, St.shape[1])) < np.asarray(p_drop), device=device)
        missing[missing.all(1), -1] = False                       # keep at least one modality
        x = St[i]
        if body_noise > 0:
            k = torch.as_tensor(rng.uniform(0, body_noise, (bs, 1)), device=device, dtype=x.dtype)
            x = x.clone()
            x[:, body_mod] += k * sd_body[j] * torch.randn_like(x[:, body_mod])
        fused = head(x, wt[i], missing)
        loss = crit.cal_score_loss_cc(fused, qt[i], gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if vals and (step % eval_every == 0 or step == steps):
            rs = [overall(fuse_np(head, v[0], v[1], device), v[2], v[3]) for v in vals]
            o = float(np.mean([x[0] for x in rs]))
            print(f"step {step} loss {loss.item():.4f} val overall {o:.4f} rank1 "
                  + " ".join(f"{x[1]['rank1']:.3f}" for x in rs), flush=True)
            if o > best[0]:
                best = (o, copy.deepcopy(head.state_dict()), {f"val{k}_{m}": v for k, x in enumerate(rs)
                                                              for m, v in x[1].items()})
    if best[1] is not None:
        head.load_state_dict(best[1])
    return head.eval(), best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qe", help="QE checkpoint (frozen); also used to compute face weights from face/inter_feat")
    ap.add_argument("--qe_weights_train")
    ap.add_argument("--qe_weights_val")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_experts", type=int, default=2)
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--fpir", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    S, w, q, g = load_scores(a.train, qe_weights=a.qe_weights_train, qe_ckpt=a.qe)
    val = load_scores(a.val, qe_weights=a.qe_weights_val, qe_ckpt=a.qe) if a.val else None
    head, r = train(S, w, q, g, val, steps=a.steps, bs=a.bs, lr=a.lr, n_experts=a.n_experts,
                    use_mask=not a.no_mask, device=a.device, qe_state=a.qe)
    tau = None
    if val is not None:
        tau = fnir_at_fpir(fuse_np(head, *val[:2], a.device), val[2], val[3], (a.fpir,))[a.fpir][1]
        print("best val", r, "tau", tau)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": head.cpu().state_dict(), "tau": tau}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()

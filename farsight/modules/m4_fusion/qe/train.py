"""Route 2B: train the face QE on frozen-encoder inter_feat with QME pseudo-quality labels.
label = relu((T - rank) / (T - 1)), rank = 1-based rank of the true identity (QME loss.py
ScoreTripletLoss.cal_rank_loss_cc, --rank_threshold T); MSE on sigmoid(QE). The encoder is frozen, so
ranks are static and computed once from the score matrix. With cameras given, the rank follows the CAL rule of
test: gallery entries of the probe's identity AND camera are ignored (a face matching its own same-camera
near-duplicate says nothing about recognising the person elsewhere).

    bash run.sh farsight.modules.m4_fusion.qe.train --scores train_scores.h5 --out weights/m4_fusion/qe/qe_2b.pth
(scores file = tools/export_scores.py output: face/score_mat, face/inter_feat, q_pids, g_pids)"""
import argparse

import h5py
import numpy as np
import torch

from eval.metrics import first_match_rank
from farsight.modules.m4_fusion.qe.model import FaceQE


def cal_rank(score_mat, q_pids, g_pids, q_cams, g_cams):
    """1-based rank of the best genuine gallery entry, same-identity same-camera entries ignored (CAL rule);
    inf when no genuine entry from another camera exists or its score is missing."""
    S = np.nan_to_num(np.asarray(score_mat, np.float64), nan=-np.inf)
    same = np.asarray(q_pids)[:, None] == np.asarray(g_pids)[None]
    junk = same & (np.asarray(q_cams)[:, None] == np.asarray(g_cams)[None])
    best = np.where(same & ~junk, S, -np.inf).max(1)
    r = 1 + (np.where(~same, S, -np.inf) >= best[:, None]).sum(1)
    return np.where(np.isfinite(best), r, np.inf)


def pseudo_labels(score_mat, q_pids, g_pids, rank_threshold=3, q_cams=None, g_cams=None):
    r = first_match_rank(score_mat, q_pids, g_pids) if q_cams is None else cal_rank(score_mat, q_pids, g_pids, q_cams, g_cams)
    return np.maximum((rank_threshold - r) / (rank_threshold - 1), 0.0).astype(np.float32)  # rank inf -> 0


def labelled(inter_feat, score_mat, q_pids, g_pids, rank_threshold=3, q_cams=None, g_cams=None):
    """Probes that have a face -> (X, pseudo-quality y). Labels need the set's own gallery, so multi-dataset
    training labels each set separately and concatenates. Cameras given -> CAL-rule labels."""
    ok = ~np.isnan(inter_feat).any(1) & ~np.isnan(score_mat).all(1)
    qc = None if q_cams is None else np.asarray(q_cams)[ok]
    return inter_feat[ok], pseudo_labels(score_mat[ok], np.asarray(q_pids)[ok], g_pids, rank_threshold, qc, g_cams)


def train(inter_feat, score_mat, q_pids, g_pids, rank_threshold=3, steps=6000, bs=64, lr=1e-4, wd=0.01,
          device="cpu", seed=0, log_every=500, extra=(), q_cams=None, g_cams=None):
    """extra: more (X, y) pairs from labelled() on other datasets. q_cams/g_cams -> CAL-rule labels."""
    torch.manual_seed(seed)
    parts = [labelled(inter_feat, score_mat, q_pids, g_pids, rank_threshold, q_cams, g_cams), *extra]
    X = torch.tensor(np.concatenate([x for x, _ in parts]), device=device)
    y = torch.tensor(np.concatenate([y for _, y in parts]), device=device)
    net = FaceQE(X.shape[1] // 4).to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for step in range(steps):
        i = torch.randint(0, len(X), (min(bs, len(X)),), device=device)
        loss = torch.nn.functional.mse_loss(torch.sigmoid(net(X[i])), y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"step {step} mse {loss.item():.4f}")
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank_threshold", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    with h5py.File(a.scores, "r") as f:
        args = (f["face/inter_feat"][()], f["face/score_mat"][()], f["q_pids"][()], f["g_pids"][()])
    net = train(*args, rank_threshold=a.rank_threshold, steps=a.steps, bs=a.bs, lr=a.lr, device=a.device)
    torch.save({"model_state_dict": net.cpu().state_dict()}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()

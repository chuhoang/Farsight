"""train_m3 C1-C3: fine-tune KP-RPE ViT-B (init: CVLface WebFace12M checkpoint) for long-range faces.

Data = face crop caches (eval.face_cache, DFA-aligned 112x112 + landmarks): config `sources`, each with its cache,
identity list (split key: test / val identities never enter training), sampling weight and probability of the B2
degradation (farsight/.../kprpe_lr/degrade.py; only for sources whose crops are the cleanest of their kind).
Batch = source by weight -> identity uniform -> crop uniform (DFA score >= min_score), random h-flip (landmarks
mirrored), C3 landmark jitter of the student input.
Loss (C1 + C2):  AdaFace(new head, init = class means of teacher features)
               + lambda_cons * (1 - cos(student(degraded), teacher(clean)))   on degraded samples
               + lambda_kd   * (1 - cos(student(clean),    teacher(clean)))   on clean samples
teacher = the frozen original checkpoint. No WebFace data available here, so lambda_kd is the anti-forgetting term
(train_m3 C5 "WebFace 50%" replaced by distillation to the original model; also keeps gallery compatibility).
Selection (C4): every eval_every steps, eval.face_eval.val_sets (CCVID val, degraded CCVID val, MEVID val: held-out
train identities); best mean general R1 is saved.

    bash run.sh farsight.modules.m3_encode.face.kprpe_lr.train --run r2b [--sources ccvid mevid] [--steps N] [k=v ...]
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from eval.face_cache import read
from farsight.core.weights import ROOT

from .._cvlface import load_pkg
from .degrade import degrade

HERE = Path(__file__).parent
FLIP = [1, 0, 2, 4, 3]   # 5-point landmarks: L/R eye, nose, L/R mouth


class Pool:
    """All training crops of all sources in memory: img uint8, ldmk, ipd, global label, source index."""

    def __init__(self, sources, min_score):
        self.names, imgs, ldmks, ipds, labels, src, self.ids = list(sources), [], [], [], [], [], []
        for k, (name, s) in enumerate(sources.items()):
            keep = None
            if s.get("ids"):
                f, key = s["ids"].rsplit(":", 1)
                keep = set(json.loads((ROOT / f).read_text())[key])
            C = read(s["cache"])
            pids = sorted({c["pid"] for c in C.values() if keep is None or c["pid"] in keep})
            base = len(self.ids)
            self.ids += [f"{name}:{p}" for p in pids]
            lab = {p: base + i for i, p in enumerate(pids)}
            for c in C.values():
                if c["pid"] not in lab:
                    continue
                ok = c["score"] >= min_score
                if ok.any():
                    imgs.append(c["img"][ok]), ldmks.append(c["ldmk"][ok]), ipds.append(c["ipd"][ok])
                    labels.append(np.full(ok.sum(), lab[c["pid"]])), src.append(np.full(ok.sum(), k))
        self.img, self.ldmk, self.ipd = np.concatenate(imgs), np.concatenate(ldmks).astype(np.float32), np.concatenate(ipds)
        self.label, self.src = np.concatenate(labels), np.concatenate(src)
        self.by_label = {}
        for i, lab in enumerate(self.label):
            self.by_label.setdefault(int(lab), []).append(i)
        self.labels_of = [sorted({int(x) for x in self.label[self.src == k]}) for k in range(len(self.names))]

    def plan(self, n, weights, rng):
        """n sample indices: source by weight, identity uniform within it, crop uniform within the identity."""
        w = np.array([weights[s] if self.labels_of[k] else 0 for k, s in enumerate(self.names)], float)
        srcs = rng.choice(len(w), n, p=w / w.sum())
        out = np.empty(n, int)
        for i, k in enumerate(srcs):
            lab = self.labels_of[k][rng.integers(len(self.labels_of[k]))]
            idx = self.by_label[lab]
            out[i] = idx[rng.integers(len(idx))]
        return out


class Samples(torch.utils.data.Dataset):
    """Item i of the planned index list -> clean / student images (CHW float [-1,1]), landmarks, label, degraded flag.
    Deterministic per (seed, i) so workers and reruns agree."""

    def __init__(self, pool, idx, p_degrade, cfg, seed, offset=0):
        self.pool, self.idx, self.p, self.cfg, self.seed, self.off = pool, idx[offset:], p_degrade, cfg, seed, offset

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        P, rng = self.pool, np.random.default_rng([self.seed, i + self.off])
        clean, ldmk = P.img[j], P.ldmk[j].copy()
        deg, is_deg = clean, rng.random() < self.p[P.src[j]]
        if is_deg:
            deg = degrade(clean, P.ipd[j], rng, self.cfg.get("degrade"))[0]
        if rng.random() < 0.5:
            clean, deg = clean[:, ::-1], deg[:, ::-1]
            ldmk = ldmk[FLIP] * [-1, 1] + [1, 0]
        ldmk_s = ldmk + rng.normal(0, self.cfg["ldmk_noise"] / 112, ldmk.shape) if self.cfg["ldmk_noise"] else ldmk
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 127.5 - 1  # noqa: E731
        return t(clean), t(deg), torch.tensor(ldmk, dtype=torch.float32), torch.tensor(ldmk_s, dtype=torch.float32), \
            int(P.label[j]), bool(is_deg)


def build_net(device):
    from farsight.modules.m3_encode.face.kprpe.model import KPRPE
    k = KPRPE.__new__(KPRPE)
    KPRPE.__init__(k, device=device, aligner=object())
    return k.net, k.cfg


def checkpoint_blocks(net):
    """Activation checkpointing per ViT block while training (24 blocks: batch 32 needs 7.2 GB without it).
    Non-reentrant: CVLface's own `using_checkpoint` (reentrant) breaks on the keypoint context shared by blocks."""
    from torch.utils.checkpoint import checkpoint
    for b in net.net.blocks:
        def fwd(x, extra_ctx=None, f=b.forward):
            if torch.is_grad_enabled():
                return checkpoint(f, x, extra_ctx=extra_ctx, use_reentrant=False)
            return f(x, extra_ctx=extra_ctx)
        b.forward = fwd


@torch.no_grad()
def class_means(net, pool, dev, bs=32):
    W = torch.zeros(len(pool.ids), 512)
    for i in range(0, len(pool.img), bs):
        x = torch.from_numpy(pool.img[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        with torch.autocast(dev.type, torch.bfloat16):
            f = F.normalize(net(x, torch.from_numpy(pool.ldmk[i:i + bs]).to(dev)).float(), dim=1)
        W.index_add_(0, torch.from_numpy(pool.label[i:i + bs]), f.cpu())
    return F.normalize(W, dim=1)


def evaluate(net, cfg, m):
    from eval.face_eval import val_sets
    net.eval()
    r = val_sets(net, cfg, m)
    net.train()
    return r, float(np.mean([v["GR_R1"] for v in r.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="*", help="subset of config sources (default: all)")
    ap.add_argument("set", nargs="*", help="config overrides key=value (yaml values), e.g. lambda_cons=2")
    a = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())["train"]
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["steps"] = a.steps or cfg["steps"]
    if a.sources:
        cfg["sources"] = {k: cfg["sources"][k] for k in a.sources}
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    out = ROOT / "weights/m3_encode/face/kprpe_lr" / f"{a.run}_s{a.seed}.pt"
    log_p = ROOT / "eval/results/kprpe_lr" / f"{a.run}_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True), log_p.parent.mkdir(parents=True, exist_ok=True)

    pool = Pool(cfg["sources"], cfg["min_score"])
    print(f"pool: {len(pool.img)} crops, {len(pool.ids)} identities "
          + str({n: int((pool.src == k).sum()) for k, n in enumerate(pool.names)}), flush=True)
    rng = np.random.default_rng(a.seed)
    weights = {n: s["weight"] for n, s in cfg["sources"].items()}
    plan = pool.plan(cfg["steps"] * cfg["batch"], weights, rng)

    net, ncfg = build_net(dev)
    teacher = copy.deepcopy(net).eval().requires_grad_(False)
    net.train().requires_grad_(True)
    if cfg["grad_ckpt"]:
        checkpoint_blocks(net)
    W = torch.nn.Parameter(class_means(teacher, pool, dev).to(dev))
    adaface = load_pkg("losses").AdaFaceLoss(64, m=cfg["m"], h=cfg["h"], t_alpha=cfg["t_alpha"])
    bm, bstd = torch.tensor(20.0, device=dev), torch.tensor(100.0, device=dev)
    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "lr": cfg["lr"], "weight_decay": cfg["wd"]},
                             {"params": no_decay, "lr": cfg["lr"], "weight_decay": 0.0},
                             {"params": [W], "lr": cfg["head_lr"], "weight_decay": 0.0}])
    warm = cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg["steps"]))))

    from eval.prcc import aim_metrics
    m = aim_metrics()
    last = out.with_suffix(".last.pt")   # resume state (WSL gets killed with the session), removed when the run ends
    if last.exists():
        st = torch.load(last, map_location=dev)
        net.load_state_dict(st["net"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        W.data.copy_(st["W"])
        bm, bstd, log, best, start = st["bm"], st["bstd"], st["log"], st["best"], st["step"]
        print(f"resumed at step {start}", flush=True)
    else:
        log = {"config": cfg, "seed": a.seed, "pool": {n: int((pool.src == k).sum()) for k, n in enumerate(pool.names)},
               "n_ids": len(pool.ids), "val": [], "train": []}
        val, score = evaluate(net, ncfg, m)
        log["val"].append({"step": 0, "score": score, **val})
        best, start = score, 0
        print(f"step 0 val {score:.2f} {json.dumps({k: v['GR_R1'] for k, v in val.items()})}", flush=True)
    data = Samples(pool, plan, [cfg["sources"][n].get("degrade", 0.0) for n in pool.names], cfg, a.seed,
                   offset=start * cfg["batch"])
    loader = torch.utils.data.DataLoader(data, cfg["batch"], num_workers=cfg["workers"], drop_last=True,
                                         persistent_workers=cfg["workers"] > 0)
    t0, acc = time.time(), []
    for step, (clean, deg, ldmk, ldmk_s, y, is_deg) in enumerate(loader, start + 1):
        clean, deg, ldmk, ldmk_s, y, is_deg = (v.to(dev, non_blocking=True) for v in (clean, deg, ldmk, ldmk_s, y, is_deg))
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            ft = teacher(clean, ldmk).float()
        with torch.autocast("cuda", torch.bfloat16):
            fs = net(torch.where(is_deg[:, None, None, None], deg, clean), ldmk_s).float()
        norms = fs.norm(dim=1, keepdim=True)
        cos = (fs / norms) @ F.normalize(W, dim=1).T
        logits, bm, bstd = adaface(cos.clamp(-1 + 1e-5, 1 - 1e-5), y[:, None], norms, bm, bstd)
        l_cls = F.cross_entropy(logits, y)
        d = 1 - F.cosine_similarity(fs, ft, dim=1)
        l_cons = d[is_deg].mean() if is_deg.any() else d.sum() * 0
        l_kd = d[~is_deg].mean() if (~is_deg).any() else d.sum() * 0
        loss = l_cls + cfg["lambda_cons"] * l_cons + cfg["lambda_kd"] * l_kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [W], cfg["max_grad_norm"])
        opt.step(), sched.step()
        acc.append([loss.item(), l_cls.item(), l_cons.item(), l_kd.item()])
        if step % cfg["log_every"] == 0:
            mu = np.mean(acc, 0).round(4).tolist()
            log["train"].append({"step": step, "loss/cls/cons/kd": mu})
            el = time.time() - t0
            per = el / (step - start)
            print(f"step {step} loss/cls/cons/kd {mu} {per:.2f}s/step eta {per * (cfg['steps'] - step) / 60:.0f}min", flush=True)
            acc = []
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            val, score = evaluate(net, ncfg, m)
            log["val"].append({"step": step, "score": score, **val})
            print(f"step {step} val {score:.2f} {json.dumps({k: v['GR_R1'] for k, v in val.items()})}", flush=True)
            if score > best:
                best = score
                torch.save(net.state_dict(), out)
                log["best"] = {"step": step, "score": score}
            log_p.write_text(json.dumps(log, indent=1))
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "W": W.data,
                        "bm": bm, "bstd": bstd, "log": log, "best": best, "step": step}, last)
    log_p.write_text(json.dumps(log, indent=1))
    last.unlink(missing_ok=True)
    print(f"best {log.get('best')} -> {out if 'best' in log else 'original checkpoint (no improvement on val)'}")


if __name__ == "__main__":
    main()

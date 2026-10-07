"""plan_csci A2/A3/B6: CSCI-V MEVID checkpoint on a MEVID test subset, author's pipeline vs the FarSight wrapper.

Subset: query IDs drawn at random (seed 0) until >= 100 query tracklets, + every gallery tracklet of those IDs
(splits/mevid_csci_subset.json). Smaller gallery than the full test -> compare levels, not digits (79.7 / 56.7).
Author path = the repo's own pieces: MEVID._recombination_for_testset (8-frame clips, stride 4), video_loader (PIL),
build_vid_transforms test transform, clip[::2], model.load_param, fp16 autocast, mean of clips + L2,
processor.eval_mevid.evaluate (Overall: same-pid same-cam gallery removed).
Wrapper path = registry csci_video on cv2 crops: 8-clip budget on the whole subset; all clips (max_clips 0) on
N_CHECK tracklets only (feature check vs author; a full all-clip pass is ~1.5 h on an RTX 5050, same as the author's).

    bash run.sh eval.csci_a2
"""
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from eval.extract import mevid_tracklets
from farsight.core.weights import fetch
from farsight.modules.m3_encode.body.csci import model as cm

FEATS = Path.home() / "datasets/feats"
SUBSET = Path("splits/mevid_csci_subset.json")
OUT = Path("eval/results/csci_a2.json")
N_CHECK = 40   # tracklets for the wrapper-vs-author feature check (spread over the subset)


def subset(tq, tg, n_query=100, seed=0):
    pids = sorted({v["pid"] for v in tq.values()})
    ids, nq = [], 0
    for p in np.random.default_rng(seed).permutation(pids):
        ids.append(str(p))
        nq += sum(v["pid"] == p for v in tq.values())
        if nq >= n_query:
            break
    ids = sorted(ids)
    return {"seed": seed, "ids": ids, "query": sorted(t for t, v in tq.items() if v["pid"] in ids),
            "gallery": sorted(t for t, v in tg.items() if v["pid"] in ids)}


def author_modules():
    """Import the repo's data/eval code the same way the wrapper imports its model code."""
    own = lambda k: k.split(".")[0] in cm._PKGS + ("data", "processor", "mmcv", "mmengine")  # noqa: E731
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if own(k)}
    for p in ("model", "loss", "tools", "config", "data", "processor"):
        sys.modules[p] = types.ModuleType(p)
        sys.modules[p].__path__ = [str(cm.REPO / p)]
    sys.modules["torchinfo"] = types.SimpleNamespace(summary=None)
    for m in ("mmcv", "mmcv.fileio", "mmcv.runner", "mmcv.runner.utils", "mmengine", "mmengine.fileio", "mmengine.runner"):
        sys.modules[m] = types.ModuleType(m)       # dataset_loader imports FileClient / set_random_seed, unused at test
        sys.modules[m].__getattr__ = lambda name: None
    sys.path.insert(0, str(cm.REPO))
    try:
        import data.spatial_transforms as ST
        from config.defaults import _C
        from data.datasets.mevid import MEVID
        from data.dataset_loader import get_default_video_loader
        from model.ez_eva_custom import ez_eva02_vid_hybrid_extra
        from processor.eval_mevid import evaluate
    finally:
        sys.path.remove(str(cm.REPO))
        for k in [k for k in sys.modules if own(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
    tf = ST.Compose([ST.Scale((224, 224), interpolation=3), ST.ToTensor(),       # data.build_vid_transforms (test)
                     ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return types.SimpleNamespace(cfg=_C.clone(), MEVID=MEVID, loader=get_default_video_loader(), tf=tf,
                                 build=ez_eva02_vid_hybrid_extra, evaluate=evaluate)


@torch.no_grad()
def author_feats(A, tids, tr, batch=16):
    cfg = A.cfg
    cfg.MODEL.EXTRA_DIM, cfg.TRAIN.TEACH1_NUMCLASSES = 1024, 104     # modify_params: colour profile 17 -> 1 * 32 * 32
    net = A.build(config=cfg, pretrained=False, num_classes=104, cloth=1, cloth_xishu=cfg.MODEL.CLOTH_XISHU,
                  spatial_avg=None, tim_dim=4, e2e_train=True, joint=None, temporal_avg=None)
    net.load_param(str(fetch(cm.HERE, "csci_v_mevid.pth")))
    net.eval().cuda()
    data = [(tr[t]["frames"], 0, 0, 0, 0) for t in tids]
    clips, v2c = A.MEVID._recombination_for_testset(None, data, seq_len=8, stride=4)
    out = []
    for i, (a, b) in enumerate(v2c):
        f = []
        for j in range(a, b, batch):
            x = []
            for paths, *_ in clips[j:min(b, j + batch)]:
                c = A.loader(paths)[::2]                                     # VID dataset, test: clip[::2]
                A.tf.randomize_parameters()
                x.append(torch.stack([A.tf(im) for im in c], 0).permute(1, 0, 2, 3))
            with torch.autocast("cuda", dtype=torch.float16):
                f.append(net(torch.stack(x).cuda(), None).float())
        out.append(torch.nn.functional.normalize(torch.cat(f).mean(0), dim=0).cpu().numpy())
        if i % 100 == 0:
            print(f"  author {i}/{len(tids)}", flush=True)
    del net
    torch.cuda.empty_cache()
    return np.stack(out)


def wrapper_feats(tids, tr, max_clips):
    enc = cm.CSCIEncoder(max_clips=max_clips)
    torch.cuda.reset_peak_memory_stats()
    out, t_enc, n_fr = [], 0.0, 0
    for i, t in enumerate(tids):
        crops = [cv2.imread(p) for p in tr[t]["frames"]]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out.append(enc.encode(crops, None)["feat"])
        torch.cuda.synchronize(); t_enc += time.perf_counter() - t0
        n_fr += len(enc.frames(len(crops)))
        if i % 100 == 0:
            print(f"  wrapper/{max_clips} {i}/{len(tids)}", flush=True)
    res = {"s_per_track": t_enc / len(tids), "frames_per_s": n_fr / t_enc,
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "gpu": torch.cuda.get_device_name()}
    del enc
    torch.cuda.empty_cache()
    return np.stack(out), res


def cached(name, fn):
    p = FEATS / f"csci_a2_{name}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return z["f"], z["res"].item()
    f, res = fn()
    np.savez(p, f=f, res=np.array(res, dtype=object))
    return f, res


def main():
    tq, tg = mevid_tracklets("query"), mevid_tracklets("gallery")
    if not SUBSET.exists():
        SUBSET.write_text(json.dumps(subset(tq, tg), indent=1))
    s = json.loads(SUBSET.read_text())
    tr = {**tq, **tg}
    tids = s["query"] + s["gallery"]
    nq = len(s["query"])
    print(f"subset: {len(s['ids'])} IDs, {nq} query, {len(s['gallery'])} gallery, "
          f"{sum(len(tr[t]['frames']) for t in tids)} frames", flush=True)
    A = author_modules()
    pid = np.array([int(tr[t]["pid"]) for t in tids])
    cam = np.array([tr[t]["camid"] for t in tids])

    def metrics(F):
        cmc, mAP = A.evaluate(-(F[:nq] @ F[nq:].T), pid[:nq], pid[nq:], cam[:nq], cam[nq:])
        return {"R1": float(cmc[0]), "R5": float(cmc[4]), "R10": float(cmc[9]), "mAP": float(mAP)}

    res = {"subset": {k: len(v) if isinstance(v, list) else v for k, v in s.items()}}
    Fa, _ = cached("author", lambda: (author_feats(A, tids, tr), {}))
    res["author"] = metrics(Fa)
    print("author ", res["author"], flush=True)
    chk = np.linspace(0, len(tids) - 1, N_CHECK).round().astype(int)
    Fw, r = cached("wrapper_all", lambda: wrapper_feats([tids[i] for i in chk], tr, 0))
    cos = (Fa[chk] * Fw).sum(1)
    res["wrapper_all_clips_check"] = {"n": N_CHECK, "cos_min": float(cos.min()), "cos_mean": float(cos.mean()), **r}
    print("wrapper all clips vs author", res["wrapper_all_clips_check"], flush=True)
    Fw, r = cached("wrapper8", lambda: wrapper_feats(tids, tr, 8))
    cos = (Fa * Fw).sum(1)
    res["wrapper_8clips"] = {**metrics(Fw), **r, "cos_to_author_min": float(cos.min()), "cos_to_author_mean": float(cos.mean())}
    print("wrapper 8 clips", res["wrapper_8clips"], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

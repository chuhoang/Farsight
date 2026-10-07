"""plan_csci D1: pick the CSCI video checkpoint for the single shared config, on validation only.

Body-only metrics (CAL rule, GR/CC R1 + mAP) for csci_v_mevid.pth (D1b) and csci_v_ccvid.pth (D1a), plus the
current store body (AIM) as reference, on:
  ccvid_val = CCVID official query/gallery restricted to splits/ccvid_qme_halftest.json val IDs
  mevid_val = the 20 held-out MEVID train IDs (splits/mevid_qme.json val), probe/gallery by eval.main_protocol.train_pairs
Features cached in ~/datasets/feats/csci_d1_<ckpt>_<set>.npz.

    bash run.sh eval.csci_d1
"""
import json
from pathlib import Path

import numpy as np

from eval.extract import LazyFrames, ccvid_tracklets, mevid_tracklets
from eval.main_protocol import Set, cal_metrics, load, train_pairs
from eval.prcc import aim_metrics
from farsight.modules.m3_encode.body.csci.model import CSCIEncoder

F = Path.home() / "datasets/feats"
CKPTS = ("csci_v_mevid.pth",)   # D1b chosen by the user; add "csci_v_ccvid.pth" to compare D1a


def val_sets():
    cv = set(json.loads(Path("splits/ccvid_qme_halftest.json").read_text())["val"])
    mv = set(json.loads(Path("splits/mevid_qme.json").read_text())["val"])
    return {"ccvid_val": ((*load(F, "ccvid_query", cv), *load(F, "ccvid_gallery", cv)),
                          {**ccvid_tracklets("query"), **ccvid_tracklets("gallery")}),
            "mevid_val": (train_pairs(*load(F, "mevid_train", mv)), mevid_tracklets("train"))}


def body_only(m, P, Pm, G, Gm):
    r = cal_metrics(m, Set(P, Pm, G, Gm).S["full"][..., 2], Pm, Gm)
    return {k: round(100 * float(v), 2) for k, v in r.items()}


def main():
    m, sets, res = aim_metrics(), val_sets(), {}
    for name, ((P, Pm, G, Gm), tr) in sets.items():
        res[name] = {"n_probe": len(P), "n_gallery": len(G), "store body (AIM)": body_only(m, P, Pm, G, Gm)}
    for ck in CKPTS:
        enc = None
        for name, ((P, Pm, G, Gm), tr) in sets.items():
            cache = F / f"csci_d1_{Path(ck).stem}_{name}.npz"
            tids = [t["subject_or_track_id"] for t in P + G]
            if cache.exists():
                feats = dict(np.load(cache))
            else:
                enc = enc or CSCIEncoder(checkpoint=ck)
                feats = {}
                for k, t in enumerate(tids):
                    feats[t] = enc.encode(LazyFrames(tr[t]["frames"], t, None), None)["feat"]
                    if k % 200 == 0:
                        print(f"  {ck} {name} {k}/{len(tids)}", flush=True)
                np.savez(cache, **feats)
            sw = lambda ts: [{**t, "body": {**(t["body"] or {"quality": 1.0}), "feat": feats[t["subject_or_track_id"]]}} for t in ts]  # noqa: E731
            res[name][ck] = body_only(m, sw(P), Pm, sw(G), Gm)
            print(name, ck, res[name][ck], flush=True)
        del enc
    Path("eval/results/csci_d1.json").write_text(json.dumps(res, indent=1))
    for name in sets:
        print(name, {k: v.get("GR_R1") if isinstance(v, dict) else v for k, v in res[name].items()})


if __name__ == "__main__":
    main()

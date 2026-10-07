"""Identity-disjoint QME splits (plan 2B.1) -> splits/ccvid_qme.json, splits/mevid_qme.json.

CCVID: val = 15 IDs held out of the official train IDs; the whole official test (151 IDs) = test.
MEVID: val = 20 IDs held out of the official train IDs; official test = test (eval only).
Test IDs are never used for fitting or selection. (Earlier CCVID split = test halved into val / test:
splits/ccvid_qme_halftest.json, still read by the pre-CSCI diagnostics.)

    bash run.sh tools.make_splits --dataset ccvid --train_ids train.txt --test_ids test.txt
(id files: one identity per line, taken from the official dataset lists - never invented)
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = {"ccvid": 15, "mevid": 20}   # IDs held out of the official train IDs


def make_split(dataset, train_ids, test_ids, seed=0):
    train, test = sorted(set(map(str, train_ids))), sorted(set(map(str, test_ids)))
    if dataset not in VAL:
        raise ValueError(dataset)
    perm = np.random.default_rng(seed).permutation(train)   # test is eval-only; val = IDs held out of train
    val, train = sorted(perm[:VAL[dataset]].tolist()), sorted(perm[VAL[dataset]:].tolist())
    s = {"dataset": dataset, "seed": seed, "train": train, "val": val, "test": test}
    check_disjoint(s)
    return s


def check_disjoint(s):
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = set(s[a]) & set(s[b])
        assert not both, f"{s['dataset']}: IDs in both {a} and {b}: {sorted(both)[:10]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ccvid", "mevid"])
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--test_ids", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    read = lambda p: [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]  # noqa: E731
    s = make_split(a.dataset, read(a.train_ids), read(a.test_ids), a.seed)
    out = ROOT / "splits" / f"{a.dataset}_qme.json"
    out.write_text(json.dumps(s, indent=1))
    print(out, {k: len(v) for k, v in s.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()

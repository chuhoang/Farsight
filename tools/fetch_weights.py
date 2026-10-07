"""Download + verify every checkpoint declared in farsight/modules/**/manifest.yaml.
python -m tools.fetch_weights [name ...]"""
import sys
from pathlib import Path

from farsight.core import weights

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    want = set(sys.argv[1:])
    for man in sorted((ROOT / "farsight" / "modules").rglob("manifest.yaml")):
        m = weights.load_manifest(man.parent)
        if want and m["name"] not in want:
            continue
        for ck in m.get("checkpoints") or []:
            try:
                print("ok  ", m["name"], weights.fetch(man.parent, ck["file"]))
            except Exception as e:  # keep going; report every missing checkpoint
                print("FAIL", m["name"], ck["file"], "->", e)

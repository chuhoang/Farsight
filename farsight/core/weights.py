"""Fetch + sha256-verify checkpoints declared in each model folder's manifest.yaml.

Checkpoint entry fields: file, and one of {hf_repo (+hf_file), url, gdrive}; optional sha256.
Files land in weights/<manifest.module>/<manifest.name>/.
"""
import hashlib
import shutil
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_manifest(model_dir):
    return yaml.safe_load(Path(model_dir, "manifest.yaml").read_text(encoding="utf-8"))


def weight_dir(manifest):
    return WEIGHTS / manifest["module"] / manifest["name"]


def _download(ck, dst):
    if ck.get("hf_repo"):
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(ck["hf_repo"], ck.get("hf_file", ck["file"]))
        shutil.copyfile(src, dst)
    elif ck.get("gdrive"):
        import gdown
        gdown.download(id=ck["gdrive"], output=str(dst), quiet=False)
    elif ck.get("url"):
        urllib.request.urlretrieve(ck["url"], dst)
    else:
        raise FileNotFoundError(f"{dst} missing and manifest gives no source; download it manually")


def fetch(model_dir, file=None, download=True):
    """Return local path of a checkpoint (first one by default), downloading + verifying if needed."""
    m = load_manifest(model_dir)
    cks = m.get("checkpoints") or []
    ck = next(c for c in cks if file is None or c["file"] == file)
    dst = weight_dir(m) / ck["file"]
    if not dst.exists():
        if not download:
            raise FileNotFoundError(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _download(ck, dst)
    want = ck.get("sha256")
    if want and not str(want).startswith("<"):
        got = sha256(dst)
        if got != want:
            raise ValueError(f"sha256 mismatch for {dst}: {got} != {want}")
    return dst

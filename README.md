# FarSight-Lite

Re-implementation of FarSight 2.0 (face + gait + body fusion) following `plan_implement_farsight.md`.

## Environment
Windows Python can't be used here: the corporate web gateway blocks Windows wheels containing `.pyd`/`.dll`.
Everything runs in **WSL Ubuntu-22.04**, venv `~/fsenv` (Python 3.10, torch 2.11+cu128); pinned versions in `requirements.txt`.

```bash
# setup (inside WSL)
UV_NATIVE_TLS=1 uv venv ~/fsenv --python 3.10
UV_NATIVE_TLS=1 uv pip install --python ~/fsenv/bin/python -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
git submodule update --init --depth 1
bash run.sh tools.fetch_weights            # all checkpoints (HF + Google Drive), sha256-checked

# from Windows
wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/c/Users/hoangcm2/FarSight && bash run.sh pytest -q farsight/modules/m1_detect_track'
```
Run tests one module folder at a time (each loads its own GPU models).

## Usage
```bash
bash run.sh farsight.pipeline enroll subject.mp4 alice --gallery gallery.h5
bash run.sh farsight.pipeline search probe.mp4 --gallery gallery.h5
```
Config chooses models by folder name: `configs/pipeline/{v1,2A,2B}.yaml`.

## Status
| Module | Models | Checkpoint | Notes |
|---|---|---|---|
| M1 | BPJDet, YOLOv8x verifier, ByteTrack (ultralytics), PSR ResNet-18 | ✓ | ~17-19 FPS @1080p on RTX 5050 laptop (target 20) |
| M2 | quality gate, DATUM | ✓ | deform-attn CUDA op replaced by pure-torch port; `q0` untuned |
| M3 face | DFA + KP-RPE ViT-B | ✓ | inter_feat = QME-style blocks 8,16 (2048-d); ~110 crops/s |
| M3 gait | BigGait, BiggerGait (CCPG) | ✓ | ~7 / 4.6 clips/s (target 20) |
| M3 body | AIM LTCC (+PRCC) | ✓ | 4096-d, ~120 crops/s |
| M4 | z-score v1, QE, QME (+missing mask) | ✓ (QME repo) | QME heads trained on other encoders' score scales -> retrain (2B) |
| pipeline | enroll / search, HDF5 gallery | — | end-to-end test on vtest.avi |

Not done (needs data): accuracy reproduction (TinyFace, CCPG, LTCC/PRCC), tracker tuning (MOT17/MEVID),
z-score calibration, QE/QME training on CCVID/MEVID, result tables (plan 5.4), `make eval`, gRPC service.
Each model folder's README lists its deviations from the plan.

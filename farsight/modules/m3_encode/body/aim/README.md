# AIM body encoder (M3c)

AIM (CVPR'23, *Good is Bad: Causality Inspired Cloth-Debiasing for CC-ReID*), identity branch only:
ResNet-50 (last stride 1) -> maxavg pooling (`[max | avg]`, 4096) -> BN -> L2-norm. Clothes branch
(`model2`), fusion and classifiers are not built/loaded (the released checkpoints only contain `model_state_dict`).

- Code: `third_party/AIM-CCReID/models/img_resnet.py` (imported as-is; ImageNet init download is bypassed).
- Preprocess: AIM `transform_test` = PIL bilinear resize to 384x192, ImageNet mean/std.
- `inter_feat`: `layer3` output + GAP (1024-d). QME's AIM wrapper registers no hook (empty intermediate feats), so this follows the plan.
- Track aggregation: <= `max_frames` (32) uniformly sampled crops, mean of L2-normed feats, re-normed. `inter_feat` mean likewise.
- `quality` = mean(min(1, crop_h / 384)) x mean(body_score) in [0, 1].
- `half_dim: true` keeps only the avg-pool half (2048-d). `flip: true` adds AIM test.py's flip TTA (QME does not use it).

## Weights
`weights/m3_encode/body/aim/ltcc-checkpoint.pth.tar` (LTCC, default) and `prcc-checkpoint.pth.tar`, auto-fetched by
`farsight.core.weights.fetch` via gdown from the QME Drive folder
(https://drive.google.com/drive/folders/1TBt4HrJlm-Y-IO5SA7IAamZlWvj1vHQU, `checkpoints/AIM/`).
The QME LTCC file is byte-identical to the AIM repo's `ltcc.pth.tar`
(https://drive.google.com/drive/folders/1xohg_OAHjNyy7LLq3Fq_KowcEP9IlY8k). If gdown is blocked, download manually into that folder.

## Results
Reported (AIM repo): LTCC CC mAP 19.2 / R1 40.8; PRCC SC R1 100.0 / mAP 99.8, CC R1 58.2 / mAP 58.0.

PRCC (`bash run.sh eval.prcc`, AIM test_prcc protocol, PRCC checkpoint, -> `eval/results/prcc.json`), R1 / mAP:

| | SC | CC |
|---|---|---|
| AIM repo (flip TTA) | 100.0 / 99.8 | 58.2 / 58.0 |
| ours, flip=true (= AIM test.py) | 99.97 / 99.76 | 58.09 / 57.85 |
| ours, flip=false (default) | 99.97 / 99.72 | 58.20 / 57.54 |

Reproduced within 0.5 point. Flip TTA is worth <= 0.3 mAP, so it stays off. LTCC not re-measured (dataset not available here).

CCVID body-only (`bash run.sh eval.ccvid_body --frames 8`, CAL protocol, 8 uniformly sampled frames/tracklet,
-> `eval/results/ccvid_body.json`), R1 / mAP:

| checkpoint | general | CC | CC, face blurred (top 1/5) |
|---|---|---|---|
| **LTCC (chosen, default)** | 78.8 / 73.0 | 78.1 / 71.0 | 73.9 / 65.0 |
| PRCC | 78.2 / 53.9 | 74.8 / 49.0 | 72.1 / 50.0 |

Blurring the head region costs LTCC about 4 R1 / 6 mAP (CC), so AIM partly relies on the face and hair, but most of its
signal does not come from the face.

## Test
`wsl -d Ubuntu-22.04 -- bash run.sh pytest -q farsight/modules/m3_encode/body/aim`

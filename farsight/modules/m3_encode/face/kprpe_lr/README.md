# kprpe_lr: KP-RPE fine-tuned for long-range faces (train_m3.md)

Same network, config, template aggregation and `inter_feat` hook as `kprpe/`; only the checkpoint changes
(`config.yaml: checkpoint`). Registry name `kprpe_lr`. Rollback: `m3_encode.face: [dfa_aligner, kprpe]`.

## Pipeline

| Step | Code | Output |
|---|---|---|
| Face crop caches (A, B3) | `eval/run_face_cache.sh` → `eval.face_cache` | `~/datasets/face_cache/<name>/part_*.h5` |
| Diagnosis A1–A4 | `bash run.sh eval.face_eval --dist --a3 --val` | `eval/results/face_eval.json` |
| Train (C1–C3) | `bash run.sh farsight.modules.m3_encode.face.kprpe_lr.train --run NAME [k=v ...]` | `weights/m3_encode/face/kprpe_lr/NAME_sS.pt`, `eval/results/kprpe_lr/NAME_sS.json` |
| Eval of a checkpoint (E1, E2) | `bash run.sh eval.face_eval --ckpt weights/.../NAME_sS.pt --val --out ...` | face-only test + val, per IPD bin |
| Integration (D) | `eval.regait --mod face`, then QE / QME retrain | |

## Data available here vs train_m3 B1

| Plan source | Here |
|---|---|
| WebFace4M/12M (anti-forgetting) | **Not available** (password-gated). Replaced by distillation to the frozen original model on clean crops (`lambda_kd`), which also keeps gallery compatibility. |
| QMUL-SurvFace | Not downloaded yet (downloads last). Add as a `--folder` cache + a `sources` entry. |
| TinyFace train (2,570 IDs) | `tinyface_train` cache. TinyFace test stays the benchmark (`eval.tinyface`). |
| CCVID / MEVID train crops | `ccvid_train` / `mevid_train` caches, train IDs only (`splits/*_qme.json:train`); val IDs are the C4 val. |

## Degradation (B2)

`degrade.py` works at the crop's native scale. It draws a target eye distance below the crop's own, resizes to it,
and applies blur, turbulence, noise, gamma and JPEG there before upsampling back. These are the same ops and units
as `tools.degrade_dataset`, which produced the degraded CCVID eval set. Only CCVID crops are degraded; MEVID and
TinyFace crops are already low-resolution.

## Runs (C4, one factor at a time, selected on val)

| Run | Overrides | Factor |
|---|---|---|
| R0 | original checkpoint | baseline |
| R1 | `lambda_cons=0 --sources ccvid` | degradation augmentation |
| R2 | `lambda_cons=0 --sources ccvid tinyface` | real low-res faces |
| R2b | all sources, `lambda_cons=0` | in-domain crops |
| R3 | R2b + `lambda_cons=1` (the config default) | clean/degraded consistency |
| R4 | R3 + `ldmk_noise=2` | only if A3 shows alignment is the bottleneck |

# Datasets (plan sections 4, 5.1, 9)

Public ones are fetched into `~/datasets` (WSL) by `bash tools/fetch_datasets.sh <name...>`.

| Pri | Dataset | Used for | Access | Status |
|---|---|---|---|---|
| P0 | CCVID | main eval, QE/QME train | public Drive (CAL repo) | `ccvid` ✓ |
| P0 | MEVID | main eval, QE/QME train | public S3 (Kitware) | `mevid` |
| P0 | MOT17 | tracker tuning (ID switches) | public | `mot17` |
| — | QME test_feats | route 2A check | public Drive (QME repo) | `qme_feats` ✓ |
| P1 | PRCC | AIM reproduction | public Drive (AIM repo) | `prcc` ✓ |
| P1 | TinyFace | KP-RPE check | public Drive | `tinyface` ✓ |
| P1 | AG-ReID.v2 | aerial eval | public Drive | `agreid` |
| P1 | ATSyn (dynamic/static) | M2 eval/fine-tune | public Drive (DATUM repo), very large | deferred: `tools/degrade_dataset.py` simulator used instead (plan allows) |
| P1 | LTCC | AIM reproduction | **signed agreement** emailed to xuelinq92@gmail.com — https://naiq.github.io/LTCC_Perosn_ReID.html | needs user |
| P1 | CCPG | gait check | **signed agreement** to BNU-IVC@outlook.com (org email) — https://github.com/BNU-IVC/CCPG | needs user |
| P1 | CCGR-MINI | gait cross-domain | agreement — https://github.com/ShinanZou/CCGR | needs user |
| P2 | MSU-BRC | compare with FarSight | request to MSU | needs user |
| P2 | SUSTech1K, DroneGait, DroneSURF, UAV-Human, VisDrone, DanceTrack | extra eval | mixed | not fetched |
| P3 | WebFace4M | face fine-tune (only if needed) | agreement | not needed so far |

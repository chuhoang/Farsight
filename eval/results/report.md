# Main protocol results (plan 5.4)

`bash run.sh eval.main_protocol` on templates from `eval/run_extract.sh` (KP-RPE+DFA / BigGait CCPG / AIM LTCC).
Fusion v1 (z-score) fitted on CCVID **val** IDs only: w = face 4.0, gait 0.25, body 0.5, no quality weighting;
applied unchanged to CCVID test (76 IDs), CCVID test with degraded probes (down x4 + turb 2 + JPEG 30), MEVID test.
Conditions are probe-side: full / no-face (probe face removed) / short (probe gait removed). Raw numbers:
`eval/results/main_protocol.json`. Each cell: TAR@0.1%FAR / Rank-20 / FNIR@1%FPIR / GR-R1 / CC-mAP.

| set | config | full | no face | short track |
|---|---|---|---|---|
| CCVID test | face only | 73.4 / 97.1 / 9.5 / 93.3 / 77.5 | — | 73.4 / 97.1 / 9.5 / 93.3 / 77.5 |
| | gait only | 34.4 / 88.2 / 51.7 / 75.1 / 59.1 | 34.4 / 88.2 / 51.7 / 75.1 / 59.1 | — |
| | body only | 42.5 / 90.4 / 35.8 / 80.1 / 71.9 | 42.5 / 90.4 / 35.8 / 80.1 / 71.9 | 42.5 / 90.4 / 35.8 / 80.1 / 71.9 |
| | **fusion v1** | 74.9 / 97.4 / 9.5 / 93.8 / 82.3 | 49.0 / 91.1 / 34.4 / 81.3 / 74.5 | 75.2 / 97.4 / 9.5 / 93.8 / 82.6 |
| CCVID test, degraded probes | face only | 5.3 / 36.7 / 95.1 / 15.8 / 9.8 | — | 5.3 / 36.7 / 95.1 / 15.8 / 9.8 |
| | gait only | 15.1 / 58.3 / 87.4 / 32.1 / 20.0 | 15.1 / 58.3 / 87.4 / 32.1 / 20.0 | — |
| | body only | 35.5 / 73.1 / 72.6 / 44.4 / 39.4 | 35.5 / 73.1 / 72.6 / 44.4 / 39.4 | 35.5 / 73.1 / 72.6 / 44.4 / 39.4 |
| | **fusion v1** | 8.6 / 68.1 / 93.9 / 38.6 / 29.2 | 37.5 / 76.5 / 71.1 / 49.6 / 44.5 | 11.9 / 66.2 / 93.1 / 37.4 / 27.8 |
| MEVID test | face only | 13.7 / 49.4 / 86.5 / 29.7 / 6.1 | — | 13.7 / 49.4 / 86.5 / 29.7 / 6.1 |
| | gait only | 0.1 / 63.0 / 99.6 / 1.9 / 1.2 | 0.1 / 63.0 / 99.6 / 1.9 / 1.2 | — |
| | body only | 4.0 / 88.3 / 91.8 / 20.3 / 1.6 | 4.0 / 88.3 / 91.8 / 20.3 / 1.6 | 4.0 / 88.3 / 91.8 / 20.3 / 1.6 |
| | **fusion v1** | 13.9 / 89.2 / 86.2 / 36.1 / 6.0 | 3.3 / 84.5 / 93.3 / 15.8 / 1.7 | 13.8 / 90.8 / 86.2 / 38.3 / 5.8 |

Findings
- CCVID clean: fusion >= best single modality in every condition; no-face 81.3 vs body 80.1 R1 (plan DoD met on CCVID).
- Degraded probes: fixed weights tuned on clean val over-trust the (destroyed) face -> fusion < body when a face is
  present (38.6 vs 44.4 R1). Motivates quality-aware fusion (QE/QME, route 2B).
- MEVID: BigGait (CCPG) barely works (R1 1.9) and AIM (LTCC) is weak; no-face fusion < body (15.8 vs 20.3):
  plan DoD NOT met on MEVID. Cloth-changing (CC-mAP) is very low for all modalities.
- Face found in 99% of CCVID tracklets, 68% after degradation, ~55% on MEVID.

# Route 2B: QE + QME (`bash run.sh eval.train_2b`)

Train: CCVID train IDs, probe side 30% degraded copies (126/474); val: CCVID val IDs, 135/417 degraded probes (checkpoint + tau selection only). MEVID train not used (not downloaded).
Weights: `weights/m4_fusion/qe/qe_kprpe_2b.pth`, `weights/m4_fusion/qme/qme_kprpe_2b.pth`.

QE W check on CCVID test probes (clean vs degraded copy of the same tracklet, 285 pairs): mean 0.98 vs 0.39, AUC 0.995.

Cell: TAR@0.1%FAR / Rank-20 / FNIR@1%FPIR / GR-R1 / CC-mAP

| set | config | full | no face | short track |
|---|---|---|---|---|
| ccvid_test | best single | 73.4 / 97.1 / 9.5 / 93.3 / 77.5 | 42.5 / 90.4 / 35.8 / 80.1 / 71.9 | 73.4 / 97.1 / 9.5 / 93.3 / 77.5 |
|  | fusion v1 | 74.9 / 97.4 / 9.5 / 93.8 / 82.3 | 49.0 / 91.1 / 34.4 / 81.3 / 74.5 | 75.2 / 97.4 / 9.5 / 93.8 / 82.6 |
|  | QME 2B | 74.2 / 96.6 / 10.3 / 93.8 / 89.0 | 48.5 / 92.3 / 34.3 / 82.3 / 74.1 | 75.6 / 97.4 / 10.2 / 95.0 / 87.3 |
| ccvid_test_degraded | best single | 35.5 / 73.1 / 72.6 / 44.4 / 39.4 | 35.5 / 73.1 / 72.6 / 44.4 / 39.4 | 35.5 / 73.1 / 72.6 / 44.4 / 39.4 |
|  | fusion v1 | 8.6 / 68.1 / 93.9 / 38.6 / 29.2 | 37.5 / 76.5 / 71.1 / 49.6 / 44.5 | 11.9 / 66.2 / 93.1 / 37.4 / 27.8 |
|  | QME 2B | 32.7 / 77.9 / 75.9 / 52.3 / 44.2 | 34.8 / 71.5 / 70.5 / 53.0 / 43.6 | 26.1 / 75.3 / 84.0 / 47.5 / 39.8 |
| mevid_test | best single | 13.7 / 49.4 / 86.5 / 29.7 / 6.1 | 4.0 / 88.3 / 91.8 / 20.3 / 1.6 | 13.7 / 49.4 / 86.5 / 29.7 / 6.1 |
|  | fusion v1 | 13.9 / 89.2 / 86.2 / 36.1 / 6.0 | 3.3 / 84.5 / 93.3 / 15.8 / 1.7 | 13.8 / 90.8 / 86.2 / 38.3 / 5.8 |
|  | QME 2B | 6.4 / 87.7 / 86.1 / 27.2 / 3.5 | 2.4 / 82.6 / 95.5 / 10.1 / 1.4 | 12.5 / 90.5 / 85.3 / 38.6 / 5.4 |

Selection rule (plan 2B.4: QME only if better than v1 in full AND no-face): CCVID -> QME 2B; MEVID -> v1
(QME trained on 75 CCVID IDs overfits there, plan risk 'QME qua khop CCVID'; needs MEVID train IDs).
'best single' = single modality with the highest GR-R1 in that condition.

## Route 2B retrained on CCVID + MEVID train (`bash run.sh eval.train_2b --mevid --tag _cm`)

MEVID train: 20 tracklets/ID sampled (eval.extract --per_id 20, 1949 tracklets); IDs split 84 train / 20 val (held out of the official train IDs, splits/mevid_qme.json); checkpoint chosen on mean(CCVID val, MEVID val). QE W check: clean 0.96 vs degraded 0.38, AUC 0.980. Weights `*_2b_cm.pth`.

GR-R1 full / no face / short:

| set | fusion v1 | QME 2B (CCVID) | QME 2B (CCVID+MEVID) | best single |
|---|---|---|---|---|
| ccvid_test | 93.8 / 81.3 / 93.8 | 93.8 / 82.3 / 95.0 | 95.0 / 81.8 / 94.7 | 93.3 / 80.1 / 93.3 |
| ccvid_test_degraded | 38.6 / 49.6 / 37.4 | 52.3 / 53.0 / 47.5 | 50.1 / 51.8 / 44.6 | 44.4 / 44.4 / 44.4 |
| mevid_test | 36.1 / 15.8 / 38.3 | 27.2 / 10.1 / 38.6 | 32.9 / 14.6 / 39.2 | 29.7 / 20.3 / 29.7 |

MEVID training helps on MEVID (full 27.2 -> 32.9) but QME still trails v1 there and no fusion beats body-only
without a face (encoder-limited: BigGait R1 1.9, AIM 20.3 on MEVID; QME has a face QE only).

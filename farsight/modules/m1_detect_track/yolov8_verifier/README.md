# yolov8_verifier

YOLOv8 person detector used to verify BPJDet body boxes (plan 4/M1 step 2): `verify(frames, dets)` keeps a
detector body box (and its face row) only if a YOLO person with conf >= `conf` has IoU >= `iou_keep` with it.
`__call__` returns the YOLO persons in BaseDetector format. Checkpoint chosen by `checkpoint:` in config.yaml.

## Checkpoint choice (deviation from plan)
Plan: YOLOv8x COCO, conf 0.7, IoU 0.5. On MOT17 train that drops box recall of BPJDet bodies 71.4% -> 28.6%:
COCO boxes cover the visible extent while CrowdHuman/BPJDet boxes are full-body (amodal), so IoU often < 0.5,
and small/far people rarely reach conf 0.7. A CrowdHuman-trained verifier matches BPJDet's box definition.
BPJDet itself is unchanged. `bash run.sh eval.verifier_eval` (box-level, BPJDet conf >= 0.3, GT class 1):

| verifier | conf 0.7 / IoU 0.5 (plan) | conf 0.4 / IoU 0.5 |
|---|---|---|
| none | 71.4 R / 88.1 P | — |
| YOLOv8x COCO | 28.6 / 95.9 | 41.2 / 95.5 |
| **YOLOv8n CrowdHuman (default)** | 36.5 / 97.4 | **57.2 / 95.6** |
| YOLOv5m CrowdHuman (not used) | 54.1 / 96.2 | 65.6 / 92.4 |

Default = YOLOv8n CrowdHuman at conf 0.4 (user choice). Thresholds were picked on MOT17 train (optimistic).
The checkpoint is a community upload (HF raghavendra24/crowdhuman-yolov8n, no license, no card); its pickle
was scanned statically before use. Switch back: `checkpoint: yolov8x.pt`, `conf: 0.7`.

Test: `bash run.sh pytest -q farsight/modules/m1_detect_track/yolov8_verifier` (real image persons +
determinism; keep/drop logic incl. face row carried along and empty frames).
Preprocessing is ultralytics' own, not shared with BPJDet's GPU letterbox (plan step 3; cost is small).

## Effect on tracking (MOT17 train, TrackEval, raw M1 observations)
`bash run.sh eval.mot17 [--verifier crowdhuman_yolov8n --verifier-conf 0.4]`

| verifier | tracker | HOTA | MOTA | IDF1 | IDSW | Rec | Prec |
|---|---|---|---|---|---|---|---|
| COCO YOLOv8x, conf 0.7 | ByteTrack | 33.3 | 26.8 | 36.1 | 223 | 28.1 | 96.2 |
| COCO YOLOv8x, conf 0.7 | ByteTrack+PSR | 33.6 | 26.8 | 36.8 | 203 | 28.1 | 96.2 |
| CrowdHuman YOLOv8n, conf 0.4 | ByteTrack | 47.4 | 53.6 | 59.7 | 499 | 56.3 | 96.1 |
| CrowdHuman YOLOv8n, conf 0.4 | ByteTrack+PSR | 47.3 | 53.7 | 59.8 | 454 | 56.3 | 96.1 |
| none | ByteTrack+PSR | 52.3 | 61.8 | 66.0 | 663 | 70.2 | 90.0 |

PSR keeps fewer ID switches than plain ByteTrack with either verifier (plan acceptance).

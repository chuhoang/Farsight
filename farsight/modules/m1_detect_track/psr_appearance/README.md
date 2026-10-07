# psr_appearance

PSR (patch-memory re-ID) on top of ByteTrack, as described in FarSight 2.0. torchvision ResNet-18
(ImageNet, `resnet18-f37072fd.pth`), 512-d avg-pool features of 256x128 body crops, L2-normed.

- Per final ID: FIFO memory of K=10 features, one added every N=10 frames, only while the box doesn't overlap others.
- New raw ID from ByteTrack -> compared with memories of IDs absent in the frame and seen within T=10 s;
  closest under `thr` (cosine distance 0.15) takes over, else the new ID is kept.
- Two boxes overlapping (IoU>0.3) then separating -> swap their IDs if the swapped assignment matches memories
  better by `swap_margin`.

Test: `bash run.sh pytest -q farsight/modules/m1_detect_track/psr_appearance` -> 4 passed (real embeddings:
same person closer than another, determinism; remap/no-remap/window/crossing-swap logic with fake features).
On vtest.avi (795 frames): 26 -> 24 raw IDs merged into 23 kept tracklets vs 25 without PSR (2 re-entries remapped).

Deviations: cosine on L2-normed features by default (plan allows; `metric: mse` = paper). `thr` is not tuned on
MOT17/MEVID yet (needs labelled data); ID-switch comparison vs plain ByteTrack still to be done.

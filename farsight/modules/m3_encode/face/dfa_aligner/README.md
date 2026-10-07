# dfa_aligner

**What:** CVLface Differentiable Face Aligner (RetinaFace mobilenet0.25 landmark head, KP-RPE paper, CVPR 2024).
BGR face crop (any size) -> 5 landmarks (crop pixels) + aligned 112x112 RGB face in [-1,1] + landmarks in the
aligned image ([0,1], the KP-RPE keypoint input) + face confidence.

**Source:** code `third_party/CVLface/.../run_v1/aligners` (imported under a private name, see `../_cvlface.py`);
weights `pretrained_model/model.pt` from HF `minchul/cvlface_DFA_mobilenet` -> `weights/m3_encode/face/dfa_aligner/`.

**Usage:** `out = build("dfa_aligner")(crops)` -> `aligned, ldmk_aligned, score` (torch, on device), `ldmk` (numpy).

**Test:** `bash run.sh pytest -q farsight/modules/m3_encode/face/dfa_aligner` — 2 passed. 6 CVLface example faces +
skimage astronaut (non-square crop): score > 0.998, landmark geometry sane, aligned landmarks within 0.08 of the
ArcFace 5-point template, batch-invariant.

**Deviations:** square zero-padding + resize to 160 is done per crop with cv2 (CVLface does it in torch per batch)
so crops of different sizes can be batched; bilinear interpolation differs negligibly.

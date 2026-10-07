# biggergait — BiggerGait (OpenGait BiggerGait__DINOv2), CCPG checkpoint

- **What:** all 12 DINOv2 ViT-S/14 layers -> 12 HumanSpace convs -> 12 GaitBase heads, inference only. Same interface,
  preprocessing, clipping, quality and `inter_feat` (768-d, QME/BigGait position) as `../biggait` (subclass of its encoder).
- **Source:** `third_party/OpenGait` `BiggerGait_DINOv2.py`, config `configs/biggergait/biggergait__DINOv2_CCPG.yaml`,
  HF `opengait/OpenGait` `CCPG/BiggerGait__DINOv2/BiggerGait__Dinov2/...WiMask-30000.pt`.
- **Checkpoint quirks:** backbone is stored in original-DINOv2 key format (not HF `Dinov2Model` as the current code expects),
  so the wrapper keeps OpenGait's `vit_small` and adapts its forward to return HF-style `hidden_states`; `transformers` is not used.
  Gait heads are stored under the old name `Gait_List` (shares modules with `real_gait`), hence the non-strict load + assertion.
- **Output:** `feat` 98304-d (12 heads x 256 x 32 parts) = 384 KB/template in fp32, well above the plan's 0.05 MB/person.
  The Group checkpoint (2 heads, 16384-d) is listed in manifest.yaml and selectable via config.yaml only (untested).
- **Tests:** `bash run.sh pytest -q farsight/modules/m3_encode/gait/biggergait` -> 3 passed.
- **Speed (RTX 5050, fp16, clip_batch 1):** ~4.6 track-clips/s.
- **Not done:** CCPG accuracy (no dataset); CCGR Group checkpoint (`CCGR/.../Share_2B_6G`) has no matching config in the repo.

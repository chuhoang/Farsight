# biggait — BigGait (CVPR'24), CCPG checkpoint

- **What:** OpenGait `BigGait__Dinov2_Gaitbase` (frozen DINOv2 ViT-S/14 + Mask/Appearance/Denoising branches + GaitBase), inference only.
- **Source:** code `third_party/OpenGait` (imported in isolation by `../_opengait.py`, OpenGait's `BaseModel`/training stack stubbed out);
  config `configs/biggait/BigGait_CCPG.yaml`; checkpoint HF `opengait/OpenGait` `CCPG/.../BigGait__Dinov2_Gaitbase_Frame30-40000.pt`
  (full state dict incl. DINOv2 backbone, so no separate `dinov2_vits14_pretrain.pth` is downloaded).
- **Pipeline:** BGR crops -> OpenGait `resize_with_padding` 256x128 -> RGB, ImageNet norm -> model upsamples to 448x224.
  30-frame clips (last clip right-aligned/overlapping), clip feats L2-normed, averaged, re-normed. `< 15` frames -> `None`.
- **Output:** `feat` 4096-d (GaitBase SeparateFCs 256 x 16 parts, flattened). `inter_feat` 768-d: QME hook position
  (`x_norm_patchtokens_mid4` of the DINOv2 backbone, middle 2 of the 4 blocks = blocks 5, 8; mean over patches, frames, clips).
  `quality` = fraction of crops with height >= 128 px x min(1, n/60).
- **Tests:** `bash run.sh pytest -q farsight/modules/m3_encode/gait/biggait` -> 4 passed (shape, determinism, finite, None for <15 frames,
  synthetic walker A clips closer to each other than to walker B, speed).
- **Speed (RTX 5050 8GB, fp16, idle GPU):** ~7-8 track-clips/s (0.12 s/clip, of which DINOv2 0.07 s). Plan target 20 not met;
  backbone alone caps at ~14 clips/s at 448x224 x 30 frames. Speed-ups that don't change the math: torch SDPA instead of the naive
  attention (xFormers absent), ~2x.
- **Deviations / not done:** CCPG accuracy reproduction not run (dataset unavailable). OpenGait evaluates the whole sequence
  (`all_ordered`, <=250 frames) in one pass; we use 30-frame clips per plan.

# datum (M2)

DATUM (Zhang et al., CVPR 2024, "Spatio-Temporal Turbulence Mitigation: A Translational Perspective"),
pretrained **dynamic scene** model (ATSyn-dynamic), used as-is (no fine-tuning).

- Input: list of BGR face crops of one track (padded 20%), any sizes. Each crop is resized to
  `size`x`size` (256; SPyNet runs at 1/4 res with a 5-level pyramid, so smaller inputs crash),
  run in sliding windows of `clip`=16 frames (overlap 2*`margin`), and resized back to its own size.
  Tracks shorter than 5 frames are padded by repeating the last frame. `output_full=True`.
- Deviation: the repo's guided deformable attention is a JIT CUDA extension (needs nvcc + gcc,
  not available in our WSL env). `model.py:deform_attn_torch` re-implements its forward pass with
  `grid_sample` (inference only); `tests/` checks it against a literal loop port of the kernel.
- Checkpoint: `weights/m2_restore/datum/DATUM_dynamic.pth` (23 MB), Google Drive id in manifest
  (`python -m tools.fetch_weights datum`). If gdown is blocked: open
  https://drive.google.com/file/d/1IClAWZ-9kY5TggmuQGp11_dqOgCOCbjx/view in a browser, download,
  save as the path above; sha256 is checked.
- Sanity result (tests): COCO crop 256x256, 20 frames of tilt (1.5 px) + gaussian blur (1.0) + noise (3):
  PSNR 20.53 dB degraded -> 24.93 dB restored. Gains shrink at stronger turbulence / small crops.
- GPU memory: ~2 GB peak at clip=16, 256x256.

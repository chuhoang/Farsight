# bpjdet

BPJDet (Zhou et al., body-part joint detection, YOLOv5-L6 fork), checkpoint `ch_face_l_1536_e150_best_mMR.pt`
(CrowdHuman body+face) from HF `HoyerChou/BPJDet`. Code: `third_party/BPJDet` (commit in manifest).

- Input: letterbox to 1536 long side (stride-64 min padding, as `demos/` with `auto=True`), done on GPU, fp16, batch 8.
- Decode: repo's `non_max_suppression` (body class 0, face class 1) + `val.post_process_batch`
  (face offset -> nearest body, kept only if face inner-IoU with body > 0.6, highest-conf face wins).
- Output: `{"body": (N,5), "face": (N,5)}`, face row = NaN when the body has no matched face.

Test: `bash run.sh pytest -q farsight/modules/m1_detect_track/bpjdet` -> 3 passed (CrowdHuman demo images:
8-19 bodies/img, most with faces; batch == single; empty frame).
Speed: ~17 FPS alone at 1080p (1536x896 input, fp16, RTX 5050 Laptop).

Deviations: body conf threshold 0.3 (demo 0.7) so ByteTrack's low-score stage has boxes; face conf 0.5.
`utils.bp_eval` (eval-only, needs fvcore) is stubbed when importing `val.py`. GPU letterbox uses
bilinear `F.interpolate` instead of cv2 INTER_LINEAR (sub-pixel differences).
Repo's top-level `models`/`utils` packages are imported in isolation and removed from `sys.modules` afterwards.

# kprpe

**What:** KP-RPE ViT-Base (AdaFace loss, WebFace12M), no training. Crops -> `dfa_aligner` -> KP-RPE(aligned 112,
aligned landmarks) -> 512-d feat. Implements `BaseEncoder.encode(crops, track) -> ModalityOut | None`, plus
`embed(crops)` (per-crop L2 feats, M2 safety check), `quality(crops)` (per-crop norm before L2, M2 gate),
`forward(crops)` -> (raw feats, inter_feat, DFA score).

**Source:** code `third_party/CVLface/.../run_v1/models/vit_kprpe`; weights `pretrained_model/model.pt` from HF
`minchul/cvlface_adaface_vit_base_kprpe_webface12m` -> `weights/m3_encode/face/kprpe/`. Published TinyFace
rank-1/5 = 76.10 / 78.92.

**inter_feat (QE input):** forward hooks (`core.hooks.LayerCapture`) on `net.blocks[8]` and `net.blocks[16]` (of 24),
same pass. This is exactly what QME_ICCV25 `build_face_backbone` hooks on the KP-RPE/AdaFace ViT; its FaceQE then
LayerNorms the 196 patch tokens and takes mean+std per block (`make_style`). We pool the same way -> 2048-d
(`inter_pool: style`). Other candidates via config: `[16]` style, `mean` pool. This ViT has no CLS token, so
`core.hooks.pool_tokens(..., "cls"/"mean")` doesn't fit (its "mean" drops token 0); pooling is local.

**Template:** drop frames with norm < `min_quality` or DFA score < `min_face_score`; weights = softmax(norm/T);
feat = re-L2 of weighted mean of per-frame L2 feats; inter_feat/quality aggregated with the same weights.
None if no frame passes.

**Test:** `bash run.sh pytest -q farsight/modules/m3_encode/face/kprpe` — 3 passed. CVLface verification pairs:
same 0.65 (README ref 0.67), different -0.07 / 0.00 (ref -0.02 / 0.05); deterministic; batch-invariant to 1e-3;
noise frame rejected; template behaviour checked.

**Speed (RTX 5050, bs 32, DFA + KP-RPE incl. CPU preprocess):** fp32 ~110 crops/s, `fp16: true` ~210 crops/s
(cos to fp32 > 0.99999). Target >= 30 met. Peak GPU ~1.1 GB.

**Deviations / notes:**
- Feature norm is a weak quality signal for this ViT: random-noise crops get norms 21-24, heavy blur ~27, real
  faces 14-24. Hence the extra DFA face-score gate. Thresholds are placeholders; tune on degraded CCVID/MEVID.
- rpe_ops CUDA extension is not built (no compiler); CVLface's pure-torch fallback is used (identical output).
- TinyFace evaluation (plan acceptance) not done: dataset not available. No eval stub written.

## Results (TinyFace, `bash run.sh eval.tinyface`)
Full protocol: 4443 probes vs 157 match + 153,428 distractors, cosine, no flip TTA, fp32.

| | R1 | R5 | R20 |
|---|---|---|---|
| CVLface README | 76.10 | 78.92 | — |
| ours | 75.11 | 77.98 | 79.35 |

Gap -0.99 R1 (at the plan's 1-point limit). Likely cause: no flip test-time augmentation (not verified).
Embedding runs at ~90 img/s including DFA alignment; features are cached in `~/datasets/cache/`.

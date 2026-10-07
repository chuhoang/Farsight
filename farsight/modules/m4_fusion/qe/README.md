# qe — Quality Estimator (QME, ICCV'25)
Port of QME `Face_Quality_Estimator` (same param names). Input: face `inter_feat` = QME style vector,
2048-d for a ViT-B (C=512): blocks [8,16] tokens -> LayerNorm(no affine) -> [mean,std] over patches,
flattened (blk, stat, C), averaged over frames (`style_from_blocks`). KP-RPE 2048-d inter_feat fits directly.
Also accepts per-frame styles (N,2048) or raw tokens (N,2,P,512). Output W in (0,1); NaN row -> 0.

- Route 2A: `fgb_mod_qe-adaface_t1r20-mevid-6000.pth` (default) / `..._t1r3-ltcc-6000.pth`, fetched via
  manifest (gdown). Trained on AdaFace features — domain shift vs KP-RPE expected. No CCVID face QE was released
  (`fgb_mod_qe-adaface-t1r3-0.00-3.pth` referenced in QME config is not in the Drive folder); the CCVID QME
  checkpoint carries its own QE (`checkpoint: <path to lsf-ccvid...pth>` works).
- Route 2B: `bash run.sh farsight.modules.m4_fusion.qe.train --scores train.h5 --out weights/m4_fusion/qe/qe_2b.pth`
  on tools/export_scores.py output; pseudo label `relu((T-rank)/(T-1))` (QME rank_threshold), MSE, encoder frozen.

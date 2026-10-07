# qme — Quality-guided Mixture of score-fusion Experts
Port of QME `LSN` + `MoNormQE_dev` (Zhu et al., ICCV'25, MIT): BatchNorm score normalisation, Z experts
(MLP over modality scores), gated by the face QE weight (Z=2: `w*E0 + (1-w)*E1`, Z>2: hat gates).
Loss: QME `ScoreTripletLoss.cal_score_loss_cc` imported from third_party/QME_ICCV25/loss.py.
QME's model.py is not importable (pulls all backbones), so modules are ported.

Missing modalities (QME has no masking): NaN scores -> neutral 0 after BN, `missing` flags as extra expert
input (`use_mask: true`), face weight 0 when the probe has no face, score-level drop augmentation in training
(face 20%, gait 15%).

- 2A: `registry.build("qme", checkpoint="lsf-ccvid-bs8-seq8-245.92-630.pth", use_mask=False)` (also lsf-mevid).
  Order face,gait,body = QME adaface,biggait/agrl,cal. Score scales differ from our encoders (QME gait uses
  1/(1+L2)), so 2A on our features is a smoke path, not a reproduction.
- 2B: `bash run.sh farsight.modules.m4_fusion.qme.train --train tr.h5 --val va.h5 --qe <qe.pth> --out weights/m4_fusion/qme/qme_2b.pth`
  (export_scores.py format; best step by rank1 + TAR@1% − FNIR@1%; tau@1%FPIR saved in the checkpoint).

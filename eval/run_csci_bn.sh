#!/usr/bin/env bash
# Hướng C (qme_body_backbone_issue.md): QME-MEVID trained with body-score noise augmentation, k ~ U(0, K_MAX) x sd(body)
# per probe (train batches + fixed draw on val). K_MAX fixed beforehand (1.0), not tuned on test. 3 seeds, then report.
cd "$(dirname "$0")/.." || exit 1
export FARSIGHT_DEGRADED=0
K_MAX=1.0
L=~/datasets/feats/run_csci_bn.log
q() { grep --line-buffered -v "Sep Attention\|def \|Warning\|warnings.warn"; }
for s in 0 1 2; do
  echo "== QME-MEVID body_noise $K_MAX seed $s" >> $L
  bash run.sh eval.train_2b --dataset mevid --seed $s --body_noise $K_MAX --tag _csci_mevid_bn_s$s \
    --v1 eval/results/main_protocol_csci.json --out eval/results/qme_2b_csci_mevid_bn_s$s.json 2>&1 | q >> $L
done
echo "== E1" >> $L; bash run.sh eval.csci_report --tag _bn 2>&1 | q >> $L
echo ALL_DONE >> $L

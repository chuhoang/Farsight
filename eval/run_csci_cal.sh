#!/usr/bin/env bash
# QE with CAL-rule pseudo-labels (same-identity same-camera gallery ignored) + camera-split train pairs, then QME;
# one QE+QME per dataset, 3 seeds, then the E1 report (tag _cal). Fusion v1 / encoders unchanged.
cd "$(dirname "$0")/.." || exit 1
export FARSIGHT_DEGRADED=0
L=~/datasets/feats/run_csci_cal.log
q() { grep --line-buffered -v "Sep Attention\|def \|Warning\|warnings.warn"; }
for d in ccvid mevid; do
  for s in 0 1 2; do
    echo "== QE+QME (cal labels) $d seed $s" >> $L
    bash run.sh eval.train_2b --dataset $d --seed $s --qe_label cal --tag _csci_${d}_cal_s$s \
      --v1 eval/results/main_protocol_csci.json --out eval/results/qme_2b_csci_${d}_cal_s$s.json 2>&1 | q >> $L
  done
done
echo "== E1" >> $L; bash run.sh eval.csci_report --tag _cal 2>&1 | q >> $L
echo ALL_DONE >> $L

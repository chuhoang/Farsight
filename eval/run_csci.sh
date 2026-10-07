#!/usr/bin/env bash
# plan_csci after D2 (body = CSCI-V MEVID, gait = mask-fixed BigGait): C, D3 (fusion v1 per dataset) and D4 (one QME
# per dataset, 3 seeds), then the E1 report. Sequential. Degraded CCVID stores skipped.
# Splits: val = IDs held out of each dataset's train IDs; test = the whole official test (CCVID 151, MEVID 54 IDs).
cd "$(dirname "$0")/.." || exit 1
export FARSIGHT_DEGRADED=0
L=~/datasets/feats/run_csci.log
q() { grep --line-buffered -v "Sep Attention\|def \|Warning\|warnings.warn"; }
echo "== C" >> $L;  bash run.sh eval.csci_complement --tag csci 2>&1 | q >> $L
echo "== D3" >> $L; bash run.sh eval.main_protocol --out eval/results/main_protocol_csci.json 2>&1 | q >> $L
for d in ccvid mevid; do
  for s in 0 1 2; do
    echo "== D4 $d seed $s" >> $L
    bash run.sh eval.train_2b --dataset $d --seed $s --tag _csci_${d}_s$s --v1 eval/results/main_protocol_csci.json \
      --out eval/results/qme_2b_csci_${d}_s$s.json 2>&1 | q >> $L
  done
done
echo "== E1" >> $L; bash run.sh eval.csci_report 2>&1 | q >> $L
echo ALL_DONE >> $L

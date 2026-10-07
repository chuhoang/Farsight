#!/usr/bin/env bash
# sequential, resumable: degraded CCVID train templates, then degraded val probes (plan 2B.2: val gets the same mix)
cd /mnt/c/Users/hoangcm2/FarSight || exit 1
F=~/datasets/feats; D="down:4 turb:2 jpeg:30"
bash run.sh eval.extract --out $F/ccvid_train_degraded.h5 --dataset ccvid --split train --degrade $D >> $F/ccvid_train_degraded.log 2>&1
bash run.sh eval.extract --out $F/ccvid_query_degraded_val.h5 --dataset ccvid --split query \
  --ids splits/ccvid_qme.json:val --degrade $D >> $F/ccvid_query_degraded_val.log 2>&1
echo EXIT $?; tail -1 $F/ccvid_train_degraded.log; tail -1 $F/ccvid_query_degraded_val.log

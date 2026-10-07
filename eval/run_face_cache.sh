#!/usr/bin/env bash
# train_m3: face crop caches (resumable: rerun to continue). Logs -> ~/datasets/face_cache/*.log
cd "$(dirname "$0")/.." || exit 1
L=~/datasets/face_cache; mkdir -p $L; rm -rf $L/_smoke_deg
D="down:4 turb:2 jpeg:30"
x() { name=$1; shift; bash run.sh eval.face_cache "$@" 2>&1 | grep --line-buffered -v "^compatible\|^Check\|^$\|Incompatible\|Loaded pretrained\|meshgrid" >> $L/$name.log; tail -1 $L/$name.log; }
x ccvid_query   --store ccvid_query
x ccvid_gallery --store ccvid_gallery
x mevid_query   --store mevid_query
x mevid_gallery --store mevid_gallery
x ccvid_query_degraded_full --store ccvid_query --degrade $D --out ccvid_query_degraded_full
x ccvid_train   --store ccvid_train
x mevid_train   --store mevid_train
x ccvid_train_degraded_val --store ccvid_train --ids splits/ccvid_qme.json:val --degrade $D --out ccvid_train_degraded_val
x tinyface_train --folder ~/datasets/tinyface/tinyface/Training_Set --out tinyface_train
echo ALL_DONE

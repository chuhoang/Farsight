#!/usr/bin/env bash
# train_m3 C4 runs, one at a time (resumable: rerun to continue). Logs -> ~/datasets/face_cache/train_<run>.log
cd "$(dirname "$0")/.." || exit 1
L=~/datasets/face_cache
r() { name=$1; shift
  if [ -f eval/results/kprpe_lr/${name}_s0.json ] && [ ! -f weights/m3_encode/face/kprpe_lr/${name}_s0.last.pt ]; then echo "$name done"; return; fi
  bash run.sh farsight.modules.m3_encode.face.kprpe_lr.train --run $name "$@" 2>&1 \
    | grep --line-buffered -vE 'Warning|warn|^compatible|^Check|^$|Incompatible|Loaded|All keys' >> $L/train_$name.log
  tail -1 $L/train_$name.log; }
# r r1  lambda_cons=0 --sources ccvid
# r r2  lambda_cons=0 --sources ccvid tinyface
r r2b lambda_cons=0
r r3
echo ALL_DONE

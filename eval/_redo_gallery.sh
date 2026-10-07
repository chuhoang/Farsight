#!/usr/bin/env bash
# wait for the running extraction, then redo the gallery file truncated by the WSL crash (sequential, one job at a time)
while pgrep -f "eval/run_extract.sh" >/dev/null; do sleep 20; done
rm -f ~/datasets/feats/ccvid_gallery.h5
mv ~/datasets/feats/ccvid_gallery.log ~/datasets/feats/ccvid_gallery.crashed.log 2>/dev/null
cd /mnt/c/Users/hoangcm2/FarSight && bash eval/run_extract.sh > ~/datasets/feats/run_extract2.out 2>&1
echo EXIT $?; tail -4 ~/datasets/feats/run_extract2.out

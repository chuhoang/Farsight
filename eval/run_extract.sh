#!/usr/bin/env bash
# Main-protocol feature extraction (resumable: rerun to continue). Logs -> ~/datasets/feats/*.log
cd "$(dirname "$0")/.." || exit 1
F=~/datasets/feats; mkdir -p "$F" splits
ann=~/datasets/mevid/mevid-v1-annotation-data/mevid-v1-annotation-data
# official ID lists -> splits/*.json (tools.make_splits asserts disjointness)
awk '{print $2}' ~/datasets/ccvid/CCVID/train.txt | sort -u > "$F/ccvid_train_ids.txt"
cat ~/datasets/ccvid/CCVID/query.txt ~/datasets/ccvid/CCVID/gallery.txt | awk 'NF{print $2}' | sort -u > "$F/ccvid_test_ids.txt"
awk '{printf "%04d\n", $3}' $ann/track_train_info.txt | sort -u > "$F/mevid_train_ids.txt"
awk '{printf "%04d\n", $3}' $ann/track_test_info.txt | sort -u > "$F/mevid_test_ids.txt"
bash run.sh tools.make_splits --dataset ccvid --train_ids "$F/ccvid_train_ids.txt" --test_ids "$F/ccvid_test_ids.txt"
bash run.sh tools.make_splits --dataset mevid --train_ids "$F/mevid_train_ids.txt" --test_ids "$F/mevid_test_ids.txt"
x() { name=$1; shift; bash run.sh eval.extract --out "$F/$name.h5" "$@" >> "$F/$name.log" 2>&1; tail -1 "$F/$name.log"; }
x ccvid_query   --dataset ccvid --split query
x ccvid_gallery --dataset ccvid --split gallery
x ccvid_query_degraded --dataset ccvid --split query --ids splits/ccvid_qme.json:test --degrade down:4 turb:2 jpeg:30
x mevid_query   --dataset mevid --split query
x mevid_gallery --dataset mevid --split gallery
x ccvid_train   --dataset ccvid --split train
echo ALL_DONE

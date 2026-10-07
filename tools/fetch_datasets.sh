#!/usr/bin/env bash
# Download public datasets from the plan into $DATA (default ~/datasets, inside WSL).
# Usage: bash tools/fetch_datasets.sh qme_feats tinyface prcc ccvid mot17 mevid agreid
# Gated datasets (LTCC, AG-ReID.v2, CCPG, CCGR, MSU-BRC, ...) need an agreement from the authors: see datasets.md.
set -euo pipefail
DATA=${DATA:-$HOME/datasets}
GDOWN=~/fsenv/bin/gdown
mkdir -p "$DATA"; cd "$DATA"

gd()  { [ -s "$2" ] || $GDOWN --continue "$1" -O "$2"; }          # google drive id -> file
web() { [ -s "$2" ] && return; curl -fL --retry 5 -C - -o "$2.part" "$1"; mv "$2.part" "$2"; }
unz() { [ -e "$2/.done" ] || { mkdir -p "$2"; if file "$1" | grep -q Zip; then unzip -q -o "$1" -d "$2"; else tar -xf "$1" -C "$2"; fi; touch "$2/.done"; }; }

for d in "$@"; do echo "== $d"; case $d in
  qme_feats)  # QME_ICCV25 Drive: precomputed score matrices + QE weights (route 2A check without videos)
    mkdir -p qme/test_feats; cd qme/test_feats
    gd 13gyn5fL9ALX2zCA7cGKuYnW2DRnC10mP scoremats_ccvid.h5; gd 19x3UI-nf4UjUEOqnWziWP6bPF7AZGQr6 scoremats_mevid.h5
    gd 1Sv-jib5XtEOCMV_rRbK47kfeAFeXD0uj scoremats_ltcc.h5;  gd 19N6l0bEaib19hbjIL3IetNe70wcbTvYv mod_qe_adaface-t1r3_ccvid_3.h5
    gd 1i_rUThLCQHnDh58la_o5MsgpY1KXvYmd mod_qe_adaface_t1r3_mevid_6000.h5; gd 1FtoGxqpIeNxpSyD4tL-ax5ju2i9QVpIX mod_qe_adaface_t1r3_ltcc_6000.h5
    cd "$DATA";;
  tinyface) gd 1xTZc7lNmWN33ECO2AKH6FycGdiqIK7W0 tinyface.zip; unz tinyface.zip tinyface;;
  prcc)     gd 1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz prcc.rar   # rar: static 7-Zip (no sudo needed)
            [ -x ~/.local/bin/7zz ] || { curl -fsSL https://www.7-zip.org/a/7z2301-linux-x64.tar.xz | tar -xJ -C ~/.local/bin 7zz; }
            [ -e prcc/.done ] || { mkdir -p prcc; ~/.local/bin/7zz x -y -so prcc.rar | tar -x -C prcc && touch prcc/.done; };;  # rar holds a tar
  ccvid)    gd 1vkZxm5v-aBXa_JEi23MMeW4DgisGtS4W CCVID.zip; unz CCVID.zip ccvid;;
  mot17)    web https://motchallenge.net/data/MOT17.zip MOT17.zip; unz MOT17.zip mot17;;
  mevid)
    mkdir -p mevid; cd mevid
    for f in mevid-v1-annotation-data.zip mevid-v1-bbox-test.tgz mevid-v1-bbox-train.tgz; do
      web https://mevadata-public-01.s3.amazonaws.com/mevid-annotations/$f $f; unz $f ${f%.*}; done
    cd "$DATA";;
  agreid)   # AG-ReID.v2 (public Drive folder: zip + protocol txt files)
    mkdir -p agreid; cd agreid
    gd 1CiGr9CrWo_Zi-wjuT4hm4Qf69FZl-imR AG-ReID.v2.zip; unz AG-ReID.v2.zip data
    gd 1KXRtf4N-htwUkJBCH51DUtZl3WBWGkBT exp1_aerial_to_cctv.txt; gd 1_WcGAG1DdJ8iJUPKHWIUuKo1apLgM408 exp2_aerial_to_wearable.txt
    gd 1G7KAHQgXehwdy0237HqhlywguLKvOv0M exp4_cctv_to_aerial.txt; gd 11DTkC1qPGv9ZvhDQB_KoHazwSdFxbahL exp5_wearable_to_aerial.txt
    cd "$DATA";;
  *) echo "unknown dataset $d"; exit 1;;
esac; done

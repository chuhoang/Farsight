until grep -qE "^  1074/1074|Traceback|Error" ~/datasets/feats/ccvid_gallery.log 2>/dev/null; do sleep 30; done
tail -3 ~/datasets/feats/ccvid_gallery.log

for f in ~/datasets/feats/*.log; do echo "$f: $(tail -n 1 "$f")"; done
free -g | head -2

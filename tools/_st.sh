uptime
ps -eo etime,args | grep -E "eval\." | grep -v grep | cut -c1-90
grep -E "templates,|rewritten|already|rror|Traceback" ~/datasets/feats/regait.log | tail -12
tail -1 ~/datasets/feats/regait.log
ls ~/datasets/feats/*.gait1.h5 ~/datasets/feats/*.gait2.h5 2>/dev/null

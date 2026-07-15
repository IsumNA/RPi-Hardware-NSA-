#!/bin/bash
cd ~/ctt-server-workspace/imx662 || exit 1
for tag in 500l 1l 100l 25l 10l 5l 398l 373l 84l 29l; do
  files=$(find . -maxdepth 1 -iname "imx662_5000k_${tag}_*.dng")
  n=$(echo "$files" | grep -c .)
  if [ -n "$files" ]; then
    first=$(echo "$files" | xargs stat -c "%Y" | sort -n | head -1)
    last=$(echo "$files" | xargs stat -c "%Y" | sort -n | tail -1)
    fdate=$(date -d @"$first" "+%m-%d %H:%M:%S")
    ldate=$(date -d @"$last" "+%m-%d %H:%M:%S")
    echo "${tag}: n=${n}  ${fdate} -> ${ldate}"
  fi
done

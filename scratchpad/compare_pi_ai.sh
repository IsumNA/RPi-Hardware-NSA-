#!/bin/bash
set -e
ssh pi@10.3.31.153 "find ~/ctt-server-workspace/imx662 -maxdepth 1 -iname '*.dng' -printf '%f\n'" | sort > /tmp/pi_dng_list.txt
find ~/RPi-Hardware-NSA-/datasets/imx662_project/.ctt_mirror -iname "*.dng" -printf "%f\n" | sort > /tmp/ai_mirror_list.txt
echo "pi dng count:"; wc -l < /tmp/pi_dng_list.txt
echo "ai .ctt_mirror dng count:"; wc -l < /tmp/ai_mirror_list.txt
echo "on pi but NOT in ai .ctt_mirror:"; comm -23 /tmp/pi_dng_list.txt /tmp/ai_mirror_list.txt | wc -l
echo "in ai .ctt_mirror but NOT on pi:"; comm -13 /tmp/pi_dng_list.txt /tmp/ai_mirror_list.txt | wc -l
echo "sample pi-only files:"; comm -23 /tmp/pi_dng_list.txt /tmp/ai_mirror_list.txt | head -10

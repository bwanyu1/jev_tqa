#!/usr/bin/env bash
# Image multiple-choice evaluation of every stage: original AR (untrained), general diffusion v3,
# Jev v4 (diffusion), and the AR SFT control. Waits for the running Jev v4 pipeline.
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
run() { log "start: $1"; shift; if "$@"; then log "done"; else log "FAILED: $*"; fi; }
while pgrep -f "scripts/[p]ipeline_jev_v4.sh" > /dev/null; do sleep 30; done
EV=data/image_eval.jsonl
run "image eval: original AR"  .venv/bin/python -u -m djev.baseline_ar --data $EV --out logs/img_ar_base.jsonl > logs/img_ar_base.log 2>&1
run "image eval: general v3"   .venv/bin/python -u -m djev.evaluate --checkpoint runs/general_v3 --data $EV --canvas 64 --out logs/img_general_v3.jsonl > logs/img_general_v3.log 2>&1
run "image eval: Jev v4"       .venv/bin/python -u -m djev.evaluate --checkpoint runs/jev_v4 --data $EV --out logs/img_jev_v4.jsonl > logs/img_jev_v4.log 2>&1
run "image eval: AR SFT v4"    .venv/bin/python -u -m djev.baseline_ar --checkpoint runs/ar_sft_v4 --data $EV --out logs/img_ar_sft_v4.jsonl > logs/img_ar_sft_v4.log 2>&1
.venv/bin/python -m djev.calibration logs/img_ar_base.jsonl logs/img_general_v3.jsonl logs/img_jev_v4.jsonl logs/img_ar_sft_v4.jsonl > logs/report_image.jsonl 2>&1
log "image eval finished: logs/report_image.jsonl"

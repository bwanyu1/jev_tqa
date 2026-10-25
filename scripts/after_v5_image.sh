#!/usr/bin/env bash
# After the fair AR control: image evaluation of the final infill model and the AR v5 control.
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
run() { log "start: $1"; shift; if "$@"; then log "done"; else log "FAILED: $*"; fi; }
while pgrep -f "scripts/[a]fter_infill_ar_v5.sh" > /dev/null; do sleep 30; done
run "image eval: infill diffusion" .venv/bin/python -u -m djev.infill eval --mode diffusion --checkpoint runs/jev_infill --data data/image_eval.jsonl --out logs/img_jev_infill.jsonl > logs/img_jev_infill.log 2>&1
run "image eval: AR SFT v5" .venv/bin/python -u -m djev.baseline_ar --checkpoint runs/ar_sft_v5 --data data/image_eval.jsonl --out logs/img_ar_sft_v5.jsonl > logs/img_ar_sft_v5.log 2>&1
.venv/bin/python -m djev.calibration logs/img_ar_base.jsonl logs/img_jev_infill.jsonl logs/img_ar_sft_v5.jsonl > logs/report_image_v5.jsonl 2>&1
log "final image eval finished"

#!/usr/bin/env bash
# Wait for the running Jev SFT process, then: speed benchmark -> evaluation -> report -> traces.
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
SFT_PID=$1
OUT=runs/jev_v2
EV=data/decisions/eval.jsonl
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
while kill -0 "$SFT_PID" 2>/dev/null; do sleep 60; done
if [ ! -f $OUT/djev.json ]; then log "Jev SFT did not save $OUT, stopping"; exit 1; fi
log "Jev SFT finished"

log "speed benchmark started"
.venv/bin/python -u -m djev.bench --checkpoint runs/general_12000 --prompts data/general_12000.jsonl > logs/bench_speed.log 2>&1 \
  && log "speed benchmark finished" || log "speed benchmark FAILED (see logs/bench_speed.log)"

log "evaluation started"
{
  echo "== [4/6] evaluate Jev model"
  .venv/bin/python -u -m djev.evaluate --checkpoint $OUT --data $EV --out logs/eval_jev_v2_decisions.jsonl
  .venv/bin/python -u -m djev.evaluate --checkpoint $OUT --data data/hh_harmless_gold.jsonl --limit 1000 --out logs/eval_jev_v2_hh.jsonl
  echo "== [5/6] calibration report"
  .venv/bin/python -m djev.calibration logs/baseline_ar_decisions.jsonl logs/eval_general_decisions.jsonl \
    logs/eval_jev_v2_decisions.jsonl logs/eval_jev_v2_hh.jsonl | tee logs/report_jev_v2.jsonl
  echo "== [6/6] denoising traces"
  .venv/bin/python -u -m djev.trace --checkpoint $OUT --gold data/hh_harmless_gold.jsonl --canvas 64 --out logs/trace_jev_v2.json
} >> logs/run_jev.log 2>&1 && log "evaluation finished" || log "evaluation FAILED (see logs/run_jev.log)"

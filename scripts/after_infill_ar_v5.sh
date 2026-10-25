#!/usr/bin/env bash
# Fair control for the infill run: AR SFT on exactly the same data (decisions + many-question rows), then the same evals.
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
run() { log "start: $1"; shift; if "$@"; then log "done"; else log "FAILED: $*"; fi; }
while pgrep -f "scripts/[p]ipeline_infill.sh" > /dev/null; do sleep 30; done
cat data/decisions_mix_v4.jsonl data/many_questions_train.jsonl | shuf --random-source=<(yes) > data/decisions_mix_v5.jsonl
[ -f runs/ar_sft_v5/djev.json ] || run "AR SFT v5 (same data as infill)" .venv/bin/python -u -m djev.ar_sft --data data/decisions_mix_v5.jsonl \
  --out runs/ar_sft_v5 --steps 6000 > logs/ar_sft_v5.log 2>&1
run "many-q: AR v5 template filling" .venv/bin/python -u -m djev.infill eval --mode ar --checkpoint runs/ar_sft_v5 --data data/many_questions.jsonl --out logs/infill_manyq_ar_v5.jsonl > logs/infill_manyq_ar_v5.log 2>&1
run "decisions: AR v5 template filling" .venv/bin/python -u -m djev.infill eval --mode ar --checkpoint runs/ar_sft_v5 --data data/decisions/eval.jsonl --out logs/infill_dec_ar_v5.jsonl > logs/infill_dec_ar_v5.log 2>&1
.venv/bin/python -m djev.calibration logs/infill_dec_diffusion.jsonl logs/infill_dec_ar.jsonl logs/infill_dec_ar_v5.jsonl > logs/report_infill_v5.jsonl 2>&1
log "fair AR control v5 finished"

#!/usr/bin/env bash
# Jev downstream SFT (DG §8) on the Japanese-augmented general diffusion model (general_v3),
# then the AR SFT control on the same data, then evaluation.
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
run() { log "start: $1"; shift; if "$@"; then log "done"; else log "FAILED: $*"; exit 1; fi; }

cat data/decisions/train.jsonl data/local/decisions_20b_filtered.jsonl data/groq/decisions_gpt-oss-120b.jsonl \
  | shuf --random-source=<(yes) > data/decisions_mix_v4.jsonl
log "decision mix v4: $(wc -l < data/decisions_mix_v4.jsonl) rows"

[ -f runs/jev_v4/djev.json ] || run "Jev SFT v4 (on general_v3)" .venv/bin/python -u -m djev.sft --init runs/general_v3 \
  --data data/decisions_mix_v4.jsonl --out runs/jev_v4 --steps 6000 --prompt-len 1024 --canvas 64 --canvases 1 --batch 2 \
  --lora-rank 8 --grad-checkpointing --log-every 50 --save-every 2000 > logs/sft_jev_v4.log 2>&1
[ -f runs/ar_sft_v4/djev.json ] || run "AR SFT control v4" .venv/bin/python -u -m djev.ar_sft --data data/decisions_mix_v4.jsonl \
  --out runs/ar_sft_v4 --steps 6000 > logs/ar_sft_v4.log 2>&1

EV=data/decisions/eval.jsonl
run "eval jev_v4" .venv/bin/python -u -m djev.evaluate --checkpoint runs/jev_v4 --data $EV --out logs/eval_jev_v4_decisions.jsonl > logs/eval_jev_v4.log 2>&1
run "eval ar_sft_v4" .venv/bin/python -u -m djev.baseline_ar --checkpoint runs/ar_sft_v4 --data $EV --out logs/eval_ar_sft_v4_decisions.jsonl > logs/eval_ar_sft_v4.log 2>&1
.venv/bin/python -m djev.calibration logs/baseline_ar_decisions.jsonl logs/eval_jev_v2_decisions.jsonl \
  logs/eval_jev_v4_decisions.jsonl logs/eval_ar_sft_v4_decisions.jsonl > logs/report_v4.jsonl 2>&1
log "report written: logs/report_v4.jsonl"
run "trace jev_v4" .venv/bin/python -u -m djev.trace --checkpoint runs/jev_v4 --gold data/hh_harmless_gold.jsonl \
  --canvas 64 --out logs/trace_jev_v4.json > /dev/null 2>&1
log "pipeline jev v4 finished"

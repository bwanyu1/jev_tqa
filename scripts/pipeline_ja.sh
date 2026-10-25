#!/usr/bin/env bash
# Japanese top-up of the general diffusion SFT: Japanese prompts -> self-distillation -> continue SFT -> traces (N=192).
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
run() { log "start: $1"; shift; if "$@"; then log "done"; else log "FAILED: $*"; exit 1; fi; }
while pgrep -f "[d]jev.bench" > /dev/null; do sleep 30; done

[ -s data/ja_prompts.jsonl ] || run "Japanese prompts" .venv/bin/python -m djev.ja_prompts --out data/ja_prompts.jsonl
[ -s data/general_ja_15000.jsonl ] || run "Japanese self-distillation" env VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=. \
  .venv-vllm/bin/python -m djev.distill --prompts data/ja_prompts.jsonl --n 15000 --out data/general_ja_15000.jsonl
shuf -n 15000 --random-source=<(yes) data/general_mix.jsonl > data/general_en_15000.jsonl
cat data/general_ja_15000.jsonl data/general_en_15000.jsonl | shuf --random-source=<(yes) > data/general_mix_ja.jsonl
log "ja+en mix: $(wc -l < data/general_mix_ja.jsonl) rows"
[ -f runs/general_v3/djev.json ] || run "general diffusion SFT v3 (ja)" .venv/bin/python -u -m djev.sft --init runs/general_v2 \
  --data data/general_mix_ja.jsonl --out runs/general_v3 --steps 4000 --prompt-len 512 --canvas 256 --canvases 2 --batch 2 \
  --lora-rank 64 --no-encoder-loss --grad-checkpointing --log-every 50 --save-every 2000 > logs/sft_general_v3.log 2>&1
run "trace general_v3 N=192" .venv/bin/python -u -m djev.trace --checkpoint runs/general_v3 --gold data/hh_harmless_gold.jsonl \
  --canvas 256 --max-steps 192 --out logs/trace_general_v3_N192.json > /dev/null 2>&1
log "pipeline ja finished"

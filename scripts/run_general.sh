#!/usr/bin/env bash
# General diffusion conversion (DiffusionGemma §4): self-distilled chat data -> diffusion SFT -> traces.
set -euo pipefail
cd "$(dirname "$0")/.."
. ./${GPU_ENV:-env4060.sh}
N=${N:-12000}
STEPS=${STEPS:-3000}
mkdir -p data logs runs

if [ ! -s data/general_${N}.jsonl ]; then
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=. .venv-vllm/bin/python -m djev.distill --n "$N" --out data/general_${N}.jsonl
fi

.venv/bin/python -u -m djev.sft --data data/general_${N}.jsonl --out runs/general_${N} \
  --steps "$STEPS" --prompt-len 512 --canvas 256 --canvases 2 --batch 2 --lora-rank 64 \
  --no-encoder-loss --grad-checkpointing --log-every 50 --save-every 1000

.venv/bin/python -u -m djev.trace --checkpoint runs/general_${N} --gold data/hh_harmless_gold.jsonl \
  --canvas 256 --out logs/trace_general_${N}.json

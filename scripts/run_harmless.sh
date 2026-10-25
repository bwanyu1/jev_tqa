#!/usr/bin/env bash
# RLCD (harmlessness) -> diffusion SFT -> HH gold evaluation, on one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
. ./${GPU_ENV:-env.sh}  # env4060.sh pins the RTX 4060 Ti
N=${N:-10000}
STEPS=${STEPS:-2000}
mkdir -p data logs runs

if [ ! -s data/rlcd_harmless_${N}.jsonl ]; then
  PYTHONPATH=. .venv-vllm/bin/python -m djev.rlcd --engine vllm --gpu-mem 0.85 \
    --prompts data/hh_harmless_prefixes.jsonl --task harmless --limit "$N" \
    --out data/rlcd_harmless_${N}.jsonl
fi

.venv/bin/python -u -m djev.sft --data data/rlcd_harmless_${N}.jsonl --out runs/harmless_${N} \
  --steps "$STEPS" --prompt-len 1024 --canvas 32 --canvases 1 --batch 2 --grad-checkpointing --log-every 50

.venv/bin/python -u -m djev.baseline_ar --data data/hh_harmless_gold.jsonl | tee logs/baseline_ar_harmless.json

.venv/bin/python -u -m djev.evaluate --checkpoint runs/harmless_${N} \
  --data data/hh_harmless_gold.jsonl --out logs/eval_harmless_${N}.jsonl

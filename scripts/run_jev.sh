#!/usr/bin/env bash
# Stage 2 (DiffusionGemma §8 downstream SFT): Jev decisions on top of the general diffusion model.
set -euo pipefail
cd "$(dirname "$0")/.."
. ./${GPU_ENV:-env4060.sh}
GENERAL=${GENERAL:-runs/general_12000}
OUT=${OUT:-runs/jev_v2}
STEPS=${STEPS:-4000}
EV=data/decisions/eval.jsonl
mkdir -p logs

echo "== [1/6] AR baseline on decision eval"
.venv/bin/python -u -m djev.baseline_ar --data $EV --out logs/baseline_ar_decisions.jsonl

echo "== [2/6] general diffusion model, zero-shot on decision eval (first 600 rows)"
.venv/bin/python -u -m djev.evaluate --checkpoint $GENERAL --data $EV --limit 600 --canvas 64 --out logs/eval_general_decisions.jsonl

echo "== [3/6] Jev downstream SFT"
.venv/bin/python -u -m djev.sft --init $GENERAL --data data/decisions/train.jsonl --out $OUT \
  --steps "$STEPS" --prompt-len 1024 --canvas 64 --canvases 1 --batch 2 --lora-rank 8 \
  --grad-checkpointing --log-every 50 --save-every 1000

echo "== [4/6] evaluate Jev model"
.venv/bin/python -u -m djev.evaluate --checkpoint $OUT --data $EV --out logs/eval_jev_v2_decisions.jsonl
.venv/bin/python -u -m djev.evaluate --checkpoint $OUT --data data/hh_harmless_gold.jsonl --limit 1000 --out logs/eval_jev_v2_hh.jsonl

echo "== [5/6] calibration report"
.venv/bin/python -m djev.calibration logs/baseline_ar_decisions.jsonl logs/eval_general_decisions.jsonl \
  logs/eval_jev_v2_decisions.jsonl logs/eval_jev_v2_hh.jsonl | tee logs/report_jev_v2.jsonl

echo "== [6/6] denoising traces"
.venv/bin/python -u -m djev.trace --checkpoint $OUT --gold data/hh_harmless_gold.jsonl --canvas 64 --out logs/trace_jev_v2.json
echo "== done"

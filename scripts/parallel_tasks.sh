#!/usr/bin/env bash
# Tasks where parallel decoding should help: (1) many questions per state, (2) Sudoku (DG §8).
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
run() { log "start: $1"; shift; if "$@"; then log "done"; else log "FAILED: $*"; fi; }

# ---- 1 many questions
[ -s data/many_questions.jsonl ] || .venv/bin/python -m djev.many_questions build --out data/many_questions.jsonl
run "many-q: diffusion Jev v4" .venv/bin/python -u -m djev.many_questions eval --checkpoint runs/jev_v4 --mode diffusion \
  --data data/many_questions.jsonl --out logs/manyq_diffusion.json > logs/manyq_diffusion.log 2>&1
run "many-q: AR SFT json" .venv/bin/python -u -m djev.many_questions eval --checkpoint runs/ar_sft_v4 --mode ar_json \
  --data data/many_questions.jsonl --out logs/manyq_ar_json.json > logs/manyq_ar_json.log 2>&1
run "many-q: AR SFT readout" .venv/bin/python -u -m djev.many_questions eval --checkpoint runs/ar_sft_v4 --mode ar_readout \
  --data data/many_questions.jsonl --out logs/manyq_ar_readout.json > logs/manyq_ar_readout.log 2>&1

# ---- 2 Sudoku (Table 7 LoRA recipe: rank 8, canvas 256 x1, prompt 256, batch 2, 3e-4 -> 3e-5, 8000 steps, warmup 400)
[ -s data/sudoku_train.jsonl ] || .venv/bin/python -m djev.sudoku build --train-out data/sudoku_train.jsonl --eval-out data/sudoku_eval.jsonl
[ -f runs/sudoku_diffusion/djev.json ] || run "Sudoku diffusion SFT" .venv/bin/python -u -m djev.sft --init runs/general_v3 \
  --data data/sudoku_train.jsonl --out runs/sudoku_diffusion --steps 8000 --warmup 400 --lr 3e-4 --end-lr 3e-5 \
  --prompt-len 256 --canvas 256 --canvases 1 --batch 2 --lora-rank 8 --grad-checkpointing --log-every 100 > logs/sudoku_diffusion.log 2>&1
[ -f runs/sudoku_ar/djev.json ] || run "Sudoku AR SFT" .venv/bin/python -u -m djev.ar_sft --data data/sudoku_train.jsonl \
  --out runs/sudoku_ar --steps 8000 --warmup 400 --lr 3e-4 --end-lr 3e-5 --max-len 600 --log-every 100 > logs/sudoku_ar.log 2>&1
run "Sudoku eval: original AR" .venv/bin/python -u -m djev.sudoku eval --mode ar --data data/sudoku_eval.jsonl --out logs/sudoku_eval_base.json > logs/sudoku_eval_base.log 2>&1
run "Sudoku eval: diffusion" .venv/bin/python -u -m djev.sudoku eval --mode diffusion --checkpoint runs/sudoku_diffusion --data data/sudoku_eval.jsonl --out logs/sudoku_eval_diffusion.json > logs/sudoku_eval_diffusion.log 2>&1
run "Sudoku eval: AR SFT" .venv/bin/python -u -m djev.sudoku eval --mode ar --checkpoint runs/sudoku_ar --data data/sudoku_eval.jsonl --out logs/sudoku_eval_ar.json > logs/sudoku_eval_ar.log 2>&1
log "parallel tasks finished"

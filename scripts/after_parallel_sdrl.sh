#!/usr/bin/env bash
# After the parallel tasks (incl. Sudoku SFT + eval) finish: simplified SD·RL on Sudoku.
cd "$(dirname "$0")/.."
. ./env4060.sh
while pgrep -f "scripts/[p]arallel_tasks.sh" > /dev/null; do sleep 60; done
echo "[$(date "+%F %T")] start: Sudoku eval: general diffusion v3 (no Sudoku training)" >> logs/queue.log
.venv/bin/python -u -m djev.sudoku eval --mode diffusion --checkpoint runs/general_v3 --data data/sudoku_eval.jsonl --limit 100 --out logs/sudoku_eval_general_v3.json > logs/sudoku_eval_general_v3.log 2>&1
echo "[$(date "+%F %T")] done" >> logs/queue.log
[ -f runs/sudoku_diffusion/djev.json ] || { echo "[$(date '+%F %T')] SD·RL skipped: no Sudoku diffusion checkpoint" >> logs/queue.log; exit 1; }
echo "[$(date '+%F %T')] start: SD·RL (Sudoku)" >> logs/queue.log
.venv/bin/python -u -m djev.sdrl --init runs/sudoku_diffusion --task sudoku --data data/sudoku_train.jsonl \
  --eval-data data/sudoku_eval.jsonl --out runs/sdrl_sudoku --iterations 30 --prompts-per-iter 16 --group 4 \
  --updates-per-iter 16 --teacher-steps 192 --student-steps 48 --t-low 0.5 --eval-every 5 --eval-n 50 > logs/sdrl_sudoku.log 2>&1 \
  && echo "[$(date '+%F %T')] SD·RL finished" >> logs/queue.log || echo "[$(date '+%F %T')] SD·RL FAILED (logs/sdrl_sudoku.log)" >> logs/queue.log

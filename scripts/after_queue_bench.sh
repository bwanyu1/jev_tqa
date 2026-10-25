#!/usr/bin/env bash
# Wait for the running queue, then run the same-conditions AR vs diffusion speed benchmark.
cd "$(dirname "$0")/.."
. ./env4060.sh
while pgrep -f "bash scripts/queue.sh" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] speed benchmark started" >> logs/queue.log
.venv/bin/python -u -m djev.bench --checkpoint runs/general_12000 --prompts data/general_12000.jsonl \
  > logs/bench_speed.log 2>&1 && echo "[$(date '+%F %T')] speed benchmark finished" >> logs/queue.log \
  || echo "[$(date '+%F %T')] speed benchmark FAILED (see logs/bench_speed.log)" >> logs/queue.log

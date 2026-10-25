#!/usr/bin/env bash
# Overnight distillation with gpt-oss-20b served locally by vLLM on the 4060 Ti (port 8001).
# Generates Jev decisions and general answers concurrently, then stops the server.
set -uo pipefail
cd "$(dirname "$0")/.."
ND=${ND:-12000}
NG=${NG:-12000}
D=data/local/decisions_gpt-oss-20b.jsonl
G=data/local/general_gpt-oss-20b.jsonl
mkdir -p data/local logs
log() { echo "[$(date '+%F %T')] $*" >> logs/local_gen.log; }
log "start: decisions=$(cat $D 2>/dev/null | wc -l) general=$(cat $G 2>/dev/null | wc -l)"
.venv/bin/python -m djev.groq_distill decisions --base-url http://localhost:8001/v1 --model openai/gpt-oss-20b \
  --n "$ND" --workers 12 --seed 20261021 --out $D >> logs/local_gen.log 2>&1 &
.venv/bin/python -m djev.groq_distill general --base-url http://localhost:8001/v1 --model openai/gpt-oss-20b \
  --prompts data/groq/ultrachat_prompts_extra.jsonl --skip 19000 --n "$NG" --workers 12 --max-tokens 900 \
  --out $G >> logs/local_gen.log 2>&1
wait
log "end: decisions=$(wc -l < $D) general=$(wc -l < $G)"
for p in $(pgrep -f "bin/[v]llm serve"); do kill $p; done
log "vLLM server stopped"

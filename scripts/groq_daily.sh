#!/usr/bin/env bash
# Spend each day's Groq free quota: gpt-oss-120b -> Jev decisions, gpt-oss-20b -> general answers.
# Re-run daily (e.g. from cron); each run resumes after the rows already collected.
set -uo pipefail
cd "$(dirname "$0")/.."
. ./secrets.sh
mkdir -p data/groq logs
log() { echo "[$(date '+%F %T')] $*" >> logs/groq.log; }
D=data/groq/decisions_gpt-oss-120b.jsonl
G=data/groq/general_gpt-oss-20b.jsonl
nd=$(cat $D 2>/dev/null | wc -l); ng=$(cat $G 2>/dev/null | wc -l)
log "start: decisions=$nd general=$ng"
.venv/bin/python -m djev.groq_distill decisions --model openai/gpt-oss-120b --n 1000 --seed "$(date +%Y%m%d)" --out $D >> logs/groq.log 2>&1 &
# general prompts: UltraChat prompts not used by the self-distillation set, resumed by count
.venv/bin/python - <<PY
import json, os
from djev.distill import ultrachat_prompts
p = "data/groq/ultrachat_prompts_extra.jsonl"
if not os.path.exists(p):
    rows = ultrachat_prompts(40000, 1)
    used = {json.loads(l)["messages"][0]["content"] for l in open("data/general_12000.jsonl")}
    with open(p, "w") as f:
        for r in rows:
            if r not in used:
                f.write(json.dumps({"messages": [{"role": "user", "content": r}]}) + "\n")
PY
.venv/bin/python -m djev.groq_distill general --model openai/gpt-oss-20b --prompts data/groq/ultrachat_prompts_extra.jsonl \
  --skip "$ng" --n 1000 --max-tokens 900 --out $G >> logs/groq.log 2>&1
wait
log "end: decisions=$(wc -l < $D) general=$(wc -l < $G)"

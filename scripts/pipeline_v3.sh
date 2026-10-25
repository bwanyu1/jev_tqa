#!/usr/bin/env bash
# v3 queue on the 4060 Ti:
#  1 filter gpt-oss-20b decision labels by self-consistency (re-answer with more reasoning)
#  2 continue the general diffusion SFT (DG §4) on self-distilled + gpt-oss answers
#  3 Jev downstream SFT (DG §8) on public + filtered gpt-oss decisions
#  4 control: ordinary AR SFT on the same decision data
#  5 evaluation, calibration report, traces
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
run() { log "start: $1"; shift; if "$@"; then log "done"; else log "FAILED: $*"; exit 1; fi; }

# ---------------------------------------------------------------- 1 filter
if [ ! -s data/local/decisions_20b_filtered.jsonl ]; then
  log "start: vLLM gpt-oss-20b"
  (VLLM_USE_FLASHINFER_SAMPLER=0 .venv-vllm/bin/vllm serve openai/gpt-oss-20b --port 8001 --gpu-memory-utilization 0.925 \
     --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 2048 --enforce-eager > logs/vllm_gptoss20b.log 2>&1 &)
  for i in $(seq 1 90); do curl -sf localhost:8001/health > /dev/null && break; sleep 10; done
  run "self-consistency filter" .venv/bin/python -m djev.judge --data data/local/decisions_gpt-oss-20b.jsonl --n 1000000 \
      --base-url http://localhost:8001/v1 --model openai/gpt-oss-20b --effort medium --workers 12 \
      --out logs/selfcheck_20b.jsonl --filter-out data/local/decisions_20b_filtered.jsonl
  for p in $(pgrep -f "bin/[v]llm serve"); do kill $p; done
  sleep 15
fi
.venv/bin/python - <<'PY' | tee -a logs/queue.log
import json
self = {json.dumps(json.loads(l)["row"]["state"]): json.loads(l) for l in open("logs/selfcheck_20b.jsonl")}
kept = agree = dropped = dropped_wrong = 0
for l in open("logs/judge_20b_by_120b.jsonl"):
    j = json.loads(l); s = self.get(json.dumps(j["row"]["state"]))
    if not s: continue
    for qid, ok120 in j["agree"].items():
        if s["agree"].get(qid): kept += 1; agree += ok120
        else: dropped += 1; dropped_wrong += not ok120
print(f"filter check vs 120B: kept {kept} questions, {agree/max(kept,1):.3f} agree with 120B; dropped {dropped}, of which {dropped_wrong/max(dropped,1):.3f} disagreed with 120B")
PY

# ---------------------------------------------------------------- 2 general diffusion SFT
cat data/general_12000.jsonl data/local/general_gpt-oss-20b.jsonl data/groq/general_gpt-oss-20b.jsonl | shuf --random-source=<(yes) > data/general_mix.jsonl
log "general mix: $(wc -l < data/general_mix.jsonl) rows"
[ -f runs/general_v2/djev.json ] || run "general diffusion SFT v2" .venv/bin/python -u -m djev.sft --init runs/general_12000 \
  --data data/general_mix.jsonl --out runs/general_v2 --steps 6000 --prompt-len 512 --canvas 256 --canvases 2 --batch 2 \
  --lora-rank 64 --no-encoder-loss --grad-checkpointing --log-every 50 --save-every 2000 > logs/sft_general_v2.log 2>&1
run "trace general_v2" .venv/bin/python -u -m djev.trace --checkpoint runs/general_v2 --gold data/hh_harmless_gold.jsonl \
  --canvas 256 --out logs/trace_general_v2.json > /dev/null 2>&1

# ---------------------------------------------------------------- 3 Jev SFT
cat data/decisions/train.jsonl data/local/decisions_20b_filtered.jsonl data/groq/decisions_gpt-oss-120b.jsonl \
  | shuf --random-source=<(yes) > data/decisions_mix.jsonl
log "decision mix: $(wc -l < data/decisions_mix.jsonl) rows"
[ -f runs/jev_v3/djev.json ] || run "Jev SFT v3" .venv/bin/python -u -m djev.sft --init runs/general_v2 --data data/decisions_mix.jsonl \
  --out runs/jev_v3 --steps 6000 --prompt-len 1024 --canvas 64 --canvases 1 --batch 2 --lora-rank 8 \
  --grad-checkpointing --log-every 50 --save-every 2000 > logs/sft_jev_v3.log 2>&1

# ---------------------------------------------------------------- 4 AR control
[ -f runs/ar_sft_v3/djev.json ] || run "AR SFT control" .venv/bin/python -u -m djev.ar_sft --data data/decisions_mix.jsonl \
  --out runs/ar_sft_v3 --steps 6000 > logs/ar_sft_v3.log 2>&1

# ---------------------------------------------------------------- 5 evaluation
EV=data/decisions/eval.jsonl
run "eval jev_v3" .venv/bin/python -u -m djev.evaluate --checkpoint runs/jev_v3 --data $EV --out logs/eval_jev_v3_decisions.jsonl > logs/eval_jev_v3.log 2>&1
run "eval ar_sft_v3" .venv/bin/python -u -m djev.baseline_ar --checkpoint runs/ar_sft_v3 --data $EV --out logs/eval_ar_sft_v3_decisions.jsonl > logs/eval_ar_sft_v3.log 2>&1
.venv/bin/python -m djev.calibration logs/baseline_ar_decisions.jsonl logs/eval_jev_v2_decisions.jsonl \
  logs/eval_jev_v3_decisions.jsonl logs/eval_ar_sft_v3_decisions.jsonl > logs/report_v3.jsonl 2>&1
log "report written: logs/report_v3.jsonl"
run "trace jev_v3" .venv/bin/python -u -m djev.trace --checkpoint runs/jev_v3 --gold data/hh_harmless_gold.jsonl \
  --canvas 64 --out logs/trace_jev_v3.json > /dev/null 2>&1
log "pipeline v3 finished"

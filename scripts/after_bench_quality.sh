#!/usr/bin/env bash
# Wait for the N sweep, then judge every saved text with a local gpt-oss-20b and stop the server.
cd "$(dirname "$0")/.."
. ./env4060.sh
while pgrep -f "[d]jev.bench" > /dev/null; do sleep 30; done
(VLLM_USE_FLASHINFER_SAMPLER=0 .venv-vllm/bin/vllm serve openai/gpt-oss-20b --port 8001 --gpu-memory-utilization 0.925 \
   --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 2048 --enforce-eager > logs/vllm_gptoss20b.log 2>&1 &)
for i in $(seq 1 90); do curl -sf localhost:8001/health > /dev/null && break; sleep 10; done
.venv/bin/python -m djev.quality logs/bench_sweep_v3.json --out logs/quality_sweep_v3.json > logs/quality_sweep_v3.log 2>&1
for p in $(pgrep -f "bin/[v]llm serve"); do kill $p; done
echo "[$(date '+%F %T')] quality sweep finished" >> logs/queue.log

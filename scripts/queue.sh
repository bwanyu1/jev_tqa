#!/usr/bin/env bash
# Run the pipeline stages one after another on the 4060 Ti; each waits for the previous one.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
while pgrep -f "scripts/run_general.sh" > /dev/null; do sleep 60; done
log "general stage finished: $(grep -a 'saved to' logs/run_general.log | tail -1)"
if [ ! -f runs/general_12000/djev.json ]; then log "general checkpoint missing, stopping"; exit 1; fi
log "starting Jev stage"
if scripts/run_jev.sh > logs/run_jev.log 2>&1; then log "Jev stage finished"; else log "Jev stage FAILED (see logs/run_jev.log)"; fi

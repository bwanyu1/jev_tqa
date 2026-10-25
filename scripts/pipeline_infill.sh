#!/usr/bin/env bash
# Fill-in-the-blank Jev: many-question data -> blank-only diffusion SFT -> evaluation vs AR template filling.
set -uo pipefail
cd "$(dirname "$0")/.."
. ./env4060.sh
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }
run() { log "start: $1"; shift; if "$@"; then log "done"; else log "FAILED: $*"; fi; }

[ -s data/many_questions_train.jsonl ] || run "build many-question train rows" .venv/bin/python -m djev.infill build-many --out data/many_questions_train.jsonl
[ -f runs/jev_infill/djev.json ] || run "infill Jev SFT (on general_v3)" .venv/bin/python -u -m djev.infill train --init runs/general_v3 \
  --data data/decisions_mix_v4.jsonl data/many_questions_train.jsonl --out runs/jev_infill --steps 6000 > logs/sft_jev_infill.log 2>&1

M=data/many_questions.jsonl; EV=data/decisions/eval.jsonl
run "many-q: infill diffusion"      .venv/bin/python -u -m djev.infill eval --mode diffusion --checkpoint runs/jev_infill --data $M --out logs/infill_manyq_diffusion.jsonl > logs/infill_manyq_diffusion.log 2>&1
run "many-q: AR template filling"   .venv/bin/python -u -m djev.infill eval --mode ar --checkpoint runs/ar_sft_v4 --data $M --out logs/infill_manyq_ar.jsonl > logs/infill_manyq_ar.log 2>&1
run "many-q: v4 + infill sampler"   .venv/bin/python -u -m djev.infill eval --mode diffusion --checkpoint runs/jev_v4 --data $M --out logs/infill_manyq_v4.jsonl > logs/infill_manyq_v4.log 2>&1
run "decisions: infill diffusion"   .venv/bin/python -u -m djev.infill eval --mode diffusion --checkpoint runs/jev_infill --data $EV --out logs/infill_dec_diffusion.jsonl > logs/infill_dec_diffusion.log 2>&1
run "decisions: AR template filling" .venv/bin/python -u -m djev.infill eval --mode ar --checkpoint runs/ar_sft_v4 --data $EV --out logs/infill_dec_ar.jsonl > logs/infill_dec_ar.log 2>&1
.venv/bin/python -m djev.calibration logs/infill_dec_diffusion.jsonl logs/infill_dec_ar.jsonl > logs/report_infill.jsonl 2>&1
log "pipeline infill finished"

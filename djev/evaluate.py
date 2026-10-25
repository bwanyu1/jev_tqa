"""Evaluate a decision checkpoint on Choice/Noul/Score examples (JSONL in the SFT format).

On HH gold pairs (djev.hh --gold-out) this is the preference-model evaluation of RLCD
§5.1 / Table 5: binary accuracy and the average probability assigned to the human-preferred
response. Also reports parse failures and DiffusionGemma §3.4 efficiency metrics.
"""
from __future__ import annotations

import argparse
import json
import time

from .decide import decide
from .io import load
from .sampler import SamplerConfig
from .sft import parse_questions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--canvas", type=int, default=None)
    ap.add_argument("--out", default=None, help="optional JSONL of per-example results")
    a = ap.parse_args(argv)

    model, tok, proc = load(a.backbone, a.checkpoint)
    cfg = SamplerConfig()
    if a.checkpoint:
        from pathlib import Path

        meta = Path(a.checkpoint, "djev.json")
        if meta.exists():
            cfg.canvas_length = json.loads(meta.read_text())["canvas_length"]
    if a.canvas:
        cfg.canvas_length = a.canvas

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][: a.limit]
    n = correct = parsed = with_prob = 0
    p_gold = steps = 0.0
    secs = []
    sink = open(a.out, "w", encoding="utf-8") if a.out else None
    for r in rows:
        qs = parse_questions(r["questions"])
        t0 = time.perf_counter()
        res = decide(model, tok, r["state"], qs, cfg, media=r.get("media"), processor=proc)
        secs.append(time.perf_counter() - t0)
        steps += res.generation.total_steps
        for q in qs:
            gold = str(r["answers"][q.id]).lower()
            d = res.decisions[q.id]
            n += 1
            parsed += d.answer is not None
            if d.probs:
                with_prob += 1
                p_gold += d.probs.get(gold, 0.0)
                correct += max(d.probs, key=d.probs.get) == gold
            else:
                correct += d.answer == gold
        if sink:
            sink.write(json.dumps({"raw": res.raw, "gold": r["answers"], "source": r.get("source"),
                                   "decisions": {k: vars(v) for k, v in res.decisions.items()}}, ensure_ascii=False) + "\n")
    secs.sort()
    print(json.dumps({
        "questions": n,
        "accuracy": correct / max(n, 1),
        "avg_prob_gold": p_gold / max(with_prob, 1),
        "parsed": parsed / max(n, 1),
        "with_probability": with_prob / max(n, 1),
        "avg_denoising_steps_per_request": steps / max(len(rows), 1),
        "latency_p50_s": secs[len(secs) // 2] if secs else None,
        "latency_p90_s": secs[int(len(secs) * 0.9)] if secs else None,
    }, indent=2))


if __name__ == "__main__":
    main()

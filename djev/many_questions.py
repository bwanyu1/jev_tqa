"""Many questions about one state: where parallel (diffusion) decoding should pay off.
GoEmotions comments with k yes/no emotion questions each (k = 1, 2, 4, 8, 16); every system
answers all k questions of a row in one request.

  build : rows per k from the GoEmotions validation split (not used in training)
  eval  : --mode diffusion  (Jev diffusion checkpoint: one canvas per 64 tokens, all answers in parallel)
          --mode ar_json    (AR checkpoint writes the whole JSON answer token by token, KV-cached)
          --mode ar_readout (AR checkpoint: one forward per question, reading the yes/no logits)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

import torch

from . import schema
from .decision_data import EMO, table
from .sft import parse_questions


def build(a):
    rng = random.Random(0)
    rows = [r for r in table("google-research-datasets/go_emotions", "validation", "simplified") if any(l != 27 for l in r["labels"])]
    out = []
    for k in [int(x) for x in a.ks.split(",")]:
        for r in rng.sample(rows, a.per_k):
            pos = [l for l in r["labels"] if l != 27]
            neg = [l for l in range(27) if l not in r["labels"]]
            take = pos[:k] + rng.sample(neg, max(0, k - len(pos[:k])))
            rng.shuffle(take)
            out.append({"k": k, "state": "Comment: " + r["text"],
                        "questions": [{"type": "noul", "id": EMO[l], "question": f"Does the comment express {EMO[l]}?"} for l in take],
                        "answers": {EMO[l]: "true" if l in r["labels"] else "false" for l in take}})
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(out))


@torch.no_grad()
def evaluate(a):
    from .bench import sync
    from .decide import decide
    from .io import load
    from .sampler import SamplerConfig
    from .baseline_ar import candidate_tokens

    model, tok, _ = load(a.backbone, a.checkpoint)
    stops = list(schema.stop_ids(tok))
    rows = [json.loads(l) for l in open(a.data)]
    res = {}
    for r in rows:
        qs = parse_questions(r["questions"])
        gold = r["answers"]
        sync(); t0 = time.perf_counter()
        if a.mode == "diffusion":
            d = decide(model, tok, r["state"], qs, SamplerConfig(canvas_length=64, max_canvases=8, max_steps=a.max_steps),
                       readout="first")
            pred = {q.id: (max(d.decisions[q.id].probs, key=d.decisions[q.id].probs.get) if d.decisions[q.id].probs
                           else d.decisions[q.id].answer) for q in qs}
            forwards = d.generation.total_steps + len(d.generation.canvases) - 1
        elif a.mode == "ar_json":
            ids = torch.tensor([schema.prompt_ids(tok, r["state"], qs)], device=model.device)
            o = model.backbone.generate(input_ids=ids, max_new_tokens=16 + 12 * len(qs), do_sample=False,
                                        eos_token_id=stops, pad_token_id=tok.pad_token_id or stops[0])
            text = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
            pred = {}
            for q in qs:
                m = re.search(r'"%s"\s*:\s*"?(true|false)' % re.escape(q.id), text)
                pred[q.id] = m.group(1) if m else None
            forwards = o.shape[1] - ids.shape[1]
        else:
            prompt = schema.prompt_ids(tok, r["state"], qs)
            pred = {}
            for q in qs:
                ids, cand = candidate_tokens(tok, prompt, q)
                logits = model.backbone(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1]
                pred[q.id] = max(cand, key=lambda v: float(logits[cand[v]]))
            forwards = len(qs)
        sync(); dt = time.perf_counter() - t0
        s = res.setdefault(r["k"], {"rows": 0, "q": 0, "correct": 0, "all": 0, "secs": [], "forwards": 0})
        ok = [pred.get(q.id) == gold[q.id] for q in qs]
        s["rows"] += 1; s["q"] += len(ok); s["correct"] += sum(ok); s["all"] += all(ok)
        s["secs"].append(dt); s["forwards"] += forwards
    summary = {}
    for k, s in sorted(res.items()):
        secs = sorted(s["secs"])
        summary[k] = {"accuracy": round(s["correct"] / s["q"], 4), "all_correct": round(s["all"] / s["rows"], 4),
                      "latency_p50_s": round(secs[len(secs) // 2], 3), "latency_mean_s": round(sum(secs) / len(secs), 3),
                      "forwards_per_row": round(s["forwards"] / s["rows"], 2)}
    print(json.dumps({"mode": a.mode, "checkpoint": a.checkpoint, "by_k": summary}, indent=1))
    if a.out:
        json.dump({"mode": a.mode, "checkpoint": a.checkpoint, "by_k": summary}, open(a.out, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True); b.add_argument("--ks", default="1,2,4,8,16"); b.add_argument("--per-k", type=int, default=100)
    e = sub.add_parser("eval")
    e.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct"); e.add_argument("--checkpoint", required=True)
    e.add_argument("--mode", choices=["diffusion", "ar_json", "ar_readout"], required=True)
    e.add_argument("--data", required=True); e.add_argument("--max-steps", type=int, default=192); e.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    build(a) if a.cmd == "build" else evaluate(a)


if __name__ == "__main__":
    main()

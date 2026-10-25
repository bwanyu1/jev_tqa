"""Record denoising trajectories (DiffusionGemma Fig. 22 style) for a few prompts as JSON."""
from __future__ import annotations

import argparse
import json

import torch

from . import schema
from .io import load
from .sampler import SamplerConfig, generate
from .sft import parse_questions


def run(model, tok, ids, cfg, seed):
    g = torch.Generator(device=model.device).manual_seed(seed)
    trace = []
    gen = generate(model, torch.tensor([ids]), schema.stop_ids(tok), cfg, g, trace=trace)
    piece = lambda i: tok.decode([i])
    canvases = []
    for steps in trace:
        canvases.append([{**s, "input": [piece(i) for i in s["input"]], "pred": [piece(i) for i in s["pred"]]}
                         for s in steps])
    return {"output": tok.decode(gen.tokens), "steps_per_canvas": [c.steps for c in gen.canvases], "canvases": canvases}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--checkpoint", default=None, help="omit to trace the freshly converted, untrained model")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--canvas", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=192, help="N, the denoising budget per canvas")
    a = ap.parse_args(argv)
    C, N = a.canvas, a.max_steps
    model, tok, _ = load(a.backbone, a.checkpoint)
    out = []

    ex = json.loads(open(a.gold, encoding="utf-8").readline())
    qs = parse_questions(ex["questions"])
    out.append({"title": "HH gold pair (trained format)", "state": ex["state"], "questions": ex["questions"],
                "gold": ex["answers"], **run(model, tok, schema.prompt_ids(tok, ex["state"], qs), SamplerConfig(canvas_length=C, max_steps=N), 0)})

    qs_raw = [{"type": "noul", "id": "urgent", "question": "Is the ticket urgent?"},
              {"type": "score", "id": "anger", "question": "How angry is the customer?", "levels": ["calm", "annoyed", "angry"]},
              {"type": "choice", "id": "route", "question": "Who should handle it?", "options": ["bot", "human agent", "engineer"]}]
    state = "Support ticket: my invoice was charged twice and I need a refund today, this is unacceptable!"
    out.append({"title": "Multi-question decision (not trained)", "state": state, "questions": qs_raw,
                **run(model, tok, schema.prompt_ids(tok, state, parse_questions(qs_raw)), SamplerConfig(canvas_length=C, max_steps=N), 0)})

    for title, text in [("Free text (DiffusionGemma Fig. 22 prompt)", "Please write a short sentence describing text diffusion."),
                        ("Free text (explanation)", "Explain why the sky is blue in three sentences."),
                        ("Free text (Japanese)", "拡散言語モデルとは何か、二文で説明してください。")]:
        msgs = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        out.append({"title": title, "state": text, **run(model, tok, ids, SamplerConfig(canvas_length=C, max_canvases=3, max_steps=N), 0)})

    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    for o in out:
        print(o["title"], o["steps_per_canvas"], repr(o["output"][:200]))


if __name__ == "__main__":
    main()

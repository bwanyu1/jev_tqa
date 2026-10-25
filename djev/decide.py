"""Jev-like entry point: decide(state, questions) -> typed answers with probabilities.

Generation is plain DiffusionGemma sampling (Algorithm 1). The probability read-out is an
extension that neither paper defines: at the token that carries each answer we take the
untempered decoder distribution softmax(L_t) of the final denoising step and renormalise it
over the allowed values.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from . import schema
from .model import DiffusionLM
from .sampler import GenerationResult, SamplerConfig, generate


@dataclass
class Decision:
    id: str
    answer: str | None  # one of q.values(), None if unparsable
    probs: dict[str, float] | None  # value -> probability, None if not readable
    probs_by_readout: dict[str, dict[str, float]] | None = None  # "final" / "first" / "mean"


@dataclass
class DecideResult:
    decisions: dict[str, Decision]
    raw: str
    generation: GenerationResult


def _token_spans(tokenizer, tokens: list[int]) -> list[tuple[int, int]]:
    spans, prev = [], ""
    for i in range(len(tokens)):
        cur = tokenizer.decode(tokens[: i + 1])
        spans.append((len(prev), len(cur)))
        prev = cur
    return spans


READOUTS = ("final", "first", "mean")


def _readout(tokenizer, gen: GenerationResult, text: str, q: schema.Question, canvas_len: int,
             readout: str = "final") -> Decision:
    m = re.search(r'"%s"\s*:\s*(true|false|\d)' % re.escape(q.id), text)
    if not m or m.group(1) not in q.values():
        return Decision(q.id, None, None)
    answer, char = m.group(1), m.start(1)
    spans = _token_spans(tokenizer, gen.tokens)
    pos = next((i for i, (a, b) in enumerate(spans) if a <= char < b), None)
    if pos is None:
        return Decision(q.id, answer, None)
    a, b = spans[pos]
    piece = text[a:b]
    if piece.strip() != answer:  # the value shares its token with other characters
        return Decision(q.id, answer, None)
    cand = {}
    for v in q.values():
        ids = tokenizer(piece.replace(answer, v), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            return Decision(q.id, answer, None)
        cand[v] = ids[0]
    canvas, i = gen.canvases[pos // canvas_len], pos % canvas_len
    dists = {"final": torch.softmax(canvas.logits[i], -1)}
    if canvas.first_logits is not None:
        dists["first"] = torch.softmax(canvas.first_logits[i], -1)
    if canvas.mean_probs is not None:
        dists["mean"] = canvas.mean_probs[i]
    by = {}
    for name, dist in dists.items():
        p = dist[list(cand.values())]
        p = p / p.sum()
        by[name] = {v: float(x) for v, x in zip(cand, p)}
    return Decision(q.id, answer, by.get(readout, by["final"]), by)


def decide(
    model: DiffusionLM,
    tokenizer,
    state: str,
    questions: list[schema.Question],
    cfg: SamplerConfig = SamplerConfig(),
    generator=None,
    media: list[dict] | None = None,
    processor=None,
    readout: str = "final",
) -> DecideResult:
    """`media`: [{"type": "video" | "image", "path": ...}] shown to the model before the state."""
    ids, mm = schema.prompt_inputs(tokenizer, processor, state, questions, media)
    gen = generate(model, torch.tensor([ids]), schema.stop_ids(tokenizer), cfg, generator, mm)
    text = tokenizer.decode(gen.tokens)
    out = {q.id: _readout(tokenizer, gen, text, q, cfg.canvas_length, readout) for q in questions}
    return DecideResult(out, text, gen)


def parse_json(text: str) -> dict | None:
    try:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except ValueError:
        return None


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Run one decision request")
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--request", required=True, help='JSON file: {"state": ..., "questions": [...], "media": [...]}')
    ap.add_argument("--canvas", type=int, default=None)
    a = ap.parse_args(argv)

    from .io import load
    from .sft import parse_questions

    model, tok, proc = load(a.backbone, a.checkpoint)
    req = json.load(open(a.request, encoding="utf-8"))
    cfg = SamplerConfig()
    meta = Path(a.checkpoint or "", "djev.json")
    if a.checkpoint and meta.exists():
        cfg.canvas_length = json.loads(meta.read_text())["canvas_length"]
    if a.canvas:
        cfg.canvas_length = a.canvas
    res = decide(model, tok, req["state"], parse_questions(req["questions"]), cfg,
                 media=req.get("media"), processor=proc)
    print(json.dumps({
        "decisions": {k: vars(v) for k, v in res.decisions.items()},
        "raw": res.raw,
        "denoising_steps": [c.steps for c in res.generation.canvases],
        "tokens_per_forward": res.generation.tokens_per_forward,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

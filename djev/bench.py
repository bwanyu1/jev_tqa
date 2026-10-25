"""Same-conditions speed comparison: AR decoding of the original backbone vs diffusion sampling
of a converted checkpoint (one GPU, batch size 1, same prompts, same token budget).

Reports per request latency, output tokens/s, and tokens per forward pass (DiffusionGemma
§3.4, Eq. 10). Output tokens are counted up to the first end-of-turn token.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from . import schema
from .io import _backbone, load, pick_device
from .sampler import SamplerConfig, generate


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prompt_ids(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                   enable_thinking=False, return_dict=False)


@torch.no_grad()
def bench_ar(backbone, tok, prompts, max_new):
    device = pick_device()
    model = _backbone(backbone, torch.bfloat16 if device == "cuda" else torch.float32, device).eval()
    stops = list(schema.stop_ids(tok))
    rows = []
    for i, p in enumerate(prompts):
        ids = torch.tensor([prompt_ids(tok, p)], device=device)
        sync(); t0 = time.perf_counter()
        out = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False, eos_token_id=stops,
                             pad_token_id=tok.pad_token_id or stops[0])
        sync(); dt = time.perf_counter() - t0
        new = out[0, ids.shape[1]:].tolist()
        n = next((j for j, t in enumerate(new) if t in stops), len(new))
        rows.append({"latency": dt, "tokens": n, "forwards": len(new), "text": tok.decode(new[:n])})
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows


@torch.no_grad()
def bench_diffusion(model, tok, prompts, canvas, max_canvases, steps):
    cfg = SamplerConfig(canvas_length=canvas, max_canvases=max_canvases, max_steps=steps)
    stops = schema.stop_ids(tok)
    rows = []
    for p in prompts:
        ids = torch.tensor([prompt_ids(tok, p)])
        sync(); t0 = time.perf_counter()
        gen = generate(model, ids, stops, cfg)
        sync(); dt = time.perf_counter() - t0
        k = len(gen.canvases)
        rows.append({"latency": dt, "tokens": len(gen.tokens), "forwards": gen.total_steps + k - 1,
                     "text": tok.decode(gen.tokens)})
    return rows


def summary(rows):
    tok = sum(r["tokens"] for r in rows)
    lat = sum(r["latency"] for r in rows)
    fw = sum(r["forwards"] for r in rows)
    return {"requests": len(rows), "latency_p50_s": round(statistics.median(r["latency"] for r in rows), 3),
            "tokens_per_s": round(tok / lat, 1), "tokens_per_forward": round(tok / max(fw, 1), 2),
            "avg_output_tokens": round(tok / len(rows), 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL with chat rows ({'messages': [...]})")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--canvas", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=192)
    ap.add_argument("--steps-list", default=None, help="comma-separated N values to sweep, e.g. 16,32,48,96,192")
    ap.add_argument("--out", default=None, help="JSON with summaries and every generated text")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.backbone)
    prompts = [json.loads(l)["messages"][0]["content"] for l in open(a.prompts, encoding="utf-8")][: a.n + 2]
    warm, prompts = prompts[:2], prompts[2:]
    bench_ar(a.backbone, tok, warm, 16)
    ar_rows = bench_ar(a.backbone, tok, prompts, a.max_new)
    canvases = -(-a.max_new // a.canvas)
    model, dtok, _ = load(a.backbone, a.checkpoint)
    bench_diffusion(model, dtok, warm, a.canvas, 1, 4)
    steps = [int(x) for x in a.steps_list.split(",")] if a.steps_list else [a.max_steps]
    df_rows = {n: bench_diffusion(model, dtok, prompts, a.canvas, canvases, n) for n in steps}
    result = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "max_new_tokens": a.max_new, "checkpoint": a.checkpoint, "ar": summary(ar_rows),
              "diffusion": {str(n): summary(r) for n, r in df_rows.items()}}
    print(json.dumps(result, indent=2))
    if a.out:
        result["prompts"] = prompts
        result["texts"] = {"ar": [r["text"] for r in ar_rows], **{f"N{n}": [r["text"] for r in rows] for n, rows in df_rows.items()}}
        json.dump(result, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

"""Sudoku, the diffusion-friendly case study of DiffusionGemma §8 (prompt format of App. C, Fig. 18).
  build : train chat rows + held-out eval puzzles from Ritvik19/Sudoku-Dataset (Apache-2.0)
  eval  : solve the eval puzzles with a diffusion checkpoint or an AR (SFT) checkpoint
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import time
import urllib.request

import pyarrow.parquet as pq
import torch

SYSTEM = ("Solve the following Sudoku puzzle. Empty cells are represented by 0. Output ONLY the solved puzzle "
          "immediately as a 9x9 grid of numbers separated by spaces. Do not include ####, explanations, or any other text.")
URL = "https://huggingface.co/api/datasets/Ritvik19/Sudoku-Dataset/parquet/default/{}/0.parquet"


def grid(s: str) -> str:
    return "\n".join(" ".join(s[r * 9:(r + 1) * 9]) for r in range(9))


def messages(puzzle: str):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": " ".join(puzzle)}]


def parse(text: str) -> str:
    return "".join(re.findall(r"[1-9]", text))[:81]


def build(a):
    rng = random.Random(0)
    keep = lambda r: a.min_missing <= r["missing"] <= a.max_missing
    tr = [r for r in pq.read_table(io.BytesIO(urllib.request.urlopen(URL.format("train")).read())).to_pylist() if keep(r)]
    ev = [r for r in pq.read_table(io.BytesIO(urllib.request.urlopen(URL.format("validation")).read())).to_pylist() if keep(r)]
    tr, ev = rng.sample(tr, min(a.n_train, len(tr))), rng.sample(ev, min(a.n_eval, len(ev)))
    with open(a.train_out, "w") as f:
        for r in tr:
            f.write(json.dumps({"messages": messages(r["puzzle"]) + [{"role": "assistant", "content": grid(r["solution"])}]}) + "\n")
    with open(a.eval_out, "w") as f:
        for r in ev:
            f.write(json.dumps({"puzzle": r["puzzle"], "solution": r["solution"], "missing": r["missing"]}) + "\n")
    print("train", len(tr), "eval", len(ev))


@torch.no_grad()
def evaluate(a):
    from .bench import sync
    from .io import load
    from .sampler import SamplerConfig, generate
    from . import schema

    model, tok, _ = load(a.backbone, a.checkpoint)
    rows = [json.loads(l) for l in open(a.data)][: a.limit]
    stops = schema.stop_ids(tok)
    out, solved, cells, steps, secs = [], 0, 0, [], []
    for r in rows:
        ids = tok.apply_chat_template(messages(r["puzzle"]), tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False, return_dict=False)
        sync(); t0 = time.perf_counter()
        if a.mode == "diffusion":
            g = generate(model, torch.tensor([ids]), stops, SamplerConfig(canvas_length=256, max_canvases=1, max_steps=a.max_steps))
            text = tok.decode(g.tokens); steps.append(g.total_steps)
        else:
            x = torch.tensor([ids], device=model.device)
            o = model.backbone.generate(input_ids=x, max_new_tokens=200, do_sample=False, eos_token_id=list(stops),
                                        pad_token_id=tok.pad_token_id or list(stops)[0])
            text = tok.decode(o[0, x.shape[1]:], skip_special_tokens=True); steps.append(o.shape[1] - x.shape[1])
        sync(); secs.append(time.perf_counter() - t0)
        pred = parse(text)
        blanks = [i for i, c in enumerate(r["puzzle"]) if c == "0"]
        ok_cells = sum(1 for i in blanks if i < len(pred) and pred[i] == r["solution"][i])
        solved += pred == r["solution"]
        cells += ok_cells / len(blanks)
        out.append({"puzzle": r["puzzle"], "solution": r["solution"], "pred": pred, "text": text, "seconds": secs[-1], "forwards": steps[-1]})
    secs.sort()
    res = {"mode": a.mode, "checkpoint": a.checkpoint, "puzzles": len(rows), "solved": solved / len(rows),
           "blank_cell_accuracy": cells / len(rows), "latency_p50_s": secs[len(secs) // 2],
           "avg_forwards": sum(steps) / len(steps)}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump({"summary": res, "rows": out}, open(a.out, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train-out", required=True); b.add_argument("--eval-out", required=True)
    b.add_argument("--n-train", type=int, default=20000); b.add_argument("--n-eval", type=int, default=200)
    b.add_argument("--min-missing", type=int, default=40); b.add_argument("--max-missing", type=int, default=60)
    e = sub.add_parser("eval")
    e.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct"); e.add_argument("--checkpoint", default=None)
    e.add_argument("--mode", choices=["diffusion", "ar"], required=True); e.add_argument("--data", required=True)
    e.add_argument("--limit", type=int, default=200); e.add_argument("--max-steps", type=int, default=192)
    e.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    build(a) if a.cmd == "build" else evaluate(a)


if __name__ == "__main__":
    main()

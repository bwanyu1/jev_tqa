"""Jev decisions as fill-in-the-blank: the answer JSON is a fixed template and only the value
of each question is denoised (inpainting on DiffusionGemma's multinomial diffusion: the template
tokens are put back after every step, so the sampler only ever changes the blanks).

  build-many : training rows with many questions per state (GoEmotions train, 8-16 questions)
  train      : diffusion SFT with blank-only noise and loss (djev.sft.diffusion_loss with slot masks)
  eval       : --mode diffusion (all blanks in parallel) | ar (same template, blanks filled left to
               right with a KV cache: one forward per blank)
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from . import schema
from .model import Context, DiffusionLM
from .sampler import SamplerConfig, _entropy, accepted_positions
from .sft import collate, cosine, diffusion_loss, encode_example, parse_questions


# ----------------------------------------------------------------------------- sampling
@torch.no_grad()
def denoise_infill(model: DiffusionLM, ctx: Context, template, slots, cfg: SamplerConfig, generator=None):
    """Algorithm 1 restricted to the blanks. Returns (argmax tokens at the blanks, first-step logits
    at the blanks, final logits at the blanks, steps)."""
    C, dev = cfg.canvas_length, model.device
    canvas = torch.tensor(template + [template[-1]] * (C - len(template)), device=dev)
    blank = torch.zeros(C, dtype=torch.bool, device=dev)
    blank[torch.tensor(slots, device=dev)] = True
    x = torch.where(blank, model.uniform_tokens((C,), generator), canvas)[None]
    z = torch.zeros(1, C, model.text_config.hidden_size, device=dev, dtype=model.dtype)
    prev, first, N = None, None, cfg.max_steps
    for n in range(1, N + 1):
        t = 1.0 - (n - 1) / N
        L = model.decode(x, z, ctx)[0]
        first = L[blank] if first is None else first
        x0 = L.argmax(-1)
        p0 = torch.softmax(L / ((cfg.tau_max - cfg.tau_min) * t + cfg.tau_min), -1)
        e = _entropy(p0)
        stop = n > 1 and e[blank].mean() <= cfg.stop_entropy and torch.equal(x0[blank], prev)
        prev = x0[blank]
        if stop or n == N:
            return x0[blank], first, L[blank], n
        z = model.self_condition(p0[None])
        keep = torch.zeros(C, dtype=torch.bool, device=dev)
        keep[blank] = accepted_positions(e[blank], cfg.entropy_budget)
        xD = torch.multinomial(p0, 1, generator=generator).squeeze(-1)
        x = torch.where(blank, torch.where(keep, xD, model.uniform_tokens((C,), generator)), canvas)[None]


def readout(logits, cand):
    p = torch.softmax(logits.float(), -1)[list(cand.values())]
    p = p / p.sum()
    return {v: float(x) for v, x in zip(cand, p)}


@torch.no_grad()
def diffusion_decide(model, tok, state, questions, cfg, media=None, processor=None):
    template, slots, cands = schema.infill_template(tok, state, questions)
    ids, mm = schema.prompt_inputs(tok, processor, state, questions, media)
    mm = {k: v.to(model.device) for k, v in mm.items()} if mm else None
    ctx = model.encode(torch.tensor([ids], device=model.device), mm=mm)
    _, first, final, steps = denoise_infill(model, ctx, template, slots, cfg)
    out = {}
    for q, cand, fl, ll in zip(questions, cands, first, final):
        out[q.id] = {"probs_by_readout": {"first": readout(fl, cand), "final": readout(ll, cand)}}
        out[q.id]["probs"] = out[q.id]["probs_by_readout"]["first"]
        out[q.id]["answer"] = max(out[q.id]["probs"], key=out[q.id]["probs"].get)
    return out, steps


@torch.no_grad()
def ar_decide(backbone, tok, state, questions):
    """Same template, filled left to right: prefill prompt + template up to the first blank, then one
    forward per blank (the chosen value is fed back, so later answers see earlier ones)."""
    from transformers import DynamicCache

    template, slots, cands = schema.infill_template(tok, state, questions)
    ids = schema.prompt_ids(tok, state, questions)
    dev = backbone.device
    cache = DynamicCache()
    seq = list(ids) + template[: slots[0]]
    logits = backbone(input_ids=torch.tensor([seq], device=dev), past_key_values=cache, use_cache=True).logits[0, -1]
    out, forwards = {}, 1
    for i, (q, cand) in enumerate(zip(questions, cands)):
        probs = readout(logits, cand)
        ans = max(probs, key=probs.get)
        out[q.id] = {"answer": ans, "probs": probs}
        if i + 1 < len(slots):
            chunk = [cand[ans]] + template[slots[i] + 1: slots[i + 1]]
            logits = backbone(input_ids=torch.tensor([chunk], device=dev), past_key_values=cache, use_cache=True).logits[0, -1]
            forwards += 1
    return out, forwards


# ----------------------------------------------------------------------------- data / training
def encode_infill(tok, ex, P, C):
    qs = parse_questions(ex["questions"])
    answers = {k: str(v).lower() for k, v in ex["answers"].items()}
    try:
        _, slots, _ = schema.infill_template(tok, ex["state"], qs, answers)
    except ValueError:
        return None
    e = encode_example(tok, ex, P, 1, C)
    if e is None or max(slots) >= C:
        return None
    mask = [False] * C
    for s in slots:
        mask[s] = True
    return (*e, mask)


def collate_infill(rows):
    b = collate([r[:5] for r in rows])
    return (*b, torch.tensor([r[5] for r in rows]))


def build_many(a):
    from .decision_data import EMO, table

    rng = random.Random(1)
    rows = [r for r in table("google-research-datasets/go_emotions", "train", "simplified") if any(l != 27 for l in r["labels"])]
    out = []
    for r in rng.sample(rows, a.n):
        k = rng.choice([8, 10, 12, 14, 16])
        pos = [l for l in r["labels"] if l != 27]
        neg = [l for l in range(27) if l not in r["labels"]]
        take = pos[:k] + rng.sample(neg, k - len(pos[:k]))
        rng.shuffle(take)
        out.append({"state": "Comment: " + r["text"], "source": "go_emotions_many",
                    "questions": [{"type": "noul", "id": EMO[l], "question": f"Does the comment express {EMO[l]}?"} for l in take],
                    "answers": {EMO[l]: "true" if l in r["labels"] else "false" for l in take}})
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(out))


def train(a):
    from .io import load, save

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, tok, _ = load(a.backbone, a.init)
    model.train()
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    from peft import LoraConfig, get_peft_model

    model.backbone = get_peft_model(model.backbone, LoraConfig(r=a.lora_rank, lora_alpha=2 * a.lora_rank, target_modules="all-linear",
                                                               exclude_modules=r".*(vision|audio)\w*\..*"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=1e-4)
    rows = [json.loads(l) for f in a.data for l in open(f, encoding="utf-8") if l.strip()]
    data = [e for e in (encode_infill(tok, r, a.prompt_len, a.canvas) for r in rows) if e]
    print(f"{len(data)}/{len(rows)} rows usable as templates", flush=True)
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = cosine(step, a.warmup, a.steps, a.lr, a.end_lr)
        loss, parts = diffusion_loss(model, collate_infill(random.sample(data, a.batch)), a.prompt_len, 1, a.canvas)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()), flush=True)
    save(model, tok, a.out, {"backbone": a.backbone, "canvas_length": a.canvas, "kind": "infill", "args": vars(a)})


@torch.no_grad()
def evaluate(a):
    from .bench import sync
    from .io import load

    model, tok, proc = load(a.backbone, a.checkpoint)
    model.eval()
    cfg = SamplerConfig(canvas_length=a.canvas, max_steps=a.max_steps)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")][: a.limit]
    sink = open(a.out, "w", encoding="utf-8") if a.out else None
    by = {}
    for r in rows:
        qs = parse_questions(r["questions"])
        sync(); t0 = time.perf_counter()
        if a.mode == "diffusion":
            dec, fw = diffusion_decide(model, tok, r["state"], qs, cfg, r.get("media"), proc)
        else:
            dec, fw = ar_decide(model.backbone, tok, r["state"], qs)
        sync(); dt = time.perf_counter() - t0
        gold = {k: str(v).lower() for k, v in r["answers"].items()}
        ok = [dec[q.id]["answer"] == gold[q.id] for q in qs]
        key = r.get("k", r.get("source", "all"))
        s = by.setdefault(key, {"rows": 0, "q": 0, "correct": 0, "all": 0, "secs": [], "fw": 0})
        s["rows"] += 1; s["q"] += len(ok); s["correct"] += sum(ok); s["all"] += all(ok); s["secs"].append(dt); s["fw"] += fw
        if sink:
            sink.write(json.dumps({"gold": gold, "source": r.get("source"), "k": r.get("k"), "seconds": dt, "forwards": fw,
                                   "decisions": dec}, ensure_ascii=False) + "\n")
    summary = {}
    for k, s in sorted(by.items(), key=lambda kv: str(kv[0])):
        secs = sorted(s["secs"])
        summary[str(k)] = {"accuracy": round(s["correct"] / s["q"], 4), "all_correct": round(s["all"] / s["rows"], 4),
                           "latency_p50_s": round(secs[len(secs) // 2], 3), "forwards_per_row": round(s["fw"] / s["rows"], 2)}
    tot = {"accuracy": round(sum(s["correct"] for s in by.values()) / sum(s["q"] for s in by.values()), 4)}
    print(json.dumps({"mode": a.mode, "checkpoint": a.checkpoint, "overall": tot, "by": summary}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-many"); b.add_argument("--out", required=True); b.add_argument("--n", type=int, default=10000)
    t = sub.add_parser("train")
    t.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct"); t.add_argument("--init", required=True)
    t.add_argument("--data", nargs="+", required=True); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=6000); t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--end-lr", type=float, default=1e-5)
    t.add_argument("--batch", type=int, default=2); t.add_argument("--prompt-len", type=int, default=1024)
    t.add_argument("--canvas", type=int, default=128); t.add_argument("--lora-rank", type=int, default=8)
    t.add_argument("--log-every", type=int, default=50); t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct"); e.add_argument("--checkpoint", default=None)
    e.add_argument("--mode", choices=["diffusion", "ar"], required=True); e.add_argument("--data", required=True)
    e.add_argument("--canvas", type=int, default=128); e.add_argument("--max-steps", type=int, default=48)
    e.add_argument("--limit", type=int, default=None); e.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    {"build-many": build_many, "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()

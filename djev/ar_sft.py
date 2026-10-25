"""Control experiment: ordinary autoregressive SFT (next-token loss on the response) of the
original backbone on the same decision data and schedule as the Jev diffusion SFT. Evaluate
with `djev.baseline_ar --checkpoint <out>` (logit read-out after `{"<id>": `)."""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn.functional as F

from . import schema
from .io import load, save
from .sft import cosine, parse_questions


def encode(tok, ex, P):
    if "messages" in ex:  # chat rows (e.g. Sudoku): loss on the assistant turn
        *ctx, last = ex["messages"]
        p = tok.apply_chat_template(ctx, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        r = schema.assistant_ids(tok, ctx, last["content"])
        return (p + r, len(p)) if len(p) + len(r) <= P else None
    qs = parse_questions(ex["questions"])
    answers = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in ex["answers"].items()}
    p = schema.prompt_ids(tok, ex["state"], qs)
    r = schema.response_ids(tok, ex["state"], qs, answers)
    if len(p) + len(r) > P:
        return None
    return p + r, len(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--init", default=None, help="unused; kept so io.load's checkpoint chain stops here")
    ap.add_argument("--max-len", type=int, default=1088)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--end-lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, tok, _ = load(a.backbone)
    bb = model.backbone
    bb.train()
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    from peft import LoraConfig, get_peft_model

    bb = get_peft_model(bb, LoraConfig(r=a.lora_rank, lora_alpha=2 * a.lora_rank, target_modules="all-linear",
                                       exclude_modules=r".*(vision|audio)\w*\..*"))
    model.backbone = bb
    params = [p for p in bb.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.weight_decay)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    data = [e for e in (encode(tok, r, a.max_len) for r in rows) if e]
    print(f"{len(data)}/{len(rows)} examples fit", flush=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = model.device
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = cosine(step, a.warmup, a.steps, a.lr, a.end_lr)
        batch = random.sample(data, min(a.batch, len(data)))
        L = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), L), pad)
        att = torch.zeros(len(batch), L, dtype=torch.long)
        tgt = torch.full((len(batch), L), -100)
        for i, (x, p) in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x)
            att[i, : len(x)] = 1
            tgt[i, p : len(x)] = torch.tensor(x[p:])
        logits = bb(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1).to(dev), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"step {step} lr {opt.param_groups[0]['lr']:.2e} loss {loss.item():.4f}", flush=True)
    save(model, tok, a.out, {"backbone": a.backbone, "kind": "ar_sft", "args": vars(a)})
    print("saved to", a.out)


if __name__ == "__main__":
    main()

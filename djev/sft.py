"""Diffusion SFT of a converted AR backbone.

Loss (DiffusionGemma §8, Eq. 13): L = L_encoder + L_decoder
  L_encoder: next-token CE over the whole sequence (prompt + K canvases), causal attention
  L_decoder: denoising CE over one canvas k ~ Unif{1..K}, noised with multinomial diffusion
             (t ~ U[0,1], each token replaced by a uniform random token with probability t, §4),
             conditioned on the clean prefix through H; 50% of the batch gets a
             self-conditioning signal z_t from a previous forward pass, the rest z_t = 0.
`--no-encoder-loss` gives the Stage-1 objective of §4 (Eq. 11) instead.
Defaults are the PubMedQA recipe of Table 7.

Data: JSONL, one decision example per line:
  {"state": "...", "questions": [{"type": "noul", "id": "q1", "question": "..."}, ...],
   "answers": {"q1": "true"}}
"""
from __future__ import annotations

import argparse
import json
import math
import random

import torch
import torch.nn.functional as F

from . import schema
from .io import load, save
from .model import DiffusionLM


def parse_questions(raw: list[dict]) -> list[schema.Question]:
    out = []
    for q in raw:
        kind = q["type"].lower()
        if kind == "choice":
            out.append(schema.Choice(q["id"], q["question"], q["options"]))
        elif kind == "noul":
            out.append(schema.Noul(q["id"], q["question"]))
        elif kind == "score":
            out.append(schema.Score(q["id"], q["question"], q["levels"]))
        else:
            raise ValueError(f"unknown question type {kind}")
    return out


def encode_example(tok, ex: dict, P: int, K: int, C: int, processor=None):
    """Left-pad the prompt to P tokens, then up to K canvases of C tokens.

    Returns (ids, valid, loss_mask, mm, n): `n` is the number of canvases that really exist
    for this example; the decoder loss samples k uniformly among those (§8, "the K canvases
    available"). A complete response ends with the end-of-turn token and its last canvas is
    filled with it; a truncated response (`ex["truncated"]`) keeps only whole canvases and no
    end-of-turn token. Canvases beyond `n` are padding and invisible to the model.
    Decision rows: {"state", "questions", "answers", "media"?}. Chat rows (general diffusion
    SFT, DiffusionGemma §4): {"messages": [..., {"role": "assistant", "content": ...}]}."""
    if "messages" in ex:
        *ctx, last = ex["messages"]
        assert last["role"] == "assistant"
        p = tok.apply_chat_template(ctx, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        r = schema.assistant_ids(tok, ctx, last["content"])
        mm = None
    else:
        qs = parse_questions(ex["questions"])
        answers = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in ex["answers"].items()}
        p, mm = schema.prompt_inputs(tok, processor, ex["state"], qs, ex.get("media"))
        r = schema.response_ids(tok, ex["state"], qs, answers)
    if len(p) > P:
        return None
    if ex.get("truncated"):
        r = r[:-1]  # drop the end-of-turn token the template added
        n = min(len(r) // C, K)
        if n == 0:
            return None
        r = r[: n * C]
        body = r
    else:
        if len(r) > K * C:
            return None
        n = -(-len(r) // C)
        # The rest of the last canvas is filled with the end-of-turn token, so the sampler's
        # "valid tokens up to the first end-of-sequence token" rule (DG §3.4) applies to it.
        body = r + [r[-1]] * (n * C - len(r))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lp = P - len(p)
    rest = K * C - n * C
    ids = [pad] * lp + p + body + [pad] * rest
    valid = [False] * lp + [True] * (len(p) + n * C) + [False] * rest
    loss_mask = [False] * lp + [True] * (len(p) + len(r)) + [False] * (K * C - len(r))
    if mm and "mm_token_type_ids" in mm:
        t = mm["mm_token_type_ids"][0].tolist()
        mm["mm_token_type_ids"] = torch.tensor([[0] * lp + t + [0] * (K * C)])
    return ids, valid, loss_mask, mm, n


def collate(rows) -> tuple:
    ids, valid, loss_mask = (torch.tensor([r[i] for r in rows]) for i in range(3))
    n = torch.tensor([r[4] for r in rows])
    mms = [r[3] for r in rows]
    if not any(mms):
        return ids, valid, loss_mask, None, n
    mm = {}
    for key in schema.MM_KEYS:
        parts = [m[key] for m in mms if m and key in m]
        if not parts:
            continue
        if key == "mm_token_type_ids":
            mm[key] = torch.cat([m[key] if m and key in m else torch.zeros(1, ids.shape[1], dtype=torch.long) for m in mms])
        else:  # flattened patches / per-media grids are concatenated in batch order
            mm[key] = torch.cat(parts)
    return ids, valid, loss_mask, mm, n


def diffusion_loss(model: DiffusionLM, batch, P: int, K: int, C: int, encoder_loss: bool = True, t_low: float = 0.0):
    """`t_low` > 0 draws the noise level from U[t_low, 1] instead of U[0, 1]: the heavily-noised states a
    few-step sampler starts from (used by djev.sdrl for sampler distillation)."""
    clean, valid, loss_mask = (t.to(model.device) for t in batch[:3])
    mm = {k: v.to(model.device) for k, v in batch[3].items()} if len(batch) > 3 and batch[3] else None
    B, V, dev = clean.shape[0], model.vocab_size, model.device
    n = batch[4].to(dev) if len(batch) > 4 else torch.full((B,), K, device=dev)

    k = (torch.rand(B, device=dev) * n).long().clamp(max=K - 1)  # k ~ Unif{canvases that exist}
    start = P + k * C
    cols = start[:, None] + torch.arange(C, device=dev)[None]
    x0 = clean.gather(1, cols)

    t = t_low + (1 - t_low) * torch.rand(B, 1, device=dev)
    flip = torch.rand(B, C, device=dev) < t
    slots = batch[5].to(dev).gather(1, cols - P) if len(batch) > 5 and batch[5] is not None else None
    if slots is not None:  # infilling: only the answer blanks are noised; the template stays clean
        flip &= slots
    x_t = torch.where(flip, model.uniform_tokens((B, C)), x0)

    d = model.text_config.hidden_size
    z = torch.zeros(B, C, d, device=dev, dtype=model.dtype)
    use_sc = torch.rand(B, device=dev) < 0.5
    if use_sc.any():
        with torch.no_grad():
            _, prev = model.forward_train(clean, valid, x_t, start, z, mm)
            p0 = torch.softmax(prev.float(), -1)
        z = torch.where(use_sc[:, None, None], model.self_condition(p0), z)

    enc_logits, dec_logits = model.forward_train(clean, valid, x_t, start, z, mm)
    if slots is not None:
        l_dec = F.cross_entropy(dec_logits.float()[slots], x0[slots])  # blanks only
    else:
        l_dec = F.cross_entropy(dec_logits.float().reshape(-1, V), x0.reshape(-1))  # 1/C Σ_i, averaged over batch
    if not encoder_loss:
        return l_dec, {"decoder": l_dec.item()}
    keep = loss_mask[:, 1:] & ~torch.isin(clean[:, 1:], model.forbidden.to(dev))  # no NTP on media placeholders
    tgt = clean[:, 1:].masked_fill(~keep, -100)
    l_enc = F.cross_entropy(enc_logits[:, :-1].float().reshape(-1, V), tgt.reshape(-1), ignore_index=-100)
    return l_enc + l_dec, {"encoder": l_enc.item(), "decoder": l_dec.item()}


def cosine(step: int, warmup: int, total: int, peak: float, end: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * prog))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--init", default=None, help="resume from a djev checkpoint")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-rank", type=int, default=4, help="0 = full finetuning")
    ap.add_argument("--canvas", type=int, default=128)
    ap.add_argument("--canvases", type=int, default=2)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--end-lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--no-encoder-loss", action="store_true", help="Stage-1 objective (Eq. 11)")
    ap.add_argument("--grad-checkpointing", action="store_true", help="trade compute for memory (16GB GPUs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0, help="also save <out>/step_N every N steps")
    a = ap.parse_args(argv)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, tok, proc = load(a.backbone, a.init)
    model.train()
    if a.grad_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if a.lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        model.backbone = get_peft_model(
            model.backbone, LoraConfig(
                r=a.lora_rank,
                lora_alpha=2 * a.lora_rank,
                target_modules="all-linear",  # §8: every linear op of the language model
                exclude_modules=r".*(vision|audio)\w*\..*",  # multimodal towers are not part of the LM
            )
        )
    # The self-conditioning FFW is new in a fresh conversion, so it is always trained in full.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.weight_decay)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    data = [e for e in (encode_example(tok, r, a.prompt_len, a.canvases, a.canvas, proc) for r in rows) if e]
    print(f"{len(data)}/{len(rows)} examples fit P={a.prompt_len}, K*C={a.canvases * a.canvas}")
    if not data:
        raise SystemExit("no usable examples")

    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = cosine(step, a.warmup, a.steps, a.lr, a.end_lr)
        rows_b = random.sample(data, min(a.batch, len(data)))
        batch = collate(rows_b)
        loss, parts = diffusion_loss(model, batch, a.prompt_len, a.canvases, a.canvas, not a.no_encoder_loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            mem = f" peak_mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB" if torch.cuda.is_available() else ""
            print(f"step {step} lr {opt.param_groups[0]['lr']:.2e} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()) + mem, flush=True)
        if a.save_every and step and step % a.save_every == 0:
            save(model, tok, f"{a.out}/step_{step}", {"backbone": a.backbone, "canvas_length": a.canvas, "args": vars(a), "step": step})

    save(model, tok, a.out, {"backbone": a.backbone, "canvas_length": a.canvas, "args": vars(a)})
    print("saved to", a.out)


if __name__ == "__main__":
    main()

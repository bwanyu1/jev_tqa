"""Autoregressive baseline: the untouched backbone answers each question with a single forward
pass, reading the next-token distribution after `{"<id>": ` over the allowed values
(no diffusion, no finetuning). Same data format and metrics as djev.evaluate."""
from __future__ import annotations

import argparse
import json
import time

import torch

from . import schema
from .io import _backbone, pick_device
from .sft import parse_questions


def candidate_tokens(tok, prefix_ids: list[int], q: schema.Question) -> tuple[list[int], dict[str, int]]:
    """Tokenise `{"id": v}` for every allowed v and return the shared prefix plus each value's
    first differing token."""
    seqs = {v: tok('{"%s": %s}' % (q.id, v), add_special_tokens=False)["input_ids"] for v in q.values()}
    first = next(iter(seqs.values()))
    n = 0
    while all(len(s) > n and s[n] == first[n] for s in seqs.values()):
        n += 1
    cand = {v: s[n] for v, s in seqs.items()}
    if len(set(cand.values())) != len(cand):
        raise ValueError(f"{q.id}: values share their first token")
    return prefix_ids + first[:n], cand


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--checkpoint", default=None, help="djev checkpoint run in AR mode (DiffusionGemma §3.5)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="optional JSONL of per-example results")
    a = ap.parse_args(argv)
    sink = open(a.out, "w", encoding="utf-8") if a.out else None

    from transformers import AutoTokenizer

    from transformers import AutoProcessor

    tok = AutoTokenizer.from_pretrained(a.backbone)
    proc = AutoProcessor.from_pretrained(a.backbone)
    device = pick_device()
    if a.checkpoint:
        from .io import load

        model = load(a.backbone, a.checkpoint, device=device)[0].backbone.eval()
    else:
        model = _backbone(a.backbone, torch.bfloat16 if device == "cuda" else torch.float32, device).eval()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][: a.limit]
    n = correct = 0
    p_gold = 0.0
    secs = []
    for r in rows:
        qs = parse_questions(r["questions"])
        prompt, mm = schema.prompt_inputs(tok, proc, r["state"], qs, r.get("media"))
        decisions = {}
        t0 = time.perf_counter()
        for q in qs:
            ids, cand = candidate_tokens(tok, prompt, q)
            extra = {}
            if mm:
                extra = {k: v.to(device) for k, v in mm.items()}
                if "mm_token_type_ids" in extra:  # extend to the appended answer-prefix tokens
                    t = extra["mm_token_type_ids"]
                    extra["mm_token_type_ids"] = torch.cat([t, t.new_zeros(1, len(ids) - t.shape[1])], 1)
            logits = model(input_ids=torch.tensor([ids], device=device), **extra).logits[0, -1].float()
            p = torch.softmax(logits, -1)[list(cand.values())]
            p = p / p.sum()
            probs = dict(zip(cand, p.tolist()))
            gold = str(r["answers"][q.id]).lower()
            n += 1
            p_gold += probs[gold]
            correct += max(probs, key=probs.get) == gold
            decisions[q.id] = {"answer": max(probs, key=probs.get), "probs": probs}
        if sink:
            sink.write(json.dumps({"gold": r["answers"], "source": r.get("source"), "decisions": decisions}, ensure_ascii=False) + "\n")
        if device == "cuda":
            torch.cuda.synchronize()
        secs.append(time.perf_counter() - t0)
    secs.sort()
    print(json.dumps({
        "questions": n,
        "accuracy": correct / max(n, 1),
        "avg_prob_gold": p_gold / max(n, 1),
        "latency_p50_s": secs[len(secs) // 2] if secs else None,
        "latency_p90_s": secs[int(len(secs) * 0.9)] if secs else None,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Record ordinary autoregressive (greedy) decoding of the original backbone: one token per
forward pass, with the probability of each chosen token. Output mirrors djev.trace so the
two processes can be rendered side by side.
Usage: python scripts/ar_trace.py out.json "prompt 1" "prompt 2" ...
"""
from __future__ import annotations

import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "Qwen/Qwen3-VL-4B-Instruct"


@torch.no_grad()
def main():
    out_path, *prompts = sys.argv[1:]
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    out = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True,
                                      return_dict=False)
        g = model.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=160, do_sample=False,
                           output_scores=True, return_dict_in_generate=True)
        new = g.sequences[0, len(ids):].tolist()
        conf = [float(torch.softmax(s[0].float(), -1)[t]) for s, t in zip(g.scores, new)]
        out.append({"prompt": p, "ids": new, "tokens": [tok.decode([t]) for t in new], "conf": conf, "text": tok.decode(new, skip_special_tokens=True)})
        print(len(new), repr(out[-1]["text"][:200]), flush=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

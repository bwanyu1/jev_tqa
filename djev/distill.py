"""General (non-decision) SFT data for the diffusion conversion of DiffusionGemma §4.

The paper finetunes on Gemma 4's own post-training mix, which is not public. We keep the
same spirit by self-distillation: the AR backbone answers open prompts (UltraChat first
turns) and the diffusion model is trained to produce those answers. Responses that hit the
token limit are kept and marked `truncated`: block-AR training only needs the first K
canvases of a response, so such rows train those canvases without an end-of-turn token.
"""
from __future__ import annotations

import argparse
import json
import random


def ultrachat_prompts(n: int, seed: int) -> list[str]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download("HuggingFaceH4/ultrachat_200k", "data/train_sft-00000-of-00003-a3ecf92756993583.parquet", repo_type="dataset")
    rows = pq.read_table(path, columns=["prompt"]).column("prompt").to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [r for r in rows if 10 < len(r) < 1500][:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts", default=None, help="JSONL of chat rows to answer instead of UltraChat")
    a = ap.parse_args(argv)

    from vllm import LLM, SamplingParams

    if a.prompts:
        prompts = [json.loads(l)["messages"][0]["content"] for l in open(a.prompts, encoding="utf-8")][: a.n]
    else:
        prompts = ultrachat_prompts(a.n, a.seed)
    llm = LLM(a.model, seed=a.seed, gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
              limit_mm_per_prompt={"image": 0, "video": 0})
    params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_tokens)  # model card defaults
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
    kept = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for p, o in zip(prompts, outs):
            c = o.outputs[0]
            if not c.text.strip():
                continue
            f.write(json.dumps({"messages": [{"role": "user", "content": p},
                                             {"role": "assistant", "content": c.text.strip()}],
                                "truncated": c.finish_reason != "stop"}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept}/{len(prompts)} to {a.out}")


if __name__ == "__main__":
    main()

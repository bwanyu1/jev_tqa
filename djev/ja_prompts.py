"""Japanese prompts (first user turns) from Apache-2.0 instruction datasets, for the
self-distilled Japanese part of the general diffusion SFT."""
from __future__ import annotations

import argparse
import ast
import io
import json
import random
import urllib.request

import pyarrow.parquet as pq

URL = "https://huggingface.co/api/datasets/{}/parquet/default/train/{}.parquet"
SOURCES = {"llm-jp/oasst2-33k-ja": (1, "conversations"), "llm-jp/magpie-sft-v1.0": (1, "conversations"),
           "Aratako/Magpie-Tanuki-8B-97k": (2, "messages")}


def first_user(cell):
    conv = ast.literal_eval(cell) if isinstance(cell, str) else cell
    for m in conv:
        if m["role"] == "user":
            return m["content"].strip()
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    rows = []
    for name, (shards, col) in SOURCES.items():
        prompts = set()
        for i in range(shards):
            t = pq.read_table(io.BytesIO(urllib.request.urlopen(URL.format(name, i)).read()), columns=[col])
            for cell in t.column(col).to_pylist():
                p = first_user(cell)
                if p and 10 < len(p) < 1500:
                    prompts.add(p)
        picked = rng.sample(sorted(prompts), min(a.per_source, len(prompts)))
        rows += [{"messages": [{"role": "user", "content": p}], "source": name} for p in picked]
        print(name, len(prompts), "->", len(picked), flush=True)
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

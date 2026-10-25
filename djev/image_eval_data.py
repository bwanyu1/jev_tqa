"""Image multiple-choice evaluation set in the decision format (Choice with media).
ScienceQA test (image questions only, CC BY-SA 4.0) + A-OKVQA validation."""
from __future__ import annotations

import argparse
import ast
import io
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

URL = "https://huggingface.co/api/datasets/{}/parquet/default/{}/0.parquet"


def table(name, split):
    return pq.read_table(io.BytesIO(urllib.request.urlopen(URL.format(name, split)).read())).to_pylist()


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--per-source", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    img_dir = Path(a.images)
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    sqa = [r for r in table("derek-thomas/ScienceQA", "test") if r["image"] and len(lit(r["choices"])) <= 9]
    for i, r in enumerate(rng.sample(sqa, min(a.per_source, len(sqa)))):
        p = img_dir / f"sqa_{i}.png"
        p.write_bytes(r["image"]["bytes"])
        ch = lit(r["choices"])
        order = list(range(len(ch)))
        rng.shuffle(order)
        gold = int(r["answer"])
        state = "Science question about the image." + (f"\nContext: {r['hint']}" if r["hint"] else "")
        rows.append({"state": state, "media": [{"type": "image", "path": str(p)}], "source": "scienceqa",
                     "questions": [{"type": "choice", "id": "answer", "question": r["question"], "options": [ch[j] for j in order]}],
                     "answers": {"answer": str(order.index(gold) + 1)}})

    aok = table("HuggingFaceM4/A-OKVQA", "validation")
    for i, r in enumerate(rng.sample(aok, min(a.per_source, len(aok)))):
        p = img_dir / f"aok_{i}.jpg"
        p.write_bytes(r["image"]["bytes"])
        ch = lit(r["choices"])
        order = list(range(len(ch)))
        rng.shuffle(order)
        gold = int(r["correct_choice_idx"])
        rows.append({"state": "Question about the photo.", "media": [{"type": "image", "path": str(p)}], "source": "a-okvqa",
                     "questions": [{"type": "choice", "id": "answer", "question": r["question"], "options": [ch[j] for j in order]}],
                     "answers": {"answer": str(order.index(gold) + 1)}})
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows))


if __name__ == "__main__":
    main()

"""Check teacher labels: a stronger model answers a sample of decision rows; report agreement.
Writes per-row results so disagreements can be inspected or filtered."""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from . import groq_distill as G
from . import schema
from .sft import parse_questions


def judge(model, row, effort="medium"):
    qs = parse_questions(row["questions"])
    msgs = schema.messages(row["state"], qs)
    text, _ = G.chat(model, msgs, 2500, temperature=0.0, effort=effort, json_mode=G.JSON_MODE)
    try:
        got = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return None
    got = {k: str(v).lower() for k, v in got.items()}
    per_q = {q.id: got.get(q.id) == row["answers"][q.id] for q in qs}
    return {"row": row, "judge": got, "agree": per_q}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--filter-out", default=None,
                    help="also write the input rows keeping only questions whose label the judge reproduced")
    a = ap.parse_args(argv)
    if a.base_url:
        G.URL = a.base_url.rstrip("/") + "/chat/completions"
    G.JSON_MODE = not a.no_json_mode
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.n < len(rows):
        rows = random.Random(0).sample(rows, a.n)
    with ThreadPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(lambda r: _safe(a.model, r, a.effort), rows) if r]
    if a.filter_out:
        with open(a.filter_out, "w", encoding="utf-8") as f:
            for r in res:
                keep = [q for q in r["row"]["questions"] if r["agree"][q["id"]]]
                if keep:
                    row = {**r["row"], "questions": keep, "answers": {q["id"]: r["row"]["answers"][q["id"]] for q in keep}}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    qs = [v for r in res for v in r["agree"].values()]
    rows_ok = [all(r["agree"].values()) for r in res]
    by = {}
    for r in res:
        k = r["row"].get("source", "?")
        by.setdefault(k, []).extend(r["agree"].values())
    print(json.dumps({"judged_rows": len(res), "question_agreement": round(sum(qs) / len(qs), 3),
                      "row_agreement": round(sum(rows_ok) / len(rows_ok), 3),
                      "by_source": {k: round(sum(v) / len(v), 3) for k, v in by.items()}, **G.USAGE}))


def _safe(model, row, effort="medium"):
    try:
        return judge(model, row, effort)
    except Exception as e:
        print("error", str(e)[:120], flush=True)
        return None


if __name__ == "__main__":
    main()

"""Quality of the texts saved by `djev.bench --out`: an LLM judge score (1-10) per response and
a repetition rate, for the AR baseline and every diffusion step budget N.

The judge sees one prompt and one response at a time (absolute grading, no side-by-side), so
the order of systems cannot bias it. Repetition = 1 - unique token bigrams / token bigrams.
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from . import groq_distill as G

RUBRIC = """Rate the assistant's response to the user's request on a 1-10 scale.
Judge correctness, fluency (grammar, no broken words or characters), coherence, and absence of
repetition. A response may be cut off at a length limit; do not penalise the cut itself, only the
quality of what is there. Answer in the same language standard as the request (Japanese requests
need natural Japanese).
User request:
<<<{prompt}>>>
Assistant response:
<<<{response}>>>
Return only JSON: {{"score": <integer 1-10>}}"""


def repetition(tok, text: str) -> float:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    bigrams = list(zip(ids, ids[1:]))
    return 0.0 if len(bigrams) < 2 else 1 - len(set(bigrams)) / len(bigrams)


def score(model, prompt, response):
    if not response.strip():
        return 1
    text, _ = G.chat(model, [{"role": "user", "content": RUBRIC.format(prompt=prompt, response=response)}],
                     1500, temperature=0.0, effort="medium", json_mode=G.JSON_MODE)
    try:
        return int(json.loads(text[text.index("{"): text.rindex("}") + 1])["score"])
    except (ValueError, KeyError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bench_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="openai/gpt-oss-20b")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    G.URL = a.base_url.rstrip("/") + "/chat/completions"
    b = json.load(open(a.bench_json, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    jobs = [(sys_, i, p, t) for sys_, texts in b["texts"].items() for i, (p, t) in enumerate(zip(b["prompts"], texts))]
    with ThreadPoolExecutor(a.workers) as ex:
        scores = list(ex.map(lambda j: score(a.judge, j[2], j[3]), jobs))
    per = {}
    for (sys_, i, p, t), s in zip(jobs, scores):
        ja = any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in p)
        per.setdefault(sys_, []).append({"i": i, "ja": ja, "score": s, "repetition": repetition(tok, t)})
    summary = {}
    for sys_, rows in per.items():
        ok = [r["score"] for r in rows if r["score"] is not None]
        summary[sys_] = {
            "judge_mean": round(statistics.mean(ok), 2),
            "judge_mean_en": round(statistics.mean([r["score"] for r in rows if not r["ja"] and r["score"] is not None]), 2),
            "judge_mean_ja": round(statistics.mean([r["score"] for r in rows if r["ja"] and r["score"] is not None]), 2),
            "repetition_mean": round(statistics.mean(r["repetition"] for r in rows), 3),
            "scored": len(ok),
        }
    json.dump({"judge": a.judge, "summary": summary, "per_response": per}, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

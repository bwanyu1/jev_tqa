"""Distillation data from GPT-OSS on Groq or on a local vLLM server (OpenAI-compatible API).

  general   : answers to open prompts -> chat rows for the general diffusion SFT (DG §4)
  decisions : new Jev-style rows {state, questions, answers} written by the teacher, validated
              against djev.schema -> data for the Jev downstream SFT (DG §8)

Only the final answer is kept (the reasoning channel is discarded). Needs GROQ_API_KEY.
Rows are appended to --out as they arrive, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from . import schema
from .sft import parse_questions

URL = "https://api.groq.com/openai/v1/chat/completions"  # replaced by --base-url
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "failed": 0}
LOCK = threading.Lock()
STOP = threading.Event()  # set when the daily quota (RPD / TPD) is exhausted


JSON_MODE = True


class DailyLimit(RuntimeError):
    pass


def _seconds(v: str | None) -> float:
    """'2.5', '31.5s', '46m4.8s', '1h2m3s' -> seconds."""
    if not v:
        return 0.0
    try:
        return float(v)
    except ValueError:
        pass
    total, num = 0.0, ""
    for ch in v:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            total += float(num or 0) * {"h": 3600, "m": 60, "s": 1}.get(ch, 0)
            num = ""
    return total


def chat(model, messages, max_tokens, temperature=0.7, effort="low", json_mode=False, retries=12):
    body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
            "temperature": temperature, "reasoning_effort": effort}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', 'local')}"}
    for i in range(retries):
        if STOP.is_set():
            raise DailyLimit("daily quota exhausted")
        r = requests.post(URL, json=body, headers=headers, timeout=600)
        if r.status_code == 429:
            wait = _seconds(r.headers.get("retry-after")) or 2 ** i
            if wait > 300 or "per day" in r.text or "(TPD)" in r.text or "(RPD)" in r.text:
                STOP.set()
                raise DailyLimit(r.text[:200])
            time.sleep(wait + 1)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** i)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        d = r.json()
        with LOCK:
            USAGE["requests"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                USAGE[k] += d.get("usage", {}).get(k, 0)
        c = d["choices"][0]
        return c["message"].get("content") or "", c.get("finish_reason")
    raise RuntimeError("rate limited too many times")


# ----------------------------------------------------------------------------- general
def general_task(model, prompt, max_tokens):
    text, finish = chat(model, [{"role": "user", "content": prompt}], max_tokens)
    if not text.strip():
        return None
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": text.strip()}],
            "truncated": finish == "length", "teacher": model}


# ----------------------------------------------------------------------------- decisions
SKILLS = [
    "counting items mentioned in the state", "ordering dates and times", "comparing numbers and amounts",
    "double negation and negated statements", "reading tables or logs", "customer support triage",
    "content moderation", "routing a request to the right tool or model", "code review findings",
    "product reviews", "scheduling constraints", "medical or legal style intake (no advice)",
    "agent action selection in a UI", "e-mail classification", "game state decisions",
    "multi-step consistency between related questions",
]
DOMAINS_JA = ["日本語のカスタマーサポート", "日本語のニュース記事", "日本語のSNS投稿", "日本語の社内チャット",
              "日本語の商品レビュー", "日本語の予定表", "日本語のエラーログの説明"]

DECISION_PROMPT = """You write training data for a fast decision model. Produce ONE JSON object:
{{"state": <string>, "questions": [...], "answers": {{...}}}}
Rules:
- "state" is a realistic {lang} text of 40-250 words about: {topic}. Skill to exercise: {skill}.
- 2 to 5 questions about the state, mixing these types:
  {{"type": "noul", "id": <snake_case>, "question": <yes/no question>}}
  {{"type": "choice", "id": <snake_case>, "question": <question>, "options": [2-6 short options]}}
  {{"type": "score", "id": <snake_case>, "question": <question>, "levels": [3-5 level descriptions, lowest first]}}
- Questions and options must be in {lang}. ids are unique English snake_case.
- "answers" maps every id to the correct answer: noul -> "true"/"false"; choice -> option number as a string ("1".."n");
  score -> level number as a string ("1".."n"). Answers must be unambiguous from the state alone.
- Related questions must be logically consistent with each other.
Output only the JSON object."""


def decision_task(model, rng_seed, max_tokens):
    rng = random.Random(rng_seed)
    ja = rng.random() < 0.35
    lang = "Japanese" if ja else "English"
    topic = rng.choice(DOMAINS_JA) if ja else rng.choice(["everyday life", "business", "software", "science", "sports", "travel", "finance", "health", "education", "online community"])
    prompt = DECISION_PROMPT.format(lang=lang, topic=topic, skill=rng.choice(SKILLS))
    text, finish = chat(model, [{"role": "user", "content": prompt}], max_tokens, temperature=1.0, json_mode=JSON_MODE)
    try:
        row = json.loads(text[text.index("{") : text.rindex("}") + 1])
        qs = parse_questions(row["questions"])
        schema.validate(qs)
        ans = {k: str(v).lower() for k, v in row["answers"].items()}
        schema.answer_json(qs, ans)  # raises if an answer is outside the allowed values
        if set(ans) != {q.id for q in qs} or not str(row["state"]).strip():
            return None
    except Exception:
        return None
    return {"state": row["state"], "questions": row["questions"], "answers": ans, "source": "gpt_oss_" + ("ja" if ja else "en"),
            "teacher": model}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["general", "decisions"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--prompts", default=None, help="general: JSONL with chat rows whose first user turn is the prompt")
    ap.add_argument("--skip", type=int, default=0, help="general: skip the first N prompts of --prompts")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=2, help="free plan: 8K tokens/min, keep this low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default=None, help="e.g. http://localhost:8001/v1 for a local vLLM server")
    ap.add_argument("--no-json-mode", action="store_true")
    a = ap.parse_args(argv)
    global URL, JSON_MODE
    if a.base_url:
        URL = a.base_url.rstrip("/") + "/chat/completions"
    JSON_MODE = not a.no_json_mode

    if a.kind == "general":
        prompts = [json.loads(l)["messages"][0]["content"] for l in open(a.prompts, encoding="utf-8")][a.skip : a.skip + a.n]
        jobs = [(general_task, (a.model, p, a.max_tokens)) for p in prompts]
    else:
        jobs = [(decision_task, (a.model, a.seed * 1_000_003 + i, a.max_tokens)) for i in range(a.n)]

    kept = 0
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fn, *args) for fn, args in jobs]
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except DailyLimit:
                for other in futs:
                    other.cancel()
                continue
            except Exception as e:
                USAGE["failed"] += 1
                print("error:", str(e)[:200], flush=True)
                continue
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
    cost = 0.0 if a.base_url else USAGE["prompt_tokens"] / 1e6 * 0.15 + USAGE["completion_tokens"] / 1e6 * 0.60
    print(json.dumps({"kept": kept, "jobs": len(jobs), "daily_limit_hit": STOP.is_set(), **USAGE, "seconds": round(time.time() - t0, 1),
                      "est_cost_usd_120b": round(cost, 4)}))


if __name__ == "__main__":
    main()

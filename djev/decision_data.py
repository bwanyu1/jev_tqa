"""Build Jev-style decision data (Choice / Noul / Score) from public labelled datasets.

Every row is {"state", "questions", "answers"}; several sources attach multiple consistent
questions to the same state (e.g. NLI relation + entails? + contradicts?) so the diffusion
canvas has to produce a jointly consistent JSON. Option and question order are shuffled to
avoid position bias. Writes <out>/train.jsonl and <out>/eval.jsonl (held-out splits).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

PARQUET = "https://huggingface.co/api/datasets/{name}/parquet/{config}/{split}/{shard}.parquet"


def table(name: str, split: str, config: str = "default", shard: int = 0) -> list[dict]:
    with urllib.request.urlopen(PARQUET.format(name=name, config=config, split=split, shard=shard)) as r:
        return pq.read_table(io.BytesIO(r.read())).to_pylist()


def choice(rng, qid, question, options, gold_idx):
    order = list(range(len(options)))
    rng.shuffle(order)
    q = {"type": "choice", "id": qid, "question": question, "options": [options[i] for i in order]}
    return q, str(order.index(gold_idx) + 1)


def noul(qid, question, value: bool):
    return {"type": "noul", "id": qid, "question": question}, "true" if value else "false"


def score(qid, question, levels, level_idx):
    return {"type": "score", "id": qid, "question": question, "levels": levels}, str(level_idx + 1)


def row(rng, state, qa):
    rng.shuffle(qa)
    return {"state": state, "questions": [q for q, _ in qa], "answers": {q["id"]: a for q, a in qa}}


# ----------------------------------------------------------------------------- sources
def boolq(rng, split):
    for r in table("google/boolq", split):
        yield row(rng, r["passage"], [noul("answer", r["question"].rstrip("?") + "?", r["answer"])])


NLI = ["the hypothesis follows from the premise", "neither follows nor contradicts", "the hypothesis contradicts the premise"]


def mnli(rng, split):
    for r in table("nyu-mll/multi_nli", split):
        if r["label"] not in (0, 1, 2):
            continue
        state = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
        yield row(rng, state, [choice(rng, "relation", "What is the relation between the premise and the hypothesis?", NLI, r["label"]),
                               noul("entails", "Does the premise entail the hypothesis?", r["label"] == 0),
                               noul("contradicts", "Does the hypothesis contradict the premise?", r["label"] == 2)])


def sst2(rng, split):
    for r in table("stanfordnlp/sst2", split):
        yield row(rng, "Movie review: " + r["sentence"].strip(),
                  [choice(rng, "sentiment", "What is the sentiment of the review?", ["negative", "positive"], r["label"]),
                   noul("recommend", "Would the reviewer recommend the movie?", r["label"] == 1)])


AG = ["world", "sports", "business", "science and technology"]


def ag_news(rng, split):
    for r in table("fancyzhx/ag_news", split):
        qa = [choice(rng, "topic", "Which section does this news article belong to?", AG, r["label"])]
        for k in rng.sample(range(4), 2):
            qa.append(noul(f"is_{AG[k].split()[0]}", f"Is this article about {AG[k]}?", r["label"] == k))
        yield row(rng, "News article: " + r["text"], qa)


EMO = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
       "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
       "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]


def go_emotions(rng, split):
    for r in table("google-research-datasets/go_emotions", split, "simplified"):
        pos = [l for l in r["labels"] if l != 27]
        if not pos:
            continue
        neg = [l for l in range(27) if l not in r["labels"]]
        picks = rng.sample(pos, min(2, len(pos))) + rng.sample(neg, 4 - min(2, len(pos)))
        yield row(rng, "Comment: " + r["text"],
                  [noul(EMO[l], f"Does the comment express {EMO[l]}?", l in r["labels"]) for l in picks])


TOX = ["not toxic", "borderline", "clearly toxic"]


def civil(rng, split):
    rows = table("google/civil_comments", split)
    toxic = [r for r in rows if r["toxicity"] >= 0.5]
    clean = rng.sample([r for r in rows if r["toxicity"] < 0.5], min(len(rows), 2 * len(toxic) + 10))
    for r in toxic + clean:
        level = 0 if r["toxicity"] < 0.2 else 1 if r["toxicity"] < 0.5 else 2
        yield row(rng, "Comment: " + r["text"],
                  [score("toxicity", "How toxic is the comment?", TOX, level),
                   noul("insult", "Is the comment insulting?", r["insult"] >= 0.5),
                   noul("threat", "Does the comment contain a threat?", r["threat"] >= 0.5),
                   noul("obscene", "Is the comment obscene?", r["obscene"] >= 0.5)])


def csqa(rng, split):
    for r in table("tau/commonsense_qa", split):
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        yield row(rng, "Commonsense question.", [choice(rng, "answer", r["question"], list(r["choices"]["text"]), labels.index(r["answerKey"]))])


def arc(rng, split):
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        for r in table("allenai/ai2_arc", split, cfg):
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels or len(labels) > 9:
                continue
            yield row(rng, "Science exam question.", [choice(rng, "answer", r["question"], list(r["choices"]["text"]), labels.index(r["answerKey"]))])


STARS = ["星1（とても悪い）", "星2", "星3", "星4", "星5（とても良い）"]


def amazon_ja(rng, split):
    for r in table("SetFit/amazon_reviews_multi_ja", split):
        qa = [score("stars", "このレビューの星の数はいくつですか？", STARS, int(r["label"]))]
        if r["label"] != 2:
            qa.append(noul("positive", "このレビューは商品に好意的ですか？", r["label"] >= 3))
        yield row(rng, "商品レビュー: " + r["text"], qa)


SOURCES = {  # name: (fn, train split, eval split)
    "boolq": (boolq, "train", "validation"),
    "mnli": (mnli, "train", "validation_matched"),
    "sst2": (sst2, "train", "validation"),
    "ag_news": (ag_news, "train", "test"),
    "go_emotions": (go_emotions, "train", "validation"),
    "civil_comments": (civil, "train", "validation"),
    "csqa": (csqa, "train", "validation"),
    "arc": (arc, "train", "validation"),
    "amazon_ja": (amazon_ja, "train", "validation"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=5000)
    ap.add_argument("--eval-per-source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, ev = [], []
    for name, (fn, tr, va) in SOURCES.items():
        t = list(fn(rng, tr))
        rng.shuffle(t)
        e = list(fn(rng, va))
        rng.shuffle(e)
        for x in t[: a.per_source]:
            train.append({**x, "source": name})
        for x in e[: a.eval_per_source]:
            ev.append({**x, "source": name})
        print(f"{name}: train {min(len(t), a.per_source)} / eval {min(len(e), a.eval_per_source)}", flush=True)
    rng.shuffle(train)
    rng.shuffle(ev)
    for fname, rows in (("train.jsonl", train), ("eval.jsonl", ev)):
        with open(out / fname, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("wrote", len(train), "train /", len(ev), "eval")


if __name__ == "__main__":
    main()

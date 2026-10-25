"""Anthropic HH-RLHF helpers: conversation prefixes for RLCD (train split) and gold
human-preference pairs for the preference-model evaluation of RLCD §5.1 / Table 5."""
from __future__ import annotations

import argparse
import gzip
import json
import random
import urllib.request

URL = "https://raw.githubusercontent.com/anthropics/hh-rlhf/master/{subset}/{split}.jsonl.gz"


def fetch(subset: str, split: str) -> list[dict]:
    with urllib.request.urlopen(URL.format(subset=subset, split=split)) as r:
        return [json.loads(l) for l in gzip.decompress(r.read()).decode("utf-8").splitlines() if l.strip()]


def split_last_turn(text: str) -> tuple[str, str]:
    """'...\\n\\nAssistant: reply' -> ('...\\n\\nAssistant:', 'reply')"""
    head, sep, reply = text.rpartition("\n\nAssistant:")
    return (head + sep).strip(), reply.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="harmless-base", choices=["harmless-base", "helpful-base"])
    ap.add_argument("--prefixes-out", required=True, help="train prefixes for djev.rlcd")
    ap.add_argument("--gold-out", required=True, help="test gold pairs as Choice examples")
    ap.add_argument("--n-gold", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)

    with open(a.prefixes_out, "w", encoding="utf-8") as f:
        for r in fetch(a.subset, "train"):
            f.write(json.dumps({"prefix": split_last_turn(r["chosen"])[0]}, ensure_ascii=False) + "\n")

    from .rlcd import TASKS, to_choice_example

    task = "harmless" if a.subset.startswith("harmless") else "helpful"
    n = 0
    with open(a.gold_out, "w", encoding="utf-8") as f:
        for r in fetch(a.subset, "test"):
            prefix, chosen = split_last_turn(r["chosen"])
            prefix_r, rejected = split_last_turn(r["rejected"])
            if prefix != prefix_r or not chosen or not rejected or chosen == rejected:
                continue
            instruction = rng.choice(TASKS[task])[2]
            f.write(json.dumps(to_choice_example(prefix, instruction, chosen, rejected, rng), ensure_ascii=False) + "\n")
            n += 1
            if n >= a.n_gold:
                break
    print("done", a.prefixes_out, a.gold_out, n)


if __name__ == "__main__":
    main()

"""RLCD preference-data simulation (arXiv:2307.12950), emitted as Jev Choice examples.

RLCD (§3.1, App. A, App. E):
  * for each conversation prefix p build p+ / p- by putting a contrasting description in
    parentheses before the final "Assistant:" and the *opposite* description before every
    earlier "Assistant:" (harmlessness: a random pair from Table 9; helpfulness: Table 11)
  * sample o+ ~ LM(p+), o- ~ LM(p-) with the same unaligned AR model, temperature 1;
    the response ends at "Human", is truncated at "Assistant", re-sampled up to 5 times
  * label o+ as preferred, no post-hoc scoring
  * train a preference model on (o+, o-)

Here the preference model *is* the diffusion decision model: each pair becomes a Choice
question in the scoring format of Table 8 (conversation, scoring instruction from Table 7
matching the sampled pair, two options in random order). Training on these examples with
`djev.sft` is the "train preference model" step. RLCD's final PPO step aligns a generator
and is outside the scope of a decision model.

Input: JSONL with {"prefix": "Human: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant:"}
(Anthropic HH format, as in the paper).
"""
from __future__ import annotations

import argparse
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import pick_device

# Table 9 (p+ description, p- description) and the Table 7 scoring instruction each pair
# corresponds to, in the same order.
HARMLESS = [
    ("helpful, honest, inoffensive response", "unhelpful, dishonest, toxic response",
     "Please choose the response that is the most helpful, honest, and harmless."),
    ("law-abiding, ethical response", "illegal, unethical response",
     "Please choose the assistant response that is as harmless and ethical as possible. Do NOT choose responses that are toxic, racist, or sexist, or that encourage or support illegal, violent, or unethical behavior. Above all the assistant’s response should be wise, peaceful, and ethical."),
    ("innocent response", "prejudiced response",
     "Compare the degree of harmfulness in the assistant responses and choose the one that’s less harmful. However, try to avoid choosing responses that are too preachy, obnoxious, or overly-reactive."),
    ("harmless, helpful response", "harmful, unhelpful response",
     "Choose the assistant response that is as harmless, helpful, polite, respectful, and thoughtful as possible without sounding overly-reactive of accusatory."),
    ("ethical and moral response", "unethical and immoral response",
     "Please choose the assistant response that’s more ethical and moral. Do NOT choose responses that exhibit toxicity, racism, sexism or any other form of physical or social harm."),
    ("inoffensive response", "offensive response",
     "Which of these assistant responses exhibits less harmful and more acceptable behavior? Choose the less harmful response."),
    ("correct, conscientious, socially acceptable response", "incorrect, unconscientious, socially unacceptable response",
     "Choose the assistant response that answers the human’s query in a more friendly, amiable, conscientious, and socially acceptable manner."),
    ("simulating a wise, ethical person", "simulating a foolish, unethical person",
     "Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say."),
    ("legal, ethical, moral response", "illegal, unethical, immoral response",
     "Choose the response that is less harmful, paying close attention to whether each response encourages illegal, unethical or immoral activity."),
    ("age-appropriate, family-friendly response", "inappropriate, not-family-friendly response",
     "Please consider these assistant responses and choose the one that is less harmful, more age-appropriate, and more family-friendly."),
    ("thoughtful response", "thoughtless response",
     "Choose the response that answers the human in the most thoughtful, respectful and cordial manner."),
    ("politically correct response", "racist, sexist, socially biased response",
     "Compare the assistant responses in terms of toxicity, racism, sexism and social bias, and choose the less harmful response."),
    ("simulating a very kind person", "simulating a really evil person",
     "Choose the response that sounds most similar to what a peaceful, ethical, and wise person like Martin Luther King Jr. or Mahatma Gandhi might say."),
    ("harmless, helpful, honest response", "harmful, unhelpful, dishonest response",
     "Which of the assistant responses is most harmless, helpful, and honest? Choose the best response based on these criteria."),
    ("ethical response", "toxic response",
     "Choose the assistant response that is more harmless, polite, pleasant, and least likely to offend a socially-aware audience."),
    ("ethically and morally aware response", "unethical and immoral response",
     "Choose the assistant response that demonstrates more ethical and moral awareness without sounding excessively condescending, reactive, annoying or condemnatory."),
]
# Table 11 / Table 10
HELPFUL = [("giving a helpful response", "giving an unhelpful response", "Please choose the response that is more helpful.")]
TASKS = {"harmless": HARMLESS, "helpful": HELPFUL}


def contrastive_prompts(prefix: str, pos: str, neg: str) -> tuple[str, str]:
    """p+ ends with 'Assistant (pos):', earlier turns get 'Assistant (neg):'; p- the reverse."""
    head, sep, _ = prefix.rstrip().rpartition("Assistant:")
    if not sep:
        raise ValueError("prefix must end with 'Assistant:'")

    def build(final: str, earlier: str) -> str:
        return head.replace("Assistant:", f"Assistant ({earlier}):") + f"Assistant ({final}):"

    return build(pos, neg), build(neg, pos)


def clean_response(text: str) -> str | None:
    """App. E: require 'Human' to mark the end, truncate at 'Assistant'."""
    if "Human" not in text:
        return None
    text = text.split("Human")[0]
    text = text.split("Assistant")[0]
    text = text.strip()
    return text or None


@torch.no_grad()
def sample_many(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int = 32, tries: int = 5):
    """o ~ LM(p) at temperature 1 for every prompt; failures (no "Human" end marker) are
    re-sampled up to `tries` times, then returned as None (App. E)."""
    out: list[str | None] = [None] * len(prompts)
    todo = list(range(len(prompts)))
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for _ in range(tries):
        failed = []
        for b in range(0, len(todo), batch_size):
            idx = todo[b : b + batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=max_new_tokens, pad_token_id=pad,
                                 stop_strings=["Human:"], tokenizer=tok)
            for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                resp = clean_response(tok.decode(row, skip_special_tokens=True))
                if resp:
                    out[i] = resp
                else:
                    failed.append(i)
        todo = failed
        if not todo:
            break
    return out


def sample_many_vllm(llm, prompts: list[str], max_new_tokens: int, seed: int, tries: int = 5):
    """Same contract as `sample_many`, using vLLM for throughput."""
    from vllm import SamplingParams

    out: list[str | None] = [None] * len(prompts)
    todo = list(range(len(prompts)))
    for t in range(tries):
        params = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=max_new_tokens,
                                stop=["Human:"], include_stop_str_in_output=True, seed=None)
        gens = llm.generate([prompts[i] for i in todo], params)
        failed = []
        for i, g in zip(todo, gens):
            resp = clean_response(g.outputs[0].text)
            if resp:
                out[i] = resp
            else:
                failed.append(i)
        todo = failed
        if not todo:
            break
    return out


def to_choice_example(prefix: str, instruction: str, o_pos: str, o_neg: str, rng: random.Random) -> dict:
    """Table 8 scoring layout as a Jev Choice question; option order randomised."""
    options = [o_pos, o_neg]
    rng.shuffle(options)
    return {
        "state": "Consider the following conversation between a human and an assistant:\n" + prefix.strip(),
        "questions": [{"type": "choice", "id": "preferred", "question": instruction, "options": options}],
        "answers": {"preferred": str(options.index(o_pos) + 1)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", default="Qwen/Qwen3-4B-Base", help="unaligned (base) AR LM, as in RLCD")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--task", choices=list(TASKS), default="harmless")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--gpu-mem", type=float, default=0.8, help="vLLM gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()][: a.limit]
    picks = [rng.choice(TASKS[a.task]) for _ in rows]
    pairs = [contrastive_prompts(r["prefix"], pos, neg) for r, (pos, neg, _) in zip(rows, picks)]
    prompts = [p for p, _ in pairs] + [n for _, n in pairs]
    if a.engine == "vllm":
        from vllm import LLM

        llm = LLM(a.generator, seed=a.seed, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len)
        outs = sample_many_vllm(llm, prompts, a.max_new_tokens, a.seed)
    else:
        tok = AutoTokenizer.from_pretrained(a.generator)
        model = AutoModelForCausalLM.from_pretrained(a.generator, dtype="auto").to(pick_device()).eval()
        outs = sample_many(model, tok, prompts, a.max_new_tokens, a.batch_size)
    o_pos, o_neg = outs[: len(pairs)], outs[len(pairs) :]
    kept = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r, (_, _, instruction), op, on in zip(rows, picks, o_pos, o_neg):
            if not op or not on:
                continue  # skipped after 5 failed re-samples (App. E)
            f.write(json.dumps(to_choice_example(r["prefix"], instruction, op, on, rng), ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept}/{len(rows)} pairs to {a.out}")


if __name__ == "__main__":
    main()

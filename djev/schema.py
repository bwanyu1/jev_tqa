"""Jev-style typed decisions (Choice / Noul / Score), serialised as a chat prompt whose
response is a strict JSON object — the structured-output setting of DiffusionGemma §9.3
(prompt style of its Figure 26).

Answer encodings are chosen so every allowed value is a single bare JSON literal:
  Choice -> option number 1..n, Noul -> true/false, Score -> level 1..n  (n <= 9)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Union


@dataclass
class Choice:
    id: str
    question: str
    options: list[str]

    def values(self) -> list[str]:
        return [str(i + 1) for i in range(len(self.options))]

    def describe(self) -> str:
        opts = "\n".join(f"    {i + 1}. {o}" for i, o in enumerate(self.options))
        return f'- "{self.id}" (choice; answer with the option number {self.values()[0]}-{self.values()[-1]}): {self.question}\n{opts}'


@dataclass
class Noul:
    id: str
    question: str

    def values(self) -> list[str]:
        return ["true", "false"]

    def describe(self) -> str:
        return f'- "{self.id}" (boolean; answer true or false): {self.question}'


@dataclass
class Score:
    id: str
    question: str
    levels: list[str]  # description of level 1..n, lowest first

    def values(self) -> list[str]:
        return [str(i + 1) for i in range(len(self.levels))]

    def describe(self) -> str:
        lv = "\n".join(f"    {i + 1}: {d}" for i, d in enumerate(self.levels))
        return f'- "{self.id}" (score; answer with the level number {self.values()[0]}-{self.values()[-1]}): {self.question}\n{lv}'


Question = Union[Choice, Noul, Score]

SYSTEM = (
    "You are a decision engine. Read the state and answer every question. "
    "Return the answers strictly as a JSON object mapping each question id to its answer. "
    "Do not include any explanations, or markdown formatting. Output only the raw JSON object."
)


def validate(questions: list[Question]) -> None:
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question ids: {ids}")
    for q in questions:
        if not 2 <= len(q.values()) <= 9:
            raise ValueError(f"{q.id}: 2..9 options/levels are supported")


def messages(state: str, questions: list[Question], media: list[dict] | None = None) -> list[dict]:
    """`media`: [{"type": "video" | "image", "path": ...}], placed before the text (the order
    DiffusionGemma's model card recommends for multimodal prompts)."""
    validate(questions)
    body = "State:\n" + state.strip() + "\n\nQuestions:\n" + "\n".join(q.describe() for q in questions)
    if not media:
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
    content = [{"type": m["type"], m["type"]: m["path"]} for m in media] + [{"type": "text", "text": body}]
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]


def answer_json(questions: list[Question], answers: dict[str, str]) -> str:
    """Training target, e.g. {"q1": 2, "q2": true}. `answers` maps id -> one of q.values()."""
    obj = {}
    for q in questions:
        v = answers[q.id]
        if v not in q.values():
            raise ValueError(f"{q.id}: {v!r} not in {q.values()}")
        obj[q.id] = json.loads(v)
    return json.dumps(obj, ensure_ascii=False)


def prompt_ids(tokenizer, state: str, questions: list[Question]) -> list[int]:
    return tokenizer.apply_chat_template(
        messages(state, questions),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,  # non-thinking mode; ignored by templates without it
        return_dict=False,
    )


MM_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids")


def prompt_inputs(tokenizer, processor, state: str, questions: list[Question], media: list[dict] | None = None):
    """(prompt ids, multimodal processor tensors or None)."""
    if not media:
        return prompt_ids(tokenizer, state, questions), None
    if processor is None:
        raise ValueError("this backbone has no multimodal processor")
    out = processor.apply_chat_template(
        messages(state, questions, media), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    )
    return out["input_ids"][0].tolist(), {k: out[k] for k in MM_KEYS if k in out}


def response_ids(tokenizer, state: str, questions: list[Question], answers: dict[str, str]) -> list[int]:
    """Assistant turn exactly as the chat template renders it, ending with the end-of-turn token."""
    return assistant_ids(tokenizer, messages(state, questions), answer_json(questions, answers))


def assistant_ids(tokenizer, msgs: list[dict], target: str) -> list[int]:
    """Tokens of the assistant turn `target` after `msgs`, as the chat template renders it."""
    kw = dict(tokenize=False, enable_thinking=False)
    head = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, **kw)
    full = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": target}], **kw)
    if full.startswith(head) and target in full[len(head):]:
        tail = full[len(head):]
        ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
    else:
        ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    stops = stop_ids(tokenizer)
    end = next((i for i, t in enumerate(ids) if t in stops), None)
    return ids[: end + 1] if end is not None else ids + [tokenizer.eos_token_id]


def stop_ids(tokenizer) -> set[int]:
    ids = {tokenizer.eos_token_id}
    for tok in ("<|im_end|>", "<end_of_turn>", "<turn|>", "<|eot_id|>"):
        i = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(i, int) and i != tokenizer.unk_token_id:
            ids.add(i)
    ids.discard(None)
    return ids


# ----------------------------------------------------------------------------- infilling
def infill_template(tokenizer, state: str, questions: list[Question], answers: dict[str, str] | None = None):
    """Answer canvas as a fixed template with one blank per question.

    Returns (tokens, slots, candidates): `tokens` = the assistant turn as the chat template renders
    it (gold values if `answers`, else each question's first allowed value as a placeholder);
    `slots[i]` = index of question i's value token; `candidates[i]` = {value: token id}. Every
    allowed value must be a single token in its context, otherwise ValueError.
    """
    ref = answers or {q.id: q.values()[0] for q in questions}
    toks = response_ids(tokenizer, state, questions, ref)
    text = tokenizer.decode(toks)
    spans, prev = [], ""
    for i in range(len(toks)):
        cur = tokenizer.decode(toks[: i + 1])
        spans.append((len(prev), len(cur)))
        prev = cur
    slots, candidates = [], []
    search_from = 0
    for q in questions:
        m = re.compile(r'"%s"\s*:\s*(true|false|\d)' % re.escape(q.id)).search(text, search_from)
        if not m:
            raise ValueError(f"{q.id}: value not found in the rendered answer")
        search_from = m.end()
        pos = next(i for i, (a, b) in enumerate(spans) if a <= m.start(1) < b)
        a, b = spans[pos]
        piece = text[a:b]
        if piece.strip() != m.group(1):
            raise ValueError(f"{q.id}: value shares its token ({piece!r})")
        cand = {}
        for v in q.values():
            ids = tokenizer(piece.replace(m.group(1), v), add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"{q.id}: value {v!r} is not a single token here")
            cand[v] = ids[0]
        slots.append(pos)
        candidates.append(cand)
    return toks, slots, candidates

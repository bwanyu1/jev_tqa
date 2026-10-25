import json
import random

import pytest
import torch

from djev import schema
from djev.decide import _readout
from djev.rlcd import HARMLESS, clean_response, contrastive_prompts, to_choice_example
from djev.sampler import CanvasResult, GenerationResult
from djev.sft import encode_example


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")


QS = [
    schema.Choice("route", "Which model should handle this?", ["small", "large", "human"]),
    schema.Noul("urgent", "Is the ticket urgent?"),
    schema.Score("tone", "How angry is the customer?", ["calm", "annoyed", "angry"]),
]


def test_response_ends_with_turn_end(tok):
    r = schema.response_ids(tok, "ticket text", QS, {"route": "2", "urgent": "true", "tone": "3"})
    text = tok.decode(r)
    assert json.loads(text[: text.rindex("}") + 1][text.index("{"):]) == {"route": 2, "urgent": True, "tone": 3}
    assert r[-1] in schema.stop_ids(tok)


def test_encode_example_layout(tok):
    ex = {"state": "s", "questions": [{"type": "noul", "id": "q", "question": "?"}], "answers": {"q": True}}
    ids, valid, lm, mm, n = encode_example(tok, ex, P=256, K=2, C=16)
    assert len(ids) == 256 + 32 and n == 1 and not valid[0] and not valid[-1] and valid[256 + 15] and not lm[-1] and mm is None


def test_readout_probabilities(tok):
    text_ids = tok('{"route": 2, "urgent": true, "tone": 3}', add_special_tokens=False)["input_ids"]
    C = 32
    toks = text_ids + [0] * (C - len(text_ids))
    V = len(tok)
    logits = torch.zeros(C, V)
    pos = next(i for i, t in enumerate(text_ids) if tok.decode([t]).strip() == "true")
    t_true = tok(tok.decode([text_ids[pos]]), add_special_tokens=False)["input_ids"][0]
    t_false = tok(tok.decode([text_ids[pos]]).replace("true", "false"), add_special_tokens=False)["input_ids"][0]
    logits[pos, t_true], logits[pos, t_false] = 2.0, 0.0
    gen = GenerationResult(tokens=text_ids, canvases=[CanvasResult(torch.tensor(toks), logits, 3)])
    d = _readout(tok, gen, tok.decode(text_ids), QS[1], C)
    assert d.answer == "true"
    assert d.probs["true"] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), rel=1e-4)
    d = _readout(tok, gen, tok.decode(text_ids), QS[0], C)
    assert d.answer == "2" and d.probs is not None and abs(sum(d.probs.values()) - 1) < 1e-5


def test_contrastive_prompts_match_rlcd_appendix_a():
    prefix = "Human: hi\n\nAssistant: hello\n\nHuman: Joe is so annoying\n\nAssistant:"
    pos, neg, _ = HARMLESS[0]
    p_pos, p_neg = contrastive_prompts(prefix, pos, neg)
    assert p_pos.endswith(f"Assistant ({pos}):") and f"Assistant ({neg}): hello" in p_pos
    assert p_neg.endswith(f"Assistant ({neg}):") and f"Assistant ({pos}): hello" in p_neg
    assert p_pos.replace(pos, "X").replace(neg, "Y") == p_neg.replace(neg, "X").replace(pos, "Y")


def test_clean_response_and_choice_label():
    assert clean_response(" Sure thing.\n\nHuman: ok") == "Sure thing."
    assert clean_response("no end marker") is None
    ex = to_choice_example("Human: x\n\nAssistant:", "Choose.", "good", "bad", random.Random(1))
    opts = ex["questions"][0]["options"]
    assert opts[int(ex["answers"]["preferred"]) - 1] == "good"


def test_truncated_chat_rows_use_whole_canvases(tok):
    long = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "word " * 300}], "truncated": True}
    ids, valid, lm, mm, n = encode_example(tok, long, P=64, K=2, C=256)
    assert n == 1 and all(valid[64:64 + 256]) and not any(valid[64 + 256:])
    assert schema.stop_ids(tok).isdisjoint(ids[64:64 + 256])  # no end-of-turn on a cut response
    short = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}
    ids, valid, lm, mm, n = encode_example(tok, short, P=64, K=2, C=256)
    assert n == 1 and ids[64 + 255] in schema.stop_ids(tok) and not any(valid[64 + 256:])

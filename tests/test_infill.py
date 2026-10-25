import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from djev import schema
from djev.infill import ar_decide, collate_infill, denoise_infill, diffusion_decide, encode_infill
from djev.model import DiffusionLM
from djev.sampler import SamplerConfig
from djev.sft import diffusion_loss


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096)
    return DiffusionLM(Qwen3ForCausalLM._from_config(cfg, attn_implementation="sdpa"))


ROW = {"state": "Comment: I love it but I am scared",
       "questions": [{"type": "noul", "id": "love", "question": "love?"}, {"type": "noul", "id": "fear", "question": "fear?"},
                     {"type": "score", "id": "intensity", "question": "how strong?", "levels": ["low", "mid", "high"]}],
       "answers": {"love": "true", "fear": "true", "intensity": "2"}}


def test_template_is_kept_and_only_blanks_change(tok, model):
    qs = schema.__dict__  # noqa: keep import used
    from djev.sft import parse_questions
    q = parse_questions(ROW["questions"])
    template, slots, cands = schema.infill_template(tok, ROW["state"], q)
    ctx = model.encode(torch.tensor([schema.prompt_ids(tok, ROW["state"], q)]))
    blanks, first, final, steps = denoise_infill(model, ctx, template, slots, SamplerConfig(canvas_length=64, max_steps=4))
    assert blanks.shape == (3,) and first.shape[0] == 3 and 1 <= steps <= 4
    dec, _ = diffusion_decide(model, tok, ROW["state"], q, SamplerConfig(canvas_length=64, max_steps=4))
    for qq in q:
        assert dec[qq.id]["answer"] in qq.values()
        assert abs(sum(dec[qq.id]["probs"].values()) - 1) < 1e-4


def test_ar_fills_left_to_right(tok, model):
    from djev.sft import parse_questions
    q = parse_questions(ROW["questions"])
    dec, forwards = ar_decide(model.backbone, tok, ROW["state"], q)
    assert forwards == 3 and all(dec[qq.id]["answer"] in qq.values() for qq in q)


def test_blank_only_loss(tok, model):
    e = encode_infill(tok, ROW, 256, 64)
    assert e is not None and sum(e[5]) == 3
    batch = collate_infill([e, e])
    loss, parts = diffusion_loss(model, batch, 256, 1, 64)
    loss.backward()
    assert torch.isfinite(loss) and "decoder" in parts

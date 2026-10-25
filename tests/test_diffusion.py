import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from djev.model import DiffusionLM
from djev.sampler import SamplerConfig, accepted_positions, denoise_canvas, generate
from djev.sft import diffusion_loss


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    bb = Qwen3ForCausalLM._from_config(cfg, attn_implementation="sdpa").eval()
    m = DiffusionLM(bb)
    torch.nn.init.normal_(m.self_cond.down.weight, std=0.02)  # non-trivial z for the tests
    return m


def test_decoder_sees_clean_prefix_only_through_cache(model):
    """forward_train (training layout, left padding, later canvases present) must give the same
    decoder logits as encode() + decode() at inference."""
    P, C, K, pad = 6, 5, 2, 3
    ctx = torch.randint(97, (1, P - pad))
    canv0 = torch.randint(97, (1, C))
    noisy = torch.randint(97, (1, C))
    p0 = torch.softmax(torch.randn(1, C, 97), -1)
    z = model.self_condition(p0).detach()

    # inference: H = Enc(ctx ⊕ canvas0); decode canvas 1
    cache = model.encode(torch.cat([ctx, canv0], 1))
    ref = model.decode(noisy, z, cache)
    assert len(cache) == P - pad + C  # canvas KV was cropped away

    clean = torch.cat([torch.zeros(1, pad, dtype=torch.long), ctx, canv0, torch.randint(97, (1, C))], 1)
    valid = torch.tensor([[False] * pad + [True] * (P - pad + K * C)])
    enc, dec = model.forward_train(clean, valid, noisy, torch.tensor([P + C]), z)
    torch.testing.assert_close(dec.float(), ref, atol=1e-4, rtol=1e-4)

    # encoder half equals a plain causal forward over the unpadded clean sequence
    plain = model.backbone(input_ids=clean[:, pad:]).logits
    torch.testing.assert_close(enc[:, pad:], plain, atol=1e-4, rtol=1e-4)


def test_canvas_is_bidirectional(model):
    cache = model.encode(torch.randint(97, (1, 4)))
    x = torch.randint(97, (1, 6))
    z = torch.zeros(1, 6, 32)
    a = model.decode(x, z, cache)
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 97
    b = model.decode(x2, z, cache)
    assert not torch.allclose(a[0, 0], b[0, 0])  # first position sees the last one


def test_entropy_bounded_selection():
    e = torch.tensor([0.05, 0.0, 0.3, 0.04, 0.02])
    # sorted: 0.0(1) 0.02(4) 0.04(3) 0.05(0) 0.3(2); prefix sums before each: 0,0,.02,.06,.11
    keep = accepted_positions(e, 0.1)
    assert keep.tolist() == [True, True, False, True, True]
    assert accepted_positions(torch.ones(4), 0.1).sum() == 1  # always accepts at least one


def test_sampler_runs(model):
    cfg = SamplerConfig(canvas_length=8, max_steps=6, max_canvases=2)
    g = torch.Generator().manual_seed(0)
    res = denoise_canvas(model, model.encode(torch.randint(97, (1, 5))), cfg, g)
    assert res.tokens.shape == (8,) and 1 <= res.steps <= 6
    out = generate(model, torch.randint(97, (1, 5)), stop_ids={-1}, cfg=cfg, generator=g)
    assert len(out.tokens) == 16 and len(out.canvases) == 2


def test_sft_loss_overfits(model):
    torch.manual_seed(0)
    P, K, C = 4, 2, 4
    clean = torch.randint(97, (2, P + K * C))
    valid = torch.ones_like(clean, dtype=torch.bool)
    batch = (clean, valid, valid.clone())
    params = [p for p in model.parameters()]
    sc_before = model.self_cond.up.weight.detach().clone()
    opt = torch.optim.Adam(params, lr=3e-3)
    first = None
    model.train()
    for _ in range(60):
        loss, _ = diffusion_loss(model, batch, P, K, C)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    assert loss.item() < 0.5 * first
    assert not torch.equal(model.self_cond.up.weight, sc_before)  # the self-conditioning FFW is trained


def test_gemma4_sliding_window_backbone():
    """Gemma 4 (12B/31B family) has sliding-window layers: training layout and cached
    inference must still agree once the context is longer than the window."""
    from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig

    cfg = Gemma4UnifiedTextConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=16,
                                  layer_types=(["sliding_attention"] * 2 + ["full_attention"]) * 2, sliding_window=8)
    torch.manual_seed(0)
    m = DiffusionLM(Gemma4UnifiedForCausalLM._from_config(cfg, attn_implementation="sdpa").eval())
    torch.nn.init.normal_(m.self_cond.down.weight, std=0.02)
    P, C, K, pad = 9, 5, 2, 2
    ctx, c0, noisy = torch.randint(300, (1, P - pad)), torch.randint(300, (1, C)), torch.randint(300, (1, C))
    z = m.self_condition(torch.softmax(torch.randn(1, C, 300), -1)).detach()
    ref = m.decode(noisy, z, m.encode(torch.cat([ctx, c0], 1)))
    clean = torch.cat([torch.zeros(1, pad, dtype=torch.long), ctx, c0, torch.randint(300, (1, C))], 1)
    valid = torch.tensor([[False] * pad + [True] * (P - pad + K * C)])
    _, dec = m.forward_train(clean, valid, noisy, torch.tensor([P + C]), z)
    torch.testing.assert_close(dec.float(), ref, atol=1e-4, rtol=1e-4)


def _tiny_qwen3_vl():
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    c = AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    c.text_config.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=16))
    c.text_config.rope_parameters = {**c.text_config.rope_parameters, "mrope_section": [2, 3, 3]}
    c.vision_config.update(dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                                deepstack_visual_indexes=[0]))
    torch.manual_seed(0)
    m = DiffusionLM(Qwen3VLForConditionalGeneration._from_config(c, attn_implementation="sdpa").eval())
    torch.nn.init.normal_(m.self_cond.down.weight, std=0.02)
    return m, c


def test_qwen3_vl_with_image_context():
    """Video/image context on the encoder side with M-RoPE: training layout == cached inference,
    and placeholder tokens never appear on the canvas."""
    m, c = _tiny_qwen3_vl()
    vs, img, ve = c.vision_start_token_id, c.image_token_id, c.vision_end_token_id
    grid = torch.tensor([[1, 4, 4]])  # 16 patches -> 4 tokens after 2x2 merge
    n_tok = 4
    pix = torch.randn(16, 3 * c.vision_config.temporal_patch_size * c.vision_config.patch_size ** 2)
    text = lambda n: torch.randint(1000, 2000, (1, n))
    prompt = torch.cat([text(3), torch.tensor([[vs] + [img] * n_tok + [ve]]), text(4)], 1)
    types = torch.zeros_like(prompt)
    types[0, 4:4 + n_tok] = 1
    mm = {"pixel_values": pix, "image_grid_thw": grid, "mm_token_type_ids": types}
    C = 5
    canv0, noisy = text(C), text(C)
    z = m.self_condition(torch.softmax(torch.randn(1, C, m.vocab_size), -1)).detach()

    ctx = m.encode(prompt, mm=mm)
    assert ctx.next_pos < prompt.shape[1]  # M-RoPE: the image occupies fewer positions than tokens
    m.encode(canv0, ctx)
    ref = m.decode(noisy, z, ctx)
    assert torch.isinf(ref[..., img]).all() or (ref[..., img] <= torch.finfo(ref.dtype).min).all()

    pad = 2
    clean = torch.cat([torch.zeros(1, pad, dtype=torch.long), prompt, canv0, text(C)], 1)
    valid = torch.ones_like(clean, dtype=torch.bool)
    valid[0, :pad] = False
    mm_tr = {**mm, "mm_token_type_ids": torch.cat([torch.zeros(1, pad, dtype=torch.long), types,
                                                   torch.zeros(1, 2 * C, dtype=torch.long)], 1)}
    start = torch.tensor([pad + prompt.shape[1] + C])
    _, dec = m.forward_train(clean, valid, noisy, start, z, mm_tr)
    torch.testing.assert_close(dec.float(), ref, atol=1e-4, rtol=1e-4)
    assert not torch.isin(m.uniform_tokens((5000,)), m.forbidden).any()

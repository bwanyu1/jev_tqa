"""AR -> discrete-diffusion conversion of an arbitrary HF causal LM.

Follows DiffusionGemma (arXiv:2608.00146) §3:
  * shared weights θ used as a causal Encoder (Eq. 3/4) and a bidirectional Decoder (Eq. 5)
  * the decoder sees the context only through the encoder KV cache H
  * self-conditioning z_t = FFW(p̂0 E) is fed back into the decoder (Eq. 6)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache, PreTrainedModel


class SelfConditioning(nn.Module):
    """FFW in Eq. (6). The last layer is zero-initialised so that a freshly converted
    model behaves exactly like the AR backbone when z_t = 0 is replaced by FFW(·)
    (implementation choice; the paper does not state the initialisation)."""

    def __init__(self, d_model: int, d_hidden: int | None = None):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.up = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


def _additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool (B, Q, K) -> additive float mask (B, 1, Q, K) accepted by every HF attention backend."""
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[:, None]


@dataclass
class Context:
    """KV cache H of clean tokens plus the next (text) position id to use after it.
    With M-RoPE backbones (Qwen3-VL) positions after an image/video are not the token count."""

    cache: DynamicCache
    next_pos: int = 0

    def __len__(self) -> int:
        return self.cache.get_seq_length()


MM_TOKEN_ATTRS = ("image_token_id", "video_token_id", "audio_token_id", "vision_start_token_id",
                  "vision_end_token_id", "image_token_index", "video_token_index", "boi_token_id",
                  "eoi_token_id", "boa_token_id", "eoa_token_id")


class DiffusionLM(nn.Module):
    def __init__(self, backbone: PreTrainedModel, self_cond: SelfConditioning | None = None):
        super().__init__()
        self.backbone = backbone
        self.self_cond = self_cond or SelfConditioning(self.text_config.hidden_size)
        self.self_cond.to(device=backbone.device, dtype=backbone.dtype)
        cfg = backbone.config
        ids = {getattr(cfg, a, None) for a in MM_TOKEN_ATTRS} | {getattr(self.text_config, a, None) for a in MM_TOKEN_ATTRS}
        # Multimodal placeholder tokens belong to the context only: they are never drawn as
        # noise nor predicted on the text canvas (needed so the backbone can locate the real
        # placeholders by embedding, and outside both papers).
        self.register_buffer("forbidden", torch.tensor(sorted(i for i in ids if isinstance(i, int)), dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------ helpers
    @property
    def text_config(self):
        """Multimodal checkpoints (e.g. Gemma 4 *ForConditionalGeneration) keep it under text_config."""
        cfg = self.backbone.config
        return getattr(cfg, "text_config", None) or cfg

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def dtype(self) -> torch.dtype:
        return self.backbone.dtype

    @property
    def vocab_size(self) -> int:
        return self.backbone.get_output_embeddings().weight.shape[0]

    def uniform_tokens(self, shape, generator=None) -> torch.Tensor:
        """Unif(V) over the text vocabulary (multimodal placeholders excluded)."""
        x = torch.randint(self.vocab_size, shape, device=self.device, generator=generator)
        if len(self.forbidden):
            bad = torch.isin(x, self.forbidden.to(x.device))
            while bad.any():
                x[bad] = torch.randint(self.vocab_size, (int(bad.sum()),), device=self.device, generator=generator)
                bad = torch.isin(x, self.forbidden.to(x.device))
        return x

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not len(self.forbidden):
            return logits
        bad = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        bad[self.forbidden.to(logits.device)] = True
        return logits.masked_fill(bad, torch.finfo(logits.dtype).min)

    def rope_positions(self, ids: torch.Tensor, valid: torch.Tensor, mm: dict | None) -> torch.Tensor:
        """(B, S) text positions, or (3, B, S) M-RoPE positions when images/videos are present."""
        if mm and hasattr(getattr(self.backbone, "model", None), "get_rope_index"):
            grids = {k: mm.get(k) for k in ("image_grid_thw", "video_grid_thw")}
            pos, _ = self.backbone.model.get_rope_index(
                ids, attention_mask=valid.long(), mm_token_type_ids=mm.get("mm_token_type_ids"), **grids
            )
            return pos.masked_fill(~valid[None], 0)
        return (valid.long().cumsum(-1) - 1).clamp(min=0)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings()(ids)

    def self_condition(self, p0: torch.Tensor) -> torch.Tensor:
        """z = FFW(p̂0 E), with E the (V, d) token-embedding matrix."""
        E = self.backbone.get_input_embeddings().weight
        return self.self_cond((p0.to(E.dtype) @ E))

    def attention_masks(self, allowed: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor):
        """Additive mask(s) for `allowed` (B, Q, K). Backbones with sliding-window layers
        (e.g. Gemma 4) get a per-layer-type dict so that those layers keep their window
        (q - k < window, the same rule the backbone's own causal masks use)."""
        cfg = self.text_config
        types = set(getattr(cfg, "layer_types", None) or [])
        full = _additive(allowed, self.dtype)
        if "sliding_attention" not in types:
            return full
        near = (q_pos[:, :, None] - k_pos[:, None, :]) < cfg.sliding_window
        return {"full_attention": full, "sliding_attention": _additive(allowed & near, self.dtype)}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def encode(self, ids: torch.Tensor, ctx: Context | None = None, mm: dict | None = None) -> Context:
        """H = Encoder_θ(h) (Eq. 3) or H ← H ⊕ Encoder_θ(x̂0) (Eq. 4). Causal attention.
        `mm` carries processor outputs (pixel_values_videos, video_grid_thw, ...) for a
        multimodal context; only the initial context may contain media (§3.1, footnote 2)."""
        ctx = ctx if ctx is not None else Context(DynamicCache())
        L = ids.shape[1]
        if mm:
            assert len(ctx) == 0, "media is only supported in the initial context"
            pos = self.rope_positions(ids, torch.ones_like(ids, dtype=torch.bool), mm)
            self.backbone(input_ids=ids, past_key_values=ctx.cache, position_ids=pos, use_cache=True, **mm)
            ctx.next_pos = int(pos.max()) + 1
        else:
            pos = torch.arange(ctx.next_pos, ctx.next_pos + L, device=ids.device)[None].expand(ids.shape[0], -1)
            self.backbone(input_ids=ids, past_key_values=ctx.cache, position_ids=pos, use_cache=True)
            ctx.next_pos += L
        return ctx

    @torch.no_grad()
    def decode(self, x_t: torch.Tensor, z_t: torch.Tensor, ctx: Context) -> torch.Tensor:
        """L_t = Decoder_θ(x_t, z_t, H) (Eq. 5): bidirectional over the canvas, full view of H.

        The canvas keys/values are removed from the cache afterwards, so H only ever holds
        clean (committed) tokens.
        """
        B, C = x_t.shape
        P = len(ctx)
        dev = x_t.device
        allowed = torch.ones(B, C, P + C, dtype=torch.bool, device=dev)
        pos = torch.arange(ctx.next_pos, ctx.next_pos + C, device=dev)[None].expand(B, -1)
        # key positions for the sliding-window rule (sequential backbones only)
        k_pos = torch.cat([torch.arange(ctx.next_pos - P, ctx.next_pos, device=dev)[None].expand(B, -1), pos], 1)
        out = self.backbone(
            inputs_embeds=self.embed(x_t) + z_t,
            attention_mask=self.attention_masks(allowed, pos, k_pos),
            position_ids=pos,
            past_key_values=ctx.cache,
            use_cache=True,
        )
        ctx.cache.crop(-C)  # drop the canvas K/V again (transformers v5: negative = tokens to remove)
        return self.mask_logits(out.logits.float())

    # ------------------------------------------------------------------ training
    def forward_train(
        self,
        clean: torch.Tensor,  # (B, S) prompt (+left pad) followed by K canvases, S = P + K*C
        valid: torch.Tensor,  # (B, S) bool, False on padding
        noisy: torch.Tensor,  # (B, C) noised copy of canvas k
        canvas_start: torch.Tensor,  # (B,) index in `clean` where canvas k starts (= P + k*C)
        z: torch.Tensor,  # (B, C, d) self-conditioning input (zeros for 50% of the batch)
        mm: dict | None = None,  # processor outputs for media inside the prompts
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward pass that yields both the encoder logits (over `clean`, causal) and the
        decoder logits (over `noisy`). Clean tokens never attend to the noisy canvas, so their
        hidden states are exactly Encoder_θ(clean); the noisy canvas attends to clean tokens
        before canvas k (= the KV cache H of §8) and bidirectionally to itself."""
        B, S = clean.shape
        C = noisy.shape[1]
        dev = clean.device
        idx = torch.arange(S, device=dev)

        allowed = torch.zeros(B, S + C, S + C, dtype=torch.bool, device=dev)
        causal = idx[None, :, None] >= idx[None, None, :]
        allowed[:, :S, :S] = causal & valid[:, None, :]
        allowed[:, S:, :S] = (idx[None, None, :] < canvas_start[:, None, None]) & valid[:, None, :]
        allowed[:, S:, S:] = True
        allowed[:, idx, idx] = True  # padded rows attend to themselves (avoids NaN)

        clean_pos = self.rope_positions(clean, valid, mm)  # (B, S) or (3, B, S)
        text_pos = clean_pos if clean_pos.ndim == 2 else clean_pos[0]
        start_pos = text_pos.gather(1, canvas_start[:, None])
        noisy_pos = start_pos + torch.arange(C, device=dev)[None]
        if clean_pos.ndim == 3:
            pos = torch.cat([clean_pos, noisy_pos[None].expand(3, -1, -1)], dim=2)
            text_all = pos[0]
        else:
            pos = text_all = torch.cat([clean_pos, noisy_pos], dim=1)

        embeds = torch.cat([self.embed(clean), self.embed(noisy) + z], dim=1)
        logits = self.backbone(
            inputs_embeds=embeds,
            attention_mask=self.attention_masks(allowed, text_all, text_all),
            position_ids=pos,
            use_cache=False,
            **(mm or {}),
        ).logits
        return logits[:, :S], self.mask_logits(logits[:, S:])

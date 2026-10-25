"""DiffusionGemma sampler: Algorithm 1 (entropy-bounded denoising + adaptive stopping)
inside the block-autoregressive loop of §3.1. Defaults are Table 2."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .model import Context, DiffusionLM


@dataclass
class SamplerConfig:
    canvas_length: int = 256  # C
    max_steps: int = 192  # N (Table 2 uses 48 after SD·RL; SFT-only models need more, §5 / Fig. 10)
    entropy_budget: float = 0.1  # b
    stop_entropy: float = 0.005  # e_stop
    tau_max: float = 0.8
    tau_min: float = 0.4
    max_canvases: int = 8


@dataclass
class CanvasResult:
    tokens: torch.Tensor  # (C,) x̂0|t
    logits: torch.Tensor  # (C, V) untempered L_t of the returning step
    steps: int  # N_k
    first_logits: torch.Tensor | None = None  # (C, V) L_1, prediction from the pure-noise canvas
    mean_probs: torch.Tensor | None = None  # (C, V) softmax(L_t) averaged over executed steps


@dataclass
class GenerationResult:
    tokens: list[int]
    canvases: list[CanvasResult] = field(default_factory=list)

    # §3.4 efficiency metrics
    @property
    def total_steps(self) -> int:
        return sum(c.steps for c in self.canvases)

    @property
    def tokens_per_forward(self) -> float:
        k = len(self.canvases)
        return len(self.tokens) / max(self.total_steps + k - 1, 1)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-30))).sum(-1)


def accepted_positions(e: torch.Tensor, budget: float) -> torch.Tensor:
    """Algorithm 1, lines 20–22: U = {π_1..π_k}, k = max{m : Σ_{j<m} e_{π_j} ≤ b}."""
    order = torch.sort(e, stable=True).indices  # non-decreasing entropy, ties by position
    before = torch.cumsum(e[order], 0) - e[order]  # Σ_{j<m} e_{π_j}
    k = int((before <= budget).sum())
    keep = torch.zeros_like(e, dtype=torch.bool)
    keep[order[:k]] = True
    return keep


@torch.no_grad()
def denoise_canvas(model: DiffusionLM, ctx: Context, cfg: SamplerConfig, generator=None, trace=None) -> CanvasResult:
    """Algorithm 1, lines 2–26, for a single canvas (batch size 1). If `trace` is a list,
    one record per denoising step is appended (input canvas, prediction, confidence, kept set)."""
    C, N, dev = cfg.canvas_length, cfg.max_steps, model.device
    dt = 1.0 / N
    x = model.uniform_tokens((1, C), generator)  # x_1 ~ Unif(V^C)
    z = torch.zeros(1, C, model.text_config.hidden_size, device=dev, dtype=model.dtype)
    prev = None
    first, psum = None, None

    for n in range(1, N + 1):
        t = 1.0 - (n - 1) * dt
        L = model.decode(x, z, ctx)[0]  # (C, V)
        first = L if first is None else first
        psum = torch.softmax(L, -1) if psum is None else psum + torch.softmax(L, -1)
        x0 = L.argmax(-1)
        tau = (cfg.tau_max - cfg.tau_min) * t + cfg.tau_min
        p0 = torch.softmax(L / tau, dim=-1)
        e = _entropy(p0)
        stop = n > 1 and e.mean() <= cfg.stop_entropy and torch.equal(x0, prev)
        if trace is not None:
            rec = {"step": n, "t": t, "tau": tau, "input": x[0].tolist(), "pred": x0.tolist(),
                   "conf": torch.softmax(L, -1).max(-1).values.tolist(), "mean_entropy": float(e.mean()),
                   "stopped": bool(stop)}
            trace.append(rec)
        if stop:
            return CanvasResult(x0, L, n, first, psum / n)  # adaptive stopping
        prev = x0
        if n < N:
            z = model.self_condition(p0[None])
            xD = torch.multinomial(p0, 1, generator=generator).squeeze(-1)
            keep = accepted_positions(e, cfg.entropy_budget)
            noise = model.uniform_tokens((C,), generator)
            x = torch.where(keep, xD, noise)[None]
            if trace is not None:
                rec["kept"] = keep.tolist()
    return CanvasResult(x0, L, N, first, psum / N)


@torch.no_grad()
def generate(
    model: DiffusionLM,
    prompt_ids: torch.Tensor,  # (1, L)
    stop_ids: set[int],
    cfg: SamplerConfig = SamplerConfig(),
    generator=None,
    mm: dict | None = None,
    trace: list | None = None,
) -> GenerationResult:
    """Block-AR generation (§3.1): encode context, denoise a canvas, append it to H, repeat
    until an end-of-turn token appears."""
    mm = {k: v.to(model.device) for k, v in mm.items()} if mm else None
    ctx = model.encode(prompt_ids.to(model.device), mm=mm)
    out = GenerationResult(tokens=[])
    for _ in range(cfg.max_canvases):
        steps = [] if trace is not None else None
        res = denoise_canvas(model, ctx, cfg, generator, steps)
        if trace is not None:
            trace.append(steps)
        out.canvases.append(res)
        toks = res.tokens.tolist()
        stop = next((i for i, tok in enumerate(toks) if tok in stop_ids), None)
        if stop is not None:
            out.tokens += toks[:stop]
            break
        out.tokens += toks
        model.encode(res.tokens[None], ctx)  # Eq. (4)
    return out

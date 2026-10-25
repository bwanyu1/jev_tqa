from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .model import DiffusionLM


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _backbone(path, dtype, device):
    """Text LMs load as *ForCausalLM; VLMs such as Qwen3-VL only register as image-text-to-text."""
    for auto in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            return auto.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device)
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
    raise ValueError(f"{path}: no causal-LM or image-text-to-text head")


def _chain(checkpoint: str | None) -> list[Path]:
    """Checkpoints to apply in order: a run started with `--init X` is a LoRA on top of X
    (e.g. Jev downstream SFT §8 on top of the general diffusion conversion §4)."""
    chain = []
    while checkpoint:
        ck = Path(checkpoint)
        chain.append(ck)
        meta = ck / "djev.json"
        checkpoint = json.loads(meta.read_text()).get("args", {}).get("init") if meta.exists() else None
    return chain[::-1]


def load(backbone: str, checkpoint: str | None = None, device: str | None = None, dtype=None):
    """Backbone = any HF causal LM (AR initialisation, DiffusionGemma §3). `checkpoint` is a
    directory written by `save` (LoRA adapter and/or full weights + self-conditioning FFW).
    Returns (model, tokenizer, processor); processor is None for text-only backbones."""
    device = device or pick_device()
    dtype = dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
    tok = AutoTokenizer.from_pretrained(checkpoint or backbone)
    try:
        proc = AutoProcessor.from_pretrained(backbone)
        proc = proc if hasattr(proc, "image_processor") or hasattr(proc, "video_processor") else None
    except Exception:
        proc = None
    base = _backbone(backbone, dtype, device)
    for ck in _chain(checkpoint):
        if (ck / "adapter_config.json").exists():
            from peft import PeftModel

            base = PeftModel.from_pretrained(base, ck).merge_and_unload()
        elif (ck / "backbone").exists():
            base = _backbone(ck / "backbone", dtype, device)
    model = DiffusionLM(base)
    if checkpoint and (Path(checkpoint) / "self_cond.pt").exists():
        model.self_cond.load_state_dict(torch.load(Path(checkpoint) / "self_cond.pt", map_location=device))
    model.eval()
    return model, tok, proc


def save(model: DiffusionLM, tokenizer, out: str, meta: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bb = model.backbone
    if hasattr(bb, "peft_config"):
        bb.save_pretrained(out)
    else:
        bb.save_pretrained(out / "backbone")
    torch.save(model.self_cond.state_dict(), out / "self_cond.pt")
    tokenizer.save_pretrained(out)
    (out / "djev.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

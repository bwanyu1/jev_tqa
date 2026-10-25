"""Render a djev.trace JSON as animated GIFs (one frame per denoising step).

Each frame shows the prediction x̂0 of that step as flowing text:
  * committed tokens (kept from the previous step by the entropy-bounded sampler) are drawn
    in ink on an accent background whose strength is the model's confidence;
  * tentative tokens (positions that were re-noised and are only guesses) are light grey.
Usage: python scripts/render_trace_gif.py trace.json out_dir [index ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 600
PAD = 36
BG = (246, 247, 249)
INK = (24, 33, 43)
MUTED = (120, 130, 140)
TENTATIVE = (178, 186, 194)
ACCENT = (61, 76, 202)
KEPT_EDGE = (196, 122, 0)
EOT = "<|im_end|>"

JP = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
JP_BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
f_body = ImageFont.truetype(JP, 21)
f_title = ImageFont.truetype(JP_BOLD, 24)
f_small = ImageFont.truetype(JP, 15)
f_mono = ImageFont.truetype(MONO, 15)


def mix(c1, c2, a):
    return tuple(int(c1[i] * (1 - a) + c2[i] * a) for i in range(3))


def clean(tok: str) -> str:
    return tok.replace("�", "▯")


def layout(tokens, x0, y0, x1, line_h):
    """Yield (token_index, text, x, y) wrapping at x1; '\\n' starts a new line."""
    x, y = x0, y0
    for i, t in enumerate(tokens):
        parts = t.split("\n")
        for j, part in enumerate(parts):
            if j > 0:
                x, y = x0, y + line_h
            if not part:
                continue
            w = f_body.getlength(part)
            if x + w > x1 and x > x0:
                x, y = x0, y + line_h
                part = part.lstrip(" ")
                w = f_body.getlength(part)
            yield i, part, x, y, w
            x += w


def frame(ex, s, prev_kept, n_steps):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, PAD - 6), ex["label"], font=f_title, fill=INK)
    d.text((PAD, PAD + 30), "入力: " + ex["prompt"][:70], font=f_small, fill=MUTED)

    # progress bar + step label
    bar_y = PAD + 62
    d.rounded_rectangle((PAD, bar_y, W - PAD, bar_y + 6), radius=3, fill=(222, 226, 231))
    d.rounded_rectangle((PAD, bar_y, PAD + (W - 2 * PAD) * s["step"] / n_steps, bar_y + 6), radius=3, fill=ACCENT)
    toks = s["pred"]
    end = next((i for i, t in enumerate(toks) if t == EOT), len(toks))
    kept_all = sum(prev_kept) if prev_kept else 0
    kept_body = sum(prev_kept[:end]) if prev_kept else 0
    d.text((PAD, bar_y + 14), f"step {s['step']:>2} / {n_steps}   t = {s['t']:.3f}", font=f_mono, fill=MUTED)
    d.text((PAD + 300, bar_y + 12), f"確定：本文 {kept_body} / {end} トークン　（キャンバス全体 {kept_all} / {len(toks)}）",
           font=f_small, fill=MUTED)
    toks = [clean(t) for t in toks[:end]]
    top = bar_y + 48
    line_h = 34
    for i, part, x, y, w in layout(toks, PAD, top, W - PAD, line_h):
        if y > H - 90:
            break
        committed = bool(prev_kept and prev_kept[i])
        if committed:
            a = 0.12 + 0.45 * s["conf"][i]
            d.rounded_rectangle((x - 1, y - 2, x + w + 1, y + 27), radius=4, fill=mix(BG, ACCENT, a))
            d.text((x, y), part, font=f_body, fill=INK)
        else:
            d.text((x, y), part, font=f_body, fill=TENTATIVE)
    if end < len(s["pred"]):
        d.text((PAD, H - 78), f"（以降 {len(s['pred']) - end} トークンは終了トークン）", font=f_small, fill=MUTED)

    # legend
    ly = H - 44
    d.rounded_rectangle((PAD, ly, PAD + 22, ly + 20), radius=4, fill=mix(BG, ACCENT, 0.5))
    d.text((PAD + 30, ly), "確定したトークン（色の濃さ＝確信度）", font=f_small, fill=MUTED)
    d.text((PAD + 330, ly), "灰色", font=f_small, fill=TENTATIVE)
    d.text((PAD + 370, ly), "＝まだ仮の予測（次のステップで作り直す）", font=f_small, fill=MUTED)
    if s.get("stopped"):
        d.text((W - PAD - 120, PAD - 2), "適応停止", font=f_title, fill=KEPT_EDGE)
    return img


def render(ex, out: Path):
    steps = ex["steps"]
    frames, durs = [], []
    for n, s in enumerate(steps):
        prev = steps[n - 1].get("kept") if n else None
        frames.append(frame(ex, s, prev, len(steps)))
        durs.append(450 if n < 3 else 220)
    # final frame: everything committed
    last = dict(steps[-1])
    frames.append(frame(ex, last, [True] * len(last["pred"]), len(steps)))
    durs.append(3500)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    print(out, len(frames), "frames")


LABELS = {
    "Free text (explanation)": "汎用拡散モデル：空が青い理由を3文で",
    "Multi-question decision (not trained)": "汎用拡散モデル：3問同時の判定（Jev化前）",
    "Free text (Japanese)": "汎用拡散モデル：日本語で説明",
    "Free text (DiffusionGemma Fig. 22 prompt)": "汎用拡散モデル：text diffusionを一文で",
    "HH gold pair (trained format)": "汎用拡散モデル：どちらの応答が良いか",
}


def main():
    src, out_dir, *idx = sys.argv[1:]
    data = json.load(open(src, encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in map(int, idx) if idx else range(len(data)):
        o = data[i]
        prompt = o["state"] if o["title"].startswith("Free") else "Support ticket + questions: urgent / anger / route" if "Multi" in o["title"] else o["state"][-70:]
        ex = {"label": LABELS.get(o["title"], o["title"]), "prompt": prompt.replace("\n", " "), "steps": o["canvases"][0]}
        render(ex, out / f"trace_{i}.gif")


if __name__ == "__main__":
    main()

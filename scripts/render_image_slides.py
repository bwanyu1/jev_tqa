"""Slides (1600x900 PNG) showing image questions with the answer of every stage.
Usage: python scripts/render_image_slides.py <dir with image_eval.jsonl, img_*.jsonl, picked.json, data/images> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
BG, INK, MUTED, LINE = (252, 252, 251), (24, 33, 43), (98, 108, 118), (226, 229, 233)
ACCENT, GOOD, BAD, NA = (42, 120, 214), (26, 127, 55), (198, 40, 40), (140, 146, 152)
JP, JPB = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
F = lambda size, bold=False: ImageFont.truetype(JPB if bold else JP, size)

MODELS = [("img_ar_base.jsonl", "① 未学習（元のモデル・自己回帰）", None),
          ("img_general_v3.jsonl", "② 拡散化（汎用 v3）", "final"),
          ("img_jev_v4.jsonl", "③ Jev化 v4（拡散）", "first"),
          ("img_ar_sft_v4.jsonl", "④ 参考：自己回帰＋同じデータでSFT", None)]


def wrap(d, text, font, width):
    lines, cur = [], ""
    for ch in text.replace("\n", " \n "):
        if ch == "\n":
            lines.append(cur.strip()); cur = ""; continue
        if d.textlength(cur + ch, font=font) > width:
            cut = cur.rfind(" ")
            if cut > len(cur) * 0.6:
                lines.append(cur[:cut]); cur = cur[cut + 1:] + ch
            else:
                lines.append(cur); cur = ch
        else:
            cur += ch
    if cur.strip():
        lines.append(cur.strip())
    return [l for l in lines if l]


def answer(r, readout):
    d = r["decisions"]["answer"]
    probs = d.get("probs")
    if readout and d.get("probs_by_readout") and readout in d["probs_by_readout"]:
        probs = d["probs_by_readout"][readout]
    if probs:
        a = max(probs, key=probs.get)
        return a, probs[a], None
    return d.get("answer"), None, r.get("raw")


def slide(n, total, ex, outs, label, base: Path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    src = "ScienceQA（理科・図）" if ex["source"] == "scienceqa" else "A-OKVQA（写真・常識）"
    d.text((48, 36), f"{n} / {total}", font=F(22), fill=MUTED)
    d.text((140, 30), label, font=F(32, True), fill=INK)
    d.text((W - 48, 40), src, font=F(22), fill=MUTED, anchor="ra")
    d.line((48, 92, W - 48, 92), fill=LINE, width=2)

    pic = Image.open(base / ex["media"][0]["path"]).convert("RGB")
    pic.thumbnail((720, 700))
    box = (48, 120, 48 + 720, 120 + 700)
    d.rectangle(box, fill=(240, 241, 243))
    img.paste(pic, (box[0] + (720 - pic.width) // 2, box[1] + (700 - pic.height) // 2))

    x0, x1 = 820, W - 48
    y = 118
    q = ex["questions"][0]
    ctx = ex["state"].split("\n", 1)[1].replace("Context: ", "") if "\n" in ex["state"] else ""
    for line in wrap(d, q["question"], F(28, True), x1 - x0)[:4]:
        d.text((x0, y), line, font=F(28, True), fill=INK); y += 40
    if ctx:
        for line in wrap(d, ctx, F(18), x1 - x0)[:3]:
            d.text((x0, y), line, font=F(18), fill=MUTED); y += 26
    y += 10
    gold = ex["answers"]["answer"]
    for k, opt in enumerate(q["options"], 1):
        is_gold = str(k) == gold
        if is_gold:
            d.rounded_rectangle((x0 - 8, y - 6, x1, y + 36), radius=8, fill=(232, 241, 252))
        d.text((x0, y), f"{k}.", font=F(24, is_gold), fill=ACCENT if is_gold else MUTED)
        d.text((x0 + 40, y), wrap(d, opt, F(24, is_gold), x1 - x0 - 140)[0], font=F(24, is_gold), fill=INK)
        if is_gold:
            d.text((x1 - 12, y + 2), "正解", font=F(20, True), fill=ACCENT, anchor="ra")
        y += 46

    y = max(y + 18, 520)
    d.text((x0, y), "各モデルの答え", font=F(20, True), fill=MUTED); y += 34
    for (fname, name, readout), r in zip(MODELS, outs):
        a, p, raw = answer(r, readout)
        d.line((x0, y - 6, x1, y - 6), fill=LINE, width=1)
        d.text((x0, y), name, font=F(20), fill=INK)
        if a is None:
            import re

            snippet = (raw or "").replace("\n", " ")[:22]
            m = re.search(r'"answer"\s*:\s*"?(\d)', raw or "")
            if m:
                ok = m.group(1) == gold
                d.text((x0 + 430, y), m.group(1), font=F(22, True), fill=GOOD if ok else BAD)
                d.text((x0 + 470, y), ("○ 正解" if ok else "× 不正解") + "（形式エラー）", font=F(20, True), fill=GOOD if ok else BAD)
            else:
                d.text((x0 + 430, y), "答えを書けず", font=F(20, True), fill=NA)
            d.text((x0 + 430, y + 26), f"出力: {snippet!r}", font=F(15), fill=NA)
        else:
            ok = a == gold
            d.text((x0 + 430, y), f"{a}", font=F(22, True), fill=GOOD if ok else BAD)
            d.text((x0 + 470, y), "○ 正解" if ok else "× 不正解", font=F(20, True), fill=GOOD if ok else BAD)
            if p is not None:
                d.text((x1, y + 2), f"確信度 {p * 100:.0f}%", font=F(18), fill=MUTED, anchor="ra")
        y += 62
    d.text((48, H - 34), "Qwen3-VL-4B-Instruct を拡散化（DiffusionGemma の手順）・画像は学習に一度も使っていない　③の確信度は first 読み出し",
           font=F(16), fill=MUTED)
    return img


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ev = [json.loads(l) for l in open(base / "image_eval.jsonl", encoding="utf-8")]
    res = {f: [json.loads(l) for l in open(base / f, encoding="utf-8")] for f, _, _ in MODELS}
    picked = json.load(open(base / "picked.json", encoding="utf-8"))
    for n, p in enumerate(picked, 1):
        i = p["i"]
        s = slide(n, len(picked), ev[i], [res[f][i] for f, _, _ in MODELS], p["label"], base)
        s.save(out / f"image_slide_{n}.png")
    print("wrote", len(picked), "slides to", out)


if __name__ == "__main__":
    main()

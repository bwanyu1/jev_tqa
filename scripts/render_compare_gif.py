"""Side-by-side GIF: ordinary autoregressive decoding (left) vs our diffusion sampler (right),
aligned so that frame k shows both models after k forward passes.

Usage: python scripts/render_compare_gif.py ar_trace.json diff_trace.json out.gif AR_INDEX DIFF_INDEX TITLE
"""
from __future__ import annotations

import json
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 680
PAD, GAP = 32, 32
COL = (W - 2 * PAD - GAP) // 2
BG = (246, 247, 249)
PANEL = (255, 255, 255)
INK = (24, 33, 43)
MUTED = (120, 130, 140)
TENTATIVE = (178, 186, 194)
ACCENT = (61, 76, 202)
AR_ACCENT = (0, 128, 110)
NEW = (196, 122, 0)
EOT = "<|im_end|>"

JP = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
JPB = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
f_body = ImageFont.truetype(JP, 18)
f_head = ImageFont.truetype(JPB, 20)
f_title = ImageFont.truetype(JPB, 24)
f_small = ImageFont.truetype(JP, 14)
f_count = ImageFont.truetype(JPB, 30)
LINE = 29


def mix(c1, c2, a):
    return tuple(int(c1[i] * (1 - a) + c2[i] * a) for i in range(3))


def clean(t):
    return t.replace("�", "▯")


def flow(d, tokens, styles, x0, y0, width, ymax):
    """Draw tokens as wrapped text; styles[i] = (fill, background or None)."""
    x, y = x0, y0
    for t, (fill, bg) in zip(tokens, styles):
        for j, part in enumerate(t.split("\n")):
            if j:
                x, y = x0, y + LINE
            if not part:
                continue
            w = f_body.getlength(part)
            if x + w > x0 + width and x > x0:
                x, y = x0, y + LINE
                part = part.lstrip(" ")
                w = f_body.getlength(part)
            if y > ymax:
                return
            if bg:
                d.rounded_rectangle((x - 1, y - 2, x + w + 1, y + LINE - 5), radius=4, fill=bg)
            d.text((x, y), part, font=f_body, fill=fill)
            x += w


def panel(d, x0, head, sub, color):
    d.rounded_rectangle((x0, 96, x0 + COL, H - 70), radius=10, fill=PANEL, outline=(222, 226, 231))
    d.rectangle((x0, 96, x0 + COL, 100), fill=color)
    d.text((x0 + 20, 112), head, font=f_head, fill=INK)
    d.text((x0 + 20, 140), sub, font=f_small, fill=MUTED)


def merge_bytes(ids, conf, tok):
    """Merge tokens that are partial UTF-8 bytes of one character (common for kanji), so each
    displayed piece is real text. Returns (pieces, confidences, forward index of each piece)."""
    pieces, confs, at, buf = [], [], [], []
    for i, t in enumerate(ids):
        buf.append(i)
        text = tok.decode([ids[j] for j in buf])
        if "\ufffd" in text and i + 1 < len(ids):
            continue
        pieces.append(text)
        confs.append(min(conf[j] for j in buf))
        at.append(buf[-1])
        buf = []
    return pieces, confs, at


def render(ar, df, out, title):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    steps = df["canvases"][0]
    ar_end = next((i for i, t in enumerate(ar["tokens"]) if t == EOT), len(ar["tokens"]))
    pieces, pconf, pat = merge_bytes(ar["ids"][:ar_end], ar["conf"], tok)
    n_ar = ar_end + 1  # the end-of-turn token also costs a forward pass
    n_df = len(steps)
    n = max(n_ar, n_df)
    frames, durs = [], []
    for k in range(1, n + 1):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((PAD, 24), title, font=f_title, fill=INK)
        d.text((PAD, 60), "1フレーム = 1回の推論（フォワードパス）。左右は同じ回数だけ推論したところを表示しています。", font=f_small, fill=MUTED)

        # left: AR
        lx = PAD
        panel(d, lx, "通常の推論（自己回帰）", "元の Qwen3-VL-4B。1回の推論で1トークンを左から順に書く", AR_ACCENT)
        shown = [j for j, a in enumerate(pat) if a < k]
        styles = [(INK, mix(PANEL, AR_ACCENT, 0.10 + 0.25 * pconf[j])) for j in shown]
        if shown and k <= ar_end and pat[shown[-1]] == k - 1:
            styles[-1] = (INK, mix(PANEL, NEW, 0.45))
        flow(d, [pieces[j] for j in shown], styles, lx + 20, 174, COL - 40, H - 140)
        done_ar = k >= n_ar
        d.text((lx + 20, H - 122), f"推論 {min(k, n_ar)} 回" + ("　完了" if done_ar else ""), font=f_count, fill=AR_ACCENT if done_ar else INK)

        # right: diffusion
        rx = PAD + COL + GAP
        panel(d, rx, "今回の推論（拡散）", "拡散化したモデル。1回の推論でキャンバス256トークン全体を更新する", ACCENT)
        s = steps[min(k, n_df) - 1]
        prev = steps[min(k, n_df) - 2].get("kept") if min(k, n_df) > 1 else None
        if k > n_df:
            prev = [True] * len(s["pred"])
        toks = s["pred"]
        end = next((i for i, t in enumerate(toks) if t == EOT), len(toks))
        styles = []
        for i in range(end):
            if prev and prev[i]:
                styles.append((INK, mix(PANEL, ACCENT, 0.12 + 0.45 * s["conf"][i])))
            else:
                styles.append((TENTATIVE, None))
        flow(d, [clean(t) for t in toks[:end]], styles, rx + 20, 174, COL - 40, H - 140)
        done_df = k >= n_df
        d.text((rx + 20, H - 122), f"推論 {min(k, n_df)} 回" + ("　完了" if done_df else ""), font=f_count, fill=ACCENT if done_df else INK)

        # legend
        ly = H - 48
        d.rounded_rectangle((PAD, ly, PAD + 20, ly + 18), radius=4, fill=mix(PANEL, NEW, 0.45))
        d.text((PAD + 28, ly), "いま書いたトークン", font=f_small, fill=MUTED)
        d.rounded_rectangle((PAD + 190, ly, PAD + 210, ly + 18), radius=4, fill=mix(PANEL, ACCENT, 0.45))
        d.text((PAD + 218, ly), "確定したトークン（濃さ＝確信度）", font=f_small, fill=MUTED)
        d.text((PAD + 470, ly), "灰色", font=f_small, fill=TENTATIVE)
        d.text((PAD + 505, ly), "＝拡散の仮の予測（次の推論で作り直す）", font=f_small, fill=MUTED)
        frames.append(img)
        durs.append(160)
    durs[-1] = 4000
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    print(out, len(frames), "frames; AR forwards", n_ar, "diffusion forwards", n_df)


def main():
    ar_path, df_path, out, ai, di, title = sys.argv[1:]
    ar = json.load(open(ar_path, encoding="utf-8"))[int(ai)]
    df = json.load(open(df_path, encoding="utf-8"))[int(di)]
    render(ar, df, out, title)


if __name__ == "__main__":
    main()

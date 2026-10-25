"""Speed vs quality figure for the AR baseline and each diffusion step budget N.
Usage: python scripts/plot_sweep.py bench_sweep.json quality_sweep.json out.png
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for path in ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"):
    font_manager.fontManager.addfont(path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc").get_name()

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
AR = "#eb6834"    # categorical slot 2
DIFF = "#2a78d6"  # categorical slot 1


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main():
    bench = json.load(open(sys.argv[1], encoding="utf-8"))
    qual = json.load(open(sys.argv[2], encoding="utf-8"))["summary"]
    out = sys.argv[3]
    steps = sorted(int(n) for n in bench["diffusion"])
    systems = ["ar"] + [f"N{n}" for n in steps]
    speed = {"ar": bench["ar"]} | {f"N{n}": bench["diffusion"][str(n)] for n in steps}
    label = {"ar": "自己回帰"} | {f"N{n}": f"拡散 N={n}" for n in steps}
    color = {s: AR if s == "ar" else DIFF for s in systems}

    fig = plt.figure(figsize=(12, 10.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1, 0.62], hspace=0.55, wspace=0.28)
    fig.suptitle("速度と品質：自己回帰 vs 拡散（デノイズ上限ステップ数 N ごと）", x=0.06, ha="left", fontsize=15,
                 color=TEXT, fontweight="bold")
    fig.text(0.06, 0.935, f"Qwen3-VL-4B-Instruct ／ 拡散版 {bench['checkpoint'].split('/')[-1]}　{bench['gpu']}・bf16・バッチ1・"
             f"同じ30問（英語19・日本語11、学習に未使用）・最大{bench['max_new_tokens']}トークン　品質＝GPT-OSS-20Bの10点満点採点",
             fontsize=9, color=TEXT2)

    # --- trade-off scatter
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=1)
    xs = [speed[f"N{n}"]["tokens_per_s"] for n in steps]
    ys = [qual[f"N{n}"]["judge_mean"] for n in steps]
    ax.plot(xs, ys, color=DIFF, linewidth=2, zorder=2)
    ax.scatter(xs, ys, s=70, color=DIFF, edgecolor=SURFACE, linewidth=2, zorder=3, label="拡散（N を変えたもの）")
    for n, x, y in zip(steps, xs, ys):
        ax.annotate(f"N={n}", (x, y), xytext=(8, 6), textcoords="offset points", fontsize=10, color=TEXT)
    ax.scatter([speed["ar"]["tokens_per_s"]], [qual["ar"]["judge_mean"]], s=90, marker="D", color=AR,
               edgecolor=SURFACE, linewidth=2, zorder=3, label="自己回帰（元のモデル）")
    ax.annotate("自己回帰", (speed["ar"]["tokens_per_s"], qual["ar"]["judge_mean"]), xytext=(8, -14),
                textcoords="offset points", fontsize=10, color=TEXT)
    ax.set_xlabel("速度（出力トークン/秒） →　右ほど速い", color=TEXT2, fontsize=10)
    ax.set_ylabel("品質（採点の平均, 10点満点） ↑", color=TEXT2, fontsize=10)
    ax.set_xlim(0, max(xs + [speed["ar"]["tokens_per_s"]]) * 1.15)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT2, loc="upper right")

    # --- small multiples
    names = [label[s] for s in systems]
    cols = [color[s] for s in systems]
    panels = [
        ("速度（トークン/秒）", [speed[s]["tokens_per_s"] for s in systems], "{:.1f}"),
        ("品質（採点の平均）", [qual[s]["judge_mean"] for s in systems], "{:.1f}"),
        ("繰り返し率（低いほど良い）", [100 * qual[s]["repetition_mean"] for s in systems], "{:.0f}%"),
    ]
    for k, (title, vals, fmt) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        style(ax)
        bars = ax.bar(range(len(systems)), vals, color=cols, width=0.68, edgecolor=SURFACE, linewidth=2)
        is_q = "品質" in title
        for i, (b, v) in enumerate(zip(bars, vals)):
            top = max(v, qual[systems[i]]["judge_mean_en"], qual[systems[i]]["judge_mean_ja"]) if is_q else v
            ax.text(b.get_x() + b.get_width() / 2, top + (0.35 if is_q else 0), fmt.format(v), ha="center", va="bottom",
                    fontsize=8, color=TEXT)
        ax.set_xticks(range(len(systems)), [n.replace("拡散 ", "") for n in names], fontsize=8, rotation=0)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        if "品質" in title:
            ax.set_ylim(0, 10)
            for i, s in enumerate(systems):
                ax.plot([i - 0.17], [qual[s]["judge_mean_en"]], marker="o", ms=3.5, color=TEXT2, ls="none")
                ax.plot([i + 0.17], [qual[s]["judge_mean_ja"]], marker="s", ms=3.5, color=TEXT, ls="none")
            ax.text(0.99, 0.98, "● 英語　■ 日本語", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TEXT2)

    # --- table
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    head = ["方式", "応答時間(中央値)", "トークン/秒", "推論1回あたり", "平均出力長", "採点(全体)", "採点(英語)", "採点(日本語)", "繰り返し率"]
    rows = [[label[s], f"{speed[s]['latency_p50_s']:.1f} 秒", f"{speed[s]['tokens_per_s']:.1f}",
             f"{speed[s]['tokens_per_forward']:.2f}", f"{speed[s]['avg_output_tokens']:.0f}",
             f"{qual[s]['judge_mean']:.2f}", f"{qual[s]['judge_mean_en']:.2f}", f"{qual[s]['judge_mean_ja']:.2f}",
             f"{100 * qual[s]['repetition_mean']:.1f}%"] for s in systems]
    t = ax.table(cellText=rows, colLabels=head, loc="center", cellLoc="right", colLoc="right")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(TEXT if r else TEXT2)
        if c == 0:
            cell.get_text().set_ha("left")
            cell.get_text().set_color(AR if rows[r - 1][0] == "自己回帰" else DIFF) if r else None
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

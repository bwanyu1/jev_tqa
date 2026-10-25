"""Break-even analysis: when is diffusion decoding faster than AR decoding?
Calibrated with the measured sweep (bench_sweep.json): t_ar = seconds per AR token,
t_df = seconds per diffusion forward pass (canvas C). Prefill is the same for both and ignored.
Usage: python scripts/plot_breakeven.py bench_sweep.json quality_sweep.json out.png
"""
from __future__ import annotations

import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for path in ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"):
    font_manager.fontManager.addfont(path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc").get_name()

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
DIFF, AR = "#2a78d6", "#eb6834"
RAMP = ["#9ec5f2", "#5c9ce6", "#2a78d6", "#1a4f94"]  # one hue, light -> dark, for S
GAIN = "#e8f1fc"  # tint of the diffusion hue for the "diffusion is faster" region


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main():
    b = json.load(open(sys.argv[1], encoding="utf-8"))
    q = json.load(open(sys.argv[2], encoding="utf-8"))["summary"]
    out = sys.argv[3]
    t_ar = 1 / b["ar"]["tokens_per_s"]
    steps = sorted(int(n) for n in b["diffusion"])
    d = {n: b["diffusion"][str(n)] for n in steps}
    t_df = sum(d[n]["tokens_per_forward"] / d[n]["tokens_per_s"] for n in steps) / len(steps)
    tpf_star = t_df / t_ar

    fig = plt.figure(figsize=(12.5, 10.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.3], hspace=0.32, wspace=0.22, bottom=0.12)
    fig.suptitle("拡散が自己回帰より得になる条件（損益分岐点）", x=0.06, ha="left", fontsize=15, color=TEXT, fontweight="bold")
    fig.text(0.06, 0.935, f"実測から推定：自己回帰 {t_ar * 1000:.0f} ms/トークン、拡散 {t_df * 1000:.0f} ms/推論（キャンバス256トークン）"
             f"　{b['gpu']}・bf16・バッチ1　プロンプトの前処理は両方に共通なので除外", fontsize=9, color=TEXT2)

    # ---- panel 1: tokens per forward -> speed-up
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    xs = [0.5 * 1.05 ** i for i in range(90)]
    ax.axvspan(tpf_star, 40, color=GAIN, zorder=0)
    ax.plot(xs, [x / tpf_star for x in xs], color=TEXT2, linewidth=1.5, zorder=2, label="理論線（速度比 = TPF ÷ %.1f）" % tpf_star)
    ax.axhline(1, color=AR, linewidth=1.5, zorder=2)
    ax.axvline(tpf_star, color=TEXT2, linewidth=1, linestyle="-", zorder=1)
    ax.text(tpf_star * 1.06, 0.33, f"分岐点\nTPF = {tpf_star:.1f}", fontsize=9, color=TEXT)
    ax.text(0.55, 1.07, "自己回帰と同じ速さ", fontsize=8.5, color=AR)
    meas_x = [d[n]["tokens_per_forward"] for n in steps]
    meas_y = [d[n]["tokens_per_s"] / b["ar"]["tokens_per_s"] for n in steps]
    ax.scatter(meas_x, meas_y, s=60, color=DIFF, edgecolor=SURFACE, linewidth=2, zorder=3, label="実測（今回の汎用拡散モデル）")
    for n, x, y in zip(steps, meas_x, meas_y):
        ax.annotate(f"N={n}  ×{y:.1f}", (x, y), xytext=(7, -12 if n == 16 else 4), textcoords="offset points", fontsize=8.5, color=TEXT)
    ax.scatter([19.74], [19.74 / tpf_star], s=70, marker="*", color=TEXT, zorder=3, label="DiffusionGemma の平均 TPF 19.7（参考）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 40)
    ax.set_ylim(0.25, 20)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks([0.25, 0.5, 1, 2, 5, 10, 20], ["×0.25", "×0.5", "×1", "×2", "×5", "×10", "×20"])
    ax.set_xlabel("推論1回あたりに進むトークン数（TPF）", color=TEXT2, fontsize=10)
    ax.set_ylabel("自己回帰に対する速度比", color=TEXT2, fontsize=10)
    ax.set_title("① 1回の推論で何トークン進めれば得か", loc="left", fontsize=11, color=TEXT)
    ax.text(0.98, 0.04, "青い領域＝拡散の方が速い", transform=ax.transAxes, ha="right", fontsize=8.5, color=DIFF)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="upper left")

    # ---- panel 2: output length x steps per canvas
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    C = 256
    Ls = sorted(set(int(round(1.06 ** i)) for i in range(0, 120)))
    Ls = [L for L in Ls if 1 <= L <= 1024]
    ax.axhspan(1, 100, color=GAIN, zorder=0)
    ax.axhline(1, color=AR, linewidth=1.5, zorder=2)
    for color, S, lab in zip(RAMP, (3, 12, 48, 192), ("3 ステップ（今回のJev判定の平均）", "12 ステップ（DiffusionGemma の平均）",
                                                      "48 ステップ", "192 ステップ（キャンバス1枚の中では分岐点なし）")):
        ys = [(L * t_ar) / (math.ceil(L / C) * (S + 1) * t_df) for L in Ls]
        ax.plot(Ls, ys, color=color, linewidth=2, zorder=2, label=lab)
        Lstar = (S + 1) * t_df / t_ar
        if Lstar <= C:
            ax.scatter([Lstar], [1], s=30, color=color, zorder=4)
            ax.annotate(f"{Lstar:.0f}", (Lstar, 1), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, color=TEXT)
    for n in (48, 192):
        L = d[n]["avg_output_tokens"]
        S = L / d[n]["tokens_per_forward"]
        y = d[n]["tokens_per_s"] / b["ar"]["tokens_per_s"]
        ax.scatter([L], [y], s=60, color=DIFF, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(f"実測 N={n}\n（約{S:.0f}ステップ）", (L, y), xytext=(-78, 8 if n == 48 else -26), textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1024)
    ax.set_ylim(0.01, 30)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    ax.set_yticks([0.01, 0.1, 1, 10], ["×0.01", "×0.1", "×1", "×10"])
    ax.set_xlabel("出力の長さ（トークン）", color=TEXT2, fontsize=10)
    ax.set_title("② 出力の長さとステップ数（キャンバス256トークン）", loc="left", fontsize=11, color=TEXT)
    ax.text(0.98, 0.04, "線の下の数字＝分岐点の長さ（これより長い出力なら拡散が速い）", transform=ax.transAxes, ha="right", fontsize=8, color=TEXT2)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="upper left", title="キャンバス1枚あたりのステップ数", title_fontsize=8)

    # ---- conclusions
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    lines = [
        ("速度の分岐点", f"推論1回で {tpf_star:.1f} トークン以上進めれば拡散が速い。今回の汎用モデルは N=96 以下で達成（N=16 で ×{d[16]['tokens_per_s'] / b['ar']['tokens_per_s']:.1f}）、N=192 では未達（×{d[192]['tokens_per_s'] / b['ar']['tokens_per_s']:.2f}）。"),
        ("出力の長さ", f"キャンバス1枚を S ステップで仕上げるなら、出力が約 {t_df / t_ar:.1f}×(S+1) トークンより長いと拡散が速い。短い出力ほど少ないステップで終わる必要がある。"),
        ("品質の条件", f"長い自由文では、どの N でも採点が {max(q[f'N{n}']['judge_mean'] for n in steps):.1f} 点以下（自己回帰 {q['ar']['judge_mean']:.1f} 点）。速度で勝てても品質で負ける。\n"
                       "今のモデルで得になるのは「決まった形の短い出力を、少ないステップで」出す判定（Jev）のような用途。"),
        ("その他の条件", "拡散の利点は同時リクエストが少ない時（バッチ小）に限られる（DiffusionGemma 論文：約32リクエストを超えると自己回帰が有利）。\n"
                       "今回の値は、どちらも最適化していない実装同士の比較。"),
    ]
    y = 1.0
    for head, body in lines:
        ax.text(0.0, y, head, fontsize=8.5, color=TEXT, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.11, y, body, fontsize=8.5, color=TEXT2, va="top", linespacing=1.5, transform=ax.transAxes)
        y -= 0.17 * (body.count("\n") + 1) + 0.04

    # ---- one-glance conclusion box
    from matplotlib.patches import FancyBboxPatch

    box = fig.add_axes([0.06, 0.02, 0.88, 0.08])
    box.axis("off")
    box.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06", transform=box.transAxes,
                                 facecolor=GAIN, edgecolor=DIFF, linewidth=1.5))
    box.text(0.02, 0.5, "結論", fontsize=15, color=DIFF, fontweight="bold", va="center", transform=box.transAxes)
    box.text(0.09, 0.70, "拡散が得なのは「短く決まった形の出力を、少ないステップで出す」とき（Jevの判定など）。",
             fontsize=12.5, color=TEXT, fontweight="bold", va="center", transform=box.transAxes)
    box.text(0.09, 0.28, f"速さは推論1回で {tpf_star:.1f} トークン以上進めれば勝ち（今回は N=96 以下）。長い文章は品質で大きく負けるので自己回帰を使う。",
             fontsize=11, color=TEXT, va="center", transform=box.transAxes)
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out, "t_ar", round(t_ar, 4), "t_df", round(t_df, 4), "TPF*", round(tpf_star, 2))


if __name__ == "__main__":
    main()

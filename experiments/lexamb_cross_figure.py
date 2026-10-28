# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""figN6 -- crossing with the natural stimuli released by Pouw et al.

Panel A (transcription measure, the Pouw schema):
  cluster 1  Published (their Figure 2, biasing context, wav2vec2-base-
             960h on the same lexamb materials): viable 15.8% vs
             unviable ~0%, N = 38/condition, digitized from the PDF.
  cluster 2  Our re-run of the same released archive (38 analyzed items
             x 2 recorded passes = 76): viable 14.5% [8.3, 24.1];
             the archive holds only viable-context recordings.
  cluster 3  Our synthetic lambda-continuum: the same ASR head returns
             non-lexical transcriptions on VocalTractLab audio
             (24/24), so the measure is not applicable there.

Panel B (probe measure, the kit's own instrument): rate of verdict
"n" (compensation) per layer, probes trained on the 18 canonical
synthetic training sentences:
  natural [m] frames (10 n->m items x 2 passes, n = 20)
  natural [N] frames (the 5 items excluded by Pouw et al.'s measure)
  synthetic references: licensed frames at lambda = 0.75 (n = 3) and
  canonical lambda = 0 (n = 3).

Style follows the kit figures (Pouw-palette bars, spines off top/
right, dashed grid, footnote, renderer-bbox assertions).

Outputs -> figures/figN6_lexamb_crossing.png (+ .pdf)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
OUT = os.path.join(KIT_ROOT, "figures", "figN6_lexamb_crossing.png")
OUT_PDF = os.path.join(KIT_ROOT, "figures", "figN6_lexamb_crossing.pdf")

COL = dict(viable="#33B44A", unviable="#6771B5", control="#BCBEC0")
EDGE = dict(viable="#1E7A2E", unviable="#3E477F", control="#6F7377")
GRAY_REF = "#9CA3AF"
GRAY_REF_D = "#4B5563"

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#444444",
})

Z = 1.959963985


def wilson(k, n, z=Z):
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)


# ---------------------------------------------------------------- data ----
# Pouw et al. Figure 2 (biasing context), digitized: 2.95 px/%, ticks
# every 59 px; bar tops y=521/521/530 on a baseline y=567.5 -> 15.8 /
# 15.8 / 12.7 % viable (neutral/biasing/random), unviable slivers <= 1
# px -> k = 0.  k/38: 6, 6, 5 viable; 0 unviable.
POUW_F2 = {"viable": (6, 38), "unviable": (0, 38)}
OURS_CTC = (11, 76)                      # pool over the two passes

LAYERS = ["CNN"] + [f"T{i}" for i in range(1, 13)]
PROBE = {
    "w2v2_base960h": {
        "nat_m":  [0, 10, 5, 0, 0, 5, 0, 0, 10, 0, 0, 0, 0],
        "nat_ng": [0, 50, 20, 0, 0, 0, 10, 20, 70, 30, 0, 0, 0],
        # v3.0 target-level cluster continuum, licensed pool (P1,P2,P4)
        "syn_075": [100, 100, 100, 100, 100, 100, 100, 100, 100, 33, 33, 33, 33],
        "syn_0":   [0, 0, 100, 100, 100, 100, 33, 33, 67, 0, 33, 33, 33],
    },
    "w2v2_base": {
        "nat_m":  [0, 5, 5, 0, 10, 5, 5, 5, 60, 35, 40, 70, 85],
        "nat_ng": [0, 30, 10, 10, 20, 30, 40, 60, 90, 90, 40, 30, 90],
        "syn_075": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 67, 100],
        "syn_0":   [0, 33, 33, 100, 67, 100, 67, 67, 33, 67, 0, 33, 33],
    },
}


def draw_bar(ax, x, rate, lo, hi, kind, lbl):
    ax.bar([x], [rate], width=0.17, color=COL[kind], alpha=0.95,
           edgecolor=EDGE[kind], linewidth=0.6, zorder=3)
    if rate < 0.5:
        ax.plot([x - 0.085, x + 0.085], [0, 0], color=EDGE[kind],
                lw=2.0, zorder=4, solid_capstyle="butt")
    ax.errorbar([x], [rate], yerr=[[max(0.0, rate - lo)],
                                   [max(0.0, hi - rate)]],
                fmt="none", ecolor="#222222", elinewidth=1.0,
                capsize=2.5, capthick=1.0, zorder=4)
    ax.text(x, hi + 3.2, lbl, ha="center", va="bottom", fontsize=7.5,
            color="#374151")


def fmt_pct(k, n):
    v = 100.0 * k / n
    return "0" if v < 0.05 else f"{v:.1f}"


def draw_na(ax, x, note):
    """Hatched not-applicable slot."""
    ax.bar([x], [3.2], width=0.17, color="none", edgecolor="#9CA3AF",
           linewidth=0.7, hatch="////", zorder=3)
    if note:
        ax.text(x, 7.5, note, ha="center", va="bottom", fontsize=7.2,
                color="#6B7280", style="italic")


def main():
    fig = plt.figure(figsize=(10.6, 5.6), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.155, 1, 0.845))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.06)

    # ================= Panel A: transcription measure ==================
    axA = fig.add_subplot(gs[0, 0])
    BT = blended_transform_factory(axA.transData, axA.transAxes)
    BW = 0.17
    g1 = [0.00, 0.24]      # published
    g2 = [0.90, 1.14]      # our re-run
    g3 = [1.94, 2.18]      # synthetic
    x_div = 1.62

    axA.axvspan(x_div, 2.46, color="#F2F5F8", zorder=0)
    axA.axvline(x_div, color="#9CA3AF", lw=0.8, ls=(0, (4, 3)), zorder=1)

    for cond, px in zip(("viable", "unviable"), g1):
        k, n = POUW_F2[cond]
        lo, hi = wilson(k, n)
        draw_bar(axA, px, 100.0 * k / n, lo, hi, cond, fmt_pct(k, n))
    k, n = OURS_CTC
    lo, hi = wilson(k, n)
    draw_bar(axA, g2[0], 100.0 * k / n, lo, hi, "viable", fmt_pct(k, n))
    draw_na(axA, g2[1], "not in\narchive")
    draw_na(axA, g3[0], None)
    draw_na(axA, g3[1], None)

    for pos, tags in ((g1, ["viable", "unviable"]),
                      (g2, ["viable", "unviable"]),
                      (g3, ["viable", "unviable"])):
        for px, tag in zip(pos, tags):
            axA.text(px, -0.030, tag, transform=BT, ha="center", va="top",
                     fontsize=7, color="#4B5563")

    # tier-2 bracket notes under clusters with hatched slots
    yb = -0.082
    for x0s, x1s, note in ((g2[1] - 0.085, g2[1] + 0.085,
                            "no unviable recordings"),
                           (g3[0] - 0.085, g3[1] + 0.085,
                            "CTC output non-lexical (24/24)")):
        axA.plot([x0s, x1s], [yb, yb], transform=BT, color="#6B7280",
                 lw=0.8, clip_on=False, zorder=5)
        for xe in (x0s, x1s):
            axA.plot([xe, xe], [yb, yb + 0.011], transform=BT,
                     color="#6B7280", lw=0.8, clip_on=False, zorder=5)
        axA.text((x0s + x1s) / 2.0, yb - 0.006, note, transform=BT,
                 ha="center", va="top", fontsize=7, color="#4B5563")

    axA.text(0.62, 112.0, "Natural stimuli (lexamb archive)",
             ha="center", va="center", fontsize=8.4, color="#374151",
             fontweight="bold", zorder=5)
    axA.text(2.06, 112.0, "Synthetic ($\\lambda$-continuum)",
             ha="center", va="center", fontsize=8.4, color="#374151",
             fontweight="bold", zorder=5)

    axA.set_xticks([np.mean(g1), np.mean(g2), np.mean(g3)])
    axA.set_xticklabels(["Published\n(Pouw et al., Fig. 2)",
                         "Our re-run\n(this study)",
                         "Our synthesis\n(this study)"], fontsize=8.2)
    axA.tick_params(axis="x", pad=42, length=2.5)
    axA.set_xlim(-0.20, 2.46)
    axA.set_ylim(0, 118)
    axA.set_yticks([0, 20, 40, 60, 80, 100])
    axA.set_ylabel("transcription compensation rate (%)")
    axA.set_title("A.  Transcription measure: wav2vec2-base-960h",
                  fontsize=9.0, color="#1a1a1a", loc="left", pad=10)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.grid(True, axis="y", ls="--", lw=0.4, alpha=0.25, zorder=0)

    # ================= Panel B: probe measure ==========================
    axesB = [fig.add_subplot(gs[0, 1])]
    gsB = gs[0, 1].subgridspec(1, 2, wspace=0.10)
    axesB = [fig.add_subplot(gsB[0, 0]), fig.add_subplot(gsB[0, 1])]
    titles = ["B.  Probe measure: 960h (CTC-tuned)",
              "self-supervised base"]
    x = np.arange(13)
    for ax, tag, ttl in zip(axesB, ("w2v2_base960h", "w2v2_base"), titles):
        d = PROBE[tag]
        ax.plot(x, d["syn_0"], color=GRAY_REF, lw=1.0, ls=":",
                zorder=2, label=None)
        ax.plot(x, d["syn_075"], color=GRAY_REF_D, lw=1.0, ls="--",
                zorder=2)
        ax.plot(x, d["nat_ng"], color=COL["unviable"], lw=1.8,
                marker="s", ms=3.4, zorder=4)
        ax.plot(x, d["nat_m"], color=COL["viable"], lw=1.8,
                marker="o", ms=3.8, zorder=5)
        # end-point value labels for the natural curves, min separation
        ym, yng = d["nat_m"][-1], d["nat_ng"][-1]
        ax.text(12.45, ym, f"{ym}", fontsize=7.2, color=EDGE["viable"],
                va="center", fontweight="bold")
        ylab = min(yng + 12.0, 97.0) if abs(yng - ym) < 10.0 else yng
        if abs(ylab - ym) < 9.0:
            ylab = max(ym - 12.0, 3.0)
        ax.text(12.45, ylab, f"{yng}", fontsize=7.2,
                color=EDGE["unviable"], va="center", fontweight="bold")
        if ylab != yng:
            ax.plot([12.15, 12.45], [yng, ylab], color=EDGE["unviable"],
                    lw=0.6, zorder=4)
        ax.set_title(ttl, fontsize=8.2, color="#374151", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [L if i % 2 == 0 or i == 12 else ""
             for i, L in enumerate(LAYERS)], fontsize=7.0)
        ax.set_xlim(-0.45, 13.75)
        ax.set_ylim(-3, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.25, zorder=0)
        if ax is axesB[0]:
            ax.set_ylabel("probe verdict rate: /n/ (%)", labelpad=4)
    axesB[0].text(6, -21.0, "layer", ha="center", va="top", fontsize=8.2,
                  color="#1a1a1a")

    handles = [
        mpatches.Patch(facecolor=COL["viable"], edgecolor=EDGE["viable"],
                       alpha=0.95, label="viable (A) / natural [m] (B)"),
        mpatches.Patch(facecolor=COL["unviable"],
                       edgecolor=EDGE["unviable"], alpha=0.95,
                       label="unviable (A) / natural [N] (B)"),
        plt.Line2D([], [], color=GRAY_REF_D, lw=1.0, ls="--",
                   label="synthetic $\\lambda=0.75$ (ref.)"),
        plt.Line2D([], [], color=GRAY_REF, lw=1.0, ls=":",
                   label="synthetic $\\lambda=0$ (ref.)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=4,
               frameon=False, fontsize=7.6, handlelength=1.7,
               columnspacing=1.2)

    # ---------------- footnote ------------------------------------------
    footnote = (
        "Error bars: 95% Wilson intervals. A (transcription measure): published bars\n"
        "digitized from Pouw et al.'s Figure 2 (biasing context, N = 38/condition;\n"
        "k/38 from pixel heights, 2.95 px/%: 6/38 viable, 0/38 unviable; their\n"
        "neutral/random viable bars read 6/38 and 5/38). Our re-run: the 38 analyzed\n"
        "items x 2 recorded passes of the released lexamb archive (11/76; passes 6/38\n"
        "and 5/38; n->m subset 2/20, stops 9/56); the archive holds only viable-\n"
        "context recordings, and the same ASR head returns non-lexical transcriptions\n"
        "on our VocalTractLab continuum (24/24) - hence the probe measure on the\n"
        "synthetic side. B (probe measure): rate of verdict /n/ per layer; the same 13\n"
        "layer probes as every experiment here, trained on the 18 canonical training\n"
        "sentences. Natural [m]: 10 n->m items x 2 passes (n = 20); natural [N]: the 5\n"
        "items excluded by Pouw et al.'s transcription measure (n = 10, exploratory).\n"
        "Synthetic references pool the 3 licensed frames per layer."
    )
    fig.text(0.012, 0.012, footnote, fontsize=7.2, color="#6B7280",
             va="bottom", ha="left", linespacing=1.35)

    # ---------------- assertions ----------------------------------------
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bb = fig.texts[-1].get_window_extent(ren)
    W_px = fig.get_figwidth() * fig.dpi
    assert bb.x1 <= W_px - 24, (
        f"footnote too wide: x1={bb.x1:.0f}px > {W_px - 24:.0f}px")
    assert bb.x0 >= 8 and bb.y0 >= 2, "footnote outside canvas"
    all_ax = [axA] + axesB
    for ax in all_ax:
        tbox = [t for t in ax.texts if t.get_text().strip()]
        for i in range(len(tbox)):
            for j in range(i + 1, len(tbox)):
                ba = tbox[i].get_window_extent(ren)
                bj = tbox[j].get_window_extent(ren)
                assert not ba.overlaps(bj), (
                    f"texts overlap: {tbox[i].get_text()[:28]!r} vs "
                    f"{tbox[j].get_text()[:28]!r}")
    # legend must not overlap any axes text
    leg = fig.legends[0] if fig.legends else None
    if leg:
        lb = leg.get_window_extent(ren)
        for ax in all_ax:
            for t in ax.texts:
                if t.get_text().strip():
                    tb = t.get_window_extent(ren)
                    assert not lb.overlaps(tb), (
                        f"legend overlaps {t.get_text()[:28]!r}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print("figure ->", OUT)


if __name__ == "__main__":
    main()

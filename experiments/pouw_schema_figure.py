#!/usr/bin/env python3
"""figN2 -- reproduction of Pouw et al. (2024) Figure 1, extended with the
lambda-continuum probing experiment of this kit.

Schema of the original (their Figure 1): grouped bars of the compensation
rate (%) per listener, three conditions -- Viable (green), Unviable
(indigo), Control (grey) -- with 95% Wilson confidence intervals.

Layout of this reproduction (v2, reassembled):
  * Left half, published comparison: Wav2Vec2 and the humans of Darcy et
    al. (2009), bars in the original Viable / Unviable / Control order,
    lowercase condition tags under the bars.
  * Right half, this study: the two wav2vec2 listeners read out by the
    13 layer probes.  Their four bars are ASSEMBLED ALONG THE LICENSED
    CONTINUUM -- Control (lambda=0, canonical [n]) -> orange midpoint
    (lambda=0.5, the condition the published design cannot express) ->
    Viable (lambda=1, fully assimilated [m]) -- closed by a bracket
    labeled "licensed continuum kli(n->m).pan (P1)"; the Unviable bar
    (unlicensed NV1, lambda=1) sits one wider step to the right, tagged
    "unlicensed (NV1)".  Both model groups share identical slot
    geometry, so bars align column-wise across listeners.

Pouw et al. values: bar heights digitized from their published Figure 1
(k/48 estimated per condition); Wilson intervals recomputed from k/48.
Our values: verdict rates of the 13 layer probes (note_stats.json).

Style follows the kit figures (Paul-Tol-adjacent palette, spines off
top/right, dashed grid, no internal title -- the caption lives in the
LaTeX note), 300 dpi.

Output -> figures/figN2_pouw_schema_reproduction.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

# --- Kit portability -------------------------------------------------------
KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
STATS = os.path.join(KIT_ROOT, "work", "lambda", "note_stats.json")
OUT = os.path.join(KIT_ROOT, "figures", "figN2_pouw_schema_reproduction.png")

# --- Pouw et al. (2024) Figure 1, digitized --------------------------------
POUW_N = 48
POUW_K = {
    "wav2vec2": {"viable": 36, "unviable": 19, "control": 48},
    "humans":   {"viable": 22, "unviable": 5,  "control": 45},
}
POUW_LBL = {
    "wav2vec2": {"viable": "75", "unviable": "40", "control": "100"},
    "humans":   {"viable": "46", "unviable": "11", "control": "94"},
}

# --- palette (bar colors sampled from the published figure) ----------------
COL = dict(viable="#33B44A", unviable="#6771B5", control="#BCBEC0",
           mid="#EE7733")
EDGE = dict(viable="#1E7A2E", unviable="#3E477F", control="#6F7377",
            mid="#9C4A1B")

# --- geometry --------------------------------------------------------------
BW = 0.17            # bar width
PITCH = 0.24         # bar-to-bar pitch inside a group / the continuum
GAP = 0.42           # between groups within a half
GAP_HALF = 0.62      # between the two halves (room for the divider)
STEP_NV = 0.42       # continuum -> unviable slot (wider: outside bracket)
YLIM = 118.0
LBL_OFF = 3.6

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#444444",
})


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n, in percent."""
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)


def ours(model, ctx, lam):
    e = json.load(open(STATS))["pools"][model][ctx][lam]
    return 100.0 * e["rate"], 100.0 * e["ci_lo"], 100.0 * e["ci_hi"]


def draw_bar(ax, x, rate, lo, hi, kind):
    ax.bar([x], [rate], width=BW, color=COL[kind], alpha=0.95,
           edgecolor=EDGE[kind], linewidth=0.6, zorder=3,
           hatch="///" if kind == "mid" else None)
    if rate < 0.5:
        # zero-height bar: short baseline tick so the slot stays visible
        ax.plot([x - BW / 2, x + BW / 2], [0, 0], color=EDGE[kind],
                lw=2.0, zorder=4, solid_capstyle="butt")
    ax.errorbar([x], [rate],
                yerr=[[max(0.0, rate - lo)], [max(0.0, hi - rate)]],
                fmt="none", ecolor="#222222", elinewidth=1.0, capsize=2.5,
                capthick=1.0, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(9.6, 5.5), constrained_layout=True)
    # reserve the bottom of the figure for the two annotation tiers, the
    # group tick labels, and the provenance footnote drawn with fig.text
    fig.get_layout_engine().set(rect=(0, 0.17, 1, 0.83))
    BT = blended_transform_factory(ax.transData, ax.transAxes)

    # ---- slots -------------------------------------------------------------
    # published half: Viable / Unviable / Control, original order
    g1 = [0.00, 0.24, 0.48]                       # Pouw wav2vec2
    g2 = [0.90, 1.14, 1.38]                       # Pouw humans
    # our half: continuum Control -> Mid -> Viable, then Unviable (NV1)
    g3 = [2.00, 2.24, 2.48, 2.90]                 # 960h
    g4 = [3.34, 3.58, 3.82, 4.24]                 # base
    centers = [np.mean(g1), np.mean(g2), np.mean(g3), np.mean(g4)]
    x_div = 1.69

    # ---- data --------------------------------------------------------------
    pouw = {}
    for who in ("wav2vec2", "humans"):
        pouw[who] = []
        for cond in ("viable", "unviable", "control"):
            k = POUW_K[who][cond]
            lo, hi = wilson(k, POUW_N)
            pouw[who].append((cond, 100.0 * k / POUW_N, lo, hi,
                              POUW_LBL[who][cond]))
    ourslots = [("control", "licensed", "0"), ("mid", "licensed", "0.5"),
                ("viable", "licensed", "1"), ("unviable", "unlicensed", "1")]
    our = {}
    for model in ("w2v2_base960h", "w2v2_base"):
        our[model] = []
        for kind, ctx, lam in ourslots:
            r, lo, hi = ours(model, ctx, lam)
            our[model].append((kind, r, lo, hi, f"{r:.1f}"))

    # ---- half shading + divider -------------------------------------------
    ax.axvspan(x_div, 4.50, color="#F2F5F8", zorder=0)
    ax.axvline(x_div, color="#9CA3AF", lw=0.8, ls=(0, (4, 3)), zorder=1)

    # ---- bars ---------------------------------------------------------------
    for pos, cells in [(g1, pouw["wav2vec2"]), (g2, pouw["humans"]),
                       (g3, our["w2v2_base960h"]), (g4, our["w2v2_base"])]:
        for (kind, r, lo, hi, vlbl), px in zip(cells, pos):
            draw_bar(ax, px, r, lo, hi, kind)
            ax.text(px, hi + LBL_OFF, vlbl, ha="center", va="bottom",
                    fontsize=7.5, color="#374151")

    # ---- per-bar sub-labels (tier 1) ----------------------------------------
    sub_pouw = ["viable", "unviable", "control"]
    for pos in (g1, g2):
        for px, tag in zip(pos, sub_pouw):
            ax.text(px, -0.026, tag, transform=BT, ha="center", va="top",
                    fontsize=7, color="#4B5563")
    sub_our = [r"$\lambda=0$", r"$\lambda=0.5$", r"$\lambda=1$",
               r"$\lambda=1$"]
    for pos in (g3, g4):
        for px, tag in zip(pos, sub_our):
            ax.text(px, -0.024, tag, transform=BT, ha="center", va="top",
                    fontsize=7, color="#4B5563")

    # ---- continuum bracket + unlicensed tag (tier 2) ------------------------
    yb = -0.078
    for pos in (g3, g4):
        x0, x1 = pos[0] - BW / 2 - 0.015, pos[2] + BW / 2 + 0.015
        ax.plot([x0, x1], [yb, yb], transform=BT, color="#6B7280",
                lw=0.8, clip_on=False, zorder=5)
        for xe in (x0, x1):
            ax.plot([xe, xe], [yb, yb + 0.011], transform=BT,
                    color="#6B7280", lw=0.8, clip_on=False, zorder=5)
        ax.text((x0 + x1) / 2.0, yb - 0.006,
                "licensed frames",
                transform=BT, ha="center", va="top", fontsize=7,
                color="#4B5563")
        ax.text(pos[3], yb - 0.006, "unlicensed frames",
                transform=BT, ha="center", va="top", fontsize=7,
                color="#3E477F")
    # extend the half divider through the annotation tiers below the axis
    ax.plot([x_div, x_div], [-0.004, -0.125], transform=BT,
            color="#9CA3AF", lw=0.8, ls=(0, (4, 3)), clip_on=False,
            zorder=5)

    # ---- half titles ---------------------------------------------------------
    ax.text(0.69, 110.0, "Published comparison (Pouw et al., 2024)",
            ha="center", va="center", fontsize=8.2, color="#374151",
            fontweight="bold", zorder=5)
    ax.text(3.12, 110.0, "This study: probe verdicts over 13 layers",
            ha="center", va="center", fontsize=8.2, color="#374151",
            fontweight="bold", zorder=5)

    # ---- axes ----------------------------------------------------------------
    ax.set_xticks(centers)
    ax.set_xticklabels(
        ["Wav2Vec2\n(Pouw et al., 2024)",
         "Humans\n(Darcy et al., 2009)",
         "wav2vec2-base-960h\n(CTC-tuned, this study)",
         "wav2vec2-base\n(self-supervised, this study)"], fontsize=8.4)
    ax.tick_params(axis="x", pad=40, length=2.5)
    ax.set_xlim(-0.24, 4.50)
    ax.set_ylim(0, YLIM)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("compensation rate (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.25, zorder=0)

    # ---- legend ---------------------------------------------------------------
    handles = [mpatches.Patch(facecolor=COL[c], edgecolor=EDGE[c],
                              alpha=0.95, hatch="///" if c == "mid" else None,
                              label=lbl)
               for c, lbl in [("viable", "Viable"),
                              ("unviable", "Unviable"),
                              ("control", "Control"),
                              ("mid", r"$\lambda=0.5$ midpoint"
                                      "\n(licensed, new)")]]
    fig.legend(handles=handles, loc="outside upper center", ncols=4,
               frameon=False, fontsize=8.2, handlelength=1.9,
               columnspacing=1.4)

    # ---- provenance footnote ----------------------------------------------------
    # NOTE: lines are kept <= ~178 chars: at 6.8 pt a longer line exceeds
    # the 9.6 in canvas and is clipped at the right edge (verified by the
    # bbox assertion below).
    footnote = (
        "Error bars: 95% Wilson confidence intervals. Left half "
        "(published): Pouw et al. (2024), N = 48 stimuli per condition; "
        "bar heights digitized from their Figure 1, intervals recomputed"
        "\n"
        "from the counts (36/19/48 model, 22/5/45 humans). Right half "
        "(this study): verdict rates pooled over the licensed frame pool "
        "(clean pan, mean boat, thin packet; 39 probes per cell)"
        "\n"
        "along the target-level cluster continuum (covtl-lambda v1.2.0): "
        "Control ($\\lambda=0$ canonical cluster), the midpoint "
        "$\\lambda=0.5$ (hatched, new condition), Viable ($\\lambda=1$);"
        "\n"
        "Unviable is the $\\lambda=1$ production of the unlicensed pool "
        "(clean tan, clean can, mean goat). Units differ across halves: "
        "48 independent stimuli per condition (left) vs 13 layer"
        "\n"
        "probes x 3 frames per cell (right; descriptive intervals).")
    fig.text(0.012, 0.012, footnote,
             fontsize=6.8, color="#6B7280", va="bottom", ha="left",
             linespacing=1.35)

    # ---- width assertion: footnote must fit inside the canvas ----------------
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bb = fig.texts[-1].get_window_extent(ren)
    W_px = fig.get_figwidth() * fig.dpi
    assert bb.x1 <= W_px - 24, (
        f"footnote too wide: x1={bb.x1:.0f}px > {W_px - 24:.0f}px; rewrap")
    assert bb.x0 >= 8 and bb.y0 >= 2, "footnote outside canvas"
    tbox = list(ax.texts)
    for i in range(len(tbox)):
        for j in range(i + 1, len(tbox)):
            ba = tbox[i].get_window_extent(ren)
            bbj = tbox[j].get_window_extent(ren)
            assert not ba.overlaps(bbj), (
                f"ax texts overlap: {tbox[i].get_text()[:30]!r} vs "
                f"{tbox[j].get_text()[:30]!r}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print("figure ->", OUT)


if __name__ == "__main__":
    main()

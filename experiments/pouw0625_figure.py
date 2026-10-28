# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""figN4 + figN5 -- lambda=0.625 production experiment figures.

figN4: the Pouw et al. (2024) Figure 1 schema with the lambda=0.625
PRODUCTION experiment, layout-identical to figN3 (which carried the
lambda=0.75 attempt): left half the published comparison (Wav2Vec2,
humans of Darcy et al. 2009), right half the two wav2vec2 listeners of
this study under the three Pouw conditions with the lambda=0.625
production degree.

figN5: the production-degree dose-response pooling ALL degrees tested
for the six-frame protocol: compensation rate of the licensed (viable)
and unlicensed (unviable) frame pools at lambda in {0, 0.625, 0.75, 1},
per listener, with 95% Wilson intervals; the pooled six-frame Control
of figN4 lies between the two lambda=0 points.  The figure makes the
two regularities of the dose-response visible at a glance: rates fall
monotonically with the production degree in BOTH frame classes, and
the licensed and unlicensed tracks sit on top of each other at every
degree (no licensing asymmetry).

Style follows the kit figures (bar colors sampled from the published
figure for reproduction fidelity; spines off top/right, dashed grid,
no suptitle -- captions live in the LaTeX note), 300 dpi, with
renderer-bbox assertions against text collisions and footnote
truncation (defects fixed in figN2/figN3 iterations).

Outputs -> figures/figN4_pouw_reproduction_lam0625.png
           figures/figN5_production_dose_response.png
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
STATS = os.path.join(KIT_ROOT, "work", "lambda0625", "pouw0625_stats.json")
OUT4 = os.path.join(KIT_ROOT, "figures",
                    "figN4_pouw_reproduction_lam0625.png")
OUT5 = os.path.join(KIT_ROOT, "figures",
                    "figN5_production_dose_response.png")

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
COL = dict(viable="#33B44A", unviable="#6771B5", control="#BCBEC0")
EDGE = dict(viable="#1E7A2E", unviable="#3E477F", control="#6F7377")
TRACK_LBL = {"viable": "Viable (licensed frames)",
             "unviable": "Unviable (unlicensed frames)"}

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#444444",
})

Z = 1.959963985


def wilson(k, n, z=Z):
    """95% Wilson score interval for k successes out of n, in percent."""
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)


# ============================================================= figN4 =======
def fig_n4():
    BW = 0.17            # bar width
    PITCH = 0.24         # bar-to-bar pitch inside a group
    YLIM = 118.0
    LBL_OFF = 3.6

    def ours(model, cond):
        e = json.load(open(STATS))["conditions"][model][cond]
        return (100.0 * e["rate"], 100.0 * e["ci_lo"], 100.0 * e["ci_hi"],
                e["k"], e["n"])

    def draw_bar(ax, x, rate, lo, hi, kind):
        ax.bar([x], [rate], width=BW, color=COL[kind], alpha=0.95,
               edgecolor=EDGE[kind], linewidth=0.6, zorder=3)
        if rate < 0.5:
            ax.plot([x - BW / 2, x + BW / 2], [0, 0], color=EDGE[kind],
                    lw=2.0, zorder=4, solid_capstyle="butt")
        ax.errorbar([x], [rate],
                    yerr=[[max(0.0, rate - lo)], [max(0.0, hi - rate)]],
                    fmt="none", ecolor="#222222", elinewidth=1.0,
                    capsize=2.5, capthick=1.0, zorder=4)

    fig, ax = plt.subplots(figsize=(9.6, 5.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.17, 1, 0.83))
    BT = blended_transform_factory(ax.transData, ax.transAxes)

    g1 = [0.00, 0.24, 0.48]                       # Pouw wav2vec2
    g2 = [0.90, 1.14, 1.38]                       # Pouw humans
    g3 = [2.00, 2.24, 2.48]                       # 960h
    g4 = [2.90, 3.14, 3.38]                       # base
    centers = [np.mean(g1), np.mean(g2), np.mean(g3), np.mean(g4)]
    x_div = 1.69
    conds = ["viable", "unviable", "control"]

    pouw = {}
    for who in ("wav2vec2", "humans"):
        pouw[who] = []
        for cond in conds:
            k = POUW_K[who][cond]
            lo, hi = wilson(k, POUW_N)
            pouw[who].append((cond, 100.0 * k / POUW_N, lo, hi,
                              POUW_LBL[who][cond]))
    our = {}
    for model in ("w2v2_base960h", "w2v2_base"):
        our[model] = []
        for cond in conds:
            r, lo, hi, k, n = ours(model, cond)
            our[model].append((cond, r, lo, hi, f"{r:.1f}"))

    ax.axvspan(x_div, 3.66, color="#F2F5F8", zorder=0)
    ax.axvline(x_div, color="#9CA3AF", lw=0.8, ls=(0, (4, 3)), zorder=1)

    for pos, cells in [(g1, pouw["wav2vec2"]), (g2, pouw["humans"]),
                       (g3, our["w2v2_base960h"]), (g4, our["w2v2_base"])]:
        for (kind, r, lo, hi, vlbl), px in zip(cells, pos):
            draw_bar(ax, px, r, lo, hi, kind)
            ax.text(px, hi + LBL_OFF, vlbl, ha="center", va="bottom",
                    fontsize=7.5, color="#374151")

    # per-bar sub-labels (tier 1)
    sub_pouw = ["viable", "unviable", "control"]
    for pos in (g1, g2):
        for px, tag in zip(pos, sub_pouw):
            ax.text(px, -0.026, tag, transform=BT, ha="center", va="top",
                    fontsize=7, color="#4B5563")
    sub_our = [r"$\lambda=0.625$", r"$\lambda=0.625$", r"$\lambda=0$"]
    for pos in (g3, g4):
        for px, tag in zip(pos, sub_our):
            ax.text(px, -0.024, tag, transform=BT, ha="center", va="top",
                    fontsize=7, color="#4B5563")

    # production bracket (tier 2, our half only)
    yb = -0.078
    for pos in (g3, g4):
        x0, x1 = pos[0] - BW / 2 - 0.015, pos[1] + BW / 2 + 0.015
        ax.plot([x0, x1], [yb, yb], transform=BT, color="#6B7280",
                lw=0.8, clip_on=False, zorder=5)
        for xe in (x0, x1):
            ax.plot([xe, xe], [yb, yb + 0.011], transform=BT,
                    color="#6B7280", lw=0.8, clip_on=False, zorder=5)
        ax.text((x0 + x1) / 2.0, yb - 0.006,
                "partial assimilation (3 frames each)",
                transform=BT, ha="center", va="top", fontsize=7,
                color="#4B5563")
    ax.plot([x_div, x_div], [-0.004, -0.125], transform=BT,
            color="#9CA3AF", lw=0.8, ls=(0, (4, 3)), clip_on=False,
            zorder=5)

    # half titles
    ax.text(0.69, 110.0, "Published comparison (Pouw et al., 2024)",
            ha="center", va="center", fontsize=8.2, color="#374151",
            fontweight="bold", zorder=5)
    ax.text(2.69, 110.0,
            "This study: probe verdicts, $\\lambda=0.625$ production",
            ha="center", va="center", fontsize=8.2, color="#374151",
            fontweight="bold", zorder=5)

    ax.set_xticks(centers)
    ax.set_xticklabels(
        ["Wav2Vec2\n(Pouw et al., 2024)",
         "Humans\n(Darcy et al., 2009)",
         "wav2vec2-base-960h\n(CTC-tuned, this study)",
         "wav2vec2-base\n(self-supervised, this study)"], fontsize=8.4)
    ax.tick_params(axis="x", pad=40, length=2.5)
    ax.set_xlim(-0.24, 3.66)
    ax.set_ylim(0, YLIM)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("compensation rate (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.25, zorder=0)

    handles = [mpatches.Patch(facecolor=COL[c], edgecolor=EDGE[c],
                              alpha=0.95, label=lbl)
               for c, lbl in [("viable", "Viable"),
                              ("unviable", "Unviable"),
                              ("control", "Control")]]
    fig.legend(handles=handles, loc="outside upper center", ncols=3,
               frameon=False, fontsize=8.2, handlelength=1.9,
               columnspacing=1.4)

    # provenance footnote (lines kept <= ~178 chars; asserted below)
    footnote = (
        "Error bars: 95% Wilson confidence intervals. Left half "
        "(published): Pouw et al. (2024), N = 48 stimuli per condition; "
        "bar heights digitized from their Figure 1, intervals recomputed"
        "\n"
        "from the counts (36/19/48 model, 22/5/45 humans). Right half "
        "(this study): production degree $\\lambda = 0.625$ of the "
        "target-level cluster morph (covtl-lambda v1.2.0:"
        "\n"
        "the C1 polar target rotates toward the C2 place; C2, context, "
        "lips, velum and glottis stay bitwise canonical); the second "
        "production degree tested after $\\lambda = 0.75$ (Figure N3)."
        "\n"
        "Viable: licensed frames (clean pan, mean boat, "
        "thin packet); Unviable: unlicensed frames (clean tan, clean"
        "\n"
        "can, mean goat); Control: $\\lambda = 0$ canonical of the same "
        "six frames. Rates are verdicts of the 13 layer probes per "
        "stimulus (39 probes per test condition, 78 for Control);"
        "\n"
        "layers are network depths, not independent samples "
        "(descriptive intervals). Reused-degree cells are byte-identical "
        "subsets of the 54-item continuum (single probing run).")
    fig.text(0.012, 0.012, footnote,
             fontsize=6.8, color="#6B7280", va="bottom", ha="left",
             linespacing=1.35)

    # width assertion: footnote must fit inside the canvas
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

    os.makedirs(os.path.dirname(OUT4), exist_ok=True)
    fig.savefig(OUT4, dpi=300)
    plt.close(fig)
    print("figure ->", OUT4)


# ============================================================= figN5 =======
def fig_n5():
    S = json.load(open(STATS))
    MODELS = [("w2v2_base960h", "wav2vec2-base-960h (CTC-tuned)"),
              ("w2v2_base", "wav2vec2-base (self-supervised)")]
    DEGS = ["0", "0.125", "0.25", "0.375", "0.5", "0.625", "0.75",
            "0.875", "1"]
    I_TEST = DEGS.index("0.625")

    def pool(model, frames, lam):
        """Read the 9-degree dose-response block (precomputed)."""
        e = S["dose_response"][model][lam][frames]
        return (100.0 * e["rate"], 100.0 * e["ci_lo"], 100.0 * e["ci_hi"],
                e["k"], e["n"])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9),
                             constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    x = np.arange(len(DEGS))
    XT = [r"$\lambda=0$" + "\n(canonical)"] + \
         [f"{float(d):g}" for d in DEGS[1:-1]] + [r"$\lambda=1$"]

    lbl_texts = []          # collected for the collision assertion
    for ax, (tag, title) in zip(axes, MODELS):
        ax.set_title(title, fontsize=8.6, color="#374151", pad=8)
        # subtle band marking the degree under test
        ax.axvspan(I_TEST - 0.38, I_TEST + 0.38, color="#F2F5F8", zorder=0)

        series = {}
        for key, frames in (("viable", "viable"), ("unviable", "unviable")):
            pts = [pool(tag, frames, lam) for lam in DEGS]
            series[key] = pts
            ax.plot(x, [p[0] for p in pts], color=COL[key], lw=1.4,
                    alpha=0.9, zorder=3, marker="o", markersize=5.5,
                    markerfacecolor=COL[key], markeredgecolor="white",
                    markeredgewidth=1.0)
            for xi, (r, lo, hi, _k, _n) in zip(x, pts):
                ax.errorbar([xi], [r],
                            yerr=[[max(0.0, r - lo)], [max(0.0, hi - r)]],
                            fmt="none", ecolor=EDGE[key], elinewidth=0.9,
                            capsize=2.2, capthick=0.9, alpha=0.85,
                            zorder=4)
        # value labels above the CI caps, with deterministic nudging
        lab = {}
        for key in ("viable", "unviable"):
            for xi, (r, lo, hi, _k, _n) in zip(x, series[key]):
                lab[(key, xi)] = hi + 2.5
        for xi in x:                       # resolve same-column collisions
            ya, yb_ = lab[("viable", xi)], lab[("unviable", xi)]
            if abs(ya - yb_) < 6.5:
                if ya >= yb_:
                    lab[("viable", xi)] = max(ya, yb_) + 6.5
                else:
                    lab[("unviable", xi)] = max(ya, yb_) + 6.5
        for key in ("viable", "unviable"):
            for xi, (r, lo, hi, _k, _n) in zip(x, series[key]):
                t = ax.text(xi, lab[(key, xi)], f"{series[key][xi][0]:.1f}",
                            ha="center", va="bottom", fontsize=6.2,
                            color=EDGE[key], zorder=6)
                lbl_texts.append(t)
        # tag over the degree under test
        t = ax.text(I_TEST, 120.5, "this test", ha="center", va="center",
                    fontsize=7.2, color="#6B7280", style="italic",
                    zorder=6)
        lbl_texts.append(t)

        ax.set_xticks(x)
        ax.set_xticklabels(XT, fontsize=7.4)
        ax.set_xlim(-0.55, len(DEGS) - 0.45)
        ax.set_ylim(0, 128)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("production degree of the cluster-target morph "
                      "(C1 polar target toward the C2 place)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.25, zorder=0)
    axes[0].set_ylabel("compensation rate (%)")

    handles = [mpatches.Patch(facecolor=COL[c], edgecolor=EDGE[c],
                              alpha=0.95, label=lbl)
               for c, lbl in [("viable", "Viable (licensed frames)"),
                              ("unviable", "Unviable (unlicensed frames)")]]
    fig.legend(handles=handles, loc="outside upper center", ncols=2,
               frameon=False, fontsize=8.2, handlelength=1.9,
               columnspacing=1.4)

    footnote = (
        "Error bars: 95% Wilson confidence intervals (descriptive; the 13 "
        "probes of one stimulus are network depths, not independent "
        "samples). Viable / Unviable:"
        "\n"
        "verdict rates pooled over the three licensed (clean pan, mean "
        "boat, thin packet) / unlicensed (clean tan, clean can, mean "
        "goat) frames at each production"
        "\n"
        "degree $\\lambda$ of the target-level cluster morph "
        "(covtl-lambda v1.2.0); the $\\lambda=0$ points of the two "
        "tracks are the canonical cluster productions of the same "
        "frame pools"
        "\n"
        "(item effect at baseline). The pooled six-frame Control of "
        "Figure N4 lies between the two $\\lambda=0$ points (64.1\\% "
        "CTC-tuned, 61.5\\% self-supervised)."
        "\n"
        "All nine degrees come from the single 54-item continuum probing "
        "run (one instrument throughout); $\\lambda = 0.625$ is the test "
        "of Figure N4.")
    fig.text(0.012, 0.012, footnote,
             fontsize=6.8, color="#6B7280", va="bottom", ha="left",
             linespacing=1.35)

    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bb = fig.texts[-1].get_window_extent(ren)
    W_px = fig.get_figwidth() * fig.dpi
    assert bb.x1 <= W_px - 24, (
        f"footnote too wide: x1={bb.x1:.0f}px > {W_px - 24:.0f}px; rewrap")
    assert bb.x0 >= 8 and bb.y0 >= 2, "footnote outside canvas"
    for i in range(len(lbl_texts)):
        for j in range(i + 1, len(lbl_texts)):
            ba = lbl_texts[i].get_window_extent(ren)
            bbj = lbl_texts[j].get_window_extent(ren)
            assert not ba.overlaps(bbj), (
                f"label overlap: {lbl_texts[i].get_text()!r} vs "
                f"{lbl_texts[j].get_text()!r}")

    os.makedirs(os.path.dirname(OUT5), exist_ok=True)
    fig.savefig(OUT5, dpi=300)
    plt.close(fig)
    print("figure ->", OUT5)


if __name__ == "__main__":
    fig_n4()
    fig_n5()

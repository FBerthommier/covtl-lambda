# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Figure for the companion note (three anchor conditions, Pouw-style).

figN1, three panels:
  (a) compensation rate (%) at lambda = 0 / 0.5 / 1.0, licensed context P1
      ("clean pan"), grouped bars for the two models, 95% Wilson CIs;
  (b) same, unlicensed context NV1 ("clean tan");
  (c) Spearman rho(lambda, P(n)) per layer for the four model x context
      combinations, with the exact two-tailed p = .05 critical value
      (n = 9, |rho| >= 0.70) as a dotted line.

Style follows the synthesis-report figures: Paul Tol palette (CTC-tuned
blue #4477AA, self-supervised purple #AA3377, licensed solid, unlicensed
dashed, circles for P1, squares for NV1), spines off top/right, dashed
grid, no internal titles (captions live in LaTeX), 300 dpi.

Reads work/lambda/note_stats.json (produced by note_stats.py).
Output -> figures/figN1_compensation_rates.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Kit portability -------------------------------------------------------
KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
STATS = os.path.join(KIT_ROOT, "work", "lambda", "note_stats.json")
OUT = os.path.join(KIT_ROOT, "figures", "figN1_compensation_rates.png")

LAYERS = ["CNN"] + [f"T{i}" for i in range(1, 13)]
ANCHORS = ["0", "0.5", "1"]
GROUP_LABELS = [r"$\lambda=0$", r"$\lambda=0.5$", r"$\lambda=1.0$"]
TOL = dict(blue="#4477AA", cyan="#66CCEE", green="#228833", yellow="#CCBB44",
           red="#EE6677", purple="#AA3377", grey="#BBBBBB", black="#000000")
MODELS = [("w2v2_base960h", TOL["blue"], "CTC-tuned (wav2vec2-base-960h)"),
          ("w2v2_base", TOL["purple"], "self-supervised (wav2vec2-base)")]

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#444444",
})


def rate_panel(ax, stats, ctx):
    """Pouw-Figure-1-style grouped bars with Wilson CIs for one pool."""
    xs = np.arange(len(ANCHORS))
    w = 0.36
    for i, (tag, color, _) in enumerate(MODELS):
        comp = stats["pools"][tag][ctx]
        rates = np.array([comp[a]["rate"] for a in ANCHORS]) * 100.0
        lo = np.array([comp[a]["ci_lo"] for a in ANCHORS]) * 100.0
        hi = np.array([comp[a]["ci_hi"] for a in ANCHORS]) * 100.0
        pos = xs + (i - 0.5) * (w + 0.03)
        ax.bar(pos, rates, width=w, color=color, alpha=0.88,
               edgecolor="#333333", linewidth=0.5, zorder=3)
        ax.errorbar(pos, rates, yerr=[rates - lo, hi - rates], fmt="none",
                    ecolor="#222222", elinewidth=1.0, capsize=2.5,
                    capthick=1.0, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(GROUP_LABELS, fontsize=8.2)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel(r"assimilation strength $\lambda$")
    ax.set_ylabel("compensation rate (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.25, zorder=0)


def main():
    stats = json.load(open(STATS))

    fig, axes = plt.subplots(
        1, 3, figsize=(7.4, 3.5), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.5]))

    # (a) licensed pool, (b) unlicensed pool ------------------------------
    rate_panel(axes[0], stats, "licensed")
    rate_panel(axes[1], stats, "unlicensed")

    # (c) Spearman rho per layer (pooled continuum) -------------------------
    ax = axes[2]
    x = np.arange(len(LAYERS))
    combos = [("w2v2_base960h", "licensed", TOL["blue"], "-", "o",
               "CTC-tuned, licensed"),
              ("w2v2_base960h", "unlicensed", TOL["blue"], "--", "s",
               "CTC-tuned, unlicensed"),
              ("w2v2_base", "licensed", TOL["purple"], "-", "o",
               "self-sup., licensed"),
              ("w2v2_base", "unlicensed", TOL["purple"], "--", "s",
               "self-sup., unlicensed")]
    for tag, ctx, color, ls, mk, lbl in combos:
        y = [stats["pooled_continuum"][tag][ctx]["spearman"][l]["rho"]
             for l in LAYERS]
        ax.plot(x, y, ls, color=color, lw=1.6, ms=3.2, marker=mk,
                label=lbl, mew=0.6, markeredgecolor="white")
    crit = stats["spearman_critical_n9"]["rho_two_tailed_p05"]
    ax.axhline(-crit, color=TOL["black"], lw=0.8, ls=":")
    ax.axhline(+crit, color=TOL["black"], lw=0.8, ls=":")
    ax.text(11.9, -crit - 0.06, "exact $p = .05$", ha="right", va="top",
            fontsize=6.8, color="#222222")
    ax.axhline(0.0, color="#999999", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYERS, fontsize=6.4, rotation=45, ha="right")
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.12, 1.12)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_ylabel(r"$\rho(\lambda,\ P(/n/))$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls="--", lw=0.4, alpha=0.25)

    # panel letters
    for a, s in zip(axes, ["(a)", "(b)", "(c)"]):
        a.text(0.015, 1.04, s, transform=a.transAxes, fontsize=9.5,
               fontweight="bold", va="bottom", ha="left")

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4,
               frameon=False, fontsize=7.2, handlelength=2.2,
               columnspacing=1.2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print("figure ->", OUT)


if __name__ == "__main__":
    main()

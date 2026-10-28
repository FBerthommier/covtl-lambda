#!/usr/bin/env python3
"""KIT30 step 22 -- the two lambda-continuum figures of the synthesis
document, regenerated for the v3.0 (target-level cluster) continuum.

figL1: 2x2 heatmaps of P(underlying /n/) over layer x lambda
       (rows: w2v2_base960h / w2v2_base; cols: licensed P1 / unlicensed
       NV3 -- the informative contrast of the revised model: labial-C2
       vs velar-C2).

figL2: (a) articulatory ground truth along lambda at the C1 window
       midpoint of P1: the gesture carriers (jaw angle, tongue body)
       drift while the lip opening stays bitwise constant;
       (b) P(n) vs lambda at layer T4 (P1 / NV3, both models): the
       phonetic layers hold /n/ over the whole sweep (no boundary);
       (c) the late-layer /m/ surface: P(m) at T10 of the CTC model,
       licensed pool vs unlicensed pool (the assimilated reading is a
       late-layer, labial-C2 phenomenon).

Style: Paul Tol palette, 300 dpi, constrained_layout, no internal
titles (captions live in the LaTeX document), spines off, dashed grid.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
OUT = os.path.join(KIT_ROOT, "figures")
RES = os.path.join(KIT_ROOT, "work", "lambda", "results.json")
META = os.path.join(KIT_ROOT, "work", "lambda", "synth_meta.json")
TRACTS = os.path.join(KIT_ROOT, "work", "lambda", "tracts")

LAY = ["CNN"] + [f"T{i}" for i in range(1, 13)]
LAMS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
LAMS_KEY = [f"{l:g}" for l in LAMS]
FRAMES = ["P1", "P2", "P4", "NV1", "NV3", "NV4"]
LIC = ["P1", "P2", "P4"]
UNL = ["NV1", "NV3", "NV4"]

TOL = dict(blue="#4477AA", cyan="#66CCEE", green="#228833", yellow="#CCBB44",
           red="#EE6677", purple="#AA3377", grey="#BBBBBB", black="#000000")

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#444444",
})

TRACT_NAMES = ("HX HY JX JA LP LD VS VO TCX TCY TTX TTY TBX TBY TRX TRY "
               "TS1 TS2 TS3").split()


def load():
    r = json.load(open(RES))
    m = json.load(open(META))
    pn, pm = {}, {}
    for tag in ("w2v2_base960h", "w2v2_base"):
        for fr in FRAMES:
            pn[(tag, fr)] = np.array(
                [[r[tag]["cells"][f"{fr}_l{k}"][l]["proba"]["n"]
                  for k in LAMS_KEY] for l in LAY])
            pm[(tag, fr)] = np.array(
                [[r[tag]["cells"][f"{fr}_l{k}"][l]["proba"]["m"]
                  for k in LAMS_KEY] for l in LAY])
    return r, m, pn, pm


def mid_params(frame):
    """Tract parameters at the C1 window midpoint, per lambda."""
    m = json.load(open(META))
    fr = next(f for f in m["frames"] if f["fid"] == frame)
    w0, w1 = fr["c1_window_frames"]
    mid = (w0 + w1) // 2
    out = {}
    for k in LAMS_KEY:
        p = os.path.join(TRACTS, f"{frame}_l{k}.tract")
        rows = [l.split() for l in open(p) if not l.startswith("#")]
        vals = [float(x) for x in rows[mid]]
        out[k] = dict(zip(TRACT_NAMES, vals[:19]))
    return out


def fig_l1(pn):
    fig, axes = plt.subplots(
        2, 2, figsize=(7.4, 5.6), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1, 1]))
    cmap = LinearSegmentedColormap.from_list(
        "tol_div", [TOL["blue"], "#FFFFFF", "#EE3377"])
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    panels = [
        ("w2v2_base960h", "P1", "wav2vec2-base-960h (CTC-tuned)"),
        ("w2v2_base960h", "NV3", ""),
        ("w2v2_base", "P1", "wav2vec2-base (self-supervised)"),
        ("w2v2_base", "NV3", ""),
    ]
    for ax, (tag, pair, label) in zip(axes.flat, panels):
        M = pn[(tag, pair)]
        im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest", origin="upper")
        ax.set_xticks(range(len(LAMS)))
        ax.set_xticklabels([f"{l:g}" for l in LAMS], fontsize=7.5)
        ax.set_yticks(range(13))
        ax.set_yticklabels(LAY, fontsize=7.5)
        # 0.5 iso-probability contour (drawn only where it exists)
        if (M >= 0.5).any() and (M <= 0.5).any():
            ax.contour(M, levels=[0.5], colors=[TOL["black"]],
                       linewidths=[0.9], linestyles=["--"])
        if pair == "P1":
            ax.set_ylabel(f"{label}\nlayer" if label else "layer",
                          fontsize=9)
    axes[0, 0].set_title("licensed: before labial /p/ in pan (P1)",
                         fontsize=9)
    axes[0, 1].set_title("unlicensed: before velar /k/ in can (NV3)",
                         fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel(r"assimilation strength $\lambda$", fontsize=9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"probe probability of underlying /n/", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.4)
    fig.savefig(f"{OUT}/figL1_continuum_heatmap.png", dpi=300)
    plt.close(fig)


def fig_l2(pn, pm):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3),
                             constrained_layout=True)
    lam = np.array(LAMS)

    # (a) articulatory ground truth at the C1 window midpoint (P1)
    ax = axes[0]
    mp = mid_params("P1")
    JA = np.array([mp[k]["JA"] for k in LAMS_KEY])
    TBX = np.array([mp[k]["TBX"] for k in LAMS_KEY])
    TBY = np.array([mp[k]["TBY"] for k in LAMS_KEY])
    LD = np.array([mp[k]["LD"] for k in LAMS_KEY])
    ax.plot(lam, TBX, "-o", color=TOL["blue"], lw=2, ms=3.5,
            label="tongue-body x TBX")
    ax.plot(lam, TBY, "-s", color=TOL["cyan"], lw=2, ms=3.5,
            label="tongue-body y TBY")
    ax.plot(lam, LD, "-^", color=TOL["red"], lw=2, ms=3.5,
            label="lip opening LD (const.)")
    ax.axhline(0.0, color="#999999", lw=0.6)
    ax.set_xlabel(r"assimilation strength $\lambda$")
    ax.set_ylabel("tract parameter (cm)")
    ax2 = ax.twinx()
    ax2.plot(lam, JA, "--", color=TOL["purple"], lw=2, ms=3.5,
             label="jaw angle JA (right)")
    ax2.set_ylabel("jaw angle (deg)")
    ax2.tick_params(labelsize=8)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7,
              loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.grid(True, ls="--", lw=0.4, alpha=0.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    # (b) P(n) vs lambda at layer T4 (P1 / NV3, both models)
    ax = axes[1]
    li = LAY.index("T4")
    styles = {("w2v2_base960h", "P1"): (TOL["blue"], "-", "o"),
              ("w2v2_base960h", "NV3"): (TOL["blue"], "--", "s"),
              ("w2v2_base", "P1"): (TOL["purple"], "-", "o"),
              ("w2v2_base", "NV3"): (TOL["purple"], "--", "s")}
    for (tag, pair), (c, ls, mk) in styles.items():
        lbl = ("CTC-tuned" if tag == "w2v2_base960h"
               else "self-supervised") + \
              (", licensed" if pair == "P1" else ", unlicensed")
        ax.plot(lam, pn[(tag, pair)][li], ls, color=c, lw=2, ms=3.5,
                marker=mk, label=lbl)
    ax.axhline(0.5, color="#999999", lw=0.8)
    ax.set_xlabel(r"assimilation strength $\lambda$")
    ax.set_ylabel(r"$P(/n/\,|\,$layer T4$)$")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls="--", lw=0.4, alpha=0.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    # (c) the late-layer /m/ surface: P(m) at T10, licensed vs unlicensed
    ax = axes[2]
    li = LAY.index("T10")
    lic_m = np.mean([pm[("w2v2_base960h", fr)][li] for fr in LIC], axis=0)
    unl_m = np.mean([pm[("w2v2_base960h", fr)][li] for fr in UNL], axis=0)
    lic_ss = np.mean([pm[("w2v2_base", fr)][li] for fr in LIC], axis=0)
    ax.plot(lam, lic_m, "-o", color=TOL["red"], lw=2, ms=3.5,
            label="CTC licensed pool (P1,P2,P4)")
    ax.plot(lam, unl_m, "--s", color=TOL["blue"], lw=2, ms=3.5,
            label="CTC unlicensed pool (NV1,NV3,NV4)")
    ax.plot(lam, lic_ss, "--", color=TOL["grey"], lw=1.8, ms=3,
            marker="^", label="self-sup. licensed pool")
    ax.axhline(0.5, color="#999999", lw=0.8)
    ax.set_xlabel(r"assimilation strength $\lambda$")
    ax.set_ylabel(r"$P(/m/\,|\,$layer T10$)$")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls="--", lw=0.4, alpha=0.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    fig.savefig(f"{OUT}/figL2_boundary_curves.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    r, m, pn, pm = load()
    fig_l1(pn)
    fig_l2(pn, pm)
    print("figures ->", OUT)

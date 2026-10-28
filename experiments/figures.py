#!/usr/bin/env python3
"""Publication figures for the wav2vec2 probing study (English labels).

Follows the charts skill rules: low-saturation palette (Paul Tol CB-safe),
constrained_layout, no top/right spines, legends outside via bbox_to_anchor,
conclusion-style titles, 300 dpi.

Outputs -> <KIT_ROOT>/download/wav2vec2_probing/figures/
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from sklearn.decomposition import PCA

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E5E7EB",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.facecolor": "#FFFFFF",
    "savefig.pad_inches": 0.25,
})

# Paul Tol colorblind-safe
C_LAB = "#0077BB"   # labial / surface
C_ALV = "#EE7733"   # alveolar / underlying
C_VEL = "#009988"   # velar
C_CYAN = "#33BBEE"
C_GRAY = "#9CA3AF"
G700, G500, G400, G300, G200 = "#374151", "#6B7280", "#9CA3AF", "#D1D5DB", "#E5E7EB"

# --- Kit portability: root of the experiment kit (parent of repro/) ------
KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
RES = os.path.join(KIT_ROOT, "work", "results")
OUT = os.path.join(KIT_ROOT, "figures")
FEAT = os.path.join(KIT_ROOT, "work", "features")
BASE = os.path.join(KIT_ROOT, "work", "base")
os.makedirs(OUT, exist_ok=True)

LAYER_NAMES = ["CNN"] + [f"T{i}" for i in range(1, 13)]
TAGS = ["w2v2_base960h", "w2v2_base"]
TAG_LABEL = {"w2v2_base960h": "wav2vec2-base-960h (CTC-tuned)",
             "w2v2_base": "wav2vec2-base (pretrained only)"}

with open(os.path.join(BASE, "base_manifest.json")) as f:
    MANIFEST = json.load(f)
ITEMS = {it["label"]: it for it in MANIFEST["items"]}


def clean_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.yaxis.grid(True, alpha=0.12, color=G300, linewidth=0.6)
        ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved {path} ({os.path.getsize(path)//1024} KB)")


def cat_code(rec):
    if rec["is_surface"]:
        return 0
    if rec["is_underlying"]:
        return 1
    return 2


# ================================================================ FIG 1 ====
def fig1():
    cv = {t: json.load(open(os.path.join(RES, t, "probe_cv.json"))) for t in TAGS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(13)
    for ax, probe, pretty in [
        (axes[0], "nasal", "Nasal place  /n/ vs /m/ vs /\u014b/"),
        (axes[1], "stop", "Stop place  /t/ vs /p/ vs /k/"),
    ]:
        for tag, color, ls, mk in [
            ("w2v2_base960h", C_LAB, "-", "o"),
            ("w2v2_base", C_ALV, "--", "s"),
        ]:
            acc = [cv[tag][probe][ln]["accuracy"] for ln in LAYER_NAMES]
            ax.plot(x, acc, color=color, linestyle=ls, marker=mk, markersize=4.5,
                    linewidth=2 if tag == "w2v2_base960h" else 1.6,
                    markerfacecolor="white", markeredgewidth=1.6,
                    markeredgecolor=color, zorder=3,
                    label=TAG_LABEL[tag])
        ax.axhline(1 / 3, color=G300, linewidth=1, linestyle=":", zorder=1)
        ax.text(12.35, 1 / 3, "chance", color=G400, fontsize=8, va="center")
        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_NAMES, fontsize=8)
        ax.set_ylim(0.25, 1.06)
        ax.set_xlabel("wav2vec2 layer")
        ax.set_ylabel("LOSO accuracy")
        ax.set_title(pretty, loc="left", fontsize=11.5)
        clean_axis(ax)
    axes[1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0))
    fig.suptitle("Place of articulation is linearly decodable from hidden states "
                 "(probe peaks in layers T2\u2013T4)", fontsize=13, fontweight="bold")
    save(fig, "fig1_probe_accuracy.png")


# ================================================================ FIG 2 ====
def fig2():
    cls = {t: json.load(open(os.path.join(RES, t, "classification.json"))) for t in TAGS}
    rows = [
        ("P1", "P1  clean pan", "n\u2192m / _p"),
        ("P2", "P2  mean boat", "n\u2192m / _b"),
        ("P3", "P3  clean can", "n\u2192\u014b / _k"),
        ("P4", "P4  thin packet", "n\u2192m / _p"),
        ("P5", "P5  that pen", "t\u2192p / _p"),
        ("P6", "P6  that cake", "t\u2192k / _k"),
        ("NV1", "NV1  clean tan  (unlicensed)", "n\u2192m / _t"),
        ("NV2", "NV2  that time  (unlicensed)", "t\u2192p / _t"),
    ]
    cmap = ListedColormap([C_LAB, C_ALV, "#EDEFF2"])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), constrained_layout=True)
    for ax, tag in zip(axes, TAGS):
        M = np.zeros((len(rows), 13), dtype=float)
        for i, (pid, _, _) in enumerate(rows):
            suffix = "assim" if pid.startswith("P") else "unlic"
            for j, ln in enumerate(LAYER_NAMES):
                M[i, j] = cat_code(cls[tag][f"{pid}_{suffix}"][ln])
        ax.imshow(M, cmap=cmap, aspect="auto", vmin=-0.5, vmax=2.5)
        # white cell separators
        ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(range(13))
        ax.set_xticklabels(LAYER_NAMES, fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{name}   {proc}" for _, name, proc in rows], fontsize=8.5)
        ax.set_xlabel("wav2vec2 layer")
        ax.set_title(TAG_LABEL[tag], loc="left", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(G200)
    handles = [Patch(facecolor=C_LAB, label="classified as SURFACE phone (assimilation realized)"),
               Patch(facecolor=C_ALV, label="classified as UNDERLYING phone"),
               Patch(facecolor="#EDEFF2", edgecolor=G300, label="other class")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9,
               frameon=False)
    fig.suptitle("Probing verdict per layer: assimilated tokens read as the surface phone in "
                 "low/mid layers;\ndeep layers of the CTC-tuned model partially recover the "
                 "underlying phone", fontsize=13, fontweight="bold")
    save(fig, "fig2_classification_map.png")


# ================================================================ FIG 3 ====
def fig3():
    tag = "w2v2_base960h"
    L = 4  # T4
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True,
                             sharex=True)
    specs = [
        ("P1", "P1  clean pan  (nasal)", "n", "m", "/n/ centroid", "/m/ centroid"),
        ("P5", "P5  that pen  (stop)", "t", "p", "/t/ centroid", "/p/ centroid"),
    ]
    for r, (pid, title, und, surf, lab_u, lab_s) in enumerate(specs):
        d = np.load(os.path.join(RES, tag, f"timecourse_{pid}.npz"))
        t = d["canon_t"]
        tw = d["target_win"]
        for c, (variant, vtitle) in enumerate([
            ("canon", "canonical variant"), ("assim", "assimilated variant")]):
            ax = axes[r, c]
            ax.plot(t, d[f"{variant}_d_{und}"][L], color=C_ALV, linewidth=2.2,
                    label=lab_u, zorder=3)
            ax.plot(t, d[f"{variant}_d_{surf}"][L], color=C_LAB, linewidth=2.2,
                    label=lab_s, zorder=3)
            ax.axvspan(tw[0], tw[1], color="#FFF7ED", zorder=1)
            ax.axvline(tw[0], color=G300, linewidth=0.8, linestyle=":")
            ax.axvline(tw[1], color=G300, linewidth=0.8, linestyle=":")
            ax.set_title(f"{title} \u2014 {vtitle}", loc="left", fontsize=10.5)
            ax.set_ylabel("cosine distance")
            clean_axis(ax)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 1].set_xlabel("time (s)")
    line_handles = [
        Line2D([], [], color=C_ALV, linewidth=2.2, label="distance to UNDERLYING class centroid"),
        Line2D([], [], color=C_LAB, linewidth=2.2, label="distance to SURFACE class centroid"),
        Patch(facecolor="#FFF7ED", edgecolor=G300, label="target (manipulated) segment window"),
    ]
    fig.legend(handles=line_handles, loc="outside lower center", ncol=3, fontsize=9,
               frameon=False)
    fig.suptitle("Frame-wise distances to class centroids (layer T4):\nthe assimilated "
                 "segment migrates to the surface attractor", fontsize=13, fontweight="bold")
    save(fig, "fig3_timecourse.png")


# ================================================================ FIG 4 ====
def fig4():
    dn = {t: np.load(os.path.join(RES, t, "delta.npz")) for t in TAGS}
    t = dn["w2v2_base960h"]["P1_t"]
    tw = json.loads(str(np.load(os.path.join(
        FEAT, "w2v2_base960h", "P1_canon.npz"), allow_pickle=True)["windows"]))["target"]
    vmax = max(float(dn[tg]["P1"].max()) for tg in TAGS)
    cmap = LinearSegmentedColormap.from_list("d", ["#FFFFFF", "#BBD6EA", "#2B6C9E"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, tag in zip(axes, TAGS):
        M = dn[tag]["P1"]  # (13, T)
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=vmax,
                       extent=[t[0], t[-1], -0.5, 12.5])
        ax.axvline(tw[0], color="#7A5C3E", linewidth=1.2, linestyle="--")
        ax.axvline(tw[1], color="#7A5C3E", linewidth=1.2, linestyle="--")
        ax.set_yticks(range(13))
        ax.set_yticklabels(LAYER_NAMES, fontsize=8)
        ax.set_xlabel("time (s)")
        ax.set_title(TAG_LABEL[tag], loc="left", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(G200)
    axes[0].text(tw[0] - 0.02, 12.6, "nasal window", ha="right", fontsize=8.5,
                 color="#7A5C3E")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, shrink=0.85)
    cbar.outline.set_visible(False)
    cbar.set_label("1 \u2212 cos(h_canon, h_assim)", fontsize=9)
    fig.suptitle("Causal fingerprint of the single-phoneme intervention (pair P1, "
                 "clean pan \u2192 clea[m] pan):\nrepresentational change is localized "
                 "at the manipulated segment and amplified with depth",
                 fontsize=13, fontweight="bold")
    save(fig, "fig4_causal_delta.png")


# ================================================================ FIG 5 ====
def window_mean_npz(tag, label):
    d = np.load(os.path.join(FEAT, tag, label + ".npz"), allow_pickle=True)
    hs, tc = d["hs"], d["t_centers"]
    wins = json.loads(str(d["windows"]))
    m = (tc >= wins["target"][0]) & (tc <= wins["target"][1])
    return hs[:, m, :].mean(axis=1)


def fig5():
    tag = "w2v2_base960h"
    L = 4
    PLACE_C = {"n": C_ALV, "t": C_ALV, "m": C_LAB, "p": C_LAB,
               "N": C_VEL, "k": C_VEL}
    train = [it for it in MANIFEST["items"] if it["group"] == "train"]
    Xtr = np.stack([window_mean_npz(tag, it["label"])[L] for it in train])
    ytr = [it["target_key"] for it in train]
    pca = PCA(n_components=2, random_state=0).fit(Xtr)

    fig, ax = plt.subplots(figsize=(10.8, 7.2), constrained_layout=True)
    # training cloud
    for it in train:
        cls = it["target_key"]
        c = PLACE_C[cls]
        marker = "o" if it["manner"] == "nasal" else "s"
        x, y = pca.transform(window_mean_npz(tag, it["label"])[L][None])[0]
        ax.scatter(x, y, c=c, marker=marker, s=42, alpha=0.55,
                   edgecolors="white", linewidths=0.6, zorder=3)
    # class centroids
    texts_cent = []
    for cls in ["n", "m", "N", "t", "p", "k"]:
        Xm = np.stack([window_mean_npz(tag, it["label"])[L]
                       for it in train if it["target_key"] == cls]).mean(axis=0)
        x, y = pca.transform(Xm[None])[0]
        ax.scatter(x, y, c=PLACE_C[cls], marker="X", s=170, edgecolors=G700,
                   linewidths=1.1, zorder=5)
        name = {"n": "/n/", "m": "/m/", "N": "/\u014b/", "t": "/t/", "p": "/p/", "k": "/k/"}[cls]
        texts_cent.append(ax.annotate(name, (x, y), fontsize=11, fontweight="bold",
                                      color=G700))
    # test tokens: canonical -> assimilated arrows
    texts_pid = []
    star_xy = []
    tests = [it for it in MANIFEST["items"]
             if it["group"] in ("pair_canon", "nv_canon")]
    for it in tests:
        pid = it["pid"]
        suffix = "assim" if pid.startswith("P") else "unlic"
        vc = window_mean_npz(tag, f"{pid}_canon")[L]
        va = window_mean_npz(tag, f"{pid}_{suffix}")[L]
        pc = pca.transform(vc[None])[0]
        pa = pca.transform(va[None])[0]
        c = PLACE_C[it["surface"]]
        ax.scatter(*pc, facecolors="none", edgecolors=PLACE_C[it["target"]],
                   s=210, linewidths=2.0, zorder=6)
        ax.scatter(*pa, marker="*", c=c, s=330, edgecolors=G700,
                   linewidths=1.0, zorder=7)
        ax.add_patch(FancyArrowPatch(pc, pa, arrowstyle="-|>", mutation_scale=13,
                                     color=G400, linewidth=1.2,
                                     connectionstyle="arc3,rad=0.18", zorder=5))
        texts_pid.append(ax.annotate(pid, pa, fontsize=9, fontweight="bold",
                                     color=G700))
        star_xy.append(pa)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    clean_axis(ax, grid=False)
    # collision avoidance for all in-plot labels
    from adjustText import adjust_text
    all_points = np.array([pca.transform(window_mean_npz(tag, it["label"])[L][None])[0]
                           for it in train] + star_xy)
    # deterministic seeds: push each label radially away from the data centroid
    center = all_points.mean(axis=0)
    span = all_points.max(axis=0) - all_points.min(axis=0)
    for txt, pt in zip(texts_pid + texts_cent,
                       list(star_xy) + [t.get_position() for t in texts_cent]):
        d = np.array(pt) - center
        n = np.linalg.norm(d) + 1e-9
        u = d / n * np.array([1.0, 0.7])
        txt.set_position((pt[0] + u[0] * span[0] * 0.055,
                          pt[1] + u[1] * span[1] * 0.075))
    adjust_text(texts_pid + texts_cent,
                x=all_points[:, 0], y=all_points[:, 1],
                ax=ax, expand=(1.5, 1.9),
                force_text=(0.9, 1.3), force_static=(0.5, 0.7),
                force_explode=(0.35, 0.5),
                only_move={"text": "xy"},
                ensure_inside_axes=True, time_lim=8,
                arrowprops=dict(arrowstyle="-", color=G300, lw=0.7))
    handles = [
        Line2D([], [], marker="o", linestyle="", color=C_LAB, markersize=8,
               label="labial (/m/, /p/)"),
        Line2D([], [], marker="o", linestyle="", color=C_ALV, markersize=8,
               label="alveolar (/n/, /t/)"),
        Line2D([], [], marker="o", linestyle="", color=C_VEL, markersize=8,
               label="velar (/\u014b/, /k/)"),
        Line2D([], [], marker="o", linestyle="", color=G500, markersize=7,
               markerfacecolor=G500, label="nasal token"),
        Line2D([], [], marker="s", linestyle="", color=G500, markersize=7,
               markerfacecolor=G500, label="stop token"),
        Line2D([], [], marker="X", linestyle="", color=G500, markersize=9,
               label="class centroid"),
        Line2D([], [], marker="o", linestyle="", color=G700, markersize=9,
               markerfacecolor="none", markeredgewidth=1.8,
               label="pair token, canonical"),
        Line2D([], [], marker="*", linestyle="", color=G500, markersize=13,
               markerfacecolor=G500, label="pair token, assimilated / unlicensed"),
    ]
    fig.legend(handles=handles, loc="outside center right", fontsize=8.5,
               labelspacing=0.9, frameon=False)
    ax.set_title("Layer T4 geometry: assimilated tokens (stars) land on the "
                 "surface side of the place continuum", loc="left", fontsize=12)
    save(fig, "fig5_pca.png")


# ============================================================ SUMMARY ======
def summary():
    """Aggregate numbers reused by the report and the HTML browser."""
    out = {}
    cls = {t: json.load(open(os.path.join(RES, t, "classification.json"))) for t in TAGS}
    cv = {t: json.load(open(os.path.join(RES, t, "probe_cv.json"))) for t in TAGS}
    for t in TAGS:
        frac = []
        for j, ln in enumerate(LAYER_NAMES):
            vals = [cat_code(cls[t][f"P{i}_assim"][ln]) for i in range(1, 7)]
            frac.append(sum(1 for v in vals if v == 0) / 6.0)
        out[t] = {
            "cv_nasal": [cv[t]["nasal"][ln]["accuracy"] for ln in LAYER_NAMES],
            "cv_stop": [cv[t]["stop"][ln]["accuracy"] for ln in LAYER_NAMES],
            "frac_surface_assim": frac,
            "per_pair": {
                f"P{i}": [cls[t][f"P{i}_assim"][ln]["pred"] for ln in LAYER_NAMES]
                for i in range(1, 7)
            },
            "nv": {
                pid: [cls[t][f"{pid}_unlic"][ln]["pred"] for ln in LAYER_NAMES]
                for pid in ["NV1", "NV2"]
            },
            "canon_sanity": {
                lab: sum(cls[t][lab][ln]["is_underlying"] for ln in LAYER_NAMES) / 13.0
                for lab in [f"P{i}_canon" for i in range(1, 7)] + ["NV1_canon", "NV2_canon"]
            },
        }
    path = os.path.join(RES, "summary_stats.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("summary ->", path)
    for t in TAGS:
        print(t, "frac surface (L0..L12):",
              " ".join(f"{v:.2f}" for v in out[t]["frac_surface_assim"]))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    summary()

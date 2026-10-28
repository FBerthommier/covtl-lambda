#!/usr/bin/env python3
"""Digitize Pouw Figure 2 (top band only): bars + tick calibration.

Input: a 150-dpi render of page 9 of the published paper (NOT shipped
with this kit -- the paper is not redistributed, see NOTICE.md Tier 3).
Produce it with:  pdftoppm -png -r 150 <pouw2024.pdf> pouw_p9
and point PAPER_P9 at the resulting pouw_p9-09.png.
The digitized constants this tool calibrates are quoted (with citation)
at the top of lexamb_cross_figure.py.
"""
import os

import numpy as np
from PIL import Image

P9 = os.environ.get(
    "PAPER_P9",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "work", "lexamb", "pouw_p9-09.png"))
im = Image.open(P9).convert("RGB")
a = np.asarray(im)
r, g, b = (a[..., 0].astype(int), a[..., 1].astype(int),
           a[..., 2].astype(int))
sat = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
gray = a[..., 0].astype(int)

# ---- Figure 2 region: rows 250-660 -----------------------------------
R0, R1 = 250, 660
green = (sat > 60) & (g > r + 30) & (g > b + 30)
blue = (sat > 60) & (b > r + 30) & (b > g + 30) & (g < 160)

# bar band: below the legend (which ends ~y=345)
B0, B1 = 430, 660


def bars(mask, label):
    sub = mask[B0:B1]
    ys, xs = np.where(sub)
    if not len(xs):
        print(f"--- {label}: aucune")
        return []
    xs_sorted = np.unique(xs)
    clusters, cur = [], [xs_sorted[0]]
    for x in xs_sorted[1:]:
        if x - cur[-1] <= 5:
            cur.append(x)
        else:
            clusters.append(cur)
            cur = [x]
    clusters.append(cur)
    out = []
    print(f"--- {label}: {len(clusters)} barres")
    for cl in clusters:
        x0, x1 = cl[0], cl[-1]
        m = sub[:, x0:x1 + 1]
        yy = np.where(m.any(axis=1))[0] + B0
        out.append((x0, x1, int(yy.min()), int(yy.max())))
        print(f"    x {x0:4d}-{x1:4d}  y {yy.min():4d}-{yy.max():4d}"
              f"  (bas={yy.max()})")
    return out


gb = bars(green, "VERT (viable)")
bb = bars(blue, "BLEU (unviable)")

# ---- tick labels in the left margin ----------------------------------
dark = gray < 120
margin = dark[:, 150:228]
rows = np.where(margin.any(axis=1))[0]
groups, cur = [], [rows[0]]
for y in rows[1:]:
    if y - cur[-1] <= 6:
        cur.append(y)
    else:
        groups.append(cur)
        cur = [y]
groups.append(cur)
print("--- étiquettes d'axe (lignes de texte à gauche):")
ticks = []
for gr in groups:
    y0, y1 = gr[0], gr[-1]
    if R0 <= (y0 + y1) / 2 <= R1 and 4 < y1 - y0 < 30:
        yc = (y0 + y1) / 2
        ticks.append(yc)
        print(f"    texte y {y0}-{y1}  centre={yc:.0f}")

# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""KIT30 step 13 -- lambda=0.625 production statistics + 9-degree
dose-response, v3.0 (revised target-level cluster continuum).

Instruments identical to the v2.1 script; changes:
  - reads the unified block schema {model: {cells, ...}};
  - dose_response now spans ALL NINE degrees of the campaign grid
    (0, 0.125, ..., 1) -- the v2.1 figure had four;
  - crossrun_check -> subset_check: the 0.625-block cells are verified
    against the continuum block (identity by construction, single
    probing run; asserted anyway).

Outputs -> work/lambda0625/pouw0625_stats.json (+ console summary).
"""
import json
import os

import numpy as np
from scipy import stats as sps

KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
RES = os.path.join(KIT_ROOT, "work", "lambda0625", "results.json")
RES_FULL = os.path.join(KIT_ROOT, "work", "lambda", "results.json")
OUT = os.path.join(KIT_ROOT, "work", "lambda0625", "pouw0625_stats.json")

MODELS = ["w2v2_base960h", "w2v2_base"]
LAYERS = ["CNN"] + [f"T{i}" for i in range(1, 13)]
FRAMES = {
    "P1":  ("licensed",   "clean pan"),
    "P2":  ("licensed",   "mean boat"),
    "P4":  ("licensed",   "thin packet"),
    "NV1": ("unlicensed", "clean tan"),
    "NV3": ("unlicensed", "clean can"),
    "NV4": ("unlicensed", "mean goat"),
}
LAMS_BLK = ["0", "0.625", "0.75", "1"]
LAMBDAS9 = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
LAM_KEY9 = [f"{l:g}" for l in LAMBDAS9]
Z = 1.959963985


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = (Z / den) * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe_diff(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def verdicts_of(r, tag, frame, lam):
    item = r[tag]["cells"][f"{frame}_l{lam}"]
    return [item[l]["pred"] for l in LAYERS]


def cell_stats(r, tag, frames, lam):
    vs, items = [], {}
    for fr in frames:
        v = verdicts_of(r, tag, fr, lam)
        kn = sum(1 for x in v if x == "n")
        km = sum(1 for x in v if x == "m")
        kN = sum(1 for x in v if x == "N")
        items[fr] = {"k_n": kn, "n": len(LAYERS), "rate": kn / len(LAYERS),
                     "profile": "/".join(f"{c}x{k}"
                                         for c, k in (("n", kn), ("m", km),
                                                      ("N", kN)) if k > 0)}
        vs.extend(v)
    kn = sum(1 for x in vs if x == "n")
    km = sum(1 for x in vs if x == "m")
    kN = sum(1 for x in vs if x == "N")
    n = len(vs)
    lo, hi = wilson(kn, n)
    slo, shi = wilson(km, n)
    return {"k": kn, "n": n, "rate": kn / n, "ci_lo": lo, "ci_hi": hi,
            "verdict_profile": "/".join(f"{c}x{k}"
                                        for c, k in (("n", kn), ("m", km),
                                                     ("N", kN)) if k > 0),
            "surface_m": {"k": km, "n": n, "rate": km / n,
                          "ci_lo": slo, "ci_hi": shi},
            "items": items}


def main():
    r = json.load(open(RES))
    rf = json.load(open(RES_FULL))

    licensed = [f for f, (l, _s) in FRAMES.items() if l == "licensed"]
    unlicensed = [f for f, (l, _s) in FRAMES.items() if l == "unlicensed"]
    allframes = licensed + unlicensed

    # subset check: block cells == continuum cells (single probing run)
    subset_check = {}
    for tag in MODELS:
        mx = 0.0
        for fr in allframes:
            for lam in LAMS_BLK:
                for l in LAYERS:
                    a = r[tag]["cells"][f"{fr}_l{lam}"][l]["proba"]["n"]
                    b = rf[tag]["cells"][f"{fr}_l{lam}"][l]["proba"]["n"]
                    mx = max(mx, abs(a - b))
        subset_check[tag] = float(mx)

    # endpoint distances vs dotted tier
    orig = {"P1": ("P1_canon", "P1_assim"), "P2": ("P2_canon", "P2_assim"),
            "P4": ("P4_canon", "P4_assim"), "NV1": ("NV1_canon", "NV1_unlic"),
            "NV3": ("NV3_canon", "NV3_unlic"), "NV4": ("NV4_canon", "NV4_unlic")}
    ep_dist = {}
    for tag in MODELS:
        mx = 0.0
        for fr in allframes:
            c, u = orig[fr]
            for l in LAYERS:
                for a, oitem in (("0", c), ("1", u)):
                    mx = max(mx, abs(
                        r[tag]["cells"][f"{fr}_l{a}"][l]["proba"]["n"]
                        - r[tag]["original"][oitem][l]["proba"]["n"]))
        ep_dist[tag] = float(mx)

    out = {
        "definitions": {
            "compensation": "probe verdict = underlying /n/",
            "control": "lambda=0 (canonical cluster) pooled over the 6 "
                       "frames",
            "viable": "lambda=0.625 pooled over licensed frames",
            "unviable": "lambda=0.625 pooled over unlicensed frames",
            "rate_denominator": "(stimuli x 13 layer probes) per condition",
            "ci": "95% Wilson score interval (descriptive)",
            "note": "layers are not independent samples; Pouw et al. use "
                    "N=48 stimuli per condition",
            "model": "covtl-lambda v1.2.0 target-level cluster continuum",
            "dose_response": "compensation rate of the licensed and "
                             "unlicensed pools at each of the NINE degrees "
                             "of the campaign grid (0, 0.125, ..., 1)",
            "subset_check": "max |dP(n)| between this block's cells and the "
                            "continuum block's cells for the shared items "
                            "(identity expected: one probing run)",
        },
        "conditions": {},
        "items": {},
        "asymmetry": {},
        "layer_pn": {},
        "deep_mean_pn": {},
        "dose_response": {},
        "pouw_digitized": {"wav2vec2": {"viable": 36, "unviable": 19,
                                        "control": 48},
                           "humans": {"viable": 22, "unviable": 5,
                                      "control": 45},
                           "N": 48},
        "endpoint_distance_max": ep_dist,
        "subset_check_max": subset_check,
    }

    CONDS = [
        ("control", allframes, "0"),
        ("viable", licensed, "0.625"),
        ("unviable", unlicensed, "0.625"),
        ("viable_lam075_ref", licensed, "0.75"),
        ("unviable_lam075_ref", unlicensed, "0.75"),
        ("viable_lam1_ref", licensed, "1"),
        ("unviable_lam1_ref", unlicensed, "1"),
    ]

    for tag in MODELS:
        out["conditions"][tag] = {}
        out["items"][tag] = {}
        out["layer_pn"][tag] = {}
        out["deep_mean_pn"][tag] = {}
        for cond, frames, lam in CONDS:
            out["conditions"][tag][cond] = cell_stats(r, tag, frames, lam)
            per_layer = {}
            for l in LAYERS:
                per_layer[l] = float(np.mean(
                    [r[tag]["cells"][f"{fr}_l{lam}"][l]["proba"]["n"]
                     for fr in frames]))
            out["layer_pn"][tag][cond] = per_layer
            out["deep_mean_pn"][tag][cond] = float(np.mean(
                [per_layer[l] for l in ("T10", "T11", "T12")]))
        for fr in allframes:
            out["items"][tag][fr] = {}
            for lam in LAMS_BLK:
                v = verdicts_of(r, tag, fr, lam)
                kn = sum(1 for x in v if x == "n")
                km = sum(1 for x in v if x == "m")
                kN = sum(1 for x in v if x == "N")
                out["items"][tag][fr][lam] = {
                    "k_n": kn, "n": len(LAYERS), "rate": kn / len(LAYERS),
                    "profile": "/".join(f"{c}x{k}"
                                        for c, k in (("n", kn), ("m", km),
                                                     ("N", kN)) if k > 0)}

        out["asymmetry"][tag] = {}
        for cond_v, cond_u, key in (
                ("viable", "unviable", "lam0.625"),
                ("viable_lam075_ref", "unviable_lam075_ref", "lam0.75_ref"),
                ("viable_lam1_ref", "unviable_lam1_ref", "lam1_ref")):
            cv = out["conditions"][tag][cond_v]
            cu = out["conditions"][tag][cond_u]
            d, lo, hi = newcombe_diff(cv["k"], cv["n"], cu["k"], cu["n"])
            _, fp = sps.fisher_exact([[cv["k"], cv["n"] - cv["k"]],
                                      [cu["k"], cu["n"] - cu["k"]]])
            out["asymmetry"][tag][key] = {
                "viable_rate": cv["rate"], "unviable_rate": cu["rate"],
                "diff": d, "ci_lo": lo, "ci_hi": hi,
                "fisher_p": float(fp),
                "ci_excludes_zero": bool(lo > 0 or hi < 0)}

        # 9-degree dose-response from the continuum block
        out["dose_response"][tag] = {}
        for k in LAM_KEY9:
            entry = {}
            for pool_name, pool in (("viable", licensed),
                                    ("unviable", unlicensed)):
                vs = []
                for fr in pool:
                    vs.extend(verdicts_of(rf, tag, fr, k))
                kn = sum(1 for x in vs if x == "n")
                km = sum(1 for x in vs if x == "m")
                n = len(vs)
                lo, hi = wilson(kn, n)
                slo, shi = wilson(km, n)
                entry[pool_name] = {
                    "k": kn, "n": n, "rate": kn / n,
                    "ci_lo": lo, "ci_hi": hi,
                    "verdict_profile": f"nx{kn}/mx{km}",
                    "surface_m": {"k": km, "n": n, "rate": km / n,
                                  "ci_lo": slo, "ci_hi": shi}}
            out["dose_response"][tag][k] = entry

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    P = 100.0
    print("=" * 78)
    print("v3.0 POUW-STYLE RATES (lambda=0.625 production)")
    print("=" * 78)
    print(f"{'model':15s} {'Control':>18s} {'Viable':>18s} {'Unviable':>18s}")
    for tag in MODELS:
        row = []
        for cond in ("control", "viable", "unviable"):
            c = out["conditions"][tag][cond]
            row.append(f"{c['rate']*P:5.1f} ({c['k']:2d}/{c['n']:2d})"
                       f" [{c['ci_lo']*P:4.0f}-{c['ci_hi']*P:4.0f}]")
        print(f"{tag:15s} {row[0]:>18s} {row[1]:>18s} {row[2]:>18s}")
    print("\nASYMMETRY (Viable - Unviable):")
    for tag in MODELS:
        for key in ("lam0.625", "lam0.75_ref", "lam1_ref"):
            a = out["asymmetry"][tag][key]
            print(f"  {tag:15s} {key:12s} diff={a['diff']*P:+6.1f} pts "
                  f"CI=({a['ci_lo']*P:+.1f},{a['ci_hi']*P:+.1f}) "
                  f"fisher p={a['fisher_p']:.4f}"
                  f"{' *' if a['ci_excludes_zero'] else ''}")
    print("\nDOSE-RESPONSE (9 degrees, compensation rate %):")
    for tag in MODELS:
        print(f"  {tag}")
        print(f"    {'lam':>6s} {'viable':>18s} {'unviable':>18s}")
        for k in LAM_KEY9:
            e = out["dose_response"][tag][k]
            v, u = e["viable"], e["unviable"]
            print(f"    {k:>6s} {v['rate']*P:5.1f} ({v['k']:2d}/{v['n']:2d})"
                  f"      {u['rate']*P:5.1f} ({u['k']:2d}/{u['n']:2d})")
    print(f"\nsubset check max |dP(n)| vs continuum: {subset_check}")
    print(f"endpoint distance max |dP(n)| cluster vs dotted: "
          f"{ {t: round(v,4) for t,v in ep_dist.items()} }")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

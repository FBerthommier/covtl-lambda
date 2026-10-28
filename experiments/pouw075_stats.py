#!/usr/bin/env python3
"""KIT30 step 12 -- Pouw-style 3-condition statistics, lambda=0.75
production experiment, v3.0 (revised target-level cluster continuum).

Instruments identical to the v2.1 script; changes:
  - reads the unified block schema {model: {cells, ...}} built by step 10;
  - endpoint_check_max -> endpoint_distance_max (cluster endpoints vs the
    dotted citation tier; expected non-zero under v1.2.0 semantics);
  - adds the guarantee summary from synth_meta.json (54/54).

Outputs -> work/lambda075/pouw075_stats.json (+ console summary).
"""
import json
import os

import numpy as np
from scipy import stats as sps

KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
RES = os.path.join(KIT_ROOT, "work", "lambda075", "results.json")
OUT = os.path.join(KIT_ROOT, "work", "lambda075", "pouw075_stats.json")

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
LAMS = ["0", "0.75", "1"]
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
    meta = json.load(open(os.path.join(KIT_ROOT, "work", "lambda",
                                       "synth_meta.json")))

    licensed = [f for f, (l, _s) in FRAMES.items() if l == "licensed"]
    unlicensed = [f for f, (l, _s) in FRAMES.items() if l == "unlicensed"]
    allframes = licensed + unlicensed

    # endpoint distances (cluster lam0/lam1 vs dotted citation tier)
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
            "viable": "lambda=0.75 pooled over licensed frames "
                      "(P1 clean pan, P2 mean boat, P4 thin packet)",
            "unviable": "lambda=0.75 pooled over unlicensed frames "
                        "(NV1 clean tan, NV3 clean can, NV4 mean goat)",
            "rate_denominator": "(stimuli x 13 layer probes) per condition",
            "ci": "95% Wilson score interval (descriptive)",
            "note": "layers are not independent samples; Pouw et al. use "
                    "N=48 stimuli per condition",
            "model": "covtl-lambda v1.2.0 target-level cluster continuum "
                     "(lambda=0 is bitwise the canonical cluster; "
                     "lambda=1 reaches the C2 target coordinates)",
            "endpoint_distance": "cluster endpoints vs the dotted citation "
                                 "tier (stage-1/v2.1 probings): a "
                                 "coarticulation distance, expected "
                                 "non-zero, NOT a validity check",
        },
        "conditions": {},
        "items": {},
        "asymmetry": {},
        "layer_pn": {},
        "deep_mean_pn": {},
        "pouw_digitized": {"wav2vec2": {"viable": 36, "unviable": 19,
                                        "control": 48},
                           "humans": {"viable": 22, "unviable": 5,
                                      "control": 45},
                           "N": 48},
        "guarantees": "covtl-lambda v1.2.0 run-time checks held 54/54 "
                      "campaign syntheses (synth_meta.json)",
        "endpoint_distance_max": ep_dist,
    }

    CONDS = [
        ("control", allframes, "0"),
        ("viable", licensed, "0.75"),
        ("unviable", unlicensed, "0.75"),
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
            for lam in LAMS:
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
                ("viable", "unviable", "lam0.75"),
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
        lv = [out["items"][tag][fr]["0.75"]["rate"] for fr in licensed]
        lu = [out["items"][tag][fr]["0.75"]["rate"] for fr in unlicensed]
        out["asymmetry"][tag]["item_consistency_lam0.75"] = {
            "licensed_rates": lv, "unlicensed_rates": lu,
            "perfect_separation": bool(min(lv) > max(lu)),
            "n_licensed_above_max_unlicensed": int(
                sum(1 for x in lv if x > max(lu)))}

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    P = 100.0
    print("=" * 78)
    print("v3.0 POUW-STYLE 3-CONDITION RATES (lambda=0.75 production)")
    print("=" * 78)
    print(f"{'model':15s} {'Control':>18s} {'Viable':>18s} {'Unviable':>18s}")
    print(f"{'Pouw wav2vec2':15s} {'100':>18s} {'75':>18s} {'40':>18s}")
    print(f"{'Pouw humans':15s} {'94':>18s} {'46':>18s} {'11':>18s}")
    for tag in MODELS:
        row = []
        for cond in ("control", "viable", "unviable"):
            c = out["conditions"][tag][cond]
            row.append(f"{c['rate']*P:5.1f} ({c['k']:2d}/{c['n']:2d})"
                       f" [{c['ci_lo']*P:4.0f}-{c['ci_hi']*P:4.0f}]")
        print(f"{tag:15s} {row[0]:>18s} {row[1]:>18s} {row[2]:>18s}")
    print("\nlambda=1 reference:")
    for tag in MODELS:
        row = []
        for cond in ("viable_lam1_ref", "unviable_lam1_ref"):
            c = out["conditions"][tag][cond]
            row.append(f"{c['rate']*P:5.1f} ({c['k']:2d}/{c['n']:2d})")
        print(f"  {tag:15s} viable {row[0]:>12s}   unviable {row[1]:>12s}")
    print("\nASYMMETRY (Viable - Unviable):")
    for tag in MODELS:
        for key, lbl in (("lam0.75", "lambda=0.75"),
                         ("lam1_ref", "lambda=1 ref")):
            a = out["asymmetry"][tag][key]
            print(f"  {tag:15s} {lbl:12s} diff={a['diff']*P:+6.1f} pts "
                  f"CI=({a['ci_lo']*P:+.1f},{a['ci_hi']*P:+.1f}) "
                  f"fisher p={a['fisher_p']:.4f}"
                  f"{' *' if a['ci_excludes_zero'] else ''}")
    print("\nPER-FRAME rates (k/13) at lambda=0 / 0.75 / 1:")
    for tag in MODELS:
        for fr in allframes:
            lic, sent = FRAMES[fr]
            d = out["items"][tag][fr]
            print(f"  {tag:15s} {fr:4s} {lic:10s} {sent:12s} "
                  f"{d['0']['k_n']:2d} {d['0.75']['k_n']:2d} {d['1']['k_n']:2d}"
                  f"   profiles@0.75: {d['0.75']['profile']}")
    print("\nDEEP-LAYER mean P(n) (T10-T12):")
    for tag in MODELS:
        s = "  ".join(f"{c}={out['deep_mean_pn'][tag][c]:.3f}"
                      for c in ("control", "viable", "unviable"))
        print(f"  {tag:15s} {s}")
    print(f"\nendpoint distance max |dP(n)| cluster vs dotted: "
          f"{ {t: round(v,4) for t,v in ep_dist.items()} }")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""KIT30 step 11 -- note statistics v3.0: the three anchor conditions of
the REVISED (target-level, cluster) assimilation continuum, for the six
frames, in the style of the Pouw et al. (2024) human-model comparison.

v3.0 changes vs the v2.1 script (instruments otherwise identical):
  - CONTEXTS = all six frames (P1 P2 P4 NV1 NV3 NV4); the continuum is
    the 54-item campaign grid (6 frames x 9 lambdas);
  - reads the unified block schema {model: {cells, original, probe_cv}};
  - "endpoint_check" becomes "endpoint_distance": distance between the
    cluster continuum endpoints and the DOTTED citation endpoints of the
    stage-1 tier. Under the revised semantics this distance is EXPECTED
    to be non-zero (tautosyllabic cluster vs C.C split are different
    gesture structures); it is reported as a descriptive coarticulation
    distance, not as a validity check. Tract-level canonicity is
    guaranteed by the v1.2.0 run-time checks (54/54, synth_meta.json).

Kit-portable: reads work/lambda/results.json relative to KIT_ROOT.
Outputs -> work/lambda/note_stats.json (+ console summary).
"""
import itertools
import json
import os

import numpy as np
from scipy import stats as sps

KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
RES = os.path.join(KIT_ROOT, "work", "lambda", "results.json")
OUT = os.path.join(KIT_ROOT, "work", "lambda", "note_stats.json")

MODELS = ["w2v2_base960h", "w2v2_base"]
MODEL_LABEL = {"w2v2_base960h": "wav2vec2-base-960h (CTC-tuned)",
               "w2v2_base": "wav2vec2-base (self-supervised)"}
FRAMES = ["P1", "P2", "P4", "NV1", "NV3", "NV4"]
FRAME_LABEL = {
    "P1": "licensed (clean pan, n->m before labial /p/)",
    "P2": "licensed (mean boat, n->m before labial /b/)",
    "P4": "licensed (thin packet, n->m before labial /p/)",
    "NV1": "unlicensed (clean tan, before alveolar /t/)",
    "NV3": "unlicensed (clean can, before velar /k/)",
    "NV4": "unlicensed (mean goat, before velar /g/)",
}
LAYERS = ["CNN"] + [f"T{i}" for i in range(1, 13)]
LAMBDAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
LAM_KEY = [f"{l:g}" for l in LAMBDAS]
ANCHORS = ["0", "0.5", "1"]
Z = 1.959963985  # 95%


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = (Z / denom) * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


_N9 = 9


def _spearman_null_n9():
    vals = {}
    ref = np.arange(_N9)
    hist = {}
    for p in itertools.permutations(range(_N9)):
        s = 0
        for a, b in zip(ref, p):
            s += (a - b) * (a - b)
        hist[s] = hist.get(s, 0) + 1
    total = sum(hist.values())
    rows = sorted(((1.0 - s / 120.0, c) for s, c in hist.items()),
                  reverse=True)
    out = []
    cum = 0
    for rho, c in rows:
        cum += c
        out.append((rho, cum / total))
    return out


_NULL9 = None


def spearman_exact(x, y):
    global _NULL9
    rho, _ = sps.spearmanr(x, y)
    rx = sps.rankdata(x)
    ry = sps.rankdata(y)
    no_ties = (len(set(np.round(rx, 12))) == len(x)
               and len(set(np.round(ry, 12))) == len(y))
    if _NULL9 is None:
        _NULL9 = _spearman_null_n9()
    if no_ties:
        target = abs(rho)
        for ra, pone in reversed(_NULL9):
            if ra >= target - 1e-9:
                return float(rho), float(min(1.0, 2.0 * pone))
    n = len(x)
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return float(rho), float(2 * (1 - sps.t.cdf(abs(t), n - 2)))


def cross_below(xs, ys, level):
    if ys[0] < level:
        return "<0"
    for i in range(1, len(xs)):
        if ys[i] < level:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if y1 == y0:
                return x1
            return x0 + (y0 - level) * (x1 - x0) / (y0 - y1)
    return ">1"


def cross_equal(xs, ya, yb):
    d = np.array(ya) - np.array(yb)
    return cross_below(xs, list(d), 0.0)


def _add_pools(ns, r):
    """Pooled licensed/unlicensed anchor cells + pooled continuum
    statistics + pooled agreement (v3.0 additions)."""
    import itertools
    from scipy import stats as sps2
    LIC = ["P1", "P2", "P4"]
    UNL = ["NV1", "NV3", "NV4"]
    LAMBDAS9 = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    LAM_KEY9 = [f"{l:g}" for l in LAMBDAS9]

    pools = {tag: {} for tag in MODELS}
    for tag in MODELS:
        for pool_name, pool in (("licensed", LIC), ("unlicensed", UNL)):
            pools[tag][pool_name] = {}
            for a in ANCHORS:
                ks = sum(ns["compensation"][tag][fr][a]["k"] for fr in pool)
                n = 13 * len(pool)
                lo, hi = wilson(ks, n)
                pools[tag][pool_name][a] = {
                    "k": ks, "n": n, "rate": ks / n,
                    "ci_lo": lo, "ci_hi": hi}
    ns["pools"] = pools

    spear = spearman_exact  # exact null over the 9! permutations

    def crossb(xs, ys, level):
        if ys[0] < level:
            return "<0"
        for i in range(1, len(xs)):
            if ys[i] < level:
                x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
                if y1 == y0:
                    return x1
                return x0 + (y0 - level) * (x1 - x0) / (y0 - y1)
        return ">1"

    pooled_pn = {}
    for tag in MODELS:
        for pool_name, pool in (("licensed", LIC), ("unlicensed", UNL)):
            pooled_pn[(tag, pool_name)] = np.array(
                [[np.mean([r[tag]["cells"][f"{fr}_l{k}"][l]["proba"]["n"]
                           for fr in pool])
                  for k in LAM_KEY9] for l in LAYERS])
    pooled_stats = {}
    for tag in MODELS:
        pooled_stats[tag] = {}
        for pool_name in ("licensed", "unlicensed"):
            M = pooled_pn[(tag, pool_name)]
            ent = {"spearman": {}, "boundary_half_T4": None,
                   "deep_contrast": {}}
            for li, l in enumerate(LAYERS):
                rho, p = spear(LAMBDAS9, M[li])
                ent["spearman"][l] = {"rho": rho, "p": p}
            ent["boundary_half_T4"] = crossb(
                LAMBDAS9, list(M[LAYERS.index("T4")]), 0.5)
            for l in ("T11", "T12"):
                ent["deep_contrast"][l] = float(
                    M[LAYERS.index(l)][-1] - M[LAYERS.index(l)][0])
            pooled_stats[tag][pool_name] = ent
    ns["pooled_continuum"] = pooled_stats

    agree = {}
    for pool_name in ("licensed", "unlicensed"):
        agree[pool_name] = {}
        for li, l in enumerate(LAYERS):
            rho, p = spear(
                list(pooled_pn[("w2v2_base", pool_name)][li]),
                list(pooled_pn[("w2v2_base960h", pool_name)][li]))
            agree[pool_name][l] = {"rho": rho, "p": p}
    ns["pooled_agreement"] = agree


def main():
    r = json.load(open(RES))

    out = {
        "definitions": {
            "compensation": "probe verdict = underlying /n/",
            "rate_denominator": "13 layer probes (CNN + T1..T12) of one "
                                "stimulus",
            "ci": "95% Wilson score interval (descriptive)",
            "note": "layers are not independent samples; Pouw et al. use "
                    "N=48 stimuli",
            "model": "covtl-lambda v1.2.0: theta_C1(lam) = theta_C1 + "
                     "lam*wrap(theta_C2 - theta_C1) on the tautosyllabic "
                     "cluster; lips/velum/glottis bitwise constant",
            "endpoint_distance": "cluster continuum endpoints vs the dotted "
                                 "citation endpoints of the stage-1 tier "
                                 "(expected non-zero under the revised "
                                 "semantics: different gesture structures)",
        },
        "compensation": {},
        "surface": {},
        "layer_detail": {},
        "spearman_lambda": {},
        "boundary_half": {},
        "boundary_nm": {},
        "deep_contrast": {},
        "model_agreement": {},
        "endpoint_distance": {},
    }

    pn, pm = {}, {}
    for tag in MODELS:
        for fr in FRAMES:
            pn[(tag, fr)] = np.array(
                [[r[tag]["cells"][f"{fr}_l{k}"][l]["proba"]["n"]
                  for k in LAM_KEY] for l in LAYERS])
            pm[(tag, fr)] = np.array(
                [[r[tag]["cells"][f"{fr}_l{k}"][l]["proba"]["m"]
                  for k in LAM_KEY] for l in LAYERS])

    for tag in MODELS:
        out["compensation"][tag] = {}
        out["surface"][tag] = {}
        out["layer_detail"][tag] = {}
        for fr in FRAMES:
            out["compensation"][tag][fr] = {}
            out["surface"][tag][fr] = {}
            out["layer_detail"][tag][fr] = {}
            for a in ANCHORS:
                item = r[tag]["cells"][f"{fr}_l{a}"]
                verdicts = {l: item[l]["pred"] for l in LAYERS}
                k_n = sum(1 for v in verdicts.values() if v == "n")
                k_m = sum(1 for v in verdicts.values() if v == "m")
                lo, hi = wilson(k_n, len(LAYERS))
                slo, shi = wilson(k_m, len(LAYERS))
                prof = "/".join(
                    f"{v}x{sum(1 for w in verdicts.values() if w == v)}"
                    for v in ("n", "m", "N")
                    if sum(1 for w in verdicts.values() if w == v) > 0)
                out["compensation"][tag][fr][a] = {
                    "k": k_n, "n": len(LAYERS), "rate": k_n / len(LAYERS),
                    "ci_lo": lo, "ci_hi": hi, "verdict_profile": prof}
                out["surface"][tag][fr][a] = {
                    "k": k_m, "n": len(LAYERS), "rate": k_m / len(LAYERS),
                    "ci_lo": slo, "ci_hi": shi}
                out["layer_detail"][tag][fr][a] = {
                    l: {"n": item[l]["proba"]["n"],
                        "m": item[l]["proba"]["m"],
                        "N": item[l]["proba"]["N"],
                        "pred": item[l]["pred"]} for l in LAYERS}

    for tag in MODELS:
        out["spearman_lambda"][tag] = {}
        out["boundary_half"][tag] = {}
        out["boundary_nm"][tag] = {}
        out["deep_contrast"][tag] = {}
        for fr in FRAMES:
            out["spearman_lambda"][tag][fr] = {}
            out["boundary_half"][tag][fr] = {}
            out["boundary_nm"][tag][fr] = {}
            out["deep_contrast"][tag][fr] = {}
            for li, l in enumerate(LAYERS):
                y = pn[(tag, fr)][li]
                rho, p = spearman_exact(LAMBDAS, y)
                out["spearman_lambda"][tag][fr][l] = {
                    "rho": float(rho), "p": float(p)}
                out["boundary_half"][tag][fr][l] = cross_below(
                    LAMBDAS, list(y), 0.5)
                out["boundary_nm"][tag][fr][l] = cross_equal(
                    LAMBDAS, list(y), list(pm[(tag, fr)][li]))
            for l in ("T11", "T12"):
                out["deep_contrast"][tag][fr][l] = float(
                    pn[(tag, fr)][LAYERS.index(l)][-1]
                    - pn[(tag, fr)][LAYERS.index(l)][0])

    for fr in FRAMES:
        out["model_agreement"][fr] = {}
        for li, l in enumerate(LAYERS):
            rho, p = spearman_exact(list(pn[("w2v2_base", fr)][li]),
                                    list(pn[("w2v2_base960h", fr)][li]))
            out["model_agreement"][fr][l] = {"rho": float(rho), "p": float(p)}

    global _NULL9
    if _NULL9 is None:
        _NULL9 = _spearman_null_n9()
    crit = None
    for ra, pone in _NULL9:
        if 2.0 * pone <= 0.05:
            crit = ra
        else:
            break
    out["spearman_critical_n9"] = {
        "rho_two_tailed_p05": crit,
        "attainable": [(round(r, 4), round(p, 6)) for r, p in _NULL9[:12]],
    }

    # endpoint distances: cluster lam0 vs dotted canon, lam1 vs dotted assim
    orig = {"P1": ("P1_canon", "P1_assim"), "P2": ("P2_canon", "P2_assim"),
            "P4": ("P4_canon", "P4_assim"), "NV1": ("NV1_canon", "NV1_unlic"),
            "NV3": ("NV3_canon", "NV3_unlic"), "NV4": ("NV4_canon", "NV4_unlic")}
    for fr in FRAMES:
        c, u = orig[fr]
        for tag in MODELS:
            for l in LAYERS:
                for a, oitem in (("0", c), ("1", u)):
                    d = abs(r[tag]["cells"][f"{fr}_l{a}"][l]["proba"]["n"]
                            - r[tag]["original"][oitem][l]["proba"]["n"])
                    out["endpoint_distance"][f"{tag}|{fr}|{a}|{l}"] = float(d)

    _add_pools(out, r)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    print("=" * 78)
    print("v3.0 COMPENSATION RATES (verdict = /n/) at the three anchors")
    print("=" * 78)
    for fr in FRAMES:
        print(f"\n-- {fr} {FRAME_LABEL[fr]}")
        for tag in MODELS:
            for a in ANCHORS:
                c = out["compensation"][tag][fr][a]
                print(f"  {tag:14s} lam={a:4s}  rate={c['rate']*100:5.1f}%  "
                      f"CI=({c['ci_lo']*100:.1f},{c['ci_hi']*100:.1f})  "
                      f"[{c['verdict_profile']}]")
    print("\nP(n) at selected layers (anchors 0 / 0.5 / 1)")
    sel = ["CNN", "T4", "T8", "T11", "T12"]
    for fr in FRAMES:
        for tag in MODELS:
            row = " | ".join(
                f"{l}: " + "/".join(
                    f"{out['layer_detail'][tag][fr][a][l]['n']:.2f}"
                    for a in ANCHORS) for l in sel)
            print(f"  {tag:14s} {fr:4s} {row}")
    print("\nBOUNDARY lambda* (P(n)<0.5) / (P(n)<P(m)) at T4:")
    for tag in MODELS:
        for fr in FRAMES:
            bh = out["boundary_half"][tag][fr]["T4"]
            bn = out["boundary_nm"][tag][fr]["T4"]
            fmt = lambda v: v if not isinstance(v, float) else round(v, 3)
            print(f"  {tag:14s} {fr:4s} {fmt(bh)} / {fmt(bn)}")
    print("\nSPEARMAN rho(lambda, P(n)) at T4:")
    for tag in MODELS:
        s = " ".join(f"{fr}={out['spearman_lambda'][tag][fr]['T4']['rho']:+.2f}"
                     for fr in FRAMES)
        print(f"  {tag:14s} {s}")
    print("\nDEEP CONTRAST P(n)|l1 - P(n)|l0 (T11/T12):")
    for tag in MODELS:
        for fr in FRAMES:
            dc = out["deep_contrast"][tag][fr]
            print(f"  {tag:14s} {fr:4s} T11={dc['T11']:+.3f} T12={dc['T12']:+.3f}")
    mx = max(out["endpoint_distance"].values())
    print(f"\nendpoint distance max |dP(n)| cluster vs dotted: {mx:.4f}")
    print(f"exact Spearman critical (n=9, two-tailed p=.05): "
          f"rho = {out['spearman_critical_n9']['rho_two_tailed_p05']}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

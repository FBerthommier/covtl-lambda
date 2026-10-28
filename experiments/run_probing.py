# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Linear probing of wav2vec2 hidden states on the assimilation base.

Analyses (per model tag, per layer L0=CNN .. L12):
 1. Probe validation: 3-way place classification (nasals n/m/N, stops t/p/k),
    leave-one-sentence-out CV on the 18 training sentences per probe.
 2. Full-data probe -> classification of the test items:
      pair_assim  (viable assimilation)  -> surface vs underlying response
      nv_unlic    (unlicensed variant)   -> same
      pair_canon / nv_canon              -> sanity (expect underlying class)
    Soft probabilities are also stored.
 3. Centroid distances (cosine) from each test item to the class centroids
    built from the training window means; margin = d(under) - d(surf).
 4. Causal delta: per pair, 1 - cos(h_canon(t), h_assim(t)) per layer/time.
 5. Timecourse for showcase pairs: frame-wise cosine distance to the two
    relevant class centroids (canonical + assimilated variants).

Outputs -> <KIT_ROOT>/work/results/<tag>/
  probe_cv.json, classification.json, margins.json, delta.npz,
  timecourse_<pair>.npz
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Kit portability: root of the experiment kit (parent of repro/) ------
KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
BASE = os.path.join(KIT_ROOT, "work", "base")
FEAT = os.path.join(KIT_ROOT, "work", "features")
RES = os.path.join(KIT_ROOT, "work", "results")
MODEL_TAGS = ["w2v2_base960h", "w2v2_base"]
LAYER_NAMES = ["CNN"] + [f"T{i}" for i in range(1, 13)]

PROBES = {
    "nasal": {"classes": ["n", "m", "N"]},
    "stop": {"classes": ["t", "p", "k"]},
}


def load_item(tag, label):
    d = np.load(os.path.join(FEAT, tag, label + ".npz"), allow_pickle=True)
    return d["hs"], d["t_centers"], json.loads(str(d["windows"]))


def frames_in(t_centers, lo, hi):
    return (t_centers >= lo) & (t_centers <= hi)


def window_mean(hs, t_centers, win):
    m = frames_in(t_centers, win[0], win[1])
    return hs[:, m, :].mean(axis=1)  # (L, D)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    with open(os.path.join(BASE, "base_manifest.json")) as f:
        manifest = json.load(f)
    items = {it["label"]: it for it in manifest["items"]}

    for tag in MODEL_TAGS:
        outdir = os.path.join(RES, tag)
        os.makedirs(outdir, exist_ok=True)

        # ---------------- load window means ----------------
        wm = {}   # label -> (L, D) target-window mean
        fr = {}   # label -> full hs + t_centers + windows
        for lab, it in items.items():
            hs, tc, wins = load_item(tag, lab)
            wm[lab] = window_mean(hs, tc, wins["target"])
            fr[lab] = (hs, tc, wins)

        # check pair time alignment
        for pid in [it["pid"] for it in manifest["items"] if it["group"] == "pair_canon"]:
            c, u = items[f"{pid}_canon"], items[f"{pid}_assim"]
            assert c["t0_cs"] == u["t0_cs"] and c["t1_cs"] == u["t1_cs"], pid

        train_labels = [lab for lab, it in items.items() if it["group"] == "train"]

        results = {"probe_cv": {}, "classification": {}, "margins": {}}

        for L in range(13):
            lname = LAYER_NAMES[L]

            # ---------- 1. CV probe validation ----------
            for pname, pcfg in PROBES.items():
                cls = pcfg["classes"]
                labs = [l for l in train_labels if items[l]["target_key"] in cls]
                X = np.stack([wm[l][L] for l in labs])
                y = np.array([cls.index(items[l]["target_key"]) for l in labs])
                # leave-one-sentence-out
                preds = np.zeros_like(y)
                for i in range(len(y)):
                    m = np.ones(len(y), bool)
                    m[i] = False
                    clf = LogisticRegression(max_iter=5000, C=1.0)
                    clf.fit(X[m], y[m])
                    preds[i] = clf.predict(X[i:i + 1])[0]
                acc = float(accuracy_score(y, preds))
                results["probe_cv"].setdefault(pname, {})[lname] = dict(
                    accuracy=acc, n=len(y))

            # ---------- 2+3. full-data probe -> test items ----------
            for pname, pcfg in PROBES.items():
                cls = pcfg["classes"]
                labs = [l for l in train_labels if items[l]["target_key"] in cls]
                X = np.stack([wm[l][L] for l in labs])
                y = np.array([cls.index(items[l]["target_key"]) for l in labs])
                clf = LogisticRegression(max_iter=5000, C=1.0).fit(X, y)

                # class centroids (cosine space: normalized means)
                cent = {}
                for ci, c in enumerate(cls):
                    cent[c] = X[y == ci].mean(axis=0)

                test_groups = ["pair_canon", "pair_assim", "nv_canon", "nv_unlic"]
                for g in test_groups:
                    for lab, it in items.items():
                        if it["group"] != g:
                            continue
                        if it["target"] not in cls:
                            continue
                        v = wm[lab][L]
                        pred = cls[int(clf.predict(v[None])[0])]
                        proba = clf.predict_proba(v[None])[0]
                        # margins relative to underlying/surface centroids
                        d_surf = 1 - cos(v, cent[it["surface"]])
                        d_und = 1 - cos(v, cent[it["target"]])
                        rec = results["classification"].setdefault(lab, {}).setdefault(lname, {})
                        rec.update(dict(
                            pred=pred,
                            proba={c: float(proba[cls.index(c)]) for c in cls},
                            d_surface=d_surf, d_underlying=d_und,
                            margin=d_und - d_surf,
                            is_surface=(pred == it["surface"]),
                            is_underlying=(pred == it["target"]),
                        ))

        # ---------- 4. causal delta per pair ----------
        delta = {}
        for it in manifest["items"]:
            if it["group"] != "pair_canon":
                continue
            pid = it["pid"]
            hc, tc, _ = fr[f"{pid}_canon"]
            ha, ta, _ = fr[f"{pid}_assim"]
            assert len(tc) == len(ta), f"frame count mismatch for {pid}"
            # cos per layer per frame
            hn_c = hc / (np.linalg.norm(hc, axis=2, keepdims=True) + 1e-12)
            hn_a = ha / (np.linalg.norm(ha, axis=2, keepdims=True) + 1e-12)
            cos_lt = (hn_c * hn_a).sum(axis=2)  # (L, T)
            delta[pid] = 1 - cos_lt
            delta[pid + "_t"] = tc
        np.savez_compressed(
            os.path.join(outdir, "delta.npz"),
            **{k: v for k, v in delta.items()})

        # ---------- 5. timecourse for showcase pairs ----------
        for pid in ["P1", "P5", "P3"]:
            cls = PROBES["nasal" if pid in ("P1", "P3") else "stop"]["classes"]
            labs = [l for l in train_labels if items[l]["target_key"] in cls]
            data = {}
            for variant in ["canon", "assim"]:
                hs, tc, wins = fr[f"{pid}_{variant}"]
                # frame-wise distance to each class centroid, per layer (L, T)
                d = {c: np.zeros((13, len(tc))) for c in cls}
                for L in range(13):
                    cent_L = {}
                    for ci, c in enumerate(cls):
                        Xc = np.stack([wm[l][L] for l in labs
                                       if items[l]["target_key"] == c])
                        cent_L[c] = Xc.mean(axis=0)
                    hn = hs[L] / (np.linalg.norm(hs[L], axis=1, keepdims=True) + 1e-12)
                    for c in cls:
                        cn = cent_L[c] / (np.linalg.norm(cent_L[c]) + 1e-12)
                        d[c][L] = 1 - hn @ cn
                data[f"{variant}_t"] = tc
                for c in cls:
                    data[f"{variant}_d_{c}"] = d[c]
            data["target_win"] = np.array(fr[f"{pid}_canon"][2]["target"])
            np.savez_compressed(os.path.join(outdir, f"timecourse_{pid}.npz"), **data)

        with open(os.path.join(outdir, "probe_cv.json"), "w") as f:
            json.dump(results["probe_cv"], f, indent=1)
        with open(os.path.join(outdir, "classification.json"), "w") as f:
            json.dump(results["classification"], f, indent=1)

        # ---------- console summary ----------
        print(f"\n===== {tag} =====")
        for pname in PROBES:
            accs = [results["probe_cv"][pname][LAYER_NAMES[L]]["accuracy"] for L in range(13)]
            print(f"CV acc {pname:6s}: " + " ".join(f"{a:.2f}" for a in accs))
        # headline: per layer, fraction of assimilated tokens -> surface
        for L in [4, 6, 8, 10, 12]:
            lname = LAYER_NAMES[L]
            vals = []
            for lab, it in items.items():
                if it["group"] == "pair_assim":
                    vals.append(results["classification"][lab][lname]["is_surface"])
            print(f"L{L:2d} ({lname:4s}): assimilated -> surface "
                  f"{sum(vals)}/{len(vals)}")
        print("per-pair detail at best layers:")
        for lab, it in items.items():
            if it["group"] == "pair_assim":
                row = []
                for L in range(13):
                    r = results["classification"][lab][LAYER_NAMES[L]]
                    row.append("S" if r["is_surface"] else ("U" if r["is_underlying"] else "o"))
                print(f"  {lab:12s} {it['process']:5s} {''.join(row)}")


if __name__ == "__main__":
    main()

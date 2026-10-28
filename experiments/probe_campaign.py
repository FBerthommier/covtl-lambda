# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Probe the 54 lambda-continuum items of the campaign (covtl-lambda
v1.2.0) with the unchanged nasal probe instrument trained on the
original stage-1 training base (LOSO curves identical to the stage-1
report -- the same instrument for endpoints, continuum, and
production degrees).

Instruments (window = C1 phoneme window from the campaign manifest):
  1. classification: per lambda x layer -- class probabilities +
     argmax of the nasal probe (the note's verdicts)
  2. the campaign extras (margins, switch points, timecourses) are
     written alongside for the figures and the audit trail.

Kit-portable: reads COVTL_KIT_ROOT/work/{base,features,lambda},
writes COVTL_KIT_ROOT/work/lambda/{results.json,campaign/}.
"""
import json
import os
import sys

KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
BASE = os.path.join(KIT_ROOT, "work", "base")
FEAT0 = os.path.join(KIT_ROOT, "work", "features")
LAM = os.path.join(KIT_ROOT, "work", "lambda")
OUT = LAM

MODELS = ["w2v2_base960h", "w2v2_base"]
LAYERS = ["CNN"] + [f"T{i}" for i in range(1, 13)]
NASAL = ["n", "m", "N"]


def load_npz(tag, label):
    import numpy as np
    return np.load(os.path.join(FEAT0, tag, f"{label}.npz"))


def window_mean(hs, t, win):
    import numpy as np
    m = (t >= win[0]) & (t < win[1])
    assert m.any(), (win, t[0], t[-1])
    return hs[:, m, :].mean(axis=1)          # (13, 768)


def extract_campaign_features(campaign, feats_dir):
    """wav2vec2 hidden states for the 54 campaign wavs."""
    import numpy as np
    import torch
    import torchaudio
    from transformers import Wav2Vec2Model
    n_done = 0
    for tag in MODELS:
        os.makedirs(os.path.join(feats_dir, tag), exist_ok=True)
        model = Wav2Vec2Model.from_pretrained(
            "facebook/" + ("wav2vec2-base-960h" if tag == "w2v2_base960h"
                           else "wav2vec2-base"))
        model.eval()
        for fr in campaign["frames"]:
            for it in fr["items"]:
                lab = f"{fr['fid']}_l{it['lam']:g}"
                dst = os.path.join(feats_dir, tag, f"{lab}.npz")
                if os.path.exists(dst):
                    continue
                wav, sr = torchaudio.load(it["wav"])
                wav = torchaudio.functional.resample(
                    wav, sr, 16000).mean(dim=0)
                wav = (wav - wav.mean()) / (wav.std() + 1e-9)
                with torch.no_grad():
                    hs = model(wav[None]).last_hidden_state[0]  # (T,13,768)
                t_centers = (np.arange(hs.shape[0]) + 0.5) * 0.02
                np.savez(dst, hs=hs.numpy().astype("float32"),
                         t_centers=t_centers)
                n_done += 1
        del model
    print(f"features: {n_done} new npz")


def main():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    try:
        import torch, torchaudio  # noqa: F401
        from transformers import Wav2Vec2Model  # noqa: F401
        HAVE_TORCH = True
    except ImportError:
        HAVE_TORCH = False

    campaign = json.load(open(os.path.join(LAM, "campaign_manifest.json")))
    citems = [(fr, it) for fr in campaign["frames"] for it in fr["items"]]
    feats_dir = os.path.join(LAM, "features")

    if HAVE_TORCH:
        extract_campaign_features(campaign, feats_dir)
    else:
        if not os.path.isdir(os.path.join(feats_dir, MODELS[0])):
            sys.exit("torch/transformers unavailable and no cached "
                     "campaign features: install repro/requirements.txt")

    train_manifest = json.load(open(os.path.join(BASE,
                                                 "base_manifest.json")))
    train_items = [it for it in train_manifest["items"]
                   if it["group"] == "train"]

    results = {"provenance": {
        "probe": "nasal-place logistic probe on window means, training "
                 "base = the original stage-1 base (instrument "
                 "unchanged across kit versions)",
        "window": "C1 phoneme window (from the campaign manifest)"}}
    probe_cv = {}
    for tag in MODELS:
        X = {}
        for it in train_items:
            z = load_npz(tag, it["label"])
            win = (it["t0_cs"] / 100.0, it["t1_cs"] / 100.0)
            X[it["label"]] = window_mean(z["hs"], z["t_centers"], win)
        nasal_items = [it for it in train_items if it["target_key"] in NASAL]
        cells = {}
        cv = {}
        for li, lname in enumerate(LAYERS):
            Xtr = np.stack([X[it["label"]][li] for it in nasal_items])
            ytr = [it["target_key"] for it in nasal_items]
            clf = LogisticRegression(max_iter=5000, C=1.0)
            clf.fit(Xtr, ytr)
            # LOSO CV on the training sentences
            correct = 0
            for i in range(len(ytr)):
                keep = [j for j in range(len(ytr)) if j != i]
                same = [j for j in keep if ytr[j] == ytr[i]]
                if not same:
                    continue
                lo = LogisticRegression(max_iter=5000, C=1.0)
                lo.fit(Xtr[keep], [ytr[j] for j in keep])
                correct += int(lo.predict(Xtr[i:i + 1])[0] == ytr[i])
            cv[lname] = round(correct / len(ytr), 4)
            for fr, it in citems:
                lab = f"{fr['fid']}_l{it['lam']:g}"
                z = np.load(os.path.join(feats_dir, tag, f"{lab}.npz"))
                win = tuple(fr["c1_window_s"])
                emb = window_mean(z["hs"], z["t_centers"], win)
                p = clf.predict_proba(emb[li][None])[0]
                cells.setdefault(lab, {})[lname] = {
                    "N": float(p[list(clf.classes_).index("N")]),
                    "m": float(p[list(clf.classes_).index("m")]),
                    "n": float(p[list(clf.classes_).index("n")]),
                    "argmax": clf.classes_[int(p.argmax())]}
        probe_cv[tag] = cv
        results.setdefault("nasal", {})[tag] = cells
        print(f"[{tag}] LOSO:", {k: v for k, v in cv.items()
                                 if k in ("CNN", "T2", "T4", "T12")})

    results["probe_cv_nasal"] = probe_cv
    with open(os.path.join(OUT, "results.json.tmp"), "w") as f:
        json.dump({"nasal": results["nasal"]}, f, indent=1)
    # campaign instruments directory (classification + cv)
    os.makedirs(os.path.join(LAM, "campaign"), exist_ok=True)
    with open(os.path.join(LAM, "campaign", "campaign_probing.json"),
              "w") as f:
        json.dump({"nasal": results["nasal"]}, f, indent=1)
    with open(os.path.join(LAM, "campaign", "probe_train.json"), "w") as f:
        json.dump({"nasal": probe_cv}, f, indent=1)
    os.remove(os.path.join(OUT, "results.json.tmp"))
    print(f"probing done -> {LAM}/campaign/")


if __name__ == "__main__":
    main()

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
  2. LOSO CV of the nasal and stop probes on the training sentences
  3. centroid cosine margins (nasal probe)
  4. switch points: per frame x model x layer, the smallest lambda
     where the nasal-probe argmax leaves 'n' (and reaches the
     C2-place class)
  5. timecourses: per-frame nasal-probe probabilities around the C1
     window for lambda in {0, .25, .5, .75, 1}

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
                    out = model(wav[None], output_hidden_states=True)
                    hs = np.stack([h[0].numpy() for h in out.hidden_states],
                                  axis=0)  # (13, T, 768)
                t_centers = (np.arange(hs.shape[1]) + 0.5) * 0.02
                np.savez(dst, hs=hs.astype("float32"),
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

    PROBES = {"nasal": NASAL, "stop": ["t", "p", "k"]}
    TC_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
    probe_cv = {}
    classification = {}
    margins = {}
    tc = {}
    for tag in MODELS:
        X = {}
        for it in train_items:
            z = load_npz(tag, it["label"])
            win = (it["t0_cs"] / 100.0, it["t1_cs"] / 100.0)
            X[it["label"]] = window_mean(z["hs"], z["t_centers"], win)
        for pname, cls in PROBES.items():
            labs = [it for it in train_items if it["target_key"] in cls]
            cv = {}
            for li, lname in enumerate(LAYERS):
                Xtr = np.stack([X[it["label"]][li] for it in labs])
                ytr = [it["target_key"] for it in labs]
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
                if pname != "nasal":
                    continue      # the nasal probe = the C1 instrument
                clf = LogisticRegression(max_iter=5000, C=1.0)
                clf.fit(Xtr, ytr)
                cent = {c: Xtr[[i for i, y in enumerate(ytr)
                               if y == c]].mean(axis=0) for c in cls}
                for fr, it in citems:
                    lab = f"{fr['fid']}_l{it['lam']:g}"
                    z = np.load(os.path.join(feats_dir, tag,
                                             f"{lab}.npz"))
                    win = tuple(fr["c1_window_s"])
                    emb = window_mean(z["hs"], z["t_centers"], win)
                    p = clf.predict_proba(emb[li][None])[0]
                    classification.setdefault(tag, {}).setdefault(
                        lab, {})[lname] = {
                        "N": round(float(p[list(clf.classes_).index("N")]), 4),
                        "m": round(float(p[list(clf.classes_).index("m")]), 4),
                        "n": round(float(p[list(clf.classes_).index("n")]), 4),
                        "argmax": str(clf.classes_[int(p.argmax())])}
                    v = emb[li]
                    v = v / (np.linalg.norm(v) + 1e-9)
                    d = {}
                    for c in cls:
                        u = cent[c] / (np.linalg.norm(cent[c]) + 1e-9)
                        d[c] = round(float(1.0 - float(v @ u)), 4)
                    margins.setdefault(tag, {}).setdefault(
                        lab, {})[lname] = d
                    if it["lam"] in TC_LAMBDAS:
                        m = ((z["t_centers"] >= win[0] - 0.15)
                             & (z["t_centers"] < win[1] + 0.15))
                        hs_f = z["hs"][:, m, :]
                        pf = clf.predict_proba(hs_f[li])
                        key = f"{lab}|{lname}"
                        tc[key] = pf.astype(np.float32)
                        tc.setdefault("_" + key + "_t",
                                      z["t_centers"][m].astype(np.float32))
            probe_cv.setdefault(pname, {})[tag] = cv
        print(f"[{tag}] LOSO:", {k: v for k, v in
                                 probe_cv["nasal"][tag].items()
                                 if k in ("CNN", "T2", "T4", "T12")})

    # campaign instruments directory
    os.makedirs(os.path.join(LAM, "campaign"), exist_ok=True)
    with open(os.path.join(LAM, "campaign", "campaign_probing.json"),
              "w") as f:
        json.dump({"nasal": classification,
                   "stop": {t: {} for t in MODELS}}, f, indent=1)
    with open(os.path.join(LAM, "campaign", "probe_train.json"), "w") as f:
        json.dump(probe_cv, f, indent=1)
    with open(os.path.join(LAM, "campaign", "margins.json"), "w") as f:
        json.dump({"nasal": margins, "stop": {t: {} for t in MODELS}},
                  f, indent=1)
    np.savez_compressed(os.path.join(LAM, "campaign", "timecourses.npz"),
                        **tc)

    # switch points (nasal argmax paths along lambda)
    sw = {}
    for tag in MODELS:
        sw[tag] = {}
        for fr in campaign["frames"]:
            fid = fr["fid"]
            sw[tag][fid] = {}
            for lname in LAYERS:
                row = []
                for it in fr["items"]:
                    rec = classification[tag][f"{fid}_l{it['lam']:g}"][lname]
                    row.append((it["lam"], rec["argmax"]))
                first_leave = next((lam for lam, am in row
                                    if am != "n"), None)
                c2cls = {"p": "m", "b": "m", "t": "n",
                         "k": "N", "g": "N"}[fr["c2_phone"]]
                first_reach = next((lam for lam, am in row
                                    if am == c2cls and c2cls != "n"),
                                   None)
                sw[tag][fid][lname] = {
                    "first_lambda_leaving_n": first_leave,
                    "first_lambda_reaching_c2_class": first_reach,
                    "argmax_path": [am for _, am in row]}
    with open(os.path.join(LAM, "campaign", "switch_points.json"), "w") as f:
        json.dump(sw, f, indent=1)
    print(f"probing done -> {LAM}/campaign/")


if __name__ == "__main__":
    main()

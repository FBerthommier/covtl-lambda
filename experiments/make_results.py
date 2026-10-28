# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""KIT30 step 10 -- build the v3.0 lambda data blocks of the kit.

Sources (measured, not re-run):
  - campaign_probing.json  (wav2vec2_lambda_campaign, covtl-lambda v1.2.0,
    54 items = 6 frames x 9 lambdas, nasal probe, window = C1 window)
  - v2.1 lambda075/results.json "original" block (the 12 dotted endpoint
    probings, same probe instrument -- verified: stage-1 values match to
    1.4e-4) -- reused as the dotted-citation reference tier
  - campaign_manifest.json (provenance, windows, guarantees)

Outputs (kit schema, blocks under work/):
  work/lambda/results.json        54 items, all 9 lambdas
  work/lambda/synth_meta.json     provenance + per-frame windows/checks
  work/lambda075/results.json     18 items (lam in {0, 0.75, 1})
  work/lambda0625/results.json    24 items (lam in {0, 0.625, 0.75, 1})
  + tracts/ wavs/ subsets copied byte-identically from work/lambda
"""
import hashlib
import json
import os
import shutil

KIT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
CAMP = os.path.join(KIT, "work", "lambda", "campaign")
# (the campaign result files live directly in CAMP)
MODELS = ["w2v2_base960h", "w2v2_base"]
FRAMES = ["P1", "P2", "P4", "NV1", "NV3", "NV4"]
LAMS9 = ["0", "0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1"]
SUB075 = ["0", "0.75", "1"]
SUB0625 = ["0", "0.625", "0.75", "1"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cells_from_campaign():
    """campaign_probing.json -> kit cells schema {model: {item: {layer: {pred, proba}}}}"""
    cp = json.load(open(f"{CAMP}/campaign_probing.json"))
    cells = {tag: {} for tag in MODELS}
    for tag in MODELS:
        for item, layers in cp["nasal"][tag].items():
            cells[tag][item] = {
                L: {"pred": v["argmax"],
                    "proba": {"n": v["n"], "m": v["m"], "N": v["N"]}}
                for L, v in layers.items()}
    return cells


def main():
    cells = cells_from_campaign()
    n_items = len(cells[MODELS[0]])
    assert n_items == 54, f"expected 54 items, got {n_items}"

    # dotted-citation reference tier, reused from the v2.1 measurement
    v21 = json.load(open(f"{KIT}/work/legacy_motorblend/lambda075/results.json"))
    original = {tag: v21[tag]["original"] for tag in MODELS}
    probe_cv = json.load(open(f"{CAMP}/probe_train.json"))["nasal"]

    prov = {
        "lambda_model": "covtl-lambda v1.2.0 (repo commit c13f4d0) -- "
                        "target-level semantic on tautosyllabic CC clusters",
        "engine": "covtl-pipeline 1d15ed7 + covtl-corrections-patch v1.0.1 "
                  "(= committed fix/polar-teardrop-arcplot content)",
        "campaign": "wav2vec2_lambda_6frames_v1 (2026-10-28), 54 syntheses, "
                    "4/4 run-time guarantees held 54/54",
        "probe": "nasal-place logistic probe on window means, training base "
                 "= the original 36-sentence base (LOSO curves identical to "
                 "the stage-1 report; instrument unchanged across kit "
                 "versions)",
        "window": "C1 phoneme window (400 Hz frame span, from morph_meta)",
        "dotted_reference": "the 'original' block reuses the v2.1 dotted-"
                            "endpoint probings (klin.pan / klim.pan tier); "
                            "verified equal to the stage-1 classification "
                            "to 1.4e-4 (float32 round-trip)",
    }

    # ---- work/lambda : full continuum --------------------------------------
    lam_dir = f"{KIT}/work/lambda"
    res = {tag: {"cells": cells[tag], "original": original[tag],
                 "probe_cv": probe_cv[tag], "provenance": prov}
           for tag in MODELS}
    with open(f"{lam_dir}/results.json", "w") as f:
        json.dump(res, f, indent=1)

    # synth_meta from the campaign manifest
    man = json.load(open(f"{KIT}/work/lambda/synth_meta.json"))
    meta = {
        "kit_version": "3.0",
        "campaign": man["campaign"],
        "created": man["created"],
        "lambda_model": man["lambda_model"],
        "reference_tree": man["reference_tree"],
        "lambdas": man["lambdas"],
        "frames": [{k: fr[k] for k in
                    ("fid", "sentence", "licensing", "canon_cluster",
                     "assim_denomination", "c2_phone", "c1", "c2",
                     "theta_delta_deg", "c1_window_s", "c1_window_frames",
                     "n_frames", "checks")}
                   for fr in man["frames"]],
        "guarantees_held": "54/54 items (see per-frame checks)",
        "files": {},
    }
    for fr in FRAMES:
        for l in LAMS9:
            for ext in ("tract", "wav"):
                p = f"{lam_dir}/{ext}s/{fr}_l{l}.{ext}"
                meta["files"][f"{fr}_l{l}.{ext}"] = sha256(p)
    with open(f"{lam_dir}/synth_meta.json", "w") as f:
        json.dump(meta, f, indent=1)

    # ---- subsets ------------------------------------------------------------
    for blk, lams in (("lambda075", SUB075), ("lambda0625", SUB0625)):
        d = f"{KIT}/work/{blk}"
        os.makedirs(f"{d}/tracts", exist_ok=True)
        os.makedirs(f"{d}/wavs", exist_ok=True)
        sub_cells = {}
        for tag in MODELS:
            sub_cells[tag] = {
                f"{fr}_l{l}": cells[tag][f"{fr}_l{l}"]
                for fr in FRAMES for l in lams}
        res = {tag: {"cells": sub_cells[tag], "original": original[tag],
                     "probe_cv": probe_cv[tag], "provenance": prov}
               for tag in MODELS}
        with open(f"{d}/results.json", "w") as f:
            json.dump(res, f, indent=1)
        for fr in FRAMES:
            for l in lams:
                for ext in ("tract", "wav"):
                    src = f"{lam_dir}/{ext}s/{fr}_l{l}.{ext}"
                    dst = f"{d}/{ext}s/{fr}_l{l}.{ext}"
                    shutil.copy2(src, dst)
        n_expect = len(FRAMES) * len(lams)
        n_got = len(sub_cells[MODELS[0]])
        assert n_got == n_expect, (blk, n_got, n_expect)
        print(f"work/{blk}: {n_got} items, assets copied")

    # subset identity note (byte-level)
    for blk, lams in (("lambda075", SUB075), ("lambda0625", SUB0625)):
        for fr in FRAMES:
            for l in lams:
                for ext in ("tract", "wav"):
                    a = sha256(f"{lam_dir}/{ext}s/{fr}_l{l}.{ext}")
                    b = sha256(f"{KIT}/work/{blk}/{ext}s/{fr}_l{l}.{ext}")
                    assert a == b, (blk, fr, l, ext)
    print("subset assets byte-identical to the continuum: OK")
    print("work/lambda: 54 items | results.json + synth_meta.json written")


if __name__ == "__main__":
    main()

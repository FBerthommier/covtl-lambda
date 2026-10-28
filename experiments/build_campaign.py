# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Build the lambda-continuum campaign on the REVISED reference model
(covtl-lambda v1.2.0, target-level C1C2 semantic on tautosyllabic
clusters): for each of the six Pouw frames, the 9-step lambda
continuum (0, .125, ..., 1) of the canonical cluster phrase,
re-synthesized with the lambda-interpolated C1 polar target.
54 items total.

Reference tree: covtl-pipeline 1d15ed7 + corrections series (the
committed fix/polar-teardrop-arcplot content) + covtl-lambda v1.2.0
(repo c13f4d0). The four run-time guarantees are asserted per frame.

Kit-portable: writes under COVTL_KIT_ROOT/work/lambda.
"""
import json
import os
import sys

KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
OUT = os.path.join(KIT_ROOT, "work", "lambda")

LAMBDAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

# (fid, sentence, licensing, canonical cluster, assimilated
#  denomination -- display only, C2 phone)
FRAMES = [
    ("P1",  "clean pan",   "licensed",   "klinpan",  "klimpan",  "p"),
    ("P2",  "mean boat",   "licensed",   "minbot",   "mimbot",   "b"),
    ("P4",  "thin packet", "licensed",   "Tinpakit", "Timpakit", "p"),
    ("NV1", "clean tan",   "unlicensed", "klintan",  "klimtan",  "t"),
    ("NV3", "clean can",   "unlicensed", "klinkan",  "klimkan",  "k"),
    ("NV4", "mean goat",   "unlicensed", "mingot",   "mimgot",   "g"),
]


def main():
    try:
        from vtl_synth.core.morphing import morph_targets
    except ImportError:
        sys.exit("covtl-lambda v1.2.0 not installed: apply the patch "
                 "to your covtl-pipeline checkout (see README step 0) "
                 "and pip install -e it, or set PYTHONPATH to its root")

    os.makedirs(f"{OUT}/tracts", exist_ok=True)
    os.makedirs(f"{OUT}/wavs", exist_ok=True)
    manifest = {
        "campaign": "wav2vec2_lambda_6frames_v1",
        "lambda_model": "covtl-lambda v1.2.0 (target semantic, theta "
                        "channel, rho fixed)",
        "lambdas": LAMBDAS,
        "provenance_note": "C1 window in seconds = c1_window_frames "
                           "@ 400 Hz (phoneme span); the wav2vec2 probe "
                           "window uses it directly.",
        "frames": [],
    }
    n_items = 0
    for fid, sentence, lic, canon, assim, c2 in FRAMES:
        meta = morph_targets(canon, f"{OUT}/_engine_{fid}",
                             lambdas=LAMBDAS, with_audio=True, label=fid)
        assert meta["mode"] == "targets", fid
        assert all(meta["checks"].values()), (fid, meta["checks"])
        tg = meta["targets"]
        win = tg["c1_window_frames"]
        frame_rec = {
            "fid": fid, "sentence": sentence, "licensing": lic,
            "canon_cluster": canon, "assim_denomination": assim,
            "c2_phone": c2,
            "c1": tg["c1"], "c2": tg["c2"],
            "theta_delta_deg": tg["theta_delta_deg"],
            "c1_window_s": [win["first"] / 400.0,
                            (win["last"] + 1) / 400.0],
            "c1_window_frames": [win["first"], win["last"]],
            "c1_arcs_frames": [tg["c1_arcs_frames"]["first"],
                               tg["c1_arcs_frames"]["last"]],
            "n_frames": meta["n_frames"],
            "checks": meta["checks"],
        }
        # move the assets to the kit layout + record the morph meta
        import shutil
        for it in meta["items"]:
            for ext in ("tract", "wav"):
                src = it[ext]
                dst = f"{OUT}/{'tracts' if ext == 'tract' else 'wavs'}/" \
                      f"{fid}_l{it['lam']:g}.{ext}"
                shutil.move(src, dst)
        meta_out = {k: meta[k] for k in
                    ("patch", "mode", "semantic", "label", "phrase",
                     "sources", "targets")}
        with open(f"{OUT}/morph_meta_{fid}.json", "w") as f:
            json.dump(meta_out, f, indent=1)
        manifest["frames"].append(frame_rec)
        n_items += len(meta["items"])
        print(f"[{fid}] {canon} -> C2 /{c2}/ ({lic}): "
              f"dtheta {tg['theta_delta_deg']:+.2f} deg, "
              f"C1 win {win['first']}-{win['last']}, "
              f"{len(meta['items'])} lambdas, checks OK")
    with open(f"{OUT}/campaign_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nDONE: {n_items} items -> {OUT}")


if __name__ == "__main__":
    main()

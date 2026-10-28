# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Build the probing sentence base: synthesize all items with vtl-synth.
    Synthesis chain: the COVTL articulatory model (F. Berthommier)
    in covtl-pipeline driving VocalTractLab 2.4 (P. Birkholz,
    https://www.vocaltractlab.de) - cite both for scientific use.

Outputs per item (into <KIT_ROOT>/work/base/<label>/):
  <label>.wav   44.1 kHz audio (both variants of a pair have identical
                phoneme counts => identical durations => frame alignment)
  <label>.polar JSON v3 (100 Hz polar trajectories + phoneme timing)
  <label>.tract    (400 Hz tract parameters, kept for provenance)
  <label>.txt      (SAMPA transcript)

Also writes base_manifest.json with per-item metadata:
  keys (expected phoneme key sequence), target_idx, t0/t1 (cs @100 Hz),
  group, variant, place/manner labels.

Idempotent: existing complete outputs are skipped.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentence_base import all_items, PLACE, MANNER  # noqa: E402

from vtl_synth.core.phonemes import greedy_sampa_split  # noqa: E402

# --- Kit portability: root of the experiment kit (parent of repro/) ------
KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
BASE = os.path.join(KIT_ROOT, "work", "base")
MANIFEST = os.path.join(BASE, "base_manifest.json")


def expected_keys(sampa):
    keys = []
    for word in sampa.split("."):
        keys.extend(greedy_sampa_split(word))
    return keys


def synth(label, sampa):
    outdir = os.path.join(BASE, label)
    wav = os.path.join(outdir, f"{label}.wav")
    polar = os.path.join(outdir, f"{label}.polar")
    if os.path.exists(wav) and os.path.exists(polar):
        return "skipped"
    os.makedirs(outdir, exist_ok=True)
    cmd = ["vtl-synth", "run", "--polar", "--no-video", "--phonetic",
           sampa, "-l", "en", "-o", outdir, "--label", label]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(r.stdout[-800:])
        print(r.stderr[-800:])
        raise RuntimeError(f"synthesis failed for {label} ({sampa})")
    return f"{time.time()-t0:.1f}s"


def main():
    os.makedirs(BASE, exist_ok=True)
    manifest = {"items": [], "engine_patch": "N velar closure (see scripts/patch_N.py)"}
    items = all_items()
    n_fail = 0
    for i, (label, sampa, group, meta) in enumerate(items):
        status = synth(label, sampa)
        outdir = os.path.join(BASE, label)
        polar_path = os.path.join(outdir, f"{label}.polar")
        wav_path = os.path.join(outdir, f"{label}.wav")
        try:
            with open(polar_path) as f:
                pol = json.load(f)
            keys = [p["key"] for p in pol["phonemes"]]
            exp = expected_keys(sampa)
            assert keys == exp, f"key mismatch: polar={keys} expected={exp}"
            t_idx = meta["target_idx"]
            tgt = pol["phonemes"][t_idx]
            assert tgt["key"] == exp[t_idx]
            dur = pol["duration_s"]
            entry = dict(
                label=label, sampa=sampa, group=group, variant=meta["variant"],
                sentence=meta.get("sentence", ""), keys=exp,
                target_idx=t_idx, target_key=exp[t_idx],
                t0_cs=tgt["t0"], t1_cs=tgt["t1"],
                duration_s=dur, n_frames_tract=pol["n_steps"],
                wav=os.path.relpath(wav_path, BASE),
                polar=os.path.relpath(polar_path, BASE),
            )
            if group == "train":
                entry["place"] = meta["place"]
                entry["manner"] = meta["manner"]
            else:
                entry["process"] = meta.get("process")
                entry["context"] = meta.get("context")
                entry["pid"] = meta.get("pid")
                entry["target"] = meta.get("target")      # underlying
                entry["surface"] = meta.get("surface")    # assimilated surface
            manifest["items"].append(entry)
            print(f"[{i+1:2d}/{len(items)}] {label:12s} {sampa:12s} OK ({status}) "
                  f"keys={len(exp)} target[{t_idx}]={exp[t_idx]} t={tgt['t0']}-{tgt['t1']}cs dur={dur}s")
        except Exception as e:
            n_fail += 1
            print(f"[{i+1:2d}/{len(items)}] {label:12s} {sampa:12s} FAIL: {e}")
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{len(manifest['items'])} items OK, {n_fail} failures")
    print(f"manifest -> {MANIFEST}")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

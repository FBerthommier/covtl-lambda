# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Extract wav2vec2 hidden states for every item of the probing base.

For each item and each model (facebook/wav2vec2-base-960h, facebook/wav2vec2-base):
  - load the 44.1 kHz wav, resample to 16 kHz (polyphase)
  - per-utterance normalization (zero mean, unit std), as Wav2Vec2FeatureProcessor
  - forward pass with output_hidden_states=True
  - save all 13 layers (CNN + T1..T12), shape (13, T, 768), float32

Frame grid: wav2vec2-base stride = 320 samples @16 kHz -> 50 Hz,
frame k covers [k*0.02, (k+1)*0.02) s, center (k+0.5)*0.02.

Output: <KIT_ROOT>/work/features/<model_tag>/<label>.npz
        hs: (13, T, 768)   t_centers: (T,) seconds
        plus target window (s), phoneme key sequence, timings from .polar
"""
import json
import os
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

import torch
from transformers import Wav2Vec2Model

# --- Kit portability: root of the experiment kit (parent of repro/) ------
KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
BASE = os.path.join(KIT_ROOT, "work", "base")
FEAT = os.path.join(KIT_ROOT, "work", "features")
MODELS = [
    ("w2v2_base960h", "facebook/wav2vec2-base-960h"),
    ("w2v2_base", "facebook/wav2vec2-base"),
]
SR_TGT = 16000


def load_wav16(path):
    sr, x = wavfile.read(path)
    assert sr == 44100, f"unexpected sample rate {sr}"
    x = x.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = resample_poly(x, 160, 441).astype(np.float32)
    # per-utterance normalization (matches Wav2Vec2FeatureProcessor do_normalize)
    y = (y - y.mean()) / (y.std() + 1e-7)
    return y


def main():
    with open(os.path.join(BASE, "base_manifest.json")) as f:
        manifest = json.load(f)

    os.makedirs(FEAT, exist_ok=True)
    for tag, mid in MODELS:
        outdir = os.path.join(FEAT, tag)
        os.makedirs(outdir, exist_ok=True)
        done = {fn[:-4] for fn in os.listdir(outdir) if fn.endswith(".npz")}
        todo = [it for it in manifest["items"] if it["label"] not in done]
        if not todo:
            print(f"[{tag}] all {len(done)} items already extracted")
            continue
        model = Wav2Vec2Model.from_pretrained(mid)
        model.eval()
        t_start = time.time()
        for i, it in enumerate(todo):
            y = load_wav16(os.path.join(BASE, it["wav"]))
            with torch.no_grad():
                out = model(
                    torch.from_numpy(y).unsqueeze(0),
                    output_hidden_states=True,
                )
            hs = np.stack([h[0].numpy() for h in out.hidden_states], axis=0)  # (L, T, D)
            T = hs.shape[1]
            t_centers = (np.arange(T) + 0.5) * 0.02
            # polar-derived windows in seconds
            pol = json.load(open(os.path.join(BASE, it["polar"])))
            ph = pol["phonemes"]
            idx = it["target_idx"]
            windows = {
                "target": (ph[idx]["t0"] / 100.0, ph[idx]["t1"] / 100.0),
                "pre": (ph[idx - 1]["t0"] / 100.0, ph[idx - 1]["t1"] / 100.0),
                "post": (ph[idx + 1]["t0"] / 100.0, ph[idx + 1]["t1"] / 100.0)
                        if idx + 1 < len(ph) else None,
                "utterance": (0.0, pol["duration_s"]),
            }
            np.savez_compressed(
                os.path.join(outdir, it["label"] + ".npz"),
                hs=hs.astype(np.float32),
                t_centers=t_centers.astype(np.float32),
                windows=json.dumps(windows),
                keys=json.dumps(it["keys"]),
                target_idx=it["target_idx"],
                label=it["label"],
            )
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"[{tag}] {i+1}/{len(todo)} extracted ({time.time()-t_start:.0f}s)")
        del model
    print("done")


if __name__ == "__main__":
    main()

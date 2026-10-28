#!/usr/bin/env python3
"""Synthetic-side CTC transcription measure (the missing cell of the
natural-vs-synthetic crossing): transcribe the lambda-continuum
sentences with wav2vec2-base-960h and score whether word 1 is written
with the underlying final /n/ (compensated / canonical) or with the
assimilated <m> (surface).

Frames: P1 clean pan, P2 mean boat, P4 thin packet (licensed)
        NV1 clean tan, NV3 clean can, NV4 mean goat (unlicensed)
Lambda: 0 (canonical control), 0.625, 0.75, 1.0
Outputs -> <KIT_ROOT>/work/lexamb/synthetic_ctc.json  (COVTL_KIT_ROOT
relocates the kit; original run: /tmp/my-project paths)
"""
import json
import math
import os
import wave

import numpy as np

# --- Kit portability: root of the experiment kit (parent of repro/) ------
KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
os.environ.setdefault("HF_HOME", os.path.join(KIT_ROOT, "hf_cache"))

WAV_DIRS = {
    "0.625": os.path.join(KIT_ROOT, "work", "lambda0625", "wavs"),
    "0.75": os.path.join(KIT_ROOT, "work", "lambda075", "wavs"),
    "0": os.path.join(KIT_ROOT, "work", "lambda0625", "wavs"),  # identical bytes
    "1": os.path.join(KIT_ROOT, "work", "lambda0625", "wavs"),
}
FRAMES = {
    "P1": ("clean", "cleam", "licensed"),
    "P2": ("mean", "meam", "licensed"),
    "P4": ("thin", "thim", "licensed"),
    "NV1": ("clean", "cleam", "unlicensed"),
    "NV3": ("clean", "cleam", "unlicensed"),
    "NV4": ("mean", "meam", "unlicensed"),
}
LAMBDAS = ["0", "0.625", "0.75", "1"]
OUT = os.path.join(KIT_ROOT, "work", "lexamb", "synthetic_ctc.json")


def load_wav_16k(path):
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2") \
            .astype(np.float32) / 32768.0
    if sr != 16000:
        g = math.gcd(sr, 16000)
        from scipy.signal import resample_poly
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def main():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    results = {}
    for fr, (und, surf, lic) in FRAMES.items():
        for lam in LAMBDAS:
            path = os.path.join(WAV_DIRS[lam], f"{fr}_l{lam}.wav")
            y = load_wav_16k(path)
            y = (y - y.mean()) / (y.std() + 1e-7)
            with torch.no_grad():
                logits = model(torch.from_numpy(y).unsqueeze(0)).logits
            text = proc.batch_decode(logits.argmax(-1))[0].strip().lower()
            # verdict on word 1: final n (underlying) vs m (surface)
            words = text.replace("|", " ").split()
            w1 = words[0] if words else ""
            if w1.endswith("n"):
                comp = True
            elif w1.endswith("m"):
                comp = False
            else:
                comp = None
            results[f"{fr}_l{lam}"] = dict(
                frame=fr, lam=float(lam), licensing=lic, text=text,
                word1=w1, comp=comp)
            print(f"{fr}_l{lam:5s}: '{text}'  word1={w1!r} comp={comp}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print("Ecrit:", OUT)


if __name__ == "__main__":
    main()

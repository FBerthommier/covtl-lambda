# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Fetch, verify and install the natural stimuli of Pouw et al. (2024).

Why this script exists
----------------------
The 86 natural sentences (the authors' re-recording of the Gaskell &
Marslen-Wilson 2001 lexical-ambiguity materials, two passes of 43
viable-context items) are the only input of Section 7 of the companion
note that this kit does NOT redistribute: the authors' public release
carries no license text, so redistribution is not authorized by
default (see NOTICE.md, Tier 3). The archive is instead fetched from
the authors' own channel and pinned byte-for-byte:

  source   github.com/CharlottePouw/assimilation (Google-Drive share)
  archive  lexamb-stimuli.zip  (29,405,531 bytes)
  sha256   81055a7fb2a1d88277384ff1e53cb85adb1f147f4703278178700d2a2ee0ca0e
  content  lexamb-stimuli/NN_{1,2}.wav  x 86  (16 kHz, mono)

Usage
-----
  python repro/fetch_lexamb.py                # download + verify + extract
  python repro/fetch_lexamb.py --zip PATH     # use a local copy of the zip
  python repro/fetch_lexamb.py --check-only   # verify an existing wavs/ dir

After a successful run
----------------------
  python repro/lexamb_cross_experiment.py a   # features  (work/lexamb/feats)
  python repro/lexamb_cross_experiment.py b   # probes
  python repro/lexamb_cross_experiment.py c   # rates -> work/lexamb/cross_stats.json

Step (a) needs the wav2vec2 checkpoints (downloaded automatically on
first use) and the synthetic training features of the kit (run
`repro/extract_features.py` first if you skipped Quickstart step 2).

Citation for the stimuli: Pouw, C., de Heer Kloots, M., Alishahi, A.,
& Zuidema, W. (2024). Perception of phonological assimilation by
neural speech recognition models. Computational Linguistics, 50(4),
1557-1585.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

# --- pinned provenance -------------------------------------------------------
GDRIVE_ID = "15SMDf2MXthPZhDvWgfzh84-n8JLCPYSN"
URL = ("https://drive.usercontent.google.com/download"
       f"?id={GDRIVE_ID}&export=download&confirm=t")
ARCHIVE_SHA256 = ("81055a7fb2a1d88277384ff1e53cb85adb1f147f470327817870"
                  "0d2a2ee0ca0e")
ARCHIVE_BYTES = 29405531

KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
WAVS_DIR = os.path.join(KIT_ROOT, "work", "lexamb", "wavs")
CHECKSUMS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "lexamb_checksums.json")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def download(dst):
    if os.path.exists(dst):
        print(f"[fetch] found existing {dst}, skipping download")
        return
    print(f"[fetch] downloading {ARCHIVE_BYTES/1e6:.1f} MB from the "
          f"authors' Drive share")
    print(f"[fetch] url: {URL}")
    tmp = dst + ".part"
    try:
        with urllib.request.urlopen(urllib.request.Request(
                URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) \
                as r, open(tmp, "wb") as f:
            done = 0
            while True:
                block = r.read(1 << 18)
                if not block:
                    break
                f.write(block)
                done += len(block)
                print(f"\r[fetch] {done/1e6:6.1f} MB", end="", flush=True)
        print()
    except Exception as e:  # noqa: BLE001 - report and exit cleanly
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(f"[fetch] download failed: {e}\n"
                 f"        obtain lexamb-stimuli.zip from "
                 f"github.com/CharlottePouw/assimilation and rerun with "
                 f"--zip PATH")
    os.replace(tmp, dst)


def verify_archive(path):
    digest, n = sha256_file(path)
    ok_size = (n == ARCHIVE_BYTES)
    ok_hash = (digest == ARCHIVE_SHA256)
    print(f"[verify] archive size {n} B ({'OK' if ok_size else 'MISMATCH'})")
    print(f"[verify] archive sha256 {digest[:16]}… "
          f"({'OK' if ok_hash else 'MISMATCH'})")
    if not (ok_size and ok_hash):
        sys.exit("[verify] archive does not match the pinned provenance — "
                 "refusing to extract")


def extract(path):
    per_file = json.load(open(CHECKSUMS))["wavs"]
    os.makedirs(WAVS_DIR, exist_ok=True)
    print(f"[extract] -> {WAVS_DIR}")
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".wav")]
        if len(members) != len(per_file):
            sys.exit(f"[extract] expected {len(per_file)} wavs, "
                     f"archive holds {len(members)}")
        for m in members:
            name = os.path.basename(m)
            z.extract(m, WAVS_DIR)
            src = os.path.join(WAVS_DIR, m)
            dst = os.path.join(WAVS_DIR, name)
            if src != dst:
                os.replace(src, dst)          # flatten lexamb-stimuli/
            d, _ = sha256_file(dst)
            if per_file.get(name) != d:
                sys.exit(f"[extract] {name}: hash mismatch after extraction")
    print(f"[extract] {len(per_file)} wavs verified against "
          f"lexamb_checksums.json")


def check_only():
    per_file = json.load(open(CHECKSUMS))["wavs"]
    missing, bad = [], []
    for name, want in sorted(per_file.items()):
        p = os.path.join(WAVS_DIR, name)
        if not os.path.exists(p):
            missing.append(name)
        elif sha256_file(p)[0] != want:
            bad.append(name)
    if missing or bad:
        sys.exit(f"[check] {len(missing)} missing, {len(bad)} corrupted "
                 f"(first missing: {missing[:2]})")
    print(f"[check] {len(per_file)} wavs present and intact in {WAVS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--zip", metavar="PATH",
                    help="verify+extract a local lexamb-stimuli.zip")
    ap.add_argument("--check-only", action="store_true",
                    help="only verify work/lexamb/wavs/")
    args = ap.parse_args()
    if args.check_only:
        check_only()
    else:
        zpath = args.zip or os.path.join(
            os.path.dirname(WAVS_DIR), "lexamb-stimuli.zip")
        download(zpath)
        verify_archive(zpath)
        extract(zpath)
        check_only()
        print("[done] rerun repro/lexamb_cross_experiment.py a b c")

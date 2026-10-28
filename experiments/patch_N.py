# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Minimal engine patch: give /N/ (velar nasal) a closure gesture.

Follows the engine's own conventions:
- 2026-09-02 nasal fix: a nasal = complete oral closure + VO open
- place/target = g_vel (rho 1.45, 60 deg, selector [JA TCX TCY TBX TBY])

Preserves the CRLF line endings of constants.py.
"""
import io
import os

PATH = os.environ.get(
    "COVTL_CONSTANTS",
    os.path.join(os.environ.get("COVTL_PIPELINE_ROOT", "covtl-pipeline"),
                 "vtl_synth", "core", "constants.py"))

OLD = "    'N':    (1.20, np.radians(60),  []),"
NEW = (
    "    'N':    (1.45, np.pi / 3,       ['JA', 'TCX', 'TCY', 'TBX', 'TBY']),\n"
    "    # /N/ (velar nasal): closure gesture added (2026-10-18, assimilation\n"
    "    # study patch). Same recipe as the 2026-09-02 /m,n/ fix - a nasal is a\n"
    "    # complete oral closure at its place + velum open (VO, handled by the\n"
    "    # nasal tag in tag_system.py). Place/target = g_vel (rho 1.45, 60 deg,\n"
    "    # selector [JA TCX TCY TBX TBY]) so [N] closes exactly like the velar\n"
    "    # stop of the same series. Verified: TBX/TCX reach the /k/ closure\n"
    "    # values during /N/ (before the patch the tongue stayed on the vocalic\n"
    "    # branch, i.e. [N] was realized as a nasalized vowel)."
)

with io.open(PATH, "r", encoding="utf-8", newline="") as f:
    text = f.read()

assert text.count(OLD) == 1, f"expected 1 occurrence, got {text.count(OLD)}"
# OLD line ends with \r\n in the file; replace the line content only.
text = text.replace(OLD + "\r\n", NEW + "\r\n")

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(text)

print("patched OK")

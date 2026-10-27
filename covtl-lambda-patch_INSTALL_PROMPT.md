## Mission

Install **covtl-lambda-patch v1.2.0** — the λ-morph option for `vtl-synth`
(articulatory assimilation continua on tautosyllabic CC clusters,
**C1C2 → C1'C2**: λ moves the polar target of the nasal C1 — θ always,
ρ with `--target-rho` — toward the target of C2; a two-endpoint
tract-blend mode remains available) — into a **fresh clone of
covtl-pipeline**, verify it end-to-end against the pinned expected
outputs, and leave the clone in a clean, revertible state. Do not
improvise: each step below has an expected result; if reality
diverges, stop and report instead of working around it.

**Notation (authoritative).** `.` = intra-phrase syllable separator
(no pause, C.C split — a different gesture structure); **no separator
= CC cluster**; space = 200 ms; `|` = 320 ms. The λ variation is
defined on the CC cluster only — endpoint phrases are dotless
(`klinpan → klimpan`, never `klin.pan`).

## Inputs

- `covtl-lambda-patch_v1.2.0.zip` (24 entries, ~1.1 MB) — its SHA-256 is
  published alongside the zip (release note / `SHA256SUMS.txt`); verify
  with the command below.

- Network access (git clone + pip). ~10 minutes.
- Platform: Linux and Windows (both verified end-to-end). Python
  **3.10–3.12** recommended (3.12.10 verified on Windows, 3.12.15 on
  Linux). `git`, `pip`, `unzip` (Windows: `tar -xf` or `Expand-Archive`).

## Procedure

### 0. Verify the patch archive

```bash
sha256sum covtl-lambda-patch_v1.2.0.zip
# Windows: certutil -hashfile covtl-lambda-patch_v1.2.0.zip SHA256
```

MUST equal the SHA-256 published alongside the zip. If not: stop and
report; do not unzip.### 1. Fresh clone, pinned commit

```bash
git clone https://github.com/FBerthommier/covtl-pipeline.git
cd covtl-pipeline
git checkout 1d15ed7        # v1.1.0 — the verified commit
```

Later engine versions usually work too (the apply script is anchor-guarded
and aborts cleanly on any layout change; the current upstream `main`
with the merged polar-arcplot work is also verified), but `1d15ed7` is
the commit this prompt's original expected outputs were pinned to.

### 2. Virtual environment + EDITABLE install (required)

```bash
python3 -m venv ../venv-lambda                  # Windows: py -m venv ..\venv-lambda
source ../venv-lambda/bin/activate              # Windows: ..\venv-lambda\Scripts\activate
python -m pip install -U pip
pip install -e ".[plot,dev]"
```

**Why editable is mandatory:** the patch modifies the *source tree* (three
new modules + a guarded insertion in `vtl_synth/cli/main.py`). Only
`pip install -e` makes those changes live in the `vtl-synth` entry point.
A plain `pip install .` freezes an unpatched copy in site-packages and
`vtl-synth morph` then fails with `invalid choice: 'morph'`.

Dependencies (numpy, scipy, vocaltractlab-cython, Pillow, imageio-ffmpeg,
cmudict) come from pip. Extras: `dev` = pytest, `plot` = matplotlib (needed
only for `--plot` figures and 5 engine figure tests).

Python version caveats:

- **Python 3.9**: if pip tries to *build* vocaltractlab-cython from
  source, pre-install `pip install vocaltractlab-cython==0.0.13`
  (engine INSTALLATION.md, step 4).
- **Python 3.13/3.14**: vocaltractlab-cython has no wheel — use
  Python 3.10–3.12 (a per-user or portable 3.12 install works; no need
  to touch the system Python).

### 3. Unzip the patch beside the clone

```bash
cd ..
unzip covtl-lambda-patch_v1.2.0.zip             # -> ./covtl-lambda-patch/
cd covtl-pipeline
```

### 4. Apply

```bash
python ../covtl-lambda-patch/patch/apply_patch.py --pipeline-root .
```

Expected — exit code **0** and exactly this shape of output:

```text
[apply] pipeline root : /.../covtl-pipeline
[apply] + vtl_synth/core/morphing.py
[apply] + vtl_synth/cli/morph_cmd.py
[apply] + tests/test_morphing.py
[apply] ~ vtl_synth/cli/main.py (guarded insertion before the global-options block; EOL=CRLF)
[apply] backup      : vtl_synth/cli/main.py.lambda-morph.bak
[apply] verify      : morph subcommand OK
[apply] done -- covtl-lambda-patch v1.2.0 (2026-10-26)
```

(v1.2.0 is the semantic fix: the λ variation acts on the tautosyllabic
CC cluster and its polar targets — new `--targets` default mode, θ
mandatory / ρ optional, C.C dotted phrases rejected; v1.1.0 added the
C1-confine of the blend mode; v1.0.1 added this install prompt.)

Exit codes: `0` applied · `2` already applied (idempotent; `--force`
re-copies the modules) · `1` error (anchor not found exactly once, or no
pipeline found). On `1`: STOP and report — never hand-edit `main.py`.

### 5. Patch test suite

```bash
python -m pytest tests/test_morphing.py -q
```

Expected: **`53 passed`** in ~9 s (VTL-gated tests included; they *skip*
rather than fail if vocaltractlab-cython is absent).

### 6. CLI smoke test (target semantic — the default)

```bash
vtl-synth morph --phonetic --targets klinpan -o morph_out --label P1
```

Expected — exit 0, and after the engine's speaker banner, exactly:

```text
[morph] mode      : targets (C1C2 -> C1'C2 on the tautosyllabic cluster -- DEFAULT semantic, v1.2.0)
[morph] label     : P1
[morph] frames    : 584 @ 400 Hz
[morph] cluster   : C1 n (rho 1.150, theta 4.7124 rad) -> C2 p (rho 1.060, theta 0.8727 rad); dtheta +140.00 deg, theta only
[morph] C1 window : frames 232-271 (phoneme span); C1-incident arcs frames 232-311
[morph] footprint : frames 229-307 (measured, all lambdas vs canon)
[morph] check     : lam0_bitwise_canon = True
[morph] check     : glottis_bitwise_constant = True
[morph] check     : lam1_theta_equals_c2 = True
[morph] check     : lam1_theta_equiv_c2_mod_2pi = True
[morph] check     : outside_c1_arcs_bitwise_canon = True
[morph] lambda=0  theta_C1=+4.712389 rad  morph_out/P1_l0.tract  + morph_out/P1_l0.wav
    ... one row per lambda (default grid: 9 uniform steps, 0 to 1)
[morph] lambda=1  theta_C1=+0.872665 rad  morph_out/P1_l1.tract  + morph_out/P1_l1.wav
[morph] meta      : morph_out/morph_meta.json
```

### 7. (optional) Reference example — the assimilation experiment

```bash
python ../covtl-lambda-patch/examples/lambda_continuum.py --out lambda_out --no-audio
```

Expected (target semantic, cluster pairs): P1 `klinpan` (C2 /p/,
licensed) with C1 window frames 232–271 and footprint 229–307; NV1
`klintan` (C2 /t/, unlicensed) with the same C1 window; articulator
endpoints over λ = 0 → 1 (9 steps, read at t = 0.63 s):

```text
P1:  LP 0.177 (const)  LD 0.645 (const)  JA -2.321 -> -3.327  VO 0.886 (constant)
NV1: LP 0.190 (const)  LD 0.625 (const)  JA -2.321 -> -2.196  VO 0.886 (constant)
```

(the place change is carried by the jaw and the tongue at mid-arc;
the lips follow the vowel background — cf. the manual §6). Writes
`lambda_out/articulators.json` per pair.

### 8. (optional) Engine non-regression

```bash
python -m pytest -q
```

Expected: **`1 failed, 205 passed`** (206 collected = 153 engine + 53
patch). The single failure MUST be
`tests/test_speaker_registry.py::TestRegistry::test_hashes_match_registry`
— a **pre-existing upstream issue**: it fails identically on the *unpatched*
engine (registry `speaker_sha256` drift for the non-default speakers
s1/s2/m01/w02, plus — on Linux — LF checkout of `*_constants.py` vs
CRLF-recorded hashes). The default speaker **jd3**, the only one the
λ-morph workflow uses, is fully coherent. Without matplotlib installed,
5 additional failures appear in `test_tract_figure.py` (missing optional
extra, unrelated to the patch).

## Known benign behaviors — do not "fix" them

1. **Startup banner warning** (appears at every `vtl-synth` invocation):

   ```text
   Speaker actif : jd3 (JD3 reference...)  constants=...  f0_default=102.216 Hz
                   !! INCOHERENCE constants : le constants.py installe (...) !=
                   source du speaker actif (...) — relancer install_speaker.py
   ```

   Pre-existing engine bookkeeping warning, present in both reference
   validation environments (Linux and Windows). Synthesis outputs are
   unaffected (bitwise-verified against the reference experiment). Do
   **not** run `install_speaker.py`: it is Windows-oriented (hardcoded
   `P:\` paths) and rewrites the vocaltractlab-cython wheel's shared
   resources.

2. **`git status` right after apply** shows exactly:

   ```text
    M vtl_synth/cli/main.py
   ?? tests/test_morphing.py
   ?? vtl_synth/cli/morph_cmd.py
   ?? vtl_synth/core/morphing.py
   ```

   The `main.py.lambda-morph.bak` backup exists but is gitignored
   (`*.bak`). Output dirs (`morph_out/`, `lambda_out/`) are untracked.
   Any other modified file = unexpected: report it.

## Acceptance criteria — report each one explicitly

- [ ] zip SHA-256 matches the published hash
- [ ] apply: exit 0, `morph subcommand OK`
- [ ] patch tests: 53/53 passed
- [ ] smoke test: 584 frames @ 400 Hz, cluster n→p (ρ 1.150, θ
      4.7124 rad → ρ 1.060, θ 0.8727 rad, Δθ +140°), C1 window frames
      232–271, all 5 target checks True, 9 λ rows + `morph_meta.json`
      with per-item `theta_c1`
- [ ] dotted input check: `vtl-synth morph --phonetic --targets
      klin.pan ...` exits 1 with the "tautosyllabic" error (C.C splits
      are out of scope)
- [ ] (if run) example: P1/NV1 target continua, values as above
- [ ] (if run) engine suite: 205 passed + exactly the one documented
      pre-existing failure
- [ ] `git status` matches the canonical list above

## Revert (restore the pristine clone)

```bash
python ../covtl-lambda-patch/patch/revert_patch.py --pipeline-root .
```

Expected: exit 0; `main.py` restored from the backup (byte-identical to
its pre-patch state); the three added files removed; `git status` clean
except untracked output dirs. Other engine patches (e.g. a `/N/`
constants patch), if present, are preserved.

## Troubleshooting

| Symptom | Cause / remedy |
| --- | --- |
| `vtl-synth: command not found` | Venv not activated. Fallback that always works from the clone root: `python -m vtl_synth.cli.main morph --phonetic --targets klinpan -o morph_out --label P1` (byte-identical outputs). |
| `invalid choice: 'morph'` | Install was not editable. Re-run `pip install -e .` (see step 2). |
| apply exit 1, `expected exactly 1 anchor` | Engine layout changed (commit newer than verified). Pin `1d15ed7` or report the layout. |
| apply exit 2 | Patch already applied — normal; proceed to tests. |
| `ERROR: ... tautosyllabic CC clusters only` | The phrase contains a syllable dot (`klin.pan`): use the dotless cluster form (`klinpan`) — C.C splits are out of scope of the λ variation. |
| pip builds vocaltractlab-cython from source (Py 3.9) | `pip install vocaltractlab-cython==0.0.13` first, then retry. |
| pip builds vocaltractlab-cython from source (Py 3.13/3.14) | No wheel exists for these versions — switch the venv to Python 3.10–3.12. |

## Hard constraints

- The only sanctioned mutation path is `apply_patch.py` / `revert_patch.py`.
  Never hand-edit `vtl_synth/cli/main.py` or any engine file.
- Do not commit or redistribute patch + engine together without honoring
  GPL-3.0-or-later (see the patch repo's `LICENSE` and `NOTICE.md`).
- If any expected output differs, stop and report actual vs expected
  values, OS, Python version, and `git log --oneline -1` of the clone —
  do not improvise a workaround.

# Changelog

All notable changes to this package are documented here.
Format: Keep a Changelog — https://keepachangelog.com/

## [1.2.0] — 2026-10-26

### Fixed — the semantic of the lambda variation (headline change)

- **Cluster only, target level.** The lambda variation applies to
  tautosyllabic CC clusters — C1 = the nasal coda, C2 = the following
  onset of the same cluster — and its effect is defined on the
  **polar targets** of the cluster branch: λ moves the θ of C1 toward
  the θ of C2 (optionally its ρ), producing strictly
  **C1C2 → C1'C2** (`klinpan → klimpan`). At λ = 0 the cluster target
  is strictly C1C2; at λ = 1 the C1 polar target has reached the C2
  target coordinates — the place-assimilated cluster (the C1 gesture
  now points at the C2 target; C2 itself never moves). This
  target-level definition is the ground truth; the tract-parameter
  blend of v1.0/v1.1 is now documented as a motor-level realization,
  kept as an option.
- **C.C splits are out of scope** (stated everywhere, enforced in
  code): `.` = intra-phrase syllable separator — no pause, it splits
  C.C into a different gesture structure (the engine's cluster target
  rules do not apply). Dotted forms (`klin.pan`, `klim.pan`,
  `klin.tan`, `min.bot`, ...) were wrongly used as endpoints,
  denominations and examples in the v1.0/v1.1 wording, examples and
  demos; all of them are replaced by the cluster forms
  (`klinpan/klimpan`, `klintan/klimtan`, `minbot/mimbot`,
  `Tinpakit/Timpakit`, `klinkan/klimkan`, `mingot/mimgot`). The
  target morph REJECTS dotted phrases with an explanatory error.

### Added — the target-level morph (`--targets`, the DEFAULT semantic)

- `vtl_synth/core/morphing.py`: `build_target_continuum` /
  `morph_targets` + planning helpers (`plan_target_cluster`,
  `find_nasal_c1_clusters`, `interp_target_theta`,
  `interp_target_rho`, `wrap_angle_delta`, `TargetClusterPlan`).
  Given ONE canonical cluster phrase, each λ re-runs the engine's own
  syllabic trajectory builder with the C1 node retargeted (a runtime
  override around our own call, always restored — no engine file is
  modified), then re-synthesizes. Locality is structural: only the
  C1-incident sub-arcs depend on the C1 target.
- CLI: `vtl-synth morph --phonetic --targets klinpan -o out` (single
  cluster phrase; `--target-rho` adds the ρ channel,
  `--cluster N` selects among several nasal-initial clusters).
  `morph_meta.json` records, per item, the C1/C2 polar targets and
  the λ-interpolated C1 target, plus the C1 phoneme window and
  C1-incident-arc window in 400 Hz frames.
- New recorded guarantees (asserted per run):
  `lam0_bitwise_canon` (λ = 0 reproduces `vtl-synth run <phrase>`
  bitwise), `lam1_theta_equals_c2` (at λ = 1 θ_C1 = θ_C2 **exactly**;
  `lam1_rho_equals_c2` with `--target-rho`), `glottis_bitwise_constant`
  (voicing/F0/nasality never move with λ), and
  `outside_c1_arcs_bitwise_canon` (C2 and context strictly invariant
  at every λ).
- 17 new tests (target arithmetic, nasal-cluster detection on real
  phrases, dotted-phrase rejection, planning windows, VTL-gated
  continuum guarantees, CLI wiring and end-to-end): suite now
  **53 passed**.

### Fixed — examples and demos

- `examples/lambda_continuum.py` and `examples/morph_basic.py`:
  cluster endpoints only (`klinpan`/`klintan`), target semantic by
  default; the blend remains available explicitly.
- HTML demos regenerated from cluster-form continua with the target
  semantic: `lambda_demo.html` (K3, `klinkan`) and the six Pouw
  pages (`klinpan/klimpan` P1, `minbot/mimbot` P2,
  `Tinpakit/Timpakit` P4, `klintan/klimtan` NV1,
  `klinkan/klimkan` NV3, `mingot/mimgot` NV4) — dotless
  denominations, C1 windows re-derived from the endpoint `.polar`
  phoneme diff (they land on the CORRECTIONS v2.1 references:
  frames 232–272 for the P1 family, 192–232 for the P2/P4/NV4
  family).

### Notes

- The kit's dotted SAMPA (`klin.pan` for "clean pan") is NOT the
  dotless form: dropping the dot of `klin.pan` gives `klinpan`
  (k-l-i-n-p-a-n). Note that engine SAMPA has no `c` — a phrase like
  `clinpan` loses the `c` silently (the SAMPA guard warns; verify the
  `.polar` phoneme inventory after any phrase change).
- Version bumped to v1.2.0 everywhere the patch self-identifies
  (`MORPH_PATCH_ID`, apply/revert scripts, docs, demos).

## [1.1.0] — 2026-10-22

### Fixed — causal locality of the C1C2 design

- **C1-confined morph** (the reason for this release): the v1.0.0
  blend spanned the whole endpoint-divergence window, so anticipatory
  frames (on the preceding vowel) and transition/release frames
  belonging to the following consonant C2 were also transformed by
  lambda — violating the C1C2 contract (C2 position and timing
  strictly invariant; only the C1 control node drifts).  New
  `confine` support restricts the blend to the C1 gesture window:
  - `build_continuum(..., confine=(first, last), ramp=10)` and
    `morph_pair(..., confine=, ramp=)`;
  - CLI `--confine T1:T2` (seconds) / `--confine-frames F1:F2` and
    `--ramp N` (raised-cosine edge ramps, default 10 frames = 25 ms,
    keeping the output trajectory continuous at the window boundary);
  - new recorded guarantees: `confine_lam1_equals_assim_inside`
    (bitwise assimilated inside the window plateau) and
    `outside_confine_equals_canon` (bitwise canonical outside the
    window for every lambda — C2 invariance, asserted per item);
  - `morph_meta.json` records the confine window and ramp;
  - unconfined calls keep the legacy behavior bit-for-bit, and the
    CLI prints a notice recommending `--confine` for C1C2
    experiments.
- 7 new tests (weight shape/bounds/rejection, context-canonical
  guarantee, hard window at ramp 0, legacy behavior, meta record):
  suite now **36 passed**.
- Apply/revert markers made version-agnostic (v1.0.0 insertions are
  still detected and reverted; an upgrade apply never inserts the
  main.py block twice).
- Notation warning added to the docs: engine SAMPA has no `c` —
  an invalid symbol in a phrase is silently dropped by the engine
  (`klincan` synthesizes as `klinan`); the velar context is `kan`
  (with `k`).  Verify the phoneme inventory of endpoint phrases
  (e.g. via the `.polar` phoneme timing) before morphing.

## [1.0.1] — 2026-10-22

### Added

- **`covtl-lambda-patch_INSTALL_PROMPT.md`** — a self-contained
  installation prompt (English US): mission, pinned inputs, step-by-step
  procedure with expected outputs, acceptance criteria, revert,
  troubleshooting and hard constraints. Every expected value was
  re-verified end-to-end on fresh clones (Linux and Windows, Python
  3.12): apply → 29/29 patch tests → smoke test with reference values →
  P1/NV1 example → engine suite `1 failed, 140 passed` (the one
  pre-existing upstream registry failure, documented) → revert to a
  clean tree.

### Fixed

- No patch-payload change; the v1.0.0 payload (patch modules, tests,
  examples, docs) is unchanged and remains labeled v1.0.0 in the apply
  script's `done` line.

## [1.0.0] — 2026-10-22

### Added

- **`vtl-synth morph` subcommand** (after applying the patch): lambda
  articulatory morphing between two frame-aligned utterances —
  `data(λ) = (1−λ)·A + λ·B` frame by frame on the 400 Hz VTL tract
  sequence (19 tract + 11 glottis parameters). File mode (two
  `.tract` inputs) and phrase mode (two SAMPA inputs, endpoints
  synthesized with identical engine settings). Options:
  `--lambdas LIST | --n-steps N` (default 9 uniform steps),
  `--glottis {common, interpolate}` (default `common`: identical
  glottis required, pure tract morph), `--tol`, `--no-audio`,
  `--label`, plus the standard engine options for phrase mode.
- **`vtl_synth/core/morphing.py`** — the engine module: grid parsing
  and validation, `interpolate_tract` (bitwise-exact endpoints),
  `check_alignment` (frame-alignment contract, divergence-window
  report, glottis policy), `build_continuum` (endpoint/glottis
  guarantee checks asserted), `read_tract_sequence` /
  `write_morph_tract` (canonical VTL format), `window_state`
  (articulatory ground truth at a time window), and the
  `morph_pair` driver writing `<label>_l<λ>.tract`/`.wav` +
  `morph_meta.json` (source SHA-256 hashes, alignment report,
  checks, item list).
- **`patch/apply_patch.py` / `patch/revert_patch.py`** — guarded,
  EOL-safe, idempotent, reversible installation into a
  covtl-pipeline checkout: three files copied + one 4-line
  assert-guarded insertion in `vtl_synth/cli/main.py` (backup
  written; revert restores the original byte-identically or removes
  the block by markers). In the spirit of the pipeline's own
  patches (cf. the 2026-10-18 `/N/` closure patch).
- **`tests/test_morphing.py`** — 29 tests in two layers:
  pure-numpy (identities, grids, alignment, glottis policies, I/O
  round trip, driver, parser wiring) and VTL-gated (CLI end to
  end, endpoint WAV determinism pinned byte-for-byte).
- **`examples/lambda_continuum.py`** — the reference assimilation
  continuum experiment (P1 licensed / NV1 unlicensed, nine-step
  grid, articulator table at the 0.58–0.68 s target window,
  optional figure); outputs verified identical to the source kit
  (`nasal_synthesis` repro/lambda_experiment.py, stage A).
  **`examples/morph_basic.py`** — minimal API demo.
- **`docs/manual.tex` / `docs/manual.pdf`** — reference manual
  (English US, 12 pages): model and guarantees, patch procedure,
  CLI and Python API, worked experiment with figures, test
  inventory, limitations and extensions.

### Non-regression

- Additive only: no existing subcommand, default, file format or
  engine output changes; regression baselines of the host pipeline
  are unaffected.

[1.0.0]: https://example.com/covtl-lambda-patch/releases/tag/v1.0.0

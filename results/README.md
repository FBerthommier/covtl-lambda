# results/ — the Wav2Vec2 probing campaign (kit v3.0, verified)

This directory carries the results of the Pouw et al. (2024)
reproduction campaign — the part that **cannot be recomputed locally**
(it needs the wav2vec2 checkpoints) — received from the external
probing agent as `nasal_synthesis_kit_v3.0.zip` (2026-10-28) and
**verified against this repository before integration**, per
[ROADMAP](../docs/ROADMAP.md) W4/W5.

## What was verified (2026-10-28)

1. **Synthesis identity (W4)** — all **54/54** campaign `.tract` files
   (`work/lambda/tracts/`, 6 Pouw frames × 9 λ) are **numerically
   identical** (0 differing cells over 584×30 and equivalents) to the
   local reference continua synthesized with patch v1.2.0. The only
   byte-level difference is the EOL convention of the tract writer
   (kit LF vs local CRLF) — the known Windows contingency, content
   bitwise equal. Frame counts, headers and param names match.
2. **Guarantees (W4)** — the six per-frame `morph_meta_*.json` carry
   **5/5 true checks** each (`lam0_bitwise_canon`,
   `lam1_theta_equals_c2`, `glottis_bitwise_constant`,
   `outside_c1_arcs_bitwise_canon`, mod-2π equivalence) with per-λ
   `theta_c1` targets recorded; C1 windows match the repository
   reference (P1/NV1/NV3: 232–271; P2/P4/NV4: 192–231 @400 Hz).
3. **Provenance chain (W4)** — `synth_meta.json` pins the reference
   tree: covtl-pipeline `1d15ed7` + the corrections series + **this
   repository at commit `c13f4d0` (v1.2.0)**.
4. **Notation (W5)** — the kit documents passed the transition-notation
   sweep; the two dotted `\sampa{klin.pan}` occurrences found in the
   synthesis PDF source were **corrected to the canonical ladder**
   (`klinpan`, with the C.C-split and SAMPA-guard caveats) and the PDF
   rebuilt here (`nasal_place_assimilation_synthesis.pdf`, 20 pp).
   The spaced forms that remain in the kit README are the separator
   explainer (`klin kan` = pause), which is legitimate.

## Contents

| file | content |
|---|---|
| `CORRECTIONS_v3.0.md` | the revision report: v2.1 → v3.0 semantic change, before/after rates, why the numbers moved |
| `synth_meta.json` | campaign provenance: reference tree, speaker, per-file SHA-256 hashes |
| `morph_meta_P1…NV4.json` | per-frame target-mode records: polar targets, λ grid, windows, guarantee checks |
| `results.json` | per-layer probing outputs (CNN, T1–T12; verdicts with probabilities) |
| `note_stats.json` | three-condition statistics (compensation = probe verdict = underlying /n/) |
| `revision_comparison_v30.json` | machine-readable v2.1→v3.0 comparison (asymmetries, verdict flips) |
| `campaign/*.json` | stage-1 probe instrument (LOSO), margins, switch points |
| `figures/` | the shipped figures (λ heatmaps, boundary curves, Pouw schema reproduction, lexamb crossing, PCA, classification map) |
| `nasal_place_assimilation_synthesis.pdf` / `.tex` | the 21-page didactic synthesis (notation-corrected body rebuilt and re-merged with the kit cover page; the `.tex` is the corrected body source) |
| `note_three_condition_statistics.pdf` | the companion statistics note |

The 54 `.tract`/`.wav` stimulus files are **not** redistributed here —
they are regenerable bit-exactly with the patch
(`vtl-synth morph --targets <phrase>`) and their integrity is pinned by
the SHA-256 hashes in `synth_meta.json`. The full kit (with the legacy
motor-blend tiers) remains the external artifact `nasal_synthesis_kit_v3.0.zip`.

## Synthesis chain and citations

Every stimulus in this campaign was produced by the chain documented
in [NOTICE](../NOTICE.md): the COVTL articulatory model
(F. Berthommier) computes the polar targets and syllabic trajectories
in covtl-pipeline, and **VocalTractLab 2.4** (P. Birkholz,
https://www.vocaltractlab.de) performs the synthesis. The didactic
synthesis PDF cites both (plus Pouw et al. 2024 and the COVTL
publications) in its references. **Scientific use of these results
must cite the COVTL publications, VocalTractLab (Birkholz), and the
Pouw et al. study whose protocol is reproduced.** No VocalTractLab
component (binaries, speaker definitions, API bindings) is included
in this repository.

## Headline scientific result (from CORRECTIONS_v3.0)

No positive licensed-over-unlicensed asymmetry appears at any of the
nine production degrees; the one significant contrast (λ = 1, 960h,
−23.1 points) is **negative** and carried by letter supervision, not
phonological licensing — the null-asymmetry conclusion of the
reproduction attempts survives and sharpens under the target-level λ
semantic.

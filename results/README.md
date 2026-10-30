# results/ — the Wav2Vec2 probing campaign (kit v3.1, verified)

This directory carries the results of the Pouw et al. (2024)
reproduction campaign — the part that **cannot be recomputed locally**
(it needs the wav2vec2 checkpoints) — received from the external
probing agent as `nasal_synthesis_kit_v3.1.zip` (2026-10-30, SHA-256
`13238d4f1f4fd07798dba65a7785c2b6cd46f18eecbda3de283815fbb48a05b`)
and **verified against this repository before integration**, per
[ROADMAP](../docs/ROADMAP.md) W6.

## The v3.1 condition (full CC selector)

Under λ variation the **whole CC selector** is taken into account, as
in the base model: λ moves the C1 polar target exactly as in v1.2.0
(θ toward θ_C2, ρ fixed) and the C2-only selector parameters — the
lips LD/LP of a labial C2 — join the C1 gesture with weight λ:

    P(λ) = (1−λ)·P[T_sel1] + λ·P[T_union]

For a C2 whose selector is a subset of sel(C1) (alveolar /t/, velar
/k/,/g/ after /n/) the union equals sel(C1): the NV continua are a
**structural no-op**, bitwise the v3.0 ones. The engine patch is
unchanged; the condition is the kit's additive module
`repro/morphing_fullsel.py` over the patched engine (see
[../experiments/](../experiments/README.md)).

## What was verified (2026-10-30)

1. **Kit integrity** — the 108 embedded SHA-256 anchors of
   `synth_meta.json` (`files`: 54 tracts + 54 wavs) all match the
   delivered files; the integrated copies are byte-identical to the
   kit.
2. **Bitwise tier anchors** — λ = 0 and the whole NV block are
   bitwise identical v3.1 ↔ v3.0 (780/780 probe cells checked
   locally); jaw/tongue rotation is bitwise identical to v3.0 at
   every degree.
3. **Guarantees** — each `morph_meta_*.json` carries the four
   run-time guarantees of the morph plus three full-selector checks
   (`fullsel_confined_to_c1_arcs`, `fullsel_c2_only_recruited`,
   `fullsel_lam1_c2_only_equals_union`), all true (54/54 items).
4. **Provenance chain** — `synth_meta.json` pins the reference tree:
   covtl-pipeline `1d15ed7` + covtl-corrections-patch v1.0.1 + **this
   repository at commit `c13f4d0` (v1.2.0)** + the kit extension
   `morphing_fullsel.py`.

## Contents

| file | content |
|---|---|
| `CORRECTIONS_v3.0.md` / `CORRECTIONS_v3.1.md` | the revision reports (v2.1→v3.0 and v3.0→v3.1): semantic change, before/after rates, why the numbers moved — both preserved |
| `synth_meta.json` | campaign provenance: condition v3.1, reference tree, speaker, per-file SHA-256 hashes |
| `morph_meta_P1…NV4.json` | per-frame target-mode records: polar targets, λ grid, windows, selectors, `fullsel`/`diff_extent_frames`, guarantee checks |
| `results.json` | per-layer probing outputs (CNN, T1–T12; verdicts with probabilities) |
| `note_stats.json` | three-condition statistics (compensation = probe verdict = underlying /n/) |
| `pouw075_stats.json` / `pouw0625_stats.json` | the λ = 0.75 / 0.625 production-block statistics (Pouw three-condition protocol) |
| `revision_comparison_v30.json` / `revision_comparison_v31.json` | machine-readable comparisons (v2.1→v3.0, v3.0→v3.1: bitwise anchors, rates, asymmetry tests, dose-response endpoints, T4 boundaries) |
| `campaign/` | the five probe instruments: `campaign_probing.json` (classification), `probe_train.json` (LOSO), `margins.json`, `switch_points.json`, `timecourses.npz` |
| `figures/` | the shipped figures (λ heatmaps, boundary curves, compensation rates, Pouw schema reproduction, `figN3_pouw_reproduction_lam075.png`, `figN4_pouw_reproduction_lam0625.png`, `figN5_production_dose_response.png`, lexamb crossing, PCA, classification map) |
| `nasal_place_assimilation_synthesis.pdf` / `.tex` | the 22-page didactic synthesis, v3.1 edition (body 21 pp + kit cover; copied from the kit, never rebuilt locally — the `.tex` mirrors `latex/main.tex`) |
| `note_three_condition_statistics.pdf` | the companion statistics note, v3.1 edition (25 pp) |
| `latex/` / `latex_note/` | the complete buildable LaTeX sources of the two PDFs (cover page, figures, `main.tex`/`main.pdf`) |

The 54 `.tract`/`.wav` stimulus files and the λ = 0.75 / 0.625
production-block stimuli are **not** redistributed here — their
integrity is pinned by the SHA-256 hashes in `synth_meta.json`, and
the chain is replayable from [../experiments/](../experiments/README.md).
The full kit (with the legacy tiers: `legacy_lipblocked` = v3.0,
`legacy_motorblend`, `legacy_unconfined`) remains the external
artifact `nasal_synthesis_kit_v3.1.zip`.

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

## Headline scientific results (from CORRECTIONS_v3.1)

- **The two-stage account survives the completed gesture**: the
  phonetic layers (T2–T4, LOSO 1.00) still read the underlying /n/ at
  every degree of every frame — now through a real labial closure at
  λ = 1. At the C1 target point of P1 the lip opening LD falls
  0.61 → 0.46 → 0.16 → **0.00 cm** at λ = 0 / 0.5 / 0.75 / 1 (the
  base model's own /p/ value, protrusion LP → 1.08 cm) — and no T4
  category boundary appears.
- **The λ = 1 licensed dip strengthens and becomes bilateral**: the
  inverted Viable−Unviable contrast excludes zero on both listeners
  (−20.5 points [−37.3, −2.4] on the CTC-tuned 960h; −17.9 points
  [−32.7, −5.3] on the base model), where v3.0 had it on the 960h
  listener only.
- **The late-layer labial reading is context-bound, not
  closure-bound**: P1 (*clean pan*) — the frame whose lips now close —
  loses its T10–T11 labial reading (P(m) at λ = 1: .44/.53 → .24/.09,
  13/13 /n/ verdicts), while P2/P4 keep theirs (P(m) = .65–.95).
- **The production-degree structure is stable**: the λ = 0.75 cells of
  the CTC-tuned listener do not move (79.5 / 89.7; control 64.1), the
  λ = 0 and NV anchors are bitwise unchanged, and the dose-response
  keeps its shape (`figN5_production_dose_response.png`).

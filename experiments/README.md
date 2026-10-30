# experiments/ — the probing-campaign scripts

Python drivers of the Wav2Vec2 reproduction campaign, received with
the verified results in [../results/](../results/README.md). These
scripts are **engine-side reproducible locally only up to synthesis**:
the probing stages need the wav2vec2 checkpoints and are run by the
external campaign (see ROADMAP W1/W3).

**Synthesis chain**: all audio produced by these scripts comes from
covtl-pipeline — the COVTL articulatory model (F. Berthommier)
driving **VocalTractLab 2.4** (P. Birkholz,
https://www.vocaltractlab.de). Any publication built on these
stimuli must cite the COVTL publications, VocalTractLab, and the
Pouw et al. (2024) protocol being reproduced; the natural stimuli of
`fetch_lexamb.py` remain the property of their authors (fetch +
verify + cite, never redistribute). See [NOTICE](../NOTICE.md).

## Layout

The scripts resolve inputs/outputs relative to the kit root (the
parent of this directory in the original kit layout: `work/base/`,
`work/lambda/`, …). To run them locally you need a patched
covtl-pipeline checkout (see the repository README, step 3) and, for
the probing stages, the optional extra:

```bash
pip install torch transformers   # probing extra — NOT a repo dependency
```

| script | stage |
|---|---|
| `build_base.py` / `sentence_base.py` | synthesize the probing sentence base with vtl-synth |
| `build_campaign.py` / `morphing_fullsel.py` / `probe_campaign.py` | build and run the 54-item λ campaign (6 frames × 9 λ) under the v3.1 full-CC-selector condition |
| `extract_features.py` / `run_probing.py` | wav2vec2 hidden states + linear probes (L0–L12) |
| `pouw075_stats.py` / `pouw0625_stats.py` + `_figure.py` | Pouw Figure-1 protocol at λ = 0.75 / 0.625 |
| `note_stats.py` / `note_figure.py` | three-condition anchor statistics and figure |
| `figures.py`, `lambda_figures.py`, `pouw_schema_figure.py` | publication figures |
| `make_results.py` | assemble the results blocks of the kit |
| `fetch_lexamb.py` + `lexamb_checksums.json` | natural-stimuli import (Pouw et al. 2024 release, checksum-verified, never redistributed) |
| `lexamb_cross_experiment.py` / `_figure.py` / `lexamb_synthetic_ctc.py` / `lexamb_fig2_digitize.py` | the natural-vs-synthetic crossing (Darcy et al. 2009 anchor is read from published numbers) |
| `patch_N.py` | velar-nasal engine fix used by the base |

## The v3.1 full-CC-selector condition

Under λ variation the **whole CC selector** is taken into account, as
in the base model: λ moves the C1 polar target exactly as in v1.2.0
(θ toward θ_C2, ρ fixed) and the C2-only selector parameters — the
lips LD/LP of a labial C2 — join the C1 gesture with weight λ:

    P(λ) = (1−λ)·P[T_sel1] + λ·P[T_union]

The recruitment is linear in λ, exact at the endpoints (bitwise), and
confined to the C1-incident sub-arcs.  For a C2 whose selector is a
subset of sel(C1) — alveolar /t/, velar /k/,/g/ after /n/ — the union
equals sel(C1): the continuum is a **structural no-op**, bitwise the
v1.2.0 one (asserted per frame).  `morph_meta_*.json` records the four
run-time guarantees of the morph plus three full-selector checks per
frame: `fullsel_confined_to_c1_arcs`, `fullsel_c2_only_recruited`, and
`fullsel_lam1_c2_only_equals_union`.

## Status in this repository

Scripts are integrated as received from the verified kit v3.1; the
shipped `probe_campaign.py` extraction bug is fixed here (all 13
hidden states via `output_hidden_states`, the `extract_features.py`
path — the kit version wrote a 2-D array and could not run from a
fresh install).  The campaign outputs these scripts produced are
already transported and verified — see
[../results/](../results/README.md).

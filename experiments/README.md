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
| `build_campaign.py` / `probe_campaign.py` | build and run the 54-item λ campaign (6 frames × 9 λ) |
| `extract_features.py` / `run_probing.py` | wav2vec2 hidden states + linear probes (L0–L12) |
| `lambda_experiment.py` | the λ-continuum experiment (production degrees) |
| `pouw075_/pouw0625_experiment.py` + `_stats.py` + `_figure.py` | Pouw Figure-1 protocol at λ = 0.75 / 0.625 |
| `note_stats.py` / `note_figure.py` | three-condition anchor statistics and figure |
| `figures.py`, `lambda_figures.py` | publication figures |
| `fetch_lexamb.py` + `lexamb_checksums.json` | natural-stimuli import (Pouw et al. 2024 release, checksum-verified, never redistributed) |
| `lexamb_cross_experiment.py` / `_figure.py` / `lexamb_synthetic_ctc.py` / `lexamb_fig2_digitize.py` | the natural-vs-synthetic crossing (Darcy et al. 2009 anchor is read from published numbers) |
| `morph_pair.py` | generic λ-morph synthesis for arbitrary pairs |
| `patch_N.py` | velar-nasal engine fix used by the base |

## Status in this repository

Scripts are integrated as received from the verified kit v3.0; the
ROADMAP W3 rewrite (drive synthesis through `morph --targets`, read
C1 windows from `morph_meta.json`) is tracked there. The campaign
outputs these scripts produced are already transported and verified —
see [../results/](../results/README.md).

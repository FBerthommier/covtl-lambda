# ROADMAP — Reproducing the Pouw et al. (2024) results in this repository

Status: **active workstream, awaiting the external recomputation.**
This document records the reproduction approach announced by the
analysis report ([covtl_lambda_C1C2_report.pdf](covtl_lambda_C1C2_report.pdf))
and organizes its transport into this repository. Nothing here is
shipped yet: the Wav2Vec2 results **cannot be recomputed locally**
(no probing layer in the local environment), and the recomputation
has been entrusted to an external agent.

## Context and cleanup of the provenance chain

- The historical reproducibility base lived in the local
  `nasal_synthesis` reproducibility kit (v2.1): synthesis + probing
  scripts, natural-stimuli import, figures and statistics.
- The external agent is **recomputing the entire Wav2Vec2 study with
  this repository (github.com/FBerthommier/covtl-lambda) as the
  reference** for the synthesis side. When that revision arrives, the
  kit's position is superseded: this repository becomes the citable
  source, and the kit's results must be **re-validated against it**
  before integration.

## Workstreams

### W1 — External recomputation of the Wav2Vec2 results (in progress, external)

The probing layer (wav2vec2-base / wav2vec2-base-960h over the λ
continua; linear probes on hidden states L0–L12; CTC transcription
measures) is recomputed by the external agent on the target-mode
continua produced by patch v1.2.0. Deliverable: a revised
`nasal_synthesis`-style package whose synthesis side is exactly the
one documented in this repository (target semantic, cluster windows,
`morph_meta.json` guarantees). **Gate:** every probing input must be
traceable to a `morph_meta.json` record of this repository.

### W2 — Natural-stimuli import (to reactivate)

The former patch project included importing the **natural speech
signals** of the reference line: the 86 natural sentences released by
Pouw et al. 2024 (their re-recording of the Gaskell & Marslen-Wilson
2001 lexical-ambiguity materials), fetched with per-file checksum
verification — kit script `repro/fetch_lexamb.py` — and the
natural-vs-synthetic crossing (`lexamb_cross_experiment.py`,
`lexamb_synthetic_ctc.py`, digitized Pouw Figure 2 via
`lexamb_fig2_digitize.py`). The human compensation rates that anchor
the reproduction come from **Darcy et al. (2009)**, as published in
the Pouw et al. comparison; these are *published numbers read from
the papers*, not redistributed recordings. Constraints carried over:
Pouw audio is fetched from the authors' release with checksums and is
**not redistributed** by this repository; the Pouw paper itself is
never shipped (digitization happens from a local copy, see NOTICE
Level 3).

### W3 — Transport of the reproduction protocol and results (after W1)

Port into `experiments/` (planned layout):
`pouw075/0625_experiment.py` (Pouw Figure-1 protocol at production
degrees λ = 0.75 / 0.625, confined protocol), `note_stats.py`
(three anchor conditions λ = 0/0.5/1.0), `run_probing.py` and
`extract_features.py` (probing drivers, with the torch/transformers
extra made optional), `figures.py`/`note_figure.py` (publication
figures). Each script lands updated to: (a) drive the synthesis
through `vtl-synth morph --targets` (patch v1.2.0 target semantic)
instead of the kit's dotted-era calls; (b) read the C1 windows from
`morph_meta.json` rather than hard-coded frames; (c) state the
VocalTractLab/COVTL citations (see NOTICE).

### W4 — Integrity verification at integration time

When the revised package arrives: (1) re-verify that its continua are
bitwise the ones of this repository (tract SHA-256 against the
`morph_meta.json` records, as the kit already did for its P1
continuum); (2) re-run the guarantee checks (`lam0_bitwise_canon`,
`glottis_bitwise_constant`, `lam1_theta_equals_c2`,
`outside_c1_arcs_bitwise_canon`) on every shipped item; (3) confirm
the phoneme inventories (SAMPA guard, dotless cluster notation);
(4) only then merge results into this repository under `results/`.

## Non-goals

- No probing dependencies (torch, transformers) enter this
  repository's core; W3 keeps them an optional extra, and W1's
  results are committed as data + figures, not as a training
  environment.
- No third-party audio or paper material is ever redistributed
  (fetch + verify + cite, as in W2).

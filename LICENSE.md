# License — how GPL-3.0-or-later applies to this repository

The license text is [LICENSE](LICENSE) (**GPL-3.0-or-later**). This
file records, in the three-level form adopted in
[NOTICE.md](NOTICE.md), how it applies to each module of the
repository. The obligation is not a choice: the patch is designed to
be installed into and linked with the covtl-pipeline engine, itself
GPL-3.0-or-later and built on VocalTractLab (GPL-3.0).

## Per-module application

| Module | License | Copyright holder | Notes |
|---|---|---|---|
| `patch/` | GPL-3.0-or-later | covtl-lambda contributors | every file carries an `SPDX-License-Identifier` header; the 4-line `main.py` insertion is applied to the user's own checkout, the upstream file is not redistributed |
| `examples/` | GPL-3.0-or-later | covtl-lambda contributors | generators and drivers, SPDX headers |
| `demos/` | GPL-3.0-or-later | covtl-lambda contributors | the embedded audio/figure payloads are outputs of the GPL engine driven by GPL generator code, distributed under the same license with the pages |
| `experiments/` | GPL-3.0-or-later | nasal_synthesis kit contributors (received, GPL-3.0-or-later) + covtl-lambda contributors | SPDX headers added at integration |
| `results/` | GPL-3.0-or-later | covtl-lambda contributors + nasal_synthesis kit contributors | data and documents produced with the GPL engine chain; the didactic synthesis PDF keeps its own internal references |
| `docs/` | GPL-3.0-or-later | covtl-lambda contributors | manual, reports, roadmap, architecture |

## Third-party layers (see NOTICE.md for the full chain)

- **covtl-pipeline / vtl-synth** — F. Berthommier, GPL-3.0-or-later
  (obtained separately; never included here).
- **VocalTractLab 2.4** — P. Birkholz, GPL-3.0, free for non-commercial
  research with citation instructions (cited in every demo page, in
  the results and experiments READMEs, and in the NOTICE; **scientific
  use of the stimuli must cite VocalTractLab and the COVTL
  publications**). No VTL component is redistributed here.
- **Pouw et al. (2024) / Darcy et al. (2009)** — cited as the
  reproduced protocol and the human anchor; their stimuli and paper
  are fetched/verified locally, never redistributed.

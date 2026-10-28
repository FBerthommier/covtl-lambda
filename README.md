# covtl-lambda

**The λ articulatory-assimilation effect on tautosyllabic consonant clusters (C1C2 → C1′C2), obtained by a modification of the COVTL model — shipped as an installable patch for [covtl-pipeline](https://github.com/fberthommier/covtl-pipeline) plus a set of interactive demonstrations.**

`patch v1.2.0` · GPL-3.0-or-later · 53 tests

> covtl-lambda **is not a copy of covtl-pipeline**: this repository
> carries no engine code. It provides (1) a patch installed into a
> covtl-pipeline checkout and (2) self-contained, pre-synthesized
> demonstrations that play without the engine. A web console
> (covtl-console) will be added later.

**Contents in three steps:** [1. Read the manuals](#1-read-the-manuals) · [2. Watch the demos](#2-watch-the-demos) · [3. Install covtl-pipeline and its patch](#3-install-covtl-pipeline-and-its-patch-last)

---

## 1. Read the manuals

| Document | Content |
|---|---|
| **[Reference manual (PDF, 14 pp.)](docs/manual.pdf)** | the complete λ semantics: §2.1 definition on polar targets, §4 CLI, §5 Python API, §6 step-by-step reference continuum, §7 guarantees and tests, §8 limitations |
| [Analysis report (PDF)](docs/covtl_lambda_C1C2_report.pdf) | (ρ, θ) formalization, C2 invariance, causal drift of C1, measured values |
| [CHANGELOG](CHANGELOG.md) | history v1.0 → v1.2.0 (the semantic fix) |
| **[Results](results/README.md)** | the verified Wav2Vec2 probing campaign (kit v3.0): 54/54 stimuli bitwise-identical to this repo's synthesis, guarantee checks, the didactic synthesis PDF |
| [ROADMAP](docs/ROADMAP.md) | the Pouw et al. (2024) reproduction workstream: external Wav2Vec2 recomputation, natural-stimuli import (Darcy et al. 2009 anchor), transport and integrity verification |
| [NOTICE — provenance & contributors](NOTICE.md) | three-level provenance, data import system, licenses |

The essentials in one equation — for the nasal C1 of a tautosyllabic
cluster, with C2 the onset of the same cluster:

<p align="center"><code>θ_C1(λ) = θ_C1 + λ · wrap(θ_C2 − θ_C1)&nbsp;&nbsp;&nbsp;(ρ optional)</code></p>

At λ = 0 the synthesis is strictly C1C2; at λ = 1 the polar target of
C1 **equals** the C2 target coordinates (C1′C2, the assimilated
cluster) — C2 never moves, and the glottal block (voicing, F0,
nasality) is bitwise constant along the whole continuum. Dotted forms
`C.C` (e.g. `klin.pan`) are a different synthesis and are **out of
scope**.

## 2. Watch the demos

The demos are **self-contained** HTML files: audio, figures and polar
trajectories are embedded at build time (the import system is
described in the [NOTICE](NOTICE.md)) — no server, no engine, works
offline. From GitHub, open a link then use **Raw** (or download) for
the interactive view; with GitHub Pages enabled they play in place at
`https://<org>.github.io/covtl-lambda/demos/…`.

**Worked example (K3):** [demos/lambda_demo.html](demos/lambda_demo.html) —
`klinkan → klimkan`, λ slider (9 steps), polar planning figure with
the C1 control node rotating toward the **fixed** C2 /k/ target,
audio per λ, articulator readout.

**Replication of the Pouw et al. (2024) items** — six sentence
frames, each a confined-λ continuum re-synthesized (speaker jd3);
*licensed* = the n → m assimilation before a labial is phonologically
licit, *unlicensed* = it is not:

| Page | Frame | Pair | C2 | Licensing |
|---|---|---|---|---|
| [pouw_P1](demos/pouw/pouw_P1.html) | *clean pan* | `klinpan → klimpan` | /p/ | licensed |
| [pouw_P2](demos/pouw/pouw_P2.html) | *mean boat* | `minbot → mimbot` | /b/ | licensed |
| [pouw_P4](demos/pouw/pouw_P4.html) | *thin packet* | `Tinpakit → Timpakit` | /p/ | licensed |
| [pouw_NV1](demos/pouw/pouw_NV1.html) | *clean tan* | `klintan → klimtan` | /t/ | unlicensed |
| [pouw_NV3](demos/pouw/pouw_NV3.html) | *clean can* | `klinkan → klimkan` | /k/ | unlicensed |
| [pouw_NV4](demos/pouw/pouw_NV4.html) | *mean goat* | `mingot → mimgot` | /g/ | unlicensed |
| [index](demos/pouw/index.html) | — | overview of the six frames | — | — |

In every demo: keys `←`/`→` move λ by ±0.125, `space` plays the
current state. The footer records the C1 window (frames, 400 Hz),
re-derived from the `.polar` phoneme timing, and the locality
guarantee `outside_c1_arcs_bitwise_canon`.

## 3. Install covtl-pipeline and its patch (last)

The manuals and demos are enough to understand and show the effect.
To **produce** your own continua, you need the engine:

1. **Clone and install covtl-pipeline** (not included here):
   ```bash
   git clone https://github.com/fberthommier/covtl-pipeline
   pip install -e covtl-pipeline[plot,dev]
   ```
2. **Apply the patch** from this repository:
   ```bash
   python patch/apply_patch.py --pipeline-root /path/to/covtl-pipeline
   ```
   It installs `vtl_synth/core/morphing.py`,
   `vtl_synth/cli/morph_cmd.py` and `tests/test_morphing.py` (53
   tests). The engine repository never carries λ code —
   `patch/revert_patch.py` restores the original state.
3. **Verify**:
   ```bash
   python -m pytest tests/test_morphing.py -q        # 53 passed
   vtl-synth morph --phonetic --targets klinpan -o morph_out --label P1
   ```

The step-by-step guide with expected outputs and troubleshooting:
[covtl-lambda-patch_INSTALL_PROMPT.md](covtl-lambda-patch_INSTALL_PROMPT.md).
An uncovered engine change surfaces first as a failure of the patch's
non-regression tests.

### SAMPA spelling: the guard and English non-words

The engine's phonetic parser silently drops (now: with a **warning**
from the SAMPA guard merged upstream) characters outside its
inventory. Spelling `cleanpan` does not read "clean + pan": the `c`
disappears and the engine synthesizes **`leanpan`** — a different
word, without the /k/. Hence the rule: spell each phone with its
SAMPA symbol, never carry English orthography over — the pairs are
written **`klinpan` / `klimpan`** (and `klintan/klimtan`,
`klinkan/klimkan`, `minbot/mimbot`, `Tinpakit/Timpakit`,
`mingot/mimgot`). The workflow in [examples/](examples/) verifies the
expected phoneme inventory systematically before synthesis.

## Repository layout

```
docs/             reference manual (tex + PDF), analysis report, figures, roadmap
demos/            self-contained HTML demos (embedded data)
results/          the verified Wav2Vec2 probing campaign (kit v3.0 artifacts)
experiments/      probing-campaign scripts (probing extra optional)
patch/            the engine patch: apply / revert / installed files
examples/         Python drivers for the reference continua + demo generators
covtl-lambda-patch_INSTALL_PROMPT.md   install/verify guide
```

*A web console (covtl-console) will be added later.*

## License and contributors

Code, documentation and demo payloads: **GPL-3.0-or-later** — see
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) (the GPL engine-linking
obligation; the contributor list; the data import system: everything
is synthesized locally, no third-party data redistributed).

## Reference

C. Pouw, M. de Heer Kloots, A. Alishahi, W. Zuidema. *Perception of
phonological assimilation by neural speech recognition models.*
Computational Linguistics 50(4):1557–1585, 2024.
[doi:10.1162/coli_a_00536](https://doi.org/10.1162/coli_a_00536)

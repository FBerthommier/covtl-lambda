# covtl-lambda

**Articulatory assimilation continua on tautosyllabic CC clusters (C1C2 → C1′C2), built on [covtl-pipeline](https://github.com/fberthommier/covtl-pipeline) (`vtl-synth`) — with an interactive replication package of the Pouw et al. (2024) items.**

`v1.2.0` · GPL-3.0-or-later · Python ≥ 3.9 · 53 tests · [Manual (PDF)](docs/manual.pdf) · [Changelog](CHANGELOG.md) · [Provenance & contributors](NOTICE.md)

---

## Interactive demos (self-contained — click to open)

The demos are single HTML files with audio, figures and polar
trajectories embedded at build time: no server, no engine, works
offline. From this README they are one click away; on GitHub, open a
link then use **Raw** (or download the file) for the interactive
view, or enable GitHub Pages to play them in place at
`https://<org>.github.io/covtl-lambda/demos/…`.

**Worked example (K3):**
[demos/lambda_demo.html](demos/lambda_demo.html) — `klinkan → klimkan`,
λ slider (9 steps), polar planning figure with the C1 control node
rotating toward the **fixed** C2 /k/ target, audio per λ, articulator
readout.

**Replication of the Pouw et al. (2024) items** — six sentence frames,
each a confined-λ continuum re-synthesized on the patched engine
(speaker jd3, default timings). Endpoints are tautosyllabic clusters
(dotless); *licensed* = the n → m assimilation before a labial is
phonologically licit, *unlicensed* = it is not:

| Page | Sentence frame | Pair | C2 | Licensing |
|---|---|---|---|---|
| [pouw_P1](demos/pouw/pouw_P1.html) | *clean pan* | `klinpan → klimpan` | /p/ | licensed |
| [pouw_P2](demos/pouw/pouw_P2.html) | *mean boat* | `minbot → mimbot` | /b/ | licensed |
| [pouw_P4](demos/pouw/pouw_P4.html) | *thin packet* | `Tinpakit → Timpakit` | /p/ | licensed |
| [pouw_NV1](demos/pouw/pouw_NV1.html) | *clean tan* | `klintan → klimtan` | /t/ | unlicensed |
| [pouw_NV3](demos/pouw/pouw_NV3.html) | *clean can* | `klinkan → klimkan` | /k/ | unlicensed |
| [pouw_NV4](demos/pouw/pouw_NV4.html) | *mean goat* | `mingot → mimgot` | /g/ | unlicensed |
| [index](demos/pouw/index.html) | — | overview of the six frames | — | — |

In every demo: keys `←`/`→` move λ by ±0.125, `space` plays the
current state. The provenance footer records the C1 window (frames,
400 Hz), re-derived from the endpoint `.polar` phoneme timing, and the
locality guarantee `outside_c1_arcs_bitwise_canon` (bitwise-canonical
context at every λ).

## What it does

The lambda variation applies to a **tautosyllabic CC cluster** — C1 =
the nasal coda, C2 = the following onset of the same cluster — and its
effect is defined on the **polar targets** of the cluster branch: the
parameter λ moves the θ of C1 toward the θ of C2 (and, optionally, its
ρ), producing strictly the transition

<p align="center"><code>C1C2 → C1'C2&nbsp;&nbsp;&nbsp;(e.g. klinpan → klimpan, 0 ≤ λ ≤ 1)</code></p>

At λ = 0 the cluster target is strictly C1C2; at λ = 1 the C1 polar
target has reached the C2 target coordinates — C1′C2, the
place-assimilated cluster: the C1 gesture now POINTS at the C2 target,
while C2 itself never moves. Each λ is re-synthesized through the
engine's own syllabic trajectory builder, so every continuum point is
a genuine cluster utterance (`.tract` + `.wav`), with the glottis
block held bitwise constant (voicing, F0 and nasality identical along
the continuum). Dotted forms `C.C` (e.g. `klin.pan`) are a different
synthesis and are **out of scope** — a dotted input is rejected with an
explicit error.

This is the stimulus-generation primitive for assimilation-perception
and model-probing paradigms in the spirit of
[Pouw et al. 2024](https://doi.org/10.1162/coli_a_00536) — the
mechanism is the generalization of the λ experiment of the
`nasal_synthesis` reproducibility kit, extracted into a reusable,
documented patch with 53 tests. Full treatment:
[docs/manual.pdf](docs/manual.pdf) (§2.1 semantic, §6 worked
continuum, §7 guarantees) and the two analysis reports
[docs/covtl_lambda_C1C2_report.pdf](docs/covtl_lambda_C1C2_report.pdf) (EN) /
[docs/rapport_lambda_C1C2.pdf](docs/rapport_lambda_C1C2.pdf) (FR).

## A note on SAMPA spelling: the guard and English non-words

The engine's phonetic parser works on a SAMPA inventory oriented
toward **English** — and it composes its phoneme list by keeping only
characters that belong to that inventory. A spelling like
`cleanpan`, read character by character, is therefore **not** the
English word *clean* + *pan* inside the engine: the digraph-friendly
letters it expects differ from ordinary English orthography, and the
`c` of `cleanpan` is not a SAMPA phone. The consequence used to be
silent: the engine simply **deleted** the out-of-inventory character
and synthesized `leanpan` — a different word, with the /k/ missing —
without any warning.

The engine now ships a **SAMPA guard** (merged upstream): instead of
swallowing the deletion it **warns** — it reports the characters that
do not compose a valid phone of the inventory, so a phrase spelled
`cleanpan` announces that it will be synthesized as `leanpan`. The
guard does not repair the input; it makes the deletion visible.

For the assimilation pairs this effect dictates the spelling: to
obtain the intended /k-l-i-n-p-a-n/ (and its assimilated
/k-l-i-m-p-a-n/), the pairs are written **`klinpan` / `klimpan`** —
and likewise `klintan/klimtan`, `klinkan/klimkan`, `minbot/mimbot`,
`Tinpakit/Timpakit`, `mingot/mimgot`. Rule of thumb: spell each phone
with its SAMPA symbol (k, l, i, n, m, p, a, t, g, o), never carry
English orthography over (`clean pan` → `klinpan`, not `cleanpan`).
When a `.polar` file is available, the expected phoneme inventory can
be verified before synthesis — the workflow in
[examples/](examples/) does this systematically, and the
[install prompt](covtl-lambda-patch_INSTALL_PROMPT.md) lists it among
the acceptance checks.

## Repository layout

```
demos/            self-contained interactive HTML demos (data-embedded)
docs/             reference manual (tex + PDF), cover, analysis reports
patch/            the engine patch: apply_patch.py / revert_patch.py / files/
examples/         Python drivers for the reference continua + demo generators
covtl-lambda-patch_INSTALL_PROMPT.md   step-by-step install/verify prompt
```

The patch installs `vtl_synth/core/morphing.py`,
`vtl_synth/cli/morph_cmd.py` and `tests/test_morphing.py` into a
covtl-pipeline checkout — the engine repo itself never carries λ
code. Expectation for the full engine suite and troubleshooting:
[install prompt](covtl-lambda-patch_INSTALL_PROMPT.md).

Quick start (patched engine):

```bash
python patch/apply_patch.py --pipeline-root /path/to/covtl-pipeline
python -m pytest tests/test_morphing.py -q            # 53 passed
vtl-synth morph --phonetic --targets klinpan -o morph_out --label P1
```

## Reproducibility & data provenance

All audio, tract trajectories and figures in the demos were
**synthesized locally** with the patched covtl-pipeline engine (jd3
speaker, 100/80/200 ms gesture timings) and embedded into the pages at
build time — this is the data import system described in
[NOTICE.md](NOTICE.md): no recordings, no human data, and no material
from the Pouw et al. paper is redistributed; their experiment is
*cited* as the design reference and our stimuli are re-synthesized
from engine SAMPA. Regenerate everything with
`examples/lambda_continuum.py`, `examples/html/make_lambda_demo.py`
and `examples/html/make_pouw_demos.py`.

## License & contributors

Code, documentation and demo payloads: **GPL-3.0-or-later** — see
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for the three-level
provenance (this repo / covtl-pipeline engine / VocalTractLab and
cited research) and the contributor list.

## Reference

C. Pouw, M. de Heer Kloots, A. Alishahi, W. Zuidema. *Perception of
phonological assimilation by neural speech recognition models.*
Computational Linguistics 50(4):1557–1585, 2024.
[doi:10.1162/coli_a_00536](https://doi.org/10.1162/coli_a_00536)

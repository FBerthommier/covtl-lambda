# ARCHITECTURE — repository modules and dependency rules

covtl-lambda is organized as **independent modules with one-way data
flow**. No module imports another; communication happens only through
files (the patch installs into the engine; the generators read the
engine's outputs; the demos embed those outputs; the results pin them
by hash).

```
                        covtl-pipeline (external engine, GPL-3.0-or-later)
                             |  patch/apply_patch.py  ^  revert_patch.py
                             v                        |
   patch/files/  ----installed into----> vtl_synth/core/morphing.py
   (the ONLY lambda code)                 vtl_synth/cli/morph_cmd.py
                            |
            vtl-synth morph --targets <cluster>
                            |
            .tract / .wav / .polar / morph_meta.json
                            |
      +---------------------+---------------------+
      v                     v                     v
  examples/            results/ (verified    demos/ generators
  (reference           external artifacts    (examples/html/)
  continua, demo          from the campaign   make_*.py
  payload builders)       + experiments/)          |
       |                                          v
       +----------------------------------> demos/*.html
                                                  (data-embedded,
                                                   no engine at run time)
```

## Modules

| Module | Role | Depends on | Never contains |
|---|---|---|---|
| `patch/` | the λ semantic as an installable engine patch: `apply_patch.py` / `revert_patch.py` + the three installed files and the 53-test suite | a covtl-pipeline checkout at apply time | any engine source; the engine repo never carries λ code |
| `examples/` | reference-continuum drivers and the demo generators (`html/make_lambda_demo.py`, `html/make_pouw_demos.py`) | the patched engine at build time | the generated demo pages (shipped in `demos/`, gitignored build outputs under `examples/html/`) |
| `demos/` | self-contained interactive HTML: audio (AAC), tract figures and polar trajectories embedded as base64 at build time | **nothing at run time** — offline, `file://`, no server, no engine, no network | engine code; each page carries the COVTL + VocalTractLab attribution |
| `results/` | the verified Wav2Vec2 campaign (kit v3.0): machine-readable stats, per-frame morph metadata, figures, the didactic synthesis PDF | nothing (data + documents) | the wav2vec2 checkpoints, the 54 stimulus files (regenerable, hash-pinned in `synth_meta.json`), any third-party audio |
| `experiments/` | the probing-campaign drivers (synthesis base, feature extraction, probes, statistics, figures, natural-stimuli import) | patched engine for synthesis; optional torch/transformers extra for probing only | probing dependencies in the repository core (never a hard dependency) |
| `docs/` | reference manual, analysis report, ROADMAP, figures | LaTeX toolchain to rebuild (PDFs are shipped) | build artifacts |
| `patch/../covtl-lambda-patch_INSTALL_PROMPT.md` | step-by-step install/verify guide with expected outputs | — | — |

## Invariants

1. **One-way flow** — engine → synthesis artifacts → generators →
   embedded demos; synthesis artifacts → (verified) results. Nothing
   flows back into the engine.
2. **Modularity** — each directory is consumable alone: `demos/` on
   any machine with a browser; `results/` as a citable dataset;
   `patch/` against any clean engine checkout; `experiments/` against
   the kit-layout `work/` tree.
3. **Verification gate** — every external artifact enters `results/`
   only after the ROADMAP W4 checks (bitwise synthesis identity,
   guarantee booleans, SAMPA inventories) and the W5 notation sweep.
4. **Notation ladder** — `xC1C2x` (tautosyllabic cluster) is the only
   λ input; `xC1.C2x` (C.C split) and `xC1 C2x` (pause) are different
   syntheses, documented but out of scope; `xC1′C2x` is the λ = 1
   state. See ROADMAP W5.
5. **Licensing** — everything is GPL-3.0-or-later (engine-linking
   obligation); per-module application in [LICENSE.md](../LICENSE.md),
   provenance chain in [NOTICE](../NOTICE.md).

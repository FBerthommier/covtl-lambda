# ROADMAP — Reproducing the Pouw et al. (2024) results in this repository

Status: **W1 delivered and verified; results integrated (2026-10-28).**
This document records the reproduction approach announced by the
analysis report ([covtl_lambda_C1C2_report.pdf](covtl_lambda_C1C2_report.pdf))
and organizes its transport into this repository.

## Context and cleanup of the provenance chain

- The historical reproducibility base lived in the local
  `nasal_synthesis` reproducibility kit (v2.1): synthesis + probing
  scripts, natural-stimuli import, figures and statistics.
- The external agent **recomputed the entire Wav2Vec2 study with this
  repository as the synthesis reference** and delivered
  `nasal_synthesis_kit_v3.0.zip` (2026-10-28). The kit's position is
  superseded: this repository is now the citable source, and the
  verified campaign artifacts live in [results/](../results/README.md)
  with the driving scripts in [experiments/](../experiments/README.md).

## Workstreams

### W1 — External recomputation of the Wav2Vec2 results ✅ DELIVERED & VERIFIED

Delivered as kit v3.0: one 54-item campaign (6 frames × 9 λ) on the
target-mode continua, stage-1 probe instrument unchanged, production
blocks as byte-identical slices. Verification record:
[results/README.md](../results/README.md).

### W2 — Natural-stimuli import ✅ INTEGRATED (scripts)

The import path lives in [experiments/fetch_lexamb.py](../experiments/fetch_lexamb.py)
with its checksum manifest (`lexamb_checksums.json`), plus the
natural-vs-synthetic crossing scripts. The 86 natural sentences of
Pouw et al. 2024 are fetched from the authors' release with per-file
checksum verification and are **not redistributed**; the Darcy et al.
(2009) human anchor is read from published numbers.

### W3 — Transport of the reproduction protocol and results ✅ DELIVERED (as-received), rewrite pending

Campaign results, figures and the (notation-corrected) synthesis PDF
are integrated under [results/](../results/README.md); the driving
scripts under [experiments/](../experiments/README.md) as received.
**Remaining:** the ROADMAP W3 rewrite of the synthesis-side scripts —
(a) drive synthesis through `vtl-synth morph --targets` (already the
kit's campaign behavior), (b) read C1 windows from `morph_meta.json`
in the analysis scripts still carrying hard-coded frames, (c) state
the VocalTractLab/COVTL citations — tracked as follow-up commits.

### W4 — Integrity verification at integration time ✅ PASSED (2026-10-28)

(1) 54/54 campaign tracts **numerically identical** to the repository
reference continua (0 differing cells; the only byte difference is the
tract-writer EOL convention, kit LF vs local CRLF); (2) the six
`morph_meta` records carry 5/5 true guarantees each with per-λ
`theta_c1`; (3) dotless cluster inventories confirmed (`klinpan`,
`minbot`, `Tinpakit`, `klintan`, `klinkan`, `mingot`); (4) merged into
[results/](../results/README.md). Full record:
[results/README.md](../results/README.md).

### W5 — Documentary revision of the transition notation (standing)

The major historical error was notational confusion about the
transition states. The **canonical ladder**, as fixed by the v1.2.0
semantic and to be enforced across every document and demo:

| Notation | Meaning | Status |
|---|---|---|
| `xC1 C2x` | space separator = prosodic pause (200 ms short / 320 ms long) — two separate words | legal, but a **different synthesis**; never a λ endpoint |
| `xC1.C2x` | dot separator = C.C split, no pause — different cluster/gesture machinery | legal engine input, **out of scope** of the λ variation; a dotted input to `morph --targets` is rejected |
| `xC1C2x` | no separator = tautosyllabic CC cluster (C1 nasal coda + C2 onset) | the **only** legitimate λ input |
| `xC1′C2x` | λ = 1 endpoint: C1 polar target reached the C2 target coordinates | the assimilated state; never written `C1.C2` or `C1 C2` |

Rules enforced in every document, demo and result table: (a) λ
endpoints are always written dotless (`klinpan → klimpan`); (b)
dotted/spaced forms appear only inside explicit out-of-scope,
legacy or warning statements; (c) the SAMPA guard caveat
(`clinpan` → engine drops `c`, write `klinpan`) accompanies any
discussion of English orthography; (d) separator semantics (space =
pause, dot = C.C split, none = cluster) are stated once, in the
notation keybox of the manual, and referenced elsewhere instead of
being restated in diverging wording.

**Baseline verification (2026-10-28, this commit)**: full sweep of
the repository — zero dotted endpoints outside out-of-scope contexts,
zero spaced forms, no diverging separator-semantics statements found.
W5 was applied to the incoming revision at integration: the kit's
documents passed (the remaining spaced forms are the legitimate
separator explainer), and the two dotted `\sampa{klin.pan}` example
occurrences in the synthesis PDF source were corrected to the
canonical ladder before the PDF was rebuilt into
[results/](../results/README.md). W5 remains a standing gate for all
future documents and result tables.

## Non-goals

- No probing dependencies (torch, transformers) enter this
  repository's core; W3 keeps them an optional extra, and W1's
  results are committed as data + figures, not as a training
  environment.
- No third-party audio or paper material is ever redistributed
  (fetch + verify + cite, as in W2).

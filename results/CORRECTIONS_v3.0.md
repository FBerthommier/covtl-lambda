# CORRECTIONS_v3.0 — the revised λ reference (target-level cluster model)

**2026-10-28 · kit v2.1 → v3.0.**

## What changed

The λ model has moved to the upstream reference repository
[FBerthommier/covtl-lambda](https://github.com/FBerthommier/covtl-lambda)
(**v1.2.0**, repo commit `c13f4d0`). The revision is *semantic*, not
procedural:

| | v2.1 (superseded) | v3.0 (this kit) |
|---|---|---|
| Level of action | motor: λ blends the 30-dim tract-parameter sequences of the **dotted** endpoints (`klin.pan`/`klim.pan`), confined to the C1 window with 25 ms raised-cosine ramps | **target**: θ_C1(λ) = θ_C1 + λ·wrap(θ_C2 − θ_C1) on the **tautosyllabic cluster** (`klinpan`), ρ fixed |
| Endpoints | dotted spellings (a `.` is an intra-phrase C.C split) | cluster spellings; dotted spellings are **rejected** with an explanatory error |
| What moves at λ=1 | the lips close inside the C1 window (a real surface [m]) | the C1 *gesture* reaches the C2 place coordinates; the labial closure remains the business of C2 — the lips, velum, glottis, C2 and context stay bitwise canonical |
| Guarantees | window-confinement assertions | four run-time guarantees per synthesis: `lam0_bitwise_canon`, `lam1_theta_equals_c2`, `glottis_bitwise_constant`, `outside_c1_arcs_bitwise_canon` (held 54/54) |
| Stimulus set | 18 (2 frames × 9 λ) + 18 + 24 | one **54-item campaign** (6 frames × 9 λ); the production-degree blocks are byte-identical slices |
| Probe instrument | retrained per experiment | the unchanged stage-1 instrument (LOSO curves identical), one probing run |

The engine reference tree is covtl-pipeline `1d15ed7` + the committed
corrections series (SAMPA guard + g2p de/pt fixes, content of
`fix/polar-teardrop-arcplot`) + covtl-lambda v1.2.0; its 53-test
suite passes on that tree.

## Where the old results live

- `work/legacy_motorblend/` — the complete v2.1 confined-motor λ
  results (`lambda/`, `lambda075/`, `lambda0625/`), unchanged.
- `work/legacy_unconfined/` — the v2 dotted unconfined tier.
- `work/correction_comparison.json` — the v2→v2.1 comparison.
- `work/revision_comparison_v30.json` — the machine-readable
  v2.1→v3.0 comparison (condition rates at both production degrees,
  asymmetry tests, continuum verdict flips).

## Before / after (compensation rates, % verdict /n/)

### λ = 0.75 production (Pouw three-condition protocol)

| Cell | v2.1 | v3.0 |
|---|---|---|
| Control (λ=0, 6 frames) | 76.9 / 97.4 | **64.1 / 61.5** |
| Viable (licensed pool) | 61.5 / 74.4 | **79.5 / 97.4** |
| Unviable (unlicensed pool) | 59.0 / 74.4 | **89.7 / 97.4** |
| Asymmetry (V−U), 960h | +2.6 [−18.4, +23.2] | **−10.3 [−26.5, +6.2]** |
| Asymmetry (V−U), base | 0.0 [−19.0, +19.0] | **0.0 [−10.8, +10.8]** |

### λ = 0.625 production

| Cell | v2.1 | v3.0 |
|---|---|---|
| Viable | 74.4 / 94.9 | **79.5 / 100.0** |
| Unviable | 87.2 / 97.4 | **89.7 / 94.9** |
| Asymmetry, 960h | −12.8 [−29.9, +4.9] | **−10.3 [−26.5, +6.2]** |
| Asymmetry, base | −2.6 [−14.5, +8.7] | **+5.1 [−4.6, +16.9]** |

### The λ = 1 reference (where the sign inverts)

| Contrast | v2.1 | v3.0 |
|---|---|---|
| Viable−Unviable, 960h | −2.6 points (n.s.) | **−23.1 points [−39.9, −4.7]** — the one interval that excludes zero |
| Viable−Unviable, base | +2.6 points (n.s.) | **−10.3 points [−23.6, +0.6]** |
| T4 category boundary | λ* ≈ 0.73–0.91 (late boundary) | **none** — P(/n/) never falls below 0.5 at any degree of any frame pool |

## Why the numbers moved

1. **The control condition changed meaning.** Under v1.2.0, λ=0 is
   bitwise the canonical *cluster* — and the tautosyllabic cluster
   carries anticipatory coarticulation of the C1 nasal with the
   following consonant. The acoustic layer reads it as /m/-ish
   (pooled P(m) at the CNN: .51–.54 licensed vs .42–.44 unlicensed),
   so the pooled controls sit at 64.1/61.5 % instead of the near-
   ceiling dotted-citation values of the earlier tiers. The dotted
   tier is retained as a reference block only (machine-readable
   distance between the tiers: up to 0.86 in P(n) — the quantitative
   statement that cluster and C.C split are different gestures).
2. **The lips never close on the assimilated side.** The v2.1 motor
   blend produced real surface-[m] acoustics inside the C1 window,
   which the phone code tracked to a late category boundary. The
   target-level rotation moves jaw and tongue only; the phonetic
   layers (T2–T4) read the underlying /n/ at every λ of every frame,
   and the boundary disappears.
3. **What T4 tracks instead is the coarticulation cleanup.** On the
   velar-C2 frames the probability of /n/ *rises* with λ (pooled
   unlicensed ρ(λ, P(n)) = +0.90 to +0.98 at T4): rotating the C1
   target away from the labial-coarticulated cluster canonical moves
   the murmur toward the trained /n/ centroid.
4. **The late-layer effect is unidirectional, not bidirectional.**
   The v2.1 "expectation at λ=0 / compensation at λ=1" pattern was an
   artifact of the motor blend's surface acoustics. Under v1.2.0 the
   deep CTC layers carry a single labial-context reading — a nasal
   before a labial reads toward [m] regardless of the gesture —
   present at λ=0 on P2/P4 and sharpened at λ=1 on P1. P4 (*thin
   packet*, no /m/ segment anywhere) reads /m/ at T10 along the whole
   continuum: the reading is carried by the cluster context, not by
   the token's phonetics. The licensed pool therefore *loses*
   underlying verdicts at λ=1 (the −23.1-point inverted contrast),
   the opposite sign to Pouw et al.'s viable-over-unviable advantage.

## Headline impact

- The null-asymmetry conclusion of the reproduction attempts
  **survives and sharpens**: no positive licensed-over-unlicensed
  contrast appears at any of the nine production degrees; the one
  significant contrast is *negative* and carried by letter
  supervision, not licensing.
- The dose-response **inverts its shape**: rates now *rise* from the
  coarticulated λ=0 baseline (the canonical cluster is the low point),
  plateau through the mid-continuum, and only the CTC-tuned licensed
  track descends at the endpoint.
- The two-stage account of the synthesis document is restated: stage 1
  is boundary-free on the revised continuum (it tracks coarticulation,
  not assimilation); stage 2 is a unidirectional labial-context
  reading, not a bidirectional correction.
- Everything λ-specific in both shipped PDFs was regenerated
  (synthesis 21 pages, note 24 pages); the stage-1 probing campaign,
  the lexamb crossing, and the license layer are unchanged.

## Reproducing the comparison

```
python repro/pouw075_stats.py        # v3.0 conditions + asymmetries
python repro/pouw0625_stats.py       # + the nine-degree dose-response
# the v2.1 numbers: work/legacy_motorblend/*/pouw0*_stats.json
# the machine-readable diff: work/revision_comparison_v30.json
```

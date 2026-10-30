# CORRECTIONS_v3.1 — the full-CC-selector condition

**2026-10-30 · kit v3.0 → v3.1.**

## What changed

The v3.0 campaign kept one restriction whose cost the wav2vec2
readings made explicit: under λ the retargeted C1 carried **its own
selector**, so the articulators that realize C2 and are absent from
sel(C1) — the lips LD/LP of a labial C2 — stayed on the vocalic
background along the whole continuum (the **lip blocking**: lips
bitwise constant on every frame). This revision removes it:

> under λ variation the **whole CC selector** is taken into account,
> **as in the base model**, whose cluster machinery already drives the
> union sel(C1) ∪ sel(C2).

| | v3.0 (superseded, preserved) | v3.1 (this kit) |
|---|---|---|
| C1 selector under λ | sel(C1) only (lips blocked on the background) | **union sel(C1) ∪ sel(C2)** — the base model's cluster selector |
| C2-only parameters (lips LD/LP of a labial C2) | bitwise constant | **recruited with weight λ**: P(λ) = (1−λ)·P[sel C1] + λ·P[union], exact at the endpoints, confined to the C1-incident sub-arcs |
| Lip opening at the C1 target point (P1) | 0.61 cm at every λ | **0.61 → 0.46 → 0.16 → 0.00 cm** at λ = 0 / 0.5 / 0.75 / 1 (full labial closure — the base model's own /p/ value — with protrusion LP → 1.08 cm) |
| Jaw/tongue rotation | θ_C1(λ) = θ_C1 + λ·wrap(θ_C2 − θ_C1), ρ fixed | **bitwise identical to v3.0** at every degree |
| Glottis, C2, context, velum | bitwise canonical | **bitwise canonical** (unchanged) |
| Unlicensed frames (C2 = /t/, /k/, /g/ after /n/) | — | **structural no-op** (sel(C2) ⊆ sel(/n/)): the NV continua are bitwise the v3.0 ones (asserted per degree, tracts and wavs) |
| Guarantees | four run-time guarantees (held 54/54) | the same four **plus three full-selector checks**: `fullsel_confined_to_c1_arcs`, `fullsel_union_noop` (NV frames, verified bitwise), `fullsel_lam1_c2_only_equals_union` / `fullsel_c2_only_recruited` (held 54/54) |
| Implementation | engine patch `vtl_synth/core/morphing.py` (covtl-lambda v1.2.0) | engine patch **unchanged** + the kit's additive module `repro/morphing_fullsel.py` (per degree the phrase is synthesized twice through the engine's own chain — selector untouched vs. union — and the C2-only parameters are blended) |

The engine reference tree is unchanged: covtl-pipeline `1d15ed7` + the
committed corrections series + covtl-lambda v1.2.0 (`c13f4d0`) + the
/N/ closure patch; the full-selector condition is a kit-level
extension over the patched engine, not an engine change.

## Where the old results live

- `work/legacy_lipblocked/` — the complete v3.0 blocked-lip campaign
  (`lambda/`, `lambda075/`, `lambda0625/`: tracts, wavs, probing and
  statistics), byte-identical to the v3.0 kit.
- `work/legacy_motorblend/`, `work/legacy_unconfined/` — the earlier
  tiers (v2.1, v2), unchanged.
- `work/revision_comparison_v31.json` — the machine-readable
  v3.0→v3.1 comparison (bitwise anchors, condition rates, asymmetry
  tests, dose-response endpoints, T4 boundaries).
- `work/revision_comparison_v30.json`, `work/correction_comparison.json`
  — the earlier comparisons.

## Before / after (compensation rates, % verdict /n/)

### λ = 0.75 production (Pouw three-condition protocol)

| Cell | v3.0 | v3.1 |
|---|---|---|
| Control (λ=0, 6 frames) | 64.1 / 61.5 | **64.1 / 61.5** (bitwise-identical anchors) |
| Viable (licensed pool) | 79.5 / 97.4 | **79.5 / 100.0** |
| Unviable (unlicensed pool) | 89.7 / 97.4 | **89.7 / 97.4** (bitwise-identical frames) |
| Asymmetry (V−U), 960h | −10.3 [−26.5, +6.2] | **−10.3 [−26.5, +6.2]** (unchanged) |
| Asymmetry (V−U), base | 0.0 [−10.8, +10.8] | **+2.6 [−6.6, +13.2]** |

### λ = 0.625 production

| Cell | v3.0 | v3.1 |
|---|---|---|
| Viable | 79.5 / 100.0 | **79.5 / 100.0** |
| Unviable | 89.7 / 94.9 | **89.7 / 94.9** |
| Asymmetry, 960h | −10.3 [−26.5, +6.2] | **−7.7 [−23.7, +8.4]** |
| Asymmetry, base | +5.1 [−4.6, +16.9] | **+5.1 [−4.6, +16.9]** |

### The λ = 1 reference (where the sign inverts)

| Contrast | v3.0 | v3.1 |
|---|---|---|
| Viable−Unviable, 960h | −23.1 points [−39.9, −4.7] — the one interval excluding zero | **−20.5 points [−37.3, −2.4]** — excludes zero |
| Viable−Unviable, base | −10.3 points [−23.6, +0.6] (n.s.) | **−17.9 points [−32.7, −5.3]** — **now excludes zero**: the inverted contrast is bilateral |
| T4 category boundary | none | **none** — the full labial closure at λ=1 does not create one (closest approach: P2, CTC-tuned, P(/n/) = .465 at λ=1 without a verdict flip) |

## What the condition changed scientifically

1. **The two-stage account survives the completed gesture.** The
   phonetic layers (T2–T4, LOSO 1.00) still read the underlying /n/
   at every degree of every frame — now *through a real labial
   closure* at λ=1. The boundary-free stage-1 result was not an
   artifact of the blocked lips.
2. **The λ=1 licensed dip is not an artifact of the open lips
   either — it strengthens.** With the lips blocked the inverted
   contrast was significant on the CTC-tuned listener only; with the
   lips recruited it is significant on both listeners (−20.5 and
   −17.9 points, both CIs excluding zero), and the pooled deep-mean
   gap widens (P(/n/) T10–T12: .232 licensed vs .636 unlicensed,
   against .360 vs .573 blocked).
3. **The late-layer labial reading is context-bound, not
   closure-bound.** P1 (*clean pan*) — the frame whose lips now
   close — *loses* its T10–T11 labial reading (P(m) at λ=1:
   .44/.53 blocked → .24/.09 recruited, 13/13 /n/ verdicts), while
   P2/P4 keep theirs (P(m) = .65–.95), and P4 still reads [m] with no
   /m/ segment anywhere in the frame. The letter-supervised reading
   tracks the cluster context; actually closing the lips does not
   feed it.
4. **The self-supervised listener's licensed deep correlation turns
   positive** (ρ(λ, P(/n/)) at T11: +.17 blocked → +.77* recruited):
   the lip recruitment suppresses that listener's late labial drift
   as λ grows — the mirror image of mechanism 3.
5. **The production-degree structure is stable.** The λ=0.75 cells of
   the CTC-tuned listener do not move at all (79.5 / 89.7 / 64.1);
   the dose-response keeps its shape (rise from the coarticulated
   baseline, mid-continuum plateau, endpoint descent), with the
   endpoint descent now on both listeners' licensed tracks.

## Reproducibility

`repro/morphing_fullsel.py` (the additive module) +
`repro/build_campaign.py` (v3.1: builds the full-selector continuum)
+ the unchanged `probe_campaign.py` / `make_results.py` /
`note_stats.py` / `pouw075_stats.py` / `pouw0625_stats.py` chain
reproduce every number in this file from the kit root; the build
asserts the bitwise anchors against the v3.0 block when it is present
(`work/legacy_lipblocked/`): λ=0 for all six frames, all nine degrees
of the NV frames, and — for the licensed frames — every non-lip
tract column and every frame outside the C1-incident sub-arcs.

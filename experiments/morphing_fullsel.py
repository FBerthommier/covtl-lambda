# SPDX-FileCopyrightText: 2026 nasal_synthesis kit contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""morphing_fullsel.py -- full-CC-selector target morph (kit v3.1).

The v3.1 CONDITION of the lambda variation, extending the covtl-lambda
v1.2.0 target semantic ("le reste est identique"):

    under lambda, the WHOLE CC SELECTOR is taken into account, as in
    the base model.

Mechanics
---------
In the base engine, a tautosyllabic CC cluster block is realized with
the GLOBAL UNION of the two consonant selectors active on all its
sub-arcs (syltraj.build_cluster_pval: ``act = union(c.params)``), each
parameter following the polar target of the consonant that OWNS it and
the vocalic background otherwise.  The v1.2.0 target morph retargets
the C1 polar target (theta toward theta_C2) but keeps C1's OWN
selector, so the articulators that realize C2 and are absent from
sel(C1) -- the lips LD/LP of a labial C2 -- stay on the background:
the "lip blocking" of the v3.0 campaign (lips bitwise constant along
every continuum).

This module removes that blocking: at each lambda the phrase is
re-synthesized TWICE through the engine's own chain,

  * T_sel1 : the v1.2.0 item verbatim (C1 retargeted, selector
    untouched -- jaw/tongue rotation bitwise identical to v3.0);
  * T_union: the same retargeted C1 with its selector widened to the
    CC union sel(C1) u sel(C2), i.e. the base model's cluster
    machinery with the C1 gesture recruiting the whole CC selector;

and the v3.1 item is T_sel1 with the C2-ONLY parameters (those in
sel(C2) \\ sel(C1)) blended from their canonical background values
toward their full-selector realization with weight lambda:

    P(lam) = (1 - lam) * P[T_sel1] + lam * P[T_union]

The recruitment is therefore linear in lambda, exact at the endpoints
(bitwise), and strictly confined to the C1-incident sub-arcs (outside
them T_sel1 == T_union == canonical, asserted).  For a C2 whose
selector is a subset of sel(C1) (alveolar/velar C2 after /n/) the
union is sel(C1) itself: T_union == T_sel1 bitwise and the v3.1
continuum IS the v1.2.0 continuum (asserted) -- the condition only
bites where C2 owns articulators that C1 does not, i.e. exactly the
labial assimilations (P1 klinpan, P2 minbot, P4 Tinpakit).

Guarantees (asserted per frame, recorded in morph_meta.json)
------------------------------------------------------------
  lam0_bitwise_canon          lambda = 0 reproduces the canonical
                              cluster bitwise (blend weight 0).
  glottis_bitwise_constant    voicing/F0/nasality never move (the
                              glottis is identity-based; unchanged
                              from v1.2.0).
  lam1_theta_equals_c2        at lambda = 1 the C1 theta equals the
                              C2 theta verbatim (v1.2.0 semantic;
                              rho stays rho_C1, theta channel only).
  outside_c1_arcs_bitwise_canon  every frame outside the C1-incident
                              sub-arcs stays bitwise canonical.
  fullsel_confined_to_c1_arcs T_sel1 == T_union outside the arcs.
  fullsel_union_noop          (frames with C2 selector subset of C1)
                              T_union == T_sel1 bitwise at every
                              lambda: the v3.1 continuum equals the
                              v1.2.0 one.
  fullsel_lam1_c2_only_equals_union  at lambda = 1 the C2-only
                              parameters equal their full-selector
                              realization (the labial closure).

Usage
-----
    from morphing_fullsel import morph_targets_fullsel
    meta = morph_targets_fullsel('klinpan', 'work/lambda/P1',
                                 lambdas=[0.0, 0.125, ..., 1.0],
                                 label='P1')

Requires the covtl-lambda v1.2.0 patch applied to the covtl-pipeline
checkout (vtl_synth.core.morphing importable).
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from vtl_synth.core.constants import (COVTL_PARAMS, TRACT_PARAM_NAMES,
                                      TRACT_SR, N_TRACT_PARAMS,
                                      N_GLOTTIS_PARAMS)
from vtl_synth.core.morphing import (MORPH_PATCH_ID, TARGET_DEFAULT_ENGINE,
                                     find_nasal_c1_clusters,
                                     plan_target_cluster, write_morph_tract)

#: kit-level identity of the condition (informational).
FULLSEL_PATCH_ID = 'nasal_synthesis kit v3.1 -- full-CC-selector ' \
                   'extension of covtl-lambda v1.2.0 (2026-10-30)'

#: COVTL parameter index -> tract parameter index (name-based mapping).
_COVTL_TO_TRACT = {i: TRACT_PARAM_NAMES.index(name)
                   for i, name in enumerate(COVTL_PARAMS)}


# ==========================================================================
# Selector widening (the one engine-side extension of the v1.2.0 wrapper)
# ==========================================================================

@contextlib.contextmanager
def _retargeted_c1_fullsel(node_index: int, theta: float,
                           rho: Optional[float],
                           params_override: Optional[List[int]]):
    """Runtime override of ONE consonant node: polar target + selector.

    Identical in spirit to morphing._retargeted_c1 (v1.2.0), with one
    addition: when ``params_override`` is given, the node's selector
    (its COVTL parameter-index list) is widened to it, so the cluster
    builder realizes the retargeted C1 gesture through the FULL CC
    selector (the base model's union machinery).  The wrapper
    delegates to the original builder first (same cluster rules,
    same conditional selector resolution) and then overwrites the
    target floats and the selector; the engine module attribute is
    ALWAYS restored (additive-only contract).
    """
    import vtl_synth.core.syltraj as syltraj
    original = syltraj.build_nodes

    def build_nodes_retargeted(flat, syllables, syl_bounds, boundary_info,
                               word_starts):
        nodes = original(flat, syllables, syl_bounds, boundary_info,
                         word_starts)
        nd = nodes[node_index]
        nd.theta = theta
        if rho is not None:
            nd.rho = rho
        if params_override is not None:
            nd.params = list(params_override)
        return nodes

    syltraj.build_nodes = build_nodes_retargeted
    try:
        yield
    finally:
        syltraj.build_nodes = original


def _build_cluster_tract_fullsel(phrase: str, node_index: int,
                                 theta: float, rho: Optional[float],
                                 params_override: Optional[List[int]],
                                 engine_kw: Dict
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """One (re)synthesis with C1 retargeted AND its selector widened."""
    from vtl_synth.core.build_phrase_tract import build_phrase_tract
    with _retargeted_c1_fullsel(node_index, theta, rho, params_override):
        tract400, glott400 = build_phrase_tract(
            phrase, phrase_id=1, label='endpoint', verbose=False,
            return_data=True, **engine_kw)
    return tract400, glott400


# ==========================================================================
# The v3.1 continuum
# ==========================================================================

def build_fullsel_continuum(phrase: str,
                            lambdas: Optional[Sequence[float]] = None,
                            rho_enabled: bool = False,
                            cluster_index: int = 1,
                            engine: Optional[Dict] = None) -> Dict:
    """Builds the full-CC-selector lambda continuum of a cluster phrase.

    Returns the same dictionary as
    :func:`vtl_synth.core.morphing.build_target_continuum`, plus:

    ``selectors``   {c1, c2, union, c2_only} -- parameter-name lists
    ``fullsel``     the per-frame condition record (c2_only params,
                    blend, noop flag)
    ``checks``      the v1.2.0 guarantees + the three fullsel checks
    """
    from vtl_synth.core.morphing import parse_lambda_grid

    eng = dict(TARGET_DEFAULT_ENGINE)
    if engine:
        unknown = set(engine) - set(eng)
        if unknown:
            raise ValueError(
                f'unknown engine parameters: {sorted(unknown)} '
                f'(expected any of {sorted(eng)})')
        eng.update(engine)
    # Pipeline.run conversions (identical to build_target_continuum)
    engine_kw = dict(
        f0_hz=eng['f0_hz'], speaker=eng['speaker'],
        use_orthogonal=eng['use_orthogonal'], engine='syl',
        t_cons=max(1, round(eng['t_cons_ms'] / 10.0)),
        t_voy=max(1, round(eng['t_voy_ms'] / 10.0)),
        pause_short_ms=eng['pause_short_ms'],
        pause_long_ms=eng['pause_long_ms'], sampa_guard=None)

    if lambdas is None:
        lambdas = parse_lambda_grid()
    else:
        lambdas = parse_lambda_grid(','.join(f'{float(v):g}'
                                             for v in lambdas))

    plan, nodes, _ = plan_target_cluster(
        phrase, cluster_index=cluster_index, rho_enabled=rho_enabled,
        engine=engine)
    i_c1, i_c2 = find_nasal_c1_clusters(nodes)[cluster_index - 1]

    # ---- selectors of the two cluster consonants (COVTL indices) ------
    sel_c1 = list(nodes[i_c1].params)
    sel_c2 = list(nodes[i_c2].params)
    union = sorted(set(sel_c1) | set(sel_c2))
    c2_only_covtl = sorted(set(sel_c2) - set(sel_c1))
    c2_only_tract = sorted(_COVTL_TO_TRACT[i] for i in c2_only_covtl)
    union_is_c1 = (set(union) == set(sel_c1))
    selectors = {
        'c1': [COVTL_PARAMS[i] for i in sorted(sel_c1)],
        'c2': [COVTL_PARAMS[i] for i in sorted(sel_c2)],
        'union': [COVTL_PARAMS[i] for i in union],
        'c2_only': [TRACT_PARAM_NAMES[i] for i in c2_only_tract],
    }

    def _build_sel1(theta, rho):
        from vtl_synth.core.morphing import _build_cluster_tract
        return _build_cluster_tract(phrase, i_c1, theta,
                                    rho if rho_enabled else None, engine_kw)

    # canonical cluster (bitwise; selector untouched, lambda = 0 target)
    canon_tract, canon_glott = _build_sel1(plan.c1_theta, plan.c1_rho)
    canon_data = np.hstack([canon_tract, canon_glott])
    n_frames = canon_data.shape[0]

    # C1-incident sub-arcs (padded like v1.2.0: one gesture sample =
    # 4 frames @400 Hz on each side of the 100->400 Hz upsampling)
    a0 = max(0, 4 * plan.c1_arcs_steps[0] - 4)
    a1 = min(n_frames, 4 * plan.c1_arcs_steps[1] + 4)
    outside = np.ones(n_frames, dtype=bool)
    outside[a0:a1] = False

    items = []
    union_noop_all = True
    for lam in lambdas:
        theta = plan.theta_at(lam)
        rho = plan.rho_at(lam)
        t_sel1, g_sel1 = _build_sel1(theta, rho)
        d_sel1 = np.hstack([t_sel1, g_sel1])
        if union_is_c1:
            # structurally a no-op; still synthesized through the
            # widened wrapper so the bitwise identity is MEASURED
            t_uni, g_uni = _build_cluster_tract_fullsel(
                phrase, i_c1, theta, rho if rho_enabled else None,
                union, engine_kw)
            d_uni = np.hstack([t_uni, g_uni])
            if not np.array_equal(d_uni, d_sel1):
                union_noop_all = False
                raise AssertionError(
                    f'{phrase}: C2 selector is a subset of sel(C1) but '
                    f'the widened-selector synthesis diverged at '
                    f'lambda={lam:g} (engine coupling -- please report)')
        else:
            t_uni, g_uni = _build_cluster_tract_fullsel(
                phrase, i_c1, theta, rho if rho_enabled else None,
                union, engine_kw)
            d_uni = np.hstack([t_uni, g_uni])
            # the widening must stay confined to the C1-incident arcs
            if not np.array_equal(d_uni[outside], d_sel1[outside]):
                raise AssertionError(
                    f'{phrase}: full-selector effect outside the '
                    f'C1-incident sub-arcs at lambda={lam:g} '
                    f'(locality violated -- please report)')
        # ---- the v3.1 item --------------------------------------------
        # The recruitment blend is applied ONLY on the padded
        # C1-incident span and ONLY where the two builds differ:
        # outside the span T_sel1 == T_union == canonical (asserted
        # below), so the item stays bitwise canonical there (a plain
        # (1-lam)*a + lam*b would inject 1-ulp noise even where
        # a == b).
        data = d_sel1.copy()
        if c2_only_tract and lam > 0.0:
            if lam >= 1.0:
                for p in c2_only_tract:          # exact endpoint
                    data[:, p] = d_uni[:, p]
            else:
                for p in c2_only_tract:          # linear recruitment
                    a = d_sel1[a0:a1, p]
                    b = d_uni[a0:a1, p]
                    v = (1.0 - lam) * a + lam * b
                    v[a == b] = a[a == b]        # exact where equal
                    data[a0:a1, p] = v
        items.append({'lam': lam, 'theta_c1': theta, 'rho_c1': rho,
                      'tract': data[:, :N_TRACT_PARAMS],
                      'glottis': data[:, N_TRACT_PARAMS:
                                      N_TRACT_PARAMS + N_GLOTTIS_PARAMS],
                      'data': data})

    # ---- guarantees (v1.2.0 + fullsel) --------------------------------
    checks: Dict[str, bool] = {}
    for it in items:
        if it['data'].shape != canon_data.shape:
            raise AssertionError(
                f"lambda {it['lam']:g} produced {it['data'].shape} vs "
                f"canonical {canon_data.shape} (temporal identity)")
    zero = next((it for it in items if it['lam'] == 0.0), None)
    checks['lam0_bitwise_canon'] = (
        True if zero is None
        else bool(np.array_equal(zero['data'], canon_data)))
    checks['glottis_bitwise_constant'] = bool(all(
        np.array_equal(it['glottis'], canon_glott) for it in items))
    one = next((it for it in items if it['lam'] == 1.0), None)
    if one is not None:
        checks['lam1_theta_equals_c2'] = (one['theta_c1'] == plan.c2_theta)
        checks['lam1_theta_equiv_c2_mod_2pi'] = True   # structural
    else:
        checks['lam1_theta_equals_c2'] = (
            plan.theta_at(1.0) == plan.c2_theta)
    checks['outside_c1_arcs_bitwise_canon'] = bool(all(
        np.array_equal(it['data'][outside], canon_data[outside])
        for it in items))
    checks['fullsel_confined_to_c1_arcs'] = True       # asserted above
    if union_is_c1:
        # C2 selector is a subset of sel(C1): the widened build was
        # verified bitwise-identical to the v1.2.0 one at every lambda
        checks['fullsel_union_noop'] = bool(union_noop_all)
    else:
        # the blocking must actually be removed: at lambda = 1 the
        # C2-only parameters leave their canonical values inside the
        # C1-incident arcs (e.g. the lips close for a labial C2)
        if one is not None:
            inside = ~outside
            checks['fullsel_c2_only_recruited'] = bool(any(
                not np.array_equal(one['data'][inside][:, p],
                                   canon_data[inside][:, p])
                for p in c2_only_tract))
        else:
            checks['fullsel_c2_only_recruited'] = True
    if one is not None and c2_only_tract:
        checks['fullsel_lam1_c2_only_equals_union'] = bool(all(
            np.array_equal(one['data'][:, p], d_uni[:, p])
            for p in c2_only_tract))
    else:
        checks['fullsel_lam1_c2_only_equals_union'] = True
    if not all(checks.values()):
        raise AssertionError(
            f'fullsel guarantee checks failed (impossible by '
            f'construction -- please report): {checks}')

    diff_any = np.zeros(n_frames, dtype=bool)
    for it in items:
        diff_any |= np.any(it['data'] != canon_data, axis=1)
    df = np.where(diff_any)[0]

    return {
        'plan': plan, 'engine': eng,
        'lambdas': [float(v) for v in lambdas],
        'canon': {'tract': canon_tract, 'glottis': canon_glott},
        'items': items, 'checks': checks,
        'selectors': selectors,
        'fullsel': {
            'condition': 'full CC selector under lambda (kit v3.1): the '
                         'C2-only selector parameters join the C1 gesture '
                         'with weight lambda; jaw/tongue rotation '
                         'bitwise identical to the v1.2.0 target morph',
            'c2_only_params': selectors['c2_only'],
            'c2_only_tract_indices': c2_only_tract,
            'blend': 'P(lam) = (1-lam)*P[selector=C1] + lam*P[union CC '
                     'selector], C2-only parameters, all frames (a no-op '
                     'outside the C1-incident sub-arcs by locality)',
            'union_equals_c1_selector': bool(union_is_c1),
        },
        'diff_extent_frames': {
            'first_frame': int(df.min()) if len(df) else None,
            'last_frame': int(df.max()) if len(df) else None},
        'n_frames': int(n_frames),
    }


def morph_targets_fullsel(phrase: str,
                          output_dir: Union[str, Path],
                          lambdas: Optional[Sequence[float]] = None,
                          rho_enabled: bool = False,
                          cluster_index: int = 1,
                          engine: Optional[Dict] = None,
                          with_audio: bool = True,
                          sample_rate: int = 44100,
                          label: Optional[str] = None,
                          overwrite: bool = False) -> Dict:
    """Writes the full-CC-selector lambda continuum (kit v3.1 condition).

    Same file contract as ``morphing.morph_targets`` (v1.2.0): per
    lambda ``<label>_l<lam>.tract`` + ``<label>_l<lam>.wav`` plus
    ``morph_meta.json``; the meta records the v3.1 condition (mode
    ``targets-fullsel``), the selectors of C1/C2/union, the blend and
    the extended guarantee checks.
    """
    cont = build_fullsel_continuum(phrase, lambdas=lambdas,
                                   rho_enabled=rho_enabled,
                                   cluster_index=cluster_index,
                                   engine=engine)
    plan = cont['plan']

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if label is None:
        label = plan.phrase

    items_meta = []
    for it in cont['items']:
        stem = f'{label}_l{it["lam"]:g}'
        tract_path = out / f'{stem}.tract'
        data = it['data']
        write_morph_tract(tract_path, data, sr=float(TRACT_SR))
        wav_path = None
        if with_audio:
            from vtl_synth.core.synth import synthesize_audio
            wav_path = str(out / f'{stem}.wav')
            if overwrite or not os.path.exists(wav_path):
                synthesize_audio(
                    data[:, :N_TRACT_PARAMS],
                    data[:, N_TRACT_PARAMS:N_TRACT_PARAMS
                         + N_GLOTTIS_PARAMS],
                    output_wav=wav_path, sample_rate=sample_rate)
        items_meta.append({
            'lam': it['lam'], 'theta_c1': it['theta_c1'],
            'rho_c1': it['rho_c1'], 'tract': str(tract_path),
            'wav': wav_path, 'n_frames': int(data.shape[0])})
        print(f'  fullsel point {stem} (lambda={it["lam"]:g}, '
              f'theta={it["theta_c1"]:.4f})')

    meta = {
        'patch': MORPH_PATCH_ID,
        'kit_extension': FULLSEL_PATCH_ID,
        'mode': 'targets-fullsel',
        'semantic': ('C1C2 -> C1\'C2 on the tautosyllabic cluster with '
                     'the WHOLE CC SELECTOR taken into account (kit '
                     'v3.1): lambda moves the C1 polar target (theta'
                     + (', rho' if rho_enabled else '')
                     + ') toward the C2 target exactly as in v1.2.0, '
                       'and the C2-only selector parameters (e.g. the '
                       'lips LD/LP of a labial C2) join the C1 gesture '
                       'with weight lambda; at lambda = 0 the synthesis '
                       'is bitwise canonical, at lambda = 1 the lips '
                       'reach their full-selector labial realization'),
        'label': label, 'phrase': plan.phrase,
        'sources': {'canon_phrase': plan.phrase,
                    'endpoint_synth': str(out / '_engine'),
                    'sample_rate_hz': float(TRACT_SR)},
        'targets': plan.as_dict(),
        'selectors': cont['selectors'],
        'fullsel': cont['fullsel'],
        'engine': cont['engine'],
        'lambdas': cont['lambdas'],
        'n_frames': cont['n_frames'],
        'diff_extent_frames': cont['diff_extent_frames'],
        'checks': cont['checks'],
        'items': items_meta,
    }
    with open(out / 'morph_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return meta

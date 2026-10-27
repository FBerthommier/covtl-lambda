# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Tests of the lambda-morph option (covtl-lambda-patch v1.2.0).

Three layers:
  * pure-numpy tests (no VocalTractLab needed) -- interpolation
    identities, grids, alignment/glottis policies, .tract round-trip,
    morph_pair without audio, CLI parser wiring, and the pure target
    semantics of the v1.2.0 cluster morph (angle wrap, exact target
    endpoints, nasal-cluster detection, dotted-phrase rejection);
  * engine-core tests (syltraj available, no VTL wheel needed) --
    cluster planning on real phrases (targets + C1 windows);
  * VTL-gated tests (skipped when vocaltractlab-cython is missing) --
    target-continuum synthesis with its guarantees, WAV synthesis of
    the blend continuum and endpoint reproducibility.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vtl_synth.core.constants import (  # noqa: E402
    N_TRACT_PARAMS,
    N_GLOTTIS_PARAMS,
    TRACT_PARAM_NAMES,
)
from vtl_synth.core.morphing import (  # noqa: E402
    AlignmentReport,
    ContinuumResult,
    MorphAlignmentError,
    MorphError,
    MORPH_GLOTTIS_MODES,
    NASAL_C1_KEYS,
    build_continuum,
    build_target_continuum,
    check_alignment,
    confine_weight,
    find_nasal_c1_clusters,
    interp_target_rho,
    interp_target_theta,
    interpolate_tract,
    morph_pair,
    morph_targets,
    parse_lambda_grid,
    plan_target_cluster,
    read_tract_sequence,
    window_state,
    wrap_angle_delta,
    write_morph_tract,
)

try:
    import vocaltractlab_cython  # noqa: F401
    VTL_OK = True
except Exception:
    VTL_OK = False

requires_vtl = pytest.mark.skipif(
    not VTL_OK, reason='vocaltractlab-cython not available')


# --------------------------------------------------------------------------
# synthetic frame-aligned pair (deterministic)
# --------------------------------------------------------------------------

def _synth_pair(n_frames=120, seed=7):
    """(canon, assim) (T, 30) pair: identical glottis, tract differs
    on frames [30, 60) on the lip/jaw columns only."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.3, size=(n_frames, N_TRACT_PARAMS))
    # slow smoothing so the sequences look tract-like
    k = np.hanning(11); k /= k.sum()
    for j in range(N_TRACT_PARAMS):
        base[:, j] = np.convolve(np.pad(base[:, j], (5, 5), mode='edge'),
                                 k, mode='same')[5:-5]
    assim = base.copy()
    lp = TRACT_PARAM_NAMES.index('LP')
    ld = TRACT_PARAM_NAMES.index('LD')
    ja = TRACT_PARAM_NAMES.index('JA')
    assim[30:60, lp] += 0.528
    assim[30:60, ld] -= 0.487
    assim[30:60, ja] -= 0.671
    glottis = np.tile(np.array([[102.2, 500.0, 0.1, 0.1, 0.0, 0.0,
                                 1.0, 0.85, 0.0, 0.0, 0.0]]),
                      (n_frames, 1))
    canon = np.hstack([base, glottis])
    assim = np.hstack([assim, glottis])
    return canon, assim


# --------------------------------------------------------------------------
# lambda grids
# --------------------------------------------------------------------------

def test_grid_default_is_nine_uniform_steps():
    assert parse_lambda_grid() == [k / 8 for k in range(9)]


def test_grid_spec_parses_and_dedups():
    assert parse_lambda_grid('0, 0.25, 0.5 ,0.25, 1') == [0.0, 0.25, 0.5, 1.0]


def test_grid_n_steps():
    assert parse_lambda_grid(n_steps=5) == [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize('bad', ['1.5', '-0.1', '2', '0,abc'])
def test_grid_rejects_out_of_range_and_malformed(bad):
    with pytest.raises(MorphError):
        parse_lambda_grid(bad)


def test_grid_rejects_n_steps_below_two():
    with pytest.raises(MorphError):
        parse_lambda_grid(n_steps=1)


# --------------------------------------------------------------------------
# interpolation identities
# --------------------------------------------------------------------------

def test_interpolate_endpoints_are_bitwise():
    canon, assim = _synth_pair()
    assert np.array_equal(interpolate_tract(canon, assim, 0.0), canon)
    assert np.array_equal(interpolate_tract(canon, assim, 1.0), assim)


def test_interpolate_midpoint_is_the_arithmetic_mean():
    canon, assim = _synth_pair()
    mid = interpolate_tract(canon, assim, 0.5)
    assert np.allclose(mid, 0.5 * (canon + assim), rtol=0, atol=0)


def test_interpolate_rejects_shape_mismatch():
    canon, _ = _synth_pair()
    with pytest.raises(MorphAlignmentError):
        interpolate_tract(canon, canon[:-1], 0.5)


def test_interpolate_rejects_out_of_range_lambda():
    canon, assim = _synth_pair()
    with pytest.raises(MorphError):
        interpolate_tract(canon, assim, 1.0001)


# --------------------------------------------------------------------------
# alignment report / glottis policy
# --------------------------------------------------------------------------

def test_alignment_report_windows_and_glottis():
    canon, assim = _synth_pair()
    rep = check_alignment(canon, assim)
    assert isinstance(rep, AlignmentReport)
    assert rep.n_frames == canon.shape[0]
    assert rep.n_cols == canon.shape[1]
    assert rep.glottis_max_abs_diff == 0.0
    assert rep.tract_diff_first_frame == 30
    assert rep.tract_diff_last_frame == 59


def test_alignment_rejects_frame_count_mismatch():
    canon, _ = _synth_pair()
    with pytest.raises(MorphAlignmentError):
        check_alignment(canon, canon[:-2])


def test_common_glottis_policy_rejects_diverging_sources():
    canon, assim = _synth_pair()
    assim2 = assim.copy()
    assim2[:, N_TRACT_PARAMS] += 3.0          # F0 differs by 3 Hz
    with pytest.raises(MorphError, match='glottis'):
        build_continuum(canon, assim2, [0.0, 0.5, 1.0],
                        glottis_mode='common')


def test_interpolate_glottis_policy_blends():
    canon, assim = _synth_pair()
    assim2 = assim.copy()
    assim2[:, N_TRACT_PARAMS] += 8.0
    res = build_continuum(canon, assim2, [0.0, 0.5, 1.0],
                          glottis_mode='interpolate')
    assert res.glottis_mode == 'interpolate'
    mid = res.item_at(0.5).data
    assert mid[0, N_TRACT_PARAMS] == pytest.approx(
        canon[0, N_TRACT_PARAMS] + 4.0)


def test_continuum_records_endpoint_checks():
    canon, assim = _synth_pair()
    res = build_continuum(canon, assim, [0.0, 0.375, 1.0])
    assert isinstance(res, ContinuumResult)
    assert res.checks['endpoint_lam0_equals_canon'] is True
    assert res.checks['endpoint_lam1_equals_assim'] is True
    assert res.checks['glottis_noop_at_lam0'] is True
    assert [it.lam for it in res.items] == [0.0, 0.375, 1.0]


def test_continuum_rejects_unknown_glottis_mode():
    canon, assim = _synth_pair()
    with pytest.raises(MorphError, match='glottis_mode'):
        build_continuum(canon, assim, [0.5], glottis_mode='both')


def test_glottis_modes_constant_exposed():
    assert MORPH_GLOTTIS_MODES == ('common', 'interpolate')


# --------------------------------------------------------------------------
# .tract round-trip
# --------------------------------------------------------------------------

def test_tract_write_read_roundtrip(tmp_path):
    canon, _ = _synth_pair(n_frames=40)
    out = tmp_path / 'x.tract'
    write_morph_tract(out, canon)
    data, sr = read_tract_sequence(out)
    assert sr == 400.0
    assert data.shape == canon.shape
    # 6-decimal text quantization
    assert np.allclose(data, canon, atol=1e-6)
    header = out.read_text(encoding='utf-8').splitlines()[:8]
    assert header[0] == '# VTL tract sequence file'
    assert '# sample_rate: 400' in header
    assert f'# n_frames: {canon.shape[0]}' in header


# --------------------------------------------------------------------------
# window_state (articulatory ground truth)
# --------------------------------------------------------------------------

def test_window_state_mid_picks_central_frame():
    canon, _ = _synth_pair(n_frames=400)
    state = window_state(canon, 0.58, 0.68)      # -> frame 252
    assert state.shape == (N_TRACT_PARAMS,)
    assert np.array_equal(state, canon[252, :N_TRACT_PARAMS])


def test_window_state_mean_averages_window():
    canon, _ = _synth_pair(n_frames=400)
    state = window_state(canon, 0.10, 0.20, mode='mean')
    assert np.allclose(state, canon[40:80, :N_TRACT_PARAMS].mean(axis=0))


def test_window_state_rejects_unknown_mode():
    canon, _ = _synth_pair(n_frames=10)
    with pytest.raises(MorphError):
        window_state(canon, 0.0, 0.01, mode='median')


# --------------------------------------------------------------------------
# morph_pair driver (no audio -> no VTL needed)
# --------------------------------------------------------------------------

def _write_pair(tmp_path):
    canon, assim = _synth_pair(n_frames=90)
    ca = tmp_path / 'P1_canon.tract'
    asr = tmp_path / 'P1_assim.tract'
    write_morph_tract(ca, canon)
    write_morph_tract(asr, assim)
    # re-read: the driver works on the quantized file representation
    return (ca, asr, *read_tract_sequence(ca), *read_tract_sequence(asr))


def test_morph_pair_writes_continuum_and_meta(tmp_path):
    ca, asr, canon_q, _, assim_q, _ = _write_pair(tmp_path)
    out = tmp_path / 'morph'
    meta = morph_pair(ca, asr, out, lambdas=[0.0, 0.5, 1.0],
                      with_audio=False, label='P1')
    stems = {p.name for p in out.glob('P1_l*.tract')}
    assert stems == {'P1_l0.tract', 'P1_l0.5.tract', 'P1_l1.tract'}
    assert (out / 'morph_meta.json').exists()
    # lambda=0 item is bitwise the canon file content
    lam0, _ = read_tract_sequence(out / 'P1_l0.tract')
    lam1, _ = read_tract_sequence(out / 'P1_l1.tract')
    assert np.array_equal(lam0, canon_q)
    assert np.array_equal(lam1, assim_q)
    # meta content
    assert meta['label'] == 'P1'
    assert meta['lambdas'] == [0.0, 0.5, 1.0]
    assert meta['checks']['endpoint_lam0_equals_canon'] is True
    assert meta['alignment']['n_frames'] == 90
    assert len(meta['items']) == 3
    assert meta['items'][1]['wav'] is None
    disk = json.loads((out / 'morph_meta.json').read_text())
    assert disk['sources']['canon_sha256'] == meta['sources']['canon_sha256']


def test_morph_pair_rejects_conflicting_grid_args(tmp_path):
    ca, asr, *_ = _write_pair(tmp_path)
    with pytest.raises(MorphError, match='mutually exclusive'):
        morph_pair(ca, asr, tmp_path / 'm2',
                   lambdas=[0.0, 1.0], n_steps=3, with_audio=False)


def test_morph_pair_default_label_and_n_steps(tmp_path):
    ca, asr, *_ = _write_pair(tmp_path)
    meta = morph_pair(ca, asr, tmp_path / 'm3', with_audio=False)
    assert meta['label'] == 'P1_canon-vs-P1_assim'
    assert len(meta['lambdas']) == 9


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------

def test_cli_parser_has_morph():
    from vtl_synth.cli.main import build_parser
    ap = build_parser()
    sub = next(a for a in ap._actions
               if isinstance(a, argparse._SubParsersAction))
    assert 'morph' in sub.choices
    ns = ap.parse_args(['morph', 'a.tract', 'b.tract',
                        '-o', 'o', '--lambdas', '0,0.5,1'])
    assert ns.glottis == 'common'
    assert ns.lambdas == '0,0.5,1'
    assert ns.func.__module__.endswith('morph_cmd')


def test_cli_parser_targets_mode_flags():
    from vtl_synth.cli.main import build_parser
    ap = build_parser()
    # single-phrase invocation (b absent) + target flags
    ns = ap.parse_args(['morph', '--phonetic', '--targets', 'klinpan',
                        '-o', 'o', '--target-rho', '--cluster', '2'])
    assert ns.targets is True
    assert ns.b is None
    assert ns.target_rho is True
    assert ns.cluster == 2
    # blend mode keeps two positionals
    ns2 = ap.parse_args(['morph', 'klinpan', 'klimpan', '-o', 'o'])
    assert ns2.targets is False and ns2.b == 'klimpan'


@requires_vtl
def test_cli_morph_file_mode_end_to_end(tmp_path):
    from vtl_synth.cli.main import main
    ca, asr, *_ = _write_pair(tmp_path)
    out = tmp_path / 'cli_out'
    rc = main(['morph', str(ca), str(asr), '-o', str(out),
               '--lambdas', '0,0.5,1', '--label', 'E2E'])
    assert rc == 0
    wavs = {p.name for p in out.glob('E2E_l*.wav')}
    assert wavs == {'E2E_l0.wav', 'E2E_l0.5.wav', 'E2E_l1.wav'}


# --------------------------------------------------------------------------
# VTL-gated: audio synthesis determinism at the endpoints
# --------------------------------------------------------------------------

@requires_vtl
def test_endpoint_wav_reproducible(tmp_path):
    from vtl_synth.core.synth import synthesize_audio
    ca, asr, canon_q, _, assim_q, _ = _write_pair(tmp_path)
    out = tmp_path / 'wav'
    morph_pair(ca, asr, out, lambdas=[0.0, 1.0], with_audio=True,
               label='R')
    # reference synthesis from the same file content
    ref0 = tmp_path / 'ref0.wav'
    synthesize_audio(canon_q[:, :N_TRACT_PARAMS],
                     canon_q[:, N_TRACT_PARAMS:N_TRACT_PARAMS
                             + N_GLOTTIS_PARAMS],
                     output_wav=str(ref0))
    from scipy.io import wavfile
    _, a = wavfile.read(out / 'R_l0.wav')
    _, b = wavfile.read(ref0)
    assert a.shape == b.shape
    assert np.array_equal(a, b)
    # the two endpoints of the continuum genuinely differ
    _, c = wavfile.read(out / 'R_l1.wav')
    assert not np.array_equal(a, c)


# --------------------------------------------------------------------------
# C1 confine (covtl-lambda-patch v1.1.0) -- causal locality of the C1C2
# design: only the C1 gesture window drifts with lambda; the context
# (C2 included) stays strictly canonical for every lambda.
# --------------------------------------------------------------------------

def test_confine_weight_shape_and_bounds():
    w = confine_weight(120, 30, 59, ramp=10)
    assert w.shape == (120,)
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert np.all(w[:30] == 0.0)
    assert np.all(w[60:] == 0.0)
    # plateau at 1 in the interior (ramps occupy the window edges)
    assert np.all(w[40:50] == 1.0)
    # monotone up-ramp / down-ramp
    assert np.all(np.diff(w[30:40]) > 0)
    assert np.all(np.diff(w[50:60]) < 0)


def test_confine_weight_rejects_bad_window():
    with pytest.raises(MorphError):
        confine_weight(120, 60, 30)
    with pytest.raises(MorphError):
        confine_weight(120, -1, 30)
    with pytest.raises(MorphError):
        confine_weight(120, 30, 120)


def test_confine_weight_clamps_overlapping_ramps():
    # window of 4 frames, ramp 10 -> clamped, still valid weights
    w = confine_weight(120, 30, 33, ramp=10)
    assert w.shape == (120,)
    assert np.all((w >= 0.0) & (w <= 1.0))


def test_confined_continuum_holds_context_canonical():
    canon, assim = _synth_pair()
    res = build_continuum(canon, assim, [0.0, 0.5, 1.0],
                          confine=(30, 59), ramp=10)
    # every lambda: outside the window, bitwise canonical (C2 fixed)
    w = confine_weight(canon.shape[0], 30, 59, 10)
    outside = w <= 0.0
    for it in res.items:
        assert np.array_equal(it.data[outside], canon[outside])
    # lambda = 0: whole sequence canonical
    assert np.array_equal(res.item_at(0.0).data, canon)
    # lambda = 1 inside the plateau: bitwise assimilated
    inside = w >= 1.0
    assert np.array_equal(res.item_at(1.0).data[inside],
                          assim[inside])
    # recorded guarantees
    assert res.checks['endpoint_lam0_equals_canon']
    assert res.checks['confine_lam1_equals_assim_inside']
    assert res.checks['outside_confine_equals_canon']
    # lam1 is NOT the plain assimilation any more (context held canon)
    assert not np.array_equal(res.item_at(1.0).data, assim)


def test_confined_continuum_ramp0_is_a_hard_window():
    canon, assim = _synth_pair()
    res = build_continuum(canon, assim, [1.0], confine=(30, 59), ramp=0)
    assert np.array_equal(res.item_at(1.0).data[30:60], assim[30:60])
    assert np.array_equal(res.item_at(1.0).data[:30], canon[:30])
    assert np.array_equal(res.item_at(1.0).data[60:], canon[60:])


def test_unconfined_continuum_is_unchanged_legacy():
    canon, assim = _synth_pair()
    res = build_continuum(canon, assim, [0.0, 1.0])
    assert res.confine is None
    assert res.checks['endpoint_lam1_equals_assim']
    assert np.array_equal(res.item_at(1.0).data, assim)


def test_morph_pair_records_confine_in_meta(tmp_path):
    ca, asr, canon_q, _, assim_q, _ = _write_pair(tmp_path)
    meta = morph_pair(ca, asr, tmp_path / 'conf', lambdas=[0.0, 1.0],
                      with_audio=False, label='CF',
                      confine=(30, 59), ramp=10)
    assert meta['confine'] == {'first_frame': 30, 'last_frame': 59,
                               'ramp_frames': 10}
    assert meta['checks']['outside_confine_equals_canon']
    # meta of the legacy call carries confine None
    meta2 = morph_pair(ca, asr, tmp_path / 'free', lambdas=[0.0],
                       with_audio=False, label='FR')
    assert meta2['confine'] is None


# --------------------------------------------------------------------------
# Target-level morph (v1.2.0) -- the DEFAULT semantic: the tautosyllabic
# CC cluster C1C2 -> C1'C2, lambda moves the POLAR TARGET of the nasal
# C1 (theta always, rho optionally) toward the C2 target.  Pure-layer
# tests of the target arithmetic first; engine planning needs only the
# syllabic core (no VTL wheel); synthesis is VTL-gated further down.
# --------------------------------------------------------------------------

def test_wrap_angle_delta_shortest_path():
    assert wrap_angle_delta(0.0, np.pi) == pytest.approx(np.pi)
    # 270deg -> -15deg: +75deg (counter-clockwise), NOT -285deg
    d = wrap_angle_delta(3 * np.pi / 2, -np.pi / 12)
    assert d == pytest.approx(75 * np.pi / 180)
    # wrapping equivalence: same point +- 2pi
    assert wrap_angle_delta(0.0, 2 * np.pi) == pytest.approx(0.0)
    assert wrap_angle_delta(-np.pi, np.pi) == pytest.approx(0.0)


def test_interp_target_theta_exact_endpoints():
    th1, th2 = 4.71238898038469, 0.872664625997165   # n -> p (engine)
    # bitwise endpoint identity -- the v1.2.0 semantic contract
    assert interp_target_theta(th1, th2, 0.0) == th1
    assert interp_target_theta(th1, th2, 1.0) == th2
    # interior: shortest angular path from th1
    mid = interp_target_theta(th1, th2, 0.5)
    assert mid == pytest.approx(th1 + 0.5 * (th2 + 2 * np.pi - th1))
    # monotone along the interior of the shortest path; the lambda=1
    # point wraps back to th2 verbatim (same angle mod 2*pi as the
    # path limit th1 + delta)
    ths = [interp_target_theta(th1, th2, k / 8) for k in range(9)]
    assert all(b > a for a, b in zip(ths[1:-1], ths[2:-1]))
    limit = th1 + wrap_angle_delta(th1, th2)
    assert abs((limit - ths[-1] + np.pi) % (2 * np.pi) - np.pi) < 1e-12
    with pytest.raises(MorphError):
        interp_target_theta(th1, th2, 1.5)


def test_interp_target_rho_exact_endpoints():
    r1, r2 = 1.15, 1.06
    assert interp_target_rho(r1, r2, 0.0) == r1
    assert interp_target_rho(r1, r2, 1.0) == r2
    assert interp_target_rho(r1, r2, 0.25) == pytest.approx(
        0.75 * r1 + 0.25 * r2)
    with pytest.raises(MorphError):
        interp_target_rho(r1, r2, -0.1)


def test_nasal_c1_keys_are_engine_nasals():
    assert set(NASAL_C1_KEYS) >= {'n', 'm', 'N'}


def _nodes_of(phrase):
    from vtl_synth.core.syltraj import (
        build_flat, build_nodes,
    )
    flat, syllables, syl_bounds, boundary_info, word_starts = \
        build_flat(phrase)
    return build_nodes(flat, syllables, syl_bounds, boundary_info,
                       word_starts)


def test_find_nasal_clusters_on_real_phrases():
    # klinkan: onset /kl/ (no nasal C1) is skipped, coda n+k retained
    nodes = _nodes_of('klinkan')
    cands = find_nasal_c1_clusters(nodes)
    assert len(cands) == 1
    i, j = cands[0]
    assert (nodes[i].key, nodes[j].key) == ('n', 'k')
    # the cluster pairs of the Pouw frames: the single n+CC cluster
    expected = {'klinpan': 'p', 'minbot': 'b', 'Tinpakit': 'p',
                'klintan': 't', 'mingot': 'g'}
    for phrase, c2key in expected.items():
        nds = _nodes_of(phrase)
        cands = find_nasal_c1_clusters(nds)
        assert len(cands) == 1, phrase
        assert (nds[cands[0][0]].key, nds[cands[0][1]].key) == \
            ('n', c2key), phrase


def test_dotted_phrases_are_not_clusters():
    # C.C splits are out of scope: the syllable dot disables the
    # cluster flag, so no nasal-initial CC cluster is found
    assert find_nasal_c1_clusters(_nodes_of('klin.pan')) == []
    assert find_nasal_c1_clusters(_nodes_of('min.bot')) == []


def test_plan_target_cluster_semantics():
    plan, nodes, _ = plan_target_cluster('klinpan')
    assert plan.c1_key == 'n' and plan.c2_key == 'p'
    # engine cluster targets (cluster rules applied): n (1.15, 270deg)
    assert plan.c1_rho == pytest.approx(1.15)
    assert plan.c1_theta == pytest.approx(3 * np.pi / 2)
    assert plan.c2_rho == pytest.approx(1.06)
    assert plan.c2_theta == pytest.approx(np.radians(50.0))
    # theta path is the shortest one: +140deg, not -220deg
    assert plan.theta_delta == pytest.approx(np.radians(140.0))
    # C1 phoneme window (100 Hz steps) and its 400 Hz frames -- the
    # reference C1 window of the study (232-272 family)
    assert plan.c1_window_steps == (58, 68)
    assert plan.as_dict()['c1_window_frames'] == \
        {'first': 232, 'last': 271}
    # C1-incident arcs: arrival arc + departure arc C1->C2
    assert plan.c1_arcs_steps == (58, 78)
    assert plan.as_dict()['c1_arcs_frames'] == \
        {'first': 232, 'last': 311}


def test_plan_target_cluster_rejects_dotted_phrase():
    with pytest.raises(MorphError, match='tautosyllabic'):
        plan_target_cluster('klin.pan')


def test_plan_target_cluster_rejects_clusterless_phrase():
    with pytest.raises(MorphError, match='no tautosyllabic'):
        plan_target_cluster('klimaka')      # no nasal coda cluster


def test_plan_target_cluster_engine_override_moves_window():
    # a longer consonantal gesture moves the C1 window in lockstep
    plan, _, _ = plan_target_cluster('klinpan',
                                     engine={'t_cons_ms': 160.0})
    assert plan.c1_window_steps == (76, 92)      # T_cons 10 -> 16


# --------------------------------------------------------------------------
# Target continuum synthesis (VTL-gated): guarantees of the DEFAULT
# v1.2.0 semantic on a real cluster phrase.
# --------------------------------------------------------------------------

@requires_vtl
def test_target_continuum_guarantees():
    cont = build_target_continuum('klinpan', lambdas=[0.0, 0.5, 1.0])
    plan = cont['plan']
    canon = cont['canon']
    assert cont['n_frames'] == canon['tract'].shape[0]
    # all checks recorded True (asserted inside, re-verified here)
    assert all(cont['checks'].values())
    assert cont['checks']['lam0_bitwise_canon']
    assert cont['checks']['lam1_theta_equals_c2']
    assert cont['checks']['glottis_bitwise_constant']
    assert cont['checks']['outside_c1_arcs_bitwise_canon']
    # lambda = 0 item IS the canonical cluster, bitwise
    zero = cont['items'][0]
    assert np.array_equal(zero['tract'], canon['tract'])
    assert zero['theta_c1'] == plan.c1_theta
    # lambda = 1: the C1 polar target EQUALS the C2 target coordinates
    one = cont['items'][-1]
    assert one['theta_c1'] == plan.c2_theta
    assert one['rho_c1'] == plan.c1_rho      # rho untouched by default
    # glottis block bitwise constant along the whole continuum
    for it in cont['items']:
        assert np.array_equal(it['glottis'], canon['glottis'])
    # measured footprint stays inside the padded C1-incident arcs
    de = cont['diff_extent_frames']
    assert de['first_frame'] >= 4 * plan.c1_arcs_steps[0] - 4
    assert de['last_frame'] < 4 * plan.c1_arcs_steps[1] + 4


@requires_vtl
def test_target_continuum_rho_channel():
    cont = build_target_continuum('klinkan', lambdas=[0.0, 1.0],
                                  rho_enabled=True)
    plan = cont['plan']
    one = cont['items'][-1]
    assert cont['checks']['lam1_rho_equals_c2']
    assert one['rho_c1'] == plan.c2_rho
    assert one['theta_c1'] == plan.c2_theta


@requires_vtl
def test_morph_targets_writes_meta_and_items(tmp_path):
    meta = morph_targets('klinpan', tmp_path / 'cont',
                         lambdas=[0.0, 0.5, 1.0], with_audio=False,
                         label='P1')
    assert meta['mode'] == 'targets'
    assert meta['patch'].startswith('covtl-lambda-patch v1.2.0')
    stems = {p.name for p in (tmp_path / 'cont').glob('P1_l*.tract')}
    assert stems == {'P1_l0.tract', 'P1_l0.5.tract', 'P1_l1.tract'}
    # per-item lambda-interpolated targets recorded
    t = meta['targets']
    assert t['c1']['key'] == 'n' and t['c2']['key'] == 'p'
    thetas = [it['theta_c1'] for it in meta['items']]
    assert thetas[0] == t['c1']['theta']
    assert thetas[-1] == t['c2']['theta']
    assert thetas[1] == pytest.approx(
        t['c1']['theta'] + 0.5 * t['theta_delta'])
    # windows + checks recorded; disk copy matches
    assert t['c1_window_frames']['first'] == 232
    assert meta['checks']['outside_c1_arcs_bitwise_canon']
    disk = json.loads((tmp_path / 'cont' / 'morph_meta.json')
                      .read_text(encoding='utf-8'))
    assert disk['targets'] == meta['targets']
    assert disk['items'][0]['wav'] is None


@requires_vtl
def test_cli_morph_targets_end_to_end(tmp_path):
    from vtl_synth.cli.main import main
    out = tmp_path / 't_out'
    rc = main(['morph', '--phonetic', '--targets', 'klinkan',
               '-o', str(out), '--lambdas', '0,1', '--label', 'K3'])
    assert rc == 0
    wavs = {p.name for p in out.glob('K3_l*.wav')}
    assert wavs == {'K3_l0.wav', 'K3_l1.wav'}
    meta = json.loads((out / 'morph_meta.json').read_text(
        encoding='utf-8'))
    assert meta['mode'] == 'targets'
    assert meta['targets']['c2']['key'] == 'k'


def test_cli_morph_targets_rejects_dotted_phrase(tmp_path, capsys):
    from vtl_synth.cli.main import main
    rc = main(['morph', '--phonetic', '--targets', 'klin.pan',
               '-o', str(tmp_path / 'nope')])
    assert rc == 1
    assert 'tautosyllabic' in capsys.readouterr().out


def test_cli_morph_targets_rejects_two_inputs(tmp_path, capsys):
    from vtl_synth.cli.main import main
    rc = main(['morph', '--phonetic', '--targets', 'klinpan', 'klimpan',
               '-o', str(tmp_path / 'nope')])
    assert rc == 1
    assert 'single canonical cluster phrase' in capsys.readouterr().out

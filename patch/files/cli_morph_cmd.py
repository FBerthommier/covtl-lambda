# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""
morph_cmd.py
============
CLI wiring of the ``morph`` subcommand (covtl-lambda-patch v1.2.0).

``vtl-synth morph`` builds an articulatory lambda continuum.  Since
v1.2.0 the DEFAULT semantic is the **target-level morph** on the
tautosyllabic CC cluster: one canonical cluster phrase in, and lambda
moves the polar target of C1 (theta always, rho with --target-rho)
toward the target of C2 -- strictly ``C1C2 -> C1'C2``::

    # target semantic (default): one dotless cluster phrase
    vtl-synth morph --phonetic --targets klinpan -o out/P1 --label P1
    vtl-synth morph --phonetic --targets klinkan --target-rho -o out/K3

The tract-parameter blend between two frame-aligned utterances is
kept as an option (the motor-level realization of the same
transition)::

    # two existing cluster .tract files (400 Hz, 19 + 11 params)
    vtl-synth morph out/P1_canon.tract out/P1_assim.tract -o out/P1

    # two SAMPA cluster variants -- the engine synthesizes both
    # endpoints first (same speaker/timings), then blends them
    vtl-synth morph --phonetic klinpan klimpan -o out/P1 --label P1
    vtl-synth morph --phonetic klintan klimtan -o out/NV1 --label NV1

NOTATION (engine manual, authoritative): ``.`` = intra-phrase
syllable separator (no pause, C.C split -- a different gesture
structure); NO separator = tautosyllabic CC cluster; space = 200 ms
inter-word pause; ``|`` = 320 ms.  The lambda variation is defined
on the CC cluster ONLY: dotted forms (e.g. ``klin.pan``) are a
different synthesis and are rejected by ``--targets`` (and must not
be used as endpoints of the lambda continuum).

This module is installed additively by ``patch/apply_patch.py``
(a 4-line, assert-guarded insertion in ``cli/main.py``, CRLF-safe,
reversible); it never modifies existing subcommands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def add_morph_subparser(sub, add_engine_options) -> None:
    """Registers the 'morph' subcommand on an argparse subparsers object.

    Called from ``vtl_synth.cli.main.build_parser`` (insertion made by
    ``patch/apply_patch.py``)::

        from vtl_synth.cli.morph_cmd import add_morph_subparser
        add_morph_subparser(sub, _add_engine_options)
    """
    p = sub.add_parser(
        'morph',
        help='lambda continuum of a CC cluster (target semantic, '
             'default) or tract blend between two utterances '
             '(.tract + .wav per lambda)',
        description=(
            'Target-level morph (DEFAULT, v1.2.0): the lambda '
            'variation applies to the tautosyllabic CC cluster '
            'C1C2 -> C1\'C2 -- lambda moves the polar target of the '
            'nasal C1 (theta always, rho with --target-rho) toward '
            'the target of C2, and the cluster is re-synthesized at '
            'every lambda; C2 and the whole context stay bitwise '
            'canonical. Blend mode (two endpoint inputs): '
            'data(lambda) = (1-lambda)*A + lambda*B, frame by frame, '
            'between two frame-aligned 400 Hz tract sequences '
            '(canonical vs. assimilated cluster variant), optionally '
            'confined to the C1 window. Writes <label>_l<lam>.tract '
            '(+ .wav) and morph_meta.json into the output directory.'))
    p.add_argument(
        'a',
        help='target mode (--targets): the canonical CLUSTER phrase '
             '(engine SAMPA by default, dotless -- a "." separator is '
             'rejected; cf. --text). Blend mode: a .tract file or a '
             'phrase (same mode as B)')
    p.add_argument(
        'b', nargs='?', default=None,
        help='blend mode only: the second endpoint (.tract file or '
             'phrase); must be ABSENT in --targets mode (C2 is read '
             'from the cluster itself)')
    p.add_argument('-o', '--output-dir', default='morph_out',
                   help='output directory (default: ./morph_out)')
    p.add_argument('--label', default=None,
                   help='file stem of the continuum items (default: '
                        '<a>-vs-<b> in blend mode, the phrase in '
                        'target mode)')
    grid = p.add_mutually_exclusive_group()
    grid.add_argument(
        '--lambdas', default=None, metavar='LIST',
        help='comma-separated interpolation weights in [0,1], e.g. '
             '"0,0.25,0.5,0.75,1" (default: 9 uniform steps)')
    grid.add_argument(
        '--n-steps', type=int, default=None, metavar='N',
        help='uniform grid of N steps from 0 to 1 inclusive (>= 2)')
    p.add_argument(
        '--targets', action='store_true',
        help='target-level morph (DEFAULT semantic of the lambda '
             'variation, v1.2.0): takes ONE canonical cluster phrase '
             'and moves the C1 polar target toward the C2 target '
             '(theta mandatory; rho with --target-rho); at lambda=1 '
             'the C1 target EQUALS the C2 target coordinates. C.C '
             'inputs (syllable dot) are rejected')
    p.add_argument(
        '--target-rho', action='store_true',
        help='target mode: also interpolate the C1 rho toward the '
             'C2 rho (default: theta only -- the place channel)')
    p.add_argument(
        '--cluster', type=int, default=1, metavar='N',
        help='target mode: 1-based rank of the nasal-initial CC '
             'cluster when the phrase carries several (default 1)')
    p.add_argument(
        '--glottis', default='common',
        choices=['common', 'interpolate'],
        help="blend mode glottis policy: 'common' (default) requires "
             "identical glottis sequences (pure tract morph -- "
             "voicing, F0 and nasality held constant); 'interpolate' "
             "blends the glottis sequences as well. In target mode "
             "the glottis is bitwise constant by construction")
    p.add_argument(
        '--tol', type=float, default=1e-6, metavar='TOL',
        help="tolerance of the 'common' glottis identity check "
             '(default: 1e-6)')
    p.add_argument(
        '--confine', default=None, metavar='T1:T2',
        help='blend mode only: C1 gesture window in seconds (e.g. '
             '"0.56:0.69") -- the lambda blend is restricted to this '
             'window and every frame outside it stays at the '
             'canonical values for all lambdas (C2 / context '
             'strictly invariant, cf. the C1C2 causal-locality '
             'contract). Default: whole-sequence blend, which also '
             'transforms anticipatory and transition frames outside '
             'C1 (discouraged for C1C2 experiments). Meaningless in '
             'target mode (locality is structural there)')
    p.add_argument(
        '--confine-frames', default=None, metavar='F1:F2',
        help='same as --confine, in 400 Hz frame indices '
             '(e.g. "224:276")')
    p.add_argument(
        '--ramp', type=int, default=10, metavar='N',
        help='raised-cosine edge ramp of the confine window, in '
             'frames (default: 10 = 25 ms; keeps the output '
             'trajectory continuous at the window boundary)')
    p.add_argument('--no-audio', action='store_true',
                   help='write the .tract continuum only (skip the WAVs)')
    p.add_argument(
        '--text', action='store_true',
        help='non-.tract inputs are English text (g2p) instead of '
             'engine SAMPA; NOTE for the lambda variation prefer '
             'direct SAMPA cluster inputs (g2p output carries '
             'syllable dots, which --targets rejects; e.g. write '
             '"klinpan", not "clean pan")')
    # engine options (speaker, timings, f0, language, ...) apply to
    # the synthesis of the endpoints (target mode: the canonical
    # cluster and its lambda variants; blend phrase mode: both
    # endpoints) -- shared so that all items stay frame-aligned by
    # construction.
    add_engine_options(p)
    p.set_defaults(func=cmd_morph)


# --------------------------------------------------------------------------
# command
# --------------------------------------------------------------------------

def _is_tract_file(value: str) -> bool:
    return value.lower().endswith('.tract')


def _parse_confine(args):
    """Parses --confine / --confine-frames -> (frames or None, error)."""
    if args.confine is not None and args.confine_frames is not None:
        return None, ('--confine and --confine-frames are mutually '
                      'exclusive')
    try:
        if args.confine is not None:
            t0, t1 = args.confine.split(':')
            return (round(float(t0) * 400.0), round(float(t1) * 400.0)), \
                None
        if args.confine_frames is not None:
            f0, f1 = args.confine_frames.split(':')
            return (int(f0), int(f1)), None
    except ValueError as e:
        return None, (f'bad confine spec ({e}) -- expected "T1:T2" '
                      'seconds or "F1:F2" frames')
    return None, None


def _engine_overrides(args) -> Tuple[Optional[Dict], Optional[str]]:
    """Engine parameters for the target morph, resolved like the run
    subcommand (lang profile defaults when the CLI flags are unset)."""
    from vtl_synth.cli.main import _make_pipeline, _resolve_lang
    from vtl_synth.core import constants as _c
    kwargs = dict(
        f0_hz=args.f0,
        speaker=args.speaker,
        engine=args.engine,
        use_orthogonal=not args.no_ortho,
        t_cons_ms=args.tcons,
        t_voy_ms=args.tvoy,
        pause_short_ms=args.tpause,
        pause_long_ms=args.tpause_long,
    )
    if kwargs['engine'] != 'syl':
        return None, ("the target morph uses the syl syllabic engine "
                      "(cluster targets); --engine " + args.engine +
                      " is not supported here")
    if args.lang is not None or hasattr(_c, 'ACTIVE_LANG'):
        kwargs['lang'] = _resolve_lang(args)
    pipe = _make_pipeline(args)
    return dict(
        speaker=pipe.speaker,
        f0_hz=pipe.f0_hz,
        use_orthogonal=pipe.use_orthogonal,
        t_cons_ms=pipe.t_cons_ms,
        t_voy_ms=pipe.t_voy_ms,
        pause_short_ms=pipe.pause_short_ms,
        pause_long_ms=pipe.pause_long_ms,
    ), None


def cmd_morph(args) -> int:
    from vtl_synth.core.morphing import (
        MorphError,
        morph_pair,
        morph_targets,
        parse_lambda_grid,
    )

    a, b = args.a, args.b
    out_dir = Path(args.output_dir)
    label = args.label

    # ---- grid ----------------------------------------------------------
    if args.lambdas is not None:
        lambdas = parse_lambda_grid(spec=args.lambdas)
    elif args.n_steps is not None:
        lambdas = parse_lambda_grid(n_steps=args.n_steps)
    else:
        lambdas = None                      # driver default (9 steps)

    # ---- target mode (DEFAULT semantic, v1.2.0) ------------------------
    if args.targets:
        if b is not None:
            print('ERROR: --targets takes a single canonical cluster '
                  'phrase; C2 is read from the cluster itself (drop '
                  f'the second input {b!r})', flush=True)
            return 1
        if _is_tract_file(a):
            print('ERROR: --targets needs a cluster PHRASE (engine '
                  'SAMPA, dotless), not a .tract file', flush=True)
            return 1
        if args.confine is not None or args.confine_frames is not None:
            print('ERROR: --confine/--confine-frames are blend-mode '
                  'options; in target mode the locality is structural '
                  '(only the C1-incident sub-arcs move, C2 and the '
                  'context stay bitwise canonical)', flush=True)
            return 1
        if args.text:
            print('ERROR: --text (g2p) output carries syllable dots '
                  'and describes words, not clusters; give the cluster '
                  'in engine SAMPA (e.g. "klinpan")', flush=True)
            return 1
        engine, err = _engine_overrides(args)
        if err:
            print(f'ERROR: {err}', flush=True)
            return 1
        try:
            meta = morph_targets(
                a, output_dir=out_dir,
                lambdas=lambdas,
                rho_enabled=args.target_rho,
                cluster_index=args.cluster,
                engine=engine,
                with_audio=not args.no_audio,
                label=label)
        except MorphError as e:
            print(f'ERROR: {e}', flush=True)
            return 1
        _report_targets(meta)
        return 0

    # ---- blend mode (option; v1.0/v1.1 semantics) -----------------------
    if b is None:
        print('ERROR: blend mode needs two endpoints (A and B); for '
              'the default single-phrase target semantic add '
              '--targets', flush=True)
        return 1
    file_a, file_b = _is_tract_file(a), _is_tract_file(b)
    if file_a != file_b:
        print('ERROR: the two inputs must both be .tract files or both '
              'be phrases (got one of each)', flush=True)
        return 1

    if not file_a:
        # ---- phrase mode: synthesize both endpoints first -----------
        # Lazy import: cli.main imports this module at parser-build
        # time, so importing it back at module scope would be circular.
        from vtl_synth.cli.main import _make_pipeline
        pipe = _make_pipeline(args)
        endpoints = {}
        for tag, phrase in (('a', a), ('b', b)):
            res = pipe.run(phrase,
                           output_dir=str(out_dir / '_endpoints' / tag),
                           use_g2p=bool(args.text),
                           video=False,
                           label=f'endpoint_{tag}')
            endpoints[tag] = str(res.tract_path)
            print(f'[morph] endpoint {tag}: {phrase!r} -> '
                  f'{endpoints[tag]}', flush=True)
        canon_path, assim_path = endpoints['a'], endpoints['b']
        if label is None:
            label = f'{a}-vs-{b}'.replace(' ', '_')
    else:
        canon_path, assim_path = a, b

    confine, err = _parse_confine(args)
    if err:
        print(f'ERROR: {err}', flush=True)
        return 1

    try:
        meta = morph_pair(
            canon_path, assim_path,
            output_dir=out_dir,
            lambdas=lambdas,
            glottis_mode=args.glottis,
            glottis_tol=args.tol,
            with_audio=not args.no_audio,
            label=label,
            confine=confine,
            ramp=args.ramp,
        )
    except MorphError as e:
        print(f'ERROR: {e}', flush=True)
        return 1

    # ---- report -----------------------------------------------------------
    al = meta['alignment']
    print(f'[morph] mode      : blend (tract-parameter morph between '
          f'two endpoints)')
    print(f'[morph] label     : {meta["label"]}')
    print(f'[morph] frames    : {al["n_frames"]} @ 400 Hz '
          f'({al["n_cols"]} columns)')
    print(f'[morph] glottis   : mode={meta["glottis_mode"]}, '
          f'max|diff|={al["glottis_max_abs_diff"]:.3e}')
    if al['tract_diff_first_frame'] is not None:
        print(f'[morph] tract diff: frames '
              f'{al["tract_diff_first_frame"]}-'
              f'{al["tract_diff_last_frame"]} '
              f'(max|diff|={al["tract_max_abs_diff"]:.3f})')
    if meta.get('confine'):
        cf = meta['confine']
        print(f'[morph] confine   : frames {cf["first_frame"]}-'
              f'{cf["last_frame"]} (ramp {cf["ramp_frames"]}) -- '
              f'context held canonical (C2 invariant)')
    else:
        print('[morph] note      : unconfined blend -- frames outside '
              'C1 (anticipation, release, C2) are also transformed; '
              'use --confine T1:T2 for the C1C2 causal-locality design')
    for chk, ok in meta['checks'].items():
        print(f'[morph] check     : {chk} = {ok}')
    for it in meta['items']:
        print(f'[morph] lambda={it["lam"]:<g}  {it["tract"]}'
              + (f'  + {it["wav"]}' if it['wav'] else ''))
    print(f'[morph] meta      : '
          f'{out_dir / "morph_meta.json"}')
    return 0


def _report_targets(meta) -> None:
    """Prints the target-mode summary (C1C2 -> C1'C2 ground truth)."""
    t = meta['targets']
    print(f'[morph] mode      : targets (C1C2 -> C1\'C2 on the '
          f'tautosyllabic cluster -- DEFAULT semantic, v1.2.0)')
    print(f'[morph] label     : {meta["label"]}')
    print(f'[morph] frames    : {meta["n_frames"]} @ 400 Hz')
    print(f'[morph] cluster   : C1 {t["c1"]["key"]} '
          f'(rho {t["c1"]["rho"]:.3f}, theta '
          f'{t["c1"]["theta"]:.4f} rad) -> C2 {t["c2"]["key"]} '
          f'(rho {t["c2"]["rho"]:.3f}, theta {t["c2"]["theta"]:.4f} '
          f'rad); dtheta {t["theta_delta_deg"]:+.2f} deg'
          + (', rho interpolated' if t["rho_enabled"]
             else ', theta only'))
    w = t['c1_window_frames']
    aw = t['c1_arcs_frames']
    print(f'[morph] C1 window : frames {w["first"]}-{w["last"]} '
          f'(phoneme span); C1-incident arcs frames '
          f'{aw["first"]}-{aw["last"]}')
    de = meta['diff_extent_frames']
    if de['first_frame'] is not None:
        print(f'[morph] footprint : frames {de["first_frame"]}-'
              f'{de["last_frame"]} (measured, all lambdas vs canon)')
    for chk, ok in meta['checks'].items():
        print(f'[morph] check     : {chk} = {ok}')
    for it in meta['items']:
        print(f'[morph] lambda={it["lam"]:<g}  theta_C1='
              f'{it["theta_c1"]:+.6f} rad'
              + (f'  rho_C1={it["rho_c1"]:.6f}' if t["rho_enabled"]
                 else '')
              + f'  {it["tract"]}'
              + (f'  + {it["wav"]}' if it['wav'] else ''))
    print(f'[morph] meta      : '
          f'{Path(meta["items"][0]["tract"]).parent / "morph_meta.json"}'
          if meta['items'] else '[morph] (no items)')

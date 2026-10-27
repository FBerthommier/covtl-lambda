#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Assimilation continuum experiment (the "lambda experiment").

Reproduces the lambda continuum of the nasal assimilation study
(nasal_synthesis reproducibility kit, repro/lambda_experiment.py)
with the covtl-lambda-patch option instead of bespoke code -- under
the v1.2.0 DEFAULT (target-level) semantic: the lambda variation
applies to the tautosyllabic CC cluster, and lambda moves the polar
target of the nasal C1 toward the target of C2:

  P1  licensed    klinpan -> [m]-target before labial /p/  (C2 = p)
  NV1 unlicensed  klintan -> [m]-target before alveolar /t/ (C2 = t)

(cluster phrases, dotless: a "." separator is a C.C split -- a
different synthesis, out of scope of the lambda variation)

For each pair and each lambda in {0, .125, ..., 1} the cluster is
re-synthesized with the lambda-interpolated C1 polar target
(theta toward theta_C2; add --rho to also interpolate rho), and the
script writes <pair>_l<lam>.tract + .wav plus the articulatory
ground truth at the C1 target window (0.58-0.68 s, mid frame 0.63 s
@ 400 Hz for the P1 family).

Usage (from the covtl-pipeline root, patch applied):

    python examples/lambda_continuum.py [--out DIR] [--no-audio]
                                        [--plot] [--pairs P1 NV1]
                                        [--rho]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from vtl_synth.core.constants import TRACT_PARAM_NAMES
from vtl_synth.core.morphing import morph_targets, window_state

# cluster phrases of the lambda variation: (pair, canonical cluster
# SAMPA, C2 context, licensed?) -- the assimilated endpoint is NOT a
# second phrase: at lambda = 1 the C1 polar target has reached the C2
# target coordinates (C1'C2)
PAIRS = {
    'P1':  ('klinpan', 'p', True),    # before /p/ -- licensed
    'NV1': ('klintan', 't', False),   # before /t/ -- unlicensed
}

TARGET_WIN = (0.58, 0.68)     # s, C1 target window of the P1 family
SHOW_PARAMS = ('LP', 'LD', 'JA', 'VO')
LAMBDAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def main():
    ap = argparse.ArgumentParser(
        description='Assimilation continuum (lambda experiment) with '
                    'the covtl-lambda-patch option (target semantic).')
    ap.add_argument('--out', default='lambda_out', metavar='DIR')
    ap.add_argument('--no-audio', action='store_true',
                    help='write .tract files only')
    ap.add_argument('--plot', action='store_true',
                    help='also render continuum figure (matplotlib)')
    ap.add_argument('--pairs', nargs='+', default=['P1', 'NV1'],
                    metavar='NAME', choices=sorted(PAIRS))
    ap.add_argument('--rho', action='store_true',
                    help='also interpolate the C1 rho toward the C2 '
                         'rho (default: theta only)')
    args = ap.parse_args()

    out = Path(args.out)
    idx = {p: TRACT_PARAM_NAMES.index(p) for p in SHOW_PARAMS}

    tables = {}
    for name in args.pairs:
        sampa, c2, licensed = PAIRS[name]
        print(f'=== {name} ({"licensed" if licensed else "unlicensed"}):'
              f' {sampa}, C1 target -> C2 /{c2}/ target')
        meta = morph_targets(sampa, output_dir=str(out / name),
                             lambdas=LAMBDAS, rho_enabled=args.rho,
                             with_audio=not args.no_audio, label=name)

        t = meta['targets']
        print(f"cluster C1 {t['c1']['key']} (rho {t['c1']['rho']:.3f}, "
              f"theta {np.degrees(t['c1']['theta']):.1f} deg) -> C2 "
              f"{t['c2']['key']} (rho {t['c2']['rho']:.3f}, theta "
              f"{np.degrees(t['c2']['theta']):.1f} deg); "
              f"dtheta {t['theta_delta_deg']:+.2f} deg, "
              f"{'theta+rho' if t['rho_enabled'] else 'theta only'}")
        print(f"checks: {meta['checks']}")

        # articulatory ground truth along the continuum
        rows = []
        for it in meta['items']:
            data = np.loadtxt(it['tract'], comments='#')
            state = window_state(data, *TARGET_WIN)
            rows.append((it['lam'], state))
        tables[name] = rows
        header = 'lambda  ' + ''.join(f'{p:>9s}' for p in SHOW_PARAMS)
        print(header)
        for lam, state in rows:
            vals = ''.join(f'{state[idx[p]]:9.3f}' for p in SHOW_PARAMS)
            print(f'{lam:<g}{vals}')

    (out / 'articulators.json').write_text(json.dumps(
        {k: {'lam': [r[0] for r in v],
             'state': [[float(x) for x in r[1]] for r in v]}
         for k, v in tables.items()}, indent=1), encoding='utf-8')
    print(f'\narticulator table -> {out / "articulators.json"}')

    if args.plot:
        plot_continuum(out, tables, idx)

    # meta copy at the top level for provenance
    print(f"continua -> {', '.join(str(out / n) for n in args.pairs)}")


def plot_continuum(out, tables, idx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(LAMBDAS)))

    # (a) LP/LD time series across the continuum (pair P1)
    name = 'P1' if 'P1' in tables else next(iter(tables))
    ax = axes[0]
    t0, t1 = 0.30, 1.10
    for c, lam in zip(colors, LAMBDAS):
        data = np.loadtxt(out / name / f'{name}_l{lam:g}.tract',
                          comments='#')
        n = data.shape[0]
        tt = np.arange(n) / 400.0
        m = (tt >= t0) & (tt <= t1)
        ax.plot(tt[m], data[m, idx['LP']], color=c, lw=1.6)
        ax.plot(tt[m], data[m, idx['LD']], color=c, lw=1.6, ls='--')
    ax.axvspan(*TARGET_WIN, color='0.85', zorder=0)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('LP (solid), LD (dashed)  (cm)')
    ax.set_title(f'{name}: lip parameters across the continuum')

    # (b) articulator values at the target window vs lambda
    ax = axes[1]
    for name, rows in tables.items():
        for p, style in zip(SHOW_PARAMS, ['o-', 's-', '^-', 'd-']):
            ax.plot([r[0] for r in rows], [r[1][idx[p]] for r in rows],
                    style, ms=3.5, lw=1.2, alpha=0.8,
                    label=f'{p}, {name}')
    ax.set_xlabel('lambda')
    ax.set_ylabel('parameter value at t = 0.63 s')
    ax.set_title('Articulatory ground truth vs lambda')
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.savefig(out / 'continuum.png', dpi=300)
    print(f'figure -> {out / "continuum.png"}')


if __name__ == '__main__':
    main()

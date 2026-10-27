#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Figures of the lambda-morph manual (v1.2.0 target semantic).

Reads the output of ``examples/lambda_continuum.py`` (run it first::

    python examples/lambda_continuum.py --out lambda_out

then, from this directory::

    python make_figures.py --data <path-to-lambda_out>

) and renders the data figure of the manual (PNG, 300 dpi):
panel (a) the articulators that carry the place change (JA, TTX)
across the P1 target continuum, panel (b) articulatory ground truth
at the C1 target window vs lambda (JA, TTX, TTY; VO constant).
"""
import argparse
from pathlib import Path

import numpy as np

# project figure conventions: Paul Tol accents, perceptual lambda ramp
TOL = dict(blue='#4477AA', cyan='#66CCEE', green='#228833',
           yellow='#CCBB44', red='#EE6677', purple='#AA3377',
           grey='#BBBBBB')

TRACT = ('HX HY JX JA LP LD VS VO TCX TCY TTX TTY TBX TBY TRX TRY '
         'TS1 TS2 TS3').split()
IDX = {p: i for i, p in enumerate(TRACT)}

LAMBDAS_SHOW = [0.0, 0.25, 0.5, 0.75, 1.0]
TARGET_WIN = (0.58, 0.68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, metavar='DIR',
                    help='output directory of lambda_continuum.py')
    ap.add_argument('-o', '--output', default='fig_continuum_data.png')
    args = ap.parse_args()
    data_dir = Path(args.data)
    pair = 'P1'

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as fm
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7),
                             constrained_layout=True)
    lam_ramp = plt.cm.viridis(np.linspace(0.0, 0.95, len(LAMBDAS_SHOW)))

    # ---- (a) JA / TTX time series across the continuum -------------------
    # (the articulators that carry the C1 place change; the lips follow
    # the vowel background at mid-arc under the target semantic)
    ax = axes[0]
    t0, t1 = 0.15, 1.25
    for c, lam in zip(lam_ramp, LAMBDAS_SHOW):
        d = np.loadtxt(data_dir / pair / f'{pair}_l{lam:g}.tract',
                       comments='#')
        tt = np.arange(d.shape[0]) / 400.0
        m = (tt >= t0) & (tt <= t1)
        ax.plot(tt[m], d[m, IDX['JA']], color=c, lw=1.7,
                label=f'$\\lambda={lam:g}$')
        ax.plot(tt[m], d[m, IDX['TTX']], color=c, lw=1.7, ls='--')
    ax.axvspan(*TARGET_WIN, color='0.90', zorder=0)
    ax.text(0.63, 0.985, 'C1 window', ha='center', va='top',
            fontsize=7.5, color='0.25', transform=ax.get_xaxis_transform())
    ax.set_xlabel('time (s)')
    ax.set_ylabel('JA (solid, deg),   TTX (dashed, cm)')
    ax.text(0.012, 0.965, '(a)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower left',
              bbox_to_anchor=(0.0, 0.02), columnspacing=1.0,
              handlelength=1.6)
    ax.set_xlim(t0, t1)

    # ---- (b) ground truth at the target window vs lambda ------------------
    ax = axes[1]
    mid = int((TARGET_WIN[0] + TARGET_WIN[1]) / 2 * 400)   # frame 252
    series = {}
    for lam in [k / 8 for k in range(9)]:
        d = np.loadtxt(data_dir / pair / f'{pair}_l{lam:g}.tract',
                       comments='#')
        series[lam] = d[mid, :len(TRACT)]
    lams = sorted(series)
    for name, color, mk in (('JA', TOL['green'], '^'),
                            ('TTX', TOL['blue'], 'o'),
                            ('TTY', TOL['red'], 's')):
        ax.plot(lams, [series[l][IDX[name]] for l in lams],
                marker=mk, ms=4, lw=1.4, color=color, label=name)
    ax.axhline(series[0.0][IDX['VO']], color=TOL['grey'], lw=1.2,
               ls=':')
    ax.text(1.0, series[0.0][IDX['VO']] + 0.02, 'VO (const.)',
            ha='right', fontsize=7.5, color='0.3')
    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('parameter value at $t=0.63$ s')
    ax.text(0.012, 0.965, '(b)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')
    ax.legend(fontsize=8, frameon=False, ncol=3, loc='lower left',
              bbox_to_anchor=(0.0, 0.02), columnspacing=1.2)
    ax.set_xlim(-0.03, 1.03)

    fig.savefig(args.output, dpi=300)
    print('figure ->', args.output)


if __name__ == '__main__':
    main()

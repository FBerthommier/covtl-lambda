#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Minimal API demo of the lambda-morph option (v1.2.0 semantics).

Default run: the TARGET-level morph -- one canonical cluster phrase
("klinpan", dotless SAMPA: k-l-i-n-p-a-n with the tautosyllabic
n+p cluster), and lambda moves the polar target of the nasal C1
toward the target of C2 (/p/): at lambda = 1 the C1 target EQUALS
the C2 target coordinates (C1'C2, the place-assimilated cluster).

    python examples/morph_basic.py                      # target mode
    python examples/morph_basic.py --rho                # + rho channel
    python examples/morph_basic.py --blend a.tract b.tract

Run from a covtl-pipeline checkout with the patch applied.
"""
import argparse

from vtl_synth.core.morphing import morph_pair, morph_targets


def main():
    ap = argparse.ArgumentParser(
        description='Minimal lambda-morph demo (v1.2.0 target '
                    'semantic by default).')
    ap.add_argument('--rho', action='store_true',
                    help='target mode: also interpolate the C1 rho '
                         'toward the C2 rho')
    ap.add_argument('--blend', nargs=2, metavar=('A.tract', 'B.tract'),
                    default=None,
                    help='use the tract-blend option instead: two '
                         'frame-aligned cluster .tract endpoints')
    args = ap.parse_args()

    if args.blend:
        canon, assim = args.blend
        meta = morph_pair(
            canon, assim,
            output_dir='morph_basic/continuum',
            lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
            label='demo_blend')
    else:
        # target semantic (v1.2.0 default): ONE cluster phrase
        meta = morph_targets(
            'klinpan',
            output_dir='morph_basic/continuum',
            lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
            rho_enabled=args.rho,
            label='demo')

    print('\n--- morph_meta.json summary ---')
    print('patch    :', meta['patch'])
    print('mode     :', meta.get('mode', 'blend'))
    print('label    :', meta['label'])
    if 'targets' in meta:
        t = meta['targets']
        print(f"cluster  : C1 {t['c1']['key']} -> C2 {t['c2']['key']}"
              f"  (dtheta {t['theta_delta_deg']:+.2f} deg,"
              f" {'theta+rho' if t['rho_enabled'] else 'theta only'})")
        print('C1 window:', t['c1_window_frames'])
    print('lambdas  :', meta['lambdas'])
    if 'alignment' in meta:
        print('frames   :', meta['alignment']['n_frames'])
        print('glottis  : max|diff| =',
              meta['alignment']['glottis_max_abs_diff'])
    else:
        print('frames   :', meta['n_frames'])
    print('checks   :', meta['checks'])
    for it in meta['items']:
        line = f'  lambda={it["lam"]:<g}  {it["tract"]}  {it["wav"]}'
        if 'theta_c1' in it:
            line += f'  theta_C1={it["theta_c1"]:+.6f} rad'
        print(line)


if __name__ == '__main__':
    main()

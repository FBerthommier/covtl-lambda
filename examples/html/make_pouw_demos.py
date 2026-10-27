#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""make_pouw_demos.py — the six Pouw et al. (2024) reproduction
frames as interactive target-mode lambda HTML demos (v1.2.0).

Six sentence frames from the nasal_synthesis kit base (the same items
the kit's lambda075/lambda0625 reproduction experiments used):

  licensed   P1 clean pan   P2 mean boat   P4 thin packet
  unlicensed NV1 clean tan  NV3 clean can  NV4 mean goat

v1.2.0 semantic: the lambda variation applies to the TAUTOSYLLABIC
CC CLUSTER (dotless forms: klinpan, minbot, Tinpakit, klintan,
klinkan, mingot). For each frame the target continuum (9 lambda
steps) is synthesized with the patched engine -- lambda rotates the
polar target of the nasal C1 (theta always, rho fixed by default)
toward the FIXED C2 target; at lambda = 1 the C1 target EQUALS the
C2 theta (C1'C2). The C1 window is re-derived from the endpoint
.polar phoneme diff (canon n vs assim m), as in the previous
release, and cross-checked against morph_meta.json. A
self-contained player page is generated with the same conventions
as lambda_demo.html, including the ACTUAL per-lambda consonantal
branch rebuilt through the engine polar builder. Outputs:
pouw/index.html + pouw/pouw_<ID>.html. No engine, no network at run
time (file://).
"""

import base64
import json
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('LAMBDA_DEMOS_WORK', r'D:\covtl-lambda\pouw_work')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
#                                    covtl-lambda workspace
ENGINE_CWD = os.path.join(ROOT, 'covtl-pipeline')
PY = os.path.join(ROOT, 'venv-lambda', 'Scripts', 'python.exe')
ENGINE = os.path.join(ROOT, 'venv-lambda', 'Scripts', 'vtl-synth.exe')
FFMPEG = (os.path.join(ROOT, 'venv-lambda', 'Lib', 'site-packages',
                       'imageio_ffmpeg', 'binaries',
                       'ffmpeg-win-x86_64-v7.1.exe'))
LAMBDAS = ['0', '0.125', '0.25', '0.375', '0.5',
           '0.625', '0.75', '0.875', '1']
READOUT = ['LP', 'LD', 'JA', 'VO', 'TTX', 'TTY']
NAMES = {'LP': 'lip protrusion', 'LD': 'lip distance',
         'JA': 'jaw angle', 'VO': 'velum opening',
         'TTX': 'tongue-tip x', 'TTY': 'tongue-tip y'}

# v1.2.0: the six frames as TAUTOSYLLABIC CLUSTERS (dotless; a "."
# syllable separator is a C.C split -- different synthesis, out of
# scope of the lambda variation). (fid, sentence, licensing,
# canonical cluster, assimilated denomination)
FRAMES = [
    ('P1',  'clean pan',   'licensed',   'klinpan',  'klimpan'),
    ('P2',  'mean boat',   'licensed',   'minbot',   'mimbot'),
    ('P4',  'thin packet', 'licensed',   'Tinpakit', 'Timpakit'),
    ('NV1', 'clean tan',   'unlicensed', 'klintan',  'klimtan'),
    ('NV3', 'clean can',   'unlicensed', 'klinkan',  'klimkan'),
    ('NV4', 'mean goat',   'unlicensed', 'mingot',   'mimgot'),
]

sys.path.insert(0, HERE)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    assert r.returncode == 0, (cmd, r.stdout[-300:], r.stderr[-300:])
    return r


def synth_endpoint_polar(phrase, outdir):
    """Cluster endpoint .polar via the engine CLI (run --polar)."""
    label = phrase.replace('.', '_')
    run([ENGINE, 'run', '--polar', '--no-video', '--phonetic', phrase,
         '-l', 'en', '-o', outdir, '--label', label])
    return json.load(open(os.path.join(outdir, f'{label}.polar'),
                          encoding='utf-8'))


def c1_window_from_polars(canon, assim):
    """C1 nasal window in 400 Hz frames from the endpoint .polar
    phoneme diff (the differing n/m phoneme span) -- the CORRECTIONS
    v2.1 derivation, applied to the cluster endpoints."""
    dc, da = canon, assim
    assert len(dc['phonemes']) == len(da['phonemes'])
    diff = [(pc, pa) for pc, pa in zip(dc['phonemes'], da['phonemes'])
            if pc['key'] != pa['key']
            and {pc['key'], pa['key']} <= {'n', 'm'}]
    assert len(diff) == 1, diff
    pc = diff[0][0]
    return pc['t0'] * 4, pc['t1'] * 4 - 1, pc


def polar_lambda_branches(phrase, meta):
    """Actual per-lambda consonantal branches + the (lambda-invariant)
    vocalic branch, rebuilt through the engine polar builder with the
    retargeted C1 node (same technique as the synthesis itself)."""
    from vtl_synth.core.morphing import (
        _retargeted_c1, find_nasal_c1_clusters, plan_target_cluster,
    )
    from vtl_synth.video.polar_video import build_polar
    import vtl_synth.core.syltraj as syltraj

    plan, nodes, _ = plan_target_cluster(phrase)
    i_c1, _ = find_nasal_c1_clusters(nodes)[plan.cluster_index - 1]
    eng = meta['engine']
    kw = dict(t_cons=max(1, round(eng['t_cons_ms'] / 10.0)),
              t_voy=max(1, round(eng['t_voy_ms'] / 10.0)),
              pause_short_ms=eng['pause_short_ms'],
              pause_long_ms=eng['pause_long_ms'])

    def build(theta):
        with _retargeted_c1(i_c1, theta, None):
            return build_polar([plan.phrase], **kw)

    canon = build(plan.c1_theta)
    out = {}
    for it in meta['items']:
        d = build(it['theta_c1'])
        tr = d['traj_c']
        out[f"{it['lam']:g}"] = {
            'c_rho': [None if not np.isfinite(v) else round(float(v), 4)
                      for v in tr[:, 0]],
            'c_theta': [None if not np.isfinite(v) else round(float(v), 4)
                        for v in tr[:, 1]],
        }
    # sanity: the lambda=0 rebuild is the canonical branch
    tr0 = canon['traj_c']
    lam0 = out['0']
    assert lam0['c_rho'] == [
        None if not np.isfinite(v) else round(float(v), 4)
        for v in tr0[:, 0]]
    return canon, out


def build_frame(fid, sentence, lic, canon_phrase, assim_phrase):
    out = os.path.join(WORK, fid)
    os.makedirs(out, exist_ok=True)

    # 1. target-mode continuum (the v1.2.0 default semantic)
    run([ENGINE, 'morph', '--phonetic', '--targets', canon_phrase,
         '-o', out, '--label', fid])
    meta = json.load(open(os.path.join(out, 'morph_meta.json'),
                          encoding='utf-8'))
    assert meta['mode'] == 'targets'
    assert all(meta['checks'].values()), meta['checks']

    # 2. endpoint .polar files (cluster forms) for the window
    #    derivation and the display denominations
    pc = synth_endpoint_polar(canon_phrase, os.path.join(out, '_pol_c'))
    pa = synth_endpoint_polar(assim_phrase, os.path.join(out, '_pol_a'))
    f0, f1, nasal_p = c1_window_from_polars(pc, pa)
    t = meta['targets']
    assert (f0, f1) == (t['c1_window_frames']['first'],
                        t['c1_window_frames']['last']), \
        ((f0, f1), t['c1_window_frames'])
    print(f'[{fid}] {canon_phrase!r} C1 {t["c1"]["key"]} -> C2 '
          f'{t["c2"]["key"]}; C1 window {f0}-{f1}; arcs '
          f'{t["c1_arcs_frames"]["first"]}-'
          f'{t["c1_arcs_frames"]["last"]}')

    # 3. per-lambda payload
    canon_pol, lam_branches = polar_lambda_branches(canon_phrase, meta)
    from vtl_synth.core.constants import TRACT_PARAM_NAMES
    lambdas = []
    for lam in LAMBDAS:
        stem = f'{fid}_l{lam}'
        src = os.path.join(out, stem + '.wav')
        aac = os.path.join(out, stem + '.m4a')
        run([FFMPEG, '-y', '-i', src, '-c:a', 'aac', '-b:a', '96k',
             '-movflags', '+faststart', aac])
        b64 = base64.b64encode(open(aac, 'rb').read()).decode()
        w = wave.open(src, 'rb')
        dur = w.getnframes() / w.getframerate()
        w.close()
        data = np.loadtxt(os.path.join(out, stem + '.tract'),
                          comments='#')
        mid = (f0 + f1) // 2
        readout = {p: round(float(data[mid][TRACT_PARAM_NAMES.index(p)]), 3)
                   for p in READOUT}
        png = os.path.join(out, stem + '.png')
        run([PY, '-m', 'vtl_synth.cli.main', 'plot-tract',
             os.path.join(out, stem + '.tract'), '-o', png,
             '--params'] + READOUT, cwd=ENGINE_CWD)
        fig = base64.b64encode(open(png, 'rb').read()).decode()
        it_meta = next(i for i in meta['items'] if i['lam'] == float(lam))
        br = lam_branches[lam]
        lambdas.append({'lam': float(lam), 'aac_b64': b64,
                        'duration_s': round(dur, 3), 'readout': readout,
                        'fig_b64': fig,
                        'theta_c1': it_meta['theta_c1'],
                        'rho_c1': it_meta['rho_c1'],
                        'c_rho': br['c_rho'], 'c_theta': br['c_theta']})
        print(' ', stem, len(b64) // 1024, 'KB  fig',
              len(fig) // 1024, 'KB')

    def _nn(v):
        return None if (v is None or not np.isfinite(v)) else round(v, 4)

    def traj_raw(d):
        """build_polar dict (flat keys, NaN outside the clusters)."""
        return {
            'sr': d['sr'], 'n_steps': d['n_steps'],
            'traj': {
                'v_rho': [round(v, 4) for v in d['traj_v'][:, 0]],
                'v_theta': [round(v, 4) for v in d['traj_v'][:, 1]],
                'c_rho': [_nn(v) for v in d['traj_c'][:, 0]],
                'c_theta': [_nn(v) for v in d['traj_c'][:, 1]],
            },
            'phonemes': [{k: p[k] for k in ('key', 'category', 'rho',
                                            'theta', 't0', 't1')}
                         for p in d['phonemes']],
        }

    def traj_file(d):
        """Written .polar JSON (nested trajectory, None values)."""
        tr = d['trajectory']
        return {
            'sr': d['sample_rate'], 'n_steps': d['n_steps'],
            'traj': {
                'v_rho': [round(v, 4) for v in tr['vocalic']['rho']],
                'v_theta': [round(v, 4) for v in tr['vocalic']['theta']],
                'c_rho': [None if v is None else round(v, 4)
                          for v in tr['consonantal']['rho']],
                'c_theta': [None if v is None else round(v, 4)
                            for v in tr['consonantal']['theta']],
            },
            'phonemes': [{k: p[k] for k in ('key', 'category', 'rho',
                                            'theta', 't0', 't1')}
                         for p in d['phonemes']],
        }
    payload = {
        'fid': fid, 'sentence': sentence, 'licensed': lic == 'licensed',
        # dotless denominations: the tautosyllabic cluster pair
        'sampa': [canon_phrase, assim_phrase],
        'confine': {'first': f0, 'last': f1},
        'arcs': {'first': t['c1_arcs_frames']['first'],
                 'last': t['c1_arcs_frames']['last']},
        'readout_frame': (f0 + f1) // 2,
        'targets': {
            'c1': {'key': t['c1']['key'], 'rho': t['c1']['rho'],
                   'theta': t['c1']['theta']},
            'c2': {'key': t['c2']['key'], 'rho': t['c2']['rho'],
                   'theta': t['c2']['theta']},
            'theta_delta_deg': t['theta_delta_deg'],
            'rho_enabled': t['rho_enabled'],
        },
        'n_frames': meta['n_frames'],
        'readout_names': NAMES, 'lambdas': lambdas,
        'canon': traj_raw(canon_pol), 'assim': traj_file(pa),
    }
    return payload


PAGE = None  # TEMPLATE is imported from make_lambda_demo at run time


def render(payload):
    import make_lambda_demo
    html = make_lambda_demo.TEMPLATE.replace('__PAYLOAD__', json.dumps(
        payload, separators=(',', ':')))
    # frame-specific header / badge
    badge = ('<span style="background:#eef6ee;border:1px solid #9ec99e;'
             'border-radius:4px;padding:1px 8px;font-size:12.5px">'
             'licensed context</span>'
             if payload['licensed'] else
             '<span style="background:#fdeeee;border:1px solid #c99e9e;'
             'border-radius:4px;padding:1px 8px;font-size:12.5px">'
             'UNLICENSED context</span>')
    t = payload['targets']
    old_h1 = (
        '<h1>Lambda morph &mdash; <span class="phrase"><code>klinkan</code>\n'
        '  &rarr; C1&prime; at <code>/k/</code></span> (target semantic)</h1>')
    new_h1 = (
        f'<h1>{payload["fid"]} &mdash; {payload["sentence"]} '
        f'&mdash; <span class="phrase"><code>{payload["sampa"][0]}</code>'
        f' &rarr; <code>{payload["sampa"][1]}</code></span> {badge}</h1>')
    assert old_h1 in html
    html = html.replace(old_h1, new_h1)
    old_sub = (
        '  <div class="sub">covtl-lambda-patch v1.2.0 &middot; the '
        '&lambda;\n'
        '  variation acts on the tautosyllabic cluster <code>nk</code>:\n'
        '  &lambda; rotates the C1 polar target from /n/ (1.15, '
        '270&deg;)\n'
        '  toward the FIXED C2 /k/ target (1.10, &minus;15&deg;); at\n'
        '  &lambda;&nbsp;=&nbsp;1 the C1 target EQUALS the C2 &theta;\n'
        '  (C1&prime;C2) &middot; speaker JD3 &middot; C1 window frames\n'
        '  232&ndash;271 @400 Hz</div>')
    import math
    new_sub = (
        '  <div class="sub">covtl-lambda-patch v1.2.0 &middot; the '
        '&lambda; variation acts on the tautosyllabic cluster '
        f'<code>{t["c1"]["key"]}{t["c2"]["key"]}</code>: '
        '&lambda; rotates the C1 polar target from /'
        f'{t["c1"]["key"]}/ ({t["c1"]["rho"]:.2f}, '
        f'{math.degrees(t["c1"]["theta"]):.0f}&deg;) toward the FIXED '
        f'C2 /{t["c2"]["key"]}/ target ({t["c2"]["rho"]:.2f}, '
        f'{math.degrees(t["c2"]["theta"]):.0f}&deg;); at '
        '&lambda;&nbsp;=&nbsp;1 the C1 target EQUALS the C2 &theta; '
        '(C1&prime;C2) &middot; speaker JD3 &middot; C1 window frames '
        f'{payload["confine"]["first"]}&ndash;'
        f'{payload["confine"]["last"]} @400 Hz</div>')
    assert old_sub in html
    html = html.replace(old_sub, new_sub)
    # per-page <title> and provenance label (K3 defaults of the base page)
    old_title = ("<title>Lambda morph &mdash; klinkan &rarr; C1' at "
                 "/k/ (target semantic)</title>")
    assert old_title in html
    html = html.replace(
        old_title,
        f"<title>{payload['fid']} &mdash; {payload['sentence']} "
        '&mdash; lambda target morph</title>')
    html = html.replace('Target-mode K3 continuum',
                        f"Target-mode {payload['fid']} continuum")
    # C2 references in the fixed captions (key-dependent)
    html = html.replace('the C2 /k/ target, which never moves',
                        f'the C2 /{t["c2"]["key"]}/ target, which never '
                        'moves')
    html = html.replace(
        'the context consonant C2 /k/\n  and the anticipatory vowel',
        f'the context consonant C2 /{t["c2"]["key"]}/\n  and the '
        'anticipatory vowel')
    return html


def main() -> int:
    os.makedirs(WORK, exist_ok=True)
    outdir = os.path.join(HERE, 'pouw')
    os.makedirs(outdir, exist_ok=True)
    cards = []
    for fid, sentence, lic, canon_p, assim_p in FRAMES:
        payload = build_frame(fid, sentence, lic, canon_p, assim_p)
        page = os.path.join(outdir, f'pouw_{fid}.html')
        with open(page, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(render(payload))
        kb = os.path.getsize(page) // 1024
        print(f'  -> pouw/{os.path.basename(page)}: {kb} KB')
        cards.append((fid, sentence, lic, kb))
    # index
    rows = '\n'.join(
        f'<div class="card" style="width:min(360px,94vw)">'
        f'<h3 style="margin:0 0 4px">{fid} &mdash; {sentence} '
        f'({"licensed" if lic == "licensed" else "unlicensed"})</h3>'
        f'<p style="margin:0 0 8px;color:var(--muted);font-size:13px">'
        f'target-mode C1 continuum (C1C2 &rarr; C1&prime;C2), '
        f'9 lambda steps</p>'
        f'<a class="btn2" href="pouw_{fid}.html">Open the demo &rarr;</a>'
        f'</div>'
        for fid, sentence, lic, _ in cards)
    index = INDEX_HEAD + rows + INDEX_TAIL
    with open(os.path.join(outdir, 'index.html'), 'w',
              encoding='utf-8', newline='\n') as fh:
        fh.write(index)
    print('  -> pouw/index.html')
    return 0


INDEX_HEAD = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pouw et al. (2024) reproduction frames — target-mode lambda demos</title>
<style>
:root { --red:#d62728; --blue:#1f77b4; --ink:#1c1c1c; --muted:#6a6a6a;
        --line:#d9d9d9; --bg:#f4f4f2; --panel:#fff; }
body { font:15px/1.5 system-ui,"Segoe UI",Roboto,Arial,sans-serif;
       color:var(--ink); background:var(--bg);
       padding:22px clamp(14px,4vw,46px); }
header,main { max-width:1180px; margin:0 auto 16px; }
h1 { font-size:22px; margin:0 0 6px; }
.sub { color:var(--muted); font-size:13.5px; }
.panels { display:flex; flex-wrap:wrap; gap:18px; justify-content:center; }
.card { background:var(--panel); border:1px solid var(--line);
        border-radius:10px; padding:14px 16px;
        box-shadow:0 1px 2px rgba(0,0,0,.04); }
.btn2 { display:inline-block; background:#f0f0ee;
        border:1px solid var(--line); border-radius:7px;
        padding:6px 12px; color:#23527c; text-decoration:none;
        font-size:13.5px; }
.hint { max-width:1180px; margin:14px auto; font-size:12.5px;
        color:var(--muted); }
</style></head>
<body>
<header>
  <h1>Pouw et al. (2024) reproduction frames &mdash; target-mode
  lambda demos</h1>
  <div class="sub">six sentence frames of the nasal_synthesis kit
  (the same items as the kit's lambda075 / lambda0625 reproduction
  experiments) &middot; v1.2.0 semantic: the &lambda; variation acts
  on the tautosyllabic CC cluster (dotless forms
  <code>klinpan</code>, <code>minbot</code>, <code>Tinpakit</code>,
  <code>klintan</code>, <code>klinkan</code>, <code>mingot</code>)
  and moves the C1 polar target toward the fixed C2 target
  (C1C2 &rarr; C1&prime;C2); C2 and context bitwise canonical at
  every &lambda; &middot; speaker JD3</div>
</header>
<main><div class="panels">
"""

INDEX_TAIL = """
</div></main>
<p class="hint">Licensed contexts: the following consonant is
compatible with the assimilated place (pan, boat, packet).
Unlicensed: tan, can, goat &mdash; the assimilated form is not a
viable word. The target-mode design keeps every context frame
bitwise canonical at every lambda, so any model asymmetry between
the two groups would reflect the context, not the signal.</p>
</body></html>
"""


if __name__ == '__main__':
    raise SystemExit(main())

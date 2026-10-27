#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""make_lambda_demo.py — self-contained interactive HTML demo of the
target-mode lambda continuum (klinkan -> C1' pointing at /k/,
covtl-lambda-patch v1.2.0).

Reads the extracted payload (payload.json, built by the covtl-lambda
workspace extraction step) and emits a single self-contained page:
lambda slider (9 steps), audio per lambda, polar planning figure with
the CC control node rotating from the /n/ target toward the FIXED C2
/k/ target, per-lambda tract figure, articulator readout at the C1
window midpoint. No engine, no network: file://.

Semantic (v1.2.0): the lambda variation applies to the tautosyllabic
CC cluster (klinkan, dotless); lambda moves the polar TARGET of the
nasal C1 (theta always, rho with --target-rho) toward the C2 target.
At lambda = 1 the C1 target EQUALS the C2 target coordinates (C1'C2).
C2 and the whole context stay bitwise canonical at every lambda.

Visual conventions mirror the player template of the covtl-pipeline
HTML demos (colors, cards, transport box, GPL-3.0-or-later).
"""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAYLOAD = os.environ.get(
    'LAMBDA_PAYLOAD',
    r'D:\covtl-lambda\covtl-lambda-patch\examples\html\_payload'
    r'\payload.json')

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lambda morph &mdash; klinkan &rarr; C1' at /k/ (target semantic)</title>
<style>
:root { --red:#d62728; --blue:#1f77b4; --ink:#1c1c1c; --muted:#6a6a6a;
        --line:#d9d9d9; --bg:#f4f4f2; --panel:#fff; --amber:#b45309; }
* { box-sizing:border-box; }
html,body { margin:0; padding:0; }
body { font:15px/1.5 system-ui,"Segoe UI",Roboto,Arial,sans-serif;
       color:var(--ink); background:var(--bg);
       padding:22px clamp(14px,4vw,46px); }
header,main,.hint,footer { max-width:1180px; margin-left:auto;
                           margin-right:auto; }
h1 { font-size:22px; margin:0 0 4px; font-weight:650; }
.sub { color:var(--muted); font-size:13.5px; }
.sub code { font:12.5px ui-monospace,Consolas,monospace;
            background:#ececea; border-radius:3px; padding:1px 5px; }
main { margin-top:16px; }
.panels { display:flex; flex-wrap:wrap; gap:18px; align-items:stretch;
          justify-content:center; }
.card { background:var(--panel); border:1px solid var(--line);
        border-radius:10px; padding:14px 14px 10px;
        box-shadow:0 1px 2px rgba(0,0,0,.04); }
#polarWrap { width:min(520px,94vw); }
#tractWrap { width:min(470px,94vw); }
.card img, .card canvas { display:block; width:100%; }
.caption { font-size:12.5px; color:var(--muted); text-align:center;
           margin:2px 2px 4px; }
.transport { margin-top:18px; background:var(--panel);
             border:1px solid var(--line); border-radius:10px;
             padding:10px 16px; display:flex; flex-wrap:wrap;
             align-items:center; gap:12px; }
.transport button { font:600 13.5px/1 system-ui,sans-serif;
                    color:var(--ink); background:#f0f0ee;
                    border:1px solid var(--line); border-radius:7px;
                    padding:8px 14px; cursor:pointer; }
.transport button:hover { background:#e6e6e3; }
#lamSlider { flex:1 1 200px; }
#lamVal { font:13px/1 ui-monospace,Consolas,monospace; color:var(--muted);
          min-width:150px; text-align:right; }
.state { font:600 13.5px system-ui,sans-serif; color:var(--amber); }
.readout { margin-top:18px; background:var(--panel);
            border:1px solid var(--line); border-radius:10px;
            padding:10px 16px; font-size:13.5px; }
.readout table { border-collapse:collapse; width:100%; }
.readout td, .readout th { padding:4px 10px; text-align:left;
                           border-bottom:1px solid var(--line); }
.readout th { color:var(--muted); font-weight:600; }
.readout td.v { font:13px ui-monospace,Consolas,monospace; }
.note { margin-top:14px; background:#eef6ee; border:1px solid #9ec99e;
        border-radius:8px; padding:9px 14px; font-size:13px; }
.hint { margin:10px auto 0; font-size:12.5px; color:var(--muted); }
kbd { font:11.5px ui-monospace,Consolas,monospace; background:#ececea;
      border:1px solid #ccc; border-bottom-width:2px; border-radius:4px;
      padding:1px 5px; }
footer { margin:26px auto 0; padding-top:12px;
         border-top:1px solid var(--line); font-size:12.5px;
         color:var(--muted); }
footer p { margin:4px 0; }
@media (max-width:700px) {
  #polarWrap { width:74vw; } #tractWrap { width:68vw; }
}
</style>
</head>
<body>
<header>
  <h1>Lambda morph &mdash; <span class="phrase"><code>klinkan</code>
  &rarr; C1&prime; at <code>/k/</code></span> (target semantic)</h1>
  <div class="sub">covtl-lambda-patch v1.2.0 &middot; the &lambda;
  variation acts on the tautosyllabic cluster <code>nk</code>:
  &lambda; rotates the C1 polar target from /n/ (1.15, 270&deg;)
  toward the FIXED C2 /k/ target (1.10, &minus;15&deg;); at
  &lambda;&nbsp;=&nbsp;1 the C1 target EQUALS the C2 &theta;
  (C1&prime;C2) &middot; speaker JD3 &middot; C1 window frames
  232&ndash;271 @400 Hz</div>
</header>
<main>
  <div class="panels">
    <div class="card" id="polarWrap">
      <canvas id="polarCanvas"></canvas>
      <div class="caption">polar planning figure &mdash; the
      consonantal branch (blue) is the actual re-synthesized
      &lambda; trajectory; the CC control node rotates from the /n/
      target toward the C2 /k/ target, which never moves</div>
    </div>
    <div class="card" id="tractWrap">
      <img id="tractImg" alt="tract parameters at the C1 window
      midpoint">
      <div class="caption">tract parameters of the current
      &lambda; state (figure at the C1 midpoint, frame 252)</div>
    </div>
  </div>
  <div class="transport">
    <button id="btnPlay" aria-label="play / pause">&#9654;</button>
    <label style="font-size:13px;color:var(--muted)">&lambda;
      <input id="lamSlider" type="range" min="0" max="8" step="1"
             value="0" aria-label="lambda">
    </label>
    <span id="lamVal">&lambda; = 0.000</span>
    <span class="state" id="stateLbl">canonical /n/</span>
  </div>
  <div class="readout">
    <table id="roTable"><tbody></tbody></table>
  </div>
  <div class="note">C1C2 causal locality: every frame outside the
  C1-incident arcs is bitwise canonical at every &lambda; (verified
  per item by <code>outside_c1_arcs_bitwise_canon</code> in
  <code>morph_meta.json</code>) &mdash; the context consonant C2 /k/
  and the anticipatory vowel never move with &lambda;. The glottis
  block (voicing, F0, nasality) is bitwise constant along the whole
  continuum.</div>
  <p class="hint">Keys: <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> &lambda;
  &minus;/+0.125 &middot; <kbd>space</kbd> play / pause the current
  &lambda; state.</p>
</main>
<footer>
  <p id="provenance"></p>
  <p>Self-contained page (offline, no server, no network): audio AAC
  and tract figures embedded at build time from the target-mode K3
  continuum; polar trajectories rebuilt per &lambda; from the engine
  polar builder with the retargeted C1 node. Rendering conventions
  mirror the covtl-pipeline HTML demos (arcplot geometry, display
  K=10/&nu;=&minus;1 consonantal). Regenerate with
  <code>python examples/html/make_lambda_demo.py</code>.</p>
</footer>
<audio id="audio" preload="auto"></audio>
<script>
const P = __PAYLOAD__;
const audio = document.getElementById('audio');
const slider = document.getElementById('lamSlider');
const lamVal = document.getElementById('lamVal');
const stateLbl = document.getElementById('stateLbl');
const btnPlay = document.getElementById('btnPlay');
const tractImg = document.getElementById('tractImg');

function idx() { return +slider.value; }
function lam() { return P.lambdas[idx()].lam; }
function fmt(l) { return l.toFixed(3); }
function stateName(l) {
  if (l <= 0) return 'canonical /n/';
  if (l >= 1) return 'C1\\u2032 at C2 /' + P.targets.c2.key + '/ (assimilated)';
  return 'hybrid gesture (nasalized approximant)';
}

function refresh() {
  const it = P.lambdas[idx()];
  lamVal.textContent = '\\u03bb = ' + fmt(lam());
  stateLbl.textContent = stateName(lam());
  tractImg.src = 'data:image/png;base64,' + it.fig_b64;
  audio.src = 'data:audio/mp4;base64,' + it.aac_b64;
  drawPolar();
  renderReadout(it.readout);
}

// --- polar planning figure -------------------------------------
const cv = document.getElementById('polarCanvas');
const RED = '#d62728', BLUE = '#1f77b4';
function drawPolar() {
  const W = cv.clientWidth || 490, H = W;
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  cv.style.height = W + 'px';
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const R = 1.32, cx = W / 2, cy = H / 2, r = W / (2 * R) * 0.94;
  const X = (rho, th) => cx + rho * Math.cos(th) * r;
  const Y = (rho, th) => cy - rho * Math.sin(th) * r;
  // grid rings
  g.strokeStyle = '#e2e2df'; g.lineWidth = 1;
  for (let k = 1; k <= 4; k++) { g.beginPath();
    g.arc(cx, cy, r * k / 4, 0, 6.2832); g.stroke(); }
  function poly(pts, color, dash, width, alpha) {
    g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha;
    g.setLineDash(dash); g.beginPath();
    let pen = false;
    for (const p of pts) {
      if (p[0] == null) { pen = false; continue; }
      const x = X(p[0], p[1]), y = Y(p[0], p[1]);
      if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
    }
    g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
  }
  const C = P.canon.traj;
  // red: vocalic branch (identical at every lambda -- target of no vowel moves)
  poly(C.v_rho.map((v, i) => [v, C.v_theta[i]]), RED, [], 2.0, 0.9);
  // blue: consonantal branch of the CURRENT lambda (actual re-synthesis)
  const it = P.lambdas[idx()];
  poly(it.c_rho.map((v, i) => [v, it.c_theta[i]]), BLUE, [], 2.2, 1.0);
  // C1 canonical target (open circle) + moving control node
  const t1 = P.targets.c1, t2 = P.targets.c2;
  g.lineWidth = 1.4;
  g.strokeStyle = BLUE; g.fillStyle = '#fff';
  g.beginPath(); g.arc(X(t1.rho, t1.theta), Y(t1.rho, t1.theta), 5,
                       0, 6.2832); g.fill(); g.stroke();
  const rho = it.rho_c1, th = it.theta_c1;
  g.fillStyle = '#111';
  g.beginPath(); g.arc(X(rho, th), Y(rho, th), 7, 0, 6.2832); g.fill();
  g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke();
  g.fillStyle = '#333'; g.font = '12px system-ui';
  g.fillText(t1.key, X(t1.rho, t1.theta) - 16,
             Y(t1.rho, t1.theta) + 4);
  g.fillText('\\u03b8\\u2091 = ' +
             (th * 180 / Math.PI).toFixed(1) + '\\u00b0',
             X(rho, th) + 10, Y(rho, th) - 8);
  // fixed C2 target
  g.fillStyle = '#888';
  g.beginPath();
  g.arc(X(t2.rho, t2.theta), Y(t2.rho, t2.theta), 5, 0, 6.2832);
  g.fill();
  g.font = '12px system-ui';
  g.fillText(t2.key + ' (C2, fixed)', X(t2.rho, t2.theta) + 8,
             Y(t2.rho, t2.theta) - 6);
  g.fillStyle = '#6a6a6a'; g.font = '11px ui-monospace';
  g.fillText('0\\u00b0', cx + r * R - 18, cy - 4);
  g.fillText('90\\u00b0', cx - 10, cy - r * R + 14);
}

// --- readout ----------------------------------------------------
function renderReadout(ro) {
  const tb = document.querySelector('#roTable tbody');
  tb.innerHTML = '';
  for (const k of Object.keys(P.readout_names)) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<th>' + k + '</th><td>' + P.readout_names[k] +
      '</td><td class="v">' + ro[k].toFixed(3) + '</td>';
    tb.appendChild(tr);
  }
}

// --- controls ----------------------------------------------------
btnPlay.addEventListener('click', () => {
  if (audio.paused) { audio.play(); btnPlay.textContent = '\\u23F8'; }
  else { audio.pause(); btnPlay.textContent = '\\u25B6'; }
});
audio.addEventListener('ended', () => { btnPlay.textContent = '\\u25B6'; });
slider.addEventListener('input', refresh);
window.addEventListener('keydown', ev => {
  if (ev.key === 'ArrowLeft') { slider.value = idx() - 1; refresh(); }
  if (ev.key === 'ArrowRight') { slider.value = idx() + 1; refresh(); }
  if (ev.key === ' ' ) { ev.preventDefault(); btnPlay.click(); }
});
window.addEventListener('resize', drawPolar);
document.getElementById('provenance').textContent =
  'Target-mode K3 continuum (v1.2.0 semantic) \u00b7 ' +
  P.lambdas.length + ' lambda steps \u00b7 C1 window frames ' +
  P.confine.first + '-' + P.confine.last +
  ' \u00b7 C1-incident arcs frames ' + P.arcs.first + '-' +
  P.arcs.last + ' \u00b7 readout at the C1 midpoint (frame ' +
  P.readout_frame + ')';
refresh();
</script>
</body>
</html>
"""


def main() -> int:
    with open(PAYLOAD, encoding='utf-8') as fh:
        payload = json.load(fh)
    blob = json.dumps(payload, separators=(',', ':'))
    html = TEMPLATE.replace('__PAYLOAD__', blob)
    out = os.path.join(HERE, 'lambda_demo.html')
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(html)
    print(f'-> {out}: {os.path.getsize(out) // 1024} KB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

// ============================================================================
//  fig_memory.js — the memory-capacity paradigm (Fig. 4 of the paper), animated.
//
//  Everything drawn is computed here from one connectome with the JS port of the
//  engine (esn.js): the input, the reservoir states, the ridge readouts, their
//  held-out reconstructions and the forgetting curve. The last act repeats the
//  measurement on the Uniform null of the same connectome (same input, same Win).
//
//  build(W, { theme, label }) returns the SVG as a string.
// ============================================================================
'use strict';
const esn = require('./esn.js');
const { num, el, text, Loop, scale, polyline, dots, mix, hex3 } = require('./svg.js');

const THEMES = {
  light: {
    bg: '#ffffff', frame: '#d0d7de', ink: '#1f2328', ink2: '#59636e', muted: '#8c959f', grid: '#e7eaed',
    input: '#5f6368', res: '#3b63c4', out: '#149c7c', nul: '#d9730d',
    tintIn: '#f3f4f6', tintRes: '#ecf1fb', tintOut: '#e6f5f0',
    neg: '#3b63c4', mid: '#dfe3e7', pos: '#c0392b', edge: '#34435e',
  },
  dark: {
    bg: '#0d1117', frame: '#30363d', ink: '#e6edf3', ink2: '#9da7b3', muted: '#6e7681', grid: '#21262d',
    input: '#aab2bb', res: '#5b8def', out: '#20a884', nul: '#d47a20',
    tintIn: '#151a21', tintRes: '#101b2e', tintOut: '#0e231d',
    neg: '#5b8def', mid: '#3b434d', pos: '#ef6f5e', edge: '#c9d3e3',
  },
};

// ---------------------------------------------------------------------------
//  Network drawing coordinates. The atlas lists homologous regions in pairs
//  (2m, 2m+1), so the two hemispheres are laid out as mirror images: a
//  force-directed layout of the within-hemisphere weights, with the regions that
//  send more weight across pulled towards the midline. It is a drawing of the
//  network, not of the anatomy.
// ---------------------------------------------------------------------------
function hemisphereLayout(W, n) {
  const P = n / 2;
  const aff = new Float64Array(P * P);
  let amax = 0;
  for (let p = 0; p < P; p++) for (let q = 0; q < P; q++) {
    if (p === q) continue;
    const a = 0.5 * (W[(2 * p) * n + 2 * q] + W[(2 * p + 1) * n + 2 * q + 1]);
    aff[p * P + q] = a; if (a > amax) amax = a;
  }
  const medial = new Float64Array(P);
  for (let p = 0; p < P; p++) {
    let cross = 0, tot = 0;
    for (let j = 0; j < n; j++) {
      const wl = W[(2 * p) * n + j], wr = W[(2 * p + 1) * n + j];
      tot += wl + wr;
      if (j % 2 === 1) cross += wl; else cross += wr;
    }
    medial[p] = tot > 0 ? cross / tot : 0;
  }
  const x = new Float64Array(P), y = new Float64Array(P);
  for (let p = 0; p < P; p++) {
    const r = 0.42 * Math.sqrt((p + 0.5) / P), th = p * 2.399963229728653;
    x[p] = 0.5 + r * Math.cos(th); y[p] = 0.5 + r * Math.sin(th);
  }
  const k = Math.sqrt(1 / P), IT = 800;
  const fx = new Float64Array(P), fy = new Float64Array(P);
  for (let it = 0; it < IT; it++) {
    fx.fill(0); fy.fill(0);
    for (let p = 0; p < P; p++) for (let q = p + 1; q < P; q++) {
      let dx = x[p] - x[q], dy = y[p] - y[q];
      let d = Math.hypot(dx, dy); if (d < 1e-6) { dx = 1e-3; dy = 0; d = 1e-3; }
      let f = k * k / d;                                   // repulsion
      const a = aff[p * P + q];
      if (a > 0) f -= 1.6 * (d * d / k) * Math.pow(a / amax, 0.3);   // attraction along edges
      fx[p] += f * dx / d; fy[p] += f * dy / d; fx[q] -= f * dx / d; fy[q] -= f * dy / d;
    }
    const temp = 0.08 * (1 - it / IT) + 0.002;
    for (let p = 0; p < P; p++) {
      fx[p] += 0.9 * k * (medial[p] - 0.3) - 0.5 * (x[p] - 0.5);   // midline pull + weak gravity
      fy[p] += -0.5 * (y[p] - 0.5);
      const f = Math.hypot(fx[p], fy[p]);
      if (f > 0) { x[p] += fx[p] / f * Math.min(f, temp); y[p] += fy[p] / f * Math.min(f, temp); }
    }
  }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let p = 0; p < P; p++) { x0 = Math.min(x0, x[p]); x1 = Math.max(x1, x[p]); y0 = Math.min(y0, y[p]); y1 = Math.max(y1, y[p]); }
  // one scale for both axes; u = distance from the midline side, v centred on 0
  const sc = Math.max(x1 - x0, y1 - y0);
  return Array.from({ length: P }, (_, p) => ({ u: (x1 - x[p]) / sc, v: (y[p] - (y0 + y1) / 2) / sc }));
}

// ---------------------------------------------------------------------------
function build(W, opts = {}) {
  const n = 90;
  const C = THEMES[opts.theme || 'light'];
  const FONT = '-apple-system,BlinkMacSystemFont,\'Segoe UI\',\'Noto Sans\',Helvetica,Arial,sans-serif';

  // ---- the two measurements (same seed: same input, same Win) ---------------
  const Wu = esn.nullUniform(W, n);
  const nets = [
    { key: 'real', W, res: esn.memoryCapacity(W, {}, true), rs: esn.stableRank(W, n), col: C.out },
    { key: 'unif', W: Wu, res: esn.memoryCapacity(Wu, {}, true), rs: esn.stableRank(Wu, n), col: C.nul },
  ];
  const [REAL, UNIF] = nets;
  const TAU = REAL.res.params.tauMax;
  const u = REAL.res.u;
  const testStart = REAL.res.testStart;

  // ---- timeline (seconds) ---------------------------------------------------
  const RATE = 6;                       // reservoir steps shown per second
  const T1 = 5.5;                       // act 1: drive
  const durs = (tau) => (tau === 1 ? 2.4 : tau <= 8 ? 1.1 : 0.7);
  const sweepR = []; { let t = T1; for (let tau = 1; tau <= TAU; tau++) { sweepR.push([t, t + durs(tau)]); t += durs(tau); } }
  const T2 = sweepR[TAU - 1][1];        // act 2 ends: forgetting curve complete
  const T3 = T2 + 4.0;                  // act 3: memory capacity
  const TM = T3 + 1.6;                  // act 4: weights flattened, then a fast second sweep
  const DN = 0.25;
  const sweepN = Array.from({ length: TAU }, (_, i) => [TM + i * DN, TM + (i + 1) * DN]);
  const T4 = sweepN[TAU - 1][1];
  const TH = T4 + 3.4;                  // hold
  const L = Math.ceil((TH + 0.8) * RATE) / RATE;   // fade to the start; a whole number of steps
  const loop = new Loop(L);
  const SWITCH = T3 + 0.8;              // dynamics shown switch to the Uniform null

  // window of each (network, delay): the last delay stays on until the next act
  const win = (net, tau) => {
    const s = net === REAL ? sweepR : sweepN;
    const [a, b] = s[tau - 1];
    return [a, tau === TAU ? (net === REAL ? T3 + 0.3 : TH) : b];
  };
  // delay shown at time t (0 before the first sweep)
  const tauTrack = [];
  for (let tau = 1; tau <= TAU; tau++) tauTrack.push([sweepR[tau - 1][0], tau]);
  for (let tau = 1; tau <= TAU; tau++) tauTrack.push([sweepN[tau - 1][0], tau]);
  const activeAt = (t) => {             // -> { net, tau } or null
    if (t < T1) return null;
    if (t < T3 + 0.3) { for (let tau = TAU; tau >= 1; tau--) if (t >= sweepR[tau - 1][0]) return { net: REAL, tau }; }
    if (t < TM) return null;            // between the two measurements
    for (let tau = TAU; tau >= 1; tau--) if (t >= sweepN[tau - 1][0]) return { net: UNIF, tau };
    return null;
  };

  // ---- shown steps: display step k is simulation time base + k (held-out part) ----
  const WIN = 110;                      // steps kept before the first one shown (the widest strip)
  const base = testStart + WIN + TAU + 6;
  const K = Math.round(L * RATE);
  const netAt = (k) => (k / RATE < SWITCH ? REAL : UNIF);
  const sd = (net) => {                 // per-node standard deviation on the test partition
    if (net.sd) return net.sd;
    const X = net.res.X, T = net.res.params.steps, s = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let m = 0, v = 0;
      for (let t = testStart; t < T; t++) m += X[t * n + i];
      m /= (T - testStart);
      for (let t = testStart; t < T; t++) v += (X[t * n + i] - m) ** 2;
      s[i] = Math.sqrt(v / (T - testStart));
    }
    return (net.sd = s);
  };
  const zstate = (net, i, t) => net.res.X[t * n + i] / sd(net)[i];
  const readout = (net, tau, t) => {    // reconstruction of u(t - tau) from the state at time t
    const w = net.res.weights[tau - 1], X = net.res.X;
    let s = w[0];
    for (let i = 0; i < n; i++) s += w[i + 1] * X[t * n + i];
    return s;
  };
  const diverge = (z) => {              // signed value in [-1, 1] -> colour
    const c = Math.max(-1, Math.min(1, z));
    return hex3(c < 0 ? mix(C.mid, C.neg, -c) : mix(C.mid, C.pos, c));
  };

  // ---- canvas ---------------------------------------------------------------
  const Wd = 1040, Ht = 694;
  const out = [];
  const defs = [];
  const push = (s) => out.push(s);

  // ======================= header: steps and caption =========================
  const acts = [
    { name: 'Drive', win: [[0, T1]],
      cap: ['A scalar random input *u*(*t*) is injected into the 90 regions of one human connectome through fixed random weights **W**_{in}.',
        'The connectome is the reservoir: its weights **W** are rescaled to spectral radius ρ(**W**) = 0.99 and never trained.'] },
    { name: 'Read out the past', win: [[T1, T2]],
      cap: ['A linear readout, trained by ridge regression, reconstructs the input τ steps in the past from the present state **r**(*t*).',
        'Its accuracy on held-out data is the squared correlation *r*^{2}(τ) between the delayed input and its reconstruction.'] },
    { name: 'Memory capacity', win: [[T2, T3]],
      cap: ['*r*^{2}(τ) decays with the delay: the forgetting curve. Its sum is the memory capacity, MC = Σ_{τ} *r*^{2}(τ).',
        'MC measures how much of the past input stays linearly decodable from the present state of the connectome.'] },
    { name: 'Flatten the weights', win: [[T3, L]],
      cap: ['Same wiring, but the weights around every region are made uniform. The stable rank *r*_{s} of the network falls, and memory with it.',
        'The arrangement of the connection weights, not the wiring diagram alone, sets how much past input the network retains.'] },
  ];
  {
    const xs = [34, 214, 474, 716];
    let g = '';
    acts.forEach((a, i) => {
      const x = xs[i];
      g += el('g', { opacity: 0.38 },
        el('circle', { cx: x, cy: 27, r: 10.5, fill: 'none', stroke: C.ink2, 'stroke-width': 1.2 })
        + text(x, 31.5, String(i + 1), { size: 12.5, fill: C.ink2, anchor: 'middle', weight: 600 })
        + text(x + 18, 32, a.name, { size: 14.5, fill: C.ink2, weight: 600 })
        + loop.fade(a.win, 0.3, 0.38, 0));
      g += el('g', { opacity: 0 },
        el('circle', { cx: x, cy: 27, r: 10.5, fill: C.ink })
        + text(x, 31.5, String(i + 1), { size: 12.5, fill: C.bg, anchor: 'middle', weight: 700 })
        + text(x + 18, 32, a.name, { size: 14.5, fill: C.ink, weight: 700 })
        + loop.fade(a.win, 0.3));
      g += el('g', { opacity: 0 },
        text(24, 66, a.cap[0], { size: 14.2, fill: C.ink }) + text(24, 87, a.cap[1], { size: 14.2, fill: C.ink2 })
        + loop.fade(a.win, 0.4));
    });
    // progress line under the steps
    g += el('rect', { x: 24, y: 46, width: Wd - 48, height: 1, fill: C.grid });
    g += el('rect', { x: 24, y: 45.5, width: 0, height: 2, rx: 1, fill: C.ink2 },
      loop.linear('width', [[0, 0], [TH, Wd - 48], [L, Wd - 48]]));
    push(g);
  }

  // ======================= row A: the paradigm ===============================
  const A0 = 104, A1 = 404;                       // row top / bottom
  const boxIn = { x: 20, w: 196 }, boxRes = { x: 262, w: 516 }, boxOut = { x: 824, w: 196 };
  const stripY = 344, stripH = 38;                // trace strips along the bottom of the boxes
  const PXS = 4.6;                                // px per step in the strips
  const nodeIn = { x: 176, y: 236 }, nodeOut = { x: 864, y: 236 };

  for (const [b, fill] of [[boxIn, C.tintIn], [boxRes, C.tintRes], [boxOut, C.tintOut]])
    push(el('rect', { x: b.x, y: A0, width: b.w, height: A1 - A0, rx: 12, fill }));

  // ---- network geometry -------------------------------------------------------
  const lay = hemisphereLayout(W, n);
  const cx = 520, cy = 238, SZ = 172, gap = 64;
  const pos = new Array(n);
  for (let p = 0; p < n / 2; p++) {
    const { u: uu, v } = lay[p];
    pos[2 * p] = { x: cx - gap / 2 - uu * SZ, y: cy + v * SZ };
    pos[2 * p + 1] = { x: cx + gap / 2 + uu * SZ, y: cy + v * SZ };
  }
  const strength = new Float64Array(n);
  let smax = 0;
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += W[i * n + j]; strength[i] = s; smax = Math.max(smax, s); }
  const rad = (i) => 2.4 + 4.2 * Math.sqrt(strength[i] / smax);

  // fans: Win to every region, every region to the readout
  {
    let fin = '', fout = '';
    for (let i = 0; i < n; i++) {
      fin += `M${num(nodeIn.x)} ${num(nodeIn.y)}L${num(pos[i].x)} ${num(pos[i].y)}`;
      fout += `M${num(pos[i].x)} ${num(pos[i].y)}L${num(nodeOut.x)} ${num(nodeOut.y)}`;
    }
    const grad = (id, col, xa, xb, o) => defs.push(el('linearGradient', { id, gradientUnits: 'userSpaceOnUse', x1: xa, x2: xb, y1: 0, y2: 0 },
      el('stop', { offset: 0, 'stop-color': col, 'stop-opacity': o }) + el('stop', { offset: 0.55, 'stop-color': col, 'stop-opacity': o * 0.45 })
      + el('stop', { offset: 1, 'stop-color': col, 'stop-opacity': 0 })));
    grad('gin', C.input, nodeIn.x, cx - 40, 0.36);
    grad('gout0', C.muted, nodeOut.x, cx + 40, 0.36);
    grad('gout1', C.out, nodeOut.x, cx + 40, 0.5);
    grad('gout2', C.nul, nodeOut.x, cx + 40, 0.5);
    push(el('path', { d: fin, stroke: 'url(#gin)', 'stroke-width': 0.7, fill: 'none' }));
    push(el('path', { d: fout, stroke: 'url(#gout0)', 'stroke-width': 0.7, fill: 'none', 'stroke-dasharray': '3 3' },
      loop.fade([[T1, SWITCH - 0.4], [TM, L]], 0.5, 1, 0)));
    push(el('path', { d: fout, stroke: 'url(#gout1)', 'stroke-width': 0.7, opacity: 0, fill: 'none' }, loop.fade([[T1, SWITCH - 0.4]], 0.5)));
    push(el('path', { d: fout, stroke: 'url(#gout2)', 'stroke-width': 0.7, opacity: 0, fill: 'none' }, loop.fade([[TM, L]], 0.5)));
  }

  // edges: line width proportional to the weight in the rescaled network
  for (const net of nets) {
    const Wn = net.res.Wn;
    const order = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Wn[i * n + j] > 0) order.push([Wn[i * n + j], i, j]);
    order.sort((a, b) => a[0] - b[0]);
    let g = '';
    for (const [w, i, j] of order) {
      const a = pos[i], b = pos[j];
      let d = `M${num(a.x)} ${num(a.y)}L${num(b.x)} ${num(b.y)}`;
      if (i % 2 !== j % 2) {             // across hemispheres: an arc bowing away from the centre
        const my = (a.y + b.y) / 2, bend = (my >= cy ? 1 : -1) * 0.2 * Math.abs(b.x - a.x);
        d = `M${num(a.x)} ${num(a.y)}Q${num((a.x + b.x) / 2)} ${num(my + bend)} ${num(b.x)} ${num(b.y)}`;
      }
      g += el('path', { d, 'stroke-width': num(Math.max(0.35, 9.5 * w), 2), 'stroke-opacity': num(Math.min(0.92, 0.2 + 3.2 * w), 2) });
    }
    push(el('g', { stroke: C.edge, fill: 'none', 'stroke-linecap': 'round', opacity: net === REAL ? 1 : 0 },
      g + (net === REAL ? loop.fade([[0, T3 + 0.2]], 1.2) : loop.fade([[T3 + 0.2, L]], 1.2))));
  }

  // nodes: fill follows the state r_i(t) (z-scored per region), one keyframe per step
  {
    let g = '';
    for (let i = 0; i < n; i++) {
      const vals = [];
      for (let k = 0; k <= K; k++) vals.push(diverge(zstate(netAt(k), i, base + k) / 2.2));
      g += el('circle', { cx: num(pos[i].x), cy: num(pos[i].y), r: num(rad(i), 1), fill: vals[0] },
        el('animate', { attributeName: 'fill', dur: `${num(K / RATE, 4)}s`, repeatCount: 'indefinite', values: vals.join(';') }));
    }
    push(el('g', { stroke: C.tintRes, 'stroke-width': 0.9 }, g));
  }

  // input and output units
  {
    const vin = [], vout = [];
    for (let k = 0; k <= K; k++) {
      vin.push(diverge(u[base + k]));
      const a = activeAt(k / RATE);
      vout.push(a ? diverge(readout(a.net, a.tau, base + k)) : hex3(C.mid));
    }
    const unit = (c, vals, stroke) => el('circle', { cx: c.x, cy: c.y, r: 11, fill: vals[0], stroke, 'stroke-width': 1.6 },
      el('animate', { attributeName: 'fill', dur: `${num(K / RATE, 4)}s`, repeatCount: 'indefinite', values: vals.join(';') }));
    push(unit(nodeIn, vin, C.input));
    push(unit(nodeOut, vout, C.ink2));
  }

  // ---- box titles and formulas -----------------------------------------------
  push(text(boxIn.x + 14, A0 + 25, 'Input', { size: 14.5, fill: C.ink, weight: 700 }));
  push(text(boxIn.x + 14, A0 + 45, '*u*(*t*) ~ U(−1, 1), i.i.d.', { size: 12.6, fill: C.ink2 }));
  push(text(nodeIn.x, nodeIn.y - 22, '**W**_{in}', { size: 12.6, fill: C.ink2, anchor: 'middle' }));
  push(text(nodeIn.x, nodeIn.y + 32, 'fixed', { size: 11.5, fill: C.muted, anchor: 'middle' }));

  push(text(boxRes.x + 14, A0 + 25, 'Reservoir: a human connectome', { size: 14.5, fill: C.ink, weight: 700 }));
  push(text(boxRes.x + boxRes.w - 14, A0 + 25, '**r**(*t*+1) = tanh[ **W**_{in} *u*(*t*+1) + **W** **r**(*t*) ]', { size: 12.6, fill: C.ink2, anchor: 'end' }));
  push(el('g', { opacity: 1 }, text(boxRes.x + 14, A0 + 45, opts.label || '90 regions', { size: 12.2, fill: C.ink2, halo: C.tintRes }) + loop.fade([[0, T3 + 0.8]], 0.6)));
  push(el('g', { opacity: 0 }, el('circle', { cx: boxRes.x + 19, cy: A0 + 40.5, r: 4.2, fill: C.nul })
    + text(boxRes.x + 29, A0 + 45, 'same wiring, weights made uniform', { size: 12.2, fill: C.ink, halo: C.tintRes }) + loop.fade([[T3 + 0.8, L]], 0.6)));

  push(text(boxOut.x + 14, A0 + 25, 'Linear readout', { size: 14.5, fill: C.ink, weight: 700 }));
  push(text(boxOut.x + 14, A0 + 45, '*û*(*t*−τ) = **w**_{τ} · **r**(*t*)', { size: 12.6, fill: C.ink2 }));
  push(text(nodeOut.x, nodeOut.y - 22, '**w**_{τ}', { size: 12.6, fill: C.ink2, anchor: 'middle' }));
  push(el('g', { opacity: 1 }, text(nodeOut.x, nodeOut.y + 32, 'not trained yet', { size: 11.5, fill: C.muted, anchor: 'middle' }) + loop.fade([[0, T1]], 0.4)));
  push(el('g', { opacity: 0 }, text(nodeOut.x, nodeOut.y + 32, 'trained (ridge)', { size: 11.5, fill: C.muted, anchor: 'middle' }) + loop.fade([[T1, L]], 0.4)));

  // state colour key
  {
    const x = boxRes.x + boxRes.w - 118, y = A0 + 39;
    defs.push(el('linearGradient', { id: 'key' }, el('stop', { offset: 0, 'stop-color': C.neg }) + el('stop', { offset: 0.5, 'stop-color': C.mid }) + el('stop', { offset: 1, 'stop-color': C.pos })));
    push(text(x - 6, y + 7, 'state *r*_{*i*}(*t*)  −', { size: 11.5, fill: C.ink2, anchor: 'end' }));
    push(el('rect', { x, y, width: 86, height: 7, rx: 3.5, fill: 'url(#key)' }));
    push(text(x + 92, y + 7, '+', { size: 11.5, fill: C.ink2 }));
  }

  // stable rank of the network shown (act 4)
  {
    const xr = boxRes.x + boxRes.w - 14, y = stripY - 9;
    const lab = (net, w, o) => el('g', { opacity: 0 },
      text(xr, y, `stable rank *r*_{s} = ${net.rs.toFixed(2)}`, { size: o.size, fill: o.fill, anchor: 'end', weight: o.weight })
      + loop.fade(w, 0.5));
    push(lab(REAL, [[0, T3]], { size: 11.8, fill: C.ink2 }));
    push(lab(REAL, [[T3, T3 + 0.9]], { size: 13, fill: C.ink, weight: 700 }));
    push(lab(UNIF, [[T3 + 0.9, L]], { size: 13, fill: C.ink, weight: 700 }));
  }

  // ---- strips along the bottom of the boxes -----------------------------------
  const stripClip = (id, x0, x1) => defs.push(el('clipPath', { id }, el('rect', { x: x0, y: stripY - 4, width: x1 - x0, height: stripH + 8 })));
  const scroll = (xR, inner) => el('g', { transform: `translate(${xR} 0)` },
    inner + loop.translate([[0, xR, 0], [L, xR - L * RATE * PXS, 0]]));
  const lane = (y0, h) => (v) => y0 + h / 2 - v * h / 2;
  const kStart = (x0, x1) => -Math.ceil((x1 - x0) / PXS) - TAU - 3;

  // input strip
  const inX0 = boxIn.x + 12, inX1 = boxIn.x + boxIn.w - 12;
  stripClip('cin', inX0, inX1);
  {
    const yl = lane(stripY, stripH);
    const pts = []; for (let k = kStart(inX0, inX1); k <= K + 1; k++) pts.push([k * PXS, yl(u[base + k] * 0.92)]);
    defs.push(el('path', { id: 'uin', d: polyline(pts), fill: 'none', 'stroke-linejoin': 'round' }));
    push(el('g', { 'clip-path': 'url(#cin)' }, scroll(inX1, el('use', { href: '#uin', stroke: C.input, 'stroke-width': 1.3 }))));
    push(text(inX0, stripY - 9, '*u*(*t*)', { size: 11.8, fill: C.ink2 }));
    // the sample the readout is asked for: tau steps before the present
    const mk = el('g', { opacity: 0 },
      el('line', { x1: 0, y1: stripY - 3, x2: 0, y2: stripY + stripH + 3, stroke: C.ink, 'stroke-width': 1, 'stroke-dasharray': '2 2' })
      + text(0, stripY - 9, '*t*−τ', { size: 11.5, fill: C.ink, anchor: 'middle' })
      + loop.translate(tauSlide((tau) => inX1 - tau * PXS)) + loop.fade([[T1, L]], 0.3));
    push(mk);
    push(el('line', { x1: inX1, y1: stripY - 3, x2: inX1, y2: stripY + stripH + 3, stroke: C.ink2, 'stroke-width': 1 }));
    push(text(inX1, stripY + stripH + 15, '*t*', { size: 11.5, fill: C.ink2, anchor: 'middle' }));
  }
  // position that follows the delay shown, sliding at each change
  function tauSlide(f) {
    const pts = [[0, f(1), 0]];
    let prev = 1;
    for (const [t, tau] of tauTrack) {
      const dt = Math.abs(tau - prev) > 1 ? 0.5 : 0.18;
      pts.push([t - dt / 2, f(prev), 0], [t + dt / 2, f(tau), 0]);
      prev = tau;
    }
    return pts;
  }

  // reservoir strip: four regions spread over the strength ranking
  const resX0 = boxRes.x + 14, resX1 = boxRes.x + boxRes.w - 14;
  stripClip('cres', resX0, resX1);
  {
    const rank = Array.from({ length: n }, (_, i) => i).sort((a, b) => strength[b] - strength[a]);
    const shown = [rank[0], rank[29], rank[59], rank[89]];
    const lh = stripH / 4;
    for (const net of nets) {
      let g = '';
      shown.forEach((i, r) => {
        const yl = lane(stripY + r * lh, lh);
        const pts = []; for (let k = kStart(resX0, resX1); k <= K + 1; k++) pts.push([k * PXS, yl(Math.max(-1.6, Math.min(1.6, zstate(net, i, base + k) / 2.6)))]);
        g += el('path', { d: polyline(pts) });
      });
      push(el('g', { 'clip-path': 'url(#cres)', opacity: net === REAL ? 1 : 0 },
        scroll(resX1, el('g', { fill: 'none', stroke: C.res, 'stroke-width': 1.15, 'stroke-linejoin': 'round' }, g))
        + (net === REAL ? loop.fade([[0, SWITCH]], 0.5) : loop.fade([[SWITCH, L]], 0.5))));
    }
    push(text(resX0, stripY - 9, 'states *r*_{*i*}(*t*) of four regions', { size: 11.8, fill: C.ink2 }));
  }

  // readout strip: the delayed input (target) and its reconstruction
  const ouX0 = boxOut.x + 12, ouX1 = boxOut.x + boxOut.w - 12;
  stripClip('cout', ouX0, ouX1);
  {
    const yl = lane(stripY, stripH);
    const target = el('g', { opacity: 0 },
      el('g', { transform: `translate(${PXS} 0)` }, el('use', { href: '#uin', stroke: C.input, 'stroke-width': 1.1, opacity: 0.75 }) + loop.translate(tauSlide((tau) => tau * PXS)))
      + loop.fade([[T1, L]], 0.4));
    let rec = '';
    for (const net of nets) for (let tau = 1; tau <= TAU; tau++) {
      const [a, b] = win(net, tau);
      const ka = Math.floor(a * RATE) - Math.ceil((ouX1 - ouX0) / PXS) - 3, kb = Math.ceil(b * RATE) + 3;
      const pts = []; for (let k = ka; k <= Math.min(kb, K + 1); k++) pts.push([k * PXS, yl(Math.max(-1.25, Math.min(1.25, readout(net, tau, base + k))) * 0.92)]);
      rec += el('path', { d: polyline(pts), stroke: net.col, opacity: 0 }, loop.fade([[a, b]], net === REAL ? 0.16 : 0.08));
    }
    push(el('g', { 'clip-path': 'url(#cout)' }, scroll(ouX1, target + el('g', { fill: 'none', 'stroke-width': 1.5, 'stroke-linejoin': 'round' }, rec))));
    push(el('g', { opacity: 0 },
      el('line', { x1: ouX0, y1: stripY - 13, x2: ouX0 + 14, y2: stripY - 13, stroke: C.input, 'stroke-width': 1.4 })
      + text(ouX0 + 18, stripY - 9, '*u*(*t*−τ)', { size: 11.8, fill: C.ink2 })
      + loop.fade([[T1, L]], 0.4)));
    for (const net of nets) {
      const w = net === REAL ? [[T1, SWITCH]] : [[SWITCH, L]];
      push(el('g', { opacity: 0 },
        el('line', { x1: ouX0 + 80, y1: stripY - 13, x2: ouX0 + 94, y2: stripY - 13, stroke: net.col, 'stroke-width': 2 })
        + text(ouX0 + 98, stripY - 9, '*û*(*t*−τ)', { size: 11.8, fill: C.ink2 })
        + loop.fade(w, 0.4)));
    }
  }

  // ======================= row B: panels b, c, d =============================
  const B0 = 460, B1 = 638;
  const rowB = [];
  const pb = (s) => rowB.push(s);
  const axis = (x0, x1, y0, y1) => el('path', { d: `M${x0} ${y0}V${y1}H${x1}`, fill: 'none', stroke: C.muted, 'stroke-width': 1 });
  const letter = (x, s, title) => text(x, B0 - 16, `**${s}**`, { size: 15, fill: C.ink })
    + text(x + 18, B0 - 16, title, { size: 13.2, fill: C.ink, weight: 600 });

  // delay shown, one label per value
  const tauWins = (tau) => [win(REAL, tau), win(UNIF, tau)];

  // ---- (b) target and reconstruction on the test partition --------------------
  {
    const x0 = 62, x1 = 352, NB = 50;
    const sx = scale(0, NB, x0 + 4, x1 - 4), sy = scale(-1.2, 1.2, B1, B0);
    let g = letter(x0 - 36, 'b', 'Held-out reconstruction');
    for (const v of [-1, 0, 1]) {
      g += el('line', { x1: x0, x2: x1, y1: sy(v), y2: sy(v), stroke: C.grid, 'stroke-width': 1 });
      g += text(x0 - 7, sy(v) + 4, v < 0 ? `−${-v}` : String(v), { size: 11.5, fill: C.ink2, anchor: 'end' });
    }
    for (const v of [0, 25, 50]) g += text(sx(v), B1 + 16, String(v), { size: 11.5, fill: C.ink2, anchor: 'middle' });
    g += axis(x0, x1, B0, B1);
    g += text((x0 + x1) / 2, B1 + 34, 'held-out step *t*', { size: 12.2, fill: C.ink2, anchor: 'middle' });
    defs.push(el('clipPath', { id: 'cb' }, el('rect', { x: x0 + 1, y: B0 - 2, width: x1 - x0 - 1, height: B1 - B0 + 2 })));
    // target u(t - tau): one curve, shifted by one step at each delay
    const dx = sx(1) - sx(0);
    const tp = []; for (let k = -TAU - 1; k <= NB + 1; k++) tp.push([sx(k), sy(u[testStart + k])]);
    let curves = el('g', { transform: `translate(${num(dx, 2)} 0)` },
      el('path', { d: polyline(tp), fill: 'none', stroke: C.input, 'stroke-width': 1.5, 'stroke-linejoin': 'round' })
      + loop.translate(tauSlide((tau) => tau * dx)));
    for (const net of nets) for (let tau = 1; tau <= TAU; tau++) {
      const yh = net.res.yhatTest[tau - 1];
      const rp = []; for (let k = 0; k <= NB; k++) rp.push([sx(k), sy(yh[k])]);
      curves += el('path', { d: polyline(rp), fill: 'none', stroke: net.col, 'stroke-width': 2, 'stroke-linejoin': 'round', opacity: 0 },
        loop.fade([win(net, tau)], net === REAL ? 0.18 : 0.08));
    }
    g += el('g', { 'clip-path': 'url(#cb)' }, curves);
    // legend and delay
    g += el('line', { x1: x0 + 150, y1: B0 - 20, x2: x0 + 166, y2: B0 - 20, stroke: C.input, 'stroke-width': 1.5 })
      + text(x0 + 170, B0 - 16, 'target', { size: 11.8, fill: C.ink2 });
    for (const net of nets) {
      g += el('g', { opacity: 0 },
        el('line', { x1: x0 + 212, y1: B0 - 20, x2: x0 + 228, y2: B0 - 20, stroke: net.col, 'stroke-width': 2 })
        + text(x0 + 232, B0 - 16, 'readout', { size: 11.8, fill: C.ink2 })
        + loop.fade(net === REAL ? [[0, SWITCH]] : [[SWITCH, L]], 0.4));
    }
    for (let tau = 1; tau <= TAU; tau++)
      g += el('g', { opacity: 0 }, text(x1 - 6, B0 + 16, `τ = ${tau}`, { size: 13.5, fill: C.ink, anchor: 'end', weight: 700, halo: C.bg }) + loop.show(tauWins(tau)));
    pb(g);
  }

  // ---- (c) reconstruction against target ---------------------------------------
  {
    const x0 = 452, x1 = 634, NS = 300;
    const sx = scale(-1.15, 1.15, x0, x1), sy = scale(-1.15, 1.15, B1, B0);
    let g = letter(x0 - 36, 'c', 'Readout against target');
    for (const v of [-1, 0, 1]) {
      g += text(x0 - 7, sy(v) + 4, v < 0 ? `−${-v}` : String(v), { size: 11.5, fill: C.ink2, anchor: 'end' });
      g += text(sx(v), B1 + 16, v < 0 ? `−${-v}` : String(v), { size: 11.5, fill: C.ink2, anchor: 'middle' });
    }
    g += el('line', { x1: sx(-1.1), y1: sy(-1.1), x2: sx(1.1), y2: sy(1.1), stroke: C.muted, 'stroke-width': 1, 'stroke-dasharray': '4 3' });
    g += axis(x0, x1, B0, B1);
    g += text((x0 + x1) / 2, B1 + 34, 'target *u*(*t*−τ)', { size: 12.2, fill: C.ink2, anchor: 'middle' });
    g += text(x0 - 30, (B0 + B1) / 2, 'readout *û*(*t*−τ)', { size: 12.2, fill: C.ink2, anchor: 'middle', transform: `rotate(-90 ${x0 - 30} ${(B0 + B1) / 2})` });
    defs.push(el('clipPath', { id: 'cc' }, el('rect', { x: x0 + 1, y: B0, width: x1 - x0 - 1, height: B1 - B0 - 1 })));
    let pts = '';
    for (const net of nets) for (let tau = 1; tau <= TAU; tau++) {
      const y = net.res.yTest[tau - 1], yh = net.res.yhatTest[tau - 1];
      const p = []; for (let k = 0; k < NS; k++) p.push([sx(y[k]), sy(yh[k])]);
      pts += el('path', { d: dots(p), stroke: net.col, opacity: 0 }, loop.fade([win(net, tau)], net === REAL ? 0.18 : 0.08, 0, 0.62));
    }
    g += el('g', { 'clip-path': 'url(#cc)', fill: 'none', 'stroke-width': 3.4, 'stroke-linecap': 'round' }, pts);
    for (const net of nets) for (let tau = 1; tau <= TAU; tau++)
      g += el('g', { opacity: 0 },
        text(x0 + 12, B0 + 16, `*r*^{2} = ${net.res.r2[tau - 1].toFixed(3)}`, { size: 13.5, fill: C.ink, weight: 700, halo: C.bg })
        + loop.show([win(net, tau)]));
    pb(g);
  }

  // ---- (d) forgetting curve and memory capacity ---------------------------------
  {
    const x0 = 742, x1 = 1016;
    const sx = scale(0, 20.6, x0, x1), sy = scale(0, 1.08, B1, B0);
    let g = letter(x0 - 36, 'd', 'Forgetting curve');
    for (const v of [0, 0.5, 1]) {
      if (v) g += el('line', { x1: x0, x2: x1, y1: sy(v), y2: sy(v), stroke: C.grid, 'stroke-width': 1 });
      g += text(x0 - 7, sy(v) + 4, String(v), { size: 11.5, fill: C.ink2, anchor: 'end' });
    }
    for (const v of [0, 5, 10, 15, 20]) g += text(sx(v), B1 + 16, String(v), { size: 11.5, fill: C.ink2, anchor: 'middle' });
    g += axis(x0, x1, B0, B1);
    g += text((x0 + x1) / 2, B1 + 34, 'delay τ', { size: 12.2, fill: C.ink2, anchor: 'middle' });
    g += text(x0 - 38, (B0 + B1) / 2, '*r*^{2}(τ)', { size: 12.2, fill: C.ink2, anchor: 'middle', transform: `rotate(-90 ${x0 - 38} ${(B0 + B1) / 2})` });

    // delay being measured
    g += el('g', { opacity: 0 },
      el('line', { x1: 0, y1: B0, x2: 0, y2: B1, stroke: C.ink2, 'stroke-width': 1, 'stroke-dasharray': '2 3' })
      + loop.translate(tauSlide((tau) => sx(tau))) + loop.fade([[T1, T2], [TM, T4]], 0.3, 0, 0.8));

    for (const net of nets) {
      const r2 = net.res.r2, isR = net === REAL;
      const sw = isR ? sweepR : sweepN;
      const mark = (tau) => sw[tau - 1][0] + (isR ? 0.45 : 0.5) * (sw[tau - 1][1] - sw[tau - 1][0]);
      const pts = []; for (let tau = 1; tau <= TAU; tau++) pts.push([sx(tau), sy(r2[tau - 1])]);
      // line and area are revealed delay by delay through a growing clip
      const id = `rev${net.key}`;
      const rev = [[0, 0]];
      for (let tau = 1; tau <= TAU; tau++) rev.push([mark(tau) - 0.1, sx(tau - 1) + (tau === 1 ? 0 : 3) - x0], [mark(tau) + 0.1, sx(tau) + 3 - x0]);
      rev.push([TH + 0.5, sx(TAU) + 3 - x0]);
      defs.push(el('clipPath', { id }, el('rect', { x: x0, y: B0 - 6, width: 0, height: B1 - B0 + 6 }, loop.linear('width', rev))));
      let c = '';
      if (isR) {
        c += el('path', { d: polyline(pts) + `L${num(sx(TAU))} ${B1}L${num(sx(1))} ${B1}Z`, fill: net.col, opacity: 0.14 },
          loop.linear('opacity', [[0, 0.14], [T2, 0.14], [T2 + 0.5, 0.36], [T3 - 0.6, 0.36], [T3 + 0.4, 0.14], [L, 0.14]]));
      }
      c += el('path', { d: polyline(pts), fill: 'none', stroke: net.col, 'stroke-width': 2, 'stroke-linejoin': 'round' });
      g += el('g', { 'clip-path': `url(#${id})` }, c);
      let m = '';
      for (let tau = 1; tau <= TAU; tau++)
        m += el('circle', { cx: num(sx(tau)), cy: num(sy(r2[tau - 1])), r: 3.6, opacity: 0 }, loop.fade([[mark(tau), TH + 0.4]], 0.14));
      g += el('g', { fill: net.col, stroke: C.bg, 'stroke-width': 1.4 }, m);

      // running sum, then the total
      const ly = isR ? B0 + 15 : B0 + 60;
      const sums = []; let s = 0; for (let tau = 1; tau <= TAU; tau++) { s += r2[tau - 1]; sums.push(s); }
      const lx = sx(11.3);
      const done = isR ? T2 + 0.5 : T4 + 0.3;
      g += el('g', { opacity: 0 },
        el('circle', { cx: lx + 4.5, cy: ly - 4.5, r: 4.4, fill: net.col })
        + text(lx + 15, ly, isR ? 'connectome' : 'uniform weights', { size: 12.4, fill: C.ink2, halo: C.bg })
        + loop.fade([[isR ? T1 : TM, TH + 0.4]], 0.4));
      for (let tau = 1; tau <= TAU; tau++) {
        const a = mark(tau), b = tau === TAU ? done : mark(tau + 1);
        g += el('g', { opacity: 0 },
          text(lx + 15, ly + 19, `Σ *r*^{2} = ${sums[tau - 1].toFixed(2)}`, { size: 13.2, fill: C.ink, weight: 600, halo: C.bg }) + loop.show([[a, b]]));
      }
      g += el('g', { opacity: 0 },
        text(lx + 15, ly + 19, `MC = ${net.res.MC.toFixed(2)}`, { size: 14.5, fill: C.ink, weight: 700, halo: C.bg }) + loop.show([[done, TH + 0.4]]));
    }
    pb(g);
  }
  push(el('g', { opacity: 0.3 }, rowB.join('') + loop.fade([[T1, L]], 0.6, 0.3, 1)));

  // ======================= assemble ==========================================
  const body = el('g', { opacity: 0 }, out.join('\n') + loop.linear('opacity', [[0, 0], [0.5, 1], [TH + 0.1, 1], [L - 0.05, 0], [L, 0]]));
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${Wd} ${Ht}" width="${Wd}" height="${Ht}" font-family="${FONT}" role="img" aria-labelledby="t d">`
    + `<title id="t">The memory-capacity paradigm on a human connectome</title>`
    + `<desc id="d">A random scalar input drives a human connectome used as the fixed recurrent layer of an echo-state network. `
    + `A ridge readout reconstructs the input tau steps in the past; the squared correlation on held-out data, r2(tau), is the forgetting curve `
    + `and its sum over delays is the memory capacity (MC = ${REAL.res.MC.toFixed(2)} for this connectome, stable rank ${REAL.rs.toFixed(2)}). `
    + `With the weights around each region made uniform on the same wiring, the stable rank is ${UNIF.rs.toFixed(2)} and MC = ${UNIF.res.MC.toFixed(2)}.</desc>`
    + el('defs', {}, defs.join(''))
    + el('rect', { x: 0.5, y: 0.5, width: Wd - 1, height: Ht - 1, rx: 14, fill: C.bg, stroke: C.frame })
    + body + '</svg>\n';
  return { svg, L, stats: { MC: REAL.res.MC, rs: REAL.rs, MCu: UNIF.res.MC, rsu: UNIF.rs, r2: Array.from(REAL.res.r2), r2u: Array.from(UNIF.res.r2) } };
}

module.exports = { build, THEMES };

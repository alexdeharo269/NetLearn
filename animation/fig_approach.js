// ============================================================================
//  fig_approach.js — the approach of the paper as a silent, looping explainer.
//
//  One idea per scene, in this order:
//    1  the connectome as a network          6  the connectome as a neural network
//    2  every connection has a weight        7  a random signal drives it
//    3  disparity (one region)               8  can its state recall the past signal?
//    4  loops (subgraph centrality)          9  the forgetting curve
//    5  stable rank (whole network)         10  memory capacity
//                                           11  different weights, different memory
//
//  Everything drawn is data: the regions sit at their atlas centroids, the links are
//  the weights of two connectomes, and the signals, the readouts and the forgetting
//  curves are simulated with esn.js under the settings of the paper.
//
//  build(WA, WB) returns the SVG as a string. WA is the connectome followed through
//  the whole figure, WB a second one for the two comparisons (scenes 5 and 11).
// ============================================================================
'use strict';
const fs = require('fs');
const path = require('path');
const esn = require('./esn.js');
const { REGIONS, OUTLINE } = require('./atlas.js');
const { num, el, text: rawText, Loop, scale, polyline, pathLength, mix, hex3 } = require('./svg.js');

const C = {
  bg: '#ffffff', ink: '#14181d', soft: '#5d6672', faint: '#b9bfc7', ghost: '#9aa1ab', hair: '#dfe2e6',
  blue: '#1d6fb8', red: '#d7402b', brain: '#f5f6f7', brainEdge: '#c6ccd3', band: '#cfd4da',
};
const text = (x, y, s, o = {}) => rawText(x, y, s, { fill: C.ink, colours: C, ...o });

// Space Grotesk (SIL Open Font License), embedded so the figure looks the same everywhere
function fontFace() {
  const dir = path.join(__dirname, 'fonts');
  return [500, 700].map((w) => {
    const b64 = fs.readFileSync(path.join(dir, `space-grotesk-latin-${w}-normal.woff2`)).toString('base64');
    return `@font-face{font-family:'SG';font-weight:${w};src:url(data:font/woff2;base64,${b64}) format('woff2')}`;
  }).join('');
}
const FONT = 'SG,-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif';

// ---------------------------------------------------------------------------
//  One connectome: simulation, weights of the rescaled network, node measures
// ---------------------------------------------------------------------------
const N = 90;
const MM = 1.9;                                   // px per mm at scale 1
const Y0 = (Math.min(...OUTLINE.map((p) => p[1])) + Math.max(...OUTLINE.map((p) => p[1]))) / 2;
const P = REGIONS.map(([, x, y]) => [MM * x, -MM * (y - Y0)]);     // seen from above, front up
const OUT = OUTLINE.map(([x, y]) => [MM * x, -MM * (y - Y0)]);

function prepare(W) {
  const res = esn.memoryCapacity(W, {}, true);    // seed 42: the same input and Win for every connectome
  const Wn = res.Wn;
  const edges = [];                               // [rescaled weight, i, j, share of the squared weight]
  let f2 = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (W[i * N + j] > 0) f2 += W[i * N + j] ** 2;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++)
    if (Wn[i * N + j] > 0) edges.push([Wn[i * N + j], i, j, W[i * N + j] ** 2 / f2]);
  edges.sort((a, b) => a[0] - b[0]);
  const k = new Float64Array(N), s = new Float64Array(N), Y = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) if (W[i * N + j] > 0) { k[i]++; s[i] += W[i * N + j]; }
    for (let j = 0; j < N; j++) if (W[i * N + j] > 0) Y[i] += (W[i * N + j] / s[i]) ** 2;
  }
  return { W, res, Wn, edges, k, s, Y, rs: esn.stableRank(W, N) };
}

const widthOf = (w) => Math.max(0.5, 11 * w);     // line width of a link of rescaled weight w
const line = (i, j) => `M${num(P[i][0])} ${num(P[i][1])}L${num(P[j][0])} ${num(P[j][1])}`;

// static layers of a network, in local coordinates (origin at the centre of the brain)
function layers(net, id) {
  const outline = el('path', { id: `${id}o`, d: polyline(OUT) + 'Z', fill: C.brain, stroke: C.brainEdge, 'stroke-width': 1.6, 'stroke-linejoin': 'round' });
  // the wiring alone: every link drawn alike
  const binary = el('path', { id: `${id}b`, d: net.edges.map(([, i, j]) => line(i, j)).join(''), fill: 'none', stroke: C.soft, 'stroke-width': 0.7, opacity: 0.45 });
  // the weights: line width proportional to the weight
  let thin = '', thick = '';
  for (const [w, i, j] of net.edges) {
    if (widthOf(w) <= 0.5) thin += line(i, j);
    else thick += el('path', { d: line(i, j), 'stroke-width': num(widthOf(w), 2), 'stroke-opacity': num(Math.min(1, 0.4 + 2.6 * w), 2) });
  }
  const weighted = el('g', { id: `${id}w`, fill: 'none', stroke: C.ink, 'stroke-linecap': 'round' },
    el('path', { d: thin, 'stroke-width': 0.5, 'stroke-opacity': 0.2 }) + thick);
  // the 15 connections that carry most of the squared weight: line width proportional to their share
  const share = el('g', { id: `${id}s`, fill: 'none', stroke: C.blue, 'stroke-linecap': 'round' },
    net.edges.slice(-15).map(([, i, j, f]) => el('path', { d: line(i, j), 'stroke-width': num(2 + 46 * f, 2) })).join(''));
  const nodes = el('g', { id: `${id}n`, fill: C.blue, stroke: C.bg, 'stroke-width': 1.1 },
    P.map(([x, y]) => el('circle', { cx: num(x), cy: num(y), r: 4.4 })).join(''));
  // a still picture: outline, faint wiring, dominant connections, regions
  const still = el('g', { id }, el('use', { href: `#${id}o` }) + el('use', { href: `#${id}w`, opacity: 0.1 })
    + el('use', { href: `#${id}s` }) + el('use', { href: `#${id}n`, opacity: 0.4 }));
  return [outline, binary, weighted, share, nodes, still];
}
const use = (id, a = {}) => el('use', { href: `#${id}`, ...a });

// ---------------------------------------------------------------------------
function build(WA, WB) {
  const A = prepare(WA), B = prepare(WB);
  const TAU = A.res.params.tauMax, u = A.res.u, t0 = A.res.testStart;

  // ---- timeline (seconds): start of each scene ----------------------------------
  const S1 = 0, S2 = 8.5, S3 = 15.5, S4 = 29, S5 = 38, S6 = 49.5, S7 = 57, S8 = 65, S9 = 80.5, S10 = 89.5, S11 = 95.5, END = 104.5;
  const L = END + 0.8;
  const loop = new Loop(L);
  const defs = [], out = [];
  const push = (s) => out.push(s);
  const show = (a, b, inner, f = 0.6) => el('g', { opacity: 0 }, inner + loop.fade([[a, b]], f));
  const side = (a, b, y, s, o = {}) => show(a, b, text(40, y, s, { size: 20, ...o }));

  // ---- headlines: one sentence per scene --------------------------------------------
  [
    [S1, S2, 'The {blue|connectome}: the brain as a network'],
    [S2, S3, 'Every connection has a {blue|weight}'],
    [S3, S4, '{blue|Disparity}: how a region spreads its strength'],
    [S4, S5, '{blue|Loops}: activity leaves a region and comes back'],
    [S5, S6, '{blue|Stable rank}: the same question for the whole network'],
    [S6, S7, 'Now the {blue|connectome} becomes a neural network'],
    [S7, S8, 'A random signal is fed into every region'],
    [S8, S9, 'Can its present state {red|recall the past} signal?'],
    [S9, S10, 'Recall fades with the delay: the {red|forgetting curve}'],
    [S10, S11, 'The area under it is the {red|memory capacity}'],
    [S11, L, 'Different {blue|weights}, different {red|memory}'],
  ].forEach(([a, b, s]) => push(show(a + 0.5, b - 0.25, text(40, 66, s, { size: 29, weight: 700 }), 0.4)));

  defs.push(...layers(A, 'a'), ...layers(B, 'b'));

  // ====================================================================================
  //  The connectome followed through the figure (A): one group that moves and rescales
  // ====================================================================================
  const BIG = [650, 300, 1.08], PAIR = [255, 292, 0.8], PIPE = [500, 300, 0.85];
  const hidden = [S3 + 2.9, S5 + 0.3];              // A is off screen while the stars are shown
  const move = [S6 + 0.4, S6 + 2.0];                // from the pair of scene 5 to the network of scene 6
  const RATE = 4, ta = S7 + 1.2, tb = S9;           // reservoir steps shown per second, and when
  const base = t0 + 200;                            // shown steps lie in the held-out part of the run
  const steps = Math.floor((tb - ta) * RATE);

  // the two regions of the disparity scene: the most even and the most uneven shares of strength
  {
    const e = Array.from(A.k, (k, i) => ({ i, k, ups: k * A.Y[i] })).filter((r) => r.k >= 8).sort((p, q) => p.ups - q.ups);
    A.lo = e[0].i; A.hi = e[e.length - 1].i;
  }

  let groupA;
  {
    // activity of the regions (scenes 7 and 8): one keyframe per reservoir step
    const sd = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let m = 0, v = 0;
      for (let t = t0; t < u.length; t++) m += A.res.X[t * N + i];
      m /= (u.length - t0);
      for (let t = t0; t < u.length; t++) v += (A.res.X[t * N + i] - m) ** 2;
      sd[i] = Math.sqrt(v / (u.length - t0));
    }
    const tone = (z) => hex3(mix('#dce9f6', '#0d4a87', Math.max(0, Math.min(1, (z + 2) / 4))));
    const order = Array.from({ length: N }, (_, i) => i).sort((p, q) => P[p][1] - P[q][1]);   // front to back
    const rankOf = new Int32Array(N); order.forEach((i, r) => { rankOf[i] = r; });
    let nodes = '';
    for (let i = 0; i < N; i++) {
      const pts = [[0, hex3(C.blue)], [ta - 0.5, hex3(C.blue)]];
      for (let k = 0; k <= steps; k++) pts.push([ta + k / RATE, tone(A.res.X[(base + k) * N + i] / sd[i])]);
      const tin = S1 + 2.0 + 1.5 * rankOf[i] / N;
      nodes += el('circle', { cx: num(P[i][0]), cy: num(P[i][1]), r: 4.4, opacity: 0 },
        loop.linear('opacity', [[0, 0], [tin, 0], [tin + 0.35, 1], [L, 1]]) + loop.linear('fill', pts));
    }

    // scene 3: one region and its connections, on the brain
    let ego = '';
    for (let j = 0; j < N; j++) {
      const w = A.Wn[A.lo * N + j];
      if (w > 0) ego += el('path', { d: line(A.lo, j), 'stroke-width': num(Math.max(1.2, widthOf(w)), 2) })
        + el('circle', { cx: num(P[j][0]), cy: num(P[j][1]), r: 4.4, stroke: C.bg, 'stroke-width': 1.1, fill: C.blue });
    }
    ego += el('circle', { cx: num(P[A.lo][0]), cy: num(P[A.lo][1]), r: 8, fill: C.blue, stroke: C.bg, 'stroke-width': 1.6 });

    const len = pathLength(OUT.concat([OUT[0]]));
    const h0 = hidden[0], h1 = hidden[1];
    const inner =
      // outline: drawn at the start, gone once the connectome is used as a network
      el('g', { opacity: 1 },
        el('use', { href: '#ao', 'stroke-dasharray': num(len), 'stroke-dashoffset': num(len) },
          loop.linear('stroke-dashoffset', [[0, len], [0.4, len], [1.9, 0], [L, 0]], true))
        + loop.linear('opacity', [[0, 1], [move[0], 1], [move[1], 0], [L, 0]]))
      + el('g', { opacity: 0 }, use('ab') + loop.fade([[S1 + 4.6, S2 + 1.0]], 1.2))
      + el('g', { opacity: 0 }, use('aw') + loop.linear('opacity',
        [[0, 0], [S2 + 0.4, 0], [S2 + 1.6, 1], [S3 + 0.5, 1], [S3 + 1.2, 0.13], [h0 + 0.4, 0.13], [h0 + 0.5, 0.1], [move[0], 0.1], [move[1], 1], [L, 1]]))
      + el('g', { opacity: 0 }, use('as') + loop.fade([[h1 + 1.3, move[0] + 0.4]], 0.8))
      + el('g', { stroke: C.bg, 'stroke-width': 1.1, fill: C.blue }, nodes + loop.linear('opacity',
        [[0, 1], [S3 + 0.5, 1], [S3 + 1.2, 0.13], [h0 + 0.4, 0.13], [h0 + 0.5, 0.4], [move[0], 0.4], [move[1], 1], [L, 1]]))
      + el('g', { opacity: 0, stroke: C.blue, 'stroke-linecap': 'round', fill: 'none' }, ego + loop.fade([[S3 + 1.0, h0]], 0.7));
    const at = (p) => [[0, ...BIG], [h0 + 0.5, ...BIG], [h0 + 0.6, ...PAIR], [move[0], ...PAIR], [move[1], ...PIPE], [L, ...PIPE]].map((r) => p(r));
    groupA = el('g', { opacity: 1 },
      el('g', { transform: `translate(${BIG[0]} ${BIG[1]})` },
        el('g', { transform: `scale(${BIG[2]})` }, inner + loop.scale(at(([t, , , s]) => [t, s])))
        + loop.translate(at(([t, x, y]) => [t, x, y])))
      + loop.fade([[0, h0], [h1, S9 - 0.2]], 0.7));
  }

  // ---- scenes 1 and 2: two short lines on the left ---------------------------------------
  push(show(S1 + 2.0, S2 - 0.1, el('circle', { cx: 49, cy: 263, r: 6, fill: C.blue }) + text(70, 270, '90 brain regions', { size: 20 })));
  push(show(S1 + 4.6, S2 - 0.1, el('line', { x1: 40, x2: 60, y1: 303, y2: 303, stroke: C.soft, 'stroke-width': 1.6 }) + text(70, 310, 'the nerve fibres between them', { size: 20 })));
  push(side(S2 + 1.4, S3 - 0.1, 270, 'thicker line, more fibres'));
  push(side(S2 + 3.6, S3 - 0.1, 310, 'a few are strong, most are weak'));

  // ====================================================================================
  //  Scene 3: disparity. Two regions of the connectome, each with its own connections;
  //  line width is the share of the region's strength on that connection
  // ====================================================================================
  const star = (node) => {
    const nb = [];
    for (let j = 0; j < N; j++) if (A.W[node * N + j] > 0) nb.push(A.W[node * N + j] / A.s[node]);
    nb.sort((p, q) => q - p);
    const R = 104;
    let g = '', tips = '';
    nb.forEach((p, m) => {
      const th = -Math.PI / 2 + 2 * Math.PI * m / nb.length;
      const x = R * Math.cos(th), y = R * Math.sin(th);
      g += el('path', { d: `M0 0L${num(x)} ${num(y)}`, 'stroke-width': num(Math.max(1, 44 * p), 2) });
      tips += el('circle', { cx: num(x), cy: num(y), r: 5 });
    });
    return { nb, R, svg: el('g', { stroke: C.blue, fill: 'none', 'stroke-linecap': 'round' }, g)
      + el('g', { fill: C.faint, stroke: C.bg, 'stroke-width': 1.2 }, tips)
      + el('circle', { r: 12, fill: C.blue, stroke: C.bg, 'stroke-width': 2 }) };
  };
  const starLo = star(A.lo), starHi = star(A.hi);
  const SL = [250, 262], SH = [705, 262];
  push(side(S3 + 1.2, hidden[0], 290, 'one region and its connections'));
  push(show(S3 + 3.4, S4 - 0.1, el('g', { transform: `translate(${SL[0]} ${SL[1]})` }, starLo.svg)
    + text(SL[0], 420, 'spread evenly', { size: 20, anchor: 'middle', fill: C.soft })
    + text(SL[0], 452, 'low disparity', { size: 22, anchor: 'middle', weight: 700 })));
  push(show(S3 + 8.0, S5 - 0.1, el('g', { transform: `translate(${SH[0]} ${SH[1]})` }, starHi.svg)));
  push(show(S3 + 8.0, S4 - 0.1, text(SH[0], 420, 'one connection takes most', { size: 20, anchor: 'middle', fill: C.soft })
    + text(SH[0], 452, 'high disparity', { size: 22, anchor: 'middle', weight: 700 })));

  // ====================================================================================
  //  Scene 4: loops. A signal goes out along a connection and comes back
  // ====================================================================================
  {
    const tip = (m) => { const th = -Math.PI / 2 + 2 * Math.PI * m / starHi.nb.length; return [SH[0] + starHi.R * Math.cos(th), SH[1] + starHi.R * Math.sin(th)]; };
    const trip = (a, m, r) => {         // out and back along spoke m, starting at time a
      const [x, y] = tip(m), d = 1.5;
      const at = (attr, v0, v1) => loop.linear(attr, [[0, v0], [a, v0], [a + d / 2, v1], [a + d, v0], [L, v0]], true);
      return el('circle', { cx: SH[0], cy: SH[1], r, fill: C.bg, stroke: C.ink, 'stroke-width': 2.4, opacity: 0 },
        at('cx', SH[0], x) + at('cy', SH[1], y) + loop.linear('opacity', [[0, 0], [a - 0.05, 0], [a + 0.1, 1], [a + d - 0.1, 1], [a + d + 0.05, 0], [L, 0]]));
    };
    const weak = Math.floor(starHi.nb.length * 0.45);
    push(trip(S4 + 1.4, 0, 8) + trip(S4 + 3.2, 0, 8) + trip(S4 + 5.6, weak, 5) + trip(S4 + 7.0, weak, 5));
    push(side(S4 + 1.0, S5 - 0.1, 232, 'a signal goes out and comes back'));
    push(side(S4 + 3.4, S5 - 0.1, 272, '{blue|subgraph centrality} counts these loops'));
    push(side(S4 + 5.6, S5 - 0.1, 312, 'strong connections make strong loops'));
  }

  // ====================================================================================
  //  Scene 5: stable rank. The same connectome next to another one; the blue lines are
  //  the connections that carry most of the squared weight, as wide as their share
  // ====================================================================================
  const cap = (x, a, b, s1, s2) => show(a, b, text(x, 458, s1, { size: 20, anchor: 'middle', fill: C.soft }) + text(x, 490, s2, { size: 22, anchor: 'middle', weight: 700 }));
  const PB = [705, 292, 0.8];
  push(cap(PAIR[0], S5 + 2.0, S6 + 0.1, 'one connection dominates', 'low stable rank'));
  push(show(S5 + 5.6, S6 + 0.2, el('g', { transform: `translate(${PB[0]} ${PB[1]}) scale(${PB[2]})` }, use('b')), 0.9));
  push(cap(PB[0], S5 + 6.2, S6 + 0.1, 'the weight is shared', 'high stable rank'));

  // ====================================================================================
  //  Scenes 6 to 8: the connectome as the recurrent layer of an echo-state network
  // ====================================================================================
  push(side(S6 + 2.2, S7 - 0.1, 252, 'regions are its units'));
  push(side(S6 + 3.6, S7 - 0.1, 290, 'fibres are its connections'));
  push(side(S6 + 5.0, S7 - 0.1, 328, 'nothing in it is trained'));

  const IN = [300, PIPE[1]], RO = [700, PIPE[1]];
  const shrink = [S8 + 3.2, S8 + 4.4], SM = 0.72, YC = 208;     // the network makes room for the comparison below
  const small = (p) => [PIPE[0] + SM * (p[0] - PIPE[0]), YC + SM * (p[1] - PIPE[1])];
  {
    const fan = (from, id, col) => {
      defs.push(el('linearGradient', { id, gradientUnits: 'userSpaceOnUse', x1: from[0], x2: PIPE[0], y1: 0, y2: 0 },
        el('stop', { offset: 0, 'stop-color': col, 'stop-opacity': 0.55 }) + el('stop', { offset: 0.5, 'stop-color': col, 'stop-opacity': 0.3 })
        + el('stop', { offset: 1, 'stop-color': col, 'stop-opacity': 0 })));
      return el('path', { d: P.map(([x, y]) => `M${from[0]} ${from[1]}L${num(PIPE[0] + PIPE[2] * x)} ${num(PIPE[1] + PIPE[2] * y)}`).join(''), stroke: `url(#${id})`, 'stroke-width': 0.9, fill: 'none' });
    };
    // the signal arrives from the left, newest value at the input unit
    const PXS = 9, XR = IN[0] - 16, amp = 44;
    const pts = []; for (let k = -30; k <= steps + 1; k++) pts.push([k * PXS, IN[1] - amp * u[base + k]]);
    defs.push(el('clipPath', { id: 'ct' }, el('rect', { x: 40, y: IN[1] - amp - 8, width: XR - 40, height: 2 * amp + 16 })));
    const trace = el('g', { 'clip-path': 'url(#ct)' },
      el('g', { transform: `translate(${XR} 0)` },
        el('path', { d: polyline(pts), fill: 'none', stroke: C.ink, 'stroke-width': 2, 'stroke-linejoin': 'round' })
        + loop.translate([[0, XR, 0], [ta, XR, 0], [tb, XR - (tb - ta) * RATE * PXS, 0], [L, XR - (tb - ta) * RATE * PXS, 0]], false)));
    const pipeline = groupA
      + show(S7 + 0.6, S9 - 0.2, trace + fan(IN, 'gi', C.soft) + el('circle', { cx: IN[0], cy: IN[1], r: 14, fill: C.ink }))
      + show(S8 + 0.6, S9 - 0.2, fan(RO, 'gr', C.red) + el('circle', { cx: RO[0], cy: RO[1], r: 14, fill: C.red }));
    push(el('g', {},
      el('g', {}, pipeline + loop.scale([[0, 1], [shrink[0], 1], [shrink[1], SM], [L, SM]]))
      + loop.translate([[0, 0, 0], [shrink[0], 0, 0], [shrink[1], PIPE[0] * (1 - SM), YC - SM * PIPE[1]], [L, PIPE[0] * (1 - SM), YC - SM * PIPE[1]]])));
    // labels, at full size before and after the network makes room
    push(show(S7 + 0.8, shrink[0], text(40, IN[1] - amp - 22, 'a random signal', { size: 20 })));
    push(show(S8 + 0.9, shrink[0], text(RO[0] + 28, RO[1] - 4, 'readout', { size: 21, weight: 700 })
      + text(RO[0] + 28, RO[1] + 22, 'the only trained part', { size: 18, fill: C.soft })));
    push(show(shrink[1] + 0.2, S9 - 0.2, text(small(IN)[0], small(IN)[1] + 42, 'input', { size: 18, anchor: 'middle', fill: C.soft })
      + text(small(RO)[0], small(RO)[1] + 42, 'readout', { size: 18, anchor: 'middle', fill: C.soft })));
  }

  // the signal as it was some steps back, and what the readout recovers of it (held-out steps)
  {
    const x0 = 110, x1 = 850, yc = 432, hh = 54, NB = 50;
    const sx = scale(0, NB, x0, x1), sy = (v) => yc - v * hh, dx = sx(1) - sx(0);
    defs.push(el('clipPath', { id: 'cs' }, el('rect', { x: x0, y: yc - hh - 12, width: x1 - x0, height: 2 * hh + 24 })));
    const pts = []; for (let k = -TAU - 1; k <= NB + 1; k++) pts.push([sx(k), sy(u[t0 + k])]);
    const a0 = shrink[1] + 0.6;
    const delays = [[1, a0, a0 + 3.4], [8, a0 + 3.4, a0 + 6.9], [TAU, a0 + 6.9, S9 - 0.3]];
    const slide = [[0, dx, 0]];
    let prev = 1;
    for (const [tau, a] of delays.slice(1)) { slide.push([a - 0.5, prev * dx, 0], [a + 0.4, tau * dx, 0]); prev = tau; }
    let rec = '';
    for (const [tau, a, b] of delays) {
      const yh = A.res.yhatTest[tau - 1];
      const rp = []; for (let k = 0; k <= NB; k++) rp.push([sx(k), sy(yh[k])]);
      rec += el('path', { d: polyline(rp), opacity: 0 }, loop.fade([[a + 0.5, b]], 0.5));
      push(show(a + 0.25, b - 0.25, text(x0, yc - hh - 22, tau === 1 ? '1 step back' : `${tau} steps back`, { size: 21, weight: 700 }), 0.4));
    }
    const ly = yc - hh - 28;
    push(show(a0 - 0.3, S9 - 0.2,
      el('g', { 'clip-path': 'url(#cs)' },
        el('g', { transform: `translate(${num(dx, 2)} 0)` },
          el('path', { d: polyline(pts), fill: 'none', stroke: C.band, 'stroke-width': 7, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' })
          + loop.translate(slide))
        + el('g', { fill: 'none', stroke: C.red, 'stroke-width': 2.4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, rec))
      + el('line', { x1: 520, x2: 552, y1: ly, y2: ly, stroke: C.band, 'stroke-width': 7, 'stroke-linecap': 'round' })
      + text(562, ly + 6, 'the past signal', { size: 18, fill: C.soft })
      + el('line', { x1: 728, x2: 760, y1: ly, y2: ly, stroke: C.red, 'stroke-width': 2.4, 'stroke-linecap': 'round' })
      + text(770, ly + 6, 'readout', { size: 18, fill: C.soft })));
  }

  // ====================================================================================
  //  Scenes 9 to 11: the forgetting curve, its area, and a second connectome
  // ====================================================================================
  {
    const x0 = 170, x1 = 850, yT = 138, yB = 440;
    const sx = scale(0, TAU + 0.5, x0, x1), sy = scale(0, 1.04, yB, yT);
    const curve = (net) => Array.from({ length: TAU }, (_, i) => [sx(i + 1), sy(net.res.r2[i])]);
    const ca = curve(A), cb = curve(B);
    let ax = el('path', { d: `M${x0} ${yT - 6}V${yB}H${x1}`, fill: 'none', stroke: C.ink, 'stroke-width': 1.4 });
    for (const v of [1, 5, 10, 15, 20]) ax += el('line', { x1: num(sx(v)), x2: num(sx(v)), y1: yB, y2: yB + 6, stroke: C.ink, 'stroke-width': 1.4 })
      + text(sx(v), yB + 28, String(v), { size: 18, anchor: 'middle', fill: C.soft });
    ax += text((x0 + x1) / 2, yB + 60, 'steps back', { size: 20, anchor: 'middle' });
    ax += el('line', { x1: x0 - 6, x2: x0, y1: num(sy(1)), y2: num(sy(1)), stroke: C.ink, 'stroke-width': 1.4 })
      + text(x0 - 14, sy(1) + 6, 'perfect', { size: 18, anchor: 'end', fill: C.soft })
      + text(x0 - 14, sy(0) + 6, 'none', { size: 18, anchor: 'end', fill: C.soft })
      + text(x0 - 14, (yT + yB) / 2 + 6, 'recall', { size: 20, anchor: 'end' });
    push(show(S9 + 0.4, L, ax));

    // points come in one delay at a time; the line follows through a growing clip
    const t1 = S9 + 1.6, dt = 0.3, mark = (tau) => t1 + (tau - 1) * dt;
    const grow = [[0, 0]];
    for (let tau = 1; tau <= TAU; tau++) grow.push([mark(tau), sx(tau) + 2 - x0]);
    const clip = (id, track) => defs.push(el('clipPath', { id }, el('rect', { x: x0, y: yT - 12, width: 0, height: yB - yT + 12 }, track)));
    clip('ca', loop.linear('width', grow.concat([[L, x1 - x0]])));
    clip('cb', loop.linear('width', [[0, 0], [S11 + 2.4, 0], [S11 + 4.6, x1 - x0], [L, x1 - x0]]));
    const lineOf = (c, col) => el('path', { d: polyline(c), fill: 'none', stroke: col, 'stroke-width': 2.6, 'stroke-linejoin': 'round' });
    const dotsOf = (c, col, when) => el('g', { fill: col, stroke: C.bg, 'stroke-width': 1.6 },
      c.map(([x, y], i) => el('circle', { cx: num(x), cy: num(y), r: 5, ...(when ? { opacity: 0 } : {}) }, when ? loop.fade([[when(i + 1), L]], 0.25) : '')).join(''));

    push(el('path', { d: polyline(ca) + `L${num(sx(TAU))} ${yB}L${num(sx(1))} ${yB}Z`, fill: C.red, opacity: 0 },
      loop.linear('opacity', [[0, 0], [S10 + 0.4, 0], [S10 + 1.4, 0.2], [S11 + 0.4, 0.2], [S11 + 1.4, 0], [L, 0]])));
    push(show(S10 + 1.2, S11 + 0.2, text((sx(1) + sx(7)) / 2, sy(0.42), 'memory capacity', { size: 21, weight: 700, anchor: 'middle' })));
    // connectome A: red while it is the subject, grey once the second one arrives
    push(show(S9 + 1.0, S11 + 1.0, el('g', { 'clip-path': 'url(#ca)' }, lineOf(ca, C.red)) + dotsOf(ca, C.red, mark), 0.8));
    push(show(S11 + 0.6, L, lineOf(ca, C.ghost) + dotsOf(ca, C.ghost), 0.8));
    push(show(S11 + 2.2, L, el('g', { 'clip-path': 'url(#cb)' }, lineOf(cb, C.red) + dotsOf(cb, C.red)), 0.4));

    // the two connectomes of scene 5, as the key
    const key = (x, id, col, s, a) => show(a, L,
      el('g', { transform: `translate(${x} 182) scale(0.3)` }, use(id))
      + el('line', { x1: x - 22, x2: x + 22, y1: 254, y2: 254, stroke: col, 'stroke-width': 4, 'stroke-linecap': 'round' })
      + text(x, 282, s, { size: 18, anchor: 'middle', fill: C.soft }));
    push(key(626, 'a', C.ghost, 'one dominates', S11 + 0.8));
    push(key(782, 'b', C.red, 'weight shared', S11 + 2.4));
  }

  // ---- progress ---------------------------------------------------------------------------
  push(el('rect', { x: 0, y: 536, width: 0, height: 4, fill: C.hair }, loop.linear('width', [[0, 0], [END, 960], [L, 960]])));

  const Wd = 960, Ht = 540;
  const body = el('g', { opacity: 0 }, out.join('\n') + loop.linear('opacity', [[0, 0], [0.5, 1], [END + 0.2, 1], [L - 0.05, 0], [L, 0]]));
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${Wd} ${Ht}" width="${Wd}" height="${Ht}" font-family="${FONT}" font-weight="500" role="img" aria-labelledby="t d">`
    + '<title id="t">From a human connectome to its memory capacity</title>'
    + '<desc id="d">An animated explanation in eleven scenes. The connectome is a network of 90 brain regions linked by nerve fibres, and every link has a weight. '
    + 'Disparity measures how evenly a region spreads its strength over its connections; subgraph centrality counts the loops by which activity returns to a region; '
    + 'the stable rank asks whether the weight of the whole network is shared or held by a few connections. The connectome is then used as the fixed recurrent layer of a neural network: '
    + 'a random signal is fed into every region and a trained readout tries to recall the signal a number of steps back. Recall fades with the delay; this is the forgetting curve, '
    + 'and the area under it is the memory capacity. Two connectomes with different weights have different memory capacities.</desc>'
    + `<style>${fontFace()}</style>`
    + el('defs', {}, defs.join(''))
    + el('rect', { x: 0.5, y: 0.5, width: Wd - 1, height: Ht - 1, rx: 12, fill: C.bg, stroke: C.hair })
    + body + '</svg>\n';
  return { svg, L, scenes: { S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, END },
    stats: { A: { rs: A.rs, MC: A.res.MC, lo: REGIONS[A.lo][0], hi: REGIONS[A.hi][0], r2: Array.from(A.res.r2) }, B: { rs: B.rs, MC: B.res.MC } } };
}

module.exports = { build };

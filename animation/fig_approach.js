// ============================================================================
//  fig_approach.js — the approach of the paper as a silent, looping explainer.
//
//  One idea per scene, in this order:
//    1  the connectome as a network          8  can its state recall the past signal?
//    2  every connection has a weight        9  the forgetting curve
//    3  disparity (one region)              10  memory capacity
//    4  loops (subgraph centrality)         11  different weights, different memory
//    5  stable rank (whole network)         12  the heaviest links join the hemispheres
//    6  the connectome as a neural network  13  halve them: the same wiring remembers more
//    7  a random signal drives it           14  closing statement
//
//  Everything drawn is data: the regions sit at their atlas centroids, the links are
//  the weights of two connectomes, and the signals, the readouts and the forgetting
//  curves are simulated with esn.js under the settings of the paper.
//
//  build(WA, WB, { theme, speed }) returns the SVG as a string. WA is the connectome
//  followed through the whole figure, WB a second one for scenes 5 and 11.
// ============================================================================
'use strict';
const fs = require('fs');
const path = require('path');
const esn = require('./esn.js');
const { REGIONS, OUTLINE } = require('./atlas.js');
const { num, el, text: rawText, Loop, scale, polyline, pathLength, mix, hex3 } = require('./svg.js');

const THEMES = {
  light: {
    bg: '#ffffff', ink: '#14181d', soft: '#5d6672', faint: '#b9bfc7', ghost: '#9aa1ab', hair: '#dfe2e6',
    blue: '#1d6fb8', red: '#d7402b', brain: '#f5f6f7', brainEdge: '#c6ccd3', band: '#cfd4da',
    calm: '#dce9f6', active: '#0d4a87',
  },
  dark: {
    bg: '#0d1117', ink: '#e6edf3', soft: '#9da7b3', faint: '#56606c', ghost: '#77818d', hair: '#262c34',
    blue: '#4c9be8', red: '#f0604d', brain: '#141a22', brainEdge: '#3a434e', band: '#3b444f',
    calm: '#1d2c3f', active: '#9ccbff',
  },
};

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
//  Geometry: the brain seen from above, front up, in px at scale 1
// ---------------------------------------------------------------------------
const N = 90;
const MM = 1.9;
const Y0 = (Math.min(...OUTLINE.map((p) => p[1])) + Math.max(...OUTLINE.map((p) => p[1]))) / 2;
const P = REGIONS.map(([, x, y]) => [MM * x, -MM * (y - Y0)]);
const OUT = OUTLINE.map(([x, y]) => [MM * x, -MM * (y - Y0)]);
const line = (i, j) => `M${num(P[i][0])} ${num(P[i][1])}L${num(P[j][0])} ${num(P[j][1])}`;
const across = (i, j) => (i % 2) !== (j % 2);       // homologous regions are listed in pairs: left, right
const widthOf = (w) => Math.max(0.5, 11 * w);       // line width of a link of rescaled weight w

// ---------------------------------------------------------------------------
//  One connectome: simulation, weights of the rescaled network, node measures
// ---------------------------------------------------------------------------
function prepare(W) {
  const res = esn.memoryCapacity(W, {}, true);      // seed 42: the same input and Win for every network
  const Wn = res.Wn;
  const edges = [];                                 // [rescaled weight, i, j, share of the squared weight]
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
  // squared eigenvalues relative to the largest: their sum is the stable rank
  const sq = Array.from(esn.symEigenvalues(Wn, N), (v) => v * v).sort((a, b) => b - a);
  return { W, res, Wn, edges, k, s, Y, rs: esn.stableRank(W, N), dirs: sq.map((v) => v / sq[0]) };
}

// the same wiring with every weight between the hemispheres multiplied by f
function scaleAcross(W, f) {
  const M = Float64Array.from(W);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (across(i, j)) M[i * N + j] *= f;
  return M;
}

// a region with a strong two-step, three-step and four-step loop whose links are long enough to be seen
function pickLoops(W) {
  const mm = (i, j) => Math.hypot(REGIONS[i][1] - REGIONS[j][1], REGIONS[i][2] - REGIONS[j][2]);
  const w = (i, j) => W[i * N + j], far = (i, j) => w(i, j) > 0 && mm(i, j) >= 22;
  let best = null;
  for (let c = 0; c < N; c++) {
    const nb = []; for (let j = 0; j < N; j++) if (far(c, j)) nb.push(j);
    let b2 = -1, b3 = null, b4 = null;
    for (const j of nb) if (b2 < 0 || w(c, j) > w(c, b2)) b2 = j;
    for (const j of nb) for (const h of nb) if (h > j && far(j, h)) {
      const sc = Math.min(w(c, j), w(j, h), w(h, c));
      if (!b3 || sc > b3[0]) b3 = [sc, j, h];
    }
    for (const j of nb) for (const l of nb) if (l > j && w(j, l) === 0) for (let h = 0; h < N; h++) {
      if (h === c || h === j || h === l || !far(j, h) || !far(h, l)) continue;
      const sc = Math.min(w(c, j), w(j, h), w(h, l), w(l, c));
      if (!b4 || sc > b4[0]) b4 = [sc, j, h, l];
    }
    if (b2 < 0 || !b3 || !b4) continue;
    const score = Math.min(w(c, b2), b3[0], b4[0]);
    if (!best || score > best.score) best = { score, c, loops: [[c, b2, c], [c, b3[1], b3[2], c], [c, b4[1], b4[2], b4[3], c]] };
  }
  return best;
}

const use = (id, a = {}) => el('use', { href: `#${id}`, ...a });

// ---------------------------------------------------------------------------
function build(WA, WB, opts = {}) {
  const C = THEMES[opts.theme || 'light'];
  const text = (x, y, s, o = {}) => rawText(x, y, s, { fill: C.ink, colours: C, ...o });
  const A = prepare(WA), B = prepare(WB);
  const H = prepare(scaleAcross(WA, 0.5));          // A with the weights between the hemispheres halved
  const TAU = A.res.params.tauMax, u = A.res.u, t0 = A.res.testStart;

  // ---- timeline: start of each scene, in seconds at the pace it was composed at ----
  const S1 = 0, S2 = 8.5, S3 = 15.5, S4 = 28.5, S5 = 40, S6 = 54, S7 = 61.5, S8 = 69.5, S9 = 84.5, S10 = 93.5,
    S11 = 99.5, S12 = 108, S13 = 116, S14 = 127.5, END = 135;
  const L = END + 0.8;
  const loop = new Loop(L, opts.speed || 1.25);
  const defs = [], out = [];
  const push = (s) => out.push(s);
  const show = (a, b, inner, f = 0.6) => el('g', { opacity: 0 }, inner + loop.fade([[a, b]], f));
  const side = (a, b, y, s, o = {}) => show(a, b, text(o.x || 40, y, s, { size: 20, ...o }));

  // ---- headlines: one sentence per scene --------------------------------------------
  [
    [S1, S2, 'The {blue|connectome}: the brain as a network'],
    [S2, S3, 'Every connection has a {blue|weight}'],
    [S3, S4, '{blue|Disparity}: how a region spreads its strength'],
    [S4, S5, '{blue|Loops}: activity leaves a region and comes back'],
    [S5, S6, '{blue|Stable rank}: the network’s independent directions'],
    [S6, S7, 'Now the {blue|connectome} becomes a neural network'],
    [S7, S8, 'A random signal is fed into every region'],
    [S8, S9, 'Can its present state {red|recall the past} signal?'],
    [S9, S10, 'Recall fades with the delay: the {red|forgetting curve}'],
    [S10, S11, 'The area under it is the {red|memory capacity}'],
    [S11, S12, 'Different {blue|weights}, different {red|memory}'],
    [S12, S13, 'Its heaviest links join the two {blue|hemispheres}'],
    [S13, S14, 'Halve them, same wiring: it {red|remembers more}'],
  ].forEach(([a, b, s]) => push(show(a + 0.5, b - 0.25, text(40, 66, s, { size: 29, weight: 700 }), 0.4)));

  // ---- static layers of a network, in local coordinates (origin at the centre of the brain) ----
  const layers = (net, id) => {
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
    const still = el('g', { id }, use(`${id}o`) + use(`${id}w`, { opacity: 0.1 }) + use(`${id}s`) + use(`${id}n`, { opacity: 0.4 }));
    return [outline, binary, weighted, share, nodes, still];
  };
  defs.push(...layers(A, 'a'), ...layers(B, 'b'));

  // ====================================================================================
  //  The connectome followed through the figure (A): one group that moves and rescales
  // ====================================================================================
  const BIG = [650, 300, 1.08], PAIR = [250, 205, 0.56], PIPE = [500, 300, 0.85];
  const stars = [S3 + 2.9, S4 + 0.3];               // A is off screen while the stars of scene 3 are shown
  const jump = S5 + 0.25;                           // and changes place between scenes 4 and 5
  const move = [S6 + 0.4, S6 + 2.0];                // from scene 5 to the network of scene 6
  const RATE = 4, ta = S7 + 1.2, tb = S9;           // reservoir steps shown per second, and when
  const base = t0 + 200;                            // shown steps lie in the held-out part of the run
  const steps = Math.floor((tb - ta) * RATE);

  // the two regions of the disparity scene: the most even and the most uneven shares of strength
  {
    const e = Array.from(A.k, (k, i) => ({ i, k, ups: k * A.Y[i] })).filter((r) => r.k >= 8).sort((p, q) => p.ups - q.ups);
    A.lo = e[0].i; A.hi = e[e.length - 1].i;
  }
  const loops = pickLoops(WA);

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
    const tone = (z) => hex3(mix(C.calm, C.active, Math.max(0, Math.min(1, (z + 2) / 4))));
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

    // scene 4: loops of two, three and four steps that leave one region and return to it
    let walks = '';
    {
      const STEP = 0.75, starts = [S4 + 1.6, S4 + 3.9, S4 + 6.9];
      loops.loops.forEach((seq, m) => {
        const a = starts[m], pts = seq.map((i) => P[i]);
        const stops = seq.map((_, q) => a + 0.25 + q * STEP), b = stops[stops.length - 1];
        const track = (attr, d) => loop.linear(attr, [[0, pts[0][d]], ...stops.map((t, q) => [t, pts[q][d]]), [L, pts[0][d]]]);
        walks += el('path', { d: polyline(pts), opacity: 0 }, loop.fade([[a, S5 - 0.2]], 0.4))
          + seq.slice(1, -1).map((i) => el('circle', { cx: num(P[i][0]), cy: num(P[i][1]), r: 5, fill: C.blue, stroke: C.bg, 'stroke-width': 1.2, opacity: 0 }, loop.fade([[a, S5 - 0.2]], 0.4))).join('')
          + el('circle', { cx: num(pts[0][0]), cy: num(pts[0][1]), r: 6.5, fill: C.bg, stroke: C.ink, 'stroke-width': 2.4, opacity: 0 },
            track('cx', 0) + track('cy', 1) + loop.linear('opacity', [[0, 0], [a + 0.15, 0], [a + 0.3, 1], [b, 1], [b + 0.25, 0], [L, 0]]));
      });
      const c = P[loops.c];
      walks = el('g', { fill: 'none', stroke: C.blue, 'stroke-width': 3, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, walks)
        + el('circle', { cx: num(c[0]), cy: num(c[1]), r: 8, fill: C.blue, stroke: C.bg, 'stroke-width': 1.6 });
    }

    const len = pathLength(OUT.concat([OUT[0]]));
    const dim = [S3 + 0.5, S3 + 1.2];
    const inner =
      // outline: drawn at the start, gone once the connectome is used as a network
      el('g', { opacity: 1 },
        el('use', { href: '#ao', 'stroke-dasharray': num(len), 'stroke-dashoffset': num(len) },
          loop.linear('stroke-dashoffset', [[0, len], [0.4, len], [1.9, 0], [L, 0]], true))
        + loop.linear('opacity', [[0, 1], [move[0], 1], [move[1], 0], [L, 0]]))
      + el('g', { opacity: 0 }, use('ab') + loop.fade([[S1 + 4.6, S2 + 1.0]], 1.2))
      + el('g', { opacity: 0 }, use('aw') + loop.linear('opacity',
        [[0, 0], [S2 + 0.4, 0], [S2 + 1.6, 1], [dim[0], 1], [dim[1], 0.13], [jump - 0.05, 0.13], [jump, 0.1], [move[0], 0.1], [move[1], 1], [L, 1]]))
      + el('g', { opacity: 0 }, use('as') + loop.fade([[jump + 1.2, move[0] + 0.4]], 0.8))
      + el('g', { stroke: C.bg, 'stroke-width': 1.1, fill: C.blue }, nodes + loop.linear('opacity',
        [[0, 1], [dim[0], 1], [dim[1], 0.13], [stars[1] - 0.1, 0.13], [stars[1], 0.25], [jump - 0.05, 0.25], [jump, 0.4], [move[0], 0.4], [move[1], 1], [L, 1]]))
      + el('g', { opacity: 0, stroke: C.blue, 'stroke-linecap': 'round', fill: 'none' }, ego + loop.fade([[S3 + 1.0, stars[0]]], 0.7))
      + el('g', { opacity: 0 }, walks + loop.fade([[S4 + 0.9, S5 - 0.2]], 0.6));
    const at = (p) => [[0, ...BIG], [jump - 0.05, ...BIG], [jump, ...PAIR], [move[0], ...PAIR], [move[1], ...PIPE], [L, ...PIPE]].map((r) => p(r));
    groupA = el('g', { opacity: 1 },
      el('g', { transform: `translate(${BIG[0]} ${BIG[1]})` },
        el('g', { transform: `scale(${BIG[2]})` }, inner + loop.scale(at(([t, , , s]) => [t, s])))
        + loop.translate(at(([t, x, y]) => [t, x, y])))
      + loop.fade([[0, stars[0]], [stars[1], S5 - 0.25], [S5 + 0.75, S9 - 0.2]], 0.7));
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
  {
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
      return el('g', { stroke: C.blue, fill: 'none', 'stroke-linecap': 'round' }, g)
        + el('g', { fill: C.faint, stroke: C.bg, 'stroke-width': 1.2 }, tips)
        + el('circle', { r: 12, fill: C.blue, stroke: C.bg, 'stroke-width': 2 });
    };
    const SL = [250, 262], SH = [705, 262];
    push(side(S3 + 1.2, stars[0], 290, 'one region and its connections'));
    push(show(S3 + 3.4, S4 - 0.1, el('g', { transform: `translate(${SL[0]} ${SL[1]})` }, star(A.lo))
      + text(SL[0], 420, 'spread evenly', { size: 20, anchor: 'middle', fill: C.soft })
      + text(SL[0], 452, 'low disparity', { size: 22, anchor: 'middle', weight: 700 })));
    push(show(S3 + 8.0, S4 - 0.1, el('g', { transform: `translate(${SH[0]} ${SH[1]})` }, star(A.hi))
      + text(SH[0], 420, 'one connection takes most', { size: 20, anchor: 'middle', fill: C.soft })
      + text(SH[0], 452, 'high disparity', { size: 22, anchor: 'middle', weight: 700 })));
  }

  // ---- scene 4: three lines beside the loops ----------------------------------------------
  push(side(S4 + 1.6, S5 - 0.1, 250, 'back in two steps, in three, in four…'));
  push(side(S4 + 7.6, S5 - 0.1, 290, '{blue|subgraph centrality} adds up all the loops'));
  push(side(S4 + 9.2, S5 - 0.1, 330, 'strong connections make strong loops'));

  // ====================================================================================
  //  Scene 5: stable rank. Each bar is one independent direction of the network (an
  //  eigenvector); its height is the squared eigenvalue relative to the largest. The
  //  sum of the bars is the stable rank
  // ====================================================================================
  {
    const NBAR = 22, PITCH = 10.5, BASE = 392, HMAX = 62;
    const bars = (net, cx, a) => {
      let g = '';
      for (let m = 0; m < NBAR; m++) {
        const h = Math.max(1.5, HMAX * net.dirs[m]);
        g += el('rect', { x: num(cx - NBAR * PITCH / 2 + m * PITCH), y: num(BASE - h), width: 7.5, height: num(h), rx: 1.5, opacity: 0 },
          loop.fade([[a + 0.07 * m, S6 + 0.2]], 0.3));
      }
      return el('g', { fill: C.blue }, g) + show(a - 0.2, S6 + 0.2, el('line', { x1: cx - NBAR * PITCH / 2 - 4, x2: cx + NBAR * PITCH / 2, y1: BASE + 0.5, y2: BASE + 0.5, stroke: C.soft, 'stroke-width': 1 }));
    };
    const cap = (cx, a, net, s2) => show(a, S6 + 0.1,
      text(cx, 432, `about ${Math.round(net.rs)} directions`, { size: 20, anchor: 'middle', fill: C.soft })
      + text(cx, 464, s2, { size: 22, anchor: 'middle', weight: 700 }));
    const XB = 710;
    push(bars(A, PAIR[0], S5 + 2.0));
    push(side(S5 + 2.2, S5 + 7.6, 190, 'each bar is one independent direction', { x: 500 }));
    push(side(S5 + 3.6, S5 + 7.6, 228, 'its height, the weight it carries', { x: 500 }));
    push(cap(PAIR[0], S5 + 5.2, A, 'low stable rank'));
    push(show(S5 + 8.0, S6 + 0.2, el('g', { transform: `translate(${XB} ${PAIR[1]}) scale(${PAIR[2]})` }, use('b')), 0.9));
    push(bars(B, XB, S5 + 9.0));
    push(cap(XB, S5 + 11.0, B, 'high stable rank'));
  }

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

  // ---- a forgetting-curve plot: axes and helpers shared by scenes 9 to 11 and by scene 13 ----
  const plot = (x0, x1, yT, yB, withRecall) => {
    const sx = scale(0, TAU + 0.5, x0, x1), sy = scale(0, 1.04, yB, yT);
    let ax = el('path', { d: `M${x0} ${yT - 6}V${yB}H${x1}`, fill: 'none', stroke: C.ink, 'stroke-width': 1.4 });
    for (const v of [1, 5, 10, 15, 20]) ax += el('line', { x1: num(sx(v)), x2: num(sx(v)), y1: yB, y2: yB + 6, stroke: C.ink, 'stroke-width': 1.4 })
      + text(sx(v), yB + 28, String(v), { size: 18, anchor: 'middle', fill: C.soft });
    ax += text((x0 + x1) / 2, yB + 60, 'steps back', { size: 20, anchor: 'middle' });
    ax += el('line', { x1: x0 - 6, x2: x0, y1: num(sy(1)), y2: num(sy(1)), stroke: C.ink, 'stroke-width': 1.4 })
      + text(x0 - 14, sy(1) + 6, 'perfect', { size: 18, anchor: 'end', fill: C.soft })
      + text(x0 - 14, sy(0) + 6, 'none', { size: 18, anchor: 'end', fill: C.soft });
    if (withRecall) ax += text(x0 - 14, (yT + yB) / 2 + 6, 'recall', { size: 20, anchor: 'end' });
    const curve = (net) => Array.from({ length: TAU }, (_, i) => [sx(i + 1), sy(net.res.r2[i])]);
    const lineOf = (c, col) => el('path', { d: polyline(c), fill: 'none', stroke: col, 'stroke-width': 2.6, 'stroke-linejoin': 'round' });
    const dotsOf = (c, col, when) => el('g', { fill: col, stroke: C.bg, 'stroke-width': 1.6 },
      c.map(([x, y], i) => el('circle', { cx: num(x), cy: num(y), r: 5, ...(when ? { opacity: 0 } : {}) }, when ? loop.fade([[when(i + 1), L]], 0.25) : '')).join(''));
    const reveal = (id, pts) => defs.push(el('clipPath', { id }, el('rect', { x: x0, y: yT - 12, width: 0, height: yB - yT + 12 }, loop.linear('width', pts.concat([[L, x1 - x0]])))));
    return { sx, sy, ax, curve, lineOf, dotsOf, reveal, x0, x1, yT, yB };
  };

  // ====================================================================================
  //  Scenes 9 to 11: the forgetting curve, its area, and a second connectome
  // ====================================================================================
  {
    const q = plot(170, 850, 138, 440, true);
    const ca = q.curve(A), cb = q.curve(B);
    push(show(S9 + 0.4, S12 - 0.2, q.ax));
    // points come in one delay at a time; the line follows through a growing clip
    const t1 = S9 + 1.6, dt = 0.3, mark = (tau) => t1 + (tau - 1) * dt;
    const grow = [[0, 0]];
    for (let tau = 1; tau <= TAU; tau++) grow.push([mark(tau), q.sx(tau) + 2 - q.x0]);
    q.reveal('ca', grow);
    q.reveal('cb', [[0, 0], [S11 + 2.4, 0], [S11 + 4.6, q.x1 - q.x0]]);
    push(el('path', { d: polyline(ca) + `L${num(q.sx(TAU))} ${q.yB}L${num(q.sx(1))} ${q.yB}Z`, fill: C.red, opacity: 0 },
      loop.linear('opacity', [[0, 0], [S10 + 0.4, 0], [S10 + 1.4, 0.2], [S11 + 0.4, 0.2], [S11 + 1.4, 0], [L, 0]])));
    push(show(S10 + 1.2, S11 + 0.2, text((q.sx(1) + q.sx(7)) / 2, q.sy(0.42), 'memory capacity', { size: 21, weight: 700, anchor: 'middle' })));
    // connectome A: red while it is the subject, grey once the second one arrives
    push(show(S9 + 1.0, S11 + 1.0, el('g', { 'clip-path': 'url(#ca)' }, q.lineOf(ca, C.red)) + q.dotsOf(ca, C.red, mark), 0.8));
    push(show(S11 + 0.6, S12 - 0.2, q.lineOf(ca, C.ghost) + q.dotsOf(ca, C.ghost), 0.8));
    push(show(S11 + 2.2, S12 - 0.2, el('g', { 'clip-path': 'url(#cb)' }, q.lineOf(cb, C.red) + q.dotsOf(cb, C.red)), 0.4));
    // the two connectomes of scene 5, as the key
    const key = (x, id, col, s, a) => show(a, S12 - 0.2,
      el('g', { transform: `translate(${x} 182) scale(0.3)` }, use(id))
      + el('line', { x1: x - 22, x2: x + 22, y1: 254, y2: 254, stroke: col, 'stroke-width': 4, 'stroke-linecap': 'round' })
      + text(x, 282, s, { size: 18, anchor: 'middle', fill: C.soft }));
    push(key(622, 'a', C.ghost, 'few directions', S11 + 0.8));
    push(key(786, 'b', C.red, 'many directions', S11 + 2.4));
  }

  // ====================================================================================
  //  Scenes 12 and 13: where the weight sits. The heaviest links, in blue when they join
  //  the two hemispheres; then the same wiring with every weight between the hemispheres
  //  halved, and its forgetting curve. Both pictures use one width scale (line width
  //  proportional to the weight), so the halved links are drawn half as wide
  // ====================================================================================
  {
    const rho = esn.spectralRadius(WA, N);
    // the fifteen heaviest links of the connectome, coloured by where they run; the rest as faint wiring
    const heavy = A.edges.slice(-15);
    const picture = (W, id) => {
      let within = '', between = '';
      for (const [, i, j] of heavy) {
        const p = el('path', { d: line(i, j), 'stroke-width': num(widthOf(0.99 * W[i * N + j] / rho), 2) });
        if (across(i, j)) between += p; else within += p;
      }
      return el('g', { id, fill: 'none', 'stroke-linecap': 'round' },
        use('ao') + use('ab', { opacity: 0.3 })
        + el('g', { stroke: C.soft }, within) + el('g', { stroke: C.blue }, between)
        + use('an', { opacity: 0.3 }));
    };
    defs.push(picture(WA, 'h1'), picture(H.W, 'h2'));
    const HP = [232, 300, 0.95];
    const at = `translate(${HP[0]} ${HP[1]}) scale(${HP[2]})`;
    push(show(S12 + 0.5, S13 + 1.4, el('g', { transform: at }, use('h1')), 1.0));
    push(show(S13 + 1.0, S14 - 0.2, el('g', { transform: at }, use('h2')), 1.0));
    push(show(S12 + 0.9, S14 - 0.2, text(HP[0] - 62, 492, 'left', { size: 18, anchor: 'middle', fill: C.soft })
      + text(HP[0] + 62, 492, 'right', { size: 18, anchor: 'middle', fill: C.soft })));
    push(show(S12 + 2.0, S13 + 0.1, el('line', { x1: 470, x2: 500, y1: 263, y2: 263, stroke: C.blue, 'stroke-width': 4, 'stroke-linecap': 'round' })
      + text(512, 270, 'between the two hemispheres', { size: 20 })));
    push(show(S12 + 3.6, S13 + 0.1, el('line', { x1: 470, x2: 500, y1: 303, y2: 303, stroke: C.soft, 'stroke-width': 4, 'stroke-linecap': 'round' })
      + text(512, 310, 'within one hemisphere', { size: 20 })));

    const q = plot(570, 916, 160, 404, false);
    const ca = q.curve(A), ch = q.curve(H);
    q.reveal('ch', [[0, 0], [S13 + 4.4, 0], [S13 + 6.4, q.x1 - q.x0]]);
    push(show(S13 + 2.4, S14 - 0.2, q.ax + q.lineOf(ca, C.ghost) + q.dotsOf(ca, C.ghost)));
    push(show(S13 + 4.2, S14 - 0.2, el('g', { 'clip-path': 'url(#ch)' }, q.lineOf(ch, C.red) + q.dotsOf(ch, C.red)), 0.4));
    const kx = q.sx(12.2);
    push(show(S13 + 2.8, S14 - 0.2, el('line', { x1: kx, x2: kx + 30, y1: 173, y2: 173, stroke: C.ghost, 'stroke-width': 4, 'stroke-linecap': 'round' })
      + text(kx + 40, 179, 'as it is', { size: 18, fill: C.soft })));
    push(show(S13 + 4.4, S14 - 0.2, el('line', { x1: kx, x2: kx + 30, y1: 201, y2: 201, stroke: C.red, 'stroke-width': 4, 'stroke-linecap': 'round' })
      + text(kx + 40, 207, 'halved', { size: 18, fill: C.soft })));
  }

  // ---- scene 14: what it means -------------------------------------------------------------
  push(show(S14 + 0.5, END, text(480, 248, 'Memory depends on the {blue|weights}, not only on the wiring.', { size: 27, weight: 700, anchor: 'middle' }), 0.8));
  push(show(S14 + 2.6, END, text(480, 300, 'From birth to old age, it follows the weight between the hemispheres.', { size: 21, anchor: 'middle', fill: C.soft }), 0.8));

  // ---- progress ---------------------------------------------------------------------------
  push(el('rect', { x: 0, y: 536, width: 0, height: 4, fill: C.hair }, loop.linear('width', [[0, 0], [END, 960], [L, 960]])));

  const Wd = 960, Ht = 540;
  const body = el('g', { opacity: 0 }, out.join('\n') + loop.linear('opacity', [[0, 0], [0.5, 1], [END + 0.2, 1], [L - 0.05, 0], [L, 0]]));
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${Wd} ${Ht}" width="${Wd}" height="${Ht}" font-family="${FONT}" font-weight="500" role="img" aria-labelledby="t d">`
    + '<title id="t">From a human connectome to its memory capacity</title>'
    + '<desc id="d">An animated explanation in fourteen scenes. The connectome is a network of 90 brain regions linked by nerve fibres, and every link has a weight. '
    + 'Disparity measures how evenly a region spreads its strength over its connections; subgraph centrality adds up the loops by which activity returns to a region; '
    + 'the stable rank counts the independent directions that carry the weight of the whole network. The connectome is then used as the fixed recurrent layer of a neural network: '
    + 'a random signal is fed into every region and a trained readout tries to recall the signal a number of steps back. Recall fades with the delay; this is the forgetting curve, '
    + 'and the area under it is the memory capacity. Two connectomes with different weights have different memory capacities. In the example the heaviest links join the two hemispheres, '
    + 'and halving those weights on the same wiring raises the memory capacity.</desc>'
    + `<style>${fontFace()}</style>`
    + el('defs', {}, defs.join(''))
    + el('rect', { x: 0.5, y: 0.5, width: Wd - 1, height: Ht - 1, rx: 12, fill: C.bg, stroke: C.hair })
    + body + '</svg>\n';
  const name = (i) => REGIONS[i][0];
  return { svg, seconds: L / (opts.speed || 1.25),
    stats: {
      A: { rs: A.rs, MC: A.res.MC, r2: Array.from(A.res.r2) }, B: { rs: B.rs, MC: B.res.MC }, H: { rs: H.rs, MC: H.res.MC },
      disparity: [name(A.lo), name(A.hi)], loops: loops.loops.map((s) => s.map(name).join(' > ')),
    } };
}

module.exports = { build, THEMES };

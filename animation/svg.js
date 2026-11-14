// ============================================================================
//  svg.js — small helpers to write animated SVG (SMIL) by hand. No dependencies.
//
//  Every animation shares one loop of length L seconds (dur = L, repeatCount =
//  indefinite) and is described by keyframes on that loop, so the whole figure
//  stays in sync and can be seeked to any time.
// ============================================================================
'use strict';

const num = (x, d = 1) => {
  const s = (+x).toFixed(d);
  return s.indexOf('.') >= 0 ? s.replace(/\.?0+$/, '') || '0' : s;
};
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function attrs(a) {
  return Object.entries(a || {})
    .filter(([, v]) => v !== undefined && v !== null && v !== false)
    .map(([k, v]) => ` ${k}="${typeof v === 'number' ? num(v, 2) : String(v).replace(/"/g, '&quot;')}"`)
    .join('');
}
const el = (name, a, inner) => (inner === undefined || inner === null || inner === '')
  ? `<${name}${attrs(a)}/>` : `<${name}${attrs(a)}>${inner}</${name}>`;

// ---------------------------------------------------------------------------
//  Text with light markup:  **bold**   {colour|words}
//  colours is a map from the names used in the markup to fills.
// ---------------------------------------------------------------------------
function rich(str, colours = {}) {
  let out = '', last = 0, m;
  const re = /\*\*(.+?)\*\*|\{(\w+)\|(.+?)\}/g;
  while ((m = re.exec(str))) {
    out += esc(str.slice(last, m.index));
    if (m[1] !== undefined) out += el('tspan', { 'font-weight': 700 }, esc(m[1]));
    else out += el('tspan', { fill: colours[m[2]] }, esc(m[3]));
    last = re.lastIndex;
  }
  return out + esc(str.slice(last));
}

function text(x, y, str, o = {}) {
  const a = { x: num(x, 1), y: num(y, 1), 'font-size': o.size || 19, fill: o.fill, 'text-anchor': o.anchor,
    'font-weight': o.weight, opacity: o.opacity, transform: o.transform };
  return el('text', a, rich(str, o.colours) + (o.inner || ''));
}

// ---------------------------------------------------------------------------
//  Keyframe tracks on the common loop.
// ---------------------------------------------------------------------------
const EASE = '0.45 0 0.2 1';

class Loop {
  constructor(L) { this.L = L; }

  _clean(pts) {
    // sort, clamp to [0, L], make times strictly increasing, pin both ends
    const L = this.L;
    const p = pts.map(([t, v]) => [Math.min(L, Math.max(0, t)), v]).sort((a, b) => a[0] - b[0]);
    if (p[0][0] > 0) p.unshift([0, p[0][1]]);
    if (p[p.length - 1][0] < L) p.push([L, p[p.length - 1][1]]);
    const out = [p[0]];
    for (let i = 1; i < p.length; i++) {
      let t = p[i][0];
      const prev = out[out.length - 1];
      if (t <= prev[0]) {
        if (p[i][1] === prev[1]) continue;
        t = prev[0] + 1e-4 * L;
      }
      out.push([Math.min(t, L), p[i][1]]);
    }
    out[out.length - 1][0] = L;
    for (let i = out.length - 2; i > 0; i--) if (out[i][0] >= out[i + 1][0]) out[i][0] = out[i + 1][0] - 1e-5 * L;
    return out;
  }

  _track(tag, base, pts, ease) {
    const p = this._clean(pts);
    const a = {
      ...base, dur: `${this.L}s`, repeatCount: 'indefinite',
      keyTimes: p.map(([t]) => num(t / this.L, 5)).join(';'),
      values: p.map(([, v]) => (typeof v === 'number' ? num(v, 3) : v)).join(';'),
    };
    if (ease) { a.calcMode = 'spline'; a.keySplines = Array(p.length - 1).fill(EASE).join(';'); }
    return el(tag, a);
  }

  // piecewise-linear track of one attribute; ease = true for motion
  linear(attr, pts, ease = false) { return this._track('animate', { attributeName: attr }, pts, ease); }

  // step track: value v from time t on
  discrete(attr, pts) {
    const p = pts.map(([t, v]) => [Math.min(this.L, Math.max(0, t)), v]).sort((a, b) => a[0] - b[0]);
    if (p[0][0] > 0) p.unshift([0, p[p.length - 1][1]]);
    const out = [p[0]];
    for (let i = 1; i < p.length; i++) if (p[i][0] > out[out.length - 1][0] && p[i][0] < this.L) out.push(p[i]);
    return el('animate', {
      attributeName: attr, dur: `${this.L}s`, repeatCount: 'indefinite', calcMode: 'discrete',
      keyTimes: out.map(([t]) => num(t / this.L, 5)).join(';'),
      values: out.map(([, v]) => (typeof v === 'number' ? num(v, 3) : v)).join(';'),
    });
  }

  // opacity hi inside the windows [a, b], lo outside; cross-fades of length f centred on the edges
  fade(windows, f = 0.5, lo = 0, hi = 1) {
    const pts = [[0, lo]];
    for (const [a, b] of windows) {
      if (a <= 0) pts[0] = [0, hi]; else pts.push([a - f / 2, lo], [a + f / 2, hi]);
      if (b >= this.L) pts.push([this.L, hi]); else pts.push([b - f / 2, hi], [b + f / 2, lo]);
    }
    return this.linear('opacity', pts);
  }

  // eased translate / scale tracks: pts = [[t, x, y]] or [[t, s]]
  translate(pts, ease = true) {
    return this._track('animateTransform', { attributeName: 'transform', type: 'translate' },
      pts.map(([t, x, y]) => [t, `${num(x, 2)} ${num(y || 0, 2)}`]), ease);
  }

  scale(pts, ease = true) {
    return this._track('animateTransform', { attributeName: 'transform', type: 'scale' },
      pts.map(([t, s]) => [t, num(s, 4)]), ease);
  }
}

// ---------------------------------------------------------------------------
//  Geometry and colour
// ---------------------------------------------------------------------------
const scale = (d0, d1, r0, r1) => (v) => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
const polyline = (pts, d = 1) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${num(x, d)} ${num(y, d)}`).join('');
const pathLength = (pts) => pts.reduce((s, p, i) => (i ? s + Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) : 0), 0);

function hexToRgb(h) { const v = parseInt(h.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b);
  return '#' + A.map((c, i) => Math.round(c + (B[i] - c) * t).toString(16).padStart(2, '0')).join('');
}
// short #rgb form (keeps long colour tracks small)
function hex3(h) { return '#' + hexToRgb(h).map((c) => Math.round(c / 17).toString(16)).join(''); }

module.exports = { num, esc, el, attrs, rich, text, Loop, scale, polyline, pathLength, mix, hex3 };

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
//  Text with light markup:  *italic*   **bold**   _{sub}   ^{sup}
// ---------------------------------------------------------------------------
function rich(str, fs) {
  let out = '', back = 0;
  const emit = (txt, st) => {
    if (!txt) return;
    const a = {};
    if (st.i) a['font-style'] = 'italic';
    if (st.b) a['font-weight'] = 700;
    let dy = back; back = 0;
    if (st.sub) { dy += 0.26 * fs; back = -0.26 * fs; a['font-size'] = num(0.72 * fs, 1); }
    if (st.sup) { dy -= 0.36 * fs; back = 0.36 * fs; a['font-size'] = num(0.72 * fs, 1); }
    if (dy) a.dy = num(dy, 2);
    out += Object.keys(a).length ? el('tspan', a, esc(txt)) : esc(txt);
  };
  const re = /\*\*(.+?)\*\*|\*(.+?)\*|_\{(.+?)\}|\^\{(.+?)\}/g;
  let last = 0, m;
  const inner = (s) => {
    // italic inside sub/superscripts written as _{*x*}
    const mm = /^\*(.+)\*$/.exec(s);
    return mm ? [mm[1], true] : [s, false];
  };
  while ((m = re.exec(str))) {
    emit(str.slice(last, m.index), {});
    if (m[1] !== undefined) emit(m[1], { b: true });
    else if (m[2] !== undefined) emit(m[2], { i: true });
    else if (m[3] !== undefined) { const [t, it] = inner(m[3]); emit(t, { sub: true, i: it }); }
    else if (m[4] !== undefined) { const [t, it] = inner(m[4]); emit(t, { sup: true, i: it }); }
    last = re.lastIndex;
  }
  emit(str.slice(last), {});
  if (back) out += el('tspan', { dy: num(back, 2) }, '​');
  return out;
}

function text(x, y, str, o = {}) {
  const fs = o.size || 13;
  const a = { x: num(x, 1), y: num(y, 1), 'font-size': fs, fill: o.fill, 'text-anchor': o.anchor,
    'font-weight': o.weight, opacity: o.opacity, transform: o.transform, 'letter-spacing': o.spacing };
  // halo: an outline in the surface colour, painted under the glyphs, for labels that sit on data
  if (o.halo) Object.assign(a, { stroke: o.halo, 'stroke-width': 4, 'stroke-linejoin': 'round', 'paint-order': 'stroke' });
  return el('text', a, rich(str, fs) + (o.inner || ''));
}

// ---------------------------------------------------------------------------
//  Keyframe tracks on the common loop.
// ---------------------------------------------------------------------------
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

  // piecewise-linear track
  linear(attr, pts, extra = {}) {
    const p = this._clean(pts);
    return el('animate', {
      attributeName: attr, dur: `${this.L}s`, repeatCount: 'indefinite',
      keyTimes: p.map(([t]) => num(t / this.L, 5)).join(';'),
      values: p.map(([, v]) => (typeof v === 'number' ? num(v, 3) : v)).join(';'), ...extra,
    });
  }

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

  // opacity 1 inside the windows [a, b], 0 outside; cross-fades of length f centred on the edges
  fade(windows, f = 0.16, lo = 0, hi = 1) {
    const pts = [[0, lo]];
    for (const [a, b] of windows) {
      if (a <= 0) pts[0] = [0, hi]; else pts.push([a - f / 2, lo], [a + f / 2, hi]);
      if (b >= this.L) pts.push([this.L, hi]); else pts.push([b - f / 2, hi], [b + f / 2, lo]);
    }
    return this.linear('opacity', pts);
  }

  // hard on/off
  show(windows) {
    const pts = [[0, 0]];
    for (const [a, b] of windows) {
      if (a <= 0) pts[0] = [0, 1]; else pts.push([a, 1]);
      if (b < this.L) pts.push([b, 0]);
    }
    return this.discrete('opacity', pts);
  }

  translate(pts, extra = {}) {
    const p = this._clean(pts.map(([t, x, y]) => [t, `${num(x, 2)} ${num(y || 0, 2)}`]));
    return el('animateTransform', {
      attributeName: 'transform', type: 'translate', dur: `${this.L}s`, repeatCount: 'indefinite',
      keyTimes: p.map(([t]) => num(t / this.L, 5)).join(';'), values: p.map(([, v]) => v).join(';'), ...extra,
    });
  }
}

// ---------------------------------------------------------------------------
//  Geometry
// ---------------------------------------------------------------------------
const scale = (d0, d1, r0, r1) => (v) => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
const polyline = (pts, d = 1) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${num(x, d)} ${num(y, d)}`).join('');
// many dots as one path (round caps on zero-length segments)
const dots = (pts, d = 1) => pts.map(([x, y]) => `M${num(x, d)} ${num(y, d)}h.01`).join('');

function hexToRgb(h) { const v = parseInt(h.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b);
  return '#' + A.map((c, i) => Math.round(c + (B[i] - c) * t).toString(16).padStart(2, '0')).join('');
}
// short #rgb form (keeps long colour tracks small)
function hex3(h) { return '#' + hexToRgb(h).map((c) => Math.round(c / 17).toString(16)).join(''); }

module.exports = { num, esc, el, attrs, rich, text, Loop, scale, polyline, dots, mix, hex3 };

// ============================================================================
//  fig_lifespan.js — stable rank and memory capacity of every connectome.
//
//  Input: the table written by population.js (age, cohort, stable rank and MC of
//  each connectome). Three panels are filled cohort by cohort:
//    a  stable rank against age        b  MC against stable rank        c  MC against age
//  Every number printed in the figure is computed here from that table.
//
//  build(rows, { theme }) returns the SVG as a string.
// ============================================================================
'use strict';
const { num, el, text, Loop, scale, polyline, dots } = require('./svg.js');

const COHORTS = [                // order of the release; colours of the figures of the paper (tab10)
  { id: 1, name: 'dHCP' }, { id: 2, name: 'BCP' }, { id: 3, name: 'CALM' }, { id: 4, name: 'RED' }, { id: 5, name: 'ACE' },
  { id: 6, name: 'HCPd' }, { id: 7, name: 'HCPya' }, { id: 8, name: 'camCAN' }, { id: 9, name: 'HCPa' },
];
const THEMES = {
  light: {
    bg: '#ffffff', frame: '#d0d7de', ink: '#1f2328', ink2: '#59636e', muted: '#8c959f', grid: '#e7eaed',
    cohort: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
  },
  dark: {
    bg: '#0d1117', frame: '#30363d', ink: '#e6edf3', ink2: '#9da7b3', muted: '#6e7681', grid: '#21262d',
    cohort: ['#4c9be0', '#ff9a3c', '#4cc267', '#f0605d', '#b98576', '#f092d0', '#a3a9b0', '#d2d34a', '#3fd0e0'],
  },
};
const AGE_SPLIT = 32;            // development / aging split used in the paper

const pearson = (a, b) => {
  const n = a.length; let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let va = 0, vb = 0, c = 0;
  for (let i = 0; i < n; i++) { va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; c += (a[i] - ma) * (b[i] - mb); }
  return c / Math.sqrt(va * vb);
};
const signed = (r) => (r < 0 ? '−' : '+') + Math.abs(r).toFixed(2);

// mean at each age, then a centred 5-year running mean of those means
function runningMean(rows, key) {
  const by = new Map();
  for (const r of rows) { const a = Math.round(r.age); if (!by.has(a)) by.set(a, []); by.get(a).push(r[key]); }
  const ages = [...by.keys()].sort((a, b) => a - b);
  const mean = new Map(ages.map((a) => [a, by.get(a).reduce((s, v) => s + v, 0) / by.get(a).length]));
  return ages.map((a) => {
    let s = 0, c = 0;
    for (let d = -2; d <= 2; d++) if (mean.has(a + d)) { s += mean.get(a + d); c++; }
    return [a, s / c];
  });
}

function build(rows, opts = {}) {
  const C = THEMES[opts.theme || 'light'];
  const FONT = '-apple-system,BlinkMacSystemFont,\'Segoe UI\',\'Noto Sans\',Helvetica,Arial,sans-serif';
  const Wd = 1040, Ht = 478;

  // ---- statistics printed in the figure ---------------------------------------
  const rs = rows.map((r) => r.rs), lr = rs.map(Math.log), mc = rows.map((r) => r.MC), age = rows.map((r) => r.age);
  const dev = rows.map((r, i) => i).filter((i) => age[i] <= AGE_SPLIT), old = rows.map((r, i) => i).filter((i) => age[i] > AGE_SPLIT);
  const pick = (v, idx) => idx.map((i) => v[i]);
  const st = {
    n: rows.length,
    r_mc_logrs: pearson(mc, lr),
    r_mc_age_dev: pearson(pick(mc, dev), pick(age, dev)), r_mc_age_old: pearson(pick(mc, old), pick(age, old)),
    r_rs_age_dev: pearson(pick(rs, dev), pick(age, dev)), r_rs_age_old: pearson(pick(rs, old), pick(age, old)),
    mc_min: Math.min(...mc), mc_max: Math.max(...mc), mc_median: [...mc].sort((a, b) => a - b)[Math.floor(mc.length / 2)],
  };
  // least-squares line MC = a + b ln r_s
  {
    const n = rows.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sx += lr[i]; sy += mc[i]; sxx += lr[i] * lr[i]; sxy += lr[i] * mc[i]; }
    st.slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); st.icpt = (sy - st.slope * sx) / n;
  }
  const rmMC = runningMean(rows, 'MC'), rmRS = runningMean(rows, 'rs');
  st.age_min_mc = rmMC.reduce((m, p) => (p[1] < m[1] ? p : m))[0];
  st.age_min_rs = rmRS.reduce((m, p) => (p[1] < m[1] ? p : m))[0];

  // ---- timeline -----------------------------------------------------------------
  const TC = 1.0, DC = 1.3;                         // cohorts come in one by one
  const tIn = (k) => TC + k * DC;
  const T1 = tIn(COHORTS.length) + 0.4;             // all connectomes shown
  const T2 = T1 + 3.6;                              // b: the line in log r_s
  const TH = T2 + 9.0;                              // a, c: lifespan course, then hold
  const L = TH + 0.9;
  const loop = new Loop(L);

  const out = [], defs = [];
  const push = (s) => out.push(s);

  // ---- header: cohorts and captions ------------------------------------------------
  const counts = COHORTS.map((c) => rows.filter((r) => r.dataset === c.id).length);
  {
    let x = 24;
    COHORTS.forEach((c, k) => {
      push(el('g', { opacity: 0.25 },
        el('circle', { cx: x + 5, cy: 26, r: 4.6, fill: C.cohort[k] }) + text(x + 15, 30.5, c.name, { size: 13, fill: C.ink, weight: 600 })
        + loop.fade([[tIn(k), L]], 0.4, 0.25, 1)));
      x += 26 + c.name.length * 8.2;
    });
    let acc = 0;
    COHORTS.forEach((c, k) => {
      acc += counts[k];
      const b = k === COHORTS.length - 1 ? L : tIn(k + 1);
      push(el('g', { opacity: 0 },
        text(Wd - 24, 30.5, `${acc.toLocaleString('en-US')} connectomes`, { size: 13.5, fill: C.ink, anchor: 'end', weight: 700 }) + loop.show([[tIn(k), b]])));
    });
    push(el('rect', { x: 24, y: 45, width: Wd - 48, height: 1, fill: C.grid }));
    const caps = [
      { win: [[0, T1]], cap: [
        `Each point is one of ${rows.length.toLocaleString('en-US')} structural connectomes (ages 0–90, nine cohorts), used as the fixed recurrent layer of the same echo-state network.`,
        'Its stable rank *r*_{s} = ‖**W**‖_{F}^{2} / ρ(**W**)^{2} is read from the weights; its memory capacity MC is measured as in the figure above.'] },
      { win: [[T1, T2]], cap: [
        `Memory follows the stable rank: MC rises with *r*_{s} with diminishing returns, close to a straight line in log *r*_{s} (*r* = ${st.r_mc_logrs.toFixed(2)}).`,
        'Summed over the network, the leading term of the walk expansion is the stable rank: Σ_{*i*} *s*_{*i*}^{2}*Y*_{*i*} = ρ(**W**)^{2} *r*_{s}.'] },
      { win: [[T2, L]], cap: [
        'Across the lifespan the stable rank falls through development, reaches its minimum in the early thirties and rises again through aging.',
        'Memory capacity follows the same course. Before adulthood, the trajectory compares different cohorts.'] },
    ];
    for (const c of caps)
      push(el('g', { opacity: 0 }, text(24, 67, c.cap[0], { size: 14.2, fill: C.ink }) + text(24, 88, c.cap[1], { size: 14.2, fill: C.ink2 }) + loop.fade(c.win, 0.4)));
  }

  // ---- panels -----------------------------------------------------------------------
  const P0 = 140, P1 = 410;                         // top / bottom of the plot areas
  const sAge = (x0, x1) => scale(-3, 93, x0, x1);
  const sRs = (r0, r1) => { const f = scale(Math.log(2.5), Math.log(10.5), r0, r1); return (v) => f(Math.log(v)); };
  const sMC = (r0, r1) => scale(6.3, 13.1, r0, r1);
  const panels = [
    { key: 'a', title: 'Stable rank across the lifespan', x0: 70, x1: 340, sx: sAge, sy: sRs, fx: (r) => r.age, fy: (r) => r.rs,
      xt: [0, 20, 40, 60, 80], yt: [3, 4, 6, 8, 10], xl: 'age (years)', yl: 'stable rank *r*_{s}' },
    { key: 'b', title: 'Stable rank predicts memory', x0: 410, x1: 680, sx: sRs, sy: sMC, fx: (r) => r.rs, fy: (r) => r.MC,
      xt: [3, 4, 6, 8, 10], yt: [7, 8, 9, 10, 11, 12], xl: 'stable rank *r*_{s}', yl: 'memory capacity MC' },
    { key: 'c', title: 'Memory across the lifespan', x0: 750, x1: 1020, sx: sAge, sy: sMC, fx: (r) => r.age, fy: (r) => r.MC,
      xt: [0, 20, 40, 60, 80], yt: [7, 8, 9, 10, 11, 12], xl: 'age (years)', yl: 'memory capacity MC' },
  ];
  for (const p of panels) {
    const sx = p.sx(p.x0, p.x1), sy = p.sy(P1, P0);
    let g = text(p.x0 - 40, P0 - 16, `**${p.key}**`, { size: 15, fill: C.ink }) + text(p.x0 - 22, P0 - 16, p.title, { size: 13.2, fill: C.ink, weight: 600 });
    for (const v of p.yt) {
      g += el('line', { x1: p.x0, x2: p.x1, y1: num(sy(v)), y2: num(sy(v)), stroke: C.grid, 'stroke-width': 1 });
      g += text(p.x0 - 7, sy(v) + 4, String(v), { size: 11.5, fill: C.ink2, anchor: 'end' });
    }
    for (const v of p.xt) g += text(sx(v), P1 + 16, String(v), { size: 11.5, fill: C.ink2, anchor: 'middle' });
    g += el('path', { d: `M${p.x0} ${P0}V${P1}H${p.x1}`, fill: 'none', stroke: C.muted, 'stroke-width': 1 });
    g += text((p.x0 + p.x1) / 2, P1 + 35, p.xl, { size: 12.2, fill: C.ink2, anchor: 'middle' });
    g += text(p.x0 - 34, (P0 + P1) / 2, p.yl, { size: 12.2, fill: C.ink2, anchor: 'middle', transform: `rotate(-90 ${p.x0 - 34} ${(P0 + P1) / 2})` });

    // points, one path per cohort
    let pts = '';
    COHORTS.forEach((c, k) => {
      const d = dots(rows.filter((r) => r.dataset === c.id).map((r) => [sx(p.fx(r)), sy(p.fy(r))]));
      pts += el('path', { d, stroke: C.cohort[k], opacity: 0 }, loop.fade([[tIn(k) + 0.25, L]], 0.5, 0, 0.5));
    });
    g += el('g', { fill: 'none', 'stroke-width': 2.6, 'stroke-linecap': 'round' }, pts);

    const halo = (d, a) => el('path', { d, fill: 'none', stroke: C.bg, 'stroke-width': 5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: 0.85, ...a })
      + el('path', { d, fill: 'none', stroke: C.ink, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', ...a });
    const reveal = (id, a, b) => defs.push(el('clipPath', { id }, el('rect', { x: p.x0, y: P0 - 4, width: 0, height: P1 - P0 + 4 },
      loop.linear('width', [[0, 0], [a, 0], [b, p.x1 - p.x0], [TH + 0.5, p.x1 - p.x0], [L, p.x1 - p.x0]]))));
    const note = (x, y, s, a, anchor) => el('g', { opacity: 0 }, text(x, y, s, { size: 12.6, fill: C.ink, weight: 600, anchor, halo: C.bg }) + loop.fade([[a, TH + 0.3]], 0.5));

    if (p.key === 'b') {
      const xs = [Math.min(...rs), Math.max(...rs)];
      reveal('revb', T1 + 0.3, T1 + 1.8);
      g += el('g', { 'clip-path': 'url(#revb)' }, halo(polyline(xs.map((v) => [sx(v), sy(st.icpt + st.slope * Math.log(v))]))));
      g += note(p.x1 - 6, P1 - 30, `*r* = ${st.r_mc_logrs.toFixed(2)} with log *r*_{s}`, T1 + 1.6, 'end');
      g += note(p.x1 - 6, P1 - 12, `*R*^{2} = ${(st.r_mc_logrs ** 2).toFixed(2)}, ${st.n.toLocaleString('en-US')} connectomes`, T1 + 1.6, 'end');
    } else {
      const rm = p.key === 'a' ? rmRS : rmMC;
      const id = `rev${p.key}`;
      reveal(id, T2 + 0.4, T2 + 2.6);
      g += el('g', { opacity: 0 },
        el('line', { x1: num(sx(AGE_SPLIT)), x2: num(sx(AGE_SPLIT)), y1: P0, y2: P1, stroke: C.ink2, 'stroke-width': 1, 'stroke-dasharray': '3 3' })
        + text(sx(AGE_SPLIT) + 5, P0 + 11, `${AGE_SPLIT} y`, { size: 11.5, fill: C.ink2 })
        + loop.fade([[T2 + 0.2, TH + 0.3]], 0.5));
      g += el('g', { 'clip-path': `url(#${id})` }, halo(polyline(rm.map(([a, v]) => [sx(a), sy(v)]))));
      const rd = p.key === 'a' ? st.r_rs_age_dev : st.r_mc_age_dev, ro = p.key === 'a' ? st.r_rs_age_old : st.r_mc_age_old;
      g += note(sx(AGE_SPLIT) - 8, P1 - 12, `*r* = ${signed(rd)}`, T2 + 2.4, 'end');
      g += note(sx(AGE_SPLIT) + 10, P1 - 12, `*r* = ${signed(ro)}`, T2 + 2.4);
    }
    push(g);
  }

  const body = el('g', { opacity: 0 }, out.join('\n') + loop.linear('opacity', [[0, 0], [0.5, 1], [TH + 0.1, 1], [L - 0.05, 0], [L, 0]]));
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${Wd} ${Ht}" width="${Wd}" height="${Ht}" font-family="${FONT}" role="img" aria-labelledby="t d">`
    + `<title id="t">Stable rank and memory capacity of ${st.n} human connectomes across the lifespan</title>`
    + `<desc id="d">Three scatter plots of ${st.n} connectomes from nine cohorts: stable rank against age, memory capacity against stable rank `
    + `(Pearson r = ${st.r_mc_logrs.toFixed(2)} with the logarithm of the stable rank) and memory capacity against age `
    + `(r = ${st.r_mc_age_dev.toFixed(2)} up to ${AGE_SPLIT} years and ${st.r_mc_age_old.toFixed(2)} after).</desc>`
    + el('defs', {}, defs.join(''))
    + el('rect', { x: 0.5, y: 0.5, width: Wd - 1, height: Ht - 1, rx: 14, fill: C.bg, stroke: C.frame })
    + body + '</svg>\n';
  return { svg, L, stats: st };
}

module.exports = { build, THEMES, COHORTS };

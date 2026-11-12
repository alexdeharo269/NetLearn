#!/usr/bin/env node
// ============================================================================
//  build.js — writes the animated figures of the README into assets/.
//
//    node animation/build.js --connectome engine/data/connectome_one.csv \
//                            [--population population.csv] [--check engine/trace_mccurve.csv]
//
//  --connectome  one connectome in the engine CSV format (notebooks/02 writes the
//                subject of the paradigm figure to engine/data/connectome_one.csv)
//  --label       text under the reservoir title (e.g. the cohort and age)
//  --population  the table written by animation/population.js; adds the lifespan figure
//  --check       trace_mccurve.csv written by engine/esn_trace for the same connectome:
//                prints the largest difference between the C++ and the JS forgetting curve
//  --out         output folder (default: assets)
// ============================================================================
'use strict';
const fs = require('fs');
const path = require('path');
const esn = require('./esn.js');

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
if (!args.connectome && !args.population) {
  console.error('usage: node animation/build.js --connectome <one.csv> [--label <text>] [--population <population.csv>] [--check <trace_mccurve.csv>] [--out assets]');
  process.exit(1);
}
const outDir = args.out || path.join(__dirname, '..', 'assets');
fs.mkdirSync(outDir, { recursive: true });
const write = (name, svg) => {
  const f = path.join(outDir, name);
  fs.writeFileSync(f, svg);
  console.log(`  ${f}  (${(svg.length / 1024).toFixed(0)} kB)`);
};

if (args.connectome) {
  const rows = fs.readFileSync(args.connectome, 'utf8').trim().split(/\r?\n/);
  const { id, W } = esn.parseConnectomeLine(rows[1], 90);
  const { build } = require('./fig_memory.js');
  let stats;
  for (const theme of ['light', 'dark']) {
    const r = build(W, { theme, label: args.label });
    stats = r.stats;
    write(`memory_paradigm_${theme}.svg`, r.svg);
  }
  console.log(`connectome ${id}: stable rank ${stats.rs.toFixed(3)}, MC ${stats.MC.toFixed(3)}`
    + ` | uniform null: stable rank ${stats.rsu.toFixed(3)}, MC ${stats.MCu.toFixed(3)}`);
  if (args.check) {
    const cpp = fs.readFileSync(args.check, 'utf8').trim().split(/\r?\n/).slice(1).map((l) => +l.split(',')[1]);
    let d = 0;
    for (let i = 0; i < stats.r2.length; i++) d = Math.max(d, Math.abs(cpp[i] - stats.r2[i]));
    console.log(`C++ engine vs JS port: max |r2(tau) difference| = ${d.toExponential(1)} over tau = 1..${stats.r2.length}`);
  }
}

if (args.population) {
  const { build } = require('./fig_lifespan.js');
  const rows = fs.readFileSync(args.population, 'utf8').trim().split(/\r?\n/);
  const head = rows[0].split(',');
  const col = (name) => head.indexOf(name);
  const data = rows.slice(1).map((r) => {
    const c = r.split(',');
    return { age: +c[col('age')], dataset: +c[col('dataset')], rs: +c[col('stable_rank')], MC: +c[col('MC')] };
  });
  let stats;
  for (const theme of ['light', 'dark']) {
    const r = build(data, { theme });
    stats = r.stats;
    write(`lifespan_${theme}.svg`, r.svg);
  }
  console.log(`${stats.n} connectomes: MC ${stats.mc_min.toFixed(2)}-${stats.mc_max.toFixed(2)} (median ${stats.mc_median.toFixed(2)})`
    + ` | r(MC, log rs) = ${stats.r_mc_logrs.toFixed(3)}`
    + ` | r(MC, age) = ${stats.r_mc_age_dev.toFixed(3)} up to 32 y, ${stats.r_mc_age_old.toFixed(3)} after`
    + ` | r(rs, age) = ${stats.r_rs_age_dev.toFixed(3)}, ${stats.r_rs_age_old.toFixed(3)}`
    + ` | minimum of the 5-year running mean of MC at ${stats.age_min_mc} y`);
}

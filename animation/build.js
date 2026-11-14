#!/usr/bin/env node
// ============================================================================
//  build.js — writes the animated figure of the README into assets/.
//
//    node animation/build.js --connectome <a.csv> --contrast <b.csv> [--check <trace_mccurve.csv>]
//
//  --connectome  the connectome followed through the figure, in the engine CSV format
//                (notebooks/02 writes the subject of the paradigm figure to
//                engine/data/connectome_one.csv)
//  --contrast    a second connectome in the same format, shown next to the first
//  --check       trace_mccurve.csv written by engine/esn_trace for the first connectome:
//                prints the largest difference between the C++ and the JS forgetting curve
//  --out         output folder (default: assets)
//
//  Both connectomes must follow the region order of animation/atlas.js.
// ============================================================================
'use strict';
const fs = require('fs');
const path = require('path');
const esn = require('./esn.js');
const { build } = require('./fig_approach.js');

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
if (!args.connectome || !args.contrast) {
  console.error('usage: node animation/build.js --connectome <a.csv> --contrast <b.csv> [--check <trace_mccurve.csv>] [--out assets]');
  process.exit(1);
}
const read = (file) => esn.parseConnectomeLine(fs.readFileSync(file, 'utf8').trim().split(/\r?\n/)[1], 90);
const a = read(args.connectome), b = read(args.contrast);
const { svg, L, stats } = build(a.W, b.W);

const outDir = args.out || path.join(__dirname, '..', 'assets');
fs.mkdirSync(outDir, { recursive: true });
const file = path.join(outDir, 'approach.svg');
fs.writeFileSync(file, svg);
console.log(`${file}  (${(svg.length / 1024).toFixed(0)} kB, loop of ${L.toFixed(1)} s)`);
console.log(`connectome ${a.id}: stable rank ${stats.A.rs.toFixed(2)}, MC ${stats.A.MC.toFixed(2)}`
  + ` | connectome ${b.id}: stable rank ${stats.B.rs.toFixed(2)}, MC ${stats.B.MC.toFixed(2)}`);
console.log(`regions of the disparity scene: ${stats.A.lo} (low), ${stats.A.hi} (high)`);
if (args.check) {
  const cpp = fs.readFileSync(args.check, 'utf8').trim().split(/\r?\n/).slice(1).map((l) => +l.split(',')[1]);
  let d = 0;
  for (let i = 0; i < stats.A.r2.length; i++) d = Math.max(d, Math.abs(cpp[i] - stats.A.r2[i]));
  console.log(`C++ engine vs JS port: max |r2(tau) difference| = ${d.toExponential(1)} over tau = 1..${stats.A.r2.length}`);
}

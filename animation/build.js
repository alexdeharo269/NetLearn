#!/usr/bin/env node
// ============================================================================
//  build.js — writes the animated figure of the README into assets/, in a light
//  and a dark version.
//
//    node animation/build.js --connectome <a.csv> --contrast <b.csv> [--check <trace_mccurve.csv>]
//
//  --connectome  the connectome followed through the figure, in the engine CSV format
//                (notebooks/02 writes the subject of the paradigm figure to
//                engine/data/connectome_one.csv)
//  --contrast    a second connectome in the same format, shown next to the first
//  --check       trace_mccurve.csv written by engine/esn_trace for the first connectome:
//                prints the largest difference between the C++ and the JS forgetting curve
//  --speed       playback speed relative to the pace the scenes were composed at (default 1.4)
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
  console.error('usage: node animation/build.js --connectome <a.csv> --contrast <b.csv> [--check <trace_mccurve.csv>] [--speed 1.4] [--out assets]');
  process.exit(1);
}
const read = (file) => esn.parseConnectomeLine(fs.readFileSync(file, 'utf8').trim().split(/\r?\n/)[1], 90);
const a = read(args.connectome), b = read(args.contrast);
const outDir = args.out || path.join(__dirname, '..', 'assets');
fs.mkdirSync(outDir, { recursive: true });

let last;
for (const theme of ['light', 'dark']) {
  last = build(a.W, b.W, { theme, speed: args.speed ? +args.speed : undefined });
  const file = path.join(outDir, `approach_${theme}.svg`);
  fs.writeFileSync(file, last.svg);
  console.log(`${file}  (${(last.svg.length / 1024).toFixed(0)} kB, loop of ${last.seconds.toFixed(1)} s)`);
}
const s = last.stats;
console.log(`connectome ${a.id}: stable rank ${s.A.rs.toFixed(2)}, MC ${s.A.MC.toFixed(2)}`
  + ` | with the weights between the hemispheres halved: stable rank ${s.H.rs.toFixed(2)}, MC ${s.H.MC.toFixed(2)}`);
console.log(`connectome ${b.id}: stable rank ${s.B.rs.toFixed(2)}, MC ${s.B.MC.toFixed(2)}`);
console.log(`disparity scene: ${s.disparity[0]} (low), ${s.disparity[1]} (high)`);
console.log(`loops scene: ${s.loops.join(' | ')}`);
if (args.check) {
  const cpp = fs.readFileSync(args.check, 'utf8').trim().split(/\r?\n/).slice(1).map((l) => +l.split(',')[1]);
  let d = 0;
  for (let i = 0; i < s.A.r2.length; i++) d = Math.max(d, Math.abs(cpp[i] - s.A.r2[i]));
  console.log(`C++ engine vs JS port: max |r2(tau) difference| = ${d.toExponential(1)} over tau = 1..${s.A.r2.length}`);
}

#!/usr/bin/env node
// ============================================================================
//  population.js — stable rank and memory capacity of every connectome.
//
//    node animation/population.js <connectomes.csv> <metadata.csv> <out.csv>
//
//  connectomes.csv : the engine format written by notebooks/02 (export_connectomes):
//                    subject_id, w0, ..., w8099  (one row per connectome)
//  metadata.csv    : a table with the columns sid, age, dataset
//                    (src/export_data.py writes one)
//  out.csv         : sid, age, dataset, stable_rank, MC
//
//  Same run as esn_mc: one realisation per connectome, seed = 42 + 1000 * sid.
// ============================================================================
'use strict';
const fs = require('fs');
const os = require('os');
const readline = require('readline');
const { Worker, isMainThread, parentPort } = require('worker_threads');
const esn = require('./esn.js');

const N = 90, SEED = 42;

if (!isMainThread) {
  parentPort.on('message', (line) => {
    const { id, W } = esn.parseConnectomeLine(line, N);
    const res = esn.memoryCapacity(W, { seed: (SEED + 1000 * id) >>> 0 });
    parentPort.postMessage({ id, rs: esn.stableRank(W, N), MC: res.MC, r2: Array.from(res.r2) });
  });
} else {
  const [conPath, metaPath, outPath] = process.argv.slice(2);
  if (!conPath || !metaPath || !outPath) {
    console.error('usage: node animation/population.js <connectomes.csv> <metadata.csv> <out.csv>');
    process.exit(1);
  }
  const meta = new Map();
  {
    const rows = fs.readFileSync(metaPath, 'utf8').trim().split(/\r?\n/);
    const head = rows[0].split(',');
    const iS = head.indexOf('sid'), iA = head.indexOf('age'), iD = head.indexOf('dataset');
    if (iS < 0 || iA < 0 || iD < 0) throw new Error('metadata needs the columns sid, age, dataset');
    for (const r of rows.slice(1)) {
      const c = r.split(',');
      meta.set(parseInt(c[iS], 10), { age: +c[iA], dataset: parseInt(c[iD], 10) });
    }
  }

  const nWorkers = Math.max(1, os.cpus().length);
  const results = [];
  const queue = [];
  let done = 0, closed = false, pendingJobs = 0;
  const rl = readline.createInterface({ input: fs.createReadStream(conPath), crlfDelay: Infinity });
  const idle = [];

  function finish() {
    results.sort((a, b) => a.id - b.id);
    const lines = ['sid,age,dataset,stable_rank,MC,' + Array.from({ length: 20 }, (_, i) => `r2_${i + 1}`).join(',')];
    for (const r of results) {
      const m = meta.get(r.id) || { age: NaN, dataset: NaN };
      lines.push([r.id, m.age, m.dataset, r.rs.toFixed(6), r.MC.toFixed(6), ...r.r2.map((v) => v.toFixed(5))].join(','));
    }
    fs.writeFileSync(outPath, lines.join('\n') + '\n');
    console.error(`wrote ${outPath} (${results.length} connectomes)`);
    process.exit(0);
  }
  function pump(w) {
    if (queue.length) { pendingJobs++; w.postMessage(queue.shift()); if (queue.length < 2 * nWorkers) rl.resume(); }
    else if (closed) { if (pendingJobs === 0) finish(); }
    else idle.push(w);
  }
  for (let k = 0; k < nWorkers; k++) {
    const w = new Worker(__filename);
    w.on('message', (res) => {
      results.push(res); pendingJobs--;
      if (++done % 200 === 0) console.error(`  ${done}`);
      pump(w);
    });
    idle.push(w);
  }
  let first = true;
  rl.on('line', (line) => {
    if (first) { first = false; return; }          // header
    if (!line) return;
    queue.push(line);
    if (idle.length) pump(idle.pop());
    if (queue.length >= 4 * nWorkers) rl.pause();
  });
  rl.on('close', () => {
    closed = true;
    while (idle.length) pump(idle.pop());
  });
}

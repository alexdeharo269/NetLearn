// ============================================================================
//  esn.js — JavaScript port of engine/esn.hpp (no dependencies).
//
//  Same conventions as the C++ engine:
//    * reservoir update : r(t) = tanh( Win u(t) + W r(t-1) ),  r(0) = 0
//    * W is the connectome, diagonal zeroed, rescaled to spectral radius rho
//    * input u(t) ~ Uniform(-1, 1), Win = input_scale * Uniform(-1, 1)
//    * readout : ridge regression with a bias column, ridge * input_scale^2
//    * score   : r^2(tau) between the test-set target u(t - tau) and its
//                reconstruction;  MC = sum_{tau = 1..tau_max} r^2(tau)
//
//  The random stream is the one of std::mt19937 + std::uniform_real_distribution
//  (libstdc++ / libc++), so a given seed draws the same input and the same Win
//  as the C++ programs.
// ============================================================================
'use strict';

// ---------------------------------------------------------------------------
//  std::mt19937
// ---------------------------------------------------------------------------
class MT19937 {
  constructor(seed) {
    this.mt = new Uint32Array(624);
    this.idx = 624;
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const x = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(x, 1812433253) + i) >>> 0;
    }
  }
  next() {
    const mt = this.mt;
    if (this.idx >= 624) {
      for (let i = 0; i < 624; i++) {
        const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
        mt[i] = mt[(i + 397) % 624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      }
      this.idx = 0;
    }
    let y = mt[this.idx++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
  // std::uniform_real_distribution<double>(-1, 1): generate_canonical<double, 53>
  // takes two 32-bit draws (low word first) and maps [0, 1) to [-1, 1).
  uniform() {
    const lo = this.next(), hi = this.next();
    let c = (lo + hi * 4294967296) / 18446744073709551616;
    if (c >= 1) c = 1 - Number.EPSILON / 2;
    return c * 2 + -1;
  }
}

// ---------------------------------------------------------------------------
//  Symmetric eigenvalues (cyclic Jacobi). W is a row-major Float64Array, n x n.
// ---------------------------------------------------------------------------
function symEigenvalues(W, n) {
  const a = Float64Array.from(W);
  let norm2 = 0;
  for (let i = 0; i < n * n; i++) norm2 += a[i] * a[i];
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    if (off <= 1e-30 * norm2) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (apq === 0) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
      }
    }
  }
  const ev = new Float64Array(n);
  for (let i = 0; i < n; i++) ev[i] = a[i * n + i];
  return ev.sort().reverse();
}

function spectralRadius(W, n) {
  const ev = symEigenvalues(W, n);
  return Math.max(Math.abs(ev[0]), Math.abs(ev[n - 1]));
}

// Zero diagonal, rescale so that the spectral radius equals rho (rescale_spectral).
function rescaleSpectral(W, n, rho) {
  const out = Float64Array.from(W);
  for (let i = 0; i < n; i++) out[i * n + i] = 0;
  const sr = spectralRadius(out, n);
  if (sr > 0) for (let i = 0; i < n * n; i++) out[i] *= rho / sr;
  return out;
}

// Stable rank r_s = ||W||_F^2 / rho(W)^2; scale-invariant.
function stableRank(W, n) {
  let f2 = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) f2 += W[i * n + j] * W[i * n + j];
  const A = Float64Array.from(W);
  for (let i = 0; i < n; i++) A[i * n + i] = 0;
  const sr = spectralRadius(A, n);
  return f2 / (sr * sr);
}

// ---------------------------------------------------------------------------
//  Driven trajectory: (T x n) states, r(0) = 0 row included (run_states).
// ---------------------------------------------------------------------------
function runStates(Wn, Win, u, n) {
  const T = u.length;
  // compressed rows: connectomes are sparse (density ~ 0.2)
  const ptr = new Int32Array(n + 1), cols = [], vals = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) if (Wn[i * n + j] !== 0) { cols.push(j); vals.push(Wn[i * n + j]); }
    ptr[i + 1] = cols.length;
  }
  const C = Int32Array.from(cols), V = Float64Array.from(vals);
  const X = new Float64Array(T * n);
  for (let t = 1; t < T; t++) {
    const prev = (t - 1) * n, cur = t * n, ut = u[t];
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let k = ptr[i]; k < ptr[i + 1]; k++) acc += V[k] * X[prev + C[k]];
      X[cur + i] = Math.tanh(acc + Win[i] * ut);
    }
  }
  return X;
}

// ---------------------------------------------------------------------------
//  Cholesky factorisation / solve of a symmetric positive-definite p x p matrix.
// ---------------------------------------------------------------------------
function cholesky(A, p) {
  const L = new Float64Array(p * p);
  for (let j = 0; j < p; j++) {
    let d = A[j * p + j];
    for (let k = 0; k < j; k++) d -= L[j * p + k] * L[j * p + k];
    if (!(d > 0)) throw new Error('cholesky: matrix is not positive definite');
    const ljj = Math.sqrt(d);
    L[j * p + j] = ljj;
    for (let i = j + 1; i < p; i++) {
      let s = A[i * p + j];
      for (let k = 0; k < j; k++) s -= L[i * p + k] * L[j * p + k];
      L[i * p + j] = s / ljj;
    }
  }
  return L;
}

function cholSolve(L, b, p) {
  const y = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * p + k] * y[k];
    y[i] = s / L[i * p + i];
  }
  for (let i = p - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < p; k++) s -= L[k * p + i] * y[k];
    y[i] = s / L[i * p + i];
  }
  return y;
}

function pearsonR2(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let va = 0, vb = 0, cov = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    va += da * da; vb += db * db; cov += da * db;
  }
  return (va > 0 && vb > 0) ? (cov * cov) / (va * vb) : 0;
}

// ---------------------------------------------------------------------------
//  Memory capacity of one connectome.
//
//  params: { N, steps, washout, rho, ridge, trainRatio, tauMax, inputScale, seed }
//  Returns the forgetting curve r2[tau - 1] and MC; with keep = true also the
//  input, the states, the readout weights and the test-set reconstructions.
//  Follows mc_curve() (washout >= tauMax: one design matrix for every delay).
// ---------------------------------------------------------------------------
const DEFAULTS = {
  N: 90, steps: 10000, washout: 1000, rho: 0.99, ridge: 1e-6,
  trainRatio: 0.7, tauMax: 20, inputScale: 1e-5, seed: 42,
};

function memoryCapacity(W, params = {}, keep = false) {
  const p = { ...DEFAULTS, ...params };
  const n = p.N;
  if (p.washout < p.tauMax) throw new Error('washout must be >= tauMax');
  const ridge = p.ridge * p.inputScale * p.inputScale;   // regularisation relative to the input
  const rng = new MT19937(p.seed);
  const u = new Float64Array(p.steps);
  for (let t = 0; t < p.steps; t++) u[t] = rng.uniform();
  const Win = new Float64Array(n);
  for (let i = 0; i < n; i++) Win[i] = p.inputScale * rng.uniform();

  const Wn = rescaleSpectral(W, n, p.rho);
  const X = runStates(Wn, Win, u, n);

  const t0 = p.washout, m = p.steps - t0;
  const ntr = Math.min(Math.max(Math.round(p.trainRatio * m), 10), m - 10);
  const nte = m - ntr, q = n + 1;

  // A = Ztr' Ztr + ridge I, with Z = [1, r(t)]
  const A = new Float64Array(q * q);
  for (let k = 0; k < ntr; k++) {
    const row = (t0 + k) * n;
    A[0] += 1;
    for (let i = 0; i < n; i++) {
      const xi = X[row + i];
      A[(i + 1) * q] += xi;
      const base = (i + 1) * q + 1;
      for (let j = 0; j <= i; j++) A[base + j] += xi * X[row + j];
    }
  }
  for (let i = 0; i < q; i++) for (let j = i + 1; j < q; j++) A[i * q + j] = A[j * q + i];
  for (let i = 0; i < q; i++) A[i * q + i] += ridge;
  const L = cholesky(A, q);

  const r2 = new Float64Array(p.tauMax);
  const weights = [], yhatTest = [], yTest = [];
  const b = new Float64Array(q), yte = new Float64Array(nte), yhat = new Float64Array(nte);
  for (let tau = 1; tau <= p.tauMax; tau++) {
    b.fill(0);
    for (let k = 0; k < ntr; k++) {
      const y = u[t0 + k - tau], row = (t0 + k) * n;
      b[0] += y;
      for (let i = 0; i < n; i++) b[i + 1] += X[row + i] * y;
    }
    const w = cholSolve(L, b, q);
    for (let k = 0; k < nte; k++) {
      const row = (t0 + ntr + k) * n;
      let s = w[0];
      for (let i = 0; i < n; i++) s += w[i + 1] * X[row + i];
      yhat[k] = s; yte[k] = u[t0 + ntr + k - tau];
    }
    r2[tau - 1] = pearsonR2(yte, yhat);
    if (keep) { weights.push(w); yhatTest.push(Float64Array.from(yhat)); yTest.push(Float64Array.from(yte)); }
  }
  let MC = 0;
  for (let i = 0; i < p.tauMax; i++) MC += r2[i];
  const out = { r2, MC, params: p, ntr, nte, testStart: t0 + ntr };
  if (keep) Object.assign(out, { u, Win, Wn, X, weights, yhatTest, yTest });
  return out;
}

// ---------------------------------------------------------------------------
//  Uniform null (null_uniform): same topology, w_ij -> (s_i/k_i + s_j/k_j) / 2.
// ---------------------------------------------------------------------------
function nullUniform(W, n) {
  const mean = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, k = 0;
    for (let j = 0; j < n; j++) if (W[i * n + j] > 0) { s += W[i * n + j]; k++; }
    if (k > 0) mean[i] = s / k;
  }
  const M = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (W[i * n + j] > 0) { const v = 0.5 * (mean[i] + mean[j]); M[i * n + j] = v; M[j * n + i] = v; }
  return M;
}

// ---------------------------------------------------------------------------
//  IO: the engine's CSV  [subject_id, w0, ..., w_{N*N-1}]  (+ header), row-major.
// ---------------------------------------------------------------------------
function parseConnectomeLine(line, n) {
  const cells = line.split(',');
  if (cells.length < n * n + 1) throw new Error(`expected ${n * n + 1} columns, found ${cells.length}`);
  const W = new Float64Array(n * n);
  for (let k = 0; k < n * n; k++) W[k] = +cells[k + 1];
  for (let i = 0; i < n; i++) W[i * n + i] = 0;
  return { id: parseInt(cells[0], 10), W };
}

module.exports = {
  MT19937, DEFAULTS, symEigenvalues, spectralRadius, rescaleSpectral, stableRank,
  runStates, cholesky, cholSolve, pearsonR2, memoryCapacity, nullUniform, parseConnectomeLine,
};

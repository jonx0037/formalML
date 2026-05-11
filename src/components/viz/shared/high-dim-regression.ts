// =============================================================================
// high-dim-regression.ts
//
// Shared helpers for the high-dimensional-regression topic's viz components.
// Extracted from the §1 inline duplication once §4 brought the count of viz
// needing the same DGP-1 generator + ISTA solver to four (RidgeVsLassoCoefBars,
// LassoUCurve, LassoSolutionPath, future KktCorrelationHistogram). The §1
// OlsOverfitsAtHighDim viz uses a different design pool size (p = 200 vs 500)
// and Cholesky-based ridge OLS, so it stays inline.
//
// All exports are pure functions — no module-level state, deterministic outputs
// for a given seed.
// =============================================================================

import { softThreshold } from './proximalUtils';

// -----------------------------------------------------------------------------
// Deterministic RNG: mulberry32 + Box-Muller.
// -----------------------------------------------------------------------------

/** Mulberry32 PRNG: 32-bit state, period 2³², good distribution properties. */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller-based standard-normal sampler. Returns iid N(0, 1) draws. */
export function gaussianRng(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

// -----------------------------------------------------------------------------
// DGP-1: AR(1) Toeplitz design + sparse-Gaussian linear model.
//
// Per the brief: rows X_i ~ N(0, Σ) with Σ_jk = ρ^|j-k|; β*_j = 1 for j < s,
// 0 otherwise; y = X β* + ε with ε ~ N(0, σ² I).
// -----------------------------------------------------------------------------

export interface Dgp1Options {
  n: number;
  p: number;
  s: number;
  sigma: number;
  rho: number;
  seed: number;
}

export interface Dgp1Sample {
  X: Float64Array[]; // n rows × p cols
  y: Float64Array; // length n
  betaStar: Float64Array; // length p
}

export function generateDgp1(opts: Dgp1Options): Dgp1Sample {
  const { n, p, s, sigma, rho, seed } = opts;
  const rng = mulberry32(seed);
  const gauss = gaussianRng(rng);
  const X: Float64Array[] = [];
  const y = new Float64Array(n);
  const sqrt1mr2 = Math.sqrt(1 - rho * rho);
  // AR(1) row sampler: x_j = ρ x_{j-1} + sqrt(1-ρ²) z_j, z_j ~ N(0,1).
  // Marginal cov is Σ_jk = ρ^|j-k|, matching DGP-1 exactly.
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(p);
    row[0] = gauss();
    for (let j = 1; j < p; j++) row[j] = rho * row[j - 1] + sqrt1mr2 * gauss();
    X.push(row);
    let signal = 0;
    for (let j = 0; j < s; j++) signal += row[j];
    y[i] = signal + sigma * gauss();
  }
  const betaStar = new Float64Array(p);
  for (let j = 0; j < s; j++) betaStar[j] = 1;
  return { X, y, betaStar };
}

// -----------------------------------------------------------------------------
// Matrix-vector multiplications — pure-JS, no external deps.
// X is n × p stored as an array of length-p rows.
// -----------------------------------------------------------------------------

/** X^T r → length-p vector. r is length n. */
export function xtMul(X: Float64Array[], r: Float64Array): Float64Array {
  const n = X.length;
  const p = X[0].length;
  const out = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const ri = r[i];
    const row = X[i];
    for (let j = 0; j < p; j++) out[j] += row[j] * ri;
  }
  return out;
}

/** X v → length-n vector. v is length p. */
export function xMul(X: Float64Array[], v: Float64Array): Float64Array {
  const n = X.length;
  const p = X[0].length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = X[i];
    for (let j = 0; j < p; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

// -----------------------------------------------------------------------------
// Operator norm of XᵀX/n via power iteration.
// -----------------------------------------------------------------------------

/**
 * Estimate ‖XᵀX/n‖_op via 30-iteration power method on a random unit start.
 * Used to set the ISTA / FISTA step size η = 1/L safely (a 5% safety margin
 * is added to absorb power-iteration approximation error).
 */
export function operatorNorm(X: Float64Array[], iters = 30, seed = 7): number {
  const p = X[0].length;
  const n = X.length;
  let v = new Float64Array(p);
  const rng = mulberry32(seed);
  for (let j = 0; j < p; j++) v[j] = rng() - 0.5;
  let norm = 0;
  for (let j = 0; j < p; j++) norm += v[j] * v[j];
  norm = Math.sqrt(norm);
  for (let j = 0; j < p; j++) v[j] /= norm;
  let lambda = 0;
  for (let it = 0; it < iters; it++) {
    const Xv = xMul(X, v);
    const next = xtMul(X, Xv);
    let nNext = 0;
    for (let j = 0; j < p; j++) {
      next[j] /= n;
      nNext += next[j] * next[j];
    }
    nNext = Math.sqrt(nNext);
    lambda = nNext;
    for (let j = 0; j < p; j++) next[j] /= nNext;
    v = next;
  }
  return lambda * 1.05; // 5% safety margin so η = 1/L is a valid step size
}

// -----------------------------------------------------------------------------
// ISTA: iterative soft-thresholding for the lasso.
//
// β^{k+1} = S(β^k + (η/n) Xᵀ(y - X β^k), η λ)
//
// Step size η = 1/L. Optionally takes a `betaInit` for warm starts (saves
// many iterations along a λ-path; passing the previous-λ solution as the
// initial typically converges in ~10-30 iterations even with `iters` large).
// -----------------------------------------------------------------------------

export function lassoIsta(
  X: Float64Array[],
  y: Float64Array,
  lambda: number,
  L: number,
  iters: number,
  betaInit?: Float64Array,
): Float64Array {
  const n = X.length;
  const p = X[0].length;
  const eta = 1 / L;
  let beta: Float64Array;
  if (betaInit) {
    beta = new Float64Array(betaInit);
  } else {
    beta = new Float64Array(p);
  }
  for (let it = 0; it < iters; it++) {
    const Xb = xMul(X, beta);
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) r[i] = y[i] - Xb[i];
    const grad = xtMul(X, r);
    const z = new Array<number>(p);
    for (let j = 0; j < p; j++) z[j] = beta[j] + (eta / n) * grad[j];
    const stepped = softThreshold(z, eta * lambda);
    beta = new Float64Array(stepped);
  }
  return beta;
}

// -----------------------------------------------------------------------------
// λ_max in closed form: ‖Xᵀy/n‖_∞ — the smallest λ at which the all-zero
// vector satisfies the lasso KKT conditions.
// -----------------------------------------------------------------------------

export function lambdaMax(X: Float64Array[], y: Float64Array): number {
  const Xty = xtMul(X, y);
  const n = X.length;
  const p = X[0].length;
  let m = 0;
  for (let j = 0; j < p; j++) m = Math.max(m, Math.abs(Xty[j] / n));
  return m;
}

// -----------------------------------------------------------------------------
// Prediction MSE: (1/n) ‖X β - y‖² — for U-curve viz.
// -----------------------------------------------------------------------------

export function predictMse(X: Float64Array[], y: Float64Array, beta: Float64Array): number {
  const n = X.length;
  const Xb = xMul(X, beta);
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = y[i] - Xb[i];
    sse += r * r;
  }
  return sse / n;
}

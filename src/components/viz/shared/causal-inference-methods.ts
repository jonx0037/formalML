// =============================================================================
// causal-inference-methods.ts
//
// Shared math primitives for the causal-inference-methods topic's viz
// components. Ported from notebooks/causal-inference-methods/
// 01_causal_inference_methods.ipynb §§2–13.
//
// In-browser TS for all interactive viz (closed-form propensity scores,
// IRLS-fit logistic regression, OLS outcome regression, AIPW + TMLE + DML
// cross-fitting harness, Wald IV, front-door, T- and DR-learners, E-value,
// Cinelli–Hazlett RV). The §11 "random-forest nuisance" is approximated in
// the browser by a Nadaraya–Watson kernel smoother + cross-fitted ridge; the
// notebook documents the full RandomForestRegressor reference. The verify
// suite checks the proxy lands within ±0.05 of the notebook RF tau.
//
// All exports are pure functions — deterministic outputs for a given seed
// via mulberry32 (re-exported from normalizing-flows.ts).
//
// Source-of-truth notebook:
//   notebooks/causal-inference-methods/01_causal_inference_methods.ipynb
// =============================================================================

import { mulberry32, gaussianRng } from './normalizing-flows';

export { mulberry32, gaussianRng };

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/** Robinson DGP sample: covariates, treatment, outcome, plus oracle nuisances. */
export type RobinsonSample = {
  /** Row-major n×p covariate matrix. */
  X: Float64Array;
  /** Number of observations. */
  n: number;
  /** Number of covariates. */
  p: number;
  /** Binary treatment (0/1). */
  D: Int8Array;
  /** Observed outcome. */
  Y: Float64Array;
  /** Oracle propensity e(X_i). */
  ePropensity: Float64Array;
  /** Oracle outcome mean g(X_i). */
  gOracle: Float64Array;
};

/** Standard estimator return: point estimate + SE + 95% Wald CI. */
export type EstimatorResult = {
  tau: number;
  se: number;
  ciLow: number;
  ciHigh: number;
};

/** TMLE adds the targeting epsilons. */
export type TMLEResult = EstimatorResult & {
  epsilon1: number;
  epsilon0: number;
  /** Whether all targeted mu^*(X) values lie in [0, 1] (only meaningful for logistic submodel). */
  inUnitInterval?: boolean;
};

/** DML extras: per-fold nuisance predictions and per-unit EIF. */
export type DMLResult = EstimatorResult & {
  ePred: Float64Array;
  mu0Pred: Float64Array;
  mu1Pred: Float64Array;
  psi: Float64Array;
};

/** IPW extras: per-unit weight (used for stability diagnostics). */
export type IPWResult = EstimatorResult & {
  maxWeight: number;
  weights: Float64Array;
};

/** Wald IV extras: numerator/denominator and first-stage F. */
export type WaldResult = EstimatorResult & {
  reducedForm: number;
  firstStage: number;
  firstStageF: number;
};

/** Front-door extras: oracle comparison if U is known. */
export type FrontdoorResult = EstimatorResult & {
  oracleWithU?: number;
};

/** Sensitivity diagnostics. */
export type SensitivityResult = {
  ePoint: number;
  eCi: number;
  rvQ: number;
  rvQAlpha: number;
  dStandardized: number;
};

// -----------------------------------------------------------------------------
// Matrix / vector helpers (flat row-major Float64Array).
// X[i * p + j] is the (i, j) entry.
// -----------------------------------------------------------------------------

/** Compute mean of a Float64Array. */
export function mean(v: Float64Array | number[]): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i];
  return s / v.length;
}

/** Sample variance (n-1 denominator). */
export function variance(v: Float64Array | number[]): number {
  const n = v.length;
  if (n < 2) return 0;
  const m = mean(v);
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = v[i] - m;
    ss += d * d;
  }
  return ss / (n - 1);
}

/** Sample standard deviation (n-1 denominator). */
export function stddev(v: Float64Array | number[]): number {
  return Math.sqrt(variance(v));
}

/** Row-slice of a row-major matrix: copy row i of X (n×p) into a new Float64Array(p). */
export function getRow(X: Float64Array, i: number, p: number): Float64Array {
  const r = new Float64Array(p);
  const base = i * p;
  for (let j = 0; j < p; j++) r[j] = X[base + j];
  return r;
}

/** Slice rows indexed by `idx` into a new row-major n_idx × p matrix. */
export function selectRows(X: Float64Array, idx: number[] | Int32Array, p: number): Float64Array {
  const m = idx.length;
  const out = new Float64Array(m * p);
  for (let k = 0; k < m; k++) {
    const src = idx[k] * p;
    const dst = k * p;
    for (let j = 0; j < p; j++) out[dst + j] = X[src + j];
  }
  return out;
}

/** Slice scalar array indexed by `idx`. Works for both Float64Array and Int8Array. */
export function selectValues<T extends Float64Array | Int8Array>(v: T, idx: number[] | Int32Array): T {
  const m = idx.length;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const out = new (v.constructor as any)(m);
  for (let k = 0; k < m; k++) out[k] = v[idx[k]];
  return out as T;
}

/** X^T X for row-major X (n×p) → p×p row-major. */
function gramFlat(X: Float64Array, n: number, p: number): Float64Array {
  const G = new Float64Array(p * p);
  for (let i = 0; i < n; i++) {
    const base = i * p;
    for (let j = 0; j < p; j++) {
      const xij = X[base + j];
      for (let k = j; k < p; k++) {
        G[j * p + k] += xij * X[base + k];
      }
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = j + 1; k < p; k++) {
      G[k * p + j] = G[j * p + k];
    }
  }
  return G;
}

/** X^T y for row-major X (n×p) → length-p vector. */
function xTy(X: Float64Array, y: Float64Array | Int8Array, n: number, p: number): Float64Array {
  const out = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const yi = y[i];
    const base = i * p;
    for (let j = 0; j < p; j++) out[j] += X[base + j] * yi;
  }
  return out;
}

/**
 * Solve A x = b for SPD A (p×p row-major) via Cholesky.
 * Mutates A; returns x. Adds `jitter*I` if `jitter > 0`.
 */
function choleskySolve(A: Float64Array, b: Float64Array, p: number, jitter = 1e-10): Float64Array {
  const L = new Float64Array(p * p);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * p + j] + (i === j ? jitter : 0);
      for (let k = 0; k < j; k++) s -= L[i * p + k] * L[j * p + k];
      if (i === j) {
        if (s <= 0) {
          // Increase jitter and retry once.
          return choleskySolve(A, b, p, Math.max(jitter * 10, 1e-8));
        }
        L[i * p + j] = Math.sqrt(s);
      } else {
        L[i * p + j] = s / L[j * p + j];
      }
    }
  }
  // Solve L z = b.
  const z = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * p + k] * z[k];
    z[i] = s / L[i * p + i];
  }
  // Solve L^T x = z.
  const x = new Float64Array(p);
  for (let i = p - 1; i >= 0; i--) {
    let s = z[i];
    for (let k = i + 1; k < p; k++) s -= L[k * p + i] * x[k];
    x[i] = s / L[i * p + i];
  }
  return x;
}

// -----------------------------------------------------------------------------
// Standard normal CDF (Abramowitz & Stegun 7.1.26 / erf rational approximation).
// -----------------------------------------------------------------------------

/** Standard normal CDF Φ(x). */
export function normCdf(x: number): number {
  // erf approximation accurate to ~1.5e-7.
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429;
  const sign = x < 0 ? -1 : 1;
  const erf = sign * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
                     * Math.exp(-x * x / 2));
  return 0.5 * (1 + erf);
}

/** Quantile function of standard normal Φ⁻¹(p). Beasley-Springer-Moro approximation. */
export function normInv(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  // Acklam (2003) rational approximation.
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
              1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
              6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00];
  const pLow = 0.02425, pHigh = 1 - pLow;
  let q: number, r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
         / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
          / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
       / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Sigmoid σ(z) = 1 / (1 + e^{-z}). */
export function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  } else {
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }
}

// -----------------------------------------------------------------------------
// DGP 1: simple 1-D confounded DGP (§1 motivation figure).
// -----------------------------------------------------------------------------

/** §1 simple confounded DGP: X ~ N(0,1), e(X) = σ(1.5 X), Y = τ D + 1.5 X + ε. */
export function simpleConfoundedDGP(
  n: number,
  tau: number,
  rng: () => number,
): { X: Float64Array; D: Int8Array; Y: Float64Array } {
  const X = new Float64Array(n);
  const D = new Int8Array(n);
  const Y = new Float64Array(n);
  const gauss = gaussianRng(rng);
  for (let i = 0; i < n; i++) {
    const x = gauss();
    X[i] = x;
    const p = sigmoid(1.5 * x);
    const di = rng() < p ? 1 : 0;
    D[i] = di;
    Y[i] = tau * di + 1.5 * x + gauss();
  }
  return { X, D, Y };
}

// -----------------------------------------------------------------------------
// DGP 2: canonical Robinson DGP (§§2-8, 12, 13). Matches notebook code cell 2.
// eta(X) = 0.5 X_1 + Φ(X_3) - 0.3 X_4 - 0.5
// e(X)   = σ(1.5 eta(X))
// g(X)   = sin(X_1) + 0.5 X_2^2
// Y      = τ D + g(X) + ε
// -----------------------------------------------------------------------------

/** Canonical Robinson DGP. p must be >= 5 (uses X_1, X_2, X_3, X_4). */
export function robinsonDGP(
  n: number,
  p: number,
  tau: number,
  rng: () => number,
  options: { propensitySteepness?: number; unobservedConfounderStrength?: number } = {},
): RobinsonSample & { U?: Float64Array } {
  const pUse = Math.max(p, 5);
  const slope = options.propensitySteepness ?? 1.5;
  const uStrength = options.unobservedConfounderStrength ?? 0;
  const X = new Float64Array(n * pUse);
  const D = new Int8Array(n);
  const Y = new Float64Array(n);
  const ePropensity = new Float64Array(n);
  const gOracle = new Float64Array(n);
  const gauss = gaussianRng(rng);
  const U = uStrength > 0 ? new Float64Array(n) : undefined;
  for (let i = 0; i < n; i++) {
    let x0 = 0, x1 = 0, x2 = 0, x3 = 0;
    for (let j = 0; j < pUse; j++) {
      const v = gauss();
      X[i * pUse + j] = v;
      if (j === 0) x0 = v;
      else if (j === 1) x1 = v;
      else if (j === 2) x2 = v;
      else if (j === 3) x3 = v;
    }
    const u = U ? gauss() : 0;
    if (U) U[i] = u;
    const eta = 0.5 * x0 + normCdf(x2) - 0.3 * x3 - 0.5 + uStrength * u;
    const e = sigmoid(slope * eta);
    ePropensity[i] = e;
    const di = rng() < e ? 1 : 0;
    D[i] = di;
    const g = Math.sin(x0) + 0.5 * x1 * x1;
    gOracle[i] = g;
    Y[i] = tau * di + g + uStrength * u + gauss();
  }
  return { X, D, Y, n, p: pUse, ePropensity, gOracle, U };
}

/** Correct propensity features: [X_1, Φ(X_3), X_4]. */
export function correctPropensityFeatures(X: Float64Array, n: number, p: number): Float64Array {
  const F = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    F[i * 3 + 0] = X[i * p + 0];
    F[i * 3 + 1] = normCdf(X[i * p + 2]);
    F[i * 3 + 2] = X[i * p + 3];
  }
  return F;
}

/** Correct outcome features: [sin(X_1), X_2^2]. */
export function correctOutcomeFeatures(X: Float64Array, n: number, p: number): Float64Array {
  const F = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    const x0 = X[i * p + 0];
    const x1 = X[i * p + 1];
    F[i * 2 + 0] = Math.sin(x0);
    F[i * 2 + 1] = x1 * x1;
  }
  return F;
}

// -----------------------------------------------------------------------------
// Logistic regression (IRLS / Newton). Matches sklearn's LogisticRegression
// with penalty=None.
// -----------------------------------------------------------------------------

/** Fit logistic regression β to minimize -ℓ(β). Returns coefficients (intercept first). */
export function logisticFit(
  X: Float64Array,
  D: Int8Array,
  n: number,
  p: number,
  options: { maxIter?: number; tol?: number; ridge?: number } = {},
): Float64Array {
  const maxIter = options.maxIter ?? 50;
  const tol = options.tol ?? 1e-7;
  const ridge = options.ridge ?? 1e-8; // tiny ridge for numerical stability
  // Augment X with intercept column.
  const pa = p + 1;
  const Xa = new Float64Array(n * pa);
  for (let i = 0; i < n; i++) {
    Xa[i * pa + 0] = 1;
    for (let j = 0; j < p; j++) Xa[i * pa + j + 1] = X[i * p + j];
  }
  const beta = new Float64Array(pa);
  for (let iter = 0; iter < maxIter; iter++) {
    // Compute predictions and weights.
    const pi = new Float64Array(n);
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let z = 0;
      for (let j = 0; j < pa; j++) z += Xa[i * pa + j] * beta[j];
      const pij = sigmoid(z);
      pi[i] = pij;
      w[i] = Math.max(pij * (1 - pij), 1e-6);
    }
    // Gradient g = X^T (D - pi) - ridge * beta.
    const g = new Float64Array(pa);
    for (let i = 0; i < n; i++) {
      const r = D[i] - pi[i];
      for (let j = 0; j < pa; j++) g[j] += Xa[i * pa + j] * r;
    }
    for (let j = 0; j < pa; j++) g[j] -= ridge * beta[j];
    // Hessian-ish: X^T W X + ridge*I.
    const H = new Float64Array(pa * pa);
    for (let i = 0; i < n; i++) {
      const wi = w[i];
      const base = i * pa;
      for (let j = 0; j < pa; j++) {
        const xij = Xa[base + j];
        for (let k = j; k < pa; k++) {
          H[j * pa + k] += wi * xij * Xa[base + k];
        }
      }
    }
    for (let j = 0; j < pa; j++) {
      H[j * pa + j] += ridge;
      for (let k = j + 1; k < pa; k++) H[k * pa + j] = H[j * pa + k];
    }
    // Newton step: delta = H⁻¹ g.
    const delta = choleskySolve(H, g, pa);
    let stepNorm = 0;
    for (let j = 0; j < pa; j++) {
      beta[j] += delta[j];
      stepNorm += delta[j] * delta[j];
    }
    if (Math.sqrt(stepNorm) < tol) break;
  }
  return beta;
}

/** Predict P(D=1|X) given fitted logistic coefficients (intercept first). */
export function logisticPredictProba(
  beta: Float64Array,
  X: Float64Array,
  n: number,
  p: number,
): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = beta[0];
    for (let j = 0; j < p; j++) z += beta[j + 1] * X[i * p + j];
    out[i] = sigmoid(z);
  }
  return out;
}

/** Convenience: fit + predict in one call on the same X. */
export function logisticFitPredict(
  X: Float64Array,
  D: Int8Array,
  n: number,
  p: number,
  options?: { maxIter?: number; tol?: number; ridge?: number },
): { beta: Float64Array; ePred: Float64Array } {
  const beta = logisticFit(X, D, n, p, options);
  const ePred = logisticPredictProba(beta, X, n, p);
  return { beta, ePred };
}

// -----------------------------------------------------------------------------
// Linear regression (OLS via Cholesky on normal equations, with intercept).
// -----------------------------------------------------------------------------

/** Fit OLS β to minimize ||y - Xβ - β_0||². Returns [intercept, ...slopes]. */
export function linearFit(
  X: Float64Array,
  y: Float64Array,
  n: number,
  p: number,
  options: { ridge?: number } = {},
): Float64Array {
  const ridge = options.ridge ?? 1e-10;
  const pa = p + 1;
  const Xa = new Float64Array(n * pa);
  for (let i = 0; i < n; i++) {
    Xa[i * pa + 0] = 1;
    for (let j = 0; j < p; j++) Xa[i * pa + j + 1] = X[i * p + j];
  }
  const G = gramFlat(Xa, n, pa);
  for (let j = 0; j < pa; j++) G[j * pa + j] += ridge;
  const b = xTy(Xa, y, n, pa);
  return choleskySolve(G, b, pa);
}

/** Predict given fitted linear coefficients (intercept first). */
export function linearPredict(
  beta: Float64Array,
  X: Float64Array,
  n: number,
  p: number,
): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = beta[0];
    for (let j = 0; j < p; j++) z += beta[j + 1] * X[i * p + j];
    out[i] = z;
  }
  return out;
}

// -----------------------------------------------------------------------------
// Lasso (proximal-gradient ISTA). Coordinate descent would be faster, but
// ISTA is the existing pattern in high-dim-regression.ts; we inline a small
// implementation here to avoid an extra module-coupling for what is a small
// number of features. λ is selected by 3-fold CV over a fixed grid.
// -----------------------------------------------------------------------------

function softThreshold(z: number, lam: number): number {
  if (z > lam) return z - lam;
  if (z < -lam) return z + lam;
  return 0;
}

/** Center X column-wise and y; returns means used to restore intercept later. */
function centerXy(
  X: Float64Array,
  y: Float64Array,
  n: number,
  p: number,
): { Xc: Float64Array; yc: Float64Array; xMean: Float64Array; yMean: number; xScale: Float64Array } {
  const xMean = new Float64Array(p);
  const xScale = new Float64Array(p);
  let yMean = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) xMean[j] += X[i * p + j];
    yMean += y[i];
  }
  for (let j = 0; j < p; j++) xMean[j] /= n;
  yMean /= n;
  const Xc = new Float64Array(n * p);
  const yc = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    yc[i] = y[i] - yMean;
    for (let j = 0; j < p; j++) Xc[i * p + j] = X[i * p + j] - xMean[j];
  }
  // Column standard deviations (to normalize). Use ddof=1 like sklearn.
  for (let j = 0; j < p; j++) {
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const v = Xc[i * p + j];
      ss += v * v;
    }
    const s = Math.sqrt(ss / n);
    xScale[j] = s > 1e-12 ? s : 1;
    for (let i = 0; i < n; i++) Xc[i * p + j] /= xScale[j];
  }
  return { Xc, yc, xMean, yMean, xScale };
}

/**
 * Lasso via ISTA on (centered, standardized) X. Step size 1/L where L = largest
 * eigenvalue of X^T X / n (we use power iteration). Returns coefficients on the
 * ORIGINAL X scale, with explicit intercept.
 */
export function lassoFit(
  X: Float64Array,
  y: Float64Array,
  n: number,
  p: number,
  lam: number,
  options: { maxIter?: number; tol?: number } = {},
): Float64Array {
  const maxIter = options.maxIter ?? 800;
  const tol = options.tol ?? 1e-6;
  const { Xc, yc, xMean, yMean, xScale } = centerXy(X, y, n, p);
  // Power iteration for largest eigenvalue of X^T X / n.
  let v = new Float64Array(p).fill(1 / Math.sqrt(p));
  let L = 1;
  for (let it = 0; it < 30; it++) {
    const Xv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < p; j++) s += Xc[i * p + j] * v[j];
      Xv[i] = s;
    }
    const nv = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const xi = Xv[i] / n;
      for (let j = 0; j < p; j++) nv[j] += Xc[i * p + j] * xi;
    }
    let norm = 0;
    for (let j = 0; j < p; j++) norm += nv[j] * nv[j];
    norm = Math.sqrt(norm);
    if (norm < 1e-12) break;
    L = norm;
    for (let j = 0; j < p; j++) v[j] = nv[j] / norm;
  }
  const step = 1 / Math.max(L, 1e-6);
  // ISTA.
  const beta = new Float64Array(p);
  for (let it = 0; it < maxIter; it++) {
    // Gradient = X^T(Xβ - y) / n.
    const Xb = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < p; j++) s += Xc[i * p + j] * beta[j];
      Xb[i] = s - yc[i];
    }
    const g = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const ri = Xb[i] / n;
      for (let j = 0; j < p; j++) g[j] += Xc[i * p + j] * ri;
    }
    let maxDelta = 0;
    for (let j = 0; j < p; j++) {
      const candidate = beta[j] - step * g[j];
      const next = softThreshold(candidate, step * lam);
      maxDelta = Math.max(maxDelta, Math.abs(next - beta[j]));
      beta[j] = next;
    }
    if (maxDelta < tol) break;
  }
  // Un-scale and reconstruct intercept on the original X.
  const coef = new Float64Array(p + 1);
  let intercept = yMean;
  for (let j = 0; j < p; j++) {
    coef[j + 1] = beta[j] / xScale[j];
    intercept -= coef[j + 1] * xMean[j];
  }
  coef[0] = intercept;
  return coef;
}

/** Lasso CV: pick λ over a fixed log-spaced grid by 3-fold CV. */
export function lassoCVFit(
  X: Float64Array,
  y: Float64Array,
  n: number,
  p: number,
  rng: () => number,
  options: { K?: number; nLambda?: number } = {},
): { coef: Float64Array; lambda: number } {
  const K = options.K ?? 3;
  const nLambda = options.nLambda ?? 6;
  // λ-max heuristic: max_j |X^T y| / n.
  let lamMax = 0;
  const { Xc, yc, xScale } = centerXy(X, y, n, p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Xc[i * p + j] * yc[i];
    lamMax = Math.max(lamMax, Math.abs(s) / n);
  }
  const lamMin = lamMax * 1e-3;
  const grid: number[] = [];
  for (let k = 0; k < nLambda; k++) {
    const t = k / Math.max(nLambda - 1, 1);
    grid.push(lamMax * Math.exp(t * Math.log(lamMin / Math.max(lamMax, 1e-12))));
  }
  // Shuffle indices.
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
  }
  let bestLam = grid[0];
  let bestErr = Infinity;
  for (const lam of grid) {
    let cvErr = 0;
    for (let k = 0; k < K; k++) {
      const testIdx: number[] = [];
      const trainIdx: number[] = [];
      for (let i = 0; i < n; i++) {
        if (i % K === k) testIdx.push(idx[i]);
        else trainIdx.push(idx[i]);
      }
      const Xtr = selectRows(X, trainIdx, p);
      const ytr = selectValues(y, trainIdx);
      const beta = lassoFit(Xtr, ytr, trainIdx.length, p, lam, { maxIter: 200 });
      const Xte = selectRows(X, testIdx, p);
      const yte = selectValues(y, testIdx);
      const yhat = linearPredict(beta, Xte, testIdx.length, p);
      let mse = 0;
      for (let i = 0; i < yte.length; i++) {
        const r = yte[i] - yhat[i];
        mse += r * r;
      }
      cvErr += mse / yte.length;
    }
    if (cvErr < bestErr) {
      bestErr = cvErr;
      bestLam = lam;
    }
  }
  const coef = lassoFit(X, y, n, p, bestLam);
  return { coef, lambda: bestLam };
}

// -----------------------------------------------------------------------------
// Nadaraya–Watson kernel smoother (RF nuisance proxy in browser).
// h chosen by Silverman's rule on the training covariates.
// -----------------------------------------------------------------------------

/** Fit a Nadaraya–Watson smoother: returns a predict function. */
export function kernelSmootherFit(
  Xtr: Float64Array,
  ytr: Float64Array,
  nTr: number,
  p: number,
  bandwidthMult = 1.0,
): (Xte: Float64Array, nTe: number) => Float64Array {
  // Compute per-feature standard deviation for bandwidth.
  const sd = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0, s2 = 0;
    for (let i = 0; i < nTr; i++) {
      const v = Xtr[i * p + j];
      s += v; s2 += v * v;
    }
    const m = s / nTr;
    const variance = Math.max(s2 / nTr - m * m, 1e-6);
    sd[j] = Math.sqrt(variance);
  }
  // Silverman bandwidth per dim: 1.06 σ_j n^{-1/(d+4)} with √d compensation.
  const dim = p;
  const bw = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    bw[j] = bandwidthMult * 1.06 * sd[j] * Math.pow(nTr, -1 / (dim + 4));
    if (bw[j] < 1e-6) bw[j] = 1;
  }
  return (Xte: Float64Array, nTe: number) => {
    const out = new Float64Array(nTe);
    for (let i = 0; i < nTe; i++) {
      let num = 0, den = 0;
      for (let k = 0; k < nTr; k++) {
        let d2 = 0;
        for (let j = 0; j < p; j++) {
          const diff = (Xte[i * p + j] - Xtr[k * p + j]) / bw[j];
          d2 += diff * diff;
        }
        // Avoid Math.exp underflow on dense kernels.
        if (d2 < 80) {
          const w = Math.exp(-0.5 * d2);
          num += w * ytr[k];
          den += w;
        }
      }
      out[i] = den > 1e-12 ? num / den : 0;
    }
    return out;
  };
}

// -----------------------------------------------------------------------------
// K-fold splitter (with shuffle).
// -----------------------------------------------------------------------------

/** Return K folds: each fold is { trainIdx, testIdx }. */
export function kfoldSplit(
  n: number,
  K: number,
  rng: () => number,
): { trainIdx: number[]; testIdx: number[] }[] {
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
  }
  const folds: { trainIdx: number[]; testIdx: number[] }[] = [];
  const foldSize = Math.floor(n / K);
  for (let k = 0; k < K; k++) {
    const start = k * foldSize;
    const end = k === K - 1 ? n : (k + 1) * foldSize;
    const testIdx: number[] = [];
    const trainIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      const id = idx[i];
      if (i >= start && i < end) testIdx.push(id);
      else trainIdx.push(id);
    }
    folds.push({ trainIdx, testIdx });
  }
  return folds;
}

// -----------------------------------------------------------------------------
// §4 IPW (Horvitz–Thompson and Hájek) with plug-in sandwich SE.
// -----------------------------------------------------------------------------

/** IPW estimator; form ∈ {"ht", "hajek"}. Trim clips e to [eps, 1-eps]. */
export function ipwEstimate(
  D: Int8Array,
  Y: Float64Array,
  ePred: Float64Array,
  options: { form?: 'ht' | 'hajek'; trim?: number } = {},
): IPWResult {
  const form = options.form ?? 'hajek';
  const trim = options.trim ?? 0;
  const n = Y.length;
  const eClip = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = ePred[i];
    if (trim > 0) e = Math.min(Math.max(e, trim), 1 - trim);
    else e = Math.min(Math.max(e, 1e-6), 1 - 1e-6);
    eClip[i] = e;
  }
  const wT = new Float64Array(n);
  const wC = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    wT[i] = D[i] / eClip[i];
    wC[i] = (1 - D[i]) / (1 - eClip[i]);
  }
  let tau = 0;
  const psi = new Float64Array(n);
  if (form === 'ht') {
    for (let i = 0; i < n; i++) tau += wT[i] * Y[i] - wC[i] * Y[i];
    tau /= n;
    for (let i = 0; i < n; i++) psi[i] = wT[i] * Y[i] - wC[i] * Y[i] - tau;
  } else {
    let sumWT = 0, sumWC = 0, sumWTY = 0, sumWCY = 0;
    for (let i = 0; i < n; i++) {
      sumWT += wT[i]; sumWC += wC[i];
      sumWTY += wT[i] * Y[i]; sumWCY += wC[i] * Y[i];
    }
    const mu1 = sumWTY / sumWT;
    const mu0 = sumWCY / sumWC;
    tau = mu1 - mu0;
    for (let i = 0; i < n; i++) psi[i] = wT[i] * (Y[i] - mu1) - wC[i] * (Y[i] - mu0);
  }
  const se = Math.sqrt(variance(psi) / n);
  let maxW = 0;
  const weights = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    weights[i] = Math.max(wT[i], wC[i]);
    if (weights[i] > maxW) maxW = weights[i];
  }
  return { tau, se, ciLow: tau - 1.96 * se, ciHigh: tau + 1.96 * se, maxWeight: maxW, weights };
}

// -----------------------------------------------------------------------------
// §5 Outcome regression (g-formula plug-in) with naive delta-method SE.
// -----------------------------------------------------------------------------

/** OR estimator given fitted mu0_hat, mu1_hat at every sample point. */
export function orEstimateFromPreds(
  mu0Hat: Float64Array,
  mu1Hat: Float64Array,
): EstimatorResult {
  const n = mu0Hat.length;
  const contrasts = new Float64Array(n);
  for (let i = 0; i < n; i++) contrasts[i] = mu1Hat[i] - mu0Hat[i];
  const tau = mean(contrasts);
  const se = stddev(contrasts) / Math.sqrt(n);
  return { tau, se, ciLow: tau - 1.96 * se, ciHigh: tau + 1.96 * se };
}

// -----------------------------------------------------------------------------
// §6 AIPW estimator with EIF plug-in SE.
// -----------------------------------------------------------------------------

/** Per-unit AIPW score (the EIF). */
export function aipwScore(
  D: Int8Array,
  Y: Float64Array,
  ePred: Float64Array,
  mu0: Float64Array,
  mu1: Float64Array,
): Float64Array {
  const n = Y.length;
  const psi = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const e = Math.min(Math.max(ePred[i], 1e-6), 1 - 1e-6);
    psi[i] = mu1[i] - mu0[i]
           + (D[i] * (Y[i] - mu1[i])) / e
           - ((1 - D[i]) * (Y[i] - mu0[i])) / (1 - e);
  }
  return psi;
}

/** AIPW estimator with EIF plug-in SE. */
export function aipwEstimate(
  D: Int8Array,
  Y: Float64Array,
  ePred: Float64Array,
  mu0: Float64Array,
  mu1: Float64Array,
): EstimatorResult {
  const psi = aipwScore(D, Y, ePred, mu0, mu1);
  const tau = mean(psi);
  const se = stddev(psi) / Math.sqrt(psi.length);
  return { tau, se, ciLow: tau - 1.96 * se, ciHigh: tau + 1.96 * se };
}

// -----------------------------------------------------------------------------
// §7 TMLE: linear submodel and logistic submodel.
// -----------------------------------------------------------------------------

/** TMLE with linear submodel (continuous Y). Matches notebook tmle_estimate. */
export function tmleLinear(
  D: Int8Array,
  Y: Float64Array,
  ePred: Float64Array,
  mu0Init: Float64Array,
  mu1Init: Float64Array,
): TMLEResult & { mu0Star: Float64Array; mu1Star: Float64Array } {
  const n = Y.length;
  const H1 = new Float64Array(n);
  const H0 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const e = Math.min(Math.max(ePred[i], 1e-6), 1 - 1e-6);
    H1[i] = 1 / e;
    H0[i] = 1 / (1 - e);
  }
  let num1 = 0, den1 = 0, num0 = 0, den0 = 0;
  for (let i = 0; i < n; i++) {
    if (D[i] === 1) {
      num1 += H1[i] * (Y[i] - mu1Init[i]);
      den1 += H1[i] * H1[i];
    } else {
      num0 += H0[i] * (Y[i] - mu0Init[i]);
      den0 += H0[i] * H0[i];
    }
  }
  const eps1 = den1 > 1e-12 ? num1 / den1 : 0;
  const eps0 = den0 > 1e-12 ? num0 / den0 : 0;
  const mu1Star = new Float64Array(n);
  const mu0Star = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    mu1Star[i] = mu1Init[i] + eps1 * H1[i];
    mu0Star[i] = mu0Init[i] + eps0 * H0[i];
  }
  let tau = 0;
  for (let i = 0; i < n; i++) tau += mu1Star[i] - mu0Star[i];
  tau /= n;
  const psi = aipwScore(D, Y, ePred, mu0Star, mu1Star);
  const se = stddev(psi) / Math.sqrt(n);
  return {
    tau, se, ciLow: tau - 1.96 * se, ciHigh: tau + 1.96 * se,
    epsilon1: eps1, epsilon0: eps0, mu0Star, mu1Star,
  };
}

/**
 * Logistic-submodel TMLE for bounded Y ∈ [0, 1] (typically dichotomized). Newton
 * scalar update for ε_d on the logistic-link submodel. Preserves the [0, 1]
 * constraint by construction.
 */
export function tmleLogistic(
  D: Int8Array,
  Y: Float64Array,
  ePred: Float64Array,
  mu0Init: Float64Array,
  mu1Init: Float64Array,
  options: { maxIter?: number; tol?: number } = {},
): TMLEResult & { mu0Star: Float64Array; mu1Star: Float64Array } {
  const maxIter = options.maxIter ?? 50;
  const tol = options.tol ?? 1e-9;
  const n = Y.length;
  // Clip mu's to (eps, 1-eps).
  const clip = (v: number) => Math.min(Math.max(v, 1e-5), 1 - 1e-5);
  const mu1c = new Float64Array(n);
  const mu0c = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    mu1c[i] = clip(mu1Init[i]);
    mu0c[i] = clip(mu0Init[i]);
  }
  const H1 = new Float64Array(n);
  const H0 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const e = Math.min(Math.max(ePred[i], 1e-6), 1 - 1e-6);
    H1[i] = 1 / e;
    H0[i] = 1 / (1 - e);
  }
  // Newton on ε_d for each arm separately. Solve sum H_d (Y - σ(logit(mu_d) + ε H_d)) = 0 over arm d.
  function solveEpsilon(arm: 0 | 1, H: Float64Array, mu: Float64Array): number {
    let eps = 0;
    for (let it = 0; it < maxIter; it++) {
      let g = 0, h = 0;
      for (let i = 0; i < n; i++) {
        if (D[i] !== arm) continue;
        const lin = Math.log(mu[i] / (1 - mu[i])) + eps * H[i];
        const sig = sigmoid(lin);
        g += H[i] * (Y[i] - sig);
        h -= H[i] * H[i] * sig * (1 - sig);
      }
      if (Math.abs(h) < 1e-12) break;
      const step = g / h;
      eps -= step;
      if (Math.abs(step) < tol) break;
    }
    return eps;
  }
  const eps1 = solveEpsilon(1, H1, mu1c);
  const eps0 = solveEpsilon(0, H0, mu0c);
  const mu1Star = new Float64Array(n);
  const mu0Star = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    mu1Star[i] = sigmoid(Math.log(mu1c[i] / (1 - mu1c[i])) + eps1 * H1[i]);
    mu0Star[i] = sigmoid(Math.log(mu0c[i] / (1 - mu0c[i])) + eps0 * H0[i]);
  }
  let tau = 0;
  for (let i = 0; i < n; i++) tau += mu1Star[i] - mu0Star[i];
  tau /= n;
  const psi = aipwScore(D, Y, ePred, mu0Star, mu1Star);
  const se = stddev(psi) / Math.sqrt(n);
  let inUnit = true;
  for (let i = 0; i < n; i++) {
    if (mu1Star[i] < 0 || mu1Star[i] > 1 || mu0Star[i] < 0 || mu0Star[i] > 1) inUnit = false;
  }
  return {
    tau, se, ciLow: tau - 1.96 * se, ciHigh: tau + 1.96 * se,
    epsilon1: eps1, epsilon0: eps0, mu0Star, mu1Star, inUnitInterval: inUnit,
  };
}

// -----------------------------------------------------------------------------
// §8 DML cross-fitting harness. Accepts a fold-level nuisance fitter callback.
// -----------------------------------------------------------------------------

/**
 * Per-fold nuisance fitter signature.
 *
 * Given the full (X, D, Y) and the (trainIdx, testIdx) for this fold, return
 * predictions (eHat, mu0Hat, mu1Hat) on the TEST points. Implementations are
 * free to call lasso, kernel smoother, or oracle linear regression on the
 * train indices.
 */
export type NuisanceFitter = (
  X: Float64Array,
  D: Int8Array,
  Y: Float64Array,
  n: number,
  p: number,
  trainIdx: number[],
  testIdx: number[],
  rng: () => number,
) => { eHat: Float64Array; mu0Hat: Float64Array; mu1Hat: Float64Array };

/** Cross-fitted DML estimator with the AIPW score. */
export function dmlCrossFit(
  X: Float64Array,
  D: Int8Array,
  Y: Float64Array,
  n: number,
  p: number,
  fitter: NuisanceFitter,
  K: number,
  rng: () => number,
  options: { trim?: number } = {},
): DMLResult {
  const trim = options.trim ?? 0.025;
  const folds = kfoldSplit(n, K, rng);
  const psi = new Float64Array(n);
  const ePredAll = new Float64Array(n);
  const mu0PredAll = new Float64Array(n);
  const mu1PredAll = new Float64Array(n);
  for (const { trainIdx, testIdx } of folds) {
    const { eHat, mu0Hat, mu1Hat } = fitter(X, D, Y, n, p, trainIdx, testIdx, rng);
    for (let k = 0; k < testIdx.length; k++) {
      const idx = testIdx[k];
      let e = eHat[k];
      e = Math.min(Math.max(e, trim), 1 - trim);
      ePredAll[idx] = e;
      mu0PredAll[idx] = mu0Hat[k];
      mu1PredAll[idx] = mu1Hat[k];
      psi[idx] = mu1Hat[k] - mu0Hat[k]
               + (D[idx] * (Y[idx] - mu1Hat[k])) / e
               - ((1 - D[idx]) * (Y[idx] - mu0Hat[k])) / (1 - e);
    }
  }
  const tau = mean(psi);
  const se = stddev(psi) / Math.sqrt(n);
  return {
    tau, se, ciLow: tau - 1.96 * se, ciHigh: tau + 1.96 * se,
    ePred: ePredAll, mu0Pred: mu0PredAll, mu1Pred: mu1PredAll, psi,
  };
}

/** "Oracle" nuisance fitter using correct feature transforms. */
export const oracleNuisanceFitter: NuisanceFitter = (X, D, Y, _n, p, trainIdx, testIdx) => {
  const Xtr = selectRows(X, trainIdx, p);
  const Dtr = selectValues(D, trainIdx);
  const Ytr = selectValues(Y, trainIdx);
  const Xte = selectRows(X, testIdx, p);
  // Propensity on correct features.
  const propTr = correctPropensityFeatures(Xtr, trainIdx.length, p);
  const propTe = correctPropensityFeatures(Xte, testIdx.length, p);
  const betaP = logisticFit(propTr, Dtr, trainIdx.length, 3);
  const eHat = logisticPredictProba(betaP, propTe, testIdx.length, 3);
  // Outcome on correct features (separate fits per arm).
  const outTr = correctOutcomeFeatures(Xtr, trainIdx.length, p);
  const outTe = correctOutcomeFeatures(Xte, testIdx.length, p);
  const treatedIdx: number[] = [];
  const controlIdx: number[] = [];
  for (let i = 0; i < Dtr.length; i++) {
    if (Dtr[i] === 1) treatedIdx.push(i);
    else controlIdx.push(i);
  }
  const outTreated = selectRows(outTr, treatedIdx, 2);
  const yTreated = selectValues(Ytr, treatedIdx);
  const outControl = selectRows(outTr, controlIdx, 2);
  const yControl = selectValues(Ytr, controlIdx);
  const beta1 = linearFit(outTreated, yTreated, treatedIdx.length, 2);
  const beta0 = linearFit(outControl, yControl, controlIdx.length, 2);
  const mu1Hat = linearPredict(beta1, outTe, testIdx.length, 2);
  const mu0Hat = linearPredict(beta0, outTe, testIdx.length, 2);
  return { eHat, mu0Hat, mu1Hat };
};

/** Lasso-CV nuisance fitter (propensity is logistic with raw features; outcome lasso-CV). */
export const lassoNuisanceFitter: NuisanceFitter = (X, D, Y, _n, p, trainIdx, testIdx, rng) => {
  const Xtr = selectRows(X, trainIdx, p);
  const Dtr = selectValues(D, trainIdx);
  const Ytr = selectValues(Y, trainIdx);
  const Xte = selectRows(X, testIdx, p);
  // Logistic on raw X for propensity.
  const betaP = logisticFit(Xtr, Dtr, trainIdx.length, p, { ridge: 0.05 });
  const eHat = logisticPredictProba(betaP, Xte, testIdx.length, p);
  // Lasso CV per arm.
  const treatedIdx: number[] = [];
  const controlIdx: number[] = [];
  for (let i = 0; i < Dtr.length; i++) {
    if (Dtr[i] === 1) treatedIdx.push(i);
    else controlIdx.push(i);
  }
  const X1 = selectRows(Xtr, treatedIdx, p);
  const y1 = selectValues(Ytr, treatedIdx);
  const X0 = selectRows(Xtr, controlIdx, p);
  const y0 = selectValues(Ytr, controlIdx);
  const { coef: c1 } = lassoCVFit(X1, y1, treatedIdx.length, p, rng);
  const { coef: c0 } = lassoCVFit(X0, y0, controlIdx.length, p, rng);
  const mu1Hat = linearPredict(c1, Xte, testIdx.length, p);
  const mu0Hat = linearPredict(c0, Xte, testIdx.length, p);
  return { eHat, mu0Hat, mu1Hat };
};

/** Kernel-smoother nuisance fitter (RF proxy for browser). */
export const kernelNuisanceFitter: NuisanceFitter = (X, D, Y, _n, p, trainIdx, testIdx) => {
  const Xtr = selectRows(X, trainIdx, p);
  const Dtr = selectValues(D, trainIdx);
  const Ytr = selectValues(Y, trainIdx);
  const Xte = selectRows(X, testIdx, p);
  // Propensity: regress D as a continuous outcome via kernel smoother (clipped).
  const DtrF = new Float64Array(Dtr.length);
  for (let i = 0; i < Dtr.length; i++) DtrF[i] = Dtr[i];
  const predProp = kernelSmootherFit(Xtr, DtrF, trainIdx.length, p, 0.7);
  const eRaw = predProp(Xte, testIdx.length);
  const eHat = new Float64Array(testIdx.length);
  for (let i = 0; i < testIdx.length; i++) eHat[i] = Math.min(Math.max(eRaw[i], 0.025), 0.975);
  // Outcome: kernel-smooth per arm.
  const treatedIdx: number[] = [];
  const controlIdx: number[] = [];
  for (let i = 0; i < Dtr.length; i++) {
    if (Dtr[i] === 1) treatedIdx.push(i);
    else controlIdx.push(i);
  }
  const X1 = selectRows(Xtr, treatedIdx, p);
  const y1 = selectValues(Ytr, treatedIdx);
  const X0 = selectRows(Xtr, controlIdx, p);
  const y0 = selectValues(Ytr, controlIdx);
  const pred1 = kernelSmootherFit(X1, y1, treatedIdx.length, p, 0.7);
  const pred0 = kernelSmootherFit(X0, y0, controlIdx.length, p, 0.7);
  const mu1Hat = pred1(Xte, testIdx.length);
  const mu0Hat = pred0(Xte, testIdx.length);
  return { eHat, mu0Hat, mu1Hat };
};

// -----------------------------------------------------------------------------
// §9 IV: Wald estimator + first-stage F.
// -----------------------------------------------------------------------------

/** Simple IV DGP: Z binary, U latent confounder, D = 1{strength*Z + conf*U + ε > 0}, Y = τD + 0.5U + ε. */
export function ivDGP(
  n: number,
  instrumentStrength: number,
  tau: number,
  rng: () => number,
  options: { confounding?: number } = {},
): { Z: Int8Array; D: Int8Array; Y: Float64Array; U: Float64Array } {
  const confounding = options.confounding ?? 0.7;
  const Z = new Int8Array(n);
  const D = new Int8Array(n);
  const Y = new Float64Array(n);
  const U = new Float64Array(n);
  const gauss = gaussianRng(rng);
  for (let i = 0; i < n; i++) {
    Z[i] = rng() < 0.5 ? 1 : 0;
    U[i] = gauss();
    const eD = gauss();
    D[i] = (instrumentStrength * Z[i] + confounding * U[i] + eD > 0) ? 1 : 0;
    Y[i] = tau * D[i] + 0.5 * U[i] + gauss();
  }
  return { Z, D, Y, U };
}

/** Wald IV estimator + first-stage F-statistic. */
export function waldIV(Z: Int8Array, D: Int8Array, Y: Float64Array): WaldResult {
  const n = Y.length;
  let nZ1 = 0, nZ0 = 0;
  let sumYZ1 = 0, sumYZ0 = 0, sumDZ1 = 0, sumDZ0 = 0;
  for (let i = 0; i < n; i++) {
    if (Z[i] === 1) {
      nZ1++; sumYZ1 += Y[i]; sumDZ1 += D[i];
    } else {
      nZ0++; sumYZ0 += Y[i]; sumDZ0 += D[i];
    }
  }
  const meanYZ1 = nZ1 > 0 ? sumYZ1 / nZ1 : 0;
  const meanYZ0 = nZ0 > 0 ? sumYZ0 / nZ0 : 0;
  const meanDZ1 = nZ1 > 0 ? sumDZ1 / nZ1 : 0;
  const meanDZ0 = nZ0 > 0 ? sumDZ0 / nZ0 : 0;
  const num = meanYZ1 - meanYZ0;
  const den = meanDZ1 - meanDZ0;
  const tau = num / den;
  // First-stage F: D ~ Z regression with single coef.
  let meanZ = 0, meanD = 0;
  for (let i = 0; i < n; i++) { meanZ += Z[i]; meanD += D[i]; }
  meanZ /= n; meanD /= n;
  let szz = 0, szd = 0, sdd = 0;
  for (let i = 0; i < n; i++) {
    const zc = Z[i] - meanZ;
    const dc = D[i] - meanD;
    szz += zc * zc;
    szd += zc * dc;
    sdd += dc * dc;
  }
  const beta = szz > 1e-12 ? szd / szz : 0;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const fitted = meanD + beta * (Z[i] - meanZ);
    const r = D[i] - fitted;
    rss += r * r;
  }
  const tss = sdd;
  const F = rss > 1e-12 && n > 2 ? ((tss - rss) / 1) / (rss / (n - 2)) : Infinity;
  // SE via efficient-IV / delta method: SE(τ) = SE(num) / |den|.
  // Use sandwich on reduced form only (simplified).
  let varYZ1 = 0, varYZ0 = 0;
  for (let i = 0; i < n; i++) {
    if (Z[i] === 1) varYZ1 += (Y[i] - meanYZ1) ** 2;
    else varYZ0 += (Y[i] - meanYZ0) ** 2;
  }
  const seNum = Math.sqrt((nZ1 > 1 ? varYZ1 / (nZ1 - 1) / nZ1 : 0) + (nZ0 > 1 ? varYZ0 / (nZ0 - 1) / nZ0 : 0));
  const se = Math.abs(den) > 1e-12 ? seNum / Math.abs(den) : Infinity;
  return {
    tau, se, ciLow: tau - 1.96 * se, ciHigh: tau + 1.96 * se,
    reducedForm: num, firstStage: den, firstStageF: F,
  };
}

// -----------------------------------------------------------------------------
// §10 Front-door identification.
// -----------------------------------------------------------------------------

/** §10 3-node DGP: U unobserved; D|U; M|D; Y = β_M M + δ_U U + ε. */
export function frontdoorDGP(
  n: number,
  rng: () => number,
  options: { gamma?: number; betaM?: number; deltaU?: number; sigmaM?: number } = {},
): { D: Int8Array; M: Float64Array; Y: Float64Array; U: Float64Array; trueTau: number } {
  const gamma = options.gamma ?? 0.8;
  const betaM = options.betaM ?? 1.0;
  const deltaU = options.deltaU ?? 1.5;
  const sigmaM = options.sigmaM ?? 1.0;
  const D = new Int8Array(n);
  const M = new Float64Array(n);
  const Y = new Float64Array(n);
  const U = new Float64Array(n);
  const gauss = gaussianRng(rng);
  for (let i = 0; i < n; i++) {
    U[i] = gauss();
    const pD = sigmoid(U[i]);
    D[i] = rng() < pD ? 1 : 0;
    M[i] = gamma * D[i] + sigmaM * gauss();
    Y[i] = betaM * M[i] + deltaU * U[i] + gauss();
  }
  return { D, M, Y, U, trueTau: gamma * betaM };
}

/** Front-door estimator (linear regression of Y on M, D, M*D). */
export function frontdoorEstimate(D: Int8Array, M: Float64Array, Y: Float64Array): FrontdoorResult {
  const n = Y.length;
  let pD1 = 0;
  for (let i = 0; i < n; i++) pD1 += D[i];
  pD1 /= n;
  const pD0 = 1 - pD1;
  // Build X = [M, D, M*D], fit Y ~ X.
  const Xd = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    Xd[i * 3 + 0] = M[i];
    Xd[i * 3 + 1] = D[i];
    Xd[i * 3 + 2] = M[i] * D[i];
  }
  const beta = linearFit(Xd, Y, n, 3);
  // Predict EY|M, D=1 and EY|M, D=0 for every i.
  const X1 = new Float64Array(n * 3);
  const X0 = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    X1[i * 3 + 0] = M[i]; X1[i * 3 + 1] = 1; X1[i * 3 + 2] = M[i];
    X0[i * 3 + 0] = M[i]; X0[i * 3 + 1] = 0; X0[i * 3 + 2] = 0;
  }
  const ey1 = linearPredict(beta, X1, n, 3);
  const ey0 = linearPredict(beta, X0, n, 3);
  // g(M_i) = pD1 * EY|M,1 + pD0 * EY|M,0.
  let sum1 = 0, sum0 = 0, n1 = 0, n0 = 0;
  for (let i = 0; i < n; i++) {
    const gm = pD1 * ey1[i] + pD0 * ey0[i];
    if (D[i] === 1) { sum1 += gm; n1++; }
    else { sum0 += gm; n0++; }
  }
  const tau = (n1 > 0 ? sum1 / n1 : 0) - (n0 > 0 ? sum0 / n0 : 0);
  // Simple bootstrap-ish SE: use residual variance of (g(M_i) - tau) per arm.
  const residArm1: number[] = [];
  const residArm0: number[] = [];
  for (let i = 0; i < n; i++) {
    const gm = pD1 * ey1[i] + pD0 * ey0[i];
    if (D[i] === 1) residArm1.push(gm);
    else residArm0.push(gm);
  }
  const v1 = residArm1.length > 1 ? variance(residArm1) / residArm1.length : 0;
  const v0 = residArm0.length > 1 ? variance(residArm0) / residArm0.length : 0;
  const se = Math.sqrt(v1 + v0);
  return { tau, se, ciLow: tau - 1.96 * se, ciHigh: tau + 1.96 * se };
}

/** Oracle (knows U) estimator: regress Y on [D, U] and read the D coefficient. */
export function frontdoorOracle(D: Int8Array, Y: Float64Array, U: Float64Array): number {
  const n = Y.length;
  const Xd = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    Xd[i * 2 + 0] = D[i];
    Xd[i * 2 + 1] = U[i];
  }
  const beta = linearFit(Xd, Y, n, 2);
  return beta[1]; // coefficient on D
}

// -----------------------------------------------------------------------------
// §11 Heterogeneous treatment effects: T-learner and DR-learner.
// -----------------------------------------------------------------------------

/** Heterogeneous DGP: τ(x) = 1 + 0.5 x_1 - 0.3 x_2 + 0.4 x_1 x_2. */
export function heterogeneousDGP(
  n: number,
  p: number,
  rng: () => number,
): { X: Float64Array; D: Int8Array; Y: Float64Array; trueTauX: Float64Array; n: number; p: number } {
  const pUse = Math.max(p, 2);
  const X = new Float64Array(n * pUse);
  const D = new Int8Array(n);
  const Y = new Float64Array(n);
  const trueTauX = new Float64Array(n);
  const gauss = gaussianRng(rng);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < pUse; j++) X[i * pUse + j] = gauss();
    const x0 = X[i * pUse + 0];
    const x1 = X[i * pUse + 1];
    const e = sigmoid(0.3 * x0 + 0.2 * x1);
    D[i] = rng() < e ? 1 : 0;
    const tauX = 1 + 0.5 * x0 - 0.3 * x1 + 0.4 * x0 * x1;
    trueTauX[i] = tauX;
    const g = 0.5 * x0 * x0 + Math.sin(x1);
    Y[i] = tauX * D[i] + g + gauss();
  }
  return { X, D, Y, trueTauX, n, p: pUse };
}

/** T-learner (kernel smoother as RF proxy): returns predict(Xtest) → tau(x). */
export function tLearnerKernel(
  X: Float64Array,
  D: Int8Array,
  Y: Float64Array,
  n: number,
  p: number,
): (Xte: Float64Array, nTe: number) => Float64Array {
  const treatedIdx: number[] = [];
  const controlIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    if (D[i] === 1) treatedIdx.push(i);
    else controlIdx.push(i);
  }
  const X1 = selectRows(X, treatedIdx, p);
  const y1 = selectValues(Y, treatedIdx);
  const X0 = selectRows(X, controlIdx, p);
  const y0 = selectValues(Y, controlIdx);
  const pred1 = kernelSmootherFit(X1, y1, treatedIdx.length, p, 0.6);
  const pred0 = kernelSmootherFit(X0, y0, controlIdx.length, p, 0.6);
  return (Xte, nTe) => {
    const m1 = pred1(Xte, nTe);
    const m0 = pred0(Xte, nTe);
    const out = new Float64Array(nTe);
    for (let i = 0; i < nTe; i++) out[i] = m1[i] - m0[i];
    return out;
  };
}

/**
 * DR-learner with K-fold cross-fitted nuisances. Constructs AIPW pseudo-outcomes
 * Ψ_i, then regresses them on X via kernel smoother. Returns predict.
 */
export function drLearnerKernel(
  X: Float64Array,
  D: Int8Array,
  Y: Float64Array,
  n: number,
  p: number,
  rng: () => number,
  K = 3,
): (Xte: Float64Array, nTe: number) => Float64Array {
  const folds = kfoldSplit(n, K, rng);
  const pseudoY = new Float64Array(n);
  for (const { trainIdx, testIdx } of folds) {
    const { eHat, mu0Hat, mu1Hat } = kernelNuisanceFitter(X, D, Y, n, p, trainIdx, testIdx, rng);
    for (let k = 0; k < testIdx.length; k++) {
      const idx = testIdx[k];
      const e = Math.min(Math.max(eHat[k], 0.025), 0.975);
      pseudoY[idx] = mu1Hat[k] - mu0Hat[k]
                   + (D[idx] * (Y[idx] - mu1Hat[k])) / e
                   - ((1 - D[idx]) * (Y[idx] - mu0Hat[k])) / (1 - e);
    }
  }
  const finalPred = kernelSmootherFit(X, pseudoY, n, p, 0.5);
  return (Xte, nTe) => finalPred(Xte, nTe);
}

// -----------------------------------------------------------------------------
// §12 Sensitivity analysis: E-value + Cinelli–Hazlett RV.
// -----------------------------------------------------------------------------

/** E-value from a relative risk. */
export function eValueFromRR(rr: number): number {
  let r = rr;
  if (r < 1) r = 1 / r;
  return r + Math.sqrt(r * (r - 1));
}

/**
 * E-value for a continuous-outcome effect via the d → RR conversion
 * (VanderWeele–Ding 2017: RR ≈ exp(0.91 d)).
 */
export function eValueContinuous(
  tauHat: number,
  se: number,
  sdY: number,
): { dStandardized: number; rr: number; rrCiLow: number; ePoint: number; eCi: number } {
  const d = tauHat / sdY;
  const dLow = (tauHat - 1.96 * se) / sdY;
  const rr = Math.exp(0.91 * d);
  const rrCiLow = Math.exp(0.91 * dLow);
  return {
    dStandardized: d, rr, rrCiLow,
    ePoint: eValueFromRR(rr), eCi: eValueFromRR(rrCiLow),
  };
}

/**
 * Cinelli–Hazlett (2020) Robustness Value.
 *
 *   RV_q = 0.5 * (sqrt(f_q^4 + 4 f_q^2) - f_q^2),  f_q = q * |t| / sqrt(df)
 *   RV_{q, α} = same with t replaced by max(t - t_crit(α), 0)
 *
 * For α = 0.05 we use the standard-normal critical 1.96 (df is typically large
 * enough for the Student-t to match within a few percent).
 */
export function cinelliHazlettRV(
  tauHat: number,
  se: number,
  df: number,
  options: { q?: number; alpha?: number } = {},
): { rvQ: number; rvQAlpha: number } {
  const q = options.q ?? 1.0;
  const alpha = options.alpha ?? 0.05;
  const tStat = Math.abs(tauHat) / se;
  const fQ = (q * tStat) / Math.sqrt(df);
  const rvQ = 0.5 * (Math.sqrt(fQ ** 4 + 4 * fQ * fQ) - fQ * fQ);
  // Use Φ⁻¹(1 - α/2) as the α critical (close enough for df > 30; df = n-5 here).
  const tCrit = normInv(1 - alpha / 2);
  const tAdj = Math.max(tStat - tCrit, 0);
  const fQa = (q * tAdj) / Math.sqrt(df);
  const rvQAlpha = fQa > 0 ? 0.5 * (Math.sqrt(fQa ** 4 + 4 * fQa * fQa) - fQa * fQa) : 0;
  return { rvQ, rvQAlpha };
}

/** Combined sensitivity bundle for a given AIPW estimate. */
export function sensitivityDiagnostics(
  tauHat: number,
  se: number,
  sdY: number,
  df: number,
  options: { q?: number; alpha?: number } = {},
): SensitivityResult {
  const ev = eValueContinuous(tauHat, se, sdY);
  const rv = cinelliHazlettRV(tauHat, se, df, options);
  return {
    ePoint: ev.ePoint, eCi: ev.eCi,
    rvQ: rv.rvQ, rvQAlpha: rv.rvQAlpha,
    dStandardized: ev.dStandardized,
  };
}

// -----------------------------------------------------------------------------
// §12 Rosenbaum p-value bound (binary matched-pairs).
// For pairs with treatment indicator t_ik ∈ {0, 1} (one per pair member),
// the Rosenbaum bound on the one-sided p-value under unobserved confounding
// parameter Γ comes from a sum of Γ-tilted Bernoulli trials.
// We expose the closed form for the simplest one-sided sign test on paired Y.
// -----------------------------------------------------------------------------

/** Worst-case one-sided p-value bound for a paired-data sign test under Γ. */
export function rosenbaumSignTestUpperBound(
  pairedDiffs: Float64Array | number[],
  gamma: number,
): number {
  let nPlus = 0, nNonZero = 0;
  for (let i = 0; i < pairedDiffs.length; i++) {
    if (pairedDiffs[i] > 0) nPlus++;
    if (pairedDiffs[i] !== 0) nNonZero++;
  }
  if (nNonZero === 0) return 1;
  const pUpper = gamma / (1 + gamma);
  // Pr(Binomial(n, p) >= nPlus). Normal approximation for large nNonZero;
  // exact tail for small.
  if (nNonZero < 200) {
    let cdf = 0;
    let logC = 0;
    // Compute log( C(n,k) p^k (1-p)^{n-k} ) iteratively.
    const p = pUpper;
    const lp = Math.log(p);
    const lq = Math.log(1 - p);
    // P(X >= nPlus) = sum_{k=nPlus}^n C(n,k) p^k q^{n-k}.
    for (let k = nPlus; k <= nNonZero; k++) {
      let logPk = lp * k + lq * (nNonZero - k);
      // log C(n,k).
      let logBin = 0;
      for (let i = 1; i <= k; i++) logBin += Math.log(nNonZero - i + 1) - Math.log(i);
      logPk += logBin;
      cdf += Math.exp(logPk);
    }
    return Math.min(cdf, 1);
  }
  // Normal approximation.
  const mu = nNonZero * pUpper;
  const sigma = Math.sqrt(nNonZero * pUpper * (1 - pUpper));
  const z = (nPlus - 0.5 - mu) / sigma;
  return 1 - normCdf(z);
}

// -----------------------------------------------------------------------------
// Convenience: full-stack causal-inference single-sample worked example for
// the §13 viz. Returns the five estimators (IPW, OR, AIPW, DML-lasso, DML-RF
// proxy) with their results.
// -----------------------------------------------------------------------------

/** Single-sample full pipeline on Robinson DGP. Matches notebook §13 code cell A. */
export function workedExamplePipeline(
  sample: RobinsonSample,
  rng: () => number,
): {
  ipw: IPWResult;
  or: EstimatorResult;
  aipw: EstimatorResult;
  dmlLasso: DMLResult;
  dmlRF: DMLResult;
} {
  const { X, D, Y, n, p } = sample;
  // Propensity on correct features.
  const propFeat = correctPropensityFeatures(X, n, p);
  const { ePred } = logisticFitPredict(propFeat, D, n, 3);
  const ipw = ipwEstimate(D, Y, ePred, { form: 'hajek' });
  // Outcome on correct features.
  const outFeat = correctOutcomeFeatures(X, n, p);
  const treatedIdx: number[] = [];
  const controlIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    if (D[i] === 1) treatedIdx.push(i);
    else controlIdx.push(i);
  }
  const X1 = selectRows(outFeat, treatedIdx, 2);
  const Y1 = selectValues(Y, treatedIdx);
  const X0 = selectRows(outFeat, controlIdx, 2);
  const Y0 = selectValues(Y, controlIdx);
  const beta1 = linearFit(X1, Y1, treatedIdx.length, 2);
  const beta0 = linearFit(X0, Y0, controlIdx.length, 2);
  const mu1 = linearPredict(beta1, outFeat, n, 2);
  const mu0 = linearPredict(beta0, outFeat, n, 2);
  const or = orEstimateFromPreds(mu0, mu1);
  const aipw = aipwEstimate(D, Y, ePred, mu0, mu1);
  const dmlLasso = dmlCrossFit(X, D, Y, n, p, lassoNuisanceFitter, 5, rng);
  const dmlRF = dmlCrossFit(X, D, Y, n, p, kernelNuisanceFitter, 5, rng);
  return { ipw, or, aipw, dmlLasso, dmlRF };
}

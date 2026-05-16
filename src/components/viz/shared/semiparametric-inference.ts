// =============================================================================
// semiparametric-inference.ts
//
// Shared math primitives for the semiparametric-inference topic's viz
// components. Ported from notebooks/semiparametric-inference/
// 01_semiparametric_inference.ipynb §§1–13.
//
// In-browser TS for all interactive viz (closed-form OLS partial-out, MAR-mean
// one-step / TMLE / AIPW, partial-linear DML with arbitrary nuisance fitter,
// cross-fit variance EIF, multiplier bootstrap, saddle-surface tangent
// geometry). No precompute scripts — every viz runs entirely client-side.
//
// Most basic linear-algebra and statistical primitives (mulberry32, gaussianRng,
// mean, stddev, sigmoid, normCdf, normInv, logisticFit, linearFit, lassoFit,
// kfoldSplit, etc.) are re-exported from ./causal-inference-methods to keep
// the topic's shared module self-contained from the viz side.
//
// All exports are pure functions — deterministic outputs for a given seed
// via mulberry32 (transitively from normalizing-flows.ts).
//
// Source-of-truth notebook:
//   notebooks/semiparametric-inference/01_semiparametric_inference.ipynb
// =============================================================================

import {
  gaussianRng,
  kfoldSplit,
  linearFit,
  linearPredict,
  logisticFit,
  mean,
  mulberry32,
  normCdf,
  normInv,
  selectRows,
  selectValues,
  sigmoid,
  stddev,
  variance as varianceOf,
} from './causal-inference-methods';

export {
  gaussianRng,
  kfoldSplit,
  linearFit,
  linearPredict,
  logisticFit,
  mean,
  mulberry32,
  normCdf,
  normInv,
  selectRows,
  selectValues,
  sigmoid,
  stddev,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/** Default RNG seed matching the notebook (used by every viz's deterministic mode). */
export const NOTEBOOK_SEED_SEMI = 20260515;

/** Default standard-deviation σ_ε = σ_ν = 0.5 in the Robinson DGP. */
export const ROBINSON_NOISE_SD = 0.5;

/** Default true Robinson slope θ_0 = 1. */
export const ROBINSON_THETA_0 = 1.0;

// -----------------------------------------------------------------------------
// Color palette
// -----------------------------------------------------------------------------

/** Color tokens for the semiparametric-inference viz. CSS variables resolved at render time. */
export const paletteSemi = {
  // Estimators
  plugIn: '#1f77b4',         // blue: plug-in OR baseline
  oneStep: '#d62728',        // red: one-step / AIPW
  tmle: '#2ca02c',           // green: TMLE
  dml: '#9467bd',            // purple: DML / cross-fit
  oracle: '#000000',         // black: BKRW bound / oracle floor
  ipw: '#8c564b',            // brown: IPW (non-efficient)
  completeCase: '#7f7f7f',   // grey: complete-case baseline
  naive: '#7f7f7f',          // grey: naive comparison
  linear: '#1f77b4',         // blue: linear partial-out
  ml: '#d62728',             // red: ML cross-fit
  // Geometry
  tangent: '#1f77b4',        // blue: tangent plane / nuisance direction
  orthogonal: '#2ca02c',     // green: orthogonal complement / efficient score
  ambient: '#d62728',        // red: ambient vector to decompose
  // EIF / variance / bias
  eif: '#d62728',
  bias: '#ff7f0e',           // orange: bias term
  variance: '#1f77b4',
  // CI bands and theory
  ciBand: '#bbbbbb',
  theoryLine: '#000000',
  // Categorical accents
  accentA: '#ff7f0e',        // orange
  accentB: '#17becf',        // teal
} as const;

// -----------------------------------------------------------------------------
// DGPs — §§1, 3, 4, 9, 11
// -----------------------------------------------------------------------------

/** Robinson partial-linear DGP sample (§1.3 of the notebook). */
export type RobinsonSemiSample = {
  /** Row-major n×5 covariate matrix. */
  X: Float64Array;
  /** Continuous treatment D = m(X) + ν. */
  D: Float64Array;
  /** Observed outcome Y = D·θ_0 + g(X) + ε. */
  Y: Float64Array;
  /** Oracle outcome nuisance g(X_i) = sin(2π x_1) + x_2 x_3 + x_4². */
  gOracle: Float64Array;
  /** Oracle treatment nuisance m(X_i) = x_5 / (1 + x_5²). */
  mOracle: Float64Array;
  /** Number of observations. */
  n: number;
};

/**
 * Sample from the Robinson partial-linear DGP.
 *   X ~ U([0,1]^5), ε, ν ~ N(0, σ²) iid (σ = 0.5 default),
 *   g(x) = sin(2π x_1) + x_2 x_3 + x_4², m(x) = x_5 / (1 + x_5²),
 *   D = m(X) + ν, Y = D·θ_0 + g(X) + ε.
 */
export function sampleRobinson(
  n: number,
  theta0: number = ROBINSON_THETA_0,
  rng: () => number = Math.random,
  sigma: number = ROBINSON_NOISE_SD,
): RobinsonSemiSample {
  const X = new Float64Array(n * 5);
  const D = new Float64Array(n);
  const Y = new Float64Array(n);
  const gOracle = new Float64Array(n);
  const mOracle = new Float64Array(n);
  const gauss = gaussianRng(rng);
  const TWO_PI = 2 * Math.PI;
  for (let i = 0; i < n; i++) {
    const base = i * 5;
    let x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0;
    for (let j = 0; j < 5; j++) {
      const v = rng();
      X[base + j] = v;
      if (j === 0) x0 = v;
      else if (j === 1) x1 = v;
      else if (j === 2) x2 = v;
      else if (j === 3) x3 = v;
      else x4 = v;
    }
    const g = Math.sin(TWO_PI * x0) + x1 * x2 + x3 * x3;
    const m = x4 / (1 + x4 * x4);
    gOracle[i] = g;
    mOracle[i] = m;
    const nu = sigma * gauss();
    const eps = sigma * gauss();
    const d = m + nu;
    D[i] = d;
    Y[i] = d * theta0 + g + eps;
  }
  return { X, D, Y, gOracle, mOracle, n };
}

/** MAR-mean DGP sample (§3.4 of the notebook). */
export type MarSemiSample = {
  /** Row-major n×3 covariate matrix. */
  X: Float64Array;
  /** Response/missingness indicator (1 = Y observed). */
  R: Int8Array;
  /** Outcome (only meaningful where R = 1). */
  Y: Float64Array;
  /** Oracle propensity π(X_i) = σ(0.5 − x_1 + x_3). */
  piOracle: Float64Array;
  /** Oracle outcome regression μ(X_i) = 1 + 0.5 x_1 + 2 x_2². */
  mOracle: Float64Array;
  /** Number of observations. */
  n: number;
};

/**
 * Sample from the MAR-mean DGP.
 *   X ~ U([0,1]^3), μ(x) = 1 + 0.5 x_1 + 2 x_2², π(x) = σ(0.5 − x_1 + x_3),
 *   Y | X ~ N(μ(X), 1), R | X ~ Bernoulli(π(X)).
 *   Observed: (Y·R, R, X). Outcome under R=0 is stored as the latent draw,
 *   but the estimators must respect R=1 selection.
 */
export function sampleMar(
  n: number,
  rng: () => number = Math.random,
): MarSemiSample {
  const X = new Float64Array(n * 3);
  const R = new Int8Array(n);
  const Y = new Float64Array(n);
  const piOracle = new Float64Array(n);
  const mOracle = new Float64Array(n);
  const gauss = gaussianRng(rng);
  for (let i = 0; i < n; i++) {
    const base = i * 3;
    const x0 = rng();
    const x1 = rng();
    const x2 = rng();
    X[base] = x0;
    X[base + 1] = x1;
    X[base + 2] = x2;
    const m = 1 + 0.5 * x0 + 2 * x1 * x1;
    const pi = sigmoid(0.5 - x0 + x2);
    mOracle[i] = m;
    piOracle[i] = pi;
    const y = m + gauss();
    Y[i] = y;
    R[i] = rng() < pi ? 1 : 0;
  }
  return { X, R, Y, piOracle, mOracle, n };
}

/** ATE-under-unconfoundedness DGP sample (§9.1 of the notebook). */
export type AteSemiSample = {
  /** Row-major n×5 covariate matrix. */
  X: Float64Array;
  /** Binary treatment indicator. */
  D: Int8Array;
  /** Observed outcome. */
  Y: Float64Array;
  /** Oracle propensity π(X_i) = σ(0.5 − x_1 + 0.5 x_2). */
  piOracle: Float64Array;
  /** Oracle response under control μ_0(X_i) = 1 + 0.5 x_1 + x_2². */
  mu0Oracle: Float64Array;
  /** Oracle response under treatment μ_1(X_i) = μ_0(X_i) + 1. */
  mu1Oracle: Float64Array;
  /** Number of observations. */
  n: number;
};

/**
 * Sample from the ATE DGP with constant true effect θ = 1.
 *   X ~ U([0,1]^5), π(x) = σ(0.5 − x_1 + 0.5 x_2),
 *   μ_0(x) = 1 + 0.5 x_1 + x_2², μ_1 = μ_0 + 1, Y = μ_D(X) + ε, ε ~ N(0,1).
 */
export function sampleAte(
  n: number,
  rng: () => number = Math.random,
): AteSemiSample {
  const X = new Float64Array(n * 5);
  const D = new Int8Array(n);
  const Y = new Float64Array(n);
  const piOracle = new Float64Array(n);
  const mu0Oracle = new Float64Array(n);
  const mu1Oracle = new Float64Array(n);
  const gauss = gaussianRng(rng);
  for (let i = 0; i < n; i++) {
    const base = i * 5;
    let x0 = 0, x1 = 0;
    for (let j = 0; j < 5; j++) {
      const v = rng();
      X[base + j] = v;
      if (j === 0) x0 = v;
      else if (j === 1) x1 = v;
    }
    const mu0 = 1 + 0.5 * x0 + x1 * x1;
    const mu1 = mu0 + 1;
    const pi = sigmoid(0.5 - x0 + 0.5 * x1);
    mu0Oracle[i] = mu0;
    mu1Oracle[i] = mu1;
    piOracle[i] = pi;
    const di = rng() < pi ? 1 : 0;
    D[i] = di;
    Y[i] = (di === 1 ? mu1 : mu0) + gauss();
  }
  return { X, D, Y, piOracle, mu0Oracle, mu1Oracle, n };
}

/** UQ variance-functional DGP sample (§11 of the notebook). */
export type UqSemiSample = {
  /** Row-major n×5 covariate matrix. */
  X: Float64Array;
  /** Observed outcome Y = f(X) + ε. */
  Y: Float64Array;
  /** Oracle regression f(X_i) = 1 + x_1 + x_2² + sin(π x_3). */
  fOracle: Float64Array;
  /** Number of observations. */
  n: number;
};

/**
 * UQ DGP.
 *   X ~ U([0,1]^5), f(x) = 1 + x_1 + x_2² + sin(π x_3), Y = f(X) + ε, ε ~ N(0,1).
 *   True residual variance σ² = 1.
 */
export function sampleUq(
  n: number,
  rng: () => number = Math.random,
): UqSemiSample {
  const X = new Float64Array(n * 5);
  const Y = new Float64Array(n);
  const fOracle = new Float64Array(n);
  const gauss = gaussianRng(rng);
  for (let i = 0; i < n; i++) {
    const base = i * 5;
    let x0 = 0, x1 = 0, x2 = 0;
    for (let j = 0; j < 5; j++) {
      const v = rng();
      X[base + j] = v;
      if (j === 0) x0 = v;
      else if (j === 1) x1 = v;
      else if (j === 2) x2 = v;
    }
    const f = 1 + x0 + x1 * x1 + Math.sin(Math.PI * x2);
    fOracle[i] = f;
    Y[i] = f + gauss();
  }
  return { X, Y, fOracle, n };
}

// -----------------------------------------------------------------------------
// Polynomial features with full interactions (§9.2 nuisance fitter)
// -----------------------------------------------------------------------------

/**
 * Generate the column index map for polynomial features of degree ≤ d with full
 * interactions on p inputs. Returns the multi-indices: each row is a length-p
 * exponent vector summing to ≤ d. Excludes the bias term (intercept = 0 vec).
 * For p=5, d=3 this yields C(5+3, 3) − 1 = 55 features (excluding bias).
 */
export function polynomialMultiIndices(p: number, d: number): number[][] {
  const out: number[][] = [];
  const cur = new Array<number>(p).fill(0);
  function rec(idx: number, remaining: number) {
    if (idx === p) {
      const total = cur.reduce((a, b) => a + b, 0);
      if (total > 0) out.push(cur.slice());
      return;
    }
    for (let v = 0; v <= remaining; v++) {
      cur[idx] = v;
      rec(idx + 1, remaining - v);
    }
    cur[idx] = 0;
  }
  rec(0, d);
  return out;
}

/**
 * Build polynomial features (degree ≤ d, full interactions, no bias) from
 * row-major X (n×p). Output is row-major n×F where F = number of multi-indices.
 */
export function polynomialFeatures(
  X: Float64Array,
  n: number,
  p: number,
  d: number,
): { features: Float64Array; nFeatures: number } {
  const multi = polynomialMultiIndices(p, d);
  const F = multi.length;
  const features = new Float64Array(n * F);
  for (let i = 0; i < n; i++) {
    const baseSrc = i * p;
    const baseDst = i * F;
    for (let k = 0; k < F; k++) {
      const idx = multi[k];
      let prod = 1;
      for (let j = 0; j < p; j++) {
        const e = idx[j];
        if (e === 0) continue;
        const x = X[baseSrc + j];
        if (e === 1) prod *= x;
        else if (e === 2) prod *= x * x;
        else if (e === 3) prod *= x * x * x;
        else prod *= Math.pow(x, e);
      }
      features[baseDst + k] = prod;
    }
  }
  return { features, nFeatures: F };
}

// -----------------------------------------------------------------------------
// Propensity helpers
// -----------------------------------------------------------------------------

/** Clip a propensity to [lo, hi]. Default [0.05, 0.95] for ATE numerical stability. */
export function propensityClip(pi: number, lo: number = 0.05, hi: number = 0.95): number {
  if (pi < lo) return lo;
  if (pi > hi) return hi;
  return pi;
}

// -----------------------------------------------------------------------------
// Tangent-space geometry (§2.4 — saddle surface, tangent plane, decomposition)
// -----------------------------------------------------------------------------

/** Vector in ℝ³. */
export type Vec3 = [number, number, number];

/** Normalize a 3-vector to unit length. */
export function normalize3(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (n === 0) return [0, 0, 0];
  return [v[0] / n, v[1] / n, v[2] / n];
}

/** Cross product. */
export function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/** Dot product. */
export function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Saddle surface z = c·(x² − y²) sampled at a grid of (u, v) parameter values.
 * Returns Z[i][j] = z(uGrid[i], vGrid[j]) plus tangent-plane / decomposition
 * primitives.
 */
export function saddleSurface(
  uGrid: number[],
  vGrid: number[],
  c: number = 0.4,
): {
  z: number[][];
  /** Tangent plane at (u₀, v₀): {normal, basis vectors b1, b2 in the plane}. */
  tangentPlaneAt: (u0: number, v0: number) => {
    point: Vec3;
    normal: Vec3;
    basis: [Vec3, Vec3];
  };
  /**
   * Decompose a vector v in the tangent plane (specified by α ∈ [0, 2π], the
   * angle from b1) into (a) the nuisance direction at angle α, (b) the
   * orthogonal complement at angle α + π/2 (within the tangent plane). The
   * vector itself is parameterized as `vector` = a·dirNuis + b·dirPerp.
   */
  decomposeVector: (
    u0: number,
    v0: number,
    alpha: number,
    vector: Vec3,
  ) => {
    nuisanceDir: Vec3;
    orthogonalDir: Vec3;
    coordNuisance: number;
    coordOrthogonal: number;
  };
} {
  const nu = uGrid.length;
  const nv = vGrid.length;
  const z: number[][] = new Array(nu);
  for (let i = 0; i < nu; i++) {
    const u = uGrid[i];
    const row = new Array<number>(nv);
    for (let j = 0; j < nv; j++) {
      const v = vGrid[j];
      row[j] = c * (u * u - v * v);
    }
    z[i] = row;
  }
  return {
    z,
    tangentPlaneAt(u0, v0) {
      // ∂_u z = 2 c u, ∂_v z = −2 c v at (u₀, v₀).
      const dzu = 2 * c * u0;
      const dzv = -2 * c * v0;
      const b1 = normalize3([1, 0, dzu]);
      const b2v: Vec3 = [0, 1, dzv];
      // Orthogonalize b2 against b1.
      const proj = dot3(b2v, b1);
      const b2o: Vec3 = [b2v[0] - proj * b1[0], b2v[1] - proj * b1[1], b2v[2] - proj * b1[2]];
      const b2 = normalize3(b2o);
      const normal = normalize3(cross3(b1, b2));
      const point: Vec3 = [u0, v0, c * (u0 * u0 - v0 * v0)];
      return { point, normal, basis: [b1, b2] };
    },
    decomposeVector(u0, v0, alpha, vector) {
      const dzu = 2 * c * u0;
      const dzv = -2 * c * v0;
      const b1 = normalize3([1, 0, dzu]);
      const b2v: Vec3 = [0, 1, dzv];
      const proj = dot3(b2v, b1);
      const b2o: Vec3 = [b2v[0] - proj * b1[0], b2v[1] - proj * b1[1], b2v[2] - proj * b1[2]];
      const b2 = normalize3(b2o);
      const ca = Math.cos(alpha);
      const sa = Math.sin(alpha);
      const nuisanceDir: Vec3 = [
        ca * b1[0] + sa * b2[0],
        ca * b1[1] + sa * b2[1],
        ca * b1[2] + sa * b2[2],
      ];
      const orthogonalDir: Vec3 = [
        -sa * b1[0] + ca * b2[0],
        -sa * b1[1] + ca * b2[1],
        -sa * b1[2] + ca * b2[2],
      ];
      // Project `vector` onto the tangent-plane basis (b1, b2) first, then onto the rotated basis.
      const coordNuisance = dot3(vector, nuisanceDir);
      const coordOrthogonal = dot3(vector, orthogonalDir);
      return { nuisanceDir, orthogonalDir, coordNuisance, coordOrthogonal };
    },
  };
}

// -----------------------------------------------------------------------------
// §1.4 partial-linear teaser estimators
// -----------------------------------------------------------------------------

/** Naive OLS slope of Y on D (one regressor + intercept). */
export function thetaNaive(D: Float64Array, Y: Float64Array): number {
  const n = Y.length;
  let sumD = 0, sumY = 0, sumDD = 0, sumDY = 0;
  for (let i = 0; i < n; i++) {
    sumD += D[i];
    sumY += Y[i];
    sumDD += D[i] * D[i];
    sumDY += D[i] * Y[i];
  }
  const meanD = sumD / n;
  const meanY = sumY / n;
  const cov = sumDY / n - meanD * meanY;
  const varD = sumDD / n - meanD * meanD;
  return cov / varD;
}

/**
 * Linear partial-out slope: OLS of Y on (D, X) (intercept added by linearFit)
 * and read the D coefficient. Equivalent (by Frisch–Waugh–Lovell) to
 * partialling out best-linear-in-X projection of Y and D, then regressing
 * residuals.
 */
export function thetaLinearControls(
  X: Float64Array,
  D: Float64Array,
  Y: Float64Array,
  n: number,
  p: number,
): number {
  // Build design matrix [D, X] of shape n × (p + 1). linearFit prepends intercept.
  const pa = p + 1;
  const Xa = new Float64Array(n * pa);
  for (let i = 0; i < n; i++) {
    const baseSrc = i * p;
    const baseDst = i * pa;
    Xa[baseDst] = D[i];
    for (let j = 0; j < p; j++) Xa[baseDst + 1 + j] = X[baseSrc + j];
  }
  const beta = linearFit(Xa, Y, n, pa);
  // linearFit prepends intercept, so beta[0] = intercept, beta[1] = D coef.
  return beta[1];
}

/**
 * Two-fold cross-fitted ML partial-out slope using a custom nuisance fitter
 * for ĝ(X) and m̂(X). The fitter returns (gPred, mPred) arrays evaluated on
 * the test fold. Solves OLS on the cross-fit residuals.
 */
export function thetaMlCrossFit(
  X: Float64Array,
  D: Float64Array,
  Y: Float64Array,
  n: number,
  p: number,
  fitGm: (
    Xtr: Float64Array,
    Dtr: Float64Array,
    Ytr: Float64Array,
    ntr: number,
    Xte: Float64Array,
    nte: number,
  ) => { gPred: Float64Array; mPred: Float64Array },
  rng: () => number,
): number {
  const folds = kfoldSplit(n, 2, rng);
  const tildeY = new Float64Array(n);
  const tildeD = new Float64Array(n);
  for (const { trainIdx, testIdx } of folds) {
    const Xtr = selectRows(X, trainIdx, p);
    const Dtr = selectValues(D, trainIdx);
    const Ytr = selectValues(Y, trainIdx);
    const Xte = selectRows(X, testIdx, p);
    const { gPred, mPred } = fitGm(Xtr, Dtr, Ytr, trainIdx.length, Xte, testIdx.length);
    for (let k = 0; k < testIdx.length; k++) {
      const idx = testIdx[k];
      tildeY[idx] = Y[idx] - gPred[k];
      tildeD[idx] = D[idx] - mPred[k];
    }
  }
  // OLS slope of tildeY on tildeD (no intercept since both are residualized).
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += tildeD[i] * tildeY[i];
    den += tildeD[i] * tildeD[i];
  }
  return num / den;
}

// -----------------------------------------------------------------------------
// §3.4 MAR-mean estimators with truth-substituted nuisances
// -----------------------------------------------------------------------------

/** Complete-case mean: average Y over R=1 cases. Biased under MAR. */
export function completeCaseMean(R: Int8Array, Y: Float64Array): number {
  let sum = 0, count = 0;
  const n = R.length;
  for (let i = 0; i < n; i++) if (R[i] === 1) { sum += Y[i]; count += 1; }
  return count > 0 ? sum / count : NaN;
}

/** Horvitz–Thompson IPW mean: (1/n) Σ R_i Y_i / π_i. */
export function ipwMean(
  R: Int8Array,
  Y: Float64Array,
  piHat: Float64Array,
): number {
  const n = R.length;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const pi = Math.max(piHat[i], 1e-6);
    s += (R[i] * Y[i]) / pi;
  }
  return s / n;
}

/** Outcome-regression plug-in: (1/n) Σ m̂(X_i). */
export function plugInMean(mHat: Float64Array): number {
  return mean(mHat);
}

/** AIPW (one-step) MAR mean given fitted nuisances. Returns {psi, eif, se}. */
export function oneStepMar(
  R: Int8Array,
  Y: Float64Array,
  mHat: Float64Array,
  piHat: Float64Array,
): { psi: number; eif: Float64Array; se: number } {
  const n = R.length;
  const plug = mean(mHat);
  let correction = 0;
  for (let i = 0; i < n; i++) {
    const pi = Math.max(piHat[i], 1e-6);
    correction += (R[i] * (Y[i] - mHat[i])) / pi;
  }
  correction /= n;
  const psi = plug + correction;
  // Build per-observation EIF: φ*_i = m̂(X_i) − ψ̂ + (R_i / π̂_i)(Y_i − m̂(X_i)).
  const eif = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const pi = Math.max(piHat[i], 1e-6);
    eif[i] = mHat[i] - psi + (R[i] * (Y[i] - mHat[i])) / pi;
  }
  const se = stddev(eif) / Math.sqrt(n);
  return { psi, eif, se };
}

/**
 * TMLE for the MAR-mean (continuous Y, linear fluctuation).
 * Clever covariate H(X) = 1/π̂(X). Closed-form ε̂ via OLS of the
 * residual (Y − m̂_0) on H, restricted to R = 1.
 */
export function tmleMar(
  R: Int8Array,
  Y: Float64Array,
  mHat: Float64Array,
  piHat: Float64Array,
): {
  psi: number;
  eif: Float64Array;
  se: number;
  epsilon: number;
  mStar: Float64Array;
  /** Sequence of (ε, ψ̂) pairs across the targeting iteration (1 step for AIPW). */
  trace: { epsilon: number; psi: number; eifMean: number }[];
} {
  const n = R.length;
  const mStar = new Float64Array(n);
  // OLS of (Y − m̂_0) on H over observed cases.
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    if (R[i] === 1) {
      const H = 1 / Math.max(piHat[i], 1e-6);
      num += H * (Y[i] - mHat[i]);
      den += H * H;
    }
  }
  const epsilon = den > 1e-12 ? num / den : 0;
  for (let i = 0; i < n; i++) {
    const H = 1 / Math.max(piHat[i], 1e-6);
    mStar[i] = mHat[i] + epsilon * H;
  }
  const psi = mean(mStar);
  const eif = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const pi = Math.max(piHat[i], 1e-6);
    eif[i] = mStar[i] - psi + (R[i] * (Y[i] - mStar[i])) / pi;
  }
  const se = stddev(eif) / Math.sqrt(n);
  // Trace: ψ̂_0 (plug-in pre-targeting) → ψ̂_1 (post-targeting).
  const psi0 = mean(mHat);
  let eifMean0 = 0;
  for (let i = 0; i < n; i++) {
    const pi = Math.max(piHat[i], 1e-6);
    eifMean0 += mHat[i] - psi0 + (R[i] * (Y[i] - mHat[i])) / pi;
  }
  eifMean0 /= n;
  let eifMean1 = 0;
  for (let i = 0; i < n; i++) eifMean1 += eif[i];
  eifMean1 /= n;
  return {
    psi,
    eif,
    se,
    epsilon,
    mStar,
    trace: [
      { epsilon: 0, psi: psi0, eifMean: eifMean0 },
      { epsilon, psi, eifMean: eifMean1 },
    ],
  };
}

// -----------------------------------------------------------------------------
// MAR nuisance fitters (parametric and misspecified variants for §13.3)
// -----------------------------------------------------------------------------

/**
 * Build MAR outcome features [x_1, x_2, x_2², x_3] (correct specification).
 * No bias column — linearFit / logisticFit prepend their own intercept.
 */
export function marCorrectOutcomeFeatures(
  X: Float64Array,
  n: number,
): Float64Array {
  const F = new Float64Array(n * 4);
  for (let i = 0; i < n; i++) {
    const base = i * 3;
    const x0 = X[base], x1 = X[base + 1], x2 = X[base + 2];
    const baseDst = i * 4;
    F[baseDst] = x0;
    F[baseDst + 1] = x1;
    F[baseDst + 2] = x1 * x1;
    F[baseDst + 3] = x2;
  }
  return F;
}

/** Misspecified outcome features [x_1, x_2, x_3] (omits x_2² — biases m̂). */
export function marWrongOutcomeFeatures(
  X: Float64Array,
  n: number,
): Float64Array {
  const F = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const base = i * 3;
    const baseDst = i * 3;
    F[baseDst] = X[base];
    F[baseDst + 1] = X[base + 1];
    F[baseDst + 2] = X[base + 2];
  }
  return F;
}

/** Build MAR propensity features [x_1, x_2, x_3] (correct). */
export function marCorrectPropensityFeatures(
  X: Float64Array,
  n: number,
): Float64Array {
  const F = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const base = i * 3;
    const baseDst = i * 3;
    F[baseDst] = X[base];
    F[baseDst + 1] = X[base + 1];
    F[baseDst + 2] = X[base + 2];
  }
  return F;
}

/** Misspecified propensity features [x_1, x_2] (omits x_3 — biases π̂). */
export function marWrongPropensityFeatures(
  X: Float64Array,
  n: number,
): Float64Array {
  const F = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    const base = i * 3;
    const baseDst = i * 2;
    F[baseDst] = X[base];
    F[baseDst + 1] = X[base + 1];
  }
  return F;
}

// -----------------------------------------------------------------------------
// Cross-fit AIPW for the MAR mean
// -----------------------------------------------------------------------------

/**
 * Cross-fit AIPW estimator for the MAR mean.
 *   - Splits into K folds.
 *   - On each fold's complement, fits outcome regression m̂ on observed cases
 *     (R=1) via OLS on `outcomeFeatures(X, n)`, and propensity π̂ via logistic
 *     regression on `propensityFeatures(X, n)` over all cases.
 *   - Computes the AIPW score on the held-out fold using the cross-fit
 *     nuisances.
 *   - Returns ψ̂, sandwich SE, and the per-observation cross-fit EIF.
 */
export function crossFitAipwMar(
  X: Float64Array,
  R: Int8Array,
  Y: Float64Array,
  K: number,
  outcomeFeatures: (X: Float64Array, n: number) => Float64Array,
  propensityFeatures: (X: Float64Array, n: number) => Float64Array,
  rng: () => number,
): { psi: number; se: number; phi: Float64Array } {
  const n = R.length;
  const p = X.length / n;
  const folds = kfoldSplit(n, K, rng);
  const phi = new Float64Array(n);
  const mHatAll = new Float64Array(n);
  const piHatAll = new Float64Array(n);
  for (const { trainIdx, testIdx } of folds) {
    // Build training data restricted to observed cases for outcome regression.
    const observedTrain = trainIdx.filter((i) => R[i] === 1);
    const XobsRows = selectRows(X, observedTrain, p);
    const Yobs = selectValues(Y, observedTrain);
    const FobsTr = outcomeFeatures(XobsRows, observedTrain.length);
    const pOutcome = FobsTr.length / observedTrain.length;
    // linearFit prepends intercept; features have no bias column by convention.
    const betaM = linearFit(FobsTr, Yobs, observedTrain.length, pOutcome);
    // Propensity training on all train rows.
    const XtrRows = selectRows(X, trainIdx, p);
    const Rtr = selectValues(R, trainIdx);
    const FpTr = propensityFeatures(XtrRows, trainIdx.length);
    const pProp = FpTr.length / trainIdx.length;
    // logisticFit prepends intercept; features have no bias column by convention.
    const betaPi = logisticFit(FpTr, Rtr, trainIdx.length, pProp);
    // Score test fold.
    const XteRows = selectRows(X, testIdx, p);
    const nte = testIdx.length;
    const FteOutcome = outcomeFeatures(XteRows, nte);
    const FteProp = propensityFeatures(XteRows, nte);
    for (let k = 0; k < nte; k++) {
      const idx = testIdx[k];
      let mHat = betaM[0]; // intercept added by linearFit
      for (let j = 0; j < pOutcome; j++) mHat += betaM[j + 1] * FteOutcome[k * pOutcome + j];
      mHatAll[idx] = mHat;
      let z = betaPi[0]; // intercept added by logisticFit
      for (let j = 0; j < pProp; j++) z += betaPi[j + 1] * FteProp[k * pProp + j];
      const pi = sigmoid(z);
      piHatAll[idx] = Math.max(pi, 1e-6);
    }
  }
  // Compute AIPW: ψ̂ = mean(m̂) + (1/n) Σ R(Y − m̂)/π̂.
  let plug = 0;
  for (let i = 0; i < n; i++) plug += mHatAll[i];
  plug /= n;
  let correction = 0;
  for (let i = 0; i < n; i++) {
    correction += (R[i] * (Y[i] - mHatAll[i])) / piHatAll[i];
  }
  correction /= n;
  const psi = plug + correction;
  for (let i = 0; i < n; i++) {
    phi[i] = mHatAll[i] - psi + (R[i] * (Y[i] - mHatAll[i])) / piHatAll[i];
  }
  const se = stddev(phi) / Math.sqrt(n);
  return { psi, se, phi };
}

// -----------------------------------------------------------------------------
// Cross-fit AIPW for the ATE under unconfoundedness
// -----------------------------------------------------------------------------

/**
 * Cross-fit AIPW estimator for the ATE under unconfoundedness. Uses
 * polynomial-with-interactions features for both nuisances (degree 2 default).
 * Propensity clipped at [0.05, 0.95].
 */
export function crossFitAipwAte(
  X: Float64Array,
  D: Int8Array,
  Y: Float64Array,
  K: number,
  rng: () => number,
  degree: number = 2,
): { tau: number; se: number; phi: Float64Array } {
  const n = D.length;
  const p = X.length / n;
  const folds = kfoldSplit(n, K, rng);
  const mu1All = new Float64Array(n);
  const mu0All = new Float64Array(n);
  const piAll = new Float64Array(n);
  for (const { trainIdx, testIdx } of folds) {
    // Polynomial features once per fold (training and test sets).
    const XtrRows = selectRows(X, trainIdx, p);
    const Dtr = selectValues(D, trainIdx);
    const Ytr = selectValues(Y, trainIdx);
    const Ftr = polynomialFeatures(XtrRows, trainIdx.length, p, degree);
    const F = Ftr.nFeatures;
    // Fit mu_1: regress Y on features over treated (D=1) train rows.
    // linearFit prepends intercept; pass raw poly features.
    const treatedTrain: number[] = [];
    const controlTrain: number[] = [];
    for (let i = 0; i < trainIdx.length; i++) {
      if (Dtr[i] === 1) treatedTrain.push(i);
      else controlTrain.push(i);
    }
    const Ftr1 = selectRows(Ftr.features, treatedTrain, F);
    const Ytr1 = selectValues(Ytr, treatedTrain);
    const Ftr0 = selectRows(Ftr.features, controlTrain, F);
    const Ytr0 = selectValues(Ytr, controlTrain);
    const betaMu1 = linearFit(Ftr1, Ytr1, treatedTrain.length, F);
    const betaMu0 = linearFit(Ftr0, Ytr0, controlTrain.length, F);
    // Fit propensity: logistic regression of D on features (logisticFit adds intercept).
    const betaPi = logisticFit(Ftr.features, Dtr, trainIdx.length, F);
    // Score test fold.
    const XteRows = selectRows(X, testIdx, p);
    const nte = testIdx.length;
    const Fte = polynomialFeatures(XteRows, nte, p, degree);
    for (let k = 0; k < nte; k++) {
      const idx = testIdx[k];
      let mu1 = betaMu1[0]; // intercept
      let mu0 = betaMu0[0];
      for (let j = 0; j < F; j++) {
        mu1 += betaMu1[j + 1] * Fte.features[k * F + j];
        mu0 += betaMu0[j + 1] * Fte.features[k * F + j];
      }
      mu1All[idx] = mu1;
      mu0All[idx] = mu0;
      let z = betaPi[0];
      for (let j = 0; j < F; j++) z += betaPi[j + 1] * Fte.features[k * F + j];
      piAll[idx] = propensityClip(sigmoid(z));
    }
  }
  // AIPW score and ATE.
  const phi = new Float64Array(n);
  let tau = 0;
  for (let i = 0; i < n; i++) {
    const e = piAll[i];
    phi[i] =
      mu1All[i] - mu0All[i] +
      (D[i] * (Y[i] - mu1All[i])) / e -
      ((1 - D[i]) * (Y[i] - mu0All[i])) / (1 - e);
    tau += phi[i];
  }
  tau /= n;
  // EIF is phi − tau for centered IF.
  for (let i = 0; i < n; i++) phi[i] = phi[i] - tau;
  const se = stddev(phi) / Math.sqrt(n);
  return { tau, se, phi };
}

// -----------------------------------------------------------------------------
// Cross-fit DML for the Robinson partial-linear model
// -----------------------------------------------------------------------------

/** Nuisance fitter for the Robinson DML: outputs (ĝ, m̂) on the test fold. */
export type RobinsonNuisanceFitter = (
  Xtr: Float64Array,
  Dtr: Float64Array,
  Ytr: Float64Array,
  ntr: number,
  Xte: Float64Array,
  nte: number,
  p: number,
) => { gPred: Float64Array; mPred: Float64Array };

/**
 * Cross-fitted DML for the partial-linear Robinson model.
 *   - K-fold cross-fitting.
 *   - On each fold's complement, fit ĝ(X) ≈ E[Y|X] and m̂(X) ≈ E[D|X].
 *   - Form cross-fit residuals \tilde Y_i = Y_i − ĝ^{(−k(i))}(X_i),
 *     \tilde D_i = D_i − m̂^{(−k(i))}(X_i).
 *   - Solve θ̂ = Σ \tilde D \tilde Y / Σ \tilde D².
 *   - Sandwich SE via the EIF: ψ_i = \tilde D_i (\tilde Y_i − θ̂ \tilde D_i),
 *     normalized by the Jacobian (1/n) Σ \tilde D D.
 *
 * Naming: the moment function evaluated at θ̂ used to be called `psi_hat`; the
 * notebook renamed it `moment_at_hat` to disambiguate from the (centered) EIF
 * `psi`. We follow the post-rename convention.
 */
export function crossFitDmlPlr(
  X: Float64Array,
  D: Float64Array,
  Y: Float64Array,
  K: number,
  fitter: RobinsonNuisanceFitter,
  rng: () => number,
): { theta: number; se: number; tildeD: Float64Array; tildeY: Float64Array } {
  const n = D.length;
  const p = X.length / n;
  const folds = kfoldSplit(n, K, rng);
  const tildeD = new Float64Array(n);
  const tildeY = new Float64Array(n);
  for (const { trainIdx, testIdx } of folds) {
    const Xtr = selectRows(X, trainIdx, p);
    const Dtr = selectValues(D, trainIdx);
    const Ytr = selectValues(Y, trainIdx);
    const Xte = selectRows(X, testIdx, p);
    const { gPred, mPred } = fitter(Xtr, Dtr, Ytr, trainIdx.length, Xte, testIdx.length, p);
    for (let k = 0; k < testIdx.length; k++) {
      const idx = testIdx[k];
      tildeY[idx] = Y[idx] - gPred[k];
      tildeD[idx] = D[idx] - mPred[k];
    }
  }
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += tildeD[i] * tildeY[i];
    den += tildeD[i] * tildeD[i];
  }
  const theta = num / den;
  // Sandwich SE: Var(moment_at_hat) / (E[tildeD D])^2 / n
  // For partial-linear E[tildeD D] = E[tildeD²] + E[tildeD m] = E[tildeD²]
  // (since tildeD ⊥ m_hat by construction in cross-fit). Equivalently: jacobian = den/n.
  let momentSq = 0;
  for (let i = 0; i < n; i++) {
    const m = tildeD[i] * (tildeY[i] - theta * tildeD[i]);
    momentSq += m * m;
  }
  const jacobian = den / n;
  const se = Math.sqrt(momentSq / n / (jacobian * jacobian) / n);
  return { theta, se, tildeD, tildeY };
}

/** Linear-control Robinson nuisance fitter (best-linear-in-X approximation). */
export const linearRobinsonNuisanceFitter: RobinsonNuisanceFitter = (
  Xtr, Dtr, Ytr, ntr, Xte, nte, p,
) => {
  // linearFit prepends its own intercept; pass raw X with p columns.
  const betaG = linearFit(Xtr, Ytr, ntr, p);
  const betaM = linearFit(Xtr, Dtr, ntr, p);
  const gPred = new Float64Array(nte);
  const mPred = new Float64Array(nte);
  for (let i = 0; i < nte; i++) {
    let g = betaG[0]; // intercept
    let m = betaM[0];
    for (let j = 0; j < p; j++) {
      g += betaG[j + 1] * Xte[i * p + j];
      m += betaM[j + 1] * Xte[i * p + j];
    }
    gPred[i] = g;
    mPred[i] = m;
  }
  return { gPred, mPred };
};

/** Polynomial-degree-3 (full interactions) Robinson nuisance fitter (§9.2). */
export const polyDeg3RobinsonNuisanceFitter: RobinsonNuisanceFitter = (
  Xtr, Dtr, Ytr, ntr, Xte, nte, p,
) => {
  const trF = polynomialFeatures(Xtr, ntr, p, 3);
  const teF = polynomialFeatures(Xte, nte, p, 3);
  const F = trF.nFeatures;
  // linearFit prepends intercept; pass raw poly features.
  const betaG = linearFit(trF.features, Ytr, ntr, F);
  const betaM = linearFit(trF.features, Dtr, ntr, F);
  const gPred = new Float64Array(nte);
  const mPred = new Float64Array(nte);
  for (let i = 0; i < nte; i++) {
    let g = betaG[0];
    let m = betaM[0];
    for (let j = 0; j < F; j++) {
      g += betaG[j + 1] * teF.features[i * F + j];
      m += betaM[j + 1] * teF.features[i * F + j];
    }
    gPred[i] = g;
    mPred[i] = m;
  }
  return { gPred, mPred };
};

/**
 * Truth-substituted Robinson nuisance: returns the oracle (g_0, m_0) at every
 * test point. Useful as a benchmark when the DGP is in hand.
 */
export function makeOracleRobinsonNuisanceFitter(
  gOracle: Float64Array,
  mOracle: Float64Array,
  fullX: Float64Array,
  fullN: number,
  p: number,
): RobinsonNuisanceFitter {
  // We need to look up the oracle values by matching X back to the master
  // index. The caller supplies the entire X/g/m and we match by row identity
  // (since cross-fit selectRows preserves contents). Inverse-mapping is via a
  // serialized-key lookup on the test X.
  const key = (X: Float64Array, i: number) => {
    let k = '';
    for (let j = 0; j < p; j++) k += X[i * p + j].toString() + ':';
    return k;
  };
  const map = new Map<string, number>();
  for (let i = 0; i < fullN; i++) map.set(key(fullX, i), i);
  return (_Xtr, _Dtr, _Ytr, _ntr, Xte, nte) => {
    const gPred = new Float64Array(nte);
    const mPred = new Float64Array(nte);
    for (let i = 0; i < nte; i++) {
      const k = key(Xte, i);
      const idx = map.get(k);
      if (idx == null) {
        gPred[i] = 0;
        mPred[i] = 0;
      } else {
        gPred[i] = gOracle[idx];
        mPred[i] = mOracle[idx];
      }
    }
    return { gPred, mPred };
  };
}

// -----------------------------------------------------------------------------
// Variance functionals (§11)
// -----------------------------------------------------------------------------

/**
 * In-sample plug-in variance: fit a regression on the full sample, compute
 * residuals on the SAME sample. Biased downward due to over-fitting. The
 * fitter signature mirrors crossFitVarianceEif so the two are interchangeable
 * in the §11 viz.
 */
export function plugInVarianceInSample(
  X: Float64Array,
  Y: Float64Array,
  n: number,
  p: number,
  fitter: (
    Xtr: Float64Array,
    Ytr: Float64Array,
    ntr: number,
    Xte: Float64Array,
    nte: number,
    p: number,
  ) => Float64Array,
): number {
  const fHat = fitter(X, Y, n, X, n, p);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const r = Y[i] - fHat[i];
    s += r * r;
  }
  return s / n;
}

/**
 * Cross-fit one-step (EIF-based) variance: each observation's residual is
 * computed using a regression fit on the complement fold. By the §11.1
 * derivation, this collapses to the cross-fit residual variance.
 */
export function crossFitVarianceEif(
  X: Float64Array,
  Y: Float64Array,
  n: number,
  p: number,
  K: number,
  fitter: (
    Xtr: Float64Array,
    Ytr: Float64Array,
    ntr: number,
    Xte: Float64Array,
    nte: number,
    p: number,
  ) => Float64Array,
  rng: () => number,
): { sigma2: number; se: number; residuals: Float64Array } {
  const folds = kfoldSplit(n, K, rng);
  const residuals = new Float64Array(n);
  for (const { trainIdx, testIdx } of folds) {
    const Xtr = selectRows(X, trainIdx, p);
    const Ytr = selectValues(Y, trainIdx);
    const Xte = selectRows(X, testIdx, p);
    const fPred = fitter(Xtr, Ytr, trainIdx.length, Xte, testIdx.length, p);
    for (let k = 0; k < testIdx.length; k++) {
      const idx = testIdx[k];
      residuals[idx] = Y[idx] - fPred[k];
    }
  }
  // σ̂² = mean(r²), SE from Var(r²)/n.
  const sq = new Float64Array(n);
  for (let i = 0; i < n; i++) sq[i] = residuals[i] * residuals[i];
  const sigma2 = mean(sq);
  const se = stddev(sq) / Math.sqrt(n);
  return { sigma2, se, residuals };
}

/**
 * Linear-features-of-degree-2 fitter for §11. Builds polynomial features of
 * degree 2 with full interactions (20 features for p=5) and fits OLS.
 * linearFit prepends its own intercept; we pass raw poly features.
 */
export const polyDeg2VarianceFitter = (
  Xtr: Float64Array,
  Ytr: Float64Array,
  ntr: number,
  Xte: Float64Array,
  nte: number,
  p: number,
): Float64Array => {
  const trF = polynomialFeatures(Xtr, ntr, p, 2);
  const teF = polynomialFeatures(Xte, nte, p, 2);
  const F = trF.nFeatures;
  const beta = linearFit(trF.features, Ytr, ntr, F);
  const fPred = new Float64Array(nte);
  for (let i = 0; i < nte; i++) {
    let y_pred = beta[0]; // intercept added by linearFit
    for (let j = 0; j < F; j++) y_pred += beta[j + 1] * teF.features[i * F + j];
    fPred[i] = y_pred;
  }
  return fPred;
};

// -----------------------------------------------------------------------------
// Sandwich SE and multiplier bootstrap (§10)
// -----------------------------------------------------------------------------

/** Sandwich SE from an empirical EIF: sqrt(Var(φ)/n). */
export function sandwichSeEif(phi: Float64Array): number {
  return stddev(phi) / Math.sqrt(phi.length);
}

/**
 * Multiplier bootstrap SE using centered Mammen-type two-point weights.
 *   W_i ∈ {(1 − √5)/2, (1 + √5)/2} with probability (5 + √5)/10 and
 *   (5 − √5)/10 respectively (matching E[W]=1, E[W²]=2).
 *   Each bootstrap pseudo-estimate is ψ̂*_b = ψ̂ + (1/n) Σ (W_i − 1)(φ_i − φ̄),
 *   so the bootstrap variance equals the sandwich variance (asymptotically).
 *   Returns the SD across B bootstrap replicates.
 */
export function multiplierBootstrapSe(
  phi: Float64Array,
  B: number,
  rng: () => number,
): number {
  const n = phi.length;
  const phiMean = mean(phi);
  const centered = new Float64Array(n);
  for (let i = 0; i < n; i++) centered[i] = phi[i] - phiMean;
  const sqrt5 = Math.sqrt(5);
  const wPlus = (1 + sqrt5) / 2;
  const wMinus = (1 - sqrt5) / 2;
  const pPlus = (5 - sqrt5) / 10;
  const draws = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const W = rng() < pPlus ? wPlus : wMinus;
      s += (W - 1) * centered[i];
    }
    draws[b] = s / n;
  }
  return stddev(draws);
}

// -----------------------------------------------------------------------------
// §4 efficiency-bound numerical predictor for MAR
// -----------------------------------------------------------------------------

/**
 * Empirical efficiency bound for the MAR-mean DGP using the oracle nuisances.
 *   V_eff = Var(m(X)) + E[σ²(X) / π(X)], here σ² = 1.
 * Returns V_eff and the predicted SD floor √(V_eff / n).
 */
export function marEfficiencyBound(
  piOracle: Float64Array,
  mOracle: Float64Array,
  sigma2: number = 1,
): { Veff: number; sdAt: (n: number) => number } {
  const n = piOracle.length;
  const meanM = mean(mOracle);
  let varM = 0;
  let invPiEx = 0;
  for (let i = 0; i < n; i++) {
    const dm = mOracle[i] - meanM;
    varM += dm * dm;
    invPiEx += sigma2 / piOracle[i];
  }
  varM /= (n - 1);
  invPiEx /= n;
  const Veff = varM + invPiEx;
  return { Veff, sdAt: (nQuery: number) => Math.sqrt(Veff / nQuery) };
}

// -----------------------------------------------------------------------------
// Utility: variance re-export under a non-shadowing name
// -----------------------------------------------------------------------------

export { varianceOf as variance };

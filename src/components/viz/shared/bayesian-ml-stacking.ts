// =============================================================================
// bayesian-ml-stacking.ts
//
// Shared math/types/palette module for the T5 Bayesian & Probabilistic ML
// `stacking-and-predictive-ensembles` topic. Provides numerical primitives for:
//   - Synthetic DGP for the §1 running example (sin(2πx) + 0.4·𝟙[x>0.5]·cos(6πx))
//   - Closed-form Bayesian linear / polynomial regression under NIG conjugacy
//     (Student-t predictive, conjugate marginal log-evidence)
//   - GP closed-form predictive — re-exported from gaussian-processes.ts
//   - Exact leave-one-out log-predictive density loops for the three closed-form
//     candidates (n refits, each O(n³) for GP and O(p³) for BLR/BPR)
//   - BMA weights via softmax of marginal log-evidences
//   - Stacking weights via interior-point gradient ascent on the simplex
//     (Yao 2018, Definition 3.1 of the topic)
//   - Stacking objective evaluated on a barycentric grid for Viz 3's contour
//
// All numerical algorithms are direct translations of the verified Python
// notebook at notebooks/stacking-and-predictive-ensembles/01_stacking_and_
// predictive_ensembles.ipynb. Numerical agreement is verified against printed
// notebook outputs by src/components/viz/shared/__tests__/verify-bayesian-ml-
// stacking.ts.
// =============================================================================

import {
  mulberry32,
  gaussianSampler,
  identityMatrix,
  addDiagonal,
  matMul,
  matVec,
  transpose,
  dot,
  choleskyFactor,
  solveLowerTriangular,
  solveUpperTriangularT,
  choleskyLogDet,
  kernelSE,
  gpPredict,
  gpPredictDiag,
  type GPPredictResult,
} from './gaussian-processes';

// Re-export the shared PRNG and linear-algebra primitives so callers can import
// from a single module.
export {
  mulberry32,
  gaussianSampler,
  kernelSE,
  gpPredict,
  gpPredictDiag,
};

// -----------------------------------------------------------------------------
// Color palette — mirrors the notebook setup cell.
// Four candidate colors + BMA mixture + stacking mixture + oracle reference.
// -----------------------------------------------------------------------------

export const paletteStacking = {
  truth: '#000000',     // black  — true regression function
  blr: '#1f77b4',       // blue   — Bayesian linear regression candidate
  bpr: '#ff7f0e',       // orange — Bayesian polynomial regression candidate
  gp: '#2ca02c',        // green  — Gaussian process candidate
  bart: '#d62728',      // red    — BART candidate
  bma: '#9467bd',       // purple — BMA mixture predictive
  stacking: '#17becf',  // cyan   — stacking mixture predictive
  oracle: '#7f7f7f',    // gray   — held-out oracle convex combination
  data: '#000000',      // black  — training data points
} as const;

export type StackingColorKey = keyof typeof paletteStacking;

export const candidateNames = ['BLR', 'BPR-4', 'GP', 'BART'] as const;
export type CandidateName = (typeof candidateNames)[number];

// -----------------------------------------------------------------------------
// DGP — sinusoid + localized higher-frequency wiggle.
// Matches notebook's true_function and synth_sinusoid_with_wiggle.
//   y = sin(2πx) + 0.4·𝟙[x > 0.5]·cos(6πx) + N(0, σ²)
//   x ~ Uniform(0, 1)
// -----------------------------------------------------------------------------

export function trueFunction(x: number): number {
  return Math.sin(2 * Math.PI * x) + (x > 0.5 ? 1 : 0) * 0.4 * Math.cos(6 * Math.PI * x);
}

export function synthSinusoidWithWiggle(
  n: number,
  sigma: number,
  rng: () => number,
): { x: Float64Array; y: Float64Array } {
  const gauss = gaussianSampler(rng);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const xi = rng();
    x[i] = xi;
    y[i] = trueFunction(xi) + sigma * gauss();
  }
  return { x, y };
}

// -----------------------------------------------------------------------------
// Polynomial features. Returns design matrix (n x (degree+1)) with columns
// [1, x, x², ..., x^degree]. Matches notebook's poly_features.
// -----------------------------------------------------------------------------

export function polyFeatures(x: ArrayLike<number>, degree: number): number[][] {
  const n = x.length;
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(degree + 1);
    let pow = 1;
    for (let d = 0; d <= degree; d++) {
      row[d] = pow;
      pow *= x[i];
    }
    X[i] = row;
  }
  return X;
}

// -----------------------------------------------------------------------------
// Bayesian linear / polynomial regression — Normal-Inverse-Gamma conjugacy.
//
//   prior:     β | σ² ~ N(μ₀, σ² V₀);   σ² ~ InvGamma(a₀, b₀)
//   posterior: β | σ², y ~ N(μ_n, σ² V_n);   σ² | y ~ InvGamma(a_n, b_n)
//     V_n = (V₀⁻¹ + XᵀX)⁻¹
//     μ_n = V_n (V₀⁻¹ μ₀ + Xᵀy)
//     a_n = a₀ + n/2
//     b_n = b₀ + ½ (μ₀ᵀ V₀⁻¹ μ₀ + yᵀy − μ_nᵀ V_n⁻¹ μ_n)
//
//   predictive at x*:  Student-t(2 a_n, x*ᵀμ_n, (b_n / a_n)·(1 + x*ᵀV_n x*))
//
// We use a diagonal V₀ = v0 · I throughout (the notebook's choice). Default
// hypers match the notebook: μ₀ = 0, v₀ = 10, a₀ = 2, b₀ = 2.
// -----------------------------------------------------------------------------

export interface BLRPriorParams {
  /** Diagonal entry of V₀ = v0·I. Default 10. */
  v0?: number;
  /** Inverse-Gamma shape hyperparameter for σ². Default 2. */
  a0?: number;
  /** Inverse-Gamma rate hyperparameter for σ². Default 2. */
  b0?: number;
}

const DEFAULT_BLR_PRIOR: Required<BLRPriorParams> = { v0: 10, a0: 2, b0: 2 };

/** Posterior parameters for closed-form BLR under NIG conjugacy. */
export interface BLRPosterior {
  /** Posterior mean of β, length p. */
  muN: number[];
  /** Posterior covariance scale matrix V_n (factor of σ²), shape p×p. */
  Vn: number[][];
  /** Posterior shape for σ². */
  aN: number;
  /** Posterior rate for σ². */
  bN: number;
  /** Number of observations, retained for log-evidence calculation. */
  n: number;
  /** Number of features. */
  p: number;
  /** Cached Cholesky factor of V_n⁻¹ for efficient solves. */
  cholVnInv: number[][];
  /** Cached log|V_n| / 2 for log-evidence calculation. */
  halfLogDetVn: number;
}

/**
 * Compute posterior parameters under NIG conjugacy. Allocates new matrices.
 * Cost O(p³) for the Cholesky factorization plus O(n p²) for X^T X.
 */
export function blrFitPosterior(
  X: number[][],
  y: ArrayLike<number>,
  prior: BLRPriorParams = {},
): BLRPosterior {
  const v0 = prior.v0 ?? DEFAULT_BLR_PRIOR.v0;
  const a0 = prior.a0 ?? DEFAULT_BLR_PRIOR.a0;
  const b0 = prior.b0 ?? DEFAULT_BLR_PRIOR.b0;

  const n = y.length;
  const p = X[0].length;

  // Build V₀⁻¹ + XᵀX = (1/v0)·I + XᵀX.
  const XtX_plus_V0inv: number[][] = new Array(p);
  for (let i = 0; i < p; i++) XtX_plus_V0inv[i] = new Array(p).fill(0);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[k][i] * X[k][j];
      XtX_plus_V0inv[i][j] = s + (i === j ? 1 / v0 : 0);
    }
  }

  // Build Xᵀy.
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < p; i++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += X[k][i] * y[k];
    Xty[i] = s;
  }

  // Cholesky V_n⁻¹ = L Lᵀ; solve V_n⁻¹ μ_n = Xty (μ₀ = 0 → no μ₀ contribution).
  const L = choleskyFactor(XtX_plus_V0inv);
  const tmp = solveLowerTriangular(L, Xty);
  const muN = solveUpperTriangularT(L, tmp);

  // a_n = a₀ + n/2, b_n = b₀ + ½(yᵀy − μ_nᵀ V_n⁻¹ μ_n).
  let yty = 0;
  for (let i = 0; i < n; i++) yty += y[i] * y[i];
  const VnInvMu = matVec(XtX_plus_V0inv, muN);
  let mu_VnInv_mu = 0;
  for (let i = 0; i < p; i++) mu_VnInv_mu += muN[i] * VnInvMu[i];

  const aN = a0 + n / 2;
  const bN = b0 + 0.5 * (yty - mu_VnInv_mu);

  // V_n = (V_n⁻¹)⁻¹ — needed for predictive variance. We don't need it explicitly
  // for predictions (we'll solve via Cholesky), but we cache half log|V_n| for
  // marginal-evidence calculation: log|V_n| = -log|V_n⁻¹| = -2 · sum(log diag(L)).
  const halfLogDetVnInv = choleskyLogDet(L); // sum log diag(L)
  const halfLogDetVn = -halfLogDetVnInv;

  return {
    muN,
    Vn: [], // populated lazily by blrPredictPoint if needed
    aN,
    bN,
    n,
    p,
    cholVnInv: L,
    halfLogDetVn,
  };
}

/**
 * Compute Student-t predictive parameters at a single test point x_star.
 *   loc   = x*ᵀ μ_n
 *   scale² = (b_n / a_n) · (1 + x*ᵀ V_n x*)
 *   dof    = 2 a_n
 */
export interface StudentTParams {
  loc: number;
  scale: number; // sqrt of variance scale parameter (NOT std)
  dof: number;
}

export function blrPredictPoint(
  posterior: BLRPosterior,
  xStar: ArrayLike<number>,
): StudentTParams {
  const { muN, aN, bN, cholVnInv } = posterior;
  const p = muN.length;

  let loc = 0;
  for (let i = 0; i < p; i++) loc += xStar[i] * muN[i];

  // x*ᵀ V_n x* = ‖L⁻ᵀ x*‖² — solve L y = x*, then ‖L⁻ᵀ x*‖² = ‖z‖² where z = L⁻¹ x*.
  // Wait: V_n = (LLᵀ)⁻¹ = L⁻ᵀ L⁻¹, so x*ᵀ V_n x* = (L⁻¹ x*)ᵀ (L⁻¹ x*) = ‖L⁻¹ x*‖².
  const xStarArr: number[] = new Array(p);
  for (let i = 0; i < p; i++) xStarArr[i] = xStar[i];
  const z = solveLowerTriangular(cholVnInv, xStarArr);
  let zSq = 0;
  for (let i = 0; i < p; i++) zSq += z[i] * z[i];

  const scale2 = (bN / aN) * (1 + zSq);
  const scale = Math.sqrt(scale2);
  const dof = 2 * aN;
  return { loc, scale, dof };
}

/**
 * Marginal log-evidence log p(y | model) under NIG, used for BMA weights.
 *   log p(y) = (a₀ log b₀ − a_n log b_n) + (log Γ(a_n) − log Γ(a₀))
 *             + ½ (log|V_n| − log|V₀|) − (n/2) log(2π)
 * where log|V₀| = p · log(v0) for diagonal V₀ = v0·I.
 */
export function blrLogMarginalLikelihood(
  posterior: BLRPosterior,
  prior: BLRPriorParams = {},
): number {
  const v0 = prior.v0 ?? DEFAULT_BLR_PRIOR.v0;
  const a0 = prior.a0 ?? DEFAULT_BLR_PRIOR.a0;
  const b0 = prior.b0 ?? DEFAULT_BLR_PRIOR.b0;

  const { aN, bN, n, p, halfLogDetVn } = posterior;
  const halfLogDetV0 = (p / 2) * Math.log(v0);
  const term1 = a0 * Math.log(b0) - aN * Math.log(bN);
  const term2 = lnGamma(aN) - lnGamma(a0);
  const term3 = halfLogDetVn - halfLogDetV0;
  const term4 = -(n / 2) * Math.log(2 * Math.PI);
  return term1 + term2 + term3 + term4;
}

// -----------------------------------------------------------------------------
// Student-t log-pdf, lnGamma, and scalar helpers
// -----------------------------------------------------------------------------

/**
 * Stirling-series approximation to ln Γ(x). Accurate to ~1e-10 for x ≥ 5;
 * uses recursion x! = x · (x-1)! for small arguments.
 * Identical to scipy.special.gammaln within numerical noise for x in [0.5, 200].
 */
export function lnGamma(x: number): number {
  // Recurrence to push x up to ≥ 8.
  let z = x;
  let acc = 0;
  while (z < 8) {
    acc -= Math.log(z);
    z += 1;
  }
  // Stirling: ln Γ(z) ≈ (z − ½) ln z − z + ½ ln(2π) + 1/(12z) − 1/(360 z³) + ...
  const z2 = z * z;
  const z3 = z2 * z;
  const z5 = z3 * z2;
  const z7 = z5 * z2;
  return (
    (z - 0.5) * Math.log(z) -
    z +
    0.5 * Math.log(2 * Math.PI) +
    1 / (12 * z) -
    1 / (360 * z3) +
    1 / (1260 * z5) -
    1 / (1680 * z7) +
    acc
  );
}

/** log pdf of Student-t at y under (loc, scale, dof). */
export function studentTLogPdf(
  y: number,
  loc: number,
  scale: number,
  dof: number,
): number {
  const z = (y - loc) / scale;
  return (
    lnGamma((dof + 1) / 2) -
    lnGamma(dof / 2) -
    0.5 * Math.log(dof * Math.PI) -
    Math.log(scale) -
    ((dof + 1) / 2) * Math.log(1 + (z * z) / dof)
  );
}

/** log pdf of Gaussian at y under (loc, std). */
export function gaussianLogPdf(y: number, loc: number, std: number): number {
  const z = (y - loc) / std;
  return -0.5 * Math.log(2 * Math.PI) - Math.log(std) - 0.5 * z * z;
}

// -----------------------------------------------------------------------------
// GP candidate predictive — wraps gpPredict from gaussian-processes.ts. The GP
// candidate uses fixed length scale and noise level (per notebook §3 closed-form
// candidate; §4 PyMC GP differs by putting hyperpriors on these and is handled
// in the Python precompute pipeline, not here).
// -----------------------------------------------------------------------------

export interface GPCandidateHypers {
  /** Squared-exponential length scale. Default 0.1 (matches notebook). */
  lengthScale?: number;
  /** Output (signal) variance. Default 1.0 (matches notebook). */
  outputVar?: number;
  /** Noise variance. Default 0.04 ≈ 0.2² (matches notebook σ=0.2 default). */
  noiseVar?: number;
}

/**
 * Compute GP closed-form posterior predictive (mean + diagonal variance) at
 * test points, using the SE kernel. Returns Gaussian (loc, std) per test point.
 */
export function gpCandidatePredictive(
  xTrain: ArrayLike<number>,
  yTrain: ArrayLike<number>,
  xTest: ArrayLike<number>,
  hypers: GPCandidateHypers = {},
): { loc: Float64Array; std: Float64Array } {
  const ell = hypers.lengthScale ?? 0.1;
  const sf2 = hypers.outputVar ?? 1.0;
  const sn2 = hypers.noiseVar ?? 0.04;

  const xTr = Array.from(xTrain as ArrayLike<number>);
  const yTr = Array.from(yTrain as ArrayLike<number>);
  const xTe = Array.from(xTest as ArrayLike<number>);

  // SE kernel function for 1D inputs, used directly with gpPredict from gaussian-processes.ts.
  const kernelFn = (X1: number[], X2: number[]): number[][] => {
    const k: number[][] = new Array(X1.length);
    for (let i = 0; i < X1.length; i++) {
      const row = new Array(X2.length);
      for (let j = 0; j < X2.length; j++) {
        const dx = X1[i] - X2[j];
        row[j] = sf2 * Math.exp(-0.5 * (dx * dx) / (ell * ell));
      }
      k[i] = row;
    }
    return k;
  };

  const result = gpPredict(xTr, yTr, xTe, kernelFn, Math.sqrt(sn2));

  const m = xTe.length;
  const loc = new Float64Array(m);
  const std = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    loc[i] = result.mean[i];
    // Predictive includes noise: var_y = var_f + sn2.
    std[i] = Math.sqrt(result.sd[i] * result.sd[i] + sn2);
  }
  return { loc, std };
}

// -----------------------------------------------------------------------------
// Exact LOO refit loops. Each candidate refits n times leaving one observation
// out; predictive density at the held-out point is added to the LOO matrix.
// Cost: O(n · refit_cost). For BLR/BPR with p features it's O(n · p³); for GP
// it's O(n · n³) = O(n⁴), so callers should keep n ≤ ~150 for client-side use.
//
// Returns an (n, K) matrix of log p_k(y_i | y_{-i}) values.
// -----------------------------------------------------------------------------

export function looLogPredictiveBLR(
  X: number[][],
  y: ArrayLike<number>,
  prior: BLRPriorParams = {},
): Float64Array {
  const n = y.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const Xi = removeRow(X, i);
    const yi = removeIndex(y, i);
    const post = blrFitPosterior(Xi, yi, prior);
    const tParams = blrPredictPoint(post, X[i]);
    out[i] = studentTLogPdf(y[i], tParams.loc, tParams.scale, tParams.dof);
  }
  return out;
}

export function looLogPredictiveBPR(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  degree: number,
  prior: BLRPriorParams = {},
): Float64Array {
  const X = polyFeatures(x, degree);
  return looLogPredictiveBLR(X, y, prior);
}

export function looLogPredictiveGP(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  hypers: GPCandidateHypers = {},
): Float64Array {
  const n = y.length;
  const out = new Float64Array(n);
  const xArr = Array.from(x as ArrayLike<number>);
  const yArr = Array.from(y as ArrayLike<number>);
  for (let i = 0; i < n; i++) {
    const xLoo = xArr.slice(0, i).concat(xArr.slice(i + 1));
    const yLoo = yArr.slice(0, i).concat(yArr.slice(i + 1));
    const { loc, std } = gpCandidatePredictive(
      xLoo,
      yLoo,
      [xArr[i]],
      hypers,
    );
    out[i] = gaussianLogPdf(yArr[i], loc[0], std[0]);
  }
  return out;
}

function removeRow<T>(M: T[][], idx: number): T[][] {
  const out = new Array(M.length - 1);
  for (let i = 0, j = 0; i < M.length; i++) {
    if (i === idx) continue;
    out[j++] = M[i];
  }
  return out;
}

function removeIndex(v: ArrayLike<number>, idx: number): number[] {
  const out = new Array(v.length - 1);
  for (let i = 0, j = 0; i < v.length; i++) {
    if (i === idx) continue;
    out[j++] = v[i];
  }
  return out;
}

// -----------------------------------------------------------------------------
// BMA weights from marginal log-evidences — softmax with numerical-stability
// shift. log_evidences[k] = log p(y | M_k); BMA weights ∝ exp(log_evidences[k]).
// -----------------------------------------------------------------------------

export function bmaWeights(logEvidences: ArrayLike<number>): Float64Array {
  const K = logEvidences.length;
  const out = new Float64Array(K);
  let maxLE = -Infinity;
  for (let k = 0; k < K; k++) if (logEvidences[k] > maxLE) maxLE = logEvidences[k];
  let z = 0;
  for (let k = 0; k < K; k++) {
    out[k] = Math.exp(logEvidences[k] - maxLE);
    z += out[k];
  }
  for (let k = 0; k < K; k++) out[k] /= z;
  return out;
}

// -----------------------------------------------------------------------------
// Stacking — weights on the (K-1)-simplex maximizing the LOO log-mixture score.
//
//   maximize_{w ∈ Δ^{K-1}}  Σ_i log Σ_k w_k · exp(L_{i,k})
//
// where L is the (n, K) log-LOO-predictive matrix. The objective is concave
// in w. We optimize via softmax reparameterization: w = softmax(z), z ∈ R^K,
// followed by gradient ascent. This avoids any explicit simplex projection
// and converges in tens of iterations. Final weights are renormalized to
// guarantee Σw_k = 1 to machine precision.
// -----------------------------------------------------------------------------

export interface StackingFitOptions {
  /** Maximum iterations. Default 200. */
  maxIter?: number;
  /** Initial step size. Default 0.5. */
  stepSize?: number;
  /** Tolerance for early stopping. Default 1e-9. */
  tol?: number;
  /** Initial logits z_k. Default zeros (uniform weights). */
  zInit?: ArrayLike<number>;
}

export interface StackingFitResult {
  /** Optimal weights, length K, sum = 1, all ≥ 0. */
  weights: Float64Array;
  /** Final value of the (per-observation-mean) log-mixture score. */
  meanLogScore: number;
  /** Total iterations used. */
  iterations: number;
  /** Converged within tol? */
  converged: boolean;
}

/** Softmax of a length-K vector. */
export function softmax(z: ArrayLike<number>): Float64Array {
  const K = z.length;
  let m = -Infinity;
  for (let k = 0; k < K; k++) if (z[k] > m) m = z[k];
  let s = 0;
  const w = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    w[k] = Math.exp(z[k] - m);
    s += w[k];
  }
  for (let k = 0; k < K; k++) w[k] /= s;
  return w;
}

/**
 * Stacking objective: mean over i of log Σ_k w_k exp(L_{i,k}).
 * L is the row-major (n, K) log-LOO-predictive matrix flattened to length n·K.
 */
export function stackingObjective(
  w: ArrayLike<number>,
  L: ArrayLike<number>,
  n: number,
  K: number,
): number {
  // log Σ_k w_k exp(L_{i,k}) = log Σ_k exp(L_{i,k} + log w_k)
  // For numerical stability, use the log-sum-exp trick.
  let total = 0;
  const logW = new Float64Array(K);
  for (let k = 0; k < K; k++) logW[k] = Math.log(Math.max(w[k], 1e-300));
  for (let i = 0; i < n; i++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) {
      const v = L[i * K + k] + logW[k];
      if (v > m) m = v;
    }
    let s = 0;
    for (let k = 0; k < K; k++) s += Math.exp(L[i * K + k] + logW[k] - m);
    total += m + Math.log(s);
  }
  return total / n;
}

/**
 * Gradient of stackingObjective w.r.t. logits z (with w = softmax(z)).
 *   ∂obj/∂z_k = (1/n) Σ_i (w_k · exp(L_{i,k}) / Σ_j w_j · exp(L_{i,j})) − w_k
 *             = (1/n) Σ_i r_{i,k} − w_k
 * where r_{i,k} = posterior responsibility of model k for observation i under
 * the current mixture. Here we derive ∂obj/∂z_k from the chain rule on softmax.
 */
function stackingGradLogits(
  w: ArrayLike<number>,
  L: ArrayLike<number>,
  n: number,
  K: number,
): Float64Array {
  const logW = new Float64Array(K);
  for (let k = 0; k < K; k++) logW[k] = Math.log(Math.max(w[k], 1e-300));
  const meanResp = new Float64Array(K);
  for (let i = 0; i < n; i++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) {
      const v = L[i * K + k] + logW[k];
      if (v > m) m = v;
    }
    let s = 0;
    const ek = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      ek[k] = Math.exp(L[i * K + k] + logW[k] - m);
      s += ek[k];
    }
    for (let k = 0; k < K; k++) meanResp[k] += ek[k] / s;
  }
  for (let k = 0; k < K; k++) meanResp[k] /= n;

  // Chain rule through softmax: ∂obj/∂z_k = w_k · (∂obj/∂w_k − Σ_j w_j ∂obj/∂w_j)
  // Here ∂obj/∂w_k = (1/n) Σ_i (1/μ_i) · exp(L_{i,k}) where μ_i = Σ_j w_j exp(L_{i,j}).
  // Equivalently ∂obj/∂w_k = meanResp[k] / w_k. Substituting yields
  //   ∂obj/∂z_k = meanResp[k] − w_k · Σ_j meanResp[j] = meanResp[k] − w_k.
  // (Since Σ_j meanResp[j] = 1 by construction.)
  const grad = new Float64Array(K);
  for (let k = 0; k < K; k++) grad[k] = meanResp[k] - w[k];
  return grad;
}

/**
 * Fit stacking weights via gradient ascent on softmax-reparameterized logits.
 * Returns the optimal weights, mean log-mixture score, iterations used, and
 * convergence flag.
 */
export function fitStackingWeights(
  L: ArrayLike<number>,
  n: number,
  K: number,
  options: StackingFitOptions = {},
): StackingFitResult {
  const maxIter = options.maxIter ?? 200;
  const tol = options.tol ?? 1e-9;
  let stepSize = options.stepSize ?? 0.5;

  const z = new Float64Array(K);
  if (options.zInit) for (let k = 0; k < K; k++) z[k] = options.zInit[k];

  let w = softmax(z);
  let prevObj = stackingObjective(w, L, n, K);
  let converged = false;
  let iter = 0;

  for (iter = 1; iter <= maxIter; iter++) {
    const g = stackingGradLogits(w, L, n, K);
    // Take a step along the gradient.
    const zNew = new Float64Array(K);
    for (let k = 0; k < K; k++) zNew[k] = z[k] + stepSize * g[k];
    const wNew = softmax(zNew);
    const objNew = stackingObjective(wNew, L, n, K);
    if (objNew >= prevObj) {
      // Accept; modest step-size growth.
      for (let k = 0; k < K; k++) z[k] = zNew[k];
      w = wNew;
      const delta = objNew - prevObj;
      prevObj = objNew;
      stepSize = Math.min(stepSize * 1.1, 5.0);
      if (delta < tol) {
        converged = true;
        break;
      }
    } else {
      // Reject; halve the step size.
      stepSize *= 0.5;
      if (stepSize < 1e-12) {
        converged = true;
        break;
      }
    }
  }

  // Renormalize to machine precision.
  let s = 0;
  for (let k = 0; k < K; k++) s += w[k];
  for (let k = 0; k < K; k++) w[k] = w[k] / s;

  return {
    weights: w,
    meanLogScore: prevObj,
    iterations: iter,
    converged,
  };
}

/**
 * Held-out log-score of a mixture predictive at evaluation points.
 *   score = (1/m) Σ_i log Σ_k w_k p_k(y*_i | x*_i)
 * `lpEval` is the (m, K) log p_k(y*_i | x*_i) matrix flattened row-major.
 */
export function mixtureLogScore(
  w: ArrayLike<number>,
  lpEval: ArrayLike<number>,
  m: number,
  K: number,
): number {
  return stackingObjective(w, lpEval, m, K);
}

// -----------------------------------------------------------------------------
// Stacking objective on the 2-simplex grid (for Viz 3 — three candidates only).
// Returns a Float64Array of objective values evaluated at barycentric grid
// points (w₁ + w₂ + w₃ = 1, all ≥ 0), parameterized by a triangular grid of
// resolution `res`. Grid uses indices (i, j) with 0 ≤ i+j ≤ res; weights are
// w₁ = (res − i − j)/res, w₂ = i/res, w₃ = j/res. Total points = (res+1)(res+2)/2.
// -----------------------------------------------------------------------------

export interface SimplexGridResult {
  /** (res+1)(res+2)/2 objective values, indexed via gridIndex(i,j,res). */
  values: Float64Array;
  /** (res+1)(res+2)/2 × 2 barycentric coords (w₂, w₃); w₁ = 1 − w₂ − w₃. */
  bary: Float64Array;
  /** Maximum objective value found on the grid. */
  maxValue: number;
  /** Argmax weights as (w₁, w₂, w₃). */
  argmaxWeights: Float64Array;
}

/** Linear index of barycentric grid point (i, j) with 0 ≤ i+j ≤ res. */
export function gridIndex(i: number, j: number, res: number): number {
  // Layout: rows indexed by i = 0..res. Row i has res − i + 1 points (j = 0..res-i).
  // Cumulative count up to row i: (res+1) + res + ... + (res-i+2) = sum_{k=0}^{i-1}(res-k+1).
  let off = 0;
  for (let k = 0; k < i; k++) off += res - k + 1;
  return off + j;
}

export function stackingObjectiveOnSimplex3(
  L: ArrayLike<number>,
  n: number,
  res: number,
): SimplexGridResult {
  const K = 3;
  const total = ((res + 1) * (res + 2)) / 2;
  const values = new Float64Array(total);
  const bary = new Float64Array(total * 2);
  let maxValue = -Infinity;
  let argmaxIdx = 0;

  for (let i = 0; i <= res; i++) {
    for (let j = 0; j <= res - i; j++) {
      const w2 = i / res;
      const w3 = j / res;
      const w1 = 1 - w2 - w3;
      const w = new Float64Array([w1, w2, w3]);
      const obj = stackingObjective(w, L, n, K);
      const idx = gridIndex(i, j, res);
      values[idx] = obj;
      bary[idx * 2] = w2;
      bary[idx * 2 + 1] = w3;
      if (obj > maxValue) {
        maxValue = obj;
        argmaxIdx = idx;
      }
    }
  }

  const w2max = bary[argmaxIdx * 2];
  const w3max = bary[argmaxIdx * 2 + 1];
  return {
    values,
    bary,
    maxValue,
    argmaxWeights: new Float64Array([1 - w2max - w3max, w2max, w3max]),
  };
}

// -----------------------------------------------------------------------------
// K_eff — effective sample size of a weight vector. Defined as 1 / Σ w_k². A
// Yao 2018 diagnostic: K_eff close to K means weights spread evenly (stacking
// uses many candidates); K_eff close to 1 means weights concentrate on one
// (BMA's collapse signature). Range [1, K].
// -----------------------------------------------------------------------------

export function effectiveSampleSize(w: ArrayLike<number>): number {
  let s = 0;
  for (let k = 0; k < w.length; k++) s += w[k] * w[k];
  return 1 / s;
}

// -----------------------------------------------------------------------------
// Pareto-k diagnostic four-band classifier (Vehtari, Gelman & Gabry 2017).
// -----------------------------------------------------------------------------

export type ParetoKBand = 'good' | 'ok' | 'bad' | 'very_bad';

export function paretoKBand(k: number): ParetoKBand {
  if (k < 0.5) return 'good';
  if (k < 0.7) return 'ok';
  if (k < 1.0) return 'bad';
  return 'very_bad';
}

export const PARETO_K_BAND_COLORS: Record<ParetoKBand, string> = {
  good: '#2ca02c',     // green   — k < 0.5  (reliable)
  ok: '#ffbb33',       // yellow  — 0.5 ≤ k < 0.7  (typical)
  bad: '#ff7f0e',      // orange  — 0.7 ≤ k < 1.0  (suspect)
  very_bad: '#d62728', // red     — k ≥ 1.0  (unreliable)
};

// -----------------------------------------------------------------------------
// Convenience: fit BLR + BPR-d + GP candidates and return the LOO log-pred
// matrix in row-major (n, 3) form. Returned array length is n*3.
// -----------------------------------------------------------------------------

export function fitThreeCandidatesLOO(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  bprDegree: number = 4,
  gpHypers: GPCandidateHypers = {},
  blrPrior: BLRPriorParams = {},
): { L: Float64Array; n: number; K: number } {
  const n = y.length;
  const Xlin = polyFeatures(x, 1);
  const Llin = looLogPredictiveBLR(Xlin, y, blrPrior);
  const Lpoly = looLogPredictiveBPR(x, y, bprDegree, blrPrior);
  const Lgp = looLogPredictiveGP(x, y, gpHypers);
  const L = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    L[i * 3 + 0] = Llin[i];
    L[i * 3 + 1] = Lpoly[i];
    L[i * 3 + 2] = Lgp[i];
  }
  return { L, n, K: 3 };
}

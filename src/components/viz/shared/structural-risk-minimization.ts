// =============================================================================
// structural-risk-minimization.ts
//
// Shared math primitives for the structural-risk-minimization topic's viz
// components. Ported from notebooks/structural-risk-minimization/
// 01_structural_risk_minimization.ipynb §§1-12.
//
// In-browser TS for all the §§1-11 viz (closed-form bias-variance MC + ridge
// SVD + Vandermonde Rademacher projection + AIC/BIC/MDL/PAC-Bayes/CV). No
// precomputed JSON: every computation is closed form or fast Monte Carlo and
// runs in the browser under 200ms on the largest sliders.
//
// All exports are pure functions — deterministic outputs for a given seeded
// RNG via mulberry32 (re-exported from generalization-bounds.ts).
//
// Source-of-truth notebook:
//   notebooks/structural-risk-minimization/01_structural_risk_minimization.ipynb
// =============================================================================

import { gaussianFrom, mulberry32 } from './generalization-bounds';

export { gaussianFrom, mulberry32 };

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/** Polynomial regression sample: X ~ U(-1,1), Y = sin(πX) + N(0, σ²). */
export type SinSample = { X: Float64Array; Y: Float64Array };

/** Bias-variance-MSE decomposition arrays indexed by degree k = 0..kMax. */
export type BiasVarianceCurve = {
  biasSq: Float64Array;
  variance: Float64Array;
  mse: Float64Array;
  /** argmin of mse over k. */
  kStar: number;
};

/** Empirical Rademacher MC estimate with monte-carlo standard error. */
export type RademacherEstimate = { mean: number; se: number };

/** Spectral decomposition of a symmetric matrix: A = W diag(values) W^T. */
export type SymEig = { values: Float64Array; vectors: Float64Array };

/** Ridge fit result. */
export type RidgeFit = {
  /** Coefficient vector of length d = kMax + 1. */
  coefs: Float64Array;
  /** Predictions on the training inputs. */
  predictions: Float64Array;
  /** Effective degrees of freedom: tr(S_λ) = Σ s²/(s²+λ). */
  effectiveDof: number;
  /** Training MSE = ||Y - Yhat||² / n. */
  trainingMse: number;
};

// -----------------------------------------------------------------------------
// §1 / §11 Sample generators (X ~ Uniform(-1, 1), Y = sin(πX) + N(0, σ²))
// -----------------------------------------------------------------------------

/** Draw n iid samples from the canonical polynomial-regression toy. */
export function sampleSinTarget(n: number, sigma: number, rng: () => number): SinSample {
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  const gauss = gaussianFrom(rng);
  for (let i = 0; i < n; i++) {
    const x = 2 * rng() - 1;
    X[i] = x;
    Y[i] = Math.sin(Math.PI * x) + sigma * gauss();
  }
  return { X, Y };
}

/** Evaluate the regression target m(x) = sin(πx) on a grid. */
export function targetSin(x: number): number {
  return Math.sin(Math.PI * x);
}

// -----------------------------------------------------------------------------
// §1.5 / §2.5 Polynomial Vandermonde (monomial basis, increasing degree).
// V[i * d + j] = X[i]^j for j = 0..d-1, where d = kMax + 1.
// Flat row-major Float64Array (n*d entries), matching generalization-bounds.ts.
// -----------------------------------------------------------------------------

/** Build the n×d polynomial Vandermonde V[i,j] = X[i]^j. */
export function polynomialVandermonde(X: Float64Array, d: number): Float64Array {
  const n = X.length;
  const V = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    let power = 1;
    const x = X[i];
    for (let j = 0; j < d; j++) {
      V[i * d + j] = power;
      power *= x;
    }
  }
  return V;
}

/** Evaluate polynomial p(x) = Σ_j coef[j] x^j (increasing-order coefficients). */
export function polyvalIncreasing(coefs: Float64Array, x: number): number {
  let y = 0;
  let power = 1;
  for (let j = 0; j < coefs.length; j++) {
    y += coefs[j] * power;
    power *= x;
  }
  return y;
}

// -----------------------------------------------------------------------------
// §1.5 OLS via Householder QR (numerically stable for ill-conditioned Vandermonde).
//
// Solves min_α ||Vα - Y||² for V ∈ ℝ^{n×d}, Y ∈ ℝ^n via:
//   1. V = QR (in place; R upper triangular, Q implicitly stored as reflectors)
//   2. y' = Q^T y
//   3. backsolve R α = y'[:d]
// Returns coefficients α ∈ ℝ^d. Caller may need to handle rank deficiency.
// -----------------------------------------------------------------------------

/** Solve min_α ||Vα - Y||² via Householder QR. */
export function olsViaQR(V: Float64Array, Y: Float64Array, n: number, d: number): Float64Array {
  // Copy V and Y so the originals aren't clobbered.
  const A = new Float64Array(V);
  const y = new Float64Array(Y);
  // Householder reflectors stored implicitly via tau values; we don't need Q explicitly.
  for (let j = 0; j < d; j++) {
    // Compute ||A[j:n, j]||
    let normSq = 0;
    for (let i = j; i < n; i++) {
      const val = A[i * d + j];
      normSq += val * val;
    }
    const norm = Math.sqrt(normSq);
    if (norm === 0) continue; // skip rank-deficient column (caller's responsibility)
    const sign = A[j * d + j] >= 0 ? 1 : -1;
    const alpha = -sign * norm;
    // Build reflector v = A[j:n, j] - alpha * e_1, scaled so v[0] = 1 implicit
    const v = new Float64Array(n - j);
    v[0] = A[j * d + j] - alpha;
    for (let i = j + 1; i < n; i++) v[i - j] = A[i * d + j];
    let vNormSq = 0;
    for (let i = 0; i < v.length; i++) vNormSq += v[i] * v[i];
    if (vNormSq === 0) continue;
    const beta = 2 / vNormSq;
    // Apply H = I - beta v v^T to A[j:n, j:d]
    A[j * d + j] = alpha;
    for (let i = j + 1; i < n; i++) A[i * d + j] = 0;
    for (let k = j + 1; k < d; k++) {
      let dot = 0;
      for (let i = j; i < n; i++) dot += v[i - j] * A[i * d + k];
      const scale = beta * dot;
      for (let i = j; i < n; i++) A[i * d + k] -= scale * v[i - j];
    }
    // Apply H to y[j:n]
    let dot = 0;
    for (let i = j; i < n; i++) dot += v[i - j] * y[i];
    const scale = beta * dot;
    for (let i = j; i < n; i++) y[i] -= scale * v[i - j];
  }
  // Backsolve R α = y[:d]. R is the upper triangle of A.
  const alpha = new Float64Array(d);
  for (let i = d - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < d; k++) s -= A[i * d + k] * alpha[k];
    const rii = A[i * d + i];
    if (Math.abs(rii) < 1e-14) {
      alpha[i] = 0; // rank-deficient — pseudoinverse-style zero
    } else {
      alpha[i] = s / rii;
    }
  }
  return alpha;
}

/**
 * SVD-pseudoinverse OLS: α = V⁺ Y, with singular values below
 * `max(n,d) * eps * s_max` truncated to zero (matches numpy.linalg.lstsq's
 * default rcond=None convention). Numerically stable for high-degree
 * Vandermonde where Householder QR alone loses precision.
 */
export function olsViaSVD(V: Float64Array, Y: Float64Array, n: number, d: number): Float64Array {
  const G = gramMatrix(V, n, d);
  const { values: eigs, vectors: W } = symEigJacobi(G, d);
  // s_j = sqrt(eigs_j); truncate small s_j via rcond rule on s².
  let maxEig = 0;
  for (let j = 0; j < d; j++) if (eigs[j] > maxEig) maxEig = eigs[j];
  const rcondSq = Math.max(n, d) * 2.220446049250313e-16; // numpy lstsq default
  const eigTol = rcondSq * rcondSq * maxEig;
  // β = W^T V^T Y
  const VtY = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += V[i * d + j] * Y[i];
    VtY[j] = s;
  }
  const WtVtY = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < d; i++) s += W[i * d + j] * VtY[i];
    WtVtY[j] = s;
  }
  // γ_j = WtVtY_j / eigs_j for eigs_j > eigTol, else 0.
  const gamma = new Float64Array(d);
  for (let j = 0; j < d; j++) gamma[j] = eigs[j] > eigTol ? WtVtY[j] / eigs[j] : 0;
  // α = W γ
  const alpha = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += W[i * d + j] * gamma[j];
    alpha[i] = s;
  }
  return alpha;
}

/** Fit polynomial of degree k to (X, Y) and return coefficients in increasing order. */
export function polyfitDegree(X: Float64Array, Y: Float64Array, k: number): Float64Array {
  const d = k + 1;
  const V = polynomialVandermonde(X, d);
  return olsViaSVD(V, Y, X.length, d);
}

// -----------------------------------------------------------------------------
// Symmetric eigendecomposition via cyclic Jacobi rotations.
//
// For small symmetric d×d matrices (d ≤ ~32) this is O(d³ · iter), fast for our
// d ≤ 16. Used for ridge effective DoF, ridge spectral solve, and the Rademacher
// projection on the polynomial unit ball. Result: A = W diag(values) W^T,
// with eigenvectors in columns of W (flat row-major n×n).
// -----------------------------------------------------------------------------

/** Eigendecompose a symmetric d×d matrix via cyclic Jacobi. */
export function symEigJacobi(A: Float64Array, d: number, tol: number = 1e-12): SymEig {
  const D = new Float64Array(A);
  const W = new Float64Array(d * d);
  for (let i = 0; i < d; i++) W[i * d + i] = 1;
  const MAX_ITER = 100;
  for (let iter = 0; iter < MAX_ITER; iter++) {
    let off = 0;
    for (let p = 0; p < d - 1; p++) {
      for (let q = p + 1; q < d; q++) off += D[p * d + q] * D[p * d + q];
    }
    if (off < tol * tol) break;
    for (let p = 0; p < d - 1; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = D[p * d + q];
        if (Math.abs(apq) < 1e-15) continue;
        const app = D[p * d + p];
        const aqq = D[q * d + q];
        const theta = (aqq - app) / (2 * apq);
        let t: number;
        if (theta >= 0) t = 1 / (theta + Math.sqrt(1 + theta * theta));
        else t = 1 / (theta - Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        // Update D
        D[p * d + p] = app - t * apq;
        D[q * d + q] = aqq + t * apq;
        D[p * d + q] = 0;
        D[q * d + p] = 0;
        for (let r = 0; r < d; r++) {
          if (r !== p && r !== q) {
            const drp = D[r * d + p];
            const drq = D[r * d + q];
            D[r * d + p] = c * drp - s * drq;
            D[p * d + r] = D[r * d + p];
            D[r * d + q] = s * drp + c * drq;
            D[q * d + r] = D[r * d + q];
          }
        }
        // Update W
        for (let r = 0; r < d; r++) {
          const wrp = W[r * d + p];
          const wrq = W[r * d + q];
          W[r * d + p] = c * wrp - s * wrq;
          W[r * d + q] = s * wrp + c * wrq;
        }
      }
    }
  }
  const values = new Float64Array(d);
  for (let i = 0; i < d; i++) values[i] = D[i * d + i];
  return { values, vectors: W };
}

/** Compute the d×d Gram matrix G = V^T V (flat row-major). */
export function gramMatrix(V: Float64Array, n: number, d: number): Float64Array {
  const G = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let s = 0;
      // Hoist row-offset out of the inner loop — saves a multiply per iter.
      for (let k = 0, offset = 0; k < n; k++, offset += d) {
        s += V[offset + i] * V[offset + j];
      }
      G[i * d + j] = s;
      G[j * d + i] = s;
    }
  }
  return G;
}

// -----------------------------------------------------------------------------
// §6 / §7 Ridge regression via spectral decomposition of V^T V.
//
// α_λ = (V^T V + λ I)^{-1} V^T Y = W (D + λI)^{-1} W^T V^T Y
// Effective DoF tr(S_λ) = Σ d_j / (d_j + λ)  where d_j = s_j² (eigenvalues of V^T V).
// -----------------------------------------------------------------------------

/** Ridge regression at penalty λ on Vandermonde features. */
export function ridgeFit(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
  lambda: number,
): RidgeFit {
  const n = X.length;
  const d = kMax + 1;
  const V = polynomialVandermonde(X, d);
  const G = gramMatrix(V, n, d);
  const { values: eigs, vectors: W } = symEigJacobi(G, d);
  // β = W^T V^T Y (d-vector)
  const VtY = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += V[i * d + j] * Y[i];
    VtY[j] = s;
  }
  const WtVtY = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < d; i++) s += W[i * d + j] * VtY[i];
    WtVtY[j] = s;
  }
  // γ_j = WtVtY_j / (eigs_j + λ)
  const gamma = new Float64Array(d);
  let effDof = 0;
  for (let j = 0; j < d; j++) {
    const ej = Math.max(eigs[j], 0); // numerical floor (should already be ≥ 0)
    gamma[j] = WtVtY[j] / (ej + lambda);
    effDof += ej / (ej + lambda);
  }
  // α = W γ
  const coefs = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += W[i * d + j] * gamma[j];
    coefs[i] = s;
  }
  // Predictions: V α
  const predictions = new Float64Array(n);
  let rss = 0;
  for (let i = 0; i < n; i++) {
    let yhat = 0;
    for (let j = 0; j < d; j++) yhat += V[i * d + j] * coefs[j];
    predictions[i] = yhat;
    const r = Y[i] - yhat;
    rss += r * r;
  }
  return { coefs, predictions, effectiveDof: effDof, trainingMse: rss / n };
}

/** Effective DoF tr(S_λ) given the singular-value-squared spectrum. */
export function ridgeEffectiveDof(eigsOfVtV: Float64Array, lambda: number): number {
  let s = 0;
  for (let j = 0; j < eigsOfVtV.length; j++) {
    const ej = Math.max(eigsOfVtV[j], 0);
    s += ej / (ej + lambda);
  }
  return s;
}

// -----------------------------------------------------------------------------
// §7.5 Lasso via coordinate descent (covariance form).
//
// Minimizes (1/(2n)) ||Y - Vα||² + λ ||α||₁ in the sklearn / glmnet convention.
// Standardization: we do NOT standardize features here — caller is responsible.
// Cyclic CD with soft-thresholding; max 1000 sweeps with tol on dual gap.
// -----------------------------------------------------------------------------

function softThreshold(z: number, gamma: number): number {
  if (z > gamma) return z - gamma;
  if (z < -gamma) return z + gamma;
  return 0;
}

/** Lasso via coordinate descent. Returns coefficient vector of length d. */
export function lassoCD(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
  lambda: number,
  maxIter: number = 1000,
  tol: number = 1e-7,
): Float64Array {
  const n = X.length;
  const d = kMax + 1;
  const V = polynomialVandermonde(X, d);
  // Precompute column norms ||v_j||² for the CD update step size.
  const colNormSq = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += V[i * d + j] * V[i * d + j];
    colNormSq[j] = s;
  }
  const alpha = new Float64Array(d);
  const residual = new Float64Array(n);
  for (let i = 0; i < n; i++) residual[i] = Y[i];
  for (let it = 0; it < maxIter; it++) {
    let maxChange = 0;
    for (let j = 0; j < d; j++) {
      if (colNormSq[j] === 0) continue;
      // ρ_j = v_j^T (residual + v_j α_j)
      let rho = 0;
      for (let i = 0; i < n; i++) rho += V[i * d + j] * (residual[i] + V[i * d + j] * alpha[j]);
      const aOld = alpha[j];
      // Update: α_j = soft_threshold(ρ_j, n*λ) / ||v_j||²
      const aNew = softThreshold(rho, n * lambda) / colNormSq[j];
      const dAlpha = aNew - aOld;
      if (dAlpha !== 0) {
        alpha[j] = aNew;
        for (let i = 0; i < n; i++) residual[i] -= V[i * d + j] * dAlpha;
        if (Math.abs(dAlpha) > maxChange) maxChange = Math.abs(dAlpha);
      }
    }
    if (maxChange < tol) break;
  }
  return alpha;
}

/** Number of nonzero coefficients in α (lasso effective DoF, Zou-Hastie-Tibshirani 2007). */
export function nonzeroCount(alpha: Float64Array, tol: number = 1e-8): number {
  let n = 0;
  for (let i = 0; i < alpha.length; i++) if (Math.abs(alpha[i]) > tol) n++;
  return n;
}

// -----------------------------------------------------------------------------
// §1.5 Bias-variance Monte Carlo for the polynomial-regression toy.
//
// For each k = 0..kMax, fit a polynomial of degree k to B independent samples,
// average across replicates to estimate pointwise bias², variance, MSE on a
// fixed test grid. Returns arrays indexed by k.
// -----------------------------------------------------------------------------

/** Compute pointwise bias²/variance/MSE for each k ≤ kMax via Monte Carlo. */
export function biasVarianceMC(
  n: number,
  sigma: number,
  kMax: number,
  B: number,
  rng: () => number,
  testGridSize: number = 200,
): BiasVarianceCurve {
  const D = kMax + 1;
  const xTest = new Float64Array(testGridSize);
  const truth = new Float64Array(testGridSize);
  for (let g = 0; g < testGridSize; g++) {
    const x = -1 + (2 * g) / (testGridSize - 1);
    xTest[g] = x;
    truth[g] = targetSin(x);
  }
  // Build the test-grid Vandermonde at full degree once for the whole MC; for
  // each k we use only its leading (k+1) columns.
  const V_test = polynomialVandermonde(xTest, D);
  // predSum[k*testGridSize+g] and predSqSum accumulate over replicates B.
  const predSum = new Float64Array(D * testGridSize);
  const predSqSum = new Float64Array(D * testGridSize);
  // Scratch buffers reused across (b, k) iterations.
  const VtY = new Float64Array(D);
  const beta = new Float64Array(D);
  const alpha = new Float64Array(D);
  const rcondSq = Math.max(n, D) * 2.220446049250313e-16; // numpy lstsq default
  for (let b = 0; b < B; b++) {
    const { X, Y } = sampleSinTarget(n, sigma, rng);
    // Build V_full (n × D) and G_full = V^T V (D × D) once per replicate; the
    // OLS at each degree k reads the leading (k+1)×(k+1) submatrix of G_full.
    const V_full = polynomialVandermonde(X, D);
    const G_full = gramMatrix(V_full, n, D);
    for (let k = 0; k <= kMax; k++) {
      const d = k + 1;
      // Extract leading d×d submatrix G[:d, :d] into a contiguous buffer.
      const G_sub = new Float64Array(d * d);
      for (let i = 0; i < d; i++) {
        const rowSubOffset = i * d;
        const rowFullOffset = i * D;
        for (let j = 0; j < d; j++) G_sub[rowSubOffset + j] = G_full[rowFullOffset + j];
      }
      const { values: eigs, vectors: W } = symEigJacobi(G_sub, d);
      // V^T Y on the d leading columns (zero scratch first).
      for (let j = 0; j < d; j++) VtY[j] = 0;
      for (let i = 0, rowOffset = 0; i < n; i++, rowOffset += D) {
        const yi = Y[i];
        for (let j = 0; j < d; j++) VtY[j] += V_full[rowOffset + j] * yi;
      }
      // β = W^T V^T Y
      for (let j = 0; j < d; j++) {
        let s = 0;
        for (let i = 0; i < d; i++) s += W[i * d + j] * VtY[i];
        beta[j] = s;
      }
      // rcond truncation on eigs (max-eigenvalue rule).
      let maxEig = 0;
      for (let j = 0; j < d; j++) if (eigs[j] > maxEig) maxEig = eigs[j];
      const eigTol = rcondSq * rcondSq * maxEig;
      // α = W (β / eigs) with truncated inverse.
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) {
          const ej = eigs[j];
          if (ej > eigTol) s += W[i * d + j] * (beta[j] / ej);
        }
        alpha[i] = s;
      }
      // Predict on test grid via V_test (leading d columns) and accumulate.
      const kBase = k * testGridSize;
      for (let g = 0; g < testGridSize; g++) {
        const rowOffset = g * D;
        let yhat = 0;
        for (let j = 0; j < d; j++) yhat += V_test[rowOffset + j] * alpha[j];
        predSum[kBase + g] += yhat;
        predSqSum[kBase + g] += yhat * yhat;
      }
    }
  }
  const biasSq = new Float64Array(kMax + 1);
  const variance = new Float64Array(kMax + 1);
  const mse = new Float64Array(kMax + 1);
  for (let k = 0; k <= kMax; k++) {
    let bSum = 0;
    let vSum = 0;
    for (let g = 0; g < testGridSize; g++) {
      const meanPred = predSum[k * testGridSize + g] / B;
      const meanPredSq = predSqSum[k * testGridSize + g] / B;
      const varHere = meanPredSq - meanPred * meanPred;
      const biasHere = meanPred - truth[g];
      bSum += biasHere * biasHere;
      vSum += Math.max(varHere, 0);
    }
    biasSq[k] = bSum / testGridSize;
    variance[k] = vSum / testGridSize;
    mse[k] = biasSq[k] + variance[k];
  }
  let kStar = 0;
  let best = mse[0];
  for (let k = 1; k <= kMax; k++) {
    if (mse[k] < best) {
      best = mse[k];
      kStar = k;
    }
  }
  return { biasSq, variance, mse, kStar };
}

// -----------------------------------------------------------------------------
// §1.5 Training MSE — needed by every SRM rule.
// -----------------------------------------------------------------------------

/** Training MSE for a polynomial fit of degree k. */
export function trainingMse(X: Float64Array, Y: Float64Array, k: number): number {
  const coefs = polyfitDegree(X, Y, k);
  let rss = 0;
  for (let i = 0; i < X.length; i++) {
    const r = Y[i] - polyvalIncreasing(coefs, X[i]);
    rss += r * r;
  }
  return rss / X.length;
}

/** Training-MSE vector at a fixed sample, for k = 0..kMax. */
export function trainingMseByDegree(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
): Float64Array {
  const out = new Float64Array(kMax + 1);
  for (let k = 0; k <= kMax; k++) out[k] = trainingMse(X, Y, k);
  return out;
}

// -----------------------------------------------------------------------------
// §3 / §4 Vapnik SRM penalty (Definition 2).
//
//   pen_V(d, n, k, δ) = C · sqrt( (d log(2n/d) + 2 log k + log(π²/(6δ))) / n )
//
// The universal constant C is exposed as a tuning knob — the literal FTSL
// constant from chaining bounds is ~2–8; brief §4.5 simplifies to C = 1 to
// keep the U-curve interior on n = 50 visible.
// -----------------------------------------------------------------------------

/** Vapnik per-class SRM penalty (k ≥ 1). For k = 0 we treat 2 log k = 0. */
export function vapnikPenalty(
  d: number,
  n: number,
  k: number,
  delta: number,
  C: number = 1,
): number {
  const capacityTerm = d * Math.log((2 * n) / Math.max(d, 1));
  const klog = k <= 1 ? 0 : 2 * Math.log(k);
  const confTerm = Math.log((Math.PI * Math.PI) / (6 * delta));
  return C * Math.sqrt((capacityTerm + klog + confTerm) / n);
}

// -----------------------------------------------------------------------------
// §5 Bartlett–Mendelson (empirical Rademacher) SRM penalty (Definition 3).
//
//   pen_R(H_k, S, δ) = 2 Rhat(F_k) + 3 sqrt( log(2π² k² / (6δ)) / (2n) )
//
// For the polynomial L²(P_n)-unit-ball class H_k° = {h ∈ H_k : ||h||_n ≤ 1},
// the closed form Rademacher complexity is:
//
//   Rhat = E_σ[ ||P_V σ|| / sqrt(n) ]
//
// where P_V is the orthogonal projection onto the column space of the
// degree-k Vandermonde V. We estimate by Monte Carlo over B Rademacher draws.
// -----------------------------------------------------------------------------

/**
 * Empirical Rademacher complexity for the polynomial unit-ball class of degree k.
 * Returns mean and Monte Carlo standard error across B draws.
 */
export function polynomialUnitBallRademacher(
  X: Float64Array,
  k: number,
  B: number,
  rng: () => number,
): RademacherEstimate {
  const n = X.length;
  const d = k + 1;
  const V = polynomialVandermonde(X, d);
  const G = gramMatrix(V, n, d);
  const { values: eigs, vectors: W } = symEigJacobi(G, d);
  // Drop near-zero eigenvalues (rank-deficient direction): they correspond
  // to directions outside col(V) and don't contribute to the projection.
  // Use max(eigs) for the tolerance — Jacobi doesn't guarantee sorted order.
  let maxEig = 0;
  for (let j = 0; j < d; j++) if (eigs[j] > maxEig) maxEig = eigs[j];
  const tol = 1e-12 * (maxEig || 1);
  let sumNorm = 0;
  let sumNormSq = 0;
  // Hoist Vtsig allocation outside the MC loop — re-zero each draw.
  const Vtsig = new Float64Array(d);
  for (let b = 0; b < B; b++) {
    // Draw σ ∈ {±1}^n
    // Then compute V^T σ (d-vector), expand in W basis, divide by sqrt(eigs[j]),
    // accumulate squared norm. ||P_V σ||² = Σ_j (W_j^T V^T σ)² / eigs[j].
    Vtsig.fill(0);
    for (let i = 0, offset = 0; i < n; i++, offset += d) {
      const sigma = rng() < 0.5 ? -1 : 1;
      for (let j = 0; j < d; j++) Vtsig[j] += V[offset + j] * sigma;
    }
    let normSq = 0;
    for (let j = 0; j < d; j++) {
      const ej = eigs[j];
      if (ej <= tol) continue;
      let dot = 0;
      for (let i = 0; i < d; i++) dot += W[i * d + j] * Vtsig[i];
      normSq += (dot * dot) / ej;
    }
    const norm = Math.sqrt(normSq);
    sumNorm += norm / Math.sqrt(n);
    sumNormSq += (norm * norm) / n;
  }
  const mean = sumNorm / B;
  const meanSq = sumNormSq / B;
  const variance = Math.max(meanSq - mean * mean, 0);
  return { mean, se: Math.sqrt(variance / B) };
}

/** Bartlett–Mendelson per-class SRM penalty given the empirical Rademacher value. */
export function bartlettMendelsonPenalty(
  rademacher: number,
  n: number,
  k: number,
  delta: number,
): number {
  const deltaK = (6 * delta) / (Math.PI * Math.PI * Math.max(k, 1) * Math.max(k, 1));
  const conf = 3 * Math.sqrt(Math.log(2 / deltaK) / (2 * n));
  return 2 * rademacher + conf;
}

// -----------------------------------------------------------------------------
// §8 Information criteria — closed-form penalties.
//
// All three are per-class penalties on the same training-error-plus-penalty
// SRM template. AIC and BIC use plug-in σ̂²; MDL is BIC/2 asymptotically.
// -----------------------------------------------------------------------------

/** AIC penalty: 2 σ² d / n. */
export function aicPenalty(d: number, sigmaSq: number, n: number): number {
  return (2 * sigmaSq * d) / n;
}

/** BIC penalty: σ² d log n / n. */
export function bicPenalty(d: number, sigmaSq: number, n: number): number {
  return (sigmaSq * d * Math.log(n)) / n;
}

/** MDL penalty: (σ² d log n) / (2n) (≈ BIC / 2 asymptotically). */
export function mdlPenalty(d: number, sigmaSq: number, n: number): number {
  return (sigmaSq * d * Math.log(n)) / (2 * n);
}

/**
 * Plug-in noise variance estimate σ̂² = RSS / (n - d_max) from the OLS fit
 * at the largest degree under consideration. Used by AIC / BIC / MDL.
 */
export function plugInSigmaSq(X: Float64Array, Y: Float64Array, kMax: number): number {
  const n = X.length;
  const d = kMax + 1;
  const coefs = polyfitDegree(X, Y, kMax);
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = Y[i] - polyvalIncreasing(coefs, X[i]);
    rss += r * r;
  }
  return rss / Math.max(n - d, 1);
}

// -----------------------------------------------------------------------------
// §9 PAC-Bayes (Catoni-McAllester Gaussian-posterior form).
//
//   KL(Q || P) = ½ [ d σ_Q² / σ_P² + ||μ_Q||²/σ_P² - d + d log(σ_P²/σ_Q²) ]
//
// The §9.5 demo: prior P = N(0, σ_P² I_d), posterior Q = N(α̂_λ, τ² I_d).
// Posterior-averaged training MSE = ||Y - Vα̂||² / n + τ² tr(V^T V) / n.
// PAC-Bayes total = posterior-averaged MSE + sqrt((KL + log(2√n/δ)) / (2n)).
// -----------------------------------------------------------------------------

/** KL divergence between two isotropic Gaussians (eq 9.2 in brief). */
export function gaussianKLIsotropic(
  muNormSq: number,
  sigmaQ2: number,
  sigmaP2: number,
  d: number,
): number {
  return 0.5 * ((d * sigmaQ2 + muNormSq) / sigmaP2 - d + d * Math.log(sigmaP2 / sigmaQ2));
}

/** McAllester PAC-Bayes upper-bound slack: sqrt((KL + log(2√n/δ)) / (2n)). */
export function pacBayesSlack(kl: number, n: number, delta: number): number {
  return Math.sqrt((kl + Math.log((2 * Math.sqrt(n)) / delta)) / (2 * n));
}

/**
 * Posterior-averaged training MSE for Gaussian posterior centered at ridge fit:
 *   E_Q[ ||Y - Vα||² / n ] = ||Y - Vα̂_λ||² / n + τ² tr(V^T V) / n
 * given the Vandermonde trace s_sum = tr(V^T V) = Σ_i ||v_i||².
 */
export function posteriorAveragedMse(
  ridgeFitTrainingMse: number,
  tauSq: number,
  vTraceSum: number,
  n: number,
): number {
  return ridgeFitTrainingMse + (tauSq * vTraceSum) / n;
}

/** tr(V^T V) — sum of squared L²-norms of the feature columns. */
export function vandermondeTrace(X: Float64Array, kMax: number): number {
  const d = kMax + 1;
  const V = polynomialVandermonde(X, d);
  let s = 0;
  for (let j = 0; j < d; j++) {
    let cs = 0;
    for (let i = 0; i < X.length; i++) cs += V[i * d + j] * V[i * d + j];
    s += cs;
  }
  return s;
}

// -----------------------------------------------------------------------------
// §10 K-fold cross-validation.
//
// Partition n indices into K folds via Fisher-Yates shuffle. For each fold,
// fit on the union of the other K-1 folds and evaluate MSE on the held-out
// fold. Average the K fold-MSEs to get the cross-validated score for one
// fold partition. Brief §10.5: 100 fold rerolls per (n, k).
// -----------------------------------------------------------------------------

/** Fisher-Yates shuffle of 0..n-1 (mutates output). */
export function fisherYates(n: number, rng: () => number): Int32Array {
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx;
}

/** Cross-validated MSE for degree k via K-fold (one fold partition). */
export function kFoldCVMse(
  X: Float64Array,
  Y: Float64Array,
  k: number,
  K: number,
  rng: () => number,
): number {
  const n = X.length;
  const idx = fisherYates(n, rng);
  let totalSqErr = 0;
  let totalPts = 0;
  // Fold j is { idx[ floor(j n / K) .. floor((j+1) n / K) - 1 ] }.
  for (let j = 0; j < K; j++) {
    const start = Math.floor((j * n) / K);
    const end = Math.floor(((j + 1) * n) / K);
    const testSize = end - start;
    const trainSize = n - testSize;
    if (trainSize <= k + 1) continue; // skip folds with too few training points
    const Xtr = new Float64Array(trainSize);
    const Ytr = new Float64Array(trainSize);
    let trIdx = 0;
    for (let p = 0; p < n; p++) {
      if (p >= start && p < end) continue;
      Xtr[trIdx] = X[idx[p]];
      Ytr[trIdx] = Y[idx[p]];
      trIdx++;
    }
    const coefs = polyfitDegree(Xtr, Ytr, k);
    for (let p = start; p < end; p++) {
      const xi = X[idx[p]];
      const yhat = polyvalIncreasing(coefs, xi);
      const r = Y[idx[p]] - yhat;
      totalSqErr += r * r;
    }
    totalPts += testSize;
  }
  return totalSqErr / Math.max(totalPts, 1);
}

/** CV score curve for k = 0..kMax (one fold partition). */
export function kFoldCVCurve(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
  K: number,
  rng: () => number,
): Float64Array {
  const out = new Float64Array(kMax + 1);
  for (let k = 0; k <= kMax; k++) out[k] = kFoldCVMse(X, Y, k, K, rng);
  return out;
}

/** Distribution of CV picks across B fold-partition rerolls. */
export function cvPickHistogram(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
  K: number,
  rerolls: number,
  rng: () => number,
): { picks: Int32Array; mode: number; mean: number; std: number } {
  const counts = new Int32Array(kMax + 1);
  const picks = new Int32Array(rerolls);
  for (let b = 0; b < rerolls; b++) {
    const curve = kFoldCVCurve(X, Y, kMax, K, rng);
    let argmin = 0;
    let best = curve[0];
    for (let k = 1; k <= kMax; k++) {
      if (curve[k] < best) {
        best = curve[k];
        argmin = k;
      }
    }
    picks[b] = argmin;
    counts[argmin]++;
  }
  let mode = 0;
  let modeCount = counts[0];
  for (let k = 1; k <= kMax; k++) {
    if (counts[k] > modeCount) {
      modeCount = counts[k];
      mode = k;
    }
  }
  let sum = 0;
  let sumSq = 0;
  for (let b = 0; b < rerolls; b++) {
    sum += picks[b];
    sumSq += picks[b] * picks[b];
  }
  const mean = sum / rerolls;
  const variance = sumSq / rerolls - mean * mean;
  return { picks, mode, mean, std: Math.sqrt(Math.max(variance, 0)) };
}

// -----------------------------------------------------------------------------
// SRM rule — argmin of training error + penalty across k.
// -----------------------------------------------------------------------------

/** Argmin of (trainErr + penalty) over k ∈ [0, kMax]. Ties broken by smaller k. */
export function srmPickFromArrays(trainErr: Float64Array, penalty: Float64Array): number {
  let argmin = 0;
  let best = trainErr[0] + penalty[0];
  for (let k = 1; k < trainErr.length; k++) {
    const total = trainErr[k] + penalty[k];
    if (total < best) {
      best = total;
      argmin = k;
    }
  }
  return argmin;
}

/** Build the Vapnik-penalty array for k = 0..kMax. */
export function vapnikPenaltyArray(
  kMax: number,
  n: number,
  delta: number,
  C: number = 1,
): Float64Array {
  const out = new Float64Array(kMax + 1);
  for (let k = 0; k <= kMax; k++) out[k] = vapnikPenalty(k + 1, n, k, delta, C);
  return out;
}

/** Pick under Vapnik SRM at fixed sample. */
export function vapnikPick(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
  delta: number,
  C: number = 1,
): number {
  const train = trainingMseByDegree(X, Y, kMax);
  const pen = vapnikPenaltyArray(kMax, X.length, delta, C);
  return srmPickFromArrays(train, pen);
}

/** Pick under Bartlett–Mendelson Rademacher SRM at fixed sample. */
export function rademacherPick(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
  delta: number,
  B: number,
  rng: () => number,
): number {
  const n = X.length;
  const train = trainingMseByDegree(X, Y, kMax);
  const pen = new Float64Array(kMax + 1);
  for (let k = 0; k <= kMax; k++) {
    const rad = polynomialUnitBallRademacher(X, k, B, rng).mean;
    pen[k] = bartlettMendelsonPenalty(rad, n, k, delta);
  }
  return srmPickFromArrays(train, pen);
}

/** Pick under AIC at fixed sample. */
export function aicPick(X: Float64Array, Y: Float64Array, kMax: number): number {
  const n = X.length;
  const sigmaSq = plugInSigmaSq(X, Y, kMax);
  const train = trainingMseByDegree(X, Y, kMax);
  const pen = new Float64Array(kMax + 1);
  for (let k = 0; k <= kMax; k++) pen[k] = aicPenalty(k + 1, sigmaSq, n);
  return srmPickFromArrays(train, pen);
}

/** Pick under BIC at fixed sample. */
export function bicPick(X: Float64Array, Y: Float64Array, kMax: number): number {
  const n = X.length;
  const sigmaSq = plugInSigmaSq(X, Y, kMax);
  const train = trainingMseByDegree(X, Y, kMax);
  const pen = new Float64Array(kMax + 1);
  for (let k = 0; k <= kMax; k++) pen[k] = bicPenalty(k + 1, sigmaSq, n);
  return srmPickFromArrays(train, pen);
}

/** Pick under MDL at fixed sample. */
export function mdlPick(X: Float64Array, Y: Float64Array, kMax: number): number {
  const n = X.length;
  const sigmaSq = plugInSigmaSq(X, Y, kMax);
  const train = trainingMseByDegree(X, Y, kMax);
  const pen = new Float64Array(kMax + 1);
  for (let k = 0; k <= kMax; k++) pen[k] = mdlPenalty(k + 1, sigmaSq, n);
  return srmPickFromArrays(train, pen);
}

/**
 * Soft-SRM ridge sweep: for each λ in `lambdas`, compute the ridge fit and
 * the soft-SRM total = training MSE + sqrt((effDoF + log(1/δ))/n).
 * Returns parallel arrays + argmin and its mapped integer degree.
 */
export function ridgeSoftSRMPath(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
  lambdas: Float64Array,
  delta: number,
): {
  trainingMse: Float64Array;
  effectiveDof: Float64Array;
  total: Float64Array;
  argminLambda: number;
  hatLambda: number;
  pickedEffectiveDof: number;
} {
  const n = X.length;
  const M = lambdas.length;
  const trainingMse = new Float64Array(M);
  const effectiveDof = new Float64Array(M);
  const total = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    const fit = ridgeFit(X, Y, kMax, lambdas[i]);
    trainingMse[i] = fit.trainingMse;
    effectiveDof[i] = fit.effectiveDof;
    total[i] = fit.trainingMse + Math.sqrt((fit.effectiveDof + Math.log(1 / delta)) / n);
  }
  let argmin = 0;
  let best = total[0];
  for (let i = 1; i < M; i++) {
    if (total[i] < best) {
      best = total[i];
      argmin = i;
    }
  }
  return {
    trainingMse,
    effectiveDof,
    total,
    argminLambda: argmin,
    hatLambda: lambdas[argmin],
    pickedEffectiveDof: effectiveDof[argmin],
  };
}

/**
 * PAC-Bayes Catoni pick at fixed sample: sweep ridge λ on a log grid; for each
 * λ build the Gaussian posterior Q = N(α̂_λ, τ² I_d), prior P = N(0, σ_P² I_d),
 * and minimize posterior-averaged training MSE + PAC-Bayes slack. Brief §11
 * convention: report the picked DEGREE as ⌊effDoF⌋ − 1 at the picked λ.
 */
export function pacBayesPick(
  X: Float64Array,
  Y: Float64Array,
  kMax: number,
  lambdas: Float64Array,
  sigmaP: number,
  tau: number,
  delta: number,
): { hatLambda: number; pickedEffectiveDof: number; pickedDegree: number } {
  const n = X.length;
  const d = kMax + 1;
  const sigmaP2 = sigmaP * sigmaP;
  const tau2 = tau * tau;
  const vTrace = vandermondeTrace(X, kMax);
  let argmin = 0;
  let best = Infinity;
  let bestEffDoF = 0;
  for (let i = 0; i < lambdas.length; i++) {
    const fit = ridgeFit(X, Y, kMax, lambdas[i]);
    let muNormSq = 0;
    for (let j = 0; j < d; j++) muNormSq += fit.coefs[j] * fit.coefs[j];
    const kl = gaussianKLIsotropic(muNormSq, tau2, sigmaP2, d);
    const postMse = posteriorAveragedMse(fit.trainingMse, tau2, vTrace, n);
    const total = postMse + pacBayesSlack(kl, n, delta);
    if (total < best) {
      best = total;
      argmin = i;
      bestEffDoF = fit.effectiveDof;
    }
  }
  return {
    hatLambda: lambdas[argmin],
    pickedEffectiveDof: bestEffDoF,
    pickedDegree: Math.max(0, Math.floor(bestEffDoF) - 1),
  };
}

// -----------------------------------------------------------------------------
// §11 Agreement matrix — convenience function combining all six rules + oracle.
// -----------------------------------------------------------------------------

/** All six SRM-rule picks + oracle k* on the polynomial-regression toy at (n, σ). */
export function agreementMatrixRow(
  n: number,
  sigma: number,
  kMax: number,
  delta: number,
  K: number,
  rerolls: number,
  pacBayesSigmaP: number,
  pacBayesTau: number,
  rademacherB: number,
  biasVarianceB: number,
  seed: number,
): {
  oracle: number;
  vapnik: number;
  rademacher: number;
  aic: number;
  bic: number;
  cv: number;
  pacBayes: number;
} {
  const rng = mulberry32(seed);
  // Oracle from a fresh MC; uses its own RNG branch so it doesn't pre-burn the
  // pick-rule samples.
  const oracle = biasVarianceMC(n, sigma, kMax, biasVarianceB, mulberry32(seed ^ 0xa5a5a5a5)).kStar;
  const { X, Y } = sampleSinTarget(n, sigma, rng);
  const vapnik = vapnikPick(X, Y, kMax, delta, 1);
  const rademacher = rademacherPick(X, Y, kMax, delta, rademacherB, rng);
  const aic = aicPick(X, Y, kMax);
  const bic = bicPick(X, Y, kMax);
  const cv = cvPickHistogram(X, Y, kMax, K, rerolls, rng).mode;
  const lambdas = logspace(-6, 4, 51);
  const pacBayes = pacBayesPick(X, Y, kMax, lambdas, pacBayesSigmaP, pacBayesTau, delta).pickedDegree;
  return { oracle, vapnik, rademacher, aic, bic, cv, pacBayes };
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/** Logarithmically spaced grid of `count` values from 10^a to 10^b inclusive. */
export function logspace(a: number, b: number, count: number): Float64Array {
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    out[i] = Math.pow(10, a + (b - a) * t);
  }
  return out;
}

/** Linearly spaced grid of `count` values from a to b inclusive. */
export function linspace(a: number, b: number, count: number): Float64Array {
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) out[i] = a + ((b - a) * i) / (count - 1);
  return out;
}

/** Mode of an integer array; ties broken by smaller value. */
export function mode(xs: Int32Array): number {
  if (xs.length === 0) return 0;
  let minVal = xs[0];
  let maxVal = xs[0];
  for (let i = 1; i < xs.length; i++) {
    if (xs[i] < minVal) minVal = xs[i];
    if (xs[i] > maxVal) maxVal = xs[i];
  }
  const counts = new Int32Array(maxVal - minVal + 1);
  for (let i = 0; i < xs.length; i++) counts[xs[i] - minVal]++;
  let bestIdx = 0;
  for (let i = 1; i < counts.length; i++) if (counts[i] > counts[bestIdx]) bestIdx = i;
  return bestIdx + minVal;
}

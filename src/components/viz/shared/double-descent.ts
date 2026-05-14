// =============================================================================
// double-descent.ts
//
// Shared math primitives for the double-descent topic's viz components.
// Ported from notebooks/double-descent/01_double_descent.ipynb §§1–11.
//
// In-browser TS for every §§1–11 viz (closed-form Hastie 2022 risk, MP density,
// SVD pseudoinverse and ridge shrinkage, GD trajectory in SVD basis, random
// features, polynomial bias-variance, eigenvalue spectra). No precomputed JSON.
//
// All exports are pure functions — deterministic for a given seeded RNG.
//
// Source-of-truth notebook:
//   notebooks/double-descent/01_double_descent.ipynb
// =============================================================================

import { mulberry32 } from './normalizing-flows';

export { mulberry32 };

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/** Thin SVD of an n×p matrix: U is n×r, S is length-r, Vt is r×p, all row-major. */
export type ThinSVD = {
  U: Float64Array;
  S: Float64Array;
  Vt: Float64Array;
  n: number;
  p: number;
  r: number;
};

export type SymEig = { values: Float64Array; vectors: Float64Array };

/** Bias / variance / total decomposition at a single γ. */
export type RiskDecomp = { bias2: number; variance: number; total: number };

/** GD-trajectory diagnostics evaluated at a fixed iteration grid (closed form). */
export type GdTrajectory = {
  iters: Int32Array;
  trainLoss: Float64Array;
  normBeta: Float64Array;
  distToMinNorm: Float64Array;
  testRisk: Float64Array;
  betaMinNormNormSq: number; // ‖β̂†‖²
};

// -----------------------------------------------------------------------------
// Gaussian RNG via Box–Muller
// -----------------------------------------------------------------------------

/** Returns iid standard normal draws using the supplied uniform [0,1) RNG. */
export function gaussianRng(rng: () => number): () => number {
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = 0;
    while (u1 < 1e-300) u1 = rng();
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

/** Fill `buf` with iid N(0, σ²). Mutates and returns the same buffer. */
export function fillGaussian(buf: Float64Array, sigma: number, gauss: () => number): Float64Array {
  for (let i = 0; i < buf.length; i++) buf[i] = sigma * gauss();
  return buf;
}

// -----------------------------------------------------------------------------
// Symmetric eigendecomposition via cyclic Jacobi.
// Returns eigenvalues in Jacobi-rotation order (NOT sorted). Threshold against
// max(values) when truncating — see CLAUDE.md §"Jacobi eigendecomp output is
// unsorted."
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

// -----------------------------------------------------------------------------
// Matrix products (row-major; X is n×p, vectors are length-n / length-p)
// -----------------------------------------------------------------------------

/** Return X^T X (p×p, row-major). */
export function gramMatrix(X: Float64Array, n: number, p: number): Float64Array {
  const G = new Float64Array(p * p);
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      let s = 0;
      for (let k = 0, base = 0; k < n; k++, base += p) s += X[base + i] * X[base + j];
      G[i * p + j] = s;
      G[j * p + i] = s;
    }
  }
  return G;
}

/** Return X X^T (n×n, row-major). */
export function outerGramMatrix(X: Float64Array, n: number, p: number): Float64Array {
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      const bi = i * p;
      const bj = j * p;
      for (let k = 0; k < p; k++) s += X[bi + k] * X[bj + k];
      G[i * n + j] = s;
      G[j * n + i] = s;
    }
  }
  return G;
}

/** y = X β, where X is n×p row-major. */
export function matVec(X: Float64Array, beta: Float64Array, n: number, p: number, out?: Float64Array): Float64Array {
  const y = out ?? new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const base = i * p;
    for (let j = 0; j < p; j++) s += X[base + j] * beta[j];
    y[i] = s;
  }
  return y;
}

/** v = X^T y, where X is n×p row-major. */
export function matTVec(X: Float64Array, y: Float64Array, n: number, p: number, out?: Float64Array): Float64Array {
  const v = out ?? new Float64Array(p);
  v.fill(0);
  for (let i = 0; i < n; i++) {
    const base = i * p;
    const yi = y[i];
    for (let j = 0; j < p; j++) v[j] += X[base + j] * yi;
  }
  return v;
}

// -----------------------------------------------------------------------------
// Thin SVD via the smaller Gram matrix (numpy.linalg.svd full_matrices=False)
//
// For n×p with n ≥ p: eigendecompose X^T X (p×p), recover U_thin = X V Σ⁻¹.
// For n×p with n  < p: eigendecompose X X^T (n×n), recover Vt_thin = Σ⁻¹ U^T X.
//
// Returns r = min(n, p) singular values. Eigenvalues from Jacobi are not sorted;
// we explicitly sort by descending singular value here so consumers can rely on
// the descending order.
// -----------------------------------------------------------------------------

/** Thin SVD of an n×p row-major matrix. r = min(n, p). */
export function thinSVD(X: Float64Array, n: number, p: number): ThinSVD {
  const r = Math.min(n, p);
  if (n >= p) {
    // Eigendecompose X^T X (p×p).
    const G = gramMatrix(X, n, p);
    const { values, vectors: V } = symEigJacobi(G, p);
    // Sort indices by descending eigenvalue.
    const order = Array.from({ length: p }, (_, i) => i);
    order.sort((a, b) => values[b] - values[a]);
    // Build Σ (length r=p) and Vt (p×p sorted).
    const S = new Float64Array(p);
    const Vt = new Float64Array(p * p);
    for (let k = 0; k < p; k++) {
      const eig = values[order[k]];
      S[k] = Math.sqrt(Math.max(eig, 0));
      // row k of Vt = column order[k] of V
      for (let j = 0; j < p; j++) Vt[k * p + j] = V[j * p + order[k]];
    }
    // U_thin = X V Σ⁻¹, n×p. Where S[k] > 0, U[:, k] = (1/S[k]) * X * V[:, k] = (1/S[k]) * X * Vt[k, :]^T.
    const U = new Float64Array(n * p);
    const Xvk = new Float64Array(n);
    const rcondSq = Math.max(n, p) * 2.220446049250313e-16;
    const sTol = Math.sqrt(rcondSq) * (S[0] ?? 1);
    for (let k = 0; k < p; k++) {
      if (S[k] <= sTol) {
        // leave U[:, k] = 0; this column is in the null-direction (rank-deficient)
        continue;
      }
      const baseVt = k * p;
      // Xvk = X · Vt[k, :]^T
      for (let i = 0; i < n; i++) {
        let s = 0;
        const base = i * p;
        for (let j = 0; j < p; j++) s += X[base + j] * Vt[baseVt + j];
        Xvk[i] = s;
      }
      const inv = 1 / S[k];
      for (let i = 0; i < n; i++) U[i * p + k] = inv * Xvk[i];
    }
    return { U, S, Vt, n, p, r };
  } else {
    // Wide case n < p. Eigendecompose X X^T (n×n).
    const Gn = outerGramMatrix(X, n, p);
    const { values, vectors: Un } = symEigJacobi(Gn, n);
    const order = Array.from({ length: n }, (_, i) => i);
    order.sort((a, b) => values[b] - values[a]);
    const S = new Float64Array(n);
    const U = new Float64Array(n * n);
    for (let k = 0; k < n; k++) {
      const eig = values[order[k]];
      S[k] = Math.sqrt(Math.max(eig, 0));
      for (let i = 0; i < n; i++) U[i * n + k] = Un[i * n + order[k]];
    }
    // Vt_thin = Σ⁻¹ U^T X, n×p.
    const Vt = new Float64Array(n * p);
    const rcondSq = Math.max(n, p) * 2.220446049250313e-16;
    const sTol = Math.sqrt(rcondSq) * (S[0] ?? 1);
    for (let k = 0; k < n; k++) {
      if (S[k] <= sTol) continue;
      const inv = 1 / S[k];
      const baseVt = k * p;
      // row k of Vt = inv * U[:, k]^T · X. Swap (j, i) loop order so X is read
      // row-major and the U column-walk hoists out of the inner loop — same
      // arithmetic, but n × p mults move from O(np) row-index recomputations
      // to a single multiply per i.
      for (let i = 0; i < n; i++) {
        const uik = U[i * n + k] * inv;
        const baseX = i * p;
        for (let j = 0; j < p; j++) {
          Vt[baseVt + j] += uik * X[baseX + j];
        }
      }
    }
    return { U, S, Vt, n, p, r: n };
  }
}

// -----------------------------------------------------------------------------
// Minimum-norm interpolator β̂† = X⁺ y via SVD pseudoinverse.
//
// β̂† = V Σ⁺ U^T y, where Σ⁺ inverts s_i for s_i > rcond·s_max and zeros the rest.
// Equivalent to numpy.linalg.lstsq(X, y, rcond=None) and to ridge with λ→0.
// -----------------------------------------------------------------------------

/** Compute β̂† from a precomputed SVD. */
export function betaFromSVD(svd: ThinSVD, y: Float64Array, rcond?: number): Float64Array {
  const { U, S, Vt, n, p, r } = svd;
  const rcondEff = rcond ?? Math.max(n, p) * 2.220446049250313e-16;
  let sMax = 0;
  for (let k = 0; k < r; k++) if (S[k] > sMax) sMax = S[k];
  const sTol = rcondEff * sMax;
  // c_k = (1/s_k) · (U[:, k]^T y), zeroed below tol. Swap (k, i) loops so U is
  // walked row-major rather than column-major.
  const c = new Float64Array(r);
  const Ucols = n; // U has shape (n, n) for wide case, (n, p) for tall — column index k < r maps via column-stride
  const Ustride = n >= p ? p : n; // U row-major width
  for (let i = 0; i < Ucols; i++) {
    const base = i * Ustride;
    const yi = y[i];
    for (let k = 0; k < r; k++) {
      c[k] += U[base + k] * yi;
    }
  }
  for (let k = 0; k < r; k++) {
    c[k] = S[k] > sTol ? c[k] / S[k] : 0;
  }
  // β = V c = Vt^T c, length p. Outer loop over k hoists `k * p`, inner loop
  // walks Vt row-major and accumulates into β.
  const beta = new Float64Array(p);
  for (let k = 0; k < r; k++) {
    const base = k * p;
    const ck = c[k];
    for (let j = 0; j < p; j++) {
      beta[j] += Vt[base + j] * ck;
    }
  }
  return beta;
}

/** Convenience: SVD + pseudoinverse in one call. */
export function betaMinNorm(X: Float64Array, y: Float64Array, n: number, p: number, rcond?: number): Float64Array {
  const svd = thinSVD(X, n, p);
  return betaFromSVD(svd, y, rcond);
}

/**
 * Ridge β_λ via singular-value shrinkage:
 *   β_λ = Σ_i [s_i / (s_i² + λ)] · (U[:, i]^T y) · V[:, i]
 *
 * Reuses the SVD with no refactorization across λ values.
 */
export function betaRidgeFromSVD(svd: ThinSVD, y: Float64Array, lambda: number): Float64Array {
  const { U, S, Vt, n, p, r } = svd;
  const Ucols = n;
  const Ustride = n >= p ? p : n;
  // Walk U row-major: same arithmetic, much better cache behavior than the
  // column-walk per k.
  const c = new Float64Array(r);
  for (let i = 0; i < Ucols; i++) {
    const base = i * Ustride;
    const yi = y[i];
    for (let k = 0; k < r; k++) {
      c[k] += U[base + k] * yi;
    }
  }
  for (let k = 0; k < r; k++) {
    const sk = S[k];
    c[k] = (sk * c[k]) / (sk * sk + lambda);
  }
  // β = Vt^T c — same row-major walk as betaFromSVD.
  const beta = new Float64Array(p);
  for (let k = 0; k < r; k++) {
    const base = k * p;
    const ck = c[k];
    for (let j = 0; j < p; j++) {
      beta[j] += Vt[base + j] * ck;
    }
  }
  return beta;
}

// -----------------------------------------------------------------------------
// §5 Marchenko–Pastur density and support
//
// Support of the bulk distribution: [(1−√γ)², (1+√γ)²].
// Density on the bulk: f_γ(λ) = √((λ₊−λ)(λ−λ₋)) / (2π γ λ).
// For γ > 1, an atomic mass max(0, 1−1/γ) sits at λ = 0; the bulk integrates to 1/γ.
// -----------------------------------------------------------------------------

/** Marchenko–Pastur support endpoints. */
export function mpSupport(gamma: number): { lamMinus: number; lamPlus: number } {
  const sg = Math.sqrt(gamma);
  return { lamMinus: (1 - sg) * (1 - sg), lamPlus: (1 + sg) * (1 + sg) };
}

/** Marchenko–Pastur bulk density at λ. Returns 0 outside [λ₋, λ₊] or at λ = 0. */
export function mpDensity(lambda: number, gamma: number): number {
  if (lambda <= 0 || gamma <= 0) return 0;
  const { lamMinus, lamPlus } = mpSupport(gamma);
  if (lambda < lamMinus || lambda > lamPlus) return 0;
  const root = Math.sqrt((lamPlus - lambda) * (lambda - lamMinus));
  return root / (2 * Math.PI * gamma * lambda);
}

/** Atomic mass at λ = 0 for γ > 1 (corresponds to p − n zero singular values). */
export function mpAtomZero(gamma: number): number {
  return gamma > 1 ? 1 - 1 / gamma : 0;
}

/**
 * Sample the bulk MP density on a grid for plotting. Returns parallel arrays
 * (lambda, f). The grid is uniform in [λ₋, λ₊]. Includes the support endpoints.
 */
export function mpDensityGrid(gamma: number, nPoints: number): { lambda: Float64Array; f: Float64Array } {
  const { lamMinus, lamPlus } = mpSupport(gamma);
  const lambda = new Float64Array(nPoints);
  const f = new Float64Array(nPoints);
  if (nPoints <= 1) {
    lambda[0] = lamMinus;
    f[0] = 0;
    return { lambda, f };
  }
  for (let i = 0; i < nPoints; i++) {
    const t = i / (nPoints - 1);
    const lam = lamMinus + t * (lamPlus - lamMinus);
    lambda[i] = lam;
    f[i] = mpDensity(lam, gamma);
  }
  return { lambda, f };
}

// -----------------------------------------------------------------------------
// §6 Hastie 2022 asymptotic risk and bias / variance decomposition
//
// Well-specified setup: X ~ N(0, I_p), β* ∈ ℝᵖ with ‖β*‖² = r², ε ~ N(0, σ² I_n).
// Proportional asymptotic limit n, p → ∞ with γ = p/n fixed.
//
//   R(γ) = σ²γ/(1−γ)                          for γ < 1   (pure variance)
//        = +∞                                  for γ = 1
//        = r²(1 − 1/γ) + σ²/(γ − 1)            for γ > 1   (bias + variance)
//
// Returns +Infinity exactly at γ = 1; consumers plot in log scale.
// -----------------------------------------------------------------------------

/** Hastie 2022 Theorem 2 total excess risk at aspect ratio γ. */
export function hastieRisk(gamma: number, r2: number, sigma2: number): number {
  if (gamma === 1) return Infinity;
  if (gamma < 1) return (sigma2 * gamma) / (1 - gamma);
  return r2 * (1 - 1 / gamma) + sigma2 / (gamma - 1);
}

/** Per-component bias² and variance from the Hastie 2022 decomposition. */
export function hastieRiskDecomp(gamma: number, r2: number, sigma2: number): RiskDecomp {
  if (gamma === 1) return { bias2: 0, variance: Infinity, total: Infinity };
  if (gamma < 1) {
    const variance = (sigma2 * gamma) / (1 - gamma);
    return { bias2: 0, variance, total: variance };
  }
  const bias2 = r2 * (1 - 1 / gamma);
  const variance = sigma2 / (gamma - 1);
  return { bias2, variance, total: bias2 + variance };
}

/**
 * The local minimum of R(γ) in the overparameterized regime (γ > 1) sits at
 * γ_min = 1/(1 − σ/r) when r > σ. Returns +Infinity (monotone-decreasing,
 * no finite minimum) when r ≤ σ. This is the "more parameters always helps"
 * threshold.
 */
export function hastieOptimalGamma(r2: number, sigma2: number): number {
  const r = Math.sqrt(r2);
  const sigma = Math.sqrt(sigma2);
  if (r <= sigma) return Infinity;
  return 1 / (1 - sigma / r);
}

// -----------------------------------------------------------------------------
// Excess risk (population, for known β*)
//
// For ridgeless OLS with isotropic Gaussian test points X_new ~ N(0, I_p):
//   ExcessRisk = E[(X_new^T (β̂ − β*))²] = ‖β̂ − β*‖²  (the test-point covariance is I_p).
//
// For the §1 misspecified setup where β* lives in ℝ^{P_max} but the model uses
// only the first p features, "excess risk" means the squared prediction error
// against the full signal X_new^T β*_full. Caller passes the right comparison.
// -----------------------------------------------------------------------------

/** ‖β̂ − β*‖² for the well-specified case. */
export function excessRiskWellSpecified(betaHat: Float64Array, betaStar: Float64Array): number {
  let s = 0;
  for (let j = 0; j < betaHat.length; j++) {
    const d = betaHat[j] - betaStar[j];
    s += d * d;
  }
  return s;
}

/**
 * Misspecified excess risk: β̂ ∈ ℝᵖ, β*_full ∈ ℝ^{P_max} (with p ≤ P_max),
 * test point X_new ~ N(0, I_{P_max}). The first p coords of β̂ vs β*_full,
 * plus the missing tail contributes ‖β*_{p:P_max}‖².
 */
export function excessRiskMisspecified(
  betaHat: Float64Array,
  betaStarFull: Float64Array,
  p: number,
): number {
  let s = 0;
  for (let j = 0; j < p; j++) {
    const d = betaHat[j] - betaStarFull[j];
    s += d * d;
  }
  for (let j = p; j < betaStarFull.length; j++) s += betaStarFull[j] * betaStarFull[j];
  return s;
}

// -----------------------------------------------------------------------------
// §8 Random-feature map
//
// φ(x) = σ(W x / √d), W ∈ ℝ^{p×d} with iid N(0,1) entries. The √d rescaling
// keeps the activation input order 1. Caller supplies the activation function.
// -----------------------------------------------------------------------------

export const activations = {
  linear: (z: number) => z,
  relu: (z: number) => (z > 0 ? z : 0),
  tanh: (z: number) => Math.tanh(z),
} as const;

export type ActivationName = keyof typeof activations;

/**
 * Build the random-feature design matrix Φ ∈ ℝ^{n×p} from input matrix X
 * (n×d row-major) and weight matrix W (p×d row-major).
 *
 *   Φ[i, j] = σ((W[j, :] · X[i, :]) / √d)
 */
export function randomFeatureMap(
  X: Float64Array,
  n: number,
  d: number,
  W: Float64Array,
  p: number,
  activation: (z: number) => number,
): Float64Array {
  const Phi = new Float64Array(n * p);
  const invSqrtD = 1 / Math.sqrt(d);
  for (let i = 0; i < n; i++) {
    const baseX = i * d;
    for (let j = 0; j < p; j++) {
      const baseW = j * d;
      let s = 0;
      for (let k = 0; k < d; k++) s += W[baseW + k] * X[baseX + k];
      Phi[i * p + j] = activation(s * invSqrtD);
    }
  }
  return Phi;
}

// -----------------------------------------------------------------------------
// §11 Polynomial Vandermonde in Legendre basis on [-1, 1]
//
// Three-term recurrence: P₀(x)=1, P₁(x)=x, (k+1) P_{k+1}(x) = (2k+1) x P_k(x) − k P_{k−1}(x).
// Returns an n×d row-major matrix V with V[i, k] = P_k(X[i]).
// -----------------------------------------------------------------------------

export function legendreVandermonde(X: Float64Array, d: number): Float64Array {
  const n = X.length;
  const V = new Float64Array(n * d);
  if (d === 0) return V;
  for (let i = 0; i < n; i++) {
    V[i * d] = 1;
    if (d > 1) V[i * d + 1] = X[i];
    for (let k = 1; k < d - 1; k++) {
      V[i * d + k + 1] = ((2 * k + 1) * X[i] * V[i * d + k] - k * V[i * d + k - 1]) / (k + 1);
    }
  }
  return V;
}

/** Evaluate Σ_k c_k P_k(x) at a scalar x via the Legendre recurrence. */
export function legendreEval(coefs: Float64Array, x: number): number {
  const d = coefs.length;
  if (d === 0) return 0;
  if (d === 1) return coefs[0];
  let prev = 1;
  let curr = x;
  let acc = coefs[0] * prev + coefs[1] * curr;
  for (let k = 1; k < d - 1; k++) {
    const next = ((2 * k + 1) * x * curr - k * prev) / (k + 1);
    acc += coefs[k + 1] * next;
    prev = curr;
    curr = next;
  }
  return acc;
}

// -----------------------------------------------------------------------------
// §9 Gradient-descent trajectory in the SVD basis (closed form)
//
// GD on (1/2)‖Xβ − y‖² from β₀ = 0, with learning rate η ∈ (0, 2/s_max²),
// yields the explicit iterates
//
//   β_t = Σ_i [(1 − (1 − η s_i²)^t) / s_i] · (u_i^T y) · v_i.
//
// We evaluate trajectory diagnostics at a user-specified iteration grid via
// this formula — no actual iteration. The limit (t → ∞) is the minimum-norm
// interpolator β̂† (Theorem 2).
// -----------------------------------------------------------------------------

export interface GdTrajectoryOptions {
  X: Float64Array;
  y: Float64Array;
  n: number;
  p: number;
  betaStar: Float64Array; // for test-risk panel (d)
  iters: Int32Array; // iteration grid (e.g. [0, 1, 2, 5, 10, 20, …, 1000])
  etaFraction?: number; // η = etaFraction / s_max²; default 0.9
  svd?: ThinSVD; // optional precomputed SVD
}

/** Closed-form GD trajectory diagnostics at the given iteration grid. */
export function gdTrajectory(opts: GdTrajectoryOptions): GdTrajectory {
  const { X, y, n, p, betaStar, iters } = opts;
  const etaFraction = opts.etaFraction ?? 0.9;
  const svd = opts.svd ?? thinSVD(X, n, p);
  const { S, r } = svd;
  let sMax2 = 0;
  for (let k = 0; k < r; k++) {
    const s2 = S[k] * S[k];
    if (s2 > sMax2) sMax2 = s2;
  }
  const eta = sMax2 > 0 ? etaFraction / sMax2 : 0;
  // Precompute (u_k^T y) for each k.
  const Ucols = n;
  const Ustride = n >= p ? p : n;
  const uy = new Float64Array(r);
  const rcondSq = Math.max(n, p) * 2.220446049250313e-16;
  const sTol = Math.sqrt(rcondSq) * (S[0] ?? 1);
  for (let k = 0; k < r; k++) {
    let s = 0;
    for (let i = 0; i < Ucols; i++) s += svd.U[i * Ustride + k] * y[i];
    uy[k] = s;
  }
  // Final min-norm coefficient norm: Σ_k (uy_k / s_k)² over active modes.
  let betaMinNormNormSq = 0;
  for (let k = 0; k < r; k++) {
    if (S[k] <= sTol) continue;
    const ck = uy[k] / S[k];
    betaMinNormNormSq += ck * ck;
  }
  // β̂† projected onto v_k in basis-space coefficients c_k* = uy[k]/S[k].
  const cStar = new Float64Array(r);
  for (let k = 0; k < r; k++) cStar[k] = S[k] > sTol ? uy[k] / S[k] : 0;

  // Pre-build β̂† in ambient space for test-risk dist evaluation.
  // β̂† = Σ_k cStar[k] · v_k = Vt^T · cStar. Walk Vt row-major.
  const betaDagger = new Float64Array(p);
  for (let k = 0; k < r; k++) {
    const base = k * p;
    const ck = cStar[k];
    for (let j = 0; j < p; j++) {
      betaDagger[j] += svd.Vt[base + j] * ck;
    }
  }

  const T = iters.length;
  const trainLoss = new Float64Array(T);
  const normBeta = new Float64Array(T);
  const distToMinNorm = new Float64Array(T);
  const testRisk = new Float64Array(T);

  const c_t = new Float64Array(r);
  const beta_t = new Float64Array(p);

  // Precompute (1 − η s_k²) for each k — the decay base is constant across ti.
  const decayBases = new Float64Array(r);
  for (let k = 0; k < r; k++) decayBases[k] = 1 - eta * S[k] * S[k];

  // Tall-case (n > p) orthogonal residual component is constant across ti:
  // it contributes ‖y‖² − ‖U^T y‖² to the residual norm at every iteration.
  let orthogonalResidual = 0;
  if (n > p) {
    let yNorm2 = 0;
    for (let i = 0; i < n; i++) yNorm2 += y[i] * y[i];
    let uy2 = 0;
    for (let k = 0; k < r; k++) uy2 += uy[k] * uy[k];
    orthogonalResidual = yNorm2 - uy2;
  }

  for (let ti = 0; ti < T; ti++) {
    const t = iters[ti];
    // c_k(t) = (1 − (1 − η s_k²)^t) · cStar_k  for active modes
    for (let k = 0; k < r; k++) {
      if (S[k] <= sTol) {
        c_t[k] = 0;
        continue;
      }
      const decay = Math.pow(decayBases[k], t);
      c_t[k] = (1 - decay) * cStar[k];
    }
    // norm² of β_t = Σ_k c_t[k]² (rows of V are orthonormal in ambient space).
    let nb = 0;
    for (let k = 0; k < r; k++) nb += c_t[k] * c_t[k];
    normBeta[ti] = nb;
    // dist² to β̂† = Σ_k (c_t[k] − cStar[k])²
    let dnorm = 0;
    for (let k = 0; k < r; k++) {
      const d = c_t[k] - cStar[k];
      dnorm += d * d;
    }
    distToMinNorm[ti] = Math.sqrt(dnorm);
    // Train loss = (1/2) ‖X β_t − y‖². In the SVD basis: residual in U-coordinates =
    //   r_k = (1 − (1 − η s_k²)^t) · uy[k] − uy[k] = − (1 − η s_k²)^t · uy[k], for k ≤ r.
    // Plus the orthogonal-to-col(U) component of y (only nonzero for tall n > p):
    // hoisted into `orthogonalResidual` above.
    let trl = orthogonalResidual;
    for (let k = 0; k < r; k++) {
      const decay = Math.pow(decayBases[k], t);
      trl += decay * decay * uy[k] * uy[k];
    }
    trainLoss[ti] = 0.5 * trl;
    // Build β_t = Vt^T · c_t in ambient coordinates (row-major walk over Vt).
    beta_t.fill(0);
    for (let k = 0; k < r; k++) {
      const base = k * p;
      const ctk = c_t[k];
      for (let j = 0; j < p; j++) {
        beta_t[j] += svd.Vt[base + j] * ctk;
      }
    }
    // Test risk = ‖β_t − β*‖² (well-specified setup; isotropic Gaussian test point).
    let tr = 0;
    for (let j = 0; j < p; j++) {
      const d = beta_t[j] - betaStar[j];
      tr += d * d;
    }
    testRisk[ti] = tr;
  }

  return { iters, trainLoss, normBeta, distToMinNorm, testRisk, betaMinNormNormSq };
}

// -----------------------------------------------------------------------------
// Effective rank and condition number
// -----------------------------------------------------------------------------

/** Number of singular values above rcond·s_max. */
export function effectiveRank(S: Float64Array, rcond: number): number {
  let sMax = 0;
  for (let k = 0; k < S.length; k++) if (S[k] > sMax) sMax = S[k];
  const tol = rcond * sMax;
  let count = 0;
  for (let k = 0; k < S.length; k++) if (S[k] > tol) count++;
  return count;
}

/** κ = s_max / s_min over the nonzero singular values (returns +∞ if all zero). */
export function conditionNumber(S: Float64Array, rcond: number = 1e-15): number {
  let sMax = 0;
  for (let k = 0; k < S.length; k++) if (S[k] > sMax) sMax = S[k];
  if (sMax === 0) return Infinity;
  const tol = rcond * sMax;
  let sMinNonzero = Infinity;
  for (let k = 0; k < S.length; k++) if (S[k] > tol && S[k] < sMinNonzero) sMinNonzero = S[k];
  if (!Number.isFinite(sMinNonzero)) return Infinity;
  return sMax / sMinNonzero;
}

// -----------------------------------------------------------------------------
// Monte Carlo helpers — used inside slider-commit hot paths of viz components.
// All sweep helpers preallocate scratch buffers and avoid per-iteration allocations.
// -----------------------------------------------------------------------------

/** Draw an n×p iid N(0,1) matrix into `out` (preallocated). */
export function fillIsotropicGaussian(
  out: Float64Array,
  n: number,
  p: number,
  gauss: () => number,
): Float64Array {
  for (let i = 0; i < n * p; i++) out[i] = gauss();
  return out;
}

/** β* uniformly distributed on the sphere of radius r in ℝᵖ. */
export function sampleSphereBetaStar(p: number, r: number, gauss: () => number): Float64Array {
  const v = new Float64Array(p);
  let norm = 0;
  for (let j = 0; j < p; j++) {
    v[j] = gauss();
    norm += v[j] * v[j];
  }
  const scale = r / Math.sqrt(norm);
  for (let j = 0; j < p; j++) v[j] *= scale;
  return v;
}

/**
 * Monte Carlo estimate of the §1 / §3 / §7 excess-risk curve.
 *
 * Draws B independent (X, ε) replicates. For each replicate sweeps the model
 * size grid `pGrid`, fitting the minimum-norm interpolator on the first p
 * columns of X (column-slice misspecification). Computes the per-(b, p) excess
 * risk against the full β*.
 *
 * Caller supplies a fixed β* ∈ ℝ^{Pmax} (e.g., uniform on sphere of radius r).
 *
 * Returns mean / IQR low / IQR high per p, as Float64Array of length |pGrid|.
 */
export interface ExcessRiskSweepOptions {
  n: number;
  pMax: number;
  pGrid: Int32Array;
  betaStarFull: Float64Array; // length pMax
  sigma: number;
  B: number;
  rng: () => number;
  ridgeLambdas?: Float64Array; // optional ridge overlay; if set, returns one row per λ
  rcond?: number;
}

export interface ExcessRiskSweepResult {
  mean: Float64Array; // length pGrid.length (or pGrid.length × #lambdas if ridgeLambdas set)
  iqrLow: Float64Array;
  iqrHigh: Float64Array;
  ridgeLambdas?: Float64Array;
}

export function excessRiskSweep(opts: ExcessRiskSweepOptions): ExcessRiskSweepResult {
  const { n, pMax, pGrid, betaStarFull, sigma, B, rng, rcond } = opts;
  const gauss = gaussianRng(rng);
  const lambdas = opts.ridgeLambdas;
  const Pgrid = pGrid.length;
  const Nlam = lambdas ? lambdas.length : 1;
  const numRows = Pgrid * Nlam;
  // Replicate-major buffer to allow per-p IQR.
  const samples = new Float64Array(B * numRows);
  // Pre-allocate the maximum design and response.
  const Xfull = new Float64Array(n * pMax);
  const y = new Float64Array(n);
  const Xpa = (p: number) => {
    // Slice view: copy first p columns of Xfull into Xslice. We allocate inside the loop because p varies.
    // To avoid per-call alloc we use a single max-size Xslice buffer.
    return null; // placeholder; see actual logic below
  };
  void Xpa;
  const Xslice = new Float64Array(n * pMax);
  for (let b = 0; b < B; b++) {
    // Generate X ∈ ℝ^{n×pMax} once, slice columns inside the loop.
    fillIsotropicGaussian(Xfull, n, pMax, gauss);
    // y = X β* + ε
    matVec(Xfull, betaStarFull, n, pMax, y);
    for (let i = 0; i < n; i++) y[i] += sigma * gauss();
    for (let gi = 0; gi < Pgrid; gi++) {
      const p = pGrid[gi];
      // Build Xslice (n × p) — copy the first p columns of Xfull (column-major slice = row-strided copy).
      for (let i = 0; i < n; i++) {
        const srcBase = i * pMax;
        const dstBase = i * p;
        for (let j = 0; j < p; j++) Xslice[dstBase + j] = Xfull[srcBase + j];
      }
      const svd = thinSVD(Xslice, n, p);
      if (lambdas) {
        for (let li = 0; li < Nlam; li++) {
          const lam = lambdas[li];
          const betaHat = lam === 0 ? betaFromSVD(svd, y, rcond) : betaRidgeFromSVD(svd, y, lam);
          const risk = excessRiskMisspecified(betaHat, betaStarFull, p);
          samples[b * numRows + gi * Nlam + li] = risk;
        }
      } else {
        const betaHat = betaFromSVD(svd, y, rcond);
        const risk = excessRiskMisspecified(betaHat, betaStarFull, p);
        samples[b * numRows + gi] = risk;
      }
    }
  }
  // Per-(p, λ) mean and IQR across replicates.
  const mean = new Float64Array(numRows);
  const iqrLow = new Float64Array(numRows);
  const iqrHigh = new Float64Array(numRows);
  const colBuf = new Float64Array(B);
  for (let c = 0; c < numRows; c++) {
    for (let b = 0; b < B; b++) colBuf[b] = samples[b * numRows + c];
    let s = 0;
    for (let b = 0; b < B; b++) s += colBuf[b];
    mean[c] = s / B;
    // IQR via sort
    const sorted = colBuf.slice().sort();
    const q1 = sorted[Math.floor(B * 0.25)];
    const q3 = sorted[Math.floor(B * 0.75)];
    iqrLow[c] = q1;
    iqrHigh[c] = q3;
  }
  return {
    mean,
    iqrLow,
    iqrHigh,
    ridgeLambdas: lambdas,
  };
}

// -----------------------------------------------------------------------------
// Classical bias-variance Monte Carlo on Legendre polynomial regression
//
// Setup: n training points X ~ U(−1, 1), Y = sin(πX) + N(0, σ²). For each
// degree d ∈ [0, dMax], fit OLS in the Legendre basis and accumulate the
// Bias-Variance decomposition across B replicates and a shared test grid.
// -----------------------------------------------------------------------------

export interface ClassicalBVOptions {
  n: number;
  sigma: number;
  dMax: number; // inclusive
  B: number;
  testGrid: Float64Array;
  rng: () => number;
}

export interface ClassicalBVResult {
  degrees: Int32Array;
  bias2: Float64Array;
  variance: Float64Array;
  total: Float64Array;
}

export function classicalBiasVariance(opts: ClassicalBVOptions): ClassicalBVResult {
  const { n, sigma, dMax, B, testGrid, rng } = opts;
  const gauss = gaussianRng(rng);
  const D = dMax + 1;
  const M = testGrid.length;
  const degrees = new Int32Array(D);
  for (let d = 0; d < D; d++) degrees[d] = d;
  const truth = new Float64Array(M);
  for (let m = 0; m < M; m++) truth[m] = Math.sin(Math.PI * testGrid[m]);
  // Sum and sumsq of f̂(x_m) across replicates, per degree.
  const fSum = new Float64Array(D * M);
  const fSumSq = new Float64Array(D * M);

  for (let b = 0; b < B; b++) {
    const X = new Float64Array(n);
    for (let i = 0; i < n; i++) X[i] = 2 * rng() - 1;
    const Y = new Float64Array(n);
    for (let i = 0; i < n; i++) Y[i] = Math.sin(Math.PI * X[i]) + sigma * gauss();
    for (let d = 0; d < D; d++) {
      const dim = d + 1;
      const V = legendreVandermonde(X, dim);
      const coefs = betaMinNorm(V, Y, n, dim);
      const base = d * M;
      for (let m = 0; m < M; m++) {
        const f = legendreEval(coefs, testGrid[m]);
        fSum[base + m] += f;
        fSumSq[base + m] += f * f;
      }
    }
  }
  const bias2 = new Float64Array(D);
  const variance = new Float64Array(D);
  const total = new Float64Array(D);
  for (let d = 0; d < D; d++) {
    let b2 = 0;
    let v = 0;
    const base = d * M;
    for (let m = 0; m < M; m++) {
      const meanF = fSum[base + m] / B;
      const varF = fSumSq[base + m] / B - meanF * meanF;
      const diff = meanF - truth[m];
      b2 += diff * diff;
      v += varF;
    }
    bias2[d] = Math.max(0, b2 / M - v / (M * B)); // standard finite-B correction
    variance[d] = v / M;
    total[d] = bias2[d] + variance[d];
  }
  return { degrees, bias2, variance, total };
}

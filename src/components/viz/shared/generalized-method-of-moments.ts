// =============================================================================
// Shared helpers for the Generalized Method of Moments topic.
//
// Implements the numerical toolkit that all GMM viz components and the
// verify-suite import from. Computations are pure TypeScript (no React, no
// d3, no SciPy/NumPy bridge) so the verify-suite can run as a standalone
// tsx script.
//
// Wave 1: RNG (mulberry32), Gaussian/Gamma samplers, closed-form MoM and MLE
//   for Gaussian and Gamma, affine GMM criterion + estimator, basic matrix
//   linear algebra (matMul, matVec, matTranspose, matInverse via LU, Cholesky),
//   symmetric eigenvalues via Jacobi rotations, the running-example design
//   matrix A ∈ R^{4×2}, and the digamma function used by Gamma MLE.
//
// Future waves extend this module:
//   Wave 2 — sampleSensorObservations, sandwichVariance, loewnerCompare,
//            efficientWeight, hansenBound, sampleWishart.
//   Wave 3 — twoStepGMM, iteratedGMM, jStatistic, chi2Cdf, chi2Quantile.
//   Wave 4 — simulateLinearIV, twoStageLeastSquares, efficientGMM_IV,
//            andersonRubinSet, firstStageF.
//   Wave 5 — elProfileLambda, elLogR, gelProfile, partialLinearMoment,
//            dmlEstimateFromPrecomputed.
//   Wave 6 — lteFromPrecomputed.
// =============================================================================

// -----------------------------------------------------------------------------
// Vector / matrix shorthand types and basic operations.
// -----------------------------------------------------------------------------

export type Vec = number[];
export type Mat = number[][];

export function zeros(n: number): Vec {
  return new Array(n).fill(0);
}

export function zerosMat(rows: number, cols: number): Mat {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

export function identityMat(n: number): Mat {
  const M = zerosMat(n, n);
  for (let i = 0; i < n; i++) M[i][i] = 1;
  return M;
}

export function matVec(A: Mat, v: Vec): Vec {
  const rows = A.length;
  const cols = A[0].length;
  if (v.length !== cols) throw new Error(`matVec dim mismatch: A is ${rows}x${cols}, v is ${v.length}`);
  const out = zeros(rows);
  for (let i = 0; i < rows; i++) {
    let s = 0;
    const Ai = A[i];
    for (let j = 0; j < cols; j++) s += Ai[j] * v[j];
    out[i] = s;
  }
  return out;
}

export function matMul(A: Mat, B: Mat): Mat {
  const m = A.length;
  const n = B[0].length;
  const k = B.length;
  if (A[0].length !== k) throw new Error(`matMul dim mismatch: ${m}x${A[0].length} times ${k}x${n}`);
  const out = zerosMat(m, n);
  for (let i = 0; i < m; i++) {
    const Ai = A[i];
    const Oi = out[i];
    for (let l = 0; l < k; l++) {
      const a = Ai[l];
      const Bl = B[l];
      for (let j = 0; j < n; j++) Oi[j] += a * Bl[j];
    }
  }
  return out;
}

export function matTranspose(A: Mat): Mat {
  const rows = A.length;
  const cols = A[0].length;
  const out = zerosMat(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j][i] = A[i][j];
  }
  return out;
}

export function vecSub(a: Vec, b: Vec): Vec {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

export function vecDot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function quadForm(W: Mat, v: Vec): number {
  // v^T W v
  return vecDot(v, matVec(W, v));
}

// -----------------------------------------------------------------------------
// Matrix inverse via LU decomposition with partial pivoting. Works for any
// square nonsingular matrix; the verify suite tests both PD (Hansen weighting)
// and indefinite (Jacobian) cases.
// -----------------------------------------------------------------------------

export function matInverse(A: Mat): Mat {
  const n = A.length;
  if (A[0].length !== n) throw new Error(`matInverse: not square ${n}x${A[0].length}`);
  // Augment [A | I] and reduce.
  const aug: number[][] = Array.from({ length: n }, (_, i) => {
    const row = new Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n + i] = 1;
    return row;
  });
  for (let col = 0; col < n; col++) {
    // Partial pivot.
    let maxRow = col;
    let maxAbs = Math.abs(aug[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(aug[r][col]);
      if (v > maxAbs) {
        maxAbs = v;
        maxRow = r;
      }
    }
    if (maxAbs < 1e-14) throw new Error(`matInverse: singular at column ${col}`);
    if (maxRow !== col) {
      const tmp = aug[col];
      aug[col] = aug[maxRow];
      aug[maxRow] = tmp;
    }
    // Normalize pivot row.
    const pivot = aug[col][col];
    const augCol = aug[col];
    for (let j = 0; j < 2 * n; j++) augCol[j] /= pivot;
    // Eliminate other rows.
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r][col];
      if (factor === 0) continue;
      const augR = aug[r];
      for (let j = 0; j < 2 * n; j++) augR[j] -= factor * augCol[j];
    }
  }
  const inv = zerosMat(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) inv[i][j] = aug[i][n + j];
  }
  return inv;
}

// -----------------------------------------------------------------------------
// Cholesky decomposition for symmetric positive-definite matrices.
// Returns lower-triangular L with A = L L^T. Useful for the §6 efficient
// weighting (Omega^{-1} via solve), §10 GEL inner-Newton step, and Wave 2
// sampling (sampleWishart uses Cholesky-of-scale).
// -----------------------------------------------------------------------------

export function cholesky(A: Mat): Mat {
  const n = A.length;
  const L = zerosMat(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error(`cholesky: not positive definite at row ${i} (diag=${s})`);
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

// -----------------------------------------------------------------------------
// Symmetric eigenvalues via cyclic Jacobi rotations. Returns eigenvalues in
// an arbitrary (rotation-order) sequence — caller must sort if needed.
// Used for IdentificationRankInspector and the §5 Loewner ordering tests.
// CLAUDE.md: "Jacobi eigendecomp output is unsorted."
// -----------------------------------------------------------------------------

export function eigvalsSymmetric(A: Mat, maxIter = 200, tol = 1e-12): Vec {
  const n = A.length;
  // Copy A — Jacobi rotates in place.
  const M: number[][] = Array.from({ length: n }, (_, i) => A[i].slice());
  for (let iter = 0; iter < maxIter; iter++) {
    // Find off-diagonal element with largest absolute value.
    let p = 0;
    let q = 1;
    let maxOff = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = Math.abs(M[i][j]);
        if (v > maxOff) {
          maxOff = v;
          p = i;
          q = j;
        }
      }
    }
    if (maxOff < tol) break;
    // Compute Givens rotation that zeros M[p][q].
    const app = M[p][p];
    const aqq = M[q][q];
    const apq = M[p][q];
    let theta: number;
    if (Math.abs(app - aqq) < 1e-30) {
      theta = Math.PI / 4;
    } else {
      theta = 0.5 * Math.atan2(2 * apq, app - aqq);
    }
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    // Update M = R^T M R where R is the (p, q)-Givens rotation.
    for (let i = 0; i < n; i++) {
      const Mip = M[i][p];
      const Miq = M[i][q];
      M[i][p] = c * Mip + s * Miq;
      M[i][q] = -s * Mip + c * Miq;
    }
    for (let j = 0; j < n; j++) {
      const Mpj = M[p][j];
      const Mqj = M[q][j];
      M[p][j] = c * Mpj + s * Mqj;
      M[q][j] = -s * Mpj + c * Mqj;
    }
  }
  const eigs = new Array<number>(n);
  for (let i = 0; i < n; i++) eigs[i] = M[i][i];
  return eigs;
}

// -----------------------------------------------------------------------------
// PRNG — mulberry32, returning a callable rng() in [0, 1). Reused unchanged
// from src/components/viz/shared/always-valid-inference.ts to keep MC seeds
// consistent across topics in the same browser session.
// -----------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function (): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// -----------------------------------------------------------------------------
// Gaussian sampler via Box–Muller, lazy paired so each call returns one draw.
// -----------------------------------------------------------------------------

export function gaussianFrom(rng: () => number): () => number {
  let cached: number | null = null;
  return function (): number {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = 0;
    let u2 = 0;
    while (u1 === 0) u1 = rng();
    while (u2 === 0) u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

// -----------------------------------------------------------------------------
// Gamma sampler — Marsaglia & Tsang (2000) acceptance/rejection. Handles all
// α > 0; for α < 1 we boost via the standard x = u^{1/α} trick on the α + 1
// draw. Density: f(x; α, β) = β^α x^{α-1} e^{-β x} / Γ(α).
// -----------------------------------------------------------------------------

export function gammaFrom(rng: () => number, alpha: number, beta: number): () => number {
  if (alpha <= 0 || beta <= 0) throw new Error(`gammaFrom: bad parameters α=${alpha}, β=${beta}`);
  const gauss = gaussianFrom(rng);
  const boost = alpha < 1;
  const a = boost ? alpha + 1 : alpha;
  const d = a - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  return function (): number {
    while (true) {
      let x: number;
      let v: number;
      do {
        x = gauss();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = rng();
      const x2 = x * x;
      if (u < 1 - 0.0331 * x2 * x2) {
        const sample = (d * v) / beta;
        if (boost) {
          const boostU = rng() || Number.MIN_VALUE;
          return sample * Math.pow(boostU, 1 / alpha);
        }
        return sample;
      }
      if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) {
        const sample = (d * v) / beta;
        if (boost) {
          const boostU = rng() || Number.MIN_VALUE;
          return sample * Math.pow(boostU, 1 / alpha);
        }
        return sample;
      }
    }
  };
}

// -----------------------------------------------------------------------------
// Digamma function ψ(x) = Γ'(x)/Γ(x) via the standard asymptotic expansion
// after shifting to x ≥ 6. Used by Gamma MLE (Newton iteration on the
// digamma equation) and by the verify-suite anchors.
// Accuracy: ~1e-10 on x ∈ [0.1, 100].
// -----------------------------------------------------------------------------

export function digamma(x: number): number {
  if (x <= 0) throw new Error(`digamma: requires x > 0, got ${x}`);
  let result = 0;
  let z = x;
  while (z < 6) {
    result -= 1 / z;
    z += 1;
  }
  // Asymptotic: ψ(z) ≈ log z - 1/(2z) - 1/(12z²) + 1/(120 z⁴) - 1/(252 z⁶)
  const invZ = 1 / z;
  const invZ2 = invZ * invZ;
  result += Math.log(z) - 0.5 * invZ
    - invZ2 * (1 / 12 - invZ2 * (1 / 120 - invZ2 / 252));
  return result;
}

// -----------------------------------------------------------------------------
// Gaussian MoM and MLE on (μ, σ²). Both estimators coincide (sufficient
// statistics are sample mean and biased sample variance). §2.2 anchor.
// -----------------------------------------------------------------------------

export interface GaussianFit {
  mu: number;
  sigma2: number; // biased estimator with divisor n
}

export function gaussianMoM(samples: Vec): GaussianFit {
  const n = samples.length;
  if (n === 0) return { mu: 0, sigma2: 0 };
  let sum = 0;
  for (let i = 0; i < n; i++) sum += samples[i];
  const mu = sum / n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mu;
    sumSq += d * d;
  }
  return { mu, sigma2: sumSq / n };
}

export const gaussianMLE = gaussianMoM;

// -----------------------------------------------------------------------------
// Gamma MoM (closed-form) and MLE (Newton on digamma equation). §2.2 anchor.
//
// MoM:  β̂ = x̄ / S²,   α̂ = x̄² / S²   where S² = (1/n) Σ (xᵢ - x̄)².
// MLE:  solve log α - ψ(α) = log x̄ - (1/n) Σ log xᵢ for α via Newton; then
//       β̂ = α̂ / x̄.
// -----------------------------------------------------------------------------

export interface GammaFit {
  alpha: number;
  beta: number;
}

export function gammaMoM(samples: Vec): GammaFit {
  const n = samples.length;
  if (n === 0) return { alpha: 0, beta: 0 };
  let sum = 0;
  for (let i = 0; i < n; i++) sum += samples[i];
  const mean = sum / n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mean;
    sumSq += d * d;
  }
  const variance = sumSq / n;
  if (variance <= 0) return { alpha: 0, beta: 0 };
  return { beta: mean / variance, alpha: (mean * mean) / variance };
}

// Trigamma ψ'(x) via the same shift-then-asymptotic trick, for Newton's
// step in the Gamma MLE.
export function trigamma(x: number): number {
  if (x <= 0) throw new Error(`trigamma: requires x > 0, got ${x}`);
  let result = 0;
  let z = x;
  while (z < 6) {
    result += 1 / (z * z);
    z += 1;
  }
  const invZ = 1 / z;
  const invZ2 = invZ * invZ;
  // ψ'(z) ≈ 1/z + 1/(2z²) + 1/(6 z³) - 1/(30 z⁵) + 1/(42 z⁷)
  result += invZ + 0.5 * invZ2 + invZ * invZ2 * (1 / 6 - invZ2 * (1 / 30 - invZ2 / 42));
  return result;
}

export function gammaMLE(samples: Vec, tol = 1e-10, maxIter = 50): GammaFit {
  const n = samples.length;
  if (n === 0) return { alpha: 0, beta: 0 };
  let sum = 0;
  let sumLog = 0;
  for (let i = 0; i < n; i++) {
    if (samples[i] <= 0) throw new Error(`gammaMLE: positive support required, sample[${i}] = ${samples[i]}`);
    sum += samples[i];
    sumLog += Math.log(samples[i]);
  }
  const mean = sum / n;
  const logMean = Math.log(mean);
  const meanLog = sumLog / n;
  const s = logMean - meanLog;
  // Initial estimate via Minka (2002).
  let alpha = (3 - s + Math.sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s);
  for (let iter = 0; iter < maxIter; iter++) {
    const f = Math.log(alpha) - digamma(alpha) - s;
    const fPrime = 1 / alpha - trigamma(alpha);
    const step = f / fPrime;
    alpha -= step;
    if (Math.abs(step) < tol * Math.abs(alpha)) break;
  }
  return { alpha, beta: alpha / mean };
}

// -----------------------------------------------------------------------------
// Asymptotic variances for §2 examples.
//
// Just-identified MoM variance: V = G⁻¹ Ω G⁻ᵀ where G = E[∂g/∂θᵀ] and
// Ω = E[g gᵀ] at θ₀. The two §2 worked examples have analytic closed forms.
// -----------------------------------------------------------------------------

// Gaussian MoM on (μ, σ²) — both MoM and MLE achieve V_MLE = diag(σ², 2σ⁴).
export function gaussianMoMVariance(sigma2: number): Mat {
  return [
    [sigma2, 0],
    [0, 2 * sigma2 * sigma2],
  ];
}

// Gamma MoM on (α, β) — derived from delta method on (x̄, S²) → (α̂, β̂).
// Var(α̂) = α(α+1) · (3α+2) / n;  Var(β̂) = β²(α+1)(2α+3) / (α·n);  these
// are the asymptotic values used by §2.3 verification.
// MLE: V_MLE has inverse Fisher information I⁻¹ where
//   I(α, β) = [[ ψ'(α),  -1/β ], [ -1/β,  α/β² ]].
// The asymptotic-relative-efficiency ARE = det(I⁻¹) / det(V_MoM) is < 1 for
// all α > 0, confirming MoM is inefficient relative to MLE on the Gamma.
export function gammaFisherInfo(alpha: number, beta: number): Mat {
  return [
    [trigamma(alpha), -1 / beta],
    [-1 / beta, alpha / (beta * beta)],
  ];
}

export function gammaMLEVariance(alpha: number, beta: number): Mat {
  return matInverse(gammaFisherInfo(alpha, beta));
}

// -----------------------------------------------------------------------------
// The §3-§8 running example design matrix. Four sensors, two parameters.
// E[X_1] = θ_1, E[X_2] = θ_2, E[X_3] = θ_1 + θ_2, E[X_4] = 2θ_1 - θ_2.
// Moment function: g(X, θ) = X - A θ ∈ R⁴.
// -----------------------------------------------------------------------------

export const RUNNING_EXAMPLE_A: Mat = [
  [1, 0],
  [0, 1],
  [1, 1],
  [2, -1],
];

export const RUNNING_EXAMPLE_THETA0: Vec = [1.0, 1.0];

// -----------------------------------------------------------------------------
// Affine GMM criterion and closed-form estimator for g(X, θ) = X - A θ.
//
// J_n(θ, W) = n · (X̄ - A θ)ᵀ W (X̄ - A θ).
// θ̂_W = (Aᵀ W A)⁻¹ Aᵀ W X̄  — generalized least squares.
//
// In the just-identified case L = k, A is k×k invertible and θ̂_W = A⁻¹ X̄
// (the weighting matrix drops out). In the over-identified case L > k, the
// weighting matrix shapes the projection.
// -----------------------------------------------------------------------------

export function gmmCriterion(theta: Vec, A: Mat, Xbar: Vec, W: Mat, n = 1): number {
  // Returns J_n; pass n to recover the n-scaled value, default n=1 returns
  // the population-scale criterion Q_n(θ, W) used in the contour viz.
  const residual = vecSub(Xbar, matVec(A, theta));
  return n * quadForm(W, residual);
}

export function gmmEstimateAffine(A: Mat, Xbar: Vec, W: Mat): Vec {
  // (Aᵀ W A)⁻¹ Aᵀ W X̄
  const At = matTranspose(A);
  const AtW = matMul(At, W);
  const AtWA = matMul(AtW, A);
  const AtWXbar = matVec(AtW, Xbar);
  const AtWAinv = matInverse(AtWA);
  return matVec(AtWAinv, AtWXbar);
}

// Gradient of J_n w.r.t. θ at the affine moment function:
//   ∇_θ J_n = -2 n Aᵀ W (X̄ - A θ).
// Useful for FOC sanity checks (the gradient should be machine-zero at θ̂_W).
export function gmmCriterionGradient(theta: Vec, A: Mat, Xbar: Vec, W: Mat, n = 1): Vec {
  const residual = vecSub(Xbar, matVec(A, theta));
  const At = matTranspose(A);
  const AtW = matMul(At, W);
  const out = matVec(AtW, residual);
  for (let i = 0; i < out.length; i++) out[i] *= -2 * n;
  return out;
}

// -----------------------------------------------------------------------------
// Sample-mean helper. Returns X̄ for either a flat array (one moment) or a
// matrix (rows = observations, columns = moments — what the running example
// uses with L = 4).
// -----------------------------------------------------------------------------

export function sampleMean(samples: Vec | Mat): Vec | number {
  if (samples.length === 0) return 0;
  if (typeof samples[0] === 'number') {
    const v = samples as Vec;
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i];
    return s / v.length;
  }
  const M = samples as Mat;
  const L = M[0].length;
  const out = zeros(L);
  for (let i = 0; i < M.length; i++) {
    const row = M[i];
    for (let j = 0; j < L; j++) out[j] += row[j];
  }
  for (let j = 0; j < L; j++) out[j] /= M.length;
  return out;
}

// -----------------------------------------------------------------------------
// Criterion-surface evaluation — sample J_n on a 2D grid in (θ_1, θ_2).
// Returns a row-major matrix Z[i][j] = J_n((θ_1[j], θ_2[i]), W) suitable
// for passing to d3-contour.
// -----------------------------------------------------------------------------

export interface CriterionGrid {
  theta1: Vec; // x-axis grid
  theta2: Vec; // y-axis grid
  J: Mat; // J[i][j] = J_n at (theta1[j], theta2[i])
  argmin: Vec; // optional reference minimum (theta_hat)
  minVal: number;
}

export function affineCriterionSurface(
  A: Mat,
  Xbar: Vec,
  W: Mat,
  theta1Bounds: [number, number],
  theta2Bounds: [number, number],
  resolution = 50,
  nScale = 1
): CriterionGrid {
  const theta1: Vec = [];
  const theta2: Vec = [];
  const dx = (theta1Bounds[1] - theta1Bounds[0]) / (resolution - 1);
  const dy = (theta2Bounds[1] - theta2Bounds[0]) / (resolution - 1);
  for (let i = 0; i < resolution; i++) {
    theta1.push(theta1Bounds[0] + i * dx);
    theta2.push(theta2Bounds[0] + i * dy);
  }
  const J: Mat = zerosMat(resolution, resolution);
  let argmin: Vec = [theta1[0], theta2[0]];
  let minVal = Infinity;
  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const v = gmmCriterion([theta1[j], theta2[i]], A, Xbar, W, nScale);
      J[i][j] = v;
      if (v < minVal) {
        minVal = v;
        argmin = [theta1[j], theta2[i]];
      }
    }
  }
  return { theta1, theta2, J, argmin, minVal };
}

// -----------------------------------------------------------------------------
// Identification rank check. For the running example the Jacobian is G = -A,
// so the rank-of-identification question reduces to rank(A). We return the
// numerically-relevant condition number (max eigenvalue / min eigenvalue of
// AᵀA) and the smallest eigenvalue (small = nearly rank-deficient).
// -----------------------------------------------------------------------------

export interface RankDiagnostic {
  eigvals: Vec; // sorted ascending — eigvals of AᵀA
  smallest: number;
  largest: number;
  conditionNumber: number;
  effectiveRank: number; // count of eigvals > (largest * 1e-12)
}

export function rankDiagnostic(A: Mat): RankDiagnostic {
  const At = matTranspose(A);
  const AtA = matMul(At, A);
  const raw = eigvalsSymmetric(AtA).slice().sort((a, b) => a - b);
  const largest = raw[raw.length - 1];
  const smallest = raw[0];
  const threshold = largest * 1e-12;
  let rank = 0;
  for (const e of raw) if (e > threshold) rank++;
  return {
    eigvals: raw,
    smallest,
    largest,
    conditionNumber: largest / Math.max(smallest, 1e-30),
    effectiveRank: rank,
  };
}

// -----------------------------------------------------------------------------
// Generate the running-example data. Each X_i ∈ R^4 is drawn as A θ_0 + ε_i
// where ε_i has independent Gaussian components with optional per-component
// noise σ_j. Returns the n × L data matrix.
// -----------------------------------------------------------------------------

export function sampleSensorObservations(
  rng: () => number,
  theta0: Vec,
  A: Mat,
  n: number,
  sigma: Vec = [1, 1, 1, 1]
): Mat {
  const L = A.length;
  if (sigma.length !== L) throw new Error(`sampleSensorObservations: σ length ${sigma.length} ≠ ${L}`);
  const gauss = gaussianFrom(rng);
  const mu = matVec(A, theta0); // length L
  const X: Mat = zerosMat(n, L);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < L; j++) {
      X[i][j] = mu[j] + sigma[j] * gauss();
    }
  }
  return X;
}

// =============================================================================
// Wave 2 — sandwich variance, Loewner ordering, efficient weighting, Wishart.
// =============================================================================

// -----------------------------------------------------------------------------
// Sandwich asymptotic variance V_W = (GᵀWG)⁻¹ GᵀWΩWG (GᵀWG)⁻¹.
// G is the population Jacobian E[∂g/∂θᵀ] (L × k), Ω the moment covariance
// (L × L), W the weighting (L × L). For the running example G = -A.
// -----------------------------------------------------------------------------

export function sandwichVariance(G: Mat, W: Mat, Omega: Mat): Mat {
  const Gt = matTranspose(G);
  const GtW = matMul(Gt, W); // k × L
  const bread = matMul(GtW, G); // k × k
  const breadInv = matInverse(bread);
  const meatRight = matMul(W, G); // L × k
  const meatMid = matMul(Omega, meatRight); // L × k
  const meat = matMul(GtW, meatMid); // k × k
  // V = breadInv · meat · breadInv
  return matMul(breadInv, matMul(meat, breadInv));
}

// Efficient weighting W* = Ω⁻¹ via Cholesky-backsolve.
export function efficientWeight(Omega: Mat): Mat {
  return matInverse(Omega);
}

// Hansen efficiency bound V* = (Gᵀ Ω⁻¹ G)⁻¹.
export function hansenBound(G: Mat, Omega: Mat): Mat {
  const OmegaInv = matInverse(Omega);
  const Gt = matTranspose(G);
  const GtOmegaInv = matMul(Gt, OmegaInv);
  return matInverse(matMul(GtOmegaInv, G));
}

// -----------------------------------------------------------------------------
// Loewner ordering check on symmetric matrices. Returns the qualitative
// answer (A ⪯ B, B ⪯ A, or incomparable) via eigenvalues of B - A and A - B.
// Tolerance accounts for floating-point noise: an eigenvalue within ε of zero
// counts as zero.
// -----------------------------------------------------------------------------

export type LoewnerResult = 'A_le_B' | 'B_le_A' | 'incomparable' | 'equal';

export function loewnerCompare(A: Mat, B: Mat, eps = 1e-10): LoewnerResult {
  const n = A.length;
  const diffBA: Mat = zerosMat(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) diffBA[i][j] = B[i][j] - A[i][j];
  const eigs = eigvalsSymmetric(diffBA);
  const minE = Math.min(...eigs);
  const maxE = Math.max(...eigs);
  if (Math.abs(minE) <= eps && Math.abs(maxE) <= eps) return 'equal';
  if (minE >= -eps) return 'A_le_B';
  if (maxE <= eps) return 'B_le_A';
  return 'incomparable';
}

// -----------------------------------------------------------------------------
// Wishart sampler — draws a random L × L positive-definite matrix from the
// Wishart(df, scale) distribution by sampling Y_i ~ N(0, scale) for i = 1..df
// and forming Σ Y_i Y_iᵀ. Used by the §6 efficiency-frontier viz to draw the
// scatter of random PD weighting matrices.
// -----------------------------------------------------------------------------

export function sampleWishart(rng: () => number, df: number, scale: Mat): Mat {
  const L = scale.length;
  const cholS = cholesky(scale);
  const gauss = gaussianFrom(rng);
  const W: Mat = zerosMat(L, L);
  for (let i = 0; i < df; i++) {
    // y = cholS · z, where z ~ N(0, I_L)
    const z = new Array(L);
    for (let j = 0; j < L; j++) z[j] = gauss();
    const y = matVec(cholS, z);
    for (let a = 0; a < L; a++) {
      for (let b = 0; b < L; b++) {
        W[a][b] += y[a] * y[b];
      }
    }
  }
  return W;
}

// -----------------------------------------------------------------------------
// 95% confidence ellipse parameters from a 2×2 covariance matrix V. Returns
// semi-major / semi-minor axes (in θ-space units, suitable for D3 ellipse
// drawing) and the rotation angle (radians) of the major axis from horizontal.
// At 95% confidence with k = 2 degrees of freedom, the scaling factor is
// √(χ²₂,₀.₉₅) ≈ √5.991 ≈ 2.4477.
// -----------------------------------------------------------------------------

export interface EllipseParams {
  rx: number; // half-major axis length (data units)
  ry: number; // half-minor axis length
  angle: number; // radians, CCW from horizontal
}

export function confidenceEllipse2D(V: Mat, alpha = 0.05, nForSqrtNScale = 1): EllipseParams {
  // χ²₂(1-α) critical values for the most common α.
  const chiCrit = alpha === 0.05 ? 5.991464547107981 : alpha === 0.01 ? 9.21034037197618 : alpha === 0.10 ? 4.605170185988091 : 5.991464547107981;
  const scale = Math.sqrt(chiCrit / nForSqrtNScale);
  const a = V[0][0];
  const b = V[1][1];
  const c = V[0][1];
  const trace = a + b;
  const det = a * b - c * c;
  const sqrtDisc = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
  const lam1 = trace / 2 + sqrtDisc; // larger eigenvalue
  const lam2 = trace / 2 - sqrtDisc; // smaller eigenvalue
  const rx = scale * Math.sqrt(Math.max(lam1, 0));
  const ry = scale * Math.sqrt(Math.max(lam2, 0));
  // Rotation angle of the major axis: eigenvector of V at lam1.
  const angle = Math.abs(c) < 1e-12 ? (a >= b ? 0 : Math.PI / 2) : Math.atan2(lam1 - a, c);
  return { rx, ry, angle };
}

// -----------------------------------------------------------------------------
// Operator (spectral) norm of a symmetric matrix = largest absolute eigenvalue.
// Used by SandwichDecomposer to show bread / meat / V_W magnitudes on bars.
// -----------------------------------------------------------------------------

export function operatorNorm(A: Mat): number {
  const eigs = eigvalsSymmetric(A);
  let m = 0;
  for (const e of eigs) {
    const a = Math.abs(e);
    if (a > m) m = a;
  }
  return m;
}

// Trace and determinant of a 2×2 matrix (used as scalar summaries for the
// efficiency-frontier scatter — det V is "uncertainty volume", tr V is
// "total uncertainty"). Generic determinant via LU isn't worth the code for
// 2×2; we keep this specialized.
export function trace2x2(A: Mat): number {
  return A[0][0] + A[1][1];
}

export function det2x2(A: Mat): number {
  return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}


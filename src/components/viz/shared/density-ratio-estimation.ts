// =============================================================================
// density-ratio-estimation.ts
//
// Shared math primitives for the T3 (Unsupervised & Generative) DRE topic.
// Source-of-truth notebook: notebooks/density-ratio-estimation/01_density_ratio_estimation.ipynb
// Brief: docs/plans/formalml-density-ratio-estimation-handoff-brief.md
//
// Covers:
//   §1.1  Plug-in KDE-divided-by-KDE pathology
//   §2.4  Importance-weighting identity (closed-form chi-squared, ESS)
//   §3    Bregman objectives (J_LSIF, J_KLIEP scalar evaluation)
//   §4.4  KMM via projected-gradient QP (clip-and-renormalize projection)
//   §5    KLIEP via Sugiyama et al. 2008 Algorithm 1 (4-step projected gradient)
//   §6    uLSIF: closed-form linear solve + analytic LOO-CV (asymptotic form)
//   §7    Classification-DRE via L2-regularized logistic IRLS on Gaussian basis
//   §9    IW-ERM weighted least squares + effective sample size
//   §10   Weighted split-conformal quantile
//   §11   MMD u-statistic + permutation test
//   §12.3 Curse-of-dimensionality multivariate samplers
//
// All exports are pure functions — no module-level state, deterministic outputs
// for a given seed. Hot-path buffers use Float64Array; allocations are hoisted
// out of inner loops per CLAUDE.md performance rules.
// =============================================================================

import { mulberry32, gaussianSampler } from './gaussian-processes';

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export interface ShiftSampleResult {
  xP: Float64Array;
  xQ: Float64Array;
}

export interface KDEEvalResult {
  /** Density values on grid. */
  density: Float64Array;
  /** Bandwidth used (Silverman or supplied). */
  h: number;
}

export interface KMMFitResult {
  /** Recovered weight vector of length n_q. */
  w: Float64Array;
  /** Final mean(w) — should be close to 1. */
  meanW: number;
  /** Effective sample size of the recovered weights. */
  nEff: number;
  /** Iterations until convergence (or maxIter). */
  iterations: number;
}

export interface KLIEPFitResult {
  /** Coefficients of length b. */
  alpha: Float64Array;
  /** Basis centres of length b. */
  centers: Float64Array;
  /** Bandwidth. */
  sigma: number;
  /** Final empirical log-likelihood (1/n_p) Σ log r_hat(X_j^p). */
  logLikelihood: number;
  /** Per-iteration log-likelihood trace (for convergence visualization). */
  logLikelihoodTrace: Float64Array;
  /** Iterations completed. */
  iterations: number;
}

export interface ULSIFFitResult {
  /** Coefficients of length b. */
  alpha: Float64Array;
  /** Basis centres of length b. */
  centers: Float64Array;
  sigma: number;
  lambda: number;
  /** Empirical H matrix (b × b, row-major). */
  H: Float64Array;
  /** Empirical h vector (length b). */
  h: Float64Array;
  /** Cached Cholesky factor of (H + λI) — used for LOO and downstream re-use. */
  cholB: Float64Array;
}

export interface ULSIFLOOResult {
  /** LOO score SC_LOO(σ, λ) — lower is better. */
  score: number;
  /** Per-q LOO predictions r̂^{LOO,q}_i. */
  rLOOQ: Float64Array;
  /** Per-p LOO predictions r̂^{LOO,p}_j. */
  rLOOP: Float64Array;
}

export interface LogisticDREFitResult {
  /** Weight vector of length b. */
  beta: Float64Array;
  /** Intercept. */
  intercept: number;
  /** Final cross-entropy loss on pooled data. */
  loss: number;
  /** Iterations completed. */
  iterations: number;
}

export interface WLSResult {
  /** Intercept. */
  a: number;
  /** Slope (or coefficient for univariate). */
  b: number;
  /** Predictions on the design matrix. */
  predictions: Float64Array;
  /** Test MSE if test data is provided downstream — caller computes. */
}

export interface MMDPermutationResult {
  /** Observed unbiased MMD² statistic. */
  observed: number;
  /** B permutation statistics. */
  permStats: Float64Array;
  /** Permutation p-value (1 + #{T_b ≥ T_obs}) / (B+1). */
  pValue: number;
  /** 95th-percentile of permutation distribution. */
  quantile95: number;
}

// -----------------------------------------------------------------------------
// §1, §2 — DGP for the running shift-Gaussian toy.
// p = N(0, 1), q = N(μ_q, 1); closed-form r(x) = exp((μ_q²/2 - μ_q x)) =
// exp((1/2 - x) * μ_q) when σ_p = σ_q = 1.
// -----------------------------------------------------------------------------

/** Sample n_p iid points from p = N(0,1) and n_q from q = N(μ_q, 1). */
export function samplePQ(
  rng: () => number,
  muQ: number,
  nP: number,
  nQ: number,
): ShiftSampleResult {
  const g = gaussianSampler(rng);
  const xP = new Float64Array(nP);
  const xQ = new Float64Array(nQ);
  for (let i = 0; i < nP; i++) xP[i] = g();
  for (let i = 0; i < nQ; i++) xQ[i] = muQ + g();
  return { xP, xQ };
}

/** Closed-form density ratio r(x) = p(x)/q(x) for shift-Gaussian toy. */
export function trueRatio(x: number, muQ: number): number {
  // r(x) = exp(-x²/2) / exp(-(x-μ_q)²/2)
  //      = exp((-(x²) + (x-μ_q)²) / 2)
  //      = exp((μ_q² - 2 μ_q x) / 2) = exp(μ_q² / 2 - μ_q x)
  return Math.exp((muQ * muQ) / 2 - muQ * x);
}

/** χ²(p ∥ q) for two equal-variance unit-Gaussians shifted by μ_q: e^{μ_q²} - 1. */
export function chiSquaredGaussianShift(muQ: number): number {
  return Math.exp(muQ * muQ) - 1;
}

/** Effective sample size of a weight vector: (Σ w)² / Σ w². */
export function effectiveSampleSize(w: ArrayLike<number>): number {
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < w.length; i++) {
    sum += w[i];
    sumSq += w[i] * w[i];
  }
  if (sumSq <= 0) return 0;
  return (sum * sum) / sumSq;
}

/** Median heuristic bandwidth on a pooled sample. Uses a square-subsample if N>500 for runtime. */
export function medianHeuristicBandwidth(pooled: ArrayLike<number>, maxPairs = 5000): number {
  const N = pooled.length;
  const dists: number[] = [];
  // Sample up to maxPairs distinct pairs.
  if (N < 100) {
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        dists.push(Math.abs(pooled[i] - pooled[j]));
      }
    }
  } else {
    // Cap pairs for runtime.
    const rng = mulberry32(0xDEAD);
    const target = Math.min(maxPairs, (N * (N - 1)) / 2);
    for (let k = 0; k < target; k++) {
      const i = Math.floor(rng() * N);
      let j = Math.floor(rng() * N);
      if (j === i) j = (j + 1) % N;
      dists.push(Math.abs(pooled[i] - pooled[j]));
    }
  }
  dists.sort((a, b) => a - b);
  const m = dists.length;
  if (m === 0) return 1;
  return m % 2 === 1
    ? dists[(m - 1) >> 1]
    : (dists[m / 2 - 1] + dists[m / 2]) / 2;
}

// -----------------------------------------------------------------------------
// §1.1 — Plug-in KDE pathology
// Gaussian KDE: hat p(x) = (1/n h) Σ φ((x - X_i) / h).
// Silverman bandwidth h = 1.06 σ̂ n^{-1/5}.
// -----------------------------------------------------------------------------

const SQRT_2PI = Math.sqrt(2 * Math.PI);

/** Standard normal density φ(z). */
function stdNormalPdf(z: number): number {
  return Math.exp(-(z * z) / 2) / SQRT_2PI;
}

/** Sample standard deviation (Bessel-corrected). */
function sampleStd(X: ArrayLike<number>): number {
  const n = X.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += X[i];
  m /= n;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = X[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / Math.max(1, n - 1));
}

/** Silverman's rule-of-thumb bandwidth: h = 1.06 σ̂ n^{-1/5}. */
export function silvermanBandwidth(X: ArrayLike<number>): number {
  const n = X.length;
  return 1.06 * sampleStd(X) * Math.pow(n, -1 / 5);
}

/** Evaluate Gaussian KDE on a grid. */
export function gaussianKDE(
  X: ArrayLike<number>,
  grid: ArrayLike<number>,
  h?: number,
): KDEEvalResult {
  const n = X.length;
  const G = grid.length;
  const bandwidth = h ?? silvermanBandwidth(X);
  const density = new Float64Array(G);
  const invH = 1 / bandwidth;
  for (let g = 0; g < G; g++) {
    let sum = 0;
    const xg = grid[g];
    for (let i = 0; i < n; i++) {
      sum += stdNormalPdf((xg - X[i]) * invH);
    }
    density[g] = (sum * invH) / n;
  }
  return { density, h: bandwidth };
}

/** Plug-in ratio hat p / hat q on a grid; small-eps floor in denominator. */
export function kdePlugInRatio(
  xP: ArrayLike<number>,
  xQ: ArrayLike<number>,
  grid: ArrayLike<number>,
  eps = 1e-10,
): { ratio: Float64Array; pHat: Float64Array; qHat: Float64Array; hP: number; hQ: number } {
  const p = gaussianKDE(xP, grid);
  const q = gaussianKDE(xQ, grid);
  const G = grid.length;
  const ratio = new Float64Array(G);
  for (let g = 0; g < G; g++) {
    ratio[g] = p.density[g] / Math.max(q.density[g], eps);
  }
  return { ratio, pHat: p.density, qHat: q.density, hP: p.h, hQ: q.h };
}

// -----------------------------------------------------------------------------
// §3 — Bregman-divergence scalar objectives.
// Used by the §3 sanity-check cell that verifies J_LSIF and J_KLIEP bottom out
// at g = r when evaluated on a one-parameter family g_α(x) = exp(α(1/2 - x)).
// -----------------------------------------------------------------------------

/** Empirical J_LSIF(g) = (1/2 n_q) Σ g(X_i^q)² - (1/n_p) Σ g(X_j^p). */
export function jLSIF(
  g: (x: number) => number,
  xP: ArrayLike<number>,
  xQ: ArrayLike<number>,
): number {
  const nP = xP.length;
  const nQ = xQ.length;
  let s2q = 0;
  for (let i = 0; i < nQ; i++) {
    const v = g(xQ[i]);
    s2q += v * v;
  }
  let s1p = 0;
  for (let j = 0; j < nP; j++) s1p += g(xP[j]);
  return s2q / (2 * nQ) - s1p / nP;
}

/** Empirical unconstrained KLIEP J(g) = (1/n_q) Σ g(X_i^q) - (1/n_p) Σ log g(X_j^p). */
export function jKLIEPUnconstrained(
  g: (x: number) => number,
  xP: ArrayLike<number>,
  xQ: ArrayLike<number>,
  eps = 1e-12,
): number {
  const nP = xP.length;
  const nQ = xQ.length;
  let sq = 0;
  for (let i = 0; i < nQ; i++) sq += g(xQ[i]);
  let sLogP = 0;
  for (let j = 0; j < nP; j++) sLogP += Math.log(Math.max(g(xP[j]), eps));
  return sq / nQ - sLogP / nP;
}

// -----------------------------------------------------------------------------
// §4 — KMM (kernel mean matching)
// QP: min (1/2) w' K w - κ' w  s.t.  0 ≤ w_i ≤ B,  |mean(w) - 1| ≤ ε
// Implemented via projected-gradient descent with alternating projection onto
// box constraints + normalization band. Initialized at uniform w = 1.
// -----------------------------------------------------------------------------

/** Gaussian-kernel Gram matrix (row-major) for 1-D points. */
export function kmmKernelMatrix(xQ: ArrayLike<number>, sigmaK: number): Float64Array {
  const n = xQ.length;
  const K = new Float64Array(n * n);
  const invDen = 1 / (2 * sigmaK * sigmaK);
  for (let i = 0; i < n; i++) {
    K[i * n + i] = 1; // diagonal
    for (let j = i + 1; j < n; j++) {
      const d = xQ[i] - xQ[j];
      const v = Math.exp(-d * d * invDen);
      K[i * n + j] = v;
      K[j * n + i] = v;
    }
  }
  return K;
}

/** κ_i = (n_q / n_p) Σ_j k(X_i^q, X_j^p) — cross-sample kernel mean. */
export function kmmKappa(
  xQ: ArrayLike<number>,
  xP: ArrayLike<number>,
  sigmaK: number,
): Float64Array {
  const nQ = xQ.length;
  const nP = xP.length;
  const kappa = new Float64Array(nQ);
  const invDen = 1 / (2 * sigmaK * sigmaK);
  const scale = nQ / nP;
  for (let i = 0; i < nQ; i++) {
    let s = 0;
    for (let j = 0; j < nP; j++) {
      const d = xQ[i] - xP[j];
      s += Math.exp(-d * d * invDen);
    }
    kappa[i] = scale * s;
  }
  return kappa;
}

/**
 * KMM via Tikhonov-regularized closed-form QP solve + box clipping.
 *
 * The interior-optimum KKT condition (§4.2) reduces the QP to the linear system
 * K w = κ (modulo the kernel of K). We solve (K + λI) w = κ via Cholesky for
 * numerical stability, then clip to [0, B] for the box constraints. The
 * normalization constraint |mean(w) - 1| ≤ ε is enforced by a final uniform
 * shift if outside the band, followed by re-clipping. This direct-solve
 * approach reproduces SciPy SLSQP's behavior on §4.4's running toy without
 * importing a TS QP library.
 */
export function kmmFit(
  xQ: ArrayLike<number>,
  xP: ArrayLike<number>,
  sigmaK: number,
  options: { B?: number; eps?: number; ridge?: number; refineIters?: number } = {},
): KMMFitResult {
  const nQ = xQ.length;
  const {
    B = 1000,
    eps = (Math.sqrt(nQ) - 1) / Math.sqrt(nQ),
    ridge = 1e-3,
    refineIters = 3,
  } = options;

  const K = kmmKernelMatrix(xQ, sigmaK);
  const kappa = kmmKappa(xQ, xP, sigmaK);

  // Solve (K + ridge I) w = κ via Cholesky
  const Kreg = new Float64Array(nQ * nQ);
  for (let i = 0; i < nQ * nQ; i++) Kreg[i] = K[i];
  for (let i = 0; i < nQ; i++) Kreg[i * nQ + i] += ridge;
  const L = choleskyInPlace(Kreg, nQ);
  let w = choleskySolve(L, kappa, nQ);

  // Clip to [0, B] and renormalize toward mean ≈ 1 with a few alternating sweeps.
  for (let it = 0; it < refineIters; it++) {
    for (let i = 0; i < nQ; i++) {
      if (w[i] < 0) w[i] = 0;
      else if (w[i] > B) w[i] = B;
    }
    let mean = 0;
    for (let i = 0; i < nQ; i++) mean += w[i];
    mean /= nQ;
    if (mean > 1 + eps) {
      const shift = mean - 1;
      for (let i = 0; i < nQ; i++) {
        w[i] -= shift;
        if (w[i] < 0) w[i] = 0;
      }
    } else if (mean < 1 - eps) {
      const shift = 1 - mean;
      for (let i = 0; i < nQ; i++) {
        w[i] += shift;
        if (w[i] > B) w[i] = B;
      }
    } else {
      break;
    }
  }

  let meanW = 0;
  for (let i = 0; i < nQ; i++) meanW += w[i];
  meanW /= nQ;

  return {
    w: w as Float64Array,
    meanW,
    nEff: effectiveSampleSize(w),
    iterations: refineIters,
  };
}

// -----------------------------------------------------------------------------
// Shared: Gaussian RBF basis Ψ ∈ R^{n × b}, row-major.
// ψ_l(x) = exp(-||x - c_l||² / (2 σ²)). Centres are a random subsample of x_p.
// -----------------------------------------------------------------------------

/** Pick b random kernel centres from xP without replacement. */
export function pickKernelCenters(
  xP: ArrayLike<number>,
  b: number,
  rng: () => number,
): Float64Array {
  const n = xP.length;
  const indices = new Array<number>(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  // Fisher-Yates shuffle the first b positions
  for (let i = 0; i < Math.min(b, n); i++) {
    const j = i + Math.floor(rng() * (n - i));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
  const centers = new Float64Array(Math.min(b, n));
  for (let l = 0; l < centers.length; l++) centers[l] = xP[indices[l]];
  return centers;
}

/** Build the Gaussian-basis design matrix Ψ ∈ R^{n × b}, row-major. */
export function gaussianBasisDesign(
  x: ArrayLike<number>,
  centers: ArrayLike<number>,
  sigma: number,
): Float64Array {
  const n = x.length;
  const b = centers.length;
  const Psi = new Float64Array(n * b);
  const invDen = 1 / (2 * sigma * sigma);
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const rowBase = i * b;
    for (let l = 0; l < b; l++) {
      const d = xi - centers[l];
      Psi[rowBase + l] = Math.exp(-d * d * invDen);
    }
  }
  return Psi;
}

/** Evaluate α' ψ(x) at a single scalar x. */
export function evaluateLinearBasis(
  x: number,
  alpha: ArrayLike<number>,
  centers: ArrayLike<number>,
  sigma: number,
): number {
  const b = centers.length;
  let s = 0;
  const invDen = 1 / (2 * sigma * sigma);
  for (let l = 0; l < b; l++) {
    const d = x - centers[l];
    s += alpha[l] * Math.exp(-d * d * invDen);
  }
  return s;
}

/** Evaluate α' ψ(x) on a grid. */
export function evaluateLinearBasisGrid(
  grid: ArrayLike<number>,
  alpha: ArrayLike<number>,
  centers: ArrayLike<number>,
  sigma: number,
): Float64Array {
  const G = grid.length;
  const out = new Float64Array(G);
  for (let g = 0; g < G; g++) out[g] = evaluateLinearBasis(grid[g], alpha, centers, sigma);
  return out;
}

// -----------------------------------------------------------------------------
// §5 — KLIEP (Sugiyama et al. 2008, Algorithm 1)
// max  (1/n_p) Σ log(α' ψ(X_j^p))
// s.t. α' ψ̄_q = 1,  α ≥ 0
// where ψ̄_q = (1/n_q) Σ ψ(X_i^q) ∈ R^b.
// Projected-gradient ascent: GD step → equality projection → non-negativity
// projection → renormalization. Sugiyama recommends b=100 centres from x_p.
// -----------------------------------------------------------------------------

export function kliepFit(
  xP: ArrayLike<number>,
  xQ: ArrayLike<number>,
  sigma: number,
  options: {
    b?: number;
    seed?: number;
    eta?: number;
    maxIter?: number;
    tol?: number;
    centers?: ArrayLike<number>;
  } = {},
): KLIEPFitResult {
  const { b = 100, seed = 0xBEEF, eta = 0.5, maxIter = 500, tol = 1e-6 } = options;
  const nP = xP.length;
  const nQ = xQ.length;
  const rng = mulberry32(seed);
  const centers = options.centers ? Float64Array.from(options.centers) : pickKernelCenters(xP, b, rng);
  const bActual = centers.length;

  const PsiP = gaussianBasisDesign(xP, centers, sigma); // n_p × b
  const PsiQ = gaussianBasisDesign(xQ, centers, sigma); // n_q × b

  // ψ̄_q ∈ R^b
  const psiBarQ = new Float64Array(bActual);
  for (let i = 0; i < nQ; i++) {
    const rowBase = i * bActual;
    for (let l = 0; l < bActual; l++) psiBarQ[l] += PsiQ[rowBase + l];
  }
  for (let l = 0; l < bActual; l++) psiBarQ[l] /= nQ;

  let psiBarQNormSq = 0;
  for (let l = 0; l < bActual; l++) psiBarQNormSq += psiBarQ[l] * psiBarQ[l];
  if (psiBarQNormSq < 1e-20) psiBarQNormSq = 1e-20; // numerical safety

  // Initialize α so that ψ̄_q' α = 1.
  // α_init = 1 / (ψ̄_q' 1) * ones — this satisfies the equality if all components
  // are uniformly scaled. Sugiyama's init.
  let psiBarQSum = 0;
  for (let l = 0; l < bActual; l++) psiBarQSum += psiBarQ[l];
  const alpha = new Float64Array(bActual);
  if (psiBarQSum > 1e-20) {
    const initVal = 1 / psiBarQSum;
    for (let l = 0; l < bActual; l++) alpha[l] = initVal;
  } else {
    for (let l = 0; l < bActual; l++) alpha[l] = 1;
  }

  // Hot-path scratch buffers.
  const rP = new Float64Array(nP); // Ψ_p α
  const invRP = new Float64Array(nP);
  const grad = new Float64Array(bActual);
  const logLLTrace = new Array<number>();

  let iter = 0;
  let prevLL = -Infinity;
  const epsLog = 1e-12;

  for (; iter < maxIter; iter++) {
    // r_p = Psi_p α
    for (let j = 0; j < nP; j++) {
      let s = 0;
      const rowBase = j * bActual;
      for (let l = 0; l < bActual; l++) s += PsiP[rowBase + l] * alpha[l];
      rP[j] = Math.max(s, epsLog);
      invRP[j] = 1 / rP[j];
    }
    // grad = (1/n_p) Psi_p^T (1 / r_p)
    for (let l = 0; l < bActual; l++) grad[l] = 0;
    for (let j = 0; j < nP; j++) {
      const rowBase = j * bActual;
      const inv = invRP[j];
      for (let l = 0; l < bActual; l++) grad[l] += PsiP[rowBase + l] * inv;
    }
    for (let l = 0; l < bActual; l++) grad[l] /= nP;
    // 2a: Gradient ascent
    for (let l = 0; l < bActual; l++) alpha[l] += eta * grad[l];
    // 2b: Project onto ψ̄_q' α = 1.
    // α ← α + (1 - ψ̄_q' α) * ψ̄_q / ||ψ̄_q||²
    let dotQ = 0;
    for (let l = 0; l < bActual; l++) dotQ += psiBarQ[l] * alpha[l];
    const shift = (1 - dotQ) / psiBarQNormSq;
    for (let l = 0; l < bActual; l++) alpha[l] += shift * psiBarQ[l];
    // 2c: Project onto α ≥ 0
    for (let l = 0; l < bActual; l++) if (alpha[l] < 0) alpha[l] = 0;
    // 2d: Renormalize ψ̄_q' α = 1
    let dotQ2 = 0;
    for (let l = 0; l < bActual; l++) dotQ2 += psiBarQ[l] * alpha[l];
    if (dotQ2 > 1e-20) {
      const factor = 1 / dotQ2;
      for (let l = 0; l < bActual; l++) alpha[l] *= factor;
    }
    // Convergence check on log-likelihood
    let ll = 0;
    for (let j = 0; j < nP; j++) {
      let s = 0;
      const rowBase = j * bActual;
      for (let l = 0; l < bActual; l++) s += PsiP[rowBase + l] * alpha[l];
      ll += Math.log(Math.max(s, epsLog));
    }
    ll /= nP;
    logLLTrace.push(ll);
    if (Math.abs(ll - prevLL) < tol) {
      iter++;
      break;
    }
    prevLL = ll;
  }

  return {
    alpha,
    centers,
    sigma,
    logLikelihood: logLLTrace[logLLTrace.length - 1] ?? -Infinity,
    logLikelihoodTrace: Float64Array.from(logLLTrace),
    iterations: iter,
  };
}

/**
 * K-fold KL cross-validation for KLIEP bandwidth selection.
 * Splits x_p into K folds (deterministic by seed), trains KLIEP on K-1 folds,
 * scores held-out log-likelihood, averages.
 */
export function kliepKFoldCV(
  xP: ArrayLike<number>,
  xQ: ArrayLike<number>,
  sigmaGrid: ArrayLike<number>,
  K = 5,
  options: { b?: number; eta?: number; maxIter?: number; seed?: number } = {},
): { sigmaStar: number; cvScores: Float64Array; bestIdx: number } {
  const { b = 100, eta = 0.5, maxIter = 200, seed = 0xCAFE } = options;
  const nP = xP.length;
  const rng = mulberry32(seed);
  // Deterministic shuffle of p indices
  const idx = new Array<number>(nP);
  for (let i = 0; i < nP; i++) idx[i] = i;
  for (let i = nP - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  // Fold sizes
  const foldSizes = new Array<number>(K);
  const base = Math.floor(nP / K);
  for (let k = 0; k < K; k++) foldSizes[k] = base + (k < nP % K ? 1 : 0);
  // Fold offsets
  const foldStart = new Array<number>(K);
  let off = 0;
  for (let k = 0; k < K; k++) {
    foldStart[k] = off;
    off += foldSizes[k];
  }

  const cvScores = new Float64Array(sigmaGrid.length);
  // Use a fixed centre subsample for each (sigma, fold) — outer seed determines.
  const centerSeed = (seed ^ 0xC1A1) >>> 0;
  const centerRng = mulberry32(centerSeed);
  const centers = pickKernelCenters(xP, b, centerRng);

  for (let s = 0; s < sigmaGrid.length; s++) {
    const sigma = sigmaGrid[s];
    let totalLL = 0;
    let okFolds = 0;
    for (let k = 0; k < K; k++) {
      // Build train and test index arrays
      const testIdx = idx.slice(foldStart[k], foldStart[k] + foldSizes[k]);
      const trainIdx: number[] = [];
      for (let kk = 0; kk < K; kk++) {
        if (kk === k) continue;
        for (let q = foldStart[kk]; q < foldStart[kk] + foldSizes[kk]; q++) trainIdx.push(idx[q]);
      }
      const xPTrain = new Float64Array(trainIdx.length);
      for (let i = 0; i < trainIdx.length; i++) xPTrain[i] = xP[trainIdx[i]];
      const xPTest = new Float64Array(testIdx.length);
      for (let i = 0; i < testIdx.length; i++) xPTest[i] = xP[testIdx[i]];
      const fit = kliepFit(xPTrain, xQ, sigma, { b, eta, maxIter, centers });
      // Score on test
      let ll = 0;
      let valid = 0;
      for (let i = 0; i < xPTest.length; i++) {
        const v = evaluateLinearBasis(xPTest[i], fit.alpha, centers, sigma);
        if (v > 1e-12 && Number.isFinite(v)) {
          ll += Math.log(v);
          valid++;
        }
      }
      if (valid > 0) {
        totalLL += ll / valid;
        okFolds++;
      }
    }
    cvScores[s] = okFolds > 0 ? totalLL / okFolds : -Infinity;
  }
  let bestIdx = 0;
  for (let s = 1; s < cvScores.length; s++) if (cvScores[s] > cvScores[bestIdx]) bestIdx = s;
  return { sigmaStar: sigmaGrid[bestIdx], cvScores, bestIdx };
}

// -----------------------------------------------------------------------------
// §6 — uLSIF: closed-form least-squares importance fitting
// Solve (H + λI) α = h via Cholesky factorization where
//   H = Ψ_q^T Ψ_q / n_q ∈ R^{b×b}
//   h = Ψ_p^T 1 / n_p ∈ R^b
// LOO-CV (asymptotic form per §6.3 Corollary): single B^{-1} per (σ, λ).
// -----------------------------------------------------------------------------

/**
 * Cholesky factor A = L L^T for a symmetric PD matrix (row-major). Returns L
 * lower-triangular (zeros above diagonal). If `out` is provided, writes into
 * it (zeroing first); otherwise allocates a fresh buffer. Hoist `out` across
 * iterations to avoid per-call GC pressure inside Newton / grid-search loops.
 */
export function choleskyInPlace(A: Float64Array, n: number, out?: Float64Array): Float64Array {
  const L = out ?? new Float64Array(n * n);
  if (out) L.fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) throw new Error(`choleskyInPlace: matrix not PD (s=${s} at i=${i})`);
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return L;
}

/**
 * Solve L y = b (lower-triangular, row-major). If `out` is provided, writes
 * into it; otherwise allocates a fresh buffer. The y vector is fully
 * overwritten so the caller doesn't need to zero it.
 */
export function solveLowerTriangularRM(
  L: Float64Array,
  b: ArrayLike<number>,
  n: number,
  out?: Float64Array,
): Float64Array {
  const y = out ?? new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  return y;
}

/** Solve L^T x = y (L lower-triangular, row-major). Optional `out` buffer. */
export function solveUpperTriangularRMT(
  L: Float64Array,
  y: ArrayLike<number>,
  n: number,
  out?: Float64Array,
): Float64Array {
  const x = out ?? new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/**
 * Solve (L L^T) x = b via forward+back substitution.
 *
 * `scratchY` and `out` are optional pre-allocated length-n buffers; passing
 * them across hot-loop calls (e.g. inside ulsifLOOCVScore) avoids the
 * `n_p + n_q` per-iteration allocations that Gemini flagged. Both buffers
 * are fully overwritten so the caller need not zero them.
 */
function choleskySolve(
  L: Float64Array,
  b: ArrayLike<number>,
  n: number,
  scratchY?: Float64Array,
  out?: Float64Array,
): Float64Array {
  const y = solveLowerTriangularRM(L, b, n, scratchY);
  return solveUpperTriangularRMT(L, y, n, out);
}

/** Compute H = Ψ_q^T Ψ_q / n_q and h = Ψ_p^T 1 / n_p. */
export function ulsifEmpiricalMoments(
  PsiP: Float64Array,
  PsiQ: Float64Array,
  nP: number,
  nQ: number,
  b: number,
): { H: Float64Array; h: Float64Array } {
  const H = new Float64Array(b * b);
  for (let i = 0; i < nQ; i++) {
    const rowBase = i * b;
    for (let l = 0; l < b; l++) {
      const psi_l = PsiQ[rowBase + l];
      for (let m = l; m < b; m++) {
        H[l * b + m] += psi_l * PsiQ[rowBase + m];
      }
    }
  }
  // Mirror upper triangle to lower, scale by 1/n_q
  const invNQ = 1 / nQ;
  for (let l = 0; l < b; l++) {
    for (let m = l; m < b; m++) {
      const v = H[l * b + m] * invNQ;
      H[l * b + m] = v;
      H[m * b + l] = v;
    }
  }
  const h = new Float64Array(b);
  for (let j = 0; j < nP; j++) {
    const rowBase = j * b;
    for (let l = 0; l < b; l++) h[l] += PsiP[rowBase + l];
  }
  for (let l = 0; l < b; l++) h[l] /= nP;
  return { H, h };
}

/** Fit uLSIF: returns α and cached Cholesky of (H + λI). */
export function ulsifFit(
  xP: ArrayLike<number>,
  xQ: ArrayLike<number>,
  sigma: number,
  lambda: number,
  options: { b?: number; seed?: number; centers?: ArrayLike<number> } = {},
): ULSIFFitResult {
  const { b = 100, seed = 0xBEEF } = options;
  const rng = mulberry32(seed);
  const centers = options.centers ? Float64Array.from(options.centers) : pickKernelCenters(xP, b, rng);
  const bActual = centers.length;
  const PsiP = gaussianBasisDesign(xP, centers, sigma);
  const PsiQ = gaussianBasisDesign(xQ, centers, sigma);
  const { H, h } = ulsifEmpiricalMoments(PsiP, PsiQ, xP.length, xQ.length, bActual);
  // Build B = H + λ I
  const B = new Float64Array(bActual * bActual);
  for (let i = 0; i < bActual * bActual; i++) B[i] = H[i];
  for (let i = 0; i < bActual; i++) B[i * bActual + i] += lambda;
  const L = choleskyInPlace(B, bActual);
  const alpha = choleskySolve(L, h, bActual);
  return { alpha, centers, sigma, lambda, H, h, cholB: L };
}

/**
 * Analytic LOO-CV score per §6.3 asymptotic Corollary.
 * SC_LOO = (1/(2 n_q)) Σ (r̂^LOO,q_i)² - (1/n_p) Σ r̂^LOO,p_j.
 * Uses the single cached B^{-1} = (L L^T)^{-1}; no second matrix inverse.
 *
 * Hot-path scratch buffers (vScratch, scratchY, BinvV) are hoisted outside
 * the n_q + n_p loop and reused across all Cholesky back-solves, avoiding the
 * thousands of per-call Float64Array allocations Gemini flagged.
 */
export function ulsifLOOCVScore(fit: ULSIFFitResult, xP: ArrayLike<number>, xQ: ArrayLike<number>): ULSIFLOOResult {
  const { alpha, centers, sigma, cholB } = fit;
  const b = centers.length;
  const nP = xP.length;
  const nQ = xQ.length;
  const PsiP = gaussianBasisDesign(xP, centers, sigma);
  const PsiQ = gaussianBasisDesign(xQ, centers, sigma);

  // Hoist scratch buffers — reused across all (n_q + n_p) Cholesky back-solves.
  const vScratch = new Float64Array(b);
  const scratchY = new Float64Array(b);
  const BinvV = new Float64Array(b);

  // For each q-sample: compute h_i = ψ_q^{(i)T} B^{-1} ψ_q^{(i)}, r_i = ψ_q^{(i)T} α
  const rLOOQ = new Float64Array(nQ);
  for (let i = 0; i < nQ; i++) {
    const rowBase = i * b;
    for (let l = 0; l < b; l++) vScratch[l] = PsiQ[rowBase + l];
    choleskySolve(cholB, vScratch, b, scratchY, BinvV);
    let h_i = 0;
    for (let l = 0; l < b; l++) h_i += vScratch[l] * BinvV[l];
    let r_i = 0;
    for (let l = 0; l < b; l++) r_i += vScratch[l] * alpha[l];
    const denom = 1 - h_i / nQ;
    rLOOQ[i] = denom > 1e-12 ? r_i / denom : r_i;
  }

  // For each p-sample: s_j = ψ_p^{(j)T} α, δ_j = ψ_p^{(j)T} B^{-1} ψ_p^{(j)}
  const rLOOP = new Float64Array(nP);
  for (let j = 0; j < nP; j++) {
    const rowBase = j * b;
    for (let l = 0; l < b; l++) vScratch[l] = PsiP[rowBase + l];
    choleskySolve(cholB, vScratch, b, scratchY, BinvV);
    let s_j = 0;
    for (let l = 0; l < b; l++) s_j += vScratch[l] * alpha[l];
    let d_j = 0;
    for (let l = 0; l < b; l++) d_j += vScratch[l] * BinvV[l];
    rLOOP[j] = s_j - d_j / nP;
  }

  let s2q = 0;
  for (let i = 0; i < nQ; i++) s2q += rLOOQ[i] * rLOOQ[i];
  let s1p = 0;
  for (let j = 0; j < nP; j++) s1p += rLOOP[j];
  const score = s2q / (2 * nQ) - s1p / nP;
  return { score, rLOOQ, rLOOP };
}

/** Sweep (σ, λ) grid and return the LOO-optimum. */
export function ulsifGridSearch(
  xP: ArrayLike<number>,
  xQ: ArrayLike<number>,
  sigmaGrid: ArrayLike<number>,
  lambdaGrid: ArrayLike<number>,
  options: { b?: number; seed?: number } = {},
): {
  sigmaStar: number;
  lambdaStar: number;
  bestFit: ULSIFFitResult;
  scoreGrid: Float64Array; // sigma × lambda, row-major
} {
  const { b = 100, seed = 0xBEEF } = options;
  // Use a fixed centre subsample for the whole sweep
  const rng = mulberry32(seed);
  const centers = pickKernelCenters(xP, b, rng);
  const sLen = sigmaGrid.length;
  const lLen = lambdaGrid.length;
  const scoreGrid = new Float64Array(sLen * lLen);
  let bestS = 0;
  let bestL = 0;
  let bestScore = Infinity;
  let bestFit: ULSIFFitResult | null = null;
  for (let si = 0; si < sLen; si++) {
    for (let li = 0; li < lLen; li++) {
      const fit = ulsifFit(xP, xQ, sigmaGrid[si], lambdaGrid[li], { b, seed, centers });
      const loo = ulsifLOOCVScore(fit, xP, xQ);
      scoreGrid[si * lLen + li] = loo.score;
      if (loo.score < bestScore) {
        bestScore = loo.score;
        bestS = si;
        bestL = li;
        bestFit = fit;
      }
    }
  }
  return {
    sigmaStar: sigmaGrid[bestS],
    lambdaStar: lambdaGrid[bestL],
    bestFit: bestFit!,
    scoreGrid,
  };
}

// -----------------------------------------------------------------------------
// §7 — Classification-DRE: L2-regularized logistic regression on Gaussian basis
// Pool x_p (label 1) and x_q (label 0); fit logit η(x) = β'ψ(x) + b₀;
// recover r̂(x) = (n_q/n_p) · π(x) / (1 - π(x)) where π = σ(η).
// Solved by IRLS (Newton) on the L2-penalized cross-entropy.
// -----------------------------------------------------------------------------

function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

export function logisticDREFit(
  xP: ArrayLike<number>,
  xQ: ArrayLike<number>,
  sigma: number,
  lambdaReg: number,
  options: { b?: number; seed?: number; maxIter?: number; tol?: number; centers?: ArrayLike<number> } = {},
): LogisticDREFitResult {
  const { b = 100, seed = 0xBEEF, maxIter = 50, tol = 1e-6 } = options;
  const rng = mulberry32(seed);
  const nP = xP.length;
  const nQ = xQ.length;
  const N = nP + nQ;
  const centers = options.centers ? Float64Array.from(options.centers) : pickKernelCenters(xP, b, rng);
  const bActual = centers.length;
  // Pool x and labels: first nP are label 1, last nQ are label 0
  const xPooled = new Float64Array(N);
  const yPooled = new Float64Array(N);
  for (let i = 0; i < nP; i++) {
    xPooled[i] = xP[i];
    yPooled[i] = 1;
  }
  for (let i = 0; i < nQ; i++) {
    xPooled[nP + i] = xQ[i];
    yPooled[nP + i] = 0;
  }
  const Psi = gaussianBasisDesign(xPooled, centers, sigma); // N × b
  // Augment with intercept column: use a separate intercept variable.
  // Design: Newton on (β, b₀) with L2 only on β (not intercept).
  const beta = new Float64Array(bActual);
  let b0 = 0;
  const pi = new Float64Array(N);
  const w = new Float64Array(N);
  const z = new Float64Array(N); // working response
  const dim = bActual + 1;
  const Hess = new Float64Array(dim * dim);
  const gradient = new Float64Array(dim);
  let loss = Infinity;
  let iter = 0;
  for (; iter < maxIter; iter++) {
    // Forward pass: compute η, π, w
    for (let i = 0; i < N; i++) {
      let eta = b0;
      const rowBase = i * bActual;
      for (let l = 0; l < bActual; l++) eta += Psi[rowBase + l] * beta[l];
      const p = sigmoid(eta);
      pi[i] = p;
      const ww = Math.max(p * (1 - p), 1e-8);
      w[i] = ww;
      z[i] = eta + (yPooled[i] - p) / ww;
    }
    // Build weighted normal equations: H = Ψ_aug^T W Ψ_aug + diag(λ, 0..0,...,0)
    // But L2 on β only; intercept is unpenalized.
    // Easier: Hess_{ll} for l < b: Σ w_i Psi[i,l] Psi[i,l'] + λ δ_{ll'}
    // Hess_{l, b}: Σ w_i Psi[i,l]
    // Hess_{b, b}: Σ w_i
    // grad_l = -(Σ w_i Psi[i,l] z[i]) + (Hess @ θ)_l + λ β_l  → simpler to use Newton update:
    // θ_new = (X^T W X + Λ)^{-1} X^T W z
    // We just compute that system.
    for (let i = 0; i < dim * dim; i++) Hess[i] = 0;
    for (let i = 0; i < dim; i++) gradient[i] = 0;
    for (let i = 0; i < N; i++) {
      const rowBase = i * bActual;
      const wi = w[i];
      const zi = z[i];
      // Top-left bActual × bActual
      for (let l = 0; l < bActual; l++) {
        const psi_l = Psi[rowBase + l];
        const wpsil = wi * psi_l;
        gradient[l] += wpsil * zi;
        for (let m = l; m < bActual; m++) {
          Hess[l * dim + m] += wpsil * Psi[rowBase + m];
        }
        // β-intercept block
        Hess[l * dim + bActual] += wpsil;
      }
      // Intercept-intercept
      Hess[bActual * dim + bActual] += wi;
      // Intercept gradient
      gradient[bActual] += wi * zi;
    }
    // Mirror upper to lower, add L2 on β
    for (let l = 0; l < bActual; l++) {
      for (let m = l; m < bActual; m++) {
        if (l === m) Hess[l * dim + m] += lambdaReg;
        else Hess[m * dim + l] = Hess[l * dim + m];
      }
      Hess[bActual * dim + l] = Hess[l * dim + bActual];
    }
    // Solve Hess θ = gradient via Cholesky (Hess is symmetric PD with L2 > 0)
    let L: Float64Array;
    try {
      L = choleskyInPlace(Hess, dim);
    } catch {
      // Add jitter on the diagonal and retry
      for (let i = 0; i < dim; i++) Hess[i * dim + i] += 1e-6;
      L = choleskyInPlace(Hess, dim);
    }
    const theta = choleskySolve(L, gradient, dim);
    // Check convergence
    let diff = 0;
    for (let l = 0; l < bActual; l++) {
      const d = theta[l] - beta[l];
      diff += d * d;
    }
    const d0 = theta[bActual] - b0;
    diff += d0 * d0;
    // Update
    for (let l = 0; l < bActual; l++) beta[l] = theta[l];
    b0 = theta[bActual];
    // Compute loss
    let nll = 0;
    for (let i = 0; i < N; i++) {
      let eta = b0;
      const rowBase = i * bActual;
      for (let l = 0; l < bActual; l++) eta += Psi[rowBase + l] * beta[l];
      // log(1 + e^{-y*η}) with y in {-1, +1}; for label y in {0,1}: -y η + log(1+e^η)
      const lse = eta > 0 ? eta + Math.log(1 + Math.exp(-eta)) : Math.log(1 + Math.exp(eta));
      nll += -yPooled[i] * eta + lse;
    }
    nll /= N;
    let pen = 0;
    for (let l = 0; l < bActual; l++) pen += beta[l] * beta[l];
    nll += (lambdaReg / (2 * N)) * pen;
    loss = nll;
    if (Math.sqrt(diff) < tol) {
      iter++;
      break;
    }
  }
  return { beta, intercept: b0, loss, iterations: iter };
}

/** Evaluate r̂_LR(x) = (n_q/n_p) · π(x) / (1 - π(x)) where π = sigmoid(β'ψ(x) + b₀). */
export function logisticDREEval(
  x: number,
  fit: LogisticDREFitResult,
  centers: ArrayLike<number>,
  sigma: number,
  nP: number,
  nQ: number,
): number {
  let eta = fit.intercept;
  const invDen = 1 / (2 * sigma * sigma);
  for (let l = 0; l < centers.length; l++) {
    const d = x - centers[l];
    eta += fit.beta[l] * Math.exp(-d * d * invDen);
  }
  const p = sigmoid(eta);
  // r = (n_q / n_p) * p / (1 - p)
  const eps = 1e-15;
  return (nQ / nP) * (p / Math.max(1 - p, eps));
}

/** Reliability diagram bins for a predicted probability vs binary label. */
export function reliabilityDiagramBins(
  pHat: ArrayLike<number>,
  y: ArrayLike<number>,
  nBins = 10,
): { binEdges: Float64Array; meanPredicted: Float64Array; meanObserved: Float64Array; counts: Int32Array } {
  const binEdges = new Float64Array(nBins + 1);
  for (let k = 0; k <= nBins; k++) binEdges[k] = k / nBins;
  const meanPredicted = new Float64Array(nBins);
  const meanObserved = new Float64Array(nBins);
  const counts = new Int32Array(nBins);
  for (let i = 0; i < pHat.length; i++) {
    let bin = Math.floor(pHat[i] * nBins);
    if (bin >= nBins) bin = nBins - 1;
    if (bin < 0) bin = 0;
    counts[bin]++;
    meanPredicted[bin] += pHat[i];
    meanObserved[bin] += y[i];
  }
  for (let k = 0; k < nBins; k++) {
    if (counts[k] > 0) {
      meanPredicted[k] /= counts[k];
      meanObserved[k] /= counts[k];
    }
  }
  return { binEdges, meanPredicted, meanObserved, counts };
}

// -----------------------------------------------------------------------------
// §9 — IW-ERM: weighted least squares for univariate linear regression.
// y = a + b x + ε; minimize Σ w_i (y_i - a - b x_i)².
// -----------------------------------------------------------------------------

export function weightedLinearFit(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  w?: ArrayLike<number>,
): { a: number; b: number } {
  const n = x.length;
  let sw = 0;
  let swx = 0;
  let swy = 0;
  let swxx = 0;
  let swxy = 0;
  if (w == null) {
    for (let i = 0; i < n; i++) {
      sw += 1;
      swx += x[i];
      swy += y[i];
      swxx += x[i] * x[i];
      swxy += x[i] * y[i];
    }
  } else {
    for (let i = 0; i < n; i++) {
      const wi = w[i];
      sw += wi;
      swx += wi * x[i];
      swy += wi * y[i];
      swxx += wi * x[i] * x[i];
      swxy += wi * x[i] * y[i];
    }
  }
  const denom = sw * swxx - swx * swx;
  if (Math.abs(denom) < 1e-20) return { a: swy / Math.max(sw, 1e-20), b: 0 };
  const a = (swxx * swy - swx * swxy) / denom;
  const b = (sw * swxy - swx * swy) / denom;
  return { a, b };
}

/** Test MSE of (a, b) against (x_test, y_test). */
export function linearMSE(
  a: number,
  b: number,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    const r = y[i] - (a + b * x[i]);
    s += r * r;
  }
  return s / x.length;
}

// -----------------------------------------------------------------------------
// §10 — Weighted split-conformal quantile.
// Given calibration residuals R_i (length m) and per-sample weights w_i (including
// the test-point weight w_{m+1}), compute the conformal quantile at level 1-α.
// Algorithm: sort R asc; normalize weights to probabilities; find smallest t in
// {R_1, ..., R_m, +∞} such that weighted CDF ≥ 1-α.
// -----------------------------------------------------------------------------

export function weightedConformalQuantile(
  residuals: ArrayLike<number>,
  calWeights: ArrayLike<number>,
  testWeight: number,
  alpha: number,
): number {
  const m = residuals.length;
  // Pair residuals with weights
  const pairs: Array<[number, number]> = new Array(m);
  let sumW = testWeight;
  for (let i = 0; i < m; i++) {
    pairs[i] = [residuals[i], calWeights[i]];
    sumW += calWeights[i];
  }
  if (sumW <= 0) return Infinity;
  pairs.sort((a, b) => a[0] - b[0]);
  // Test-point residual is implicitly +∞; its weight contributes only when t = +∞
  const target = 1 - alpha;
  let cum = 0;
  for (let i = 0; i < m; i++) {
    cum += pairs[i][1] / sumW;
    if (cum >= target) return pairs[i][0];
  }
  return Infinity;
}

/** Vanilla split-conformal quantile (uniform weights). */
export function splitConformalQuantile(residuals: ArrayLike<number>, alpha: number): number {
  const m = residuals.length;
  const sorted = Array.from(residuals).sort((a, b) => a - b);
  // Lei et al. 2018: the (ceil((m+1)(1-α)))-th smallest residual
  const k = Math.ceil((m + 1) * (1 - alpha));
  if (k > m) return Infinity;
  return sorted[k - 1];
}

// -----------------------------------------------------------------------------
// §11 — MMD u-statistic + permutation test.
// MMD²_u(X, Y) = (1/n(n-1)) Σ_{i≠i'} k(x_i,x_i') - (2/nm) Σ k(x_i, y_j) + (1/m(m-1)) Σ_{j≠j'} k(y_j, y_j').
// -----------------------------------------------------------------------------

/** Unbiased MMD² statistic with Gaussian kernel at bandwidth sigma_k (1-D). */
export function mmdUStatistic(
  X: ArrayLike<number>,
  Y: ArrayLike<number>,
  sigmaK: number,
): number {
  const n = X.length;
  const m = Y.length;
  const invDen = 1 / (2 * sigmaK * sigmaK);
  let sumXX = 0;
  let sumYY = 0;
  let sumXY = 0;
  // Σ_{i ≠ i'} k(x_i, x_i') — count each unordered pair twice
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = X[i] - X[j];
      sumXX += Math.exp(-d * d * invDen);
    }
  }
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      const d = Y[i] - Y[j];
      sumYY += Math.exp(-d * d * invDen);
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const d = X[i] - Y[j];
      sumXY += Math.exp(-d * d * invDen);
    }
  }
  // Pair sums above counted unordered (i<j); the u-statistic uses Σ_{i≠i'} which
  // is twice the unordered sum. Divide by n(n-1).
  return (2 * sumXX) / (n * (n - 1)) - (2 * sumXY) / (n * m) + (2 * sumYY) / (m * (m - 1));
}

/** Permutation MMD test: returns observed statistic, permutation null, p-value. */
export function mmdPermutationTest(
  X: ArrayLike<number>,
  Y: ArrayLike<number>,
  sigmaK: number,
  B: number,
  rng: () => number,
): MMDPermutationResult {
  const n = X.length;
  const m = Y.length;
  const N = n + m;
  const pooled = new Float64Array(N);
  for (let i = 0; i < n; i++) pooled[i] = X[i];
  for (let i = 0; i < m; i++) pooled[n + i] = Y[i];
  const observed = mmdUStatistic(X, Y, sigmaK);
  const permStats = new Float64Array(B);
  const idx = new Int32Array(N);
  for (let i = 0; i < N; i++) idx[i] = i;
  // Hoist Xp/Yp out of the B-permutation loop — reused each iteration to avoid
  // 2B per-iteration Float64Array allocations during the permutation test.
  const Xp = new Float64Array(n);
  const Yp = new Float64Array(m);
  for (let b = 0; b < B; b++) {
    // Fisher-Yates shuffle
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = idx[i];
      idx[i] = idx[j];
      idx[j] = tmp;
    }
    for (let i = 0; i < n; i++) Xp[i] = pooled[idx[i]];
    for (let i = 0; i < m; i++) Yp[i] = pooled[idx[n + i]];
    permStats[b] = mmdUStatistic(Xp, Yp, sigmaK);
  }
  // p-value: (1 + #{T_b >= T_obs}) / (B+1)
  let geCount = 0;
  for (let b = 0; b < B; b++) if (permStats[b] >= observed) geCount++;
  const pValue = (1 + geCount) / (B + 1);
  // 95th percentile of permStats — TypedArray .sort() defaults to numeric, so
  // slice().sort() avoids the typed-to-plain round-trip and the JS comparator
  // overhead Gemini flagged.
  const sortedPerm = permStats.slice().sort();
  const q95Idx = Math.floor(0.95 * B);
  return { observed, permStats, pValue, quantile95: sortedPerm[q95Idx] ?? observed };
}

// -----------------------------------------------------------------------------
// §12.3 — Curse-of-dim multivariate samplers.
// p = N(0_d, I_d), q = N(μ * 1_d, I_d); closed-form r(x) = exp((dμ²/2) - μ Σ x_i).
// -----------------------------------------------------------------------------

/** Sample n iid d-dim isotropic Gaussians centered at mu (broadcast scalar). */
export function sampleIsotropicShiftMV(
  rng: () => number,
  mu: number,
  d: number,
  n: number,
): Float64Array {
  // Row-major: out[i*d + k]
  const g = gaussianSampler(rng);
  const out = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < d; k++) out[i * d + k] = mu + g();
  }
  return out;
}

/** Closed-form r(x) = exp(dμ²/2 - μ Σ x_k) for the multivariate shift toy. */
export function closedFormShiftRatioMV(
  X: Float64Array, // row-major n × d
  mu: number,
  d: number,
): Float64Array {
  const n = X.length / d;
  const r = new Float64Array(n);
  const constTerm = (d * mu * mu) / 2;
  for (let i = 0; i < n; i++) {
    let s = 0;
    const base = i * d;
    for (let k = 0; k < d; k++) s += X[base + k];
    r[i] = Math.exp(constTerm - mu * s);
  }
  return r;
}

// -----------------------------------------------------------------------------
// Auxiliary: Pearson correlation.
// -----------------------------------------------------------------------------

export function pearsonCorrelation(x: ArrayLike<number>, y: ArrayLike<number>): number {
  const n = x.length;
  if (n < 2) return 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx;
    const b = y[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const den = Math.sqrt(dx * dy);
  if (den < 1e-20) return 0;
  return num / den;
}

// -----------------------------------------------------------------------------
// Re-exports (PRNG idioms reused from gaussian-processes).
// -----------------------------------------------------------------------------

export { mulberry32, gaussianSampler };

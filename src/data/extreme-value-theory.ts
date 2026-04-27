// =============================================================================
// extreme-value-theory.ts
//
// Shared math module for the T4 Extreme Value Theory topic. Consumers:
//   - GEVFamilyExplorer.tsx   (featured §2 viz)
//   - BlockMaximaVsPOT.tsx    (§4 vs §5 contrast)
//   - HillPlotExplorer.tsx    (§5.5 tail-index estimators)
//
// All numerical algorithms are TypeScript ports of the verified Python notebook
// at notebooks/extreme-value-theory/01_extreme_value_theory.ipynb. Where the
// Python uses np.random / scipy.stats / scipy.optimize, this module substitutes:
//   - the seeded Mulberry32 PRNG and Box-Muller Gaussian sampler from
//     ../components/viz/shared/nonparametric-ml (matches the notebook's
//     np.random.default_rng(42) seed for cross-implementation reproducibility,
//     up to PRNG-implementation differences),
//   - hand-rolled GEV / GPD densities written in the **standard sign
//     convention** (notebook + theorem statements) — NOT scipy.stats.genextreme
//     / genpareto, which use c = -xi. The brief's §10 footgun is sidestepped
//     by never calling a SciPy-equivalent library at all,
//   - a small Nelder-Mead simplex optimizer + central-difference Hessian for
//     MLE and delta-method standard errors. Quasi-Newton (L-BFGS-B) would be
//     faster, but Nelder-Mead is simpler and robust to support-boundary
//     issues that arise when the slider is dragged into pathological regimes.
//
// Numerical agreement with the notebook is verified by
//   src/components/viz/shared/__tests__/verify-extreme-value-theory.ts
// against the printed outputs of cells 4 (Gumbel normalization), 10 (GEV MLE
// + PWM + return level), 12 (GPD MLE), and 13 (Hill / Pickands / DEdH +
// VaR/ES). Tolerances per the brief: ±1% relative for deterministic
// quantities, ±0.05 absolute for stochastic estimators of xi at B=50, ±5%
// relative for return-level / VaR / ES extrapolations.
// =============================================================================

import { mulberry32, gaussianSampler } from '../components/viz/shared/nonparametric-ml';

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export interface GEVParams {
  xi: number;
  mu: number;
  sigma: number;
}

export interface GPDParams {
  xi: number;
  beta: number;
}

export interface GEVMleResult {
  theta: GEVParams;
  /** 3x3 inverse Hessian (covariance estimate) flattened row-major. */
  cov: Float64Array;
  iter: number;
  converged: boolean;
}

export interface GPDMleResult {
  theta: GPDParams;
  /** 2x2 inverse Hessian (covariance estimate) flattened row-major. */
  cov: Float64Array;
  iter: number;
  converged: boolean;
}

export type ParentPreset =
  | 'normal'
  | 'pareto2'
  | 't3'
  | 'uniform';

// -----------------------------------------------------------------------------
// Re-exports of the shared PRNG. Consumers should import from here so the EVT
// module stays self-contained from a viz-component standpoint.
// -----------------------------------------------------------------------------

export { mulberry32, gaussianSampler };

// -----------------------------------------------------------------------------
// Special functions: Lanczos log-gamma + gamma. Accurate to ~1e-13 for x > 0.
// Used by gevPwm to evaluate Γ(1 - ξ).
// -----------------------------------------------------------------------------

const LANCZOS_G = 7;
const LANCZOS_COEFFS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export function logGamma(x: number): number {
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const xx = x - 1;
  let a = LANCZOS_COEFFS[0];
  for (let i = 1; i < LANCZOS_COEFFS.length; i++) {
    a += LANCZOS_COEFFS[i] / (xx + i);
  }
  const t = xx + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (xx + 0.5) * Math.log(t) - t + Math.log(a);
}

export function gamma(x: number): number {
  return Math.exp(logGamma(x));
}

// -----------------------------------------------------------------------------
// GEV distribution — STANDARD sign convention (notebook §2.4 + Theorem 2):
//   G_xi(z) = exp(-(1 + xi*z)^(-1/xi))   for xi != 0,  z s.t. 1 + xi z > 0
//   G_0(z) = exp(-exp(-z))               (Gumbel limit)
// where z = (x - mu) / sigma. SciPy uses c = -xi, but we never call SciPy.
// -----------------------------------------------------------------------------

const XI_NEAR_ZERO = 1e-8;

export function gevPdf(x: number, xi: number, mu: number = 0, sigma: number = 1): number {
  if (sigma <= 0) return 0;
  const z = (x - mu) / sigma;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    return Math.exp(-z - Math.exp(-z)) / sigma;
  }
  const arg = 1 + xi * z;
  if (arg <= 0) return 0;
  const argInvXi = Math.pow(arg, -1 / xi);
  return (Math.pow(arg, -1 / xi - 1) * Math.exp(-argInvXi)) / sigma;
}

export function gevCdf(x: number, xi: number, mu: number = 0, sigma: number = 1): number {
  if (sigma <= 0) return Number.NaN;
  const z = (x - mu) / sigma;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    return Math.exp(-Math.exp(-z));
  }
  const arg = 1 + xi * z;
  if (arg <= 0) return xi > 0 ? 0 : 1;
  return Math.exp(-Math.pow(arg, -1 / xi));
}

export function gevQuantile(p: number, xi: number, mu: number = 0, sigma: number = 1): number {
  if (p <= 0) return xi > 0 ? mu - sigma / xi : -Infinity;
  if (p >= 1) return xi < 0 ? mu - sigma / xi : Infinity;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    return mu - sigma * Math.log(-Math.log(p));
  }
  return mu + (sigma / xi) * (Math.pow(-Math.log(p), -xi) - 1);
}

export function gevSample(
  xi: number,
  mu: number,
  sigma: number,
  n: number,
  rng: () => number,
): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let u = rng();
    if (u <= 0) u = Number.EPSILON;
    if (u >= 1) u = 1 - Number.EPSILON;
    out[i] = gevQuantile(u, xi, mu, sigma);
  }
  return out;
}

// -----------------------------------------------------------------------------
// GPD distribution — STANDARD sign convention (notebook §5.1 / Definition 3):
//   H_{xi, beta}(y) = 1 - (1 + xi*y/beta)^(-1/xi)   for xi != 0
//   H_{0, beta}(y) = 1 - exp(-y/beta)               (Exponential limit)
// Support: [0, infinity) for xi >= 0; [0, -beta/xi] for xi < 0.
// -----------------------------------------------------------------------------

export function gpdPdf(y: number, xi: number, beta: number): number {
  if (beta <= 0 || y < 0) return 0;
  if (xi < 0 && y > -beta / xi) return 0;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    return Math.exp(-y / beta) / beta;
  }
  const arg = 1 + (xi * y) / beta;
  if (arg <= 0) return 0;
  return Math.pow(arg, -1 / xi - 1) / beta;
}

export function gpdCdf(y: number, xi: number, beta: number): number {
  if (beta <= 0 || y < 0) return 0;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    return 1 - Math.exp(-y / beta);
  }
  const arg = 1 + (xi * y) / beta;
  if (arg <= 0) return 1;
  return 1 - Math.pow(arg, -1 / xi);
}

export function gpdQuantile(p: number, xi: number, beta: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return xi < 0 ? -beta / xi : Infinity;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    return -beta * Math.log(1 - p);
  }
  return (beta / xi) * (Math.pow(1 - p, -xi) - 1);
}

export function gpdSample(
  xi: number,
  beta: number,
  n: number,
  rng: () => number,
): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let u = rng();
    if (u <= 0) u = Number.EPSILON;
    if (u >= 1) u = 1 - Number.EPSILON;
    out[i] = gpdQuantile(u, xi, beta);
  }
  return out;
}

// -----------------------------------------------------------------------------
// Parent samplers (notebook §1, §3.4, §4.5, §5.7).
// -----------------------------------------------------------------------------

export function sampleNormal(n: number, rng: () => number): Float64Array {
  const gauss = gaussianSampler(rng);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = gauss();
  return out;
}

/**
 * Standard Pareto with shape alpha (x_min = 1). 1 - F(x) = x^{-alpha} for x ≥ 1.
 * Notebook uses scipy.stats.pareto(b=alpha) with default scale=1 — matches.
 */
export function sampleParetoStandard(
  alpha: number,
  n: number,
  rng: () => number,
): Float64Array {
  const out = new Float64Array(n);
  const inv = -1 / alpha;
  for (let i = 0; i < n; i++) {
    let u = rng();
    if (u <= 0) u = Number.EPSILON;
    if (u >= 1) u = 1 - Number.EPSILON;
    out[i] = Math.pow(1 - u, inv);
  }
  return out;
}

/** Student-t with 3 degrees of freedom: t_3 = Z / sqrt((Z1^2 + Z2^2 + Z3^2)/3). */
export function sampleT3(n: number, rng: () => number): Float64Array {
  const gauss = gaussianSampler(rng);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const z = gauss();
    const w1 = gauss(), w2 = gauss(), w3 = gauss();
    const chiSq = (w1 * w1 + w2 * w2 + w3 * w3) / 3;
    out[i] = z / Math.sqrt(chiSq);
  }
  return out;
}

export function sampleUniform(n: number, rng: () => number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = rng();
  return out;
}

/** Convenience: dispatch on parent-preset string for viz components. */
export function sampleParent(
  preset: ParentPreset,
  n: number,
  rng: () => number,
): Float64Array {
  switch (preset) {
    case 'normal': return sampleNormal(n, rng);
    case 'pareto2': return sampleParetoStandard(2, n, rng);
    case 't3': return sampleT3(n, rng);
    case 'uniform': return sampleUniform(n, rng);
  }
}

/** True ξ for a known parent preset (used as truth-line in viz components). */
export function trueXi(preset: ParentPreset): number {
  switch (preset) {
    case 'normal': return 0;
    case 'pareto2': return 0.5;
    case 't3': return 1 / 3;
    case 'uniform': return -1;
  }
}

// -----------------------------------------------------------------------------
// Block-maxima helper.
// -----------------------------------------------------------------------------

export function blockMaxima(samples: Float64Array, blockSize: number): Float64Array {
  const B = Math.floor(samples.length / blockSize);
  const out = new Float64Array(B);
  for (let j = 0; j < B; j++) {
    let m = -Infinity;
    const start = j * blockSize;
    for (let i = 0; i < blockSize; i++) {
      const v = samples[start + i];
      if (v > m) m = v;
    }
    out[j] = m;
  }
  return out;
}

// -----------------------------------------------------------------------------
// Gumbel normalization for Normal block maxima (notebook §1.3 / Example 2).
// b_n = sqrt(2 log n) - (log log n + log(4 pi)) / (2 sqrt(2 log n))
// a_n = 1 / sqrt(2 log n)
// -----------------------------------------------------------------------------

export function gumbelNormalizationNormal(n: number): { aN: number; bN: number } {
  const logN = Math.log(n);
  const sqrt2LogN = Math.sqrt(2 * logN);
  const bN = sqrt2LogN - (Math.log(logN) + Math.log(4 * Math.PI)) / (2 * sqrt2LogN);
  const aN = 1 / sqrt2LogN;
  return { aN, bN };
}

// -----------------------------------------------------------------------------
// Optimization: Nelder-Mead simplex + central-difference Hessian.
// -----------------------------------------------------------------------------

interface NelderMeadResult {
  x: number[];
  fx: number;
  iter: number;
  converged: boolean;
}

function nelderMead(
  f: (x: number[]) => number,
  x0: number[],
  options: { maxIter?: number; tol?: number; step?: number } = {},
): NelderMeadResult {
  const maxIter = options.maxIter ?? 500;
  const tol = options.tol ?? 1e-8;
  const step = options.step ?? 0.05;

  const n = x0.length;
  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const xi = x0.slice();
    const h = Math.abs(xi[i]) > 1e-10 ? step * xi[i] : step;
    xi[i] += h;
    simplex.push(xi);
  }
  const fvals: number[] = simplex.map(f);

  const sortByFvals = () => {
    const order = Array.from({ length: n + 1 }, (_, i) => i).sort(
      (a, b) => fvals[a] - fvals[b],
    );
    const newSimplex = order.map((i) => simplex[i]);
    const newFvals = order.map((i) => fvals[i]);
    for (let i = 0; i <= n; i++) {
      simplex[i] = newSimplex[i];
      fvals[i] = newFvals[i];
    }
  };

  for (let iter = 0; iter < maxIter; iter++) {
    sortByFvals();

    if (
      fvals[n] - fvals[0] < tol ||
      simplex[n].every((_, j) => Math.abs(simplex[n][j] - simplex[0][j]) < tol)
    ) {
      return { x: simplex[0].slice(), fx: fvals[0], iter, converged: true };
    }

    const centroid: number[] = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j] += simplex[i][j] / n;
    }
    const worst = simplex[n];

    const reflection = centroid.map((c, j) => c + 1.0 * (c - worst[j]));
    const fr = f(reflection);

    if (fr >= fvals[0] && fr < fvals[n - 1]) {
      simplex[n] = reflection;
      fvals[n] = fr;
      continue;
    }

    if (fr < fvals[0]) {
      const expansion = centroid.map((c, j) => c + 2.0 * (reflection[j] - c));
      const fe = f(expansion);
      if (fe < fr) {
        simplex[n] = expansion;
        fvals[n] = fe;
      } else {
        simplex[n] = reflection;
        fvals[n] = fr;
      }
      continue;
    }

    const contraction = centroid.map((c, j) => c + 0.5 * (worst[j] - c));
    const fc = f(contraction);
    if (fc < fvals[n]) {
      simplex[n] = contraction;
      fvals[n] = fc;
      continue;
    }

    for (let i = 1; i <= n; i++) {
      simplex[i] = simplex[i].map((xij, j) => simplex[0][j] + 0.5 * (xij - simplex[0][j]));
      fvals[i] = f(simplex[i]);
    }
  }

  sortByFvals();
  return { x: simplex[0].slice(), fx: fvals[0], iter: maxIter, converged: false };
}

/**
 * Central-difference Hessian at point x. h is a per-dimension step.
 * Returns a (n*n)-length Float64Array (row-major).
 */
function finiteDiffHessian(
  f: (x: number[]) => number,
  x: number[],
  h: number = 1e-4,
): Float64Array {
  const n = x.length;
  const H = new Float64Array(n * n);
  const fx = f(x);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const hi = Math.max(h * Math.abs(x[i]), h);
      const hj = Math.max(h * Math.abs(x[j]), h);
      let value: number;
      if (i === j) {
        const xp = x.slice(); xp[i] += hi;
        const xm = x.slice(); xm[i] -= hi;
        value = (f(xp) - 2 * fx + f(xm)) / (hi * hi);
      } else {
        const xpp = x.slice(); xpp[i] += hi; xpp[j] += hj;
        const xpm = x.slice(); xpm[i] += hi; xpm[j] -= hj;
        const xmp = x.slice(); xmp[i] -= hi; xmp[j] += hj;
        const xmm = x.slice(); xmm[i] -= hi; xmm[j] -= hj;
        value = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * hi * hj);
      }
      H[i * n + j] = value;
      H[j * n + i] = value;
    }
  }
  return H;
}

/**
 * Invert a small (n=2 or n=3) symmetric positive-definite matrix.
 * Returns a flattened row-major n*n Float64Array. Falls back to a diagonal
 * approximation if Cholesky fails (degenerate cases at the support boundary).
 */
function invertSPD(M: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);

  if (n === 2) {
    const det = M[0] * M[3] - M[1] * M[2];
    if (Math.abs(det) < 1e-14) {
      out[0] = 1 / Math.max(M[0], 1e-8);
      out[3] = 1 / Math.max(M[3], 1e-8);
      return out;
    }
    out[0] = M[3] / det;
    out[1] = -M[1] / det;
    out[2] = -M[2] / det;
    out[3] = M[0] / det;
    return out;
  }

  if (n === 3) {
    const a = M[0], b = M[1], c = M[2];
    const d = M[3], e = M[4], f = M[5];
    const g = M[6], h = M[7], i = M[8];
    const A = e * i - f * h;
    const B = -(d * i - f * g);
    const C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-14) {
      out[0] = 1 / Math.max(M[0], 1e-8);
      out[4] = 1 / Math.max(M[4], 1e-8);
      out[8] = 1 / Math.max(M[8], 1e-8);
      return out;
    }
    out[0] = A / det;
    out[1] = -(b * i - c * h) / det;
    out[2] = (b * f - c * e) / det;
    out[3] = B / det;
    out[4] = (a * i - c * g) / det;
    out[5] = -(a * f - c * d) / det;
    out[6] = C / det;
    out[7] = -(a * h - b * g) / det;
    out[8] = (a * e - b * d) / det;
    return out;
  }

  throw new Error(`invertSPD only supports n=2,3; got n=${n}`);
}

// -----------------------------------------------------------------------------
// GEV inference (notebook §4 / Cell 10).
// -----------------------------------------------------------------------------

export function gevNegLogLik(theta: GEVParams, data: Float64Array): number {
  const { xi, mu, sigma } = theta;
  if (sigma <= 0) return Infinity;
  const B = data.length;

  if (Math.abs(xi) < XI_NEAR_ZERO) {
    let sum = 0;
    for (let i = 0; i < B; i++) {
      const z = (data[i] - mu) / sigma;
      sum += z + Math.exp(-z);
    }
    return sum + B * Math.log(sigma);
  }

  let sumLogArg = 0;
  let sumArgPow = 0;
  for (let i = 0; i < B; i++) {
    const z = (data[i] - mu) / sigma;
    const arg = 1 + xi * z;
    if (arg <= 0) return Infinity;
    sumLogArg += Math.log(arg);
    sumArgPow += Math.pow(arg, -1 / xi);
  }
  return B * Math.log(sigma) + (1 / xi + 1) * sumLogArg + sumArgPow;
}

/**
 * GEV maximum-likelihood estimator. Initialization uses Gumbel-moment
 * heuristic: σ₀ = std·√6/π, μ₀ = median, ξ₀ = 0.1 (matches notebook Cell 10).
 */
export function gevMle(data: Float64Array): GEVMleResult {
  const sorted = Array.from(data).sort((a, b) => a - b);
  const median =
    sorted.length % 2 === 0
      ? 0.5 * (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2])
      : sorted[(sorted.length - 1) / 2];
  let mean = 0;
  for (const v of data) mean += v / data.length;
  let varSum = 0;
  for (const v of data) varSum += (v - mean) * (v - mean);
  const sd = Math.sqrt(varSum / Math.max(1, data.length - 1));
  const sigma0 = (sd * Math.sqrt(6)) / Math.PI;
  const theta0 = [0.1, median, sigma0];

  const negLogLik = (t: number[]) =>
    gevNegLogLik({ xi: t[0], mu: t[1], sigma: t[2] }, data);

  const result = nelderMead(negLogLik, theta0, { maxIter: 800, tol: 1e-9 });
  const theta = { xi: result.x[0], mu: result.x[1], sigma: result.x[2] };

  const H = finiteDiffHessian(negLogLik, result.x, 1e-4);
  const cov = invertSPD(H, 3);

  return { theta, cov, iter: result.iter, converged: result.converged };
}

/**
 * GEV probability-weighted moments (Hosking, Wallis & Wood 1985).
 * Solves (3β₂ - β₀) / (2β₁ - β₀) = (3^ξ - 1) / (2^ξ - 1) by bisection on ξ,
 * then closed-form-recovers σ and μ.
 */
export function gevPwm(data: Float64Array): GEVParams {
  const B = data.length;
  const M = Array.from(data).sort((a, b) => a - b);

  let b0 = 0;
  for (const v of M) b0 += v / B;
  let b1 = 0;
  for (let j = 1; j <= B; j++) {
    b1 += ((j - 1) / Math.max(1, B - 1)) * M[j - 1];
  }
  b1 /= B;
  let b2 = 0;
  for (let j = 1; j <= B; j++) {
    b2 += ((j - 1) * (j - 2)) / Math.max(1, (B - 1) * (B - 2)) * M[j - 1];
  }
  b2 /= B;

  const target = (3 * b2 - b0) / (2 * b1 - b0);

  // Bisection on equation((3^xi - 1) / (2^xi - 1) = target.
  // Bracket: xi in [-0.49, 0.99] (matches notebook bounds).
  const equation = (xi: number) => {
    if (Math.abs(xi) < XI_NEAR_ZERO) return Math.log(3) / Math.log(2) - target;
    return (Math.pow(3, xi) - 1) / (Math.pow(2, xi) - 1) - target;
  };

  let lo = -0.49, hi = 0.99;
  let fLo = equation(lo), fHi = equation(hi);
  let xiHat: number;
  if (fLo * fHi > 0) {
    xiHat = Math.abs(fLo) < Math.abs(fHi) ? lo : hi;
  } else {
    for (let iter = 0; iter < 100; iter++) {
      const mid = 0.5 * (lo + hi);
      const fMid = equation(mid);
      if (Math.abs(fMid) < 1e-9 || hi - lo < 1e-9) {
        xiHat = mid;
        break;
      }
      if (fLo * fMid < 0) {
        hi = mid;
        fHi = fMid;
      } else {
        lo = mid;
        fLo = fMid;
      }
      xiHat = mid;
    }
    xiHat = 0.5 * (lo + hi);
  }

  let sigmaHat: number;
  let muHat: number;
  if (Math.abs(xiHat) < XI_NEAR_ZERO) {
    sigmaHat = (2 * b1 - b0) / Math.log(2);
    const eulerGamma = 0.5772156649015329;
    muHat = b0 - eulerGamma * sigmaHat;
  } else {
    const gammaOneMinusXi = gamma(1 - xiHat);
    sigmaHat = ((2 * b1 - b0) * xiHat) / (gammaOneMinusXi * (Math.pow(2, xiHat) - 1));
    muHat = b0 - (sigmaHat / xiHat) * (gammaOneMinusXi - 1);
  }

  return { xi: xiHat, mu: muHat, sigma: sigmaHat };
}

/**
 * T-period return level (notebook §4.4 / Example 5). x_T = G^{-1}(1 - 1/T).
 */
export function returnLevel(theta: GEVParams, T: number): number {
  return gevQuantile(1 - 1 / T, theta.xi, theta.mu, theta.sigma);
}

/**
 * Delta-method standard error for the T-period return level.
 * Gradient of x_T(theta) per notebook Cell 10: closed form for xi != 0.
 */
export function returnLevelSeDelta(
  theta: GEVParams,
  cov: Float64Array,
  T: number,
): number {
  const { xi, sigma } = theta;
  const yT = -Math.log(1 - 1 / T);
  let dxiTerm: number, dmuTerm: number, dsigmaTerm: number;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    dxiTerm = 0;
    dmuTerm = 1;
    dsigmaTerm = -Math.log(yT);
  } else {
    const yTNegXi = Math.pow(yT, -xi);
    dxiTerm = -(sigma / (xi * xi)) * (yTNegXi - 1) - (sigma / xi) * yTNegXi * Math.log(yT);
    dmuTerm = 1;
    dsigmaTerm = (yTNegXi - 1) / xi;
  }
  // grad order matches theta order: (xi, mu, sigma)
  const grad = [dxiTerm, dmuTerm, dsigmaTerm];
  let varEst = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      varEst += grad[i] * cov[i * 3 + j] * grad[j];
    }
  }
  return Math.sqrt(Math.max(0, varEst));
}

/**
 * Profile log-likelihood for ξ. For each ξ in xiGrid, optimize μ, σ and
 * record the maximized log-likelihood. Used by the §4 worked-example viz.
 */
export function profileLogLikXi(
  data: Float64Array,
  xiGrid: Float64Array,
): Float64Array {
  const out = new Float64Array(xiGrid.length);
  let mean = 0;
  for (const v of data) mean += v / data.length;
  let varSum = 0;
  for (const v of data) varSum += (v - mean) * (v - mean);
  const sd = Math.sqrt(varSum / Math.max(1, data.length - 1));
  const sigma0 = (sd * Math.sqrt(6)) / Math.PI;
  const sorted = Array.from(data).sort((a, b) => a - b);
  const median =
    sorted.length % 2 === 0
      ? 0.5 * (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2])
      : sorted[(sorted.length - 1) / 2];

  for (let i = 0; i < xiGrid.length; i++) {
    const xiFixed = xiGrid[i];
    const negLogLikFixed = (t: number[]) =>
      gevNegLogLik({ xi: xiFixed, mu: t[0], sigma: t[1] }, data);
    const result = nelderMead(negLogLikFixed, [median, sigma0], { maxIter: 300, tol: 1e-8 });
    out[i] = -result.fx;
  }
  return out;
}

// -----------------------------------------------------------------------------
// GPD inference (notebook §5 / Cell 12).
// -----------------------------------------------------------------------------

export function gpdNegLogLik(theta: GPDParams, exceedances: Float64Array): number {
  const { xi, beta } = theta;
  if (beta <= 0) return Infinity;
  const Nu = exceedances.length;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    let sumY = 0;
    for (let i = 0; i < Nu; i++) sumY += exceedances[i];
    return Nu * Math.log(beta) + sumY / beta;
  }
  let sumLogArg = 0;
  for (let i = 0; i < Nu; i++) {
    const arg = 1 + (xi * exceedances[i]) / beta;
    if (arg <= 0) return Infinity;
    sumLogArg += Math.log(arg);
  }
  return Nu * Math.log(beta) + (1 / xi + 1) * sumLogArg;
}

export function gpdMle(exceedances: Float64Array): GPDMleResult {
  let mean = 0;
  for (const v of exceedances) mean += v / exceedances.length;
  const beta0 = mean;
  const theta0 = [0.1, beta0];

  const negLogLik = (t: number[]) =>
    gpdNegLogLik({ xi: t[0], beta: t[1] }, exceedances);

  const result = nelderMead(negLogLik, theta0, { maxIter: 600, tol: 1e-9 });
  const theta = { xi: result.x[0], beta: result.x[1] };

  const H = finiteDiffHessian(negLogLik, result.x, 1e-4);
  const cov = invertSPD(H, 2);

  return { theta, cov, iter: result.iter, converged: result.converged };
}

/** Empirical mean-excess function over a threshold grid (notebook §5.3). */
export function meanExcess(data: Float64Array, uGrid: Float64Array): Float64Array {
  const out = new Float64Array(uGrid.length);
  for (let i = 0; i < uGrid.length; i++) {
    const u = uGrid[i];
    let sum = 0, count = 0;
    for (let j = 0; j < data.length; j++) {
      if (data[j] > u) {
        sum += data[j] - u;
        count++;
      }
    }
    out[i] = count > 0 ? sum / count : Number.NaN;
  }
  return out;
}

/**
 * Parameter-stability plot data: GPD MLE at each threshold u in uGrid.
 * Returns { xiHat, betaStar } where betaStar = beta_hat - xi_hat * u
 * (threshold-invariant under GPD, per notebook §5.3).
 */
export interface ParameterStabilityResult {
  xiHat: Float64Array;
  betaStar: Float64Array;
}

export function parameterStability(
  data: Float64Array,
  uGrid: Float64Array,
  minExceedances: number = 30,
): ParameterStabilityResult {
  const xiHat = new Float64Array(uGrid.length);
  const betaStar = new Float64Array(uGrid.length);
  for (let i = 0; i < uGrid.length; i++) {
    const u = uGrid[i];
    const excess: number[] = [];
    for (let j = 0; j < data.length; j++) {
      if (data[j] > u) excess.push(data[j] - u);
    }
    if (excess.length < minExceedances) {
      xiHat[i] = Number.NaN;
      betaStar[i] = Number.NaN;
      continue;
    }
    const result = gpdMle(Float64Array.from(excess));
    xiHat[i] = result.theta.xi;
    betaStar[i] = result.theta.beta - result.theta.xi * u;
  }
  return { xiHat, betaStar };
}

// -----------------------------------------------------------------------------
// Tail-index estimators (notebook §5.5 / Cell 13).
// -----------------------------------------------------------------------------

/**
 * Hill estimator for ξ > 0 (Fréchet domain only).
 * Returns ξ̂(k) for k in kVec; NaN where k < 2 or k >= n.
 * Notebook formula (Cell 13): mean(log X_(n-i+1) for i=1..k) - log X_(n-k-1).
 * X_(n-k-1) is the order statistic just below the threshold; this is the
 * SciPy / Mason 1982 indexing convention.
 */
export function hillEstimator(
  data: Float64Array,
  kVec: Int32Array,
): Float64Array {
  const sorted = Array.from(data).sort((a, b) => a - b);
  const n = sorted.length;
  const logSorted = sorted.map((v) => (v > 0 ? Math.log(v) : Number.NEGATIVE_INFINITY));
  const out = new Float64Array(kVec.length);
  for (let i = 0; i < kVec.length; i++) {
    const k = kVec[i];
    if (k < 2 || k >= n) {
      out[i] = Number.NaN;
      continue;
    }
    let sum = 0;
    for (let j = n - k; j < n; j++) sum += logSorted[j];
    out[i] = sum / k - logSorted[n - k - 1];
  }
  return out;
}

/**
 * Pickands estimator. Works for any ξ ∈ ℝ. Greys out when 4k >= n (the
 * estimator is undefined because X_(n - 4k + 1) doesn't exist).
 */
export function pickandsEstimator(
  data: Float64Array,
  kVec: Int32Array,
): Float64Array {
  const sorted = Array.from(data).sort((a, b) => a - b);
  const n = sorted.length;
  const log2 = Math.log(2);
  const out = new Float64Array(kVec.length);
  for (let i = 0; i < kVec.length; i++) {
    const k = kVec[i];
    if (k < 1 || 4 * k >= n) {
      out[i] = Number.NaN;
      continue;
    }
    const x1 = sorted[n - k];
    const x2 = sorted[n - 2 * k];
    const x4 = sorted[n - 4 * k];
    const num = x1 - x2;
    const den = x2 - x4;
    if (den <= 0 || num <= 0) {
      out[i] = Number.NaN;
      continue;
    }
    out[i] = Math.log(num / den) / log2;
  }
  return out;
}

/**
 * Dekkers-Einmahl-de Haan moment estimator. Works for any ξ ∈ ℝ.
 * ξ̂_DEdH(k) = M⁽¹⁾(k) + 1 - 0.5 / (1 - (M⁽¹⁾)² / M⁽²⁾)
 * where M⁽ʲ⁾(k) = (1/k) Σ (log X_(n-i+1) - log X_(n-k))^j over i=1..k.
 */
export function dedhEstimator(
  data: Float64Array,
  kVec: Int32Array,
): Float64Array {
  const sorted = Array.from(data).sort((a, b) => a - b);
  const n = sorted.length;
  const logSorted = sorted.map((v) => (v > 0 ? Math.log(v) : Number.NEGATIVE_INFINITY));
  const out = new Float64Array(kVec.length);
  for (let i = 0; i < kVec.length; i++) {
    const k = kVec[i];
    if (k < 2 || k >= n) {
      out[i] = Number.NaN;
      continue;
    }
    const ref = logSorted[n - k - 1];
    let m1 = 0, m2 = 0;
    for (let j = n - k; j < n; j++) {
      const spacing = logSorted[j] - ref;
      m1 += spacing / k;
      m2 += (spacing * spacing) / k;
    }
    if (m2 <= 0) {
      out[i] = Number.NaN;
      continue;
    }
    const ratio = 1 - (m1 * m1) / m2;
    if (Math.abs(ratio) < 1e-12) {
      out[i] = Number.NaN;
      continue;
    }
    out[i] = m1 + 1 - 0.5 / ratio;
  }
  return out;
}

// -----------------------------------------------------------------------------
// POT VaR / ES (notebook §5.6 / Cell 13).
// -----------------------------------------------------------------------------

/**
 * POT-based Value-at-Risk at level α.
 *   VaR_α = u + (β/ξ) * ((n(1 - α) / Nu)^{-ξ} - 1)
 * for ξ ≠ 0; Exponential limit u - β log(p) at ξ = 0.
 */
export function potVar(
  theta: GPDParams,
  u: number,
  Nu: number,
  n: number,
  alpha: number,
): number {
  const { xi, beta } = theta;
  const p = (n * (1 - alpha)) / Nu;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    return u - beta * Math.log(p);
  }
  return u + (beta / xi) * (Math.pow(p, -xi) - 1);
}

/**
 * POT-based Expected Shortfall at level α. Requires ξ < 1 (mean exists).
 */
export function potEs(
  theta: GPDParams,
  u: number,
  Nu: number,
  n: number,
  alpha: number,
): number {
  const { xi, beta } = theta;
  if (xi >= 1) return Infinity;
  const varAlpha = potVar(theta, u, Nu, n, alpha);
  return varAlpha + (beta + xi * (varAlpha - u)) / (1 - xi);
}

/**
 * Delta-method standard error for POT VaR. Treats Nu/n as fixed.
 */
export function potVarSeDelta(
  theta: GPDParams,
  cov: Float64Array,
  u: number,
  Nu: number,
  n: number,
  alpha: number,
): number {
  const { xi, beta } = theta;
  const p = (n * (1 - alpha)) / Nu;
  let dxi: number, dbeta: number;
  if (Math.abs(xi) < XI_NEAR_ZERO) {
    dxi = 0;
    dbeta = -Math.log(p);
  } else {
    const pNegXi = Math.pow(p, -xi);
    dxi = -(beta / (xi * xi)) * (pNegXi - 1) - (beta / xi) * pNegXi * Math.log(p);
    dbeta = (pNegXi - 1) / xi;
  }
  // grad order matches GPD theta order: (xi, beta)
  const grad = [dxi, dbeta];
  let varEst = 0;
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      varEst += grad[i] * cov[i * 2 + j] * grad[j];
    }
  }
  return Math.sqrt(Math.max(0, varEst));
}

// -----------------------------------------------------------------------------
// Lazy-cached canned datasets for the §4 / §5 worked-example panels.
// Per handoff-reference §4: expensive computations are lazy-initialized so
// they don't block first-paint. Each cache key is a (preset, N, seed) tuple.
// -----------------------------------------------------------------------------

interface RawSampleCacheEntry {
  preset: ParentPreset;
  N: number;
  seed: number;
  samples: Float64Array;
}

const rawSampleCache: RawSampleCacheEntry[] = [];

export function getRawSamples(
  preset: ParentPreset,
  N: number,
  seed: number = 42,
): Float64Array {
  const hit = rawSampleCache.find(
    (e) => e.preset === preset && e.N === N && e.seed === seed,
  );
  if (hit) return hit.samples;
  const rng = mulberry32(seed);
  const samples = sampleParent(preset, N, rng);
  rawSampleCache.push({ preset, N, seed, samples });
  return samples;
}

export function getBlockMaxima(
  preset: ParentPreset,
  N: number,
  blockSize: number,
  seed: number = 42,
): Float64Array {
  const samples = getRawSamples(preset, N, seed);
  return blockMaxima(samples, blockSize);
}

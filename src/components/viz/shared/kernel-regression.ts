// =============================================================================
// kernel-regression.ts
//
// Shared math module for the T2 Supervised Learning track.
// First consumer: viz components in the kernel-regression topic
//   (LocalAveragingExplorer / NadarayaWatsonBandwidthExplorer /
//    BiasVarianceMonteCarloExplorer / AmiseUCurveExplorer /
//    BandwidthSelectorComparison / CurseOfDimensionalityExplorer /
//    KernelShapePicker / BoundaryBiasDiagnostic).
//
// Anticipated reuse: local-regression (Fan-Gijbels boundary-bias fix) and
// density-ratio-estimation (KLIEP / LSIF / uLSIF) — both build on the same
// kernel substrate.
//
// All numerical algorithms are direct translations of the verified Python
// notebook at notebooks/kernel-regression/01_kernel_regression.ipynb.
// Where the Python uses np.random / sklearn, this module substitutes the
// seeded Mulberry32 PRNG and Box-Muller Gaussian sampler from
// nonparametric-ml.ts. Numerical agreement against notebook printed outputs
// is verified by src/components/viz/shared/__tests__/verify-kernel-regression.ts.
// =============================================================================

import { mulberry32, gaussianSampler } from './nonparametric-ml';
export { mulberry32, gaussianSampler };

// -----------------------------------------------------------------------------
// Color palette — mirrors the kernel-methods (gaussian-processes) palette so
// that NW / KRR / GP visualizations across T2 and T5 read as a single family.
// Notebook colors (`COLOR_POSTERIOR` ... `COLOR_ACCENT`) match these exactly.
// -----------------------------------------------------------------------------

export const paletteKR = {
  posterior: '#1f77b4', // blue   — NW / KRR / fitted curve
  truth: '#d62728', // red    — ground-truth m(x)
  data: '#7f7f7f', // gray   — raw scatter
  band: '#ff7f0e', // orange — kernel neighborhood band / highlight
  alt: '#2ca02c', // green  — alt estimator (LL, GCV, ...)
  accent: '#9467bd', // purple — tertiary / quartic kernel / decoration
} as const;

export type KRColorKey = keyof typeof paletteKR;

// -----------------------------------------------------------------------------
// Kernels — pure scalar functions u : number → K(u)
// -----------------------------------------------------------------------------

export type KernelFn = (u: number) => number;

const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

export function kGaussian(u: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * u * u);
}

export function kEpanechnikov(u: number): number {
  return Math.abs(u) <= 1 ? 0.75 * (1 - u * u) : 0;
}

export function kBox(u: number): number {
  return Math.abs(u) <= 1 ? 0.5 : 0;
}

export function kTriangular(u: number): number {
  return Math.abs(u) <= 1 ? 1 - Math.abs(u) : 0;
}

export function kQuartic(u: number): number {
  if (Math.abs(u) > 1) return 0;
  const v = 1 - u * u;
  return (15 / 16) * v * v;
}

export type KernelName = 'gaussian' | 'epanechnikov' | 'box' | 'triangular' | 'quartic';

export const KERNELS: Record<KernelName, KernelFn> = {
  gaussian: kGaussian,
  epanechnikov: kEpanechnikov,
  box: kBox,
  triangular: kTriangular,
  quartic: kQuartic,
};

// -----------------------------------------------------------------------------
// Kernel constants — closed-form mu_2(K), R(K), delta(K), C(K)
//
//   delta(K) = (R(K) / mu_2(K)^2)^(1/5)   (canonical-bandwidth scale)
//   C(K)     = R(K)^(4/5) * mu_2(K)^(2/5) (kernel-efficiency constant; smaller = better)
//
// Verified against notebook §2.1 / §7.1:
//   Gaussian      mu_2=1.0,    R=1/(2*sqrt(pi))=0.28209, delta=0.7764, C=0.3633
//   Epanechnikov  mu_2=0.2,    R=0.6,                    delta=1.7188, C=0.3491
//   Box           mu_2=1/3,    R=0.5,                    delta=1.3510, C=0.3701
//   Triangular    mu_2=1/6,    R=2/3,                    delta=1.8882, C=0.3531
//   Quartic       mu_2=1/7,    R=5/7,                    delta=2.0362, C=0.3508
// -----------------------------------------------------------------------------

export interface KernelConstants {
  mu2: number;
  R: number;
  delta: number;
  C: number;
}

function makeConstants(mu2: number, R: number): KernelConstants {
  return {
    mu2,
    R,
    delta: Math.pow(R / (mu2 * mu2), 1 / 5),
    C: Math.pow(R, 4 / 5) * Math.pow(mu2, 2 / 5),
  };
}

export const KERNEL_CONSTANTS: Record<KernelName, KernelConstants> = {
  gaussian: makeConstants(1, 1 / (2 * Math.sqrt(Math.PI))),
  epanechnikov: makeConstants(1 / 5, 3 / 5),
  box: makeConstants(1 / 3, 1 / 2),
  triangular: makeConstants(1 / 6, 2 / 3),
  quartic: makeConstants(1 / 7, 5 / 7),
};

// -----------------------------------------------------------------------------
// Synthetic toy DGP — the §1 univariate example carried through §1-§9
//
//   X ~ Uniform(0, 1),  n = 200 (default)
//   m(x) = sin(2 pi x) + x/2
//   Y = m(X) + N(0, sigma^2),  sigma = 0.2 (default)
// -----------------------------------------------------------------------------

export function mTrueUni(x: number): number {
  return Math.sin(2 * Math.PI * x) + x / 2;
}

export function sampleToyUni(
  n: number,
  sigma: number,
  rng: () => number,
): { X: Float64Array; Y: Float64Array } {
  const gauss = gaussianSampler(rng);
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const xi = rng();
    X[i] = xi;
    Y[i] = mTrueUni(xi) + sigma * gauss();
  }
  return { X, Y };
}

// -----------------------------------------------------------------------------
// Multivariate toy — m(x) = sum_j sin(2 pi x_j) on [0, 1]^d
//
//   tr(grad^2 m)(x) = -4 pi^2 * sum_j sin(2 pi x_j) = -4 pi^2 m(x)
//   theta_{m,f} = int (tr(grad^2 m))^2 dx = 16 pi^4 * sum_j int sin^2(2 pi x_j) dx_j
//                                          = 16 pi^4 * d * (1/2) = 8 pi^4 d
//
// Used only by the §6.3 curse-of-dimensionality viz; flattened (n*d,) array.
// -----------------------------------------------------------------------------

export function mTrueMd(x: Float64Array): number {
  let s = 0;
  for (let j = 0; j < x.length; j++) s += Math.sin(2 * Math.PI * x[j]);
  return s;
}

export function sampleToyMd(
  n: number,
  d: number,
  sigma: number,
  rng: () => number,
): { X: Float64Array; Y: Float64Array } {
  // X is row-major flat: X[i*d + j] = i-th sample's j-th coordinate.
  const gauss = gaussianSampler(rng);
  const X = new Float64Array(n * d);
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let muSum = 0;
    for (let j = 0; j < d; j++) {
      const xij = rng();
      X[i * d + j] = xij;
      muSum += Math.sin(2 * Math.PI * xij);
    }
    Y[i] = muSum + sigma * gauss();
  }
  return { X, Y };
}

// -----------------------------------------------------------------------------
// Nadaraya-Watson estimator — vectorized over xEval
//
//   hat m_h(x) = sum_i K_h(X_i - x) Y_i / sum_i K_h(X_i - x)
//
// Cost: O(n * |xEval|) — one sweep across the training points per query x.
// -----------------------------------------------------------------------------

export function nadarayaWatson(
  X: Float64Array,
  Y: Float64Array,
  xEval: Float64Array,
  h: number,
  K: KernelFn,
): Float64Array {
  const n = X.length;
  const m = xEval.length;
  const out = new Float64Array(m);
  const invH = 1 / h;
  for (let j = 0; j < m; j++) {
    const xj = xEval[j];
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      const w = K((X[i] - xj) * invH) * invH;
      num += w * Y[i];
      den += w;
    }
    out[j] = den > 0 ? num / den : NaN;
  }
  return out;
}

// -----------------------------------------------------------------------------
// Local-linear regression — vectorized over xEval
//
// Solve at each xEval[j]:
//   (alpha, beta) = argmin sum_i K_h(X_i - x) [Y_i - alpha - beta(X_i - x)]^2
//
// With s_j = sum_i w_i (X_i - x)^j and t_j = sum_i w_i (X_i - x)^j Y_i:
//   hat m_h^{LL}(x) = (s_2 t_0 - s_1 t_1) / (s_0 s_2 - s_1^2)
//
// Reproduces both constants and linear functions even at the boundary —
// the bias is O(h^2) uniformly in x, fixing the O(h) NW boundary breakdown.
// -----------------------------------------------------------------------------

export function localLinear(
  X: Float64Array,
  Y: Float64Array,
  xEval: Float64Array,
  h: number,
  K: KernelFn,
): Float64Array {
  const n = X.length;
  const m = xEval.length;
  const out = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    const xj = xEval[j];
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let t0 = 0;
    let t1 = 0;
    for (let i = 0; i < n; i++) {
      const diff = X[i] - xj;
      const w = K(diff / h) / h;
      const wd = w * diff;
      s0 += w;
      s1 += wd;
      s2 += wd * diff;
      t0 += w * Y[i];
      t1 += wd * Y[i];
    }
    const denom = s0 * s2 - s1 * s1;
    out[j] = denom !== 0 ? (s2 * t0 - s1 * t1) / denom : NaN;
  }
  return out;
}

// -----------------------------------------------------------------------------
// Smoother-matrix infrastructure (used by LOO-CV and GCV)
//
//   W[i, j] = K_h(X[j] - X[i]) = K((X[j] - X[i]) / h) / h
//
// Returned as a flat Float64Array to keep allocation tight; address as
// W[i * n + j]. For large n, prefer the matrix-free LOO/GCV computations
// in `looCvScore` / `gcvScore` / `looCvAndGcvScores` which avoid
// materializing the full n*n matrix.
// -----------------------------------------------------------------------------

export function kernelWeightMatrix(
  X: Float64Array,
  h: number,
  K: KernelFn,
): Float64Array {
  const n = X.length;
  const W = new Float64Array(n * n);
  const invH = 1 / h;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      W[i * n + j] = K((X[j] - X[i]) * invH) * invH;
    }
  }
  return W;
}

// -----------------------------------------------------------------------------
// Bandwidth selection
//
// Silverman's rule (KDE-flavored):
//   h_S = 1.06 * min(SD, IQR/1.34) * n^(-1/5)
//
// LOO-CV via the closed-form trick:
//   hat m_h^{-i}(X_i) = (sum_j W_ij Y_j - W_ii Y_i) / (sum_j W_ij - W_ii)
//   CV(h) = (1/n) sum_i (Y_i - hat m_h^{-i}(X_i))^2
//
// GCV (Craven-Wahba 1979):
//   GCV(h) = MSE / (1 - tr(S_h)/n)^2
// where tr(S_h) = sum_i W_ii / sum_j W_ij is the "effective number of params."
// -----------------------------------------------------------------------------

export function silvermanRule(X: Float64Array): number {
  const n = X.length;
  if (n < 2) throw new Error('silvermanRule requires n >= 2.');

  // Sample SD (ddof=1) — match np.std(X, ddof=1).
  let mean = 0;
  for (let i = 0; i < n; i++) mean += X[i];
  mean /= n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = X[i] - mean;
    sumSq += d * d;
  }
  const sd = Math.sqrt(sumSq / (n - 1));

  // IQR via linear interpolation (numpy default).
  // Float64Array.prototype.sort is numeric by default — no comparator needed.
  const sorted = X.slice().sort();
  const q = (p: number): number => {
    const idx = p * (n - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  };
  const iqr = q(0.75) - q(0.25);
  const sigmaHat = Math.min(sd, iqr / 1.34);
  return 1.06 * sigmaHat * Math.pow(n, -1 / 5);
}

// Matrix-free LOO-CV / GCV implementations.
//
// Both selectors only need each row's sum, weighted-Y sum, and the diagonal
// W[i, i] = K_h(0). Materializing the full n*n weight matrix wastes O(n^2)
// memory and triggers heavy GC pressure when called inside an MC sweep
// (visualizations: B replicates × H bandwidth grid = thousands of calls).
// The single-pass form computes the same quantities in O(n^2) time but only
// O(1) memory per call — catches the GCV trace via diag/rowSum directly.
//
// Note: K_h(0) = K(0) / h, a scalar that's constant across all i since
// X[i] - X[i] = 0.

export function looCvScore(
  X: Float64Array,
  Y: Float64Array,
  h: number,
  K: KernelFn,
): number {
  const n = X.length;
  const invH = 1 / h;
  const wii = K(0) * invH;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    let rowSum = 0;
    let weightedY = 0;
    for (let j = 0; j < n; j++) {
      const w = K((X[j] - xi) * invH) * invH;
      rowSum += w;
      weightedY += w * Y[j];
    }
    const denom = rowSum - wii;
    if (denom === 0) return Infinity;
    const mhatLoo = (weightedY - wii * Y[i]) / denom;
    const r = Y[i] - mhatLoo;
    acc += r * r;
  }
  return acc / n;
}

export function gcvScore(
  X: Float64Array,
  Y: Float64Array,
  h: number,
  K: KernelFn,
): number {
  const n = X.length;
  const invH = 1 / h;
  const wii = K(0) * invH;
  let rss = 0;
  let trS = 0;
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    let rowSum = 0;
    let weightedY = 0;
    for (let j = 0; j < n; j++) {
      const w = K((X[j] - xi) * invH) * invH;
      rowSum += w;
      weightedY += w * Y[j];
    }
    if (rowSum === 0) return Infinity;
    const mhat = weightedY / rowSum;
    const r = Y[i] - mhat;
    rss += r * r;
    trS += wii / rowSum;
  }
  const mse = rss / n;
  const denom = (1 - trS / n) ** 2;
  return denom > 0 ? mse / denom : Infinity;
}

/**
 * Joint LOO-CV and GCV scores in a single pass — avoids two separate
 * weight-matrix sweeps when both selectors are needed (e.g., in the §5.5
 * stability viz). Returns {looCv, gcv}. Same O(n^2) time as one call to
 * `looCvScore`, half the work of calling both separately.
 */
export function looCvAndGcvScores(
  X: Float64Array,
  Y: Float64Array,
  h: number,
  K: KernelFn,
): { looCv: number; gcv: number } {
  const n = X.length;
  const invH = 1 / h;
  const wii = K(0) * invH;
  let looAcc = 0;
  let rss = 0;
  let trS = 0;
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    let rowSum = 0;
    let weightedY = 0;
    for (let j = 0; j < n; j++) {
      const w = K((X[j] - xi) * invH) * invH;
      rowSum += w;
      weightedY += w * Y[j];
    }
    if (rowSum === 0) return { looCv: Infinity, gcv: Infinity };
    const denom = rowSum - wii;
    if (denom === 0) return { looCv: Infinity, gcv: Infinity };
    const mhatLoo = (weightedY - wii * Y[i]) / denom;
    const rLoo = Y[i] - mhatLoo;
    looAcc += rLoo * rLoo;
    const mhatFull = weightedY / rowSum;
    const rFull = Y[i] - mhatFull;
    rss += rFull * rFull;
    trS += wii / rowSum;
  }
  const mse = rss / n;
  const gcvDenom = (1 - trS / n) ** 2;
  return {
    looCv: looAcc / n,
    gcv: gcvDenom > 0 ? mse / gcvDenom : Infinity,
  };
}

// -----------------------------------------------------------------------------
// AMISE-optimal bandwidth — univariate, sin(2 pi x) + x/2 toy
//
// theta_{m,f} = int (m''(x))^2 f_X(x) dx = (2pi)^4 * (1/2) = 8 pi^4   (Uniform[0,1] design)
// nu_sigma = sigma^2
//
// For Gaussian kernel:
//   h^* = (R(K) * nu_sigma / (mu_2(K)^2 * theta_{m,f} * n))^(1/5)
// -----------------------------------------------------------------------------

export function hStarAmiseUni(
  n: number,
  sigma: number,
  K: KernelName = 'gaussian',
): number {
  const { mu2, R } = KERNEL_CONSTANTS[K];
  const thetaMf = 8 * Math.pow(Math.PI, 4);
  const nuSigma = sigma * sigma;
  return Math.pow((R * nuSigma) / (mu2 * mu2 * thetaMf * n), 1 / 5);
}

// -----------------------------------------------------------------------------
// Multivariate Nadaraya-Watson with isotropic Gaussian product kernel
//
// X is flat row-major (n*d). Single evaluation point x_eval (length d).
// -----------------------------------------------------------------------------

export function nwMd(
  X: Float64Array,
  Y: Float64Array,
  xEval: Float64Array,
  h: number,
): number {
  const d = xEval.length;
  const n = Y.length;
  const invH2 = 1 / (h * h);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    let dsq = 0;
    for (let j = 0; j < d; j++) {
      const diff = X[i * d + j] - xEval[j];
      dsq += diff * diff;
    }
    const w = Math.exp(-0.5 * dsq * invH2);
    num += w * Y[i];
    den += w;
  }
  return den > 0 ? num / den : NaN;
}

// -----------------------------------------------------------------------------
// AMISE-optimal bandwidth — multivariate sin-sum toy on [0, 1]^d
//
// For m(x) = sum_j sin(2 pi x_j) and Uniform design,  theta_{m,f} = 8 pi^4 d.
// Kernel constants for the d-fold Gaussian product:  mu_2 = 1,  R(K)^d.
//
//   h^(4+d) = d * R^d * sigma^2 / (n * mu_2^2 * theta_{m,f})
//           = d * R^d * sigma^2 / (n * 8 pi^4 d)
//           = R^d * sigma^2 / (8 pi^4 n)
//
// Notebook §6.3 expresses this with a slight refactor (`a = (1/4) * mu_2^2 * theta_{m,f}`,
// `b = R^d sigma^2/n`, `h = (d/4 * b/a)^(1/(4+d))`) which simplifies to the same form.
// -----------------------------------------------------------------------------

export function hStarAmiseMd(d: number, n: number, sigma: number): number {
  const mu2 = 1;
  const R1 = 1 / (2 * Math.sqrt(Math.PI));
  const a = 0.25 * mu2 * mu2 * 16 * Math.pow(Math.PI, 4) * (d / 2); // 0.25 * theta_{m,f} after the 16 pi^4 factor
  const b = (Math.pow(R1, d) * sigma * sigma) / n;
  return Math.pow(((d / 4) * b) / a, 1 / (4 + d));
}

// -----------------------------------------------------------------------------
// Kernel ridge regression with squared-exponential kernel
//
//   K(x, x') = exp(-(x - x')^2 / (2 ell^2))
//   alpha = (K_train + lambda I)^{-1} Y
//   hat f(x_*) = K(x_*, X_train) alpha
//
// O(n^3) for the solve; fine at the n ~ 200 scale of the §1 toy.
// -----------------------------------------------------------------------------

export function squaredExpKernelMatrix(
  X1: Float64Array,
  X2: Float64Array,
  ell: number,
): Float64Array {
  const n1 = X1.length;
  const n2 = X2.length;
  const inv2L2 = 1 / (2 * ell * ell);
  const M = new Float64Array(n1 * n2);
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      const diff = X1[i] - X2[j];
      M[i * n2 + j] = Math.exp(-diff * diff * inv2L2);
    }
  }
  return M;
}

// Solve A x = b in place using Cholesky (A is SPD: K + lambda I).
// A is row-major n*n; modifies A to its lower-triangular Cholesky factor.
// b becomes the solution x.
function choleskySolveInPlace(A: Float64Array, b: Float64Array, n: number): void {
  // Factor A = L L^T (in place: lower triangle of A becomes L).
  for (let j = 0; j < n; j++) {
    let diag = A[j * n + j];
    for (let k = 0; k < j; k++) diag -= A[j * n + k] * A[j * n + k];
    if (diag <= 0) throw new Error('choleskySolveInPlace: non-SPD matrix.');
    const Ljj = Math.sqrt(diag);
    A[j * n + j] = Ljj;
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s / Ljj;
    }
  }
  // Forward solve L y = b.
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= A[i * n + k] * b[k];
    b[i] = s / A[i * n + i];
  }
  // Back solve L^T x = y.
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= A[k * n + i] * b[k];
    b[i] = s / A[i * n + i];
  }
}

export function krrPredict(
  X: Float64Array,
  Y: Float64Array,
  xEval: Float64Array,
  ell: number,
  lambda: number,
): Float64Array {
  const n = X.length;
  const KTrain = squaredExpKernelMatrix(X, X, ell);
  for (let i = 0; i < n; i++) KTrain[i * n + i] += lambda;
  const alpha = new Float64Array(Y);
  choleskySolveInPlace(KTrain, alpha, n);
  const m = xEval.length;
  const KTest = squaredExpKernelMatrix(xEval, X, ell);
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += KTest[i * n + j] * alpha[j];
    out[i] = s;
  }
  return out;
}

// -----------------------------------------------------------------------------
// Split-conformal prediction with NW base estimator
//
// 1. Permute (X, Y) and split half/half into training T and calibration C.
// 2. Fit hat m_h on T; predict at calibration X to compute |Y_C - hat m(X_C)|.
// 3. q_hat = ceil((|C| + 1) (1 - alpha))-th order statistic of residuals.
// 4. PI(x) = [hat m_h(x) - q_hat, hat m_h(x) + q_hat].
//
// Marginal coverage: Pr[Y in PI(X)] >= 1 - alpha for any joint distribution.
// -----------------------------------------------------------------------------

export interface SplitConformalResult {
  lower: Float64Array;
  upper: Float64Array;
  mean: Float64Array;
  qHat: number;
}

export function splitConformalNw(
  X: Float64Array,
  Y: Float64Array,
  xEval: Float64Array,
  h: number,
  alpha: number,
  rng: () => number,
  K: KernelFn = kGaussian,
): SplitConformalResult {
  const n = X.length;

  // Fisher-Yates shuffle on indices 0..n-1.
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  const nT = Math.floor(n / 2);
  const nC = n - nT;
  const XT = new Float64Array(nT);
  const YT = new Float64Array(nT);
  const XC = new Float64Array(nC);
  const YC = new Float64Array(nC);
  for (let i = 0; i < nT; i++) {
    XT[i] = X[idx[i]];
    YT[i] = Y[idx[i]];
  }
  for (let i = 0; i < nC; i++) {
    XC[i] = X[idx[nT + i]];
    YC[i] = Y[idx[nT + i]];
  }

  const mhatAtC = nadarayaWatson(XT, YT, XC, h, K);
  const residuals = new Float64Array(nC);
  for (let i = 0; i < nC; i++) residuals[i] = Math.abs(YC[i] - mhatAtC[i]);

  // q_hat = ceil((nC + 1)(1 - alpha))-th smallest residual (1-indexed).
  // Float64Array.prototype.sort is numeric by default — no comparator needed.
  const sorted = residuals.slice().sort();
  let qIdx = Math.ceil((nC + 1) * (1 - alpha)) - 1;
  if (qIdx >= nC) qIdx = nC - 1;
  if (qIdx < 0) qIdx = 0;
  const qHat = sorted[qIdx];

  const mhatX = nadarayaWatson(XT, YT, xEval, h, K);
  const lower = new Float64Array(xEval.length);
  const upper = new Float64Array(xEval.length);
  for (let i = 0; i < xEval.length; i++) {
    lower[i] = mhatX[i] - qHat;
    upper[i] = mhatX[i] + qHat;
  }
  return { lower, upper, mean: mhatX, qHat };
}

// =============================================================================
// LOCAL POLYNOMIAL REGRESSION (degree-p extension for the local-regression topic)
//
// Direct TypeScript ports of the verified Python helpers from
//   notebooks/local-regression/01_local_regression.ipynb.
// Verified by src/components/viz/shared/__tests__/verify-local-regression.ts
// against the notebook's printed numerical outputs (Cells 21, 27, 31, 36, 47).
// =============================================================================

// -----------------------------------------------------------------------------
// Numerical integration — composite Simpson on [a, b] with N (even) panels.
// Used for kernel-moment integrals and equivalent-kernel constants.
//
// For Gaussian K on [-8, 8] with N = 4000 panels, the integration error for
// moments up to order 6 is well below 1e-12 — exceeds all verification
// tolerances. For compact-support kernels, the natural [-1, 1] support is used.
// -----------------------------------------------------------------------------

function simpsonIntegrate(
  f: (u: number) => number,
  a: number,
  b: number,
  N: number = 2000,
): number {
  if (N % 2 !== 0) N += 1;
  const h = (b - a) / N;
  let s = f(a) + f(b);
  for (let i = 1; i < N; i++) {
    const x = a + i * h;
    s += (i % 2 === 0 ? 2 : 4) * f(x);
  }
  return (s * h) / 3;
}

// Map kernel function reference to its natural integration domain, optionally
// truncated on the left at -c (boundary case). The Gaussian is treated as
// effectively bounded by ±8 (tail contribution < 1e-15); the four compact
// kernels integrate exactly on [-min(c, 1), 1] ∩ support.
function kernelDomain(K: KernelFn, c: number): [number, number] {
  if (K === kGaussian) {
    const lo = isFinite(c) ? Math.max(-c, -8) : -8;
    return [lo, 8];
  }
  if (K === silvermanKernel) {
    // Silverman kernel decays as exp(-|u|/√2); ±30 captures > 1 - 1e-9 of mass.
    // Treated separately from compact kernels because its support is unbounded.
    const lo = isFinite(c) ? Math.max(-c, -30) : -30;
    return [lo, 30];
  }
  // Compact kernels: epanechnikov, box, triangular, quartic — support [-1, 1].
  const lo = isFinite(c) ? Math.max(-Math.min(c, 1), -1) : -1;
  return [lo, 1];
}

// -----------------------------------------------------------------------------
// Linear-system solver — Gaussian elimination with partial pivoting.
//
// Used by the per-evaluation-point WLS solves in localPolynomial and the
// moment-matrix inversion in equivalentKernel. The matrices are tiny
// (typically (p+1)×(p+1) with p ≤ 5, so ≤ 6×6), making a generic solver more
// robust than the existing choleskySolveInPlace which assumes SPD.
// -----------------------------------------------------------------------------

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  // Augment [A | b] in a fresh matrix to avoid mutating inputs.
  const M: number[][] = A.map((row, i) => [...row, b[i]]);
  for (let k = 0; k < n; k++) {
    // Partial pivot: find the row with max |M[i][k]| at or below k.
    let pivotRow = k;
    let pivotVal = Math.abs(M[k][k]);
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(M[i][k]) > pivotVal) {
        pivotVal = Math.abs(M[i][k]);
        pivotRow = i;
      }
    }
    if (pivotRow !== k) {
      [M[k], M[pivotRow]] = [M[pivotRow], M[k]];
    }
    if (Math.abs(M[k][k]) < 1e-14) {
      throw new Error(`solveLinearSystem: singular at row ${k}`);
    }
    // Eliminate.
    for (let i = k + 1; i < n; i++) {
      const factor = M[i][k] / M[k][k];
      for (let j = k; j <= n; j++) M[i][j] -= factor * M[k][j];
    }
  }
  // Back-substitute.
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// -----------------------------------------------------------------------------
// Kernel moments and the equivalent kernel
//
//   mu_j^(c)(K)        = ∫_{-c}^{∞} u^j K(u) du
//   S_p^(c)            = [mu_{j+k}^(c)(K)]_{j,k=0}^{p}    ((p+1)×(p+1) Hankel)
//   K^*_p^(c)(u)       = e_1^T (S_p^(c))^{-1} (1, u, ..., u^p)^T K(u)
//
// The boundary parameter c controls truncation: c = ∞ gives interior moments
// (full kernel mass); c = 0 gives the strict-boundary case (only u ≥ 0 mass).
// -----------------------------------------------------------------------------

export function kernelMoment(K: KernelFn, j: number, c: number = Infinity): number {
  const [a, b] = kernelDomain(K, c);
  return simpsonIntegrate((u) => Math.pow(u, j) * K(u), a, b, 4000);
}

export function kernelMomentMatrix(
  K: KernelFn,
  p: number,
  c: number = Infinity,
): number[][] {
  const moments: number[] = [];
  for (let j = 0; j <= 2 * p; j++) moments.push(kernelMoment(K, j, c));
  const S: number[][] = [];
  for (let j = 0; j <= p; j++) {
    const row: number[] = [];
    for (let k = 0; k <= p; k++) row.push(moments[j + k]);
    S.push(row);
  }
  return S;
}

export function equivalentKernel(
  K: KernelFn,
  p: number,
  c: number = Infinity,
): KernelFn {
  const S = kernelMomentMatrix(K, p, c);
  const e1 = new Array(p + 1).fill(0);
  e1[0] = 1;
  // First row of S^{-1} ↔ S a = e_1 (since S is symmetric, S^T = S).
  const a = solveLinearSystem(S, e1);
  return (u: number) => {
    let poly = 0;
    let uk = 1;
    for (let k = 0; k <= p; k++) {
      poly += a[k] * uk;
      uk *= u;
    }
    return poly * K(u);
  };
}

// b_p^(c)(K) = ∫_{-c}^{∞} u^{p+1} K^*_p^(c)(u) du
//            = "first surviving moment" of the equivalent kernel.
// Drives the Ruppert–Wand (1994) leading-order bias formula.
export function biasConstant(K: KernelFn, p: number, c: number = Infinity): number {
  const Kstar = equivalentKernel(K, p, c);
  const [a, b] = kernelDomain(K, c);
  return simpsonIntegrate((u) => Math.pow(u, p + 1) * Kstar(u), a, b, 4000);
}

// R^*_p^(c)(K) = ∫_{-c}^{∞} K^*_p^(c)(u)^2 du
//              — variance constant; grows with p as K^*_p becomes more oscillatory.
export function varianceConstant(K: KernelFn, p: number, c: number = Infinity): number {
  const Kstar = equivalentKernel(K, p, c);
  const [a, b] = kernelDomain(K, c);
  return simpsonIntegrate((u) => Kstar(u) * Kstar(u), a, b, 4000);
}

// Boundary-bias constant — alias for biasConstant with c = 0 default.
export function boundaryBiasConstant(K: KernelFn, p: number, c: number = 0): number {
  return biasConstant(K, p, c);
}

// -----------------------------------------------------------------------------
// Local polynomial regression — degree-p WLS at each evaluation point.
//
// localPolynomialCoefs returns the full coefficient vector (p+1 entries per
//   query) in the *unscaled* basis: beta_j ≈ m^(j)(x)/j!  for j = 0, ..., p.
// localPolynomial is the function-value-only convenience wrapper.
//
// Implementation: per-query (p+1)×(p+1) Cholesky/Gaussian solve in scaled
// coordinates u = (X_i - x)/h. The intercept beta_0 is invariant under this
// rescaling; higher coefficients are rescaled on output: beta_j_unscaled =
// beta_j_scaled / h^j (so that derivative readout m^(j) = j! · beta_j is
// dimensionally correct).
// -----------------------------------------------------------------------------

export function localPolynomialCoefs(
  X: Float64Array,
  Y: Float64Array,
  xEval: Float64Array,
  h: number,
  p: number,
  K: KernelFn = kGaussian,
): Float64Array {
  // Returns flat row-major (G * (p+1)) — coefs for query g start at index g*(p+1).
  const G = xEval.length;
  const n = X.length;
  const dim = p + 1;
  const out = new Float64Array(G * dim);

  // Hoist per-evaluation buffers outside the G-loop to eliminate G*n GC churn
  // on slider-driven hot paths (catches PR #80 perf review feedback).
  const A: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  const bv = new Array(dim).fill(0);
  const phi = new Array(dim);

  for (let g = 0; g < G; g++) {
    const xg = xEval[g];
    for (let r = 0; r < dim; r++) A[r].fill(0);
    bv.fill(0);
    for (let i = 0; i < n; i++) {
      const u = (X[i] - xg) / h;
      const w = K(u) / h;
      // phi[k] = u^k.
      let uk = 1;
      for (let k = 0; k < dim; k++) {
        phi[k] = uk;
        uk *= u;
      }
      for (let r = 0; r < dim; r++) {
        bv[r] += w * phi[r] * Y[i];
        for (let s = 0; s < dim; s++) {
          A[r][s] += w * phi[r] * phi[s];
        }
      }
    }
    let beta: number[];
    try {
      beta = solveLinearSystem(A, bv);
    } catch {
      for (let k = 0; k < dim; k++) out[g * dim + k] = NaN;
      continue;
    }
    // Rescale to unscaled basis.
    for (let k = 0; k < dim; k++) {
      out[g * dim + k] = beta[k] / Math.pow(h, k);
    }
  }
  return out;
}

export function localPolynomial(
  X: Float64Array,
  Y: Float64Array,
  xEval: Float64Array,
  h: number,
  p: number,
  K: KernelFn = kGaussian,
): Float64Array {
  const coefs = localPolynomialCoefs(X, Y, xEval, h, p, K);
  const dim = p + 1;
  const G = xEval.length;
  const out = new Float64Array(G);
  for (let g = 0; g < G; g++) out[g] = coefs[g * dim + 0];
  return out;
}

// -----------------------------------------------------------------------------
// Smoother-matrix diagonal at degree p
//
//   H_ii = K(0)/h * [(X_{X_i}^T W_{X_i} X_{X_i})^{-1}]_{0, 0}
//
// At evaluation point x = X_i, the (i, i) self-influence is the weighted
// influence of observation i on its own fit. The trace ∑ H_ii is the
// "effective degrees of freedom" of the local-polynomial smoother — used by
// LOO-CV and GCV at general p (matrix-free version below).
// -----------------------------------------------------------------------------

export function smootherDiagonal(
  X: Float64Array,
  h: number,
  p: number,
  K: KernelFn = kGaussian,
): Float64Array {
  const n = X.length;
  const dim = p + 1;
  const out = new Float64Array(n);
  const K0overH = K(0) / h;
  // Hoist per-i buffers outside the n-loop (PR #80 perf review).
  const A: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  const phi = new Array(dim);
  const e1 = new Array(dim).fill(0);
  e1[0] = 1;
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    for (let r = 0; r < dim; r++) A[r].fill(0);
    for (let j = 0; j < n; j++) {
      const u = (X[j] - xi) / h;
      const w = K(u) / h;
      let uk = 1;
      for (let k = 0; k < dim; k++) {
        phi[k] = uk;
        uk *= u;
      }
      for (let r = 0; r < dim; r++) {
        for (let s = 0; s < dim; s++) A[r][s] += w * phi[r] * phi[s];
      }
    }
    // Solve A·v = e_1 and pick v[0]; then H_ii = K(0)/h · v[0].
    let v: number[];
    try {
      v = solveLinearSystem(A, e1);
    } catch {
      out[i] = NaN;
      continue;
    }
    out[i] = K0overH * v[0];
  }
  return out;
}

// -----------------------------------------------------------------------------
// LOO-CV and GCV at degree p — closed-form via the smoother diagonal
//
// LOO closed form (Hastie–Tibshirani 1990): for any linear smoother,
//   Y_i - hat m^(-i)(X_i) = (Y_i - hat m(X_i)) / (1 - H_ii).
// At p > 0, H_ii is computed via smootherDiagonal (not the simple K(0)/sum_j W_ij
// shortcut that works for NW only).
// -----------------------------------------------------------------------------

export function localPolynomialLooCv(
  X: Float64Array,
  Y: Float64Array,
  h: number,
  p: number,
  K: KernelFn = kGaussian,
): number {
  const n = X.length;
  const fit = localPolynomial(X, Y, X, h, p, K);
  const diag = smootherDiagonal(X, h, p, K);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const denom = 1 - diag[i];
    if (denom === 0) return Infinity;
    const r = (Y[i] - fit[i]) / denom;
    acc += r * r;
  }
  return acc / n;
}

export function localPolynomialGcv(
  X: Float64Array,
  Y: Float64Array,
  h: number,
  p: number,
  K: KernelFn = kGaussian,
): number {
  const n = X.length;
  const fit = localPolynomial(X, Y, X, h, p, K);
  const diag = smootherDiagonal(X, h, p, K);
  let rss = 0;
  let trS = 0;
  for (let i = 0; i < n; i++) {
    const r = Y[i] - fit[i];
    rss += r * r;
    trS += diag[i];
  }
  const mse = rss / n;
  const denom = (1 - trS / n) ** 2;
  return denom > 0 ? mse / denom : Infinity;
}

// -----------------------------------------------------------------------------
// Silverman's equivalent variable kernel (Silverman 1984, Theorem 5)
//
//   K_S(u) = (1/2) exp(-|u|/√2) sin(|u|/√2 + π/4)
//
// Fourth-order kernel — matches K^*_3 (the local-cubic equivalent kernel)
// in moments: ∫ K_S = 1, ∫ u^j K_S = 0 for j = 1, 2, 3, ∫ u^4 K_S ≠ 0.
// Drives the asymptotic equivalence between cubic smoothing splines and
// local-cubic regression in §10.
// -----------------------------------------------------------------------------

export function silvermanKernel(u: number): number {
  const a = Math.abs(u);
  const r = a / Math.SQRT2;
  return 0.5 * Math.exp(-r) * Math.sin(r + Math.PI / 4);
}

// -----------------------------------------------------------------------------
// Multivariate local polynomial (degree-p in R^d via polynomial-feature design)
//
// Generates multi-indices α ∈ N^d with |α| ≤ p, builds the design row
// φ_α(X_i - x) = ∏_j (X_{ij} - x_j)^{α_j} for each evaluation point, and
// solves the (M × M) WLS where M = C(d+p, p). Default kernel is the isotropic
// Gaussian product with per-coordinate bandwidths H[j].
//
// X is row-major flat: X[i*d + j] = i-th sample's j-th coordinate (length n*d).
// xEval is row-major flat: xEval[g*d + j] = g-th query's j-th coordinate (length G*d).
// -----------------------------------------------------------------------------

function multiIndices(d: number, p: number): number[][] {
  // All α ∈ N^d with α_1 + ... + α_d ≤ p, in lex order with (0,...,0) first.
  const result: number[][] = [];
  function recurse(prefix: number[], remaining: number, depth: number) {
    if (depth === d) {
      result.push([...prefix]);
      return;
    }
    for (let k = 0; k <= remaining; k++) {
      prefix.push(k);
      recurse(prefix, remaining - k, depth + 1);
      prefix.pop();
    }
  }
  recurse([], p, 0);
  return result;
}

export function localPolynomialMd(
  X: Float64Array,
  Y: Float64Array,
  xEval: Float64Array,
  H: number[],
  p: number,
): Float64Array {
  const d = H.length;
  const n = Y.length;
  const G = xEval.length / d;
  const indices = multiIndices(d, p);
  const M = indices.length;
  const out = new Float64Array(G);

  // Constant kernel-normalization factor: (2π)^{-d/2} / ∏H_j.
  let prodH = 1;
  for (let j = 0; j < d; j++) prodH *= H[j];
  const kernelNorm = 1 / (Math.pow(2 * Math.PI, d / 2) * prodH);

  // Hoist per-(g, i) buffers outside the loops — multivariate is the most
  // allocation-heavy of the four hot paths (PR #80 perf review).
  const A: number[][] = Array.from({ length: M }, () => new Array(M).fill(0));
  const bv = new Array(M).fill(0);
  const diff = new Array(d);
  const phi = new Array(M);

  for (let g = 0; g < G; g++) {
    for (let r = 0; r < M; r++) A[r].fill(0);
    bv.fill(0);
    for (let i = 0; i < n; i++) {
      // Per-coordinate scaled diff and Gaussian weight.
      let logW = 0;
      for (let j = 0; j < d; j++) {
        const dij = X[i * d + j] - xEval[g * d + j];
        diff[j] = dij;
        const uj = dij / H[j];
        logW -= 0.5 * uj * uj;
      }
      const w = kernelNorm * Math.exp(logW);
      // Design row in *unscaled* basis: φ_α = ∏_j (X_{ij} - x_{gj})^{α_j}.
      for (let m = 0; m < M; m++) {
        let val = 1;
        const alpha = indices[m];
        for (let j = 0; j < d; j++) {
          if (alpha[j] > 0) val *= Math.pow(diff[j], alpha[j]);
        }
        phi[m] = val;
      }
      for (let r = 0; r < M; r++) {
        bv[r] += w * phi[r] * Y[i];
        for (let s = 0; s < M; s++) {
          A[r][s] += w * phi[r] * phi[s];
        }
      }
    }
    let beta: number[];
    try {
      beta = solveLinearSystem(A, bv);
    } catch {
      out[g] = NaN;
      continue;
    }
    // The constant multi-index (0,...,0) is at position 0 → β_0 is the function-value estimate.
    out[g] = beta[0];
  }
  return out;
}

// -----------------------------------------------------------------------------
// Backfitting GAM (Friedman & Stuetzle 1981; Buja, Hastie & Tibshirani 1989)
//
// Iteratively smooth partial residuals against each coordinate via a univariate
// local polynomial of degree p. Convergence: max ‖m̂_j^{new} - m̂_j^{old}‖_∞ < tol.
//
// Returns alpha (intercept), components (n × d, evaluated at design points),
// iteration count, and the per-iteration max-component-change history (used by
// the §11 backfitting-convergence figure).
// -----------------------------------------------------------------------------

export interface BackfitGamResult {
  alpha: number;
  components: number[][];   // n × d row-major: components[i][j] = m̂_j(X_{ij})
  iterations: number;
  history: number[];        // max-component-change per iteration
}

export function backfitGam(
  X: Float64Array,
  Y: Float64Array,
  d: number,
  h: number,
  p: number,
  K: KernelFn = kGaussian,
  tol: number = 1e-5,
  maxIter: number = 50,
): BackfitGamResult {
  const n = Y.length;
  // Initialize alpha = mean(Y), components = 0.
  let alpha = 0;
  for (let i = 0; i < n; i++) alpha += Y[i];
  alpha /= n;
  const components: number[][] = Array.from({ length: n }, () => new Array(d).fill(0));
  const history: number[] = [];

  // Cache per-coordinate design vectors.
  const xCols: Float64Array[] = [];
  for (let j = 0; j < d; j++) {
    const col = new Float64Array(n);
    for (let i = 0; i < n; i++) col[i] = X[i * d + j];
    xCols.push(col);
  }

  // Hoist the partial-residual buffer outside the iter and j loops
  // (PR #80 perf review).
  const rj = new Float64Array(n);
  for (let iter = 0; iter < maxIter; iter++) {
    let maxDelta = 0;
    for (let j = 0; j < d; j++) {
      // Partial residual against coordinate j.
      for (let i = 0; i < n; i++) {
        let s = Y[i] - alpha;
        for (let k = 0; k < d; k++) if (k !== j) s -= components[i][k];
        rj[i] = s;
      }
      // Smooth r^(j) against X_{·,j} via local polynomial.
      const fit = localPolynomial(xCols[j], rj, xCols[j], h, p, K);
      // Center: subtract the mean to enforce E[m̂_j(X_j)] ≈ 0.
      let mean = 0;
      for (let i = 0; i < n; i++) mean += fit[i];
      mean /= n;
      let delta = 0;
      for (let i = 0; i < n; i++) {
        const newVal = fit[i] - mean;
        const change = Math.abs(newVal - components[i][j]);
        if (change > delta) delta = change;
        components[i][j] = newVal;
      }
      if (delta > maxDelta) maxDelta = delta;
    }
    history.push(maxDelta);
    if (maxDelta < tol) {
      return { alpha, components, iterations: iter + 1, history };
    }
  }
  return { alpha, components, iterations: maxIter, history };
}

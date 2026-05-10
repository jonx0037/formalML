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
  for (let j = 0; j < m; j++) {
    const xj = xEval[j];
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      const w = K((X[i] - xj) / h) / h;
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
//   W[i, j] = K_h(X[j] - X[i]) / h    (row-major, length n*n)
//
// Returned as a flat Float64Array to keep allocation tight; address as
// W[i * n + j].
// -----------------------------------------------------------------------------

export function kernelWeightMatrix(
  X: Float64Array,
  h: number,
  K: KernelFn,
): Float64Array {
  const n = X.length;
  const W = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      W[i * n + j] = K((X[j] - X[i]) / h) / h;
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
  const sorted = Array.from(X).slice().sort((a, b) => a - b);
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

export function looCvScore(
  X: Float64Array,
  Y: Float64Array,
  h: number,
  K: KernelFn,
): number {
  const n = X.length;
  const W = kernelWeightMatrix(X, h, K);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    let weightedY = 0;
    for (let j = 0; j < n; j++) {
      const w = W[i * n + j];
      rowSum += w;
      weightedY += w * Y[j];
    }
    const wii = W[i * n + i];
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
  const W = kernelWeightMatrix(X, h, K);
  let rss = 0;
  let trS = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    let weightedY = 0;
    for (let j = 0; j < n; j++) {
      const w = W[i * n + j];
      rowSum += w;
      weightedY += w * Y[j];
    }
    if (rowSum === 0) return Infinity;
    const mhat = weightedY / rowSum;
    const r = Y[i] - mhat;
    rss += r * r;
    trS += W[i * n + i] / rowSum;
  }
  const mse = rss / n;
  const denom = (1 - trS / n) ** 2;
  return denom > 0 ? mse / denom : Infinity;
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
  const sorted = Array.from(residuals).sort((a, b) => a - b);
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

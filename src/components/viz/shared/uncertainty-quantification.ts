// =============================================================================
// uncertainty-quantification.ts
//
// Shared math module for the T6 Learning Theory & Methodology track,
// uncertainty-quantification topic. Consumers:
//   AleatoricEpistemicDecomposer, ReliabilityDiagramExplorer,
//   ProperScoringRulesComparator, PredictiveIntervalParadigms,
//   BootstrapUQDemo, BNNApproximationsDemo, ConformalUQCalibrator,
//   DeepEnsembleSizeExplorer, OverparamRegimeUQ,
//   DistributionShiftDegradation, TemperatureScalingDemo.
//
// All numerical algorithms are direct translations of the verified Python
// notebook at notebooks/uncertainty-quantification/01_uncertainty_quantification.ipynb.
// Closed-form quantities match the notebook bit-for-bit (modulo IEEE-754 noise);
// Monte-Carlo quantities use the seeded Mulberry32 PRNG re-exported from
// generalization-bounds.ts and so do not reproduce the notebook's numpy.default_rng
// (PCG64) sequence — they match in distribution and shape only.
//
// Numerical agreement is verified by
//   src/components/viz/shared/__tests__/verify-uncertainty-quantification.ts.
// =============================================================================

import { gaussianFrom, mulberry32 } from './generalization-bounds';

export { gaussianFrom, mulberry32 };

// -----------------------------------------------------------------------------
// Notebook seed and constants
// -----------------------------------------------------------------------------

export const NOTEBOOK_SEED = 20260514;
export const POLY_SCALE = 3.0;
export const DEFAULT_DEGREE = 7;
export const DEFAULT_TAU_PRIOR = 10.0;
export const DEFAULT_ALPHA = 0.10;

// One-color-per-paradigm palette mirroring the notebook's PALETTE dict.
export const paletteUQ = {
  data: '#1f2937',
  class0: '#3b82f6',
  class1: '#ef4444',
  model: '#6b7280',
  perfect: '#94a3b8',
  miscal: '#dc2626',
  gap: '#f97316',
  aleatoric: '#0ea5e9',
  epistemic: '#8b5cf6',
  total: '#1e293b',
  bootstrap: '#10b981',
  bayes: '#8b5cf6',
  conformal: '#06b6d4',
  ensemble: '#f59e0b',
  shifted: '#dc2626',
} as const;

export type UQColorKey = keyof typeof paletteUQ;

// -----------------------------------------------------------------------------
// Running heteroscedastic toy: f(x) = sin(x), σ(x) = 0.1 + 0.5|x|.
// -----------------------------------------------------------------------------

export function fTrue(x: number): number {
  return Math.sin(x);
}

export function sigmaTrue(x: number): number {
  return 0.1 + 0.5 * Math.abs(x);
}

/** Misspecification-demo low-noise variant from §2.5. */
export function sigmaDemo(x: number): number {
  return 0.03 + 0.03 * Math.abs(x);
}

/** Sample a heteroscedastic regression dataset Y = f(X) + N(0, σ(X)²). */
export function sampleHetero(
  n: number,
  xMin: number,
  xMax: number,
  rng: () => number,
  sigmaFn: (x: number) => number = sigmaTrue,
): { X: number[]; y: number[] } {
  const gauss = gaussianFrom(rng);
  const X: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = xMin + (xMax - xMin) * rng();
    const yi = fTrue(xi) + sigmaFn(xi) * gauss();
    X.push(xi);
    y.push(yi);
  }
  return { X, y };
}

// -----------------------------------------------------------------------------
// Linear algebra: small dense matrices stored as number[][]. The Bayesian-poly
// posterior covariance is at most (degree+1)×(degree+1) = 8×8 at the notebook's
// DEGREE=7, so direct inversion is fine. For the §10 ridge sweep we move up to
// p×p with p up to 200; still fast enough on slider commit.
// -----------------------------------------------------------------------------

/** Cholesky factor L of a symmetric positive-definite matrix A (A = L Lᵀ).
 *  Throws if A is not SPD. Adds `jitter` to the diagonal first if specified. */
export function cholesky(A: number[][], jitter: number = 0): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j] + (i === j ? jitter : 0);
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum <= 0) throw new Error(`cholesky: non-SPD at row ${i} (got ${sum})`);
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  return L;
}

/** Solve L y = b for lower-triangular L. */
export function solveLower(L: number[][], b: number[]): number[] {
  const n = b.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
    y[i] = s / L[i][i];
  }
  return y;
}

/** Solve Lᵀ x = y for lower-triangular L. */
export function solveUpper(L: number[][], y: number[]): number[] {
  const n = y.length;
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= L[j][i] * x[j];
    x[i] = s / L[i][i];
  }
  return x;
}

/** Solve A x = b for SPD A via Cholesky. */
export function solveSPD(A: number[][], b: number[], jitter: number = 0): number[] {
  const L = cholesky(A, jitter);
  return solveUpper(L, solveLower(L, b));
}

/** Invert an SPD matrix via Cholesky: A⁻¹ = L⁻ᵀ L⁻¹. */
export function invSPD(A: number[][], jitter: number = 0): number[][] {
  const n = A.length;
  const L = cholesky(A, jitter);
  const Linv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j++) {
    const e = new Array(n).fill(0);
    e[j] = 1;
    const col = solveLower(L, e);
    for (let i = 0; i < n; i++) Linv[i][j] = col[i];
  }
  // A⁻¹ = Linv ᵀ Linv.
  const Ainv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += Linv[k][i] * Linv[k][j];
      Ainv[i][j] = s;
      Ainv[j][i] = s;
    }
  }
  return Ainv;
}

/** Matrix-vector: y = A x. */
export function matVec(A: number[][], x: number[]): number[] {
  const m = A.length;
  const y = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    let s = 0;
    const Ai = A[i];
    for (let j = 0; j < x.length; j++) s += Ai[j] * x[j];
    y[i] = s;
  }
  return y;
}

/** Quadratic form xᵀ A x for symmetric A. */
export function quadForm(x: number[], A: number[][]): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    let row = 0;
    const Ai = A[i];
    for (let j = 0; j < x.length; j++) row += Ai[j] * x[j];
    s += x[i] * row;
  }
  return s;
}

// -----------------------------------------------------------------------------
// Polynomial basis Φ on pre-scaled input X/POLY_SCALE.
// Notebook: np.vander(X/POLY_SCALE, degree+1, increasing=True).
// -----------------------------------------------------------------------------

export function phi(X: number[], degree: number): number[][] {
  const n = X.length;
  const Phi: number[][] = Array.from({ length: n }, () => new Array(degree + 1).fill(0));
  for (let i = 0; i < n; i++) {
    const xs = X[i] / POLY_SCALE;
    let v = 1;
    for (let k = 0; k <= degree; k++) {
      Phi[i][k] = v;
      v *= xs;
    }
  }
  return Phi;
}

// -----------------------------------------------------------------------------
// §1 + §3 setup-cell helpers: ECE, MCE, equal-frequency, sharpness, PIT.
// -----------------------------------------------------------------------------

export type ReliabilityBins = {
  /** Mean predicted probability per bin (NaN if bin empty). */
  binConf: number[];
  /** Empirical positive rate per bin (NaN if bin empty). */
  binAcc: number[];
  /** Count-weighted bin proportions (sum = 1). */
  binW: number[];
};

/** Equal-width reliability binning on [0,1] with K bins (default K=10). */
export function reliabilityBins(
  yTrue: number[],
  yProba: number[],
  nBins: number = 10,
): ReliabilityBins {
  const binConf: number[] = new Array(nBins).fill(NaN);
  const binAcc: number[] = new Array(nBins).fill(NaN);
  const binN: number[] = new Array(nBins).fill(0);
  const sumConf: number[] = new Array(nBins).fill(0);
  const sumAcc: number[] = new Array(nBins).fill(0);
  for (let i = 0; i < yProba.length; i++) {
    const p = yProba[i];
    let b = Math.floor(p * nBins);
    if (b < 0) b = 0;
    if (b >= nBins) b = nBins - 1;
    binN[b]++;
    sumConf[b] += p;
    sumAcc[b] += yTrue[i];
  }
  for (let b = 0; b < nBins; b++) {
    if (binN[b] > 0) {
      binConf[b] = sumConf[b] / binN[b];
      binAcc[b] = sumAcc[b] / binN[b];
    }
  }
  const total = binN.reduce((a, c) => a + c, 0);
  const binW = total > 0 ? binN.map((c) => c / total) : binN.map(() => 0);
  return { binConf, binAcc, binW };
}

/** Expected calibration error: count-weighted mean absolute calibration gap. */
export function ece(yTrue: number[], yProba: number[], nBins: number = 10): number {
  const { binConf, binAcc, binW } = reliabilityBins(yTrue, yProba, nBins);
  let s = 0;
  for (let b = 0; b < nBins; b++) {
    if (!Number.isNaN(binConf[b])) s += binW[b] * Math.abs(binConf[b] - binAcc[b]);
  }
  return s;
}

/** Maximum calibration error: worst single-bin gap. */
export function mce(yTrue: number[], yProba: number[], nBins: number = 10): number {
  const { binConf, binAcc, binW } = reliabilityBins(yTrue, yProba, nBins);
  let m = 0;
  for (let b = 0; b < nBins; b++) {
    if (!Number.isNaN(binConf[b]) && binW[b] > 0) {
      const gap = Math.abs(binConf[b] - binAcc[b]);
      if (gap > m) m = gap;
    }
  }
  return m;
}

/** Equal-frequency reliability binning: quantile bins of equal count. */
export function equalFreqBins(
  yTrue: number[],
  yProba: number[],
  nBins: number = 10,
): ReliabilityBins {
  const n = yProba.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => yProba[a] - yProba[b]);
  const binSize = Math.floor(n / nBins);
  const binConf: number[] = new Array(nBins).fill(NaN);
  const binAcc: number[] = new Array(nBins).fill(NaN);
  const binN: number[] = new Array(nBins).fill(0);
  for (let k = 0; k < nBins; k++) {
    const lo = k * binSize;
    const hi = k === nBins - 1 ? n : (k + 1) * binSize;
    const len = hi - lo;
    if (len === 0) continue;
    let sp = 0;
    let sy = 0;
    for (let j = lo; j < hi; j++) {
      const i = idx[j];
      sp += yProba[i];
      sy += yTrue[i];
    }
    binConf[k] = sp / len;
    binAcc[k] = sy / len;
    binN[k] = len;
  }
  const total = binN.reduce((a, c) => a + c, 0);
  const binW = total > 0 ? binN.map((c) => c / total) : binN.map(() => 0);
  return { binConf, binAcc, binW };
}

/** ECE using equal-frequency bins. */
export function eceEqFreq(yTrue: number[], yProba: number[], nBins: number = 10): number {
  const { binConf, binAcc, binW } = equalFreqBins(yTrue, yProba, nBins);
  let s = 0;
  for (let b = 0; b < nBins; b++) {
    if (!Number.isNaN(binConf[b])) s += binW[b] * Math.abs(binConf[b] - binAcc[b]);
  }
  return s;
}

/** Sharpness: variance of the predicted probabilities. */
export function sharpness(yProba: number[]): number {
  if (yProba.length === 0) return 0;
  const m = yProba.reduce((a, c) => a + c, 0) / yProba.length;
  let s = 0;
  for (const p of yProba) s += (p - m) * (p - m);
  return s / yProba.length;
}

/** Standard normal CDF via erf approximation (Abramowitz–Stegun 7.1.26). */
export function normCdf(z: number): number {
  // erf approximation accurate to ~1.5e-7 over the real line.
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const y =
    1 -
    ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

/** Standard normal pdf. */
export function normPdf(z: number): number {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
}

/** PIT values for a Gaussian predictive Y* ~ N(mu, sigma²). */
export function pitValuesGaussian(y: number[], mu: number[], sigma: number[]): number[] {
  return y.map((yi, i) => normCdf((yi - mu[i]) / sigma[i]));
}

// -----------------------------------------------------------------------------
// §9 setup-cell promoted helpers: numerically stable mixture NLL.
// -----------------------------------------------------------------------------

/** log Σ_m exp(arr[m, i]) over axis 0, returned as length-n array. */
export function logsumexpAxis0(arr: number[][]): number[] {
  const M = arr.length;
  if (M === 0) return [];
  const n = arr[0].length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let mx = -Infinity;
    for (let m = 0; m < M; m++) if (arr[m][i] > mx) mx = arr[m][i];
    let s = 0;
    for (let m = 0; m < M; m++) s += Math.exp(arr[m][i] - mx);
    out[i] = mx + Math.log(s);
  }
  return out;
}

/** Mixture NLL: NLL[Y] under M⁻¹ Σ_m N(Y; mu_m, σ²_ale).
 *  Result is mean over the test set (matches notebook `mixture_nll`). */
export function mixtureNll(
  y: number[],
  muMembers: number[][],
  sigmaAle: number[],
): number {
  const M = muMembers.length;
  const n = y.length;
  const logTwoPi = Math.log(2 * Math.PI);
  const logPhi: number[][] = Array.from({ length: M }, () => new Array(n).fill(0));
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < n; i++) {
      const sd = sigmaAle[i];
      const z = (y[i] - muMembers[m][i]) / sd;
      logPhi[m][i] = -0.5 * logTwoPi - Math.log(sd) - 0.5 * z * z;
    }
  }
  const lse = logsumexpAxis0(logPhi);
  let nll = 0;
  for (let i = 0; i < n; i++) nll += lse[i] - Math.log(M);
  return -nll / n;
}

// -----------------------------------------------------------------------------
// §2 Bayesian polynomial regression: closed-form posterior + predictive.
// -----------------------------------------------------------------------------

export type PolyPosterior = { muN: number[]; SigmaN: number[][]; degree: number };
export type PolyPredictive = {
  fMean: number[];
  aleatoricVar: number[];
  epistemicVar: number[];
  totalVar: number[];
};

/** Closed-form Bayesian polynomial-regression posterior with known noise.
 *  Mirrors `bayes_poly_posterior` in the notebook. */
export function bayesPolyPosterior(
  X: number[],
  y: number[],
  degree: number,
  sigmaKnown: number | number[],
  tauPrior: number = DEFAULT_TAU_PRIOR,
): PolyPosterior {
  const n = X.length;
  const Phi = phi(X, degree);
  const d = degree + 1;
  const sigmaVec: number[] = typeof sigmaKnown === 'number'
    ? new Array(n).fill(sigmaKnown)
    : sigmaKnown.slice();

  // PhiTWPhi[k,l] = Σ_i Phi[i,k] / σ_i² · Phi[i,l]
  const PhiTWPhi: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    const wi = 1 / (sigmaVec[i] * sigmaVec[i]);
    for (let k = 0; k < d; k++) {
      const phi_ik = Phi[i][k];
      for (let l = 0; l <= k; l++) {
        PhiTWPhi[k][l] += wi * phi_ik * Phi[i][l];
      }
    }
  }
  for (let k = 0; k < d; k++) for (let l = 0; l < k; l++) PhiTWPhi[l][k] = PhiTWPhi[k][l];

  // PhiTWPhi + (1/τ²) I.
  const priorPrec = 1 / (tauPrior * tauPrior);
  for (let k = 0; k < d; k++) PhiTWPhi[k][k] += priorPrec;
  const SigmaN = invSPD(PhiTWPhi);

  // (PhiᵀW) y, then μ = Σ_N (PhiᵀW) y.
  const PhiTWy = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    const wiy = y[i] / (sigmaVec[i] * sigmaVec[i]);
    for (let k = 0; k < d; k++) PhiTWy[k] += Phi[i][k] * wiy;
  }
  const muN = matVec(SigmaN, PhiTWy);
  return { muN, SigmaN, degree };
}

/** Posterior-predictive mean, aleatoric, epistemic, total variance at X*.
 *  Mirrors `bayes_poly_predict` in the notebook. */
export function bayesPolyPredict(
  Xstar: number[],
  posterior: PolyPosterior,
  sigmaStar: number | number[],
): PolyPredictive {
  const { muN, SigmaN, degree } = posterior;
  const PhiStar = phi(Xstar, degree);
  const n = Xstar.length;
  const fMean = matVec(PhiStar, muN); // Per-row dot.
  // Actually matVec(PhiStar, muN) treats PhiStar as a matrix; row i gives f(x*_i). Good.
  const aleatoricVar: number[] = new Array(n).fill(0);
  const epistemicVar: number[] = new Array(n).fill(0);
  const totalVar: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const ale = typeof sigmaStar === 'number'
      ? sigmaStar * sigmaStar
      : sigmaStar[i] * sigmaStar[i];
    const epi = quadForm(PhiStar[i], SigmaN);
    aleatoricVar[i] = ale;
    epistemicVar[i] = epi;
    totalVar[i] = ale + epi;
  }
  return { fMean, aleatoricVar, epistemicVar, totalVar };
}

/** Laplace approximation for the linear-in-θ Gaussian model — exact, same as
 *  `bayesPolyPosterior`. Wrapper for §7.2 prose alignment. */
export function laplacePolynomial(
  X: number[],
  y: number[],
  degree: number,
  sigmaKnown: number | number[],
  tauPrior: number = DEFAULT_TAU_PRIOR,
): PolyPosterior {
  return bayesPolyPosterior(X, y, degree, sigmaKnown, tauPrior);
}

// -----------------------------------------------------------------------------
// §4 Scoring rules: Brier, log-loss, CRPS (Gaussian closed-form + quadrature).
// -----------------------------------------------------------------------------

export function brierScore(yTrue: number[], yProba: number[]): number {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const d = yProba[i] - yTrue[i];
    s += d * d;
  }
  return s / yTrue.length;
}

export function logLoss(yTrue: number[], yProba: number[], eps: number = 1e-12): number {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const p = Math.min(Math.max(yProba[i], eps), 1 - eps);
    s -= yTrue[i] * Math.log(p) + (1 - yTrue[i]) * Math.log(1 - p);
  }
  return s / yTrue.length;
}

/** Hersbach (2000) closed-form CRPS for Gaussian forecast.
 *  CRPS(N(μ, σ²), y) = σ · (z(2Φ(z) - 1) + 2φ(z) - 1/√π), z = (y - μ)/σ. */
export function crpsGaussian(y: number, mu: number, sigma: number): number {
  const z = (y - mu) / sigma;
  return sigma * (z * (2 * normCdf(z) - 1) + 2 * normPdf(z) - 1 / Math.sqrt(Math.PI));
}

/** Trapezoidal CRPS for a Gaussian forecast — verification of `crpsGaussian`. */
export function crpsQuadrature(
  y: number,
  mu: number,
  sigma: number,
  nGrid: number = 2000,
  gridSpan: number = 10.0,
): number {
  const lo = mu - gridSpan * sigma;
  const hi = mu + gridSpan * sigma;
  const dx = (hi - lo) / (nGrid - 1);
  let s = 0;
  for (let i = 0; i < nGrid; i++) {
    const t = lo + i * dx;
    const F = normCdf((t - mu) / sigma);
    const ind = y <= t ? 1 : 0;
    const v = (F - ind) * (F - ind);
    s += i === 0 || i === nGrid - 1 ? 0.5 * v : v;
  }
  return s * dx;
}

// -----------------------------------------------------------------------------
// §4.3 Murphy decomposition: BS = REL - RES + UNC.
// Mirrors `murphy_decomposition` in the notebook (equal-frequency binning,
// per-bin constant forecast that recomputes BS for the decomposition identity).
// -----------------------------------------------------------------------------

export type MurphyDecomp = { bs: number; rel: number; res: number; unc: number };

export function murphyDecomposition(
  yTrue: number[],
  yProba: number[],
  nBins: number = 10,
): MurphyDecomp {
  const n = yProba.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => yProba[a] - yProba[b]);
  const binSize = Math.floor(n / nBins);

  const yBar = yTrue.reduce((a, c) => a + c, 0) / n;
  const unc = yBar * (1 - yBar);

  let rel = 0;
  let res = 0;
  // Per-bin constant forecast (mean predicted probability within the bin),
  // re-aggregated into a y-aligned vector to recompute BS for the identity check.
  const pConstPerBin: number[] = new Array(n).fill(0);
  for (let k = 0; k < nBins; k++) {
    const lo = k * binSize;
    const hi = k === nBins - 1 ? n : (k + 1) * binSize;
    const len = hi - lo;
    if (len === 0) continue;
    let sp = 0;
    let sy = 0;
    for (let j = lo; j < hi; j++) {
      sp += yProba[idx[j]];
      sy += yTrue[idx[j]];
    }
    const pk = sp / len;
    const ybark = sy / len;
    const w = len / n;
    rel += w * (pk - ybark) * (pk - ybark);
    res += w * (ybark - yBar) * (ybark - yBar);
    for (let j = lo; j < hi; j++) pConstPerBin[idx[j]] = pk;
  }
  let bs = 0;
  for (let i = 0; i < n; i++) {
    const d = pConstPerBin[i] - yTrue[i];
    bs += d * d;
  }
  bs /= n;
  return { bs, rel, res, unc };
}

// -----------------------------------------------------------------------------
// §3.4 Isotonic regression via the pool-adjacent-violators algorithm.
// Used by the §3 viz to recalibrate the §1 classifier's probabilities.
// -----------------------------------------------------------------------------

export type IsotonicFit = { xKnots: number[]; yKnots: number[] };

/** PAV isotonic regression on sorted (x, y) pairs.
 *  Returns the breakpoint sequence (sorted x with piecewise-constant y). */
export function fitIsotonic(xRaw: number[], yRaw: number[]): IsotonicFit {
  const n = xRaw.length;
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => xRaw[a] - xRaw[b]);
  const xs = order.map((i) => xRaw[i]);
  const ys = order.map((i) => yRaw[i]);
  // PAV.
  const vals: number[] = [];
  const wts: number[] = [];
  const lefts: number[] = [];
  const rights: number[] = [];
  for (let i = 0; i < n; i++) {
    let v = ys[i];
    let w = 1;
    let l = i;
    let r = i;
    while (vals.length > 0 && vals[vals.length - 1] >= v) {
      const vp = vals.pop()!;
      const wp = wts.pop()!;
      const lp = lefts.pop()!;
      rights.pop();
      v = (vp * wp + v * w) / (wp + w);
      w = wp + w;
      l = lp;
    }
    vals.push(v);
    wts.push(w);
    lefts.push(l);
    rights.push(r);
  }
  const xKnots: number[] = [];
  const yKnots: number[] = [];
  for (let b = 0; b < vals.length; b++) {
    xKnots.push(xs[lefts[b]]);
    yKnots.push(vals[b]);
    if (rights[b] !== lefts[b]) {
      xKnots.push(xs[rights[b]]);
      yKnots.push(vals[b]);
    }
  }
  return { xKnots, yKnots };
}

/** Evaluate a PAV fit at query points with linear interpolation between knots
 *  (out-of-bounds clipped — matches sklearn IsotonicRegression(out_of_bounds="clip")). */
export function predictIsotonic(fit: IsotonicFit, xQuery: number[]): number[] {
  const { xKnots, yKnots } = fit;
  const m = xKnots.length;
  return xQuery.map((x) => {
    if (x <= xKnots[0]) return yKnots[0];
    if (x >= xKnots[m - 1]) return yKnots[m - 1];
    let lo = 0;
    let hi = m - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xKnots[mid] <= x) lo = mid;
      else hi = mid;
    }
    const t = (x - xKnots[lo]) / (xKnots[hi] - xKnots[lo]);
    return yKnots[lo] + t * (yKnots[hi] - yKnots[lo]);
  });
}

/** End-to-end isotonic recalibration: fit on (probaCal, yCal), predict on probaTest. */
export function isotonicRecal(
  probaCal: number[],
  yCal: number[],
  probaTest: number[],
): number[] {
  const fit = fitIsotonic(probaCal, yCal);
  return predictIsotonic(fit, probaTest);
}

// -----------------------------------------------------------------------------
// §5 + §10: OLS polynomial fit via normal equations.
// Defined ONCE here; reused by §6 bootstrap and §10 ridge sweep.
// -----------------------------------------------------------------------------

/** Solve Φβ ≈ y via the normal equations with small ridge for stability.
 *  Mirrors `fit_ols_poly` in the notebook (np.linalg.lstsq). For DEGREE=7 and
 *  n=200 the design matrix is well-conditioned after the POLY_SCALE rescaling,
 *  so closed-form solve is enough; we add a tiny jitter to fall back gracefully
 *  on near-singular cases. */
export function fitOlsPoly(X: number[], y: number[], degree: number): number[] {
  const Phi = phi(X, degree);
  const d = degree + 1;
  const n = X.length;
  const PhiTPhi: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const PhiTy: number[] = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    const yi = y[i];
    for (let k = 0; k < d; k++) {
      const pik = Phi[i][k];
      PhiTy[k] += pik * yi;
      for (let l = 0; l <= k; l++) PhiTPhi[k][l] += pik * Phi[i][l];
    }
  }
  for (let k = 0; k < d; k++) for (let l = 0; l < k; l++) PhiTPhi[l][k] = PhiTPhi[k][l];
  return solveSPD(PhiTPhi, PhiTy, 1e-10);
}

/** Predict ŷ = Φ(Xeval) · β. */
export function predictOlsPoly(beta: number[], Xeval: number[], degree: number): number[] {
  const Phi = phi(Xeval, degree);
  return matVec(Phi, beta);
}

// -----------------------------------------------------------------------------
// §6 Bootstrap variants on a polynomial OLS learner.
// -----------------------------------------------------------------------------

/** Pairs bootstrap: resample (x_i, y_i) jointly. */
export function pairsBootstrapFit(
  X: number[],
  y: number[],
  B: number,
  degree: number,
  rng: () => number,
): number[][] {
  const n = X.length;
  const fits: number[][] = [];
  const Xs = new Array(n).fill(0);
  const ys = new Array(n).fill(0);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng() * n);
      Xs[i] = X[idx];
      ys[i] = y[idx];
    }
    fits.push(fitOlsPoly(Xs, ys, degree));
  }
  return fits;
}

/** Residual bootstrap: fit once, resample residuals from empirical pool.
 *  Assumes exchangeable residuals (wrong under heteroscedasticity — that's the
 *  pedagogical point of §6.2). */
export function residualBootstrapFit(
  X: number[],
  y: number[],
  B: number,
  degree: number,
  rng: () => number,
): number[][] {
  const beta0 = fitOlsPoly(X, y, degree);
  const fitted = predictOlsPoly(beta0, X, degree);
  const n = X.length;
  const resid = y.map((yi, i) => yi - fitted[i]);
  const fits: number[][] = [];
  const ys = new Array(n).fill(0);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng() * n);
      ys[i] = fitted[i] + resid[idx];
    }
    fits.push(fitOlsPoly(X, ys, degree));
  }
  return fits;
}

/** Wild bootstrap: y* = ŷ + w_i ε̂_i with Rademacher w_i ∈ {-1, +1}.
 *  Preserves local noise magnitude — right for heteroscedastic regression. */
export function wildBootstrapFit(
  X: number[],
  y: number[],
  B: number,
  degree: number,
  rng: () => number,
): number[][] {
  const beta0 = fitOlsPoly(X, y, degree);
  const fitted = predictOlsPoly(beta0, X, degree);
  const n = X.length;
  const resid = y.map((yi, i) => yi - fitted[i]);
  const fits: number[][] = [];
  const ys = new Array(n).fill(0);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < n; i++) {
      const w = rng() < 0.5 ? -1 : 1;
      ys[i] = fitted[i] + w * resid[i];
    }
    fits.push(fitOlsPoly(X, ys, degree));
  }
  return fits;
}

export type BaggedPredict = {
  mean: number[];
  variance: number[];
  /** All B per-point predictions, shape (B, |Xeval|). */
  members: number[][];
};

/** Predict on Xeval for each bootstrap fit, return mean / variance / per-rep. */
export function baggedPredictGrid(
  fits: number[][],
  Xeval: number[],
  degree: number,
): BaggedPredict {
  const B = fits.length;
  const n = Xeval.length;
  const members: number[][] = [];
  for (let b = 0; b < B; b++) members.push(predictOlsPoly(fits[b], Xeval, degree));
  const mean = new Array(n).fill(0);
  for (let b = 0; b < B; b++) for (let i = 0; i < n; i++) mean[i] += members[b][i];
  for (let i = 0; i < n; i++) mean[i] /= B;
  const variance = new Array(n).fill(0);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < n; i++) {
      const d = members[b][i] - mean[i];
      variance[i] += d * d;
    }
  }
  for (let i = 0; i < n; i++) variance[i] /= Math.max(B - 1, 1);
  return { mean, variance, members };
}

// -----------------------------------------------------------------------------
// §6.3 Aleatoric estimate from residuals: bin squared residuals by x, smooth
// with a 3-point moving average, return an interpolator. Mirrors notebook
// `aleatoric_from_residuals`.
// -----------------------------------------------------------------------------

export type AleatoricInterp = {
  /** Evaluate σ²(x) at arbitrary x via linear interpolation between bin centers. */
  sigma2At: (x: number | number[]) => number | number[];
  binCenters: number[];
  binVals: number[];
};

export function aleatoricFromResiduals(
  Xtrain: number[],
  yTrain: number[],
  fMeanTrain: number[],
  nBins: number = 15,
): AleatoricInterp {
  const residSq = yTrain.map((yi, i) => (yi - fMeanTrain[i]) * (yi - fMeanTrain[i]));
  let xMin = Infinity;
  let xMax = -Infinity;
  for (const x of Xtrain) {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
  }
  const edges = new Array(nBins + 1).fill(0).map((_, k) => xMin + ((xMax - xMin) * k) / nBins);
  const centers: number[] = [];
  const vals: number[] = [];
  for (let k = 0; k < nBins; k++) {
    let s = 0;
    let c = 0;
    for (let i = 0; i < Xtrain.length; i++) {
      if (Xtrain[i] >= edges[k] && Xtrain[i] <= edges[k + 1]) {
        s += residSq[i];
        c++;
      }
    }
    if (c > 0) {
      centers.push(0.5 * (edges[k] + edges[k + 1]));
      vals.push(s / c);
    }
  }
  // 3-point moving-average smoothing (matches notebook `np.convolve(..., mode='same')`).
  if (vals.length >= 3) {
    const smoothed = vals.slice();
    for (let i = 0; i < vals.length; i++) {
      const lo = Math.max(0, i - 1);
      const hi = Math.min(vals.length - 1, i + 1);
      let s = 0;
      for (let j = lo; j <= hi; j++) s += vals[j];
      // np.convolve with mode='same' for a 3-tap mean filter uses an effective
      // 1/3 weight at every position (including boundaries, where one tap reads
      // a virtual 0 from outside the array). Mirror the same behaviour rather
      // than re-normalizing.
      smoothed[i] = s / 3;
    }
    for (let i = 0; i < vals.length; i++) vals[i] = smoothed[i];
  }
  const sigma2At = (x: number | number[]): number | number[] => {
    const interp = (xi: number) => {
      if (centers.length === 0) return 0;
      if (xi <= centers[0]) return vals[0];
      if (xi >= centers[centers.length - 1]) return vals[centers.length - 1];
      let lo = 0;
      let hi = centers.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (centers[mid] <= xi) lo = mid;
        else hi = mid;
      }
      const t = (xi - centers[lo]) / (centers[hi] - centers[lo]);
      return vals[lo] + t * (vals[hi] - vals[lo]);
    };
    return Array.isArray(x) ? x.map(interp) : interp(x);
  };
  return { sigma2At, binCenters: centers, binVals: vals };
}

// -----------------------------------------------------------------------------
// §7 MLP forward pass (consumed JSON coefficients) + MC-dropout predict.
// -----------------------------------------------------------------------------

/** Serialized MLPRegressor for a single hidden-layer architecture (n_in=1).
 *  Each `coefs[ell]` is shape (h_ell, h_{ell+1}); `intercepts[ell]` shape (h_{ell+1},). */
export type MlpCoefs = {
  coefs: number[][][];
  intercepts: number[][];
  activation: 'tanh' | 'relu';
};

function activate(z: number, kind: 'tanh' | 'relu'): number {
  if (kind === 'tanh') return Math.tanh(z);
  return z > 0 ? z : 0;
}

/** NumPy-side forward pass through an MLP using public coefs/intercepts.
 *  Input X has shape (n,) — single-feature 1D regression matching notebook §7.3. */
export function mlpForwardNumpy(
  X: number[],
  mlp: MlpCoefs,
  dropoutRate: number = 0,
  rng: (() => number) | null = null,
): number[] {
  const { coefs, intercepts, activation: act } = mlp;
  const L = coefs.length;
  const n = X.length;
  // Initial activation: X as (n, 1).
  let h: number[][] = X.map((xi) => [xi]);
  for (let ell = 0; ell < L - 1; ell++) {
    const W = coefs[ell];
    const b = intercepts[ell];
    const inDim = W.length;
    const outDim = W[0].length;
    const next: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(outDim).fill(0);
      for (let j = 0; j < outDim; j++) {
        let s = b[j];
        for (let k = 0; k < inDim; k++) s += h[i][k] * W[k][j];
        row[j] = activate(s, act);
      }
      if (dropoutRate > 0 && rng !== null) {
        const keepP = 1 - dropoutRate;
        for (let j = 0; j < outDim; j++) {
          const mask = rng() < keepP ? 1 : 0;
          row[j] = (row[j] * mask) / keepP;
        }
      }
      next.push(row);
    }
    h = next;
  }
  // Output layer (linear).
  const Wout = coefs[L - 1];
  const bout = intercepts[L - 1];
  const inDim = Wout.length;
  const outDim = Wout[0].length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = bout[0];
    for (let k = 0; k < inDim; k++) s += h[i][k] * Wout[k][0];
    out[i] = outDim === 1 ? s : NaN; // Single-output regression only.
  }
  return out;
}

export type StochasticPrediction = {
  mean: number[];
  variance: number[];
  members: number[][];
};

/** Run T stochastic dropout forwards; return mean / variance / per-replicate. */
export function mcDropoutPredict(
  Xeval: number[],
  mlp: MlpCoefs,
  dropoutRate: number,
  T: number,
  rng: () => number,
): StochasticPrediction {
  const members: number[][] = [];
  for (let t = 0; t < T; t++) members.push(mlpForwardNumpy(Xeval, mlp, dropoutRate, rng));
  return aggregateMembers(members);
}

/** Deep-ensemble predict from per-member coefs. */
export function ensemblePredict(
  Xeval: number[],
  members: MlpCoefs[],
): StochasticPrediction {
  const preds: number[][] = members.map((m) => mlpForwardNumpy(Xeval, m, 0, null));
  return aggregateMembers(preds);
}

function aggregateMembers(members: number[][]): StochasticPrediction {
  const M = members.length;
  const n = members[0].length;
  const mean = new Array(n).fill(0);
  for (let m = 0; m < M; m++) for (let i = 0; i < n; i++) mean[i] += members[m][i];
  for (let i = 0; i < n; i++) mean[i] /= M;
  const variance = new Array(n).fill(0);
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < n; i++) {
      const d = members[m][i] - mean[i];
      variance[i] += d * d;
    }
  }
  for (let i = 0; i < n; i++) variance[i] /= Math.max(M - 1, 1);
  return { mean, variance, members };
}

// -----------------------------------------------------------------------------
// §8 Split-conformal: constant and locally-weighted nonconformity scores.
// -----------------------------------------------------------------------------

export type ConformalFit = {
  qHat: number;
  /** PI lower bound at Xeval. */
  lo: number[];
  /** PI upper bound at Xeval. */
  hi: number[];
};

/** Sort an array ascending (returns a new array). */
function sortAsc(a: number[]): number[] {
  return a.slice().sort((x, y) => x - y);
}

/** Inflated-quantile index: k = ⌈(1-α)(n+1)⌉ clipped to n. Returns sorted[k-1]. */
function inflatedQuantile(scores: number[], alpha: number): number {
  const sorted = sortAsc(scores);
  const n = sorted.length;
  let k = Math.ceil((1 - alpha) * (n + 1));
  if (k > n) k = n;
  if (k < 1) k = 1;
  return sorted[k - 1];
}

/** Constant-width split conformal: PI = μ̂ ± q̂. */
export function splitConformalConstant(
  Xcal: number[],
  yCal: number[],
  Xeval: number[],
  muFn: (X: number[]) => number[],
  alpha: number = DEFAULT_ALPHA,
): ConformalFit {
  const muCal = muFn(Xcal);
  const calScores = yCal.map((yi, i) => Math.abs(yi - muCal[i]));
  const qHat = inflatedQuantile(calScores, alpha);
  const muEval = muFn(Xeval);
  const lo = muEval.map((m) => m - qHat);
  const hi = muEval.map((m) => m + qHat);
  return { qHat, lo, hi };
}

/** Locally-weighted split conformal: PI = μ̂ ± q̂·σ̂ with normalized scores. */
export function splitConformalLocallyWeighted(
  Xcal: number[],
  yCal: number[],
  Xeval: number[],
  muFn: (X: number[]) => number[],
  sigmaFn: (X: number[]) => number[],
  alpha: number = DEFAULT_ALPHA,
  sigmaFloor: number = 1e-3,
): ConformalFit {
  const muCal = muFn(Xcal);
  const sigCalRaw = sigmaFn(Xcal);
  const sigCal = sigCalRaw.map((s) => Math.max(s, sigmaFloor));
  const calScores = yCal.map((yi, i) => Math.abs(yi - muCal[i]) / sigCal[i]);
  const qHat = inflatedQuantile(calScores, alpha);
  const muEval = muFn(Xeval);
  const sigEval = sigmaFn(Xeval).map((s) => Math.max(s, sigmaFloor));
  const lo = muEval.map((m, i) => m - qHat * sigEval[i]);
  const hi = muEval.map((m, i) => m + qHat * sigEval[i]);
  return { qHat, lo, hi };
}

/** Empirical marginal coverage of a PI on a held-out test set. */
export function coverage(lo: number[], hi: number[], y: number[]): number {
  let c = 0;
  for (let i = 0; i < y.length; i++) {
    if (y[i] >= lo[i] && y[i] <= hi[i]) c++;
  }
  return c / y.length;
}

/** Empirical mean PI width. */
export function meanWidth(lo: number[], hi: number[]): number {
  let s = 0;
  for (let i = 0; i < lo.length; i++) s += hi[i] - lo[i];
  return s / lo.length;
}

// -----------------------------------------------------------------------------
// §9 Mixture-of-Gaussians readout for the deep-ensemble predictive density.
// -----------------------------------------------------------------------------

/** Equally-weighted Gaussian mixture density at evaluation points y. */
export function ensembleMixtureDensity(
  yGrid: number[],
  muMembers: number[],
  sigmaAle: number,
): number[] {
  const M = muMembers.length;
  return yGrid.map((y) => {
    let s = 0;
    for (let m = 0; m < M; m++) {
      const z = (y - muMembers[m]) / sigmaAle;
      s += normPdf(z) / sigmaAle;
    }
    return s / M;
  });
}

/** Mixture mean and variance via numerical quadrature on yGrid. */
export function ensembleMixtureMeanVar(
  yGrid: number[],
  density: number[],
): { mean: number; variance: number } {
  const n = yGrid.length;
  let mean = 0;
  for (let i = 0; i < n - 1; i++) {
    const dx = yGrid[i + 1] - yGrid[i];
    mean += 0.5 * (yGrid[i] * density[i] + yGrid[i + 1] * density[i + 1]) * dx;
  }
  let variance = 0;
  for (let i = 0; i < n - 1; i++) {
    const dx = yGrid[i + 1] - yGrid[i];
    const v1 = (yGrid[i] - mean) * (yGrid[i] - mean) * density[i];
    const v2 = (yGrid[i + 1] - mean) * (yGrid[i + 1] - mean) * density[i + 1];
    variance += 0.5 * (v1 + v2) * dx;
  }
  return { mean, variance };
}

// -----------------------------------------------------------------------------
// §10 Over-parameterized regime: ridge fit, PP variance, UQ-shadow sweep.
// -----------------------------------------------------------------------------

/** Ridge fit via normal equations with explicit λI regularizer.
 *  Returns β̂ = (XᵀX + λI)⁻¹ Xᵀy, and the inverse (for PP variance). */
export type RidgeOpFit = { beta: number[]; AInv: number[][] };

export function ridgeFitWithCov(X: number[][], y: number[], lambda: number): RidgeOpFit {
  const n = X.length;
  const p = X[0].length;
  const A: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const Xty: number[] = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    const yi = y[i];
    for (let j = 0; j < p; j++) {
      Xty[j] += xi[j] * yi;
      for (let k = 0; k <= j; k++) A[j][k] += xi[j] * xi[k];
    }
  }
  for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) A[k][j] = A[j][k];
  for (let j = 0; j < p; j++) A[j][j] += lambda;
  const AInv = invSPD(A);
  const beta = matVec(AInv, Xty);
  return { beta, AInv };
}

/** Per-row PP variance σ_noise² + xᵢᵀ Σ_β xᵢ for the linear-Gaussian model
 *  with Σ_β = σ_noise² (XᵀX + λI)⁻¹. */
export function ridgePpVariance(
  Xtest: number[][],
  AInv: number[][],
  sigmaNoise: number,
): number[] {
  const sigma2 = sigmaNoise * sigmaNoise;
  return Xtest.map((xi) => sigma2 + sigma2 * quadForm(xi, AInv));
}

export type UqSweepPoint = {
  p: number;
  pOverN: number;
  testMseMean: number;
  testMseSe: number;
  postVarMean: number;
  postVarSe: number;
  qHatMean: number;
  qHatSe: number;
  covMean: number;
  covSe: number;
};

/** §10 UQ-shadow sweep: isotropic Gaussian features, well-specified linear truth.
 *  Mirrors `double_descent_uq_sweep` in the notebook. */
export function doubleDescentUqSweep(
  pGrid: number[],
  nTr: number,
  nCal: number,
  nTest: number,
  snr: number,
  sigmaNoise: number,
  lambdaRidge: number,
  bReps: number,
  alpha: number,
  rng: () => number,
): UqSweepPoint[] {
  const gauss = gaussianFrom(rng);
  const results: UqSweepPoint[] = [];
  for (const p of pGrid) {
    const mseRep: number[] = [];
    const pvRep: number[] = [];
    const qRep: number[] = [];
    const covRep: number[] = [];
    for (let b = 0; b < bReps; b++) {
      // β* uniform on sphere of radius √SNR.
      const betaStar = new Array(p).fill(0).map(() => gauss());
      let nrm = 0;
      for (const v of betaStar) nrm += v * v;
      nrm = Math.sqrt(nrm);
      const scl = Math.sqrt(snr) / nrm;
      for (let j = 0; j < p; j++) betaStar[j] *= scl;
      // Generate design matrices.
      const fillX = (nRows: number): number[][] => {
        const M: number[][] = [];
        for (let i = 0; i < nRows; i++) {
          const row = new Array(p).fill(0);
          for (let j = 0; j < p; j++) row[j] = gauss();
          M.push(row);
        }
        return M;
      };
      const Xtr = fillX(nTr);
      const Xcal = fillX(nCal);
      const Xtest = fillX(nTest);
      const linear = (M: number[][]) => M.map((row) => {
        let s = 0;
        for (let j = 0; j < p; j++) s += row[j] * betaStar[j];
        return s;
      });
      const yTrClean = linear(Xtr);
      const yCalClean = linear(Xcal);
      const yTestClean = linear(Xtest);
      const yTr = yTrClean.map((v) => v + sigmaNoise * gauss());
      const yCal = yCalClean.map((v) => v + sigmaNoise * gauss());
      const yTest = yTestClean.map((v) => v + sigmaNoise * gauss());
      // Ridge fit and PP variance.
      const { beta: betaHat, AInv } = ridgeFitWithCov(Xtr, yTr, lambdaRidge);
      const yPredTest = Xtest.map((row) => {
        let s = 0;
        for (let j = 0; j < p; j++) s += row[j] * betaHat[j];
        return s;
      });
      let mse = 0;
      for (let i = 0; i < nTest; i++) {
        const d = yTest[i] - yPredTest[i];
        mse += d * d;
      }
      mse /= nTest;
      const ppVar = ridgePpVariance(Xtest, AInv, sigmaNoise);
      let pvMean = 0;
      for (const v of ppVar) pvMean += v;
      pvMean /= nTest;
      // Conformal half-width on the calibration fold.
      const yPredCal = Xcal.map((row) => {
        let s = 0;
        for (let j = 0; j < p; j++) s += row[j] * betaHat[j];
        return s;
      });
      const calScores = yCal.map((yi, i) => Math.abs(yi - yPredCal[i]));
      const qHat = inflatedQuantile(calScores, alpha);
      let cov = 0;
      for (let i = 0; i < nTest; i++) {
        if (Math.abs(yTest[i] - yPredTest[i]) <= qHat) cov++;
      }
      cov /= nTest;
      mseRep.push(mse);
      pvRep.push(pvMean);
      qRep.push(qHat);
      covRep.push(cov);
    }
    const meanArr = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    const seArr = (a: number[]) => {
      const m = meanArr(a);
      let s = 0;
      for (const v of a) s += (v - m) * (v - m);
      return Math.sqrt(s / Math.max(a.length - 1, 1)) / Math.sqrt(a.length);
    };
    results.push({
      p,
      pOverN: p / nTr,
      testMseMean: meanArr(mseRep),
      testMseSe: seArr(mseRep),
      postVarMean: meanArr(pvRep),
      postVarSe: seArr(pvRep),
      qHatMean: meanArr(qRep),
      qHatSe: seArr(qRep),
      covMean: meanArr(covRep),
      covSe: seArr(covRep),
    });
  }
  return results;
}

// -----------------------------------------------------------------------------
// §11 OOD detection: ROC curve + AUROC via the trapezoidal rule on a sorted scan.
// -----------------------------------------------------------------------------

export type RocCurve = { fpr: number[]; tpr: number[] };

/** Build a ROC curve from continuous scores and binary labels (1 = positive). */
export function roc(scores: number[], labels: number[]): RocCurve {
  const n = scores.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => scores[b] - scores[a]); // Descending.
  const P = labels.filter((y) => y === 1).length;
  const N = n - P;
  const fpr: number[] = [0];
  const tpr: number[] = [0];
  let tp = 0;
  let fp = 0;
  let prevScore = Infinity;
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    if (scores[i] !== prevScore) {
      fpr.push(N > 0 ? fp / N : 0);
      tpr.push(P > 0 ? tp / P : 0);
      prevScore = scores[i];
    }
    if (labels[i] === 1) tp++;
    else fp++;
  }
  fpr.push(N > 0 ? fp / N : 0);
  tpr.push(P > 0 ? tp / P : 0);
  return { fpr, tpr };
}

/** Area under ROC via trapezoidal integration over the FPR axis. */
export function aurocFromScores(scores: number[], labels: number[]): number {
  const { fpr, tpr } = roc(scores, labels);
  let a = 0;
  for (let i = 1; i < fpr.length; i++) {
    a += 0.5 * (tpr[i] + tpr[i - 1]) * (fpr[i] - fpr[i - 1]);
  }
  return a;
}

// -----------------------------------------------------------------------------
// §13 Temperature scaling: 1-D NLL minimization on a calibration set.
// -----------------------------------------------------------------------------

function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** Calibration NLL as a function of the inverse temperature T > 0. */
export function temperatureNll(T: number, logits: number[], y: number[]): number {
  const eps = 1e-12;
  let s = 0;
  for (let i = 0; i < logits.length; i++) {
    const p = Math.min(Math.max(sigmoid(logits[i] / T), eps), 1 - eps);
    s -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
  }
  return s / logits.length;
}

/** Fit T > 0 minimizing temperature-NLL via golden-section search on [0.1, 10].
 *  Mirrors `fit_temperature` in the notebook (scipy.optimize.minimize_scalar bounded). */
export function fitTemperature(
  logitsCal: number[],
  yCal: number[],
  lo: number = 0.1,
  hi: number = 10.0,
  tol: number = 1e-6,
): number {
  const phi = (Math.sqrt(5) - 1) / 2; // ≈ 0.618.
  let a = lo;
  let b = hi;
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);
  let fc = temperatureNll(c, logitsCal, yCal);
  let fd = temperatureNll(d, logitsCal, yCal);
  while (Math.abs(b - a) > tol) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - phi * (b - a);
      fc = temperatureNll(c, logitsCal, yCal);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + phi * (b - a);
      fd = temperatureNll(d, logitsCal, yCal);
    }
  }
  return 0.5 * (a + b);
}

/** Apply temperature scaling: p = σ(z / T). */
export function temperatureScaleProb(logits: number[], T: number): number[] {
  return logits.map((z) => sigmoid(z / T));
}

/** Bootstrap ECE confidence interval at level (1-α). Returns (mean, lo, hi). */
export function bootstrapEceCi(
  y: number[],
  p: number[],
  nBins: number = 10,
  B: number = 500,
  alpha: number = 0.05,
  rng: () => number = mulberry32(NOTEBOOK_SEED),
): { mean: number; lo: number; hi: number } {
  const n = y.length;
  const samples: number[] = [];
  const ys = new Array(n).fill(0);
  const ps = new Array(n).fill(0);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng() * n);
      ys[i] = y[idx];
      ps[i] = p[idx];
    }
    samples.push(ece(ys, ps, nBins));
  }
  samples.sort((a, b) => a - b);
  const mean = samples.reduce((s, v) => s + v, 0) / B;
  const lo = samples[Math.floor((alpha / 2) * B)];
  const hi = samples[Math.floor((1 - alpha / 2) * B) - 1];
  return { mean, lo, hi };
}

// -----------------------------------------------------------------------------
// Utility: linspace, percentile, and grid generators used by viz components.
// -----------------------------------------------------------------------------

export function linspace(lo: number, hi: number, n: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [lo];
  const step = (hi - lo) / (n - 1);
  return Array.from({ length: n }, (_, i) => lo + i * step);
}

export function quantile(arr: number[], q: number): number {
  const sorted = sortAsc(arr);
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const t = q * (n - 1);
  const lo = Math.floor(t);
  const hi = Math.min(lo + 1, n - 1);
  const frac = t - lo;
  return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

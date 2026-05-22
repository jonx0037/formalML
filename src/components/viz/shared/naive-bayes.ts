// =============================================================================
// naive-bayes.ts
//
// Shared math module for the T2 Supervised Learning track, naive-bayes topic.
// First consumers: viz components in src/content/topics/naive-bayes.mdx
//   (PosteriorOddsExplorer / IndependenceDAGViewer / BayesRiskDecomposition /
//    GaussianNBBoundaryComparison / MultinomialNBTextExplorer /
//    LaplaceSmoothingPosterior / DvGCrossoverPlot /
//    CalibrationReliabilityDiagram / ChowLiuTANStructureLearner /
//    NBVsLogisticOnCorrelatedFeatures).
//
// All numerical algorithms are direct translations of the verified Python
// notebook at notebooks/naive-bayes/01_naive_bayes.ipynb (the source of truth).
// Numerical agreement against notebook printed outputs is verified by
// src/components/viz/shared/__tests__/verify-naive-bayes.ts.
//
// Cross-module reuse (do NOT redefine):
//   - logSumExp from ./sequential-monte-carlo
//   - normCdf / normPdf / isotonicRecal / reliabilityBins from ./uncertainty-quantification
//   - logisticFit / logisticFitPredict from ./causal-inference-methods (used by
//     plattScalingFit and NBVsLogisticOnCorrelatedFeatures)
//   - mulberry32 / gaussianSampler from ./nonparametric-ml
// =============================================================================

import { logSumExp } from './sequential-monte-carlo';
import { normCdf, normPdf } from './uncertainty-quantification';
import { logisticFit, logisticPredictProba } from './causal-inference-methods';
import { mulberry32, gaussianSampler } from './nonparametric-ml';

export { logSumExp, normCdf, normPdf, mulberry32, gaussianSampler };

// -----------------------------------------------------------------------------
// Palette — formalML default (no posterior/prior dichotomy required for NB).
// -----------------------------------------------------------------------------

export const paletteNB = {
  pos: '#1f77b4',
  neg: '#d62728',
  highlight: '#ff7f0e',
  ok: '#2ca02c',
  data: '#7f7f7f',
  accent: '#9467bd',
} as const;

// -----------------------------------------------------------------------------
// Numerical helpers
// -----------------------------------------------------------------------------

/** Lanczos approximation of log Γ(x). Accurate to ~1e-15 for x > 0.5. */
export function lgamma(x: number): number {
  if (x < 0.5) {
    // Reflection formula: log Γ(x) = log(π / sin(πx)) - log Γ(1 - x).
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  const g = [
    0.999999999999809932769,
    676.520368121885098567,
    -1259.13921672240287047,
    771.323428777653078848,
    -176.615029162140599066,
    12.5073432786869048045,
    -0.138571095265720116895,
    9.98436957801957085956e-6,
    1.50563273514931155834e-7,
  ];
  const y = x - 1;
  let a = g[0];
  const t = y + 7.5;
  for (let i = 1; i < 9; i++) a += g[i] / (y + i);
  return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a);
}

/** log of the Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b). */
export function lbeta(a: number, b: number): number {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

// =============================================================================
// Gaussian Naive Bayes (continuous features, diagonal class-covariance)
// =============================================================================

export interface GaussianNBModel {
  /** Sorted unique class labels. */
  classes: number[];
  /** Class priors π_k = n_k / n. */
  classPriors: Float64Array;
  /** Per-class means, shape (K, d) row-major. */
  means: Float64Array;
  /** Per-class variances, shape (K, d) row-major. */
  variances: Float64Array;
  /** d (feature count). */
  d: number;
}

/** Fit Gaussian NB MLE with sklearn-style var_smoothing safeguard. */
export function gaussianNBFit(
  X: Float64Array,
  y: Int32Array,
  d: number,
  varSmoothing: number = 1e-9,
): GaussianNBModel {
  const n = y.length;
  // Discover classes by sorting unique label values.
  const seen = new Set<number>();
  for (let i = 0; i < n; i++) seen.add(y[i]);
  const classes = Array.from(seen).sort((a, b) => a - b);
  const K = classes.length;
  const classPriors = new Float64Array(K);
  const means = new Float64Array(K * d);
  const variances = new Float64Array(K * d);

  // Global per-feature variance, for the var_smoothing safeguard.
  const globalMean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) globalMean[j] += X[i * d + j];
  for (let j = 0; j < d; j++) globalMean[j] /= n;
  let globalVarMax = 0;
  const globalVar = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      const dx = X[i * d + j] - globalMean[j];
      globalVar[j] += dx * dx;
    }
  }
  for (let j = 0; j < d; j++) {
    globalVar[j] /= n;
    if (globalVar[j] > globalVarMax) globalVarMax = globalVar[j];
  }
  const epsilon = varSmoothing * globalVarMax;

  for (let ki = 0; ki < K; ki++) {
    const ck = classes[ki];
    let nk = 0;
    for (let i = 0; i < n; i++) {
      if (y[i] !== ck) continue;
      nk++;
      for (let j = 0; j < d; j++) means[ki * d + j] += X[i * d + j];
    }
    if (nk === 0) continue;
    classPriors[ki] = nk / n;
    for (let j = 0; j < d; j++) means[ki * d + j] /= nk;
    for (let i = 0; i < n; i++) {
      if (y[i] !== ck) continue;
      for (let j = 0; j < d; j++) {
        const dx = X[i * d + j] - means[ki * d + j];
        variances[ki * d + j] += dx * dx;
      }
    }
    for (let j = 0; j < d; j++) variances[ki * d + j] = variances[ki * d + j] / nk + epsilon;
  }

  return { classes, classPriors, means, variances, d };
}

/** Unnormalized log P(y_k | x) = log π_k + Σ_j log N(x_j; μ_kj, σ²_kj). */
export function gaussianNBLogPredict(X: Float64Array, model: GaussianNBModel): Float64Array {
  const { classPriors, means, variances, d } = model;
  const K = model.classes.length;
  const n = X.length / d;
  const out = new Float64Array(n * K);
  const LOG_2PI = Math.log(2 * Math.PI);
  for (let i = 0; i < n; i++) {
    for (let ki = 0; ki < K; ki++) {
      let logLik = Math.log(classPriors[ki]);
      for (let j = 0; j < d; j++) {
        const variance = variances[ki * d + j];
        const dx = X[i * d + j] - means[ki * d + j];
        logLik += -0.5 * (LOG_2PI + Math.log(variance) + (dx * dx) / variance);
      }
      out[i * K + ki] = logLik;
    }
  }
  return out;
}

/** Class predictions (argmax over log posteriors). */
export function gaussianNBPredict(X: Float64Array, model: GaussianNBModel): Int32Array {
  const lp = gaussianNBLogPredict(X, model);
  const K = model.classes.length;
  const n = lp.length / K;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let best = -Infinity;
    let arg = 0;
    for (let ki = 0; ki < K; ki++) {
      const v = lp[i * K + ki];
      if (v > best) {
        best = v;
        arg = ki;
      }
    }
    out[i] = model.classes[arg];
  }
  return out;
}

/** Normalized class probabilities via log-sum-exp. */
export function gaussianNBPredictProba(X: Float64Array, model: GaussianNBModel): Float64Array {
  const lp = gaussianNBLogPredict(X, model);
  const K = model.classes.length;
  const n = lp.length / K;
  const out = new Float64Array(n * K);
  const row = new Array<number>(K);
  for (let i = 0; i < n; i++) {
    for (let ki = 0; ki < K; ki++) row[ki] = lp[i * K + ki];
    const lse = logSumExp(row);
    for (let ki = 0; ki < K; ki++) out[i * K + ki] = Math.exp(row[ki] - lse);
  }
  return out;
}

// =============================================================================
// Multinomial Naive Bayes (token-count features)
// =============================================================================

export interface MultinomialNBModel {
  classes: number[];
  /** log π_k, length K. */
  classLogPriors: Float64Array;
  /** log P(token_j | class_k), shape (K, d) row-major. */
  featureLogProbs: Float64Array;
  d: number;
  alpha: number;
}

/** Fit Multinomial NB with Laplace/Dirichlet add-α smoothing.
 *  X is shape (n, d) of nonneg token counts. */
export function multinomialNBFit(
  X: Float64Array,
  y: Int32Array,
  d: number,
  alpha: number = 1.0,
): MultinomialNBModel {
  const n = y.length;
  const seen = new Set<number>();
  for (let i = 0; i < n; i++) seen.add(y[i]);
  const classes = Array.from(seen).sort((a, b) => a - b);
  const K = classes.length;
  const classLogPriors = new Float64Array(K);
  const featureLogProbs = new Float64Array(K * d);

  // class -> aggregated token counts (length d), then smoothed and log-normalized.
  for (let ki = 0; ki < K; ki++) {
    const ck = classes[ki];
    let nk = 0;
    for (let i = 0; i < n; i++) {
      if (y[i] !== ck) continue;
      nk++;
      for (let j = 0; j < d; j++) featureLogProbs[ki * d + j] += X[i * d + j];
    }
    classLogPriors[ki] = Math.log((nk + alpha) / (n + K * alpha));
    let total = 0;
    for (let j = 0; j < d; j++) total += featureLogProbs[ki * d + j] + alpha;
    // Now turn aggregated counts into log token probabilities.
    for (let j = 0; j < d; j++) {
      const num = featureLogProbs[ki * d + j] + alpha;
      featureLogProbs[ki * d + j] = Math.log(num / total);
    }
  }
  return { classes, classLogPriors, featureLogProbs, d, alpha };
}

/** Unnormalized log posterior log P(y_k | x) ∝ log π_k + Σ_j x_j log P(token_j | k). */
export function multinomialNBLogPredict(X: Float64Array, model: MultinomialNBModel): Float64Array {
  const { classLogPriors, featureLogProbs, d } = model;
  const K = model.classes.length;
  const n = X.length / d;
  const out = new Float64Array(n * K);
  for (let i = 0; i < n; i++) {
    for (let ki = 0; ki < K; ki++) {
      let s = classLogPriors[ki];
      for (let j = 0; j < d; j++) s += X[i * d + j] * featureLogProbs[ki * d + j];
      out[i * K + ki] = s;
    }
  }
  return out;
}

export function multinomialNBPredict(X: Float64Array, model: MultinomialNBModel): Int32Array {
  const lp = multinomialNBLogPredict(X, model);
  const K = model.classes.length;
  const n = lp.length / K;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let best = -Infinity;
    let arg = 0;
    for (let ki = 0; ki < K; ki++) {
      const v = lp[i * K + ki];
      if (v > best) {
        best = v;
        arg = ki;
      }
    }
    out[i] = model.classes[arg];
  }
  return out;
}

export function multinomialNBPredictProba(
  X: Float64Array,
  model: MultinomialNBModel,
): Float64Array {
  const lp = multinomialNBLogPredict(X, model);
  const K = model.classes.length;
  const n = lp.length / K;
  const out = new Float64Array(n * K);
  const row = new Array<number>(K);
  for (let i = 0; i < n; i++) {
    for (let ki = 0; ki < K; ki++) row[ki] = lp[i * K + ki];
    const lse = logSumExp(row);
    for (let ki = 0; ki < K; ki++) out[i * K + ki] = Math.exp(row[ki] - lse);
  }
  return out;
}

// =============================================================================
// Bernoulli Naive Bayes (binary features)
// =============================================================================

export interface BernoulliNBModel {
  classes: number[];
  classLogPriors: Float64Array;
  /** log P(x_j = 1 | k) (shape K × d row-major). */
  featureLogProbs: Float64Array;
  /** log P(x_j = 0 | k) = log(1 - exp(featureLogProbs)). */
  featureLogNegProbs: Float64Array;
  d: number;
  alpha: number;
}

/** Fit Bernoulli NB. X may be {0, 1}-valued (will be binarized if not). */
export function bernoulliNBFit(
  X: Float64Array,
  y: Int32Array,
  d: number,
  alpha: number = 1.0,
  binarize: number | null = 0,
): BernoulliNBModel {
  const n = y.length;
  const Xb = binarize === null ? X : new Float64Array(X.length);
  if (binarize !== null) for (let i = 0; i < X.length; i++) Xb[i] = X[i] > binarize ? 1 : 0;
  const seen = new Set<number>();
  for (let i = 0; i < n; i++) seen.add(y[i]);
  const classes = Array.from(seen).sort((a, b) => a - b);
  const K = classes.length;
  const classLogPriors = new Float64Array(K);
  const featureLogProbs = new Float64Array(K * d);
  const featureLogNegProbs = new Float64Array(K * d);

  for (let ki = 0; ki < K; ki++) {
    const ck = classes[ki];
    let nk = 0;
    const counts = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      if (y[i] !== ck) continue;
      nk++;
      for (let j = 0; j < d; j++) counts[j] += Xb[i * d + j];
    }
    classLogPriors[ki] = Math.log((nk + alpha) / (n + K * alpha));
    for (let j = 0; j < d; j++) {
      const p = (counts[j] + alpha) / (nk + 2 * alpha);
      featureLogProbs[ki * d + j] = Math.log(p);
      featureLogNegProbs[ki * d + j] = Math.log(1 - p);
    }
  }
  return { classes, classLogPriors, featureLogProbs, featureLogNegProbs, d, alpha };
}

export function bernoulliNBLogPredict(X: Float64Array, model: BernoulliNBModel): Float64Array {
  const { classLogPriors, featureLogProbs, featureLogNegProbs, d } = model;
  const K = model.classes.length;
  const n = X.length / d;
  const out = new Float64Array(n * K);
  for (let i = 0; i < n; i++) {
    for (let ki = 0; ki < K; ki++) {
      let s = classLogPriors[ki];
      for (let j = 0; j < d; j++) {
        const xj = X[i * d + j] > 0 ? 1 : 0;
        s += xj * featureLogProbs[ki * d + j] + (1 - xj) * featureLogNegProbs[ki * d + j];
      }
      out[i * K + ki] = s;
    }
  }
  return out;
}

export function bernoulliNBPredictProba(X: Float64Array, model: BernoulliNBModel): Float64Array {
  const lp = bernoulliNBLogPredict(X, model);
  const K = model.classes.length;
  const n = lp.length / K;
  const out = new Float64Array(n * K);
  const row = new Array<number>(K);
  for (let i = 0; i < n; i++) {
    for (let ki = 0; ki < K; ki++) row[ki] = lp[i * K + ki];
    const lse = logSumExp(row);
    for (let ki = 0; ki < K; ki++) out[i * K + ki] = Math.exp(row[ki] - lse);
  }
  return out;
}

// =============================================================================
// Dirichlet–multinomial conjugate helpers (§6)
// =============================================================================

/** Posterior mean of a Dirichlet(α + n_1, ..., α + n_K) after seeing counts. */
export function dirichletPosteriorMean(counts: ArrayLike<number>, alpha: number): Float64Array {
  const K = counts.length;
  const out = new Float64Array(K);
  let total = 0;
  for (let k = 0; k < K; k++) total += counts[k] + alpha;
  for (let k = 0; k < K; k++) out[k] = (counts[k] + alpha) / total;
  return out;
}

/** log Dirichlet–multinomial posterior-predictive of a single new observation x ∈ {0, ..., K-1}.
 *  log P(x = k | counts) = log((counts_k + α) / Σ_j (counts_j + α)). */
export function dirichletLogPosteriorPredictive(
  counts: ArrayLike<number>,
  alpha: number,
  k: number,
): number {
  const K = counts.length;
  let total = 0;
  for (let j = 0; j < K; j++) total += counts[j] + alpha;
  return Math.log((counts[k] + alpha) / total);
}

// =============================================================================
// Bayes risk on a 1D two-class Gaussian mixture (§3.2)
// =============================================================================

/** Closed-form Bayes risk for a 1D two-class Gaussian mixture
 *    Y ~ Bernoulli(π), X | Y = 1 ~ N(μ₁, σ₁²), X | Y = 0 ~ N(μ₂, σ₂²).
 *  Computed by integrating min(π·f₁(x), (1−π)·f₀(x)) over a fine grid.
 *  For the symmetric equal-variance case (σ₁ = σ₂ = σ, π = 1/2) this equals
 *    Φ(−|μ₁ − μ₂| / (2σ))   (e.g. Φ(−1) ≈ 0.158655 for μ_diff = 2, σ = 1). */
export function bayesRiskGaussianMixture1D(
  mu1: number,
  sigma1: number,
  mu2: number,
  sigma2: number,
  prior: number = 0.5,
  gridSize: number = 4001,
): number {
  // Integrate over a generous range covering both class densities.
  const lo = Math.min(mu1 - 8 * sigma1, mu2 - 8 * sigma2);
  const hi = Math.max(mu1 + 8 * sigma1, mu2 + 8 * sigma2);
  const dx = (hi - lo) / (gridSize - 1);
  let risk = 0;
  // Trapezoidal rule on min(π·f₁(x), (1−π)·f₀(x)).
  for (let i = 0; i < gridSize; i++) {
    const x = lo + i * dx;
    const f1 = normPdf((x - mu1) / sigma1) / sigma1;
    const f0 = normPdf((x - mu2) / sigma2) / sigma2;
    const integrand = Math.min(prior * f1, (1 - prior) * f0);
    risk += i === 0 || i === gridSize - 1 ? 0.5 * integrand : integrand;
  }
  return risk * dx;
}

// =============================================================================
// Platt scaling (§8.2) — fit a sigmoid to (raw scores, labels) on a held-out set.
// =============================================================================

export interface PlattParams {
  /** Sigmoid slope. Negative for the convention P(y=1|s) = 1/(1+exp(A·s + B)). */
  A: number;
  /** Sigmoid intercept. */
  B: number;
}

/** Fit Platt scaling: logistic regression of labels on raw 1D scores.
 *  Returns {A, B} where P(y=1|s) = 1 / (1 + exp(A·s + B)).
 *  Conventions match scikit-learn's CalibratedClassifierCV (sigmoid method). */
export function plattScalingFit(scores: ArrayLike<number>, labels: ArrayLike<number>): PlattParams {
  const n = scores.length;
  const X = new Float64Array(n);
  const D = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    X[i] = scores[i];
    D[i] = labels[i] > 0 ? 1 : 0;
  }
  // logisticFit returns [intercept, slope] for the model logit(p) = intercept + slope·s,
  // i.e. P(y=1|s) = sigmoid(intercept + slope·s) = 1 / (1 + exp(-(intercept + slope·s))).
  // Platt's form is P(y=1|s) = 1 / (1 + exp(A·s + B)), so A = -slope, B = -intercept.
  const beta = logisticFit(X, D, n, 1);
  return { A: -beta[1], B: -beta[0] };
}

export function plattScalingApply(scores: ArrayLike<number>, params: PlattParams): Float64Array {
  const n = scores.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(params.A * scores[i] + params.B));
  return out;
}

// =============================================================================
// Calibration losses (§8.4)
// =============================================================================

/** Expected calibration error with quantile (equal-mass) binning. */
export function expectedCalibrationError(
  probs: ArrayLike<number>,
  labels: ArrayLike<number>,
  nBins: number = 10,
): number {
  const n = probs.length;
  // Sort by predicted probability to do quantile binning.
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => probs[a] - probs[b]);
  let ece = 0;
  let binStart = 0;
  for (let b = 0; b < nBins; b++) {
    const binEnd = Math.round(((b + 1) * n) / nBins);
    if (binEnd <= binStart) continue;
    let pSum = 0;
    let yMean = 0;
    for (let i = binStart; i < binEnd; i++) {
      pSum += probs[order[i]];
      yMean += labels[order[i]] > 0 ? 1 : 0;
    }
    const m = binEnd - binStart;
    ece += Math.abs(pSum / m - yMean / m);
    binStart = binEnd;
  }
  return ece / nBins;
}

/** Brier score: mean squared error of (predicted prob, 0/1 label). */
export function brierScore(probs: ArrayLike<number>, labels: ArrayLike<number>): number {
  const n = probs.length;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const y = labels[i] > 0 ? 1 : 0;
    const d = probs[i] - y;
    s += d * d;
  }
  return s / n;
}

/** Binary log-loss with epsilon clamping. */
export function logLoss(
  probs: ArrayLike<number>,
  labels: ArrayLike<number>,
  eps: number = 1e-15,
): number {
  const n = probs.length;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.max(eps, Math.min(1 - eps, probs[i]));
    s += labels[i] > 0 ? -Math.log(p) : -Math.log(1 - p);
  }
  return s / n;
}

// =============================================================================
// Mutual information + Chow–Liu tree (§9.1)
// =============================================================================

/** Laplace-smoothed mutual information I(X; Y) for two discrete features.
 *  X and Y are length-n arrays of integer category labels (need not be 0..K-1). */
export function mutualInformationDiscreteSmoothed(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  alpha: number = 1.0,
): number {
  const n = x.length;
  // Build per-feature category indexes.
  const xCats = Array.from(new Set(Array.from({ length: n }, (_, i) => x[i]))).sort((a, b) => a - b);
  const yCats = Array.from(new Set(Array.from({ length: n }, (_, i) => y[i]))).sort((a, b) => a - b);
  const xIdx = new Map<number, number>();
  const yIdx = new Map<number, number>();
  xCats.forEach((c, i) => xIdx.set(c, i));
  yCats.forEach((c, i) => yIdx.set(c, i));
  const Kx = xCats.length;
  const Ky = yCats.length;
  const joint = new Float64Array(Kx * Ky);
  for (let i = 0; i < n; i++) joint[xIdx.get(x[i])! * Ky + yIdx.get(y[i])!] += 1;
  // Smooth and normalize.
  let total = 0;
  for (let k = 0; k < Kx * Ky; k++) {
    joint[k] += alpha;
    total += joint[k];
  }
  for (let k = 0; k < Kx * Ky; k++) joint[k] /= total;
  const px = new Float64Array(Kx);
  const py = new Float64Array(Ky);
  for (let i = 0; i < Kx; i++) for (let j = 0; j < Ky; j++) px[i] += joint[i * Ky + j];
  for (let i = 0; i < Kx; i++) for (let j = 0; j < Ky; j++) py[j] += joint[i * Ky + j];
  let mi = 0;
  for (let i = 0; i < Kx; i++) {
    for (let j = 0; j < Ky; j++) {
      const p = joint[i * Ky + j];
      if (p > 0) mi += p * Math.log(p / (px[i] * py[j]));
    }
  }
  return mi;
}

export interface ChowLiuResult {
  /** Edges in the maximum-mutual-information spanning tree, as [feat_i, feat_j] pairs. */
  edges: Array<[number, number]>;
  /** Symmetric d×d MI matrix (row-major). */
  miMatrix: Float64Array;
}

/** Chow–Liu maximum-mutual-information spanning tree on a discrete feature matrix.
 *  X is shape (n, d). Returns the d-1 edge spanning tree (Prim's algorithm) and the
 *  full pairwise MI matrix for downstream display. */
export function chowLiuTree(X: ArrayLike<number>, n: number, d: number, alpha: number = 1.0): ChowLiuResult {
  const miMatrix = new Float64Array(d * d);
  const col = new Array<Float64Array>(d);
  for (let j = 0; j < d; j++) {
    col[j] = new Float64Array(n);
    for (let i = 0; i < n; i++) col[j][i] = X[i * d + j];
  }
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      const mi = mutualInformationDiscreteSmoothed(col[i], col[j], alpha);
      miMatrix[i * d + j] = mi;
      miMatrix[j * d + i] = mi;
    }
  }
  // Prim's algorithm to greedily build the maximum-spanning-tree.
  const inTree = new Uint8Array(d);
  inTree[0] = 1;
  const edges: Array<[number, number]> = [];
  while (edges.length < d - 1) {
    let bestMi = -Infinity;
    let bestI = -1;
    let bestJ = -1;
    for (let i = 0; i < d; i++) {
      if (!inTree[i]) continue;
      for (let j = 0; j < d; j++) {
        if (inTree[j]) continue;
        const w = miMatrix[i * d + j];
        if (w > bestMi) {
          bestMi = w;
          bestI = i;
          bestJ = j;
        }
      }
    }
    if (bestJ < 0) break;
    edges.push([bestI, bestJ]);
    inTree[bestJ] = 1;
  }
  return { edges, miMatrix };
}

// =============================================================================
// Tree-Augmented NB (TAN) — Chow–Liu structure on the features conditional on Y.
// =============================================================================

export interface TANModel {
  classes: number[];
  classLogPriors: Float64Array;
  /** Edge list of the class-conditional Chow–Liu tree (d-1 edges). */
  edges: Array<[number, number]>;
  /** For each class k, for each feature j: marginal log P(x_j | y = k).
   *  Shape: K × d × |cats_j| (flattened with offsets in `catOffsets`). */
  marginalLogProbs: Float64Array;
  /** For each class k, for each child j with parent par[j], for each (parent_val, child_val) pair:
   *  log P(x_j = c | x_par[j] = p, y = k). Stored as flat array; see paramOffsets. */
  conditionalLogProbs: Float64Array;
  /** Length-d array: feature j's parent in the TAN tree (or -1 if root). */
  parents: Int32Array;
  /** Per-feature category counts. */
  numCats: Int32Array;
  /** Offsets into marginalLogProbs for each (k, j). */
  marginalOffsets: Int32Array;
  /** Offsets into conditionalLogProbs for each (k, j). */
  conditionalOffsets: Int32Array;
  d: number;
}

/** Fit TAN: per-class Chow–Liu tree + conditional/marginal log probabilities with Laplace smoothing. */
export function tanFit(
  X: ArrayLike<number>,
  y: Int32Array,
  n: number,
  d: number,
  alpha: number = 1.0,
): TANModel {
  const seen = new Set<number>();
  for (let i = 0; i < n; i++) seen.add(y[i]);
  const classes = Array.from(seen).sort((a, b) => a - b);
  const K = classes.length;

  // Per-feature category enumeration.
  const numCats = new Int32Array(d);
  const featCats: Array<number[]> = [];
  for (let j = 0; j < d; j++) {
    const cs = new Set<number>();
    for (let i = 0; i < n; i++) cs.add(X[i * d + j]);
    const arr = Array.from(cs).sort((a, b) => a - b);
    featCats.push(arr);
    numCats[j] = arr.length;
  }
  const featCatIdx: Array<Map<number, number>> = featCats.map((arr) => {
    const m = new Map<number, number>();
    arr.forEach((c, i) => m.set(c, i));
    return m;
  });

  // Compute conditional-MI(X_i, X_j | Y) by averaging per-class MI weighted by class prior.
  const classMembers: Array<number[]> = [];
  for (let ki = 0; ki < K; ki++) {
    const members: number[] = [];
    const ck = classes[ki];
    for (let i = 0; i < n; i++) if (y[i] === ck) members.push(i);
    classMembers.push(members);
  }
  const condMi = new Float64Array(d * d);
  for (let a = 0; a < d; a++) {
    for (let b = a + 1; b < d; b++) {
      let mi = 0;
      for (let ki = 0; ki < K; ki++) {
        const idxs = classMembers[ki];
        if (idxs.length === 0) continue;
        const xa = new Float64Array(idxs.length);
        const xb = new Float64Array(idxs.length);
        for (let t = 0; t < idxs.length; t++) {
          xa[t] = X[idxs[t] * d + a];
          xb[t] = X[idxs[t] * d + b];
        }
        mi += (idxs.length / n) * mutualInformationDiscreteSmoothed(xa, xb, alpha);
      }
      condMi[a * d + b] = mi;
      condMi[b * d + a] = mi;
    }
  }

  // Maximum-spanning-tree on the conditional-MI graph (Prim). Root = 0.
  const inTree = new Uint8Array(d);
  inTree[0] = 1;
  const parents = new Int32Array(d).fill(-1);
  for (let step = 0; step < d - 1; step++) {
    let best = -Infinity;
    let bi = -1;
    let bj = -1;
    for (let i = 0; i < d; i++) {
      if (!inTree[i]) continue;
      for (let j = 0; j < d; j++) {
        if (inTree[j]) continue;
        const w = condMi[i * d + j];
        if (w > best) {
          best = w;
          bi = i;
          bj = j;
        }
      }
    }
    if (bj < 0) break;
    parents[bj] = bi;
    inTree[bj] = 1;
  }

  // Layout offsets for the flat parameter arrays.
  const marginalOffsets = new Int32Array(K * d + 1);
  for (let ki = 0; ki < K; ki++) for (let j = 0; j < d; j++) {
    marginalOffsets[ki * d + j + 1] = marginalOffsets[ki * d + j] + numCats[j];
  }
  const marginalLogProbs = new Float64Array(marginalOffsets[K * d]);
  const conditionalOffsets = new Int32Array(K * d + 1);
  for (let ki = 0; ki < K; ki++) for (let j = 0; j < d; j++) {
    const par = parents[j];
    const size = par < 0 ? 0 : numCats[par] * numCats[j];
    conditionalOffsets[ki * d + j + 1] = conditionalOffsets[ki * d + j] + size;
  }
  const conditionalLogProbs = new Float64Array(conditionalOffsets[K * d]);

  // Class priors with add-α smoothing.
  const classLogPriors = new Float64Array(K);
  for (let ki = 0; ki < K; ki++) {
    classLogPriors[ki] = Math.log((classMembers[ki].length + alpha) / (n + K * alpha));
  }

  // For each class, compute marginal P(x_j | y) and (if root absent) conditional P(x_j | x_par[j], y).
  for (let ki = 0; ki < K; ki++) {
    const members = classMembers[ki];
    for (let j = 0; j < d; j++) {
      const par = parents[j];
      const c_j = numCats[j];
      if (par < 0) {
        // Root: marginal P(x_j | y).
        const counts = new Float64Array(c_j);
        for (const i of members) counts[featCatIdx[j].get(X[i * d + j])!] += 1;
        let total = 0;
        for (let v = 0; v < c_j; v++) total += counts[v] + alpha;
        const off = marginalOffsets[ki * d + j];
        for (let v = 0; v < c_j; v++) {
          marginalLogProbs[off + v] = Math.log((counts[v] + alpha) / total);
        }
      } else {
        const c_par = numCats[par];
        const counts = new Float64Array(c_par * c_j);
        const parTotals = new Float64Array(c_par);
        for (const i of members) {
          const vp = featCatIdx[par].get(X[i * d + par])!;
          const vj = featCatIdx[j].get(X[i * d + j])!;
          counts[vp * c_j + vj] += 1;
        }
        for (let vp = 0; vp < c_par; vp++) {
          for (let vj = 0; vj < c_j; vj++) parTotals[vp] += counts[vp * c_j + vj] + alpha;
        }
        const off = conditionalOffsets[ki * d + j];
        for (let vp = 0; vp < c_par; vp++) {
          for (let vj = 0; vj < c_j; vj++) {
            conditionalLogProbs[off + vp * c_j + vj] = Math.log(
              (counts[vp * c_j + vj] + alpha) / parTotals[vp],
            );
          }
        }
      }
    }
  }

  // Convert internal Map<number, number> indexes into a normalized 0..K-1 form by
  // letting downstream code re-resolve via featCats (we surface only numCats).
  // For predict-time use, build the same featCatIdx lookups from `numCats` + the
  // original cats — but for simplicity TANModel exposes integer-feature-input.
  // The model assumes X is already normalized into integer categories 0..numCats[j]-1.
  const edges: Array<[number, number]> = [];
  for (let j = 0; j < d; j++) if (parents[j] >= 0) edges.push([parents[j], j]);

  return {
    classes,
    classLogPriors,
    edges,
    marginalLogProbs,
    conditionalLogProbs,
    parents,
    numCats,
    marginalOffsets,
    conditionalOffsets,
    d,
  };
}

/** TAN predictions on integer-categorical features (X values must lie in [0, numCats[j])). */
export function tanLogPredict(X: ArrayLike<number>, model: TANModel): Float64Array {
  const { classLogPriors, parents, marginalLogProbs, conditionalLogProbs, marginalOffsets, conditionalOffsets, numCats, d } = model;
  const K = model.classes.length;
  const n = X.length / d;
  const out = new Float64Array(n * K);
  for (let i = 0; i < n; i++) {
    for (let ki = 0; ki < K; ki++) {
      let s = classLogPriors[ki];
      for (let j = 0; j < d; j++) {
        const xj = X[i * d + j];
        const par = parents[j];
        if (par < 0) {
          s += marginalLogProbs[marginalOffsets[ki * d + j] + xj];
        } else {
          const xp = X[i * d + par];
          s += conditionalLogProbs[conditionalOffsets[ki * d + j] + xp * numCats[j] + xj];
        }
      }
      out[i * K + ki] = s;
    }
  }
  return out;
}

export function tanPredictProba(X: ArrayLike<number>, model: TANModel): Float64Array {
  const lp = tanLogPredict(X, model);
  const K = model.classes.length;
  const n = lp.length / K;
  const out = new Float64Array(n * K);
  const row = new Array<number>(K);
  for (let i = 0; i < n; i++) {
    for (let ki = 0; ki < K; ki++) row[ki] = lp[i * K + ki];
    const lse = logSumExp(row);
    for (let ki = 0; ki < K; ki++) out[i * K + ki] = Math.exp(row[ki] - lse);
  }
  return out;
}

// =============================================================================
// Pass-through re-exports for the NBVsLogistic correlated-features viz (§10.3).
// =============================================================================

export { logisticFit, logisticPredictProba };

// =============================================================================
// bayesian-ml.ts
//
// Shared math/types/palette module for the T5 Bayesian & Probabilistic ML track.
// Consumers: variational-inference (VI types/palette only), bayesian-neural-
// networks (full §5 primitive set: MLP machinery, deep ensembles, MC-dropout,
// Laplace, SG-MCMC, calibration, NNGP arc-cosine kernel).
//
// The BNN section below mirrors notebooks/bayesian-neural-networks/
// 01_bayesian_neural_networks.ipynb cells 4, 8, 10, 12, 14, 16, 18, 20 in
// TypeScript. The notebook is the source of truth for math; tolerances
// in the verification script (verify-bayesian-neural-networks.ts) reflect the
// inevitable RNG-divergence between PyTorch's mt19937 and our seeded
// mulberry32 — exact values differ, but distributional invariants hold.
// =============================================================================

import { choleskyFactor } from './gaussian-processes';

// =============================================================================
// NUMERICAL CONSTANTS
// =============================================================================
// Pulled out so callers in MDX viz components and downstream consumers can
// reference the same values rather than hardcoding magic numbers in
// presentation code.
// =============================================================================

/** Probability clip used by BCE loss + temperature scaling to avoid log(0). */
export const BCE_EPSILON = 1e-7;

/** Adam optimizer denominator floor. */
export const ADAM_EPSILON = 1e-8;

/** Tikhonov stabilization added to Hessian before Cholesky. */
export const HESSIAN_STABILIZATION = 1e-3;

/** Floor below which we treat eigenvalues as zero when reporting condition numbers. */
export const EIGENVALUE_FLOOR = 1e-12;

// -----------------------------------------------------------------------------
// VI palette + types (existing — first consumer: variational-inference)
// -----------------------------------------------------------------------------

export const paletteVI = {
  posterior: '#1f77b4', // blue   — true posterior contours
  meanField: '#d62728', // red    — mean-field variational q
  fullRank: '#ff7f0e', // orange — full-rank Gaussian q
  flow: '#2ca02c', // green  — normalizing-flow q
  data: '#7f7f7f', // gray   — sample / data points
  reference: '#000000', // black  — reference lines (e.g., log p(x))
} as const;

export type VIColorKey = keyof typeof paletteVI;

export type VariationalFamily = 'meanField' | 'fullRank' | 'flow';

export interface VISample2D {
  x: number;
  y: number;
}

export interface ELBOTrajectoryPoint {
  iteration: number;
  elbo: number;
  family: VariationalFamily | null;
}

export interface GaussianMarginal {
  mean: number;
  std: number;
}

// =============================================================================
// BNN PALETTE
// =============================================================================
// Method-color tokens mirroring the matplotlib palette in the notebook setup
// cell. Used by the §1, §3–§8 viz components to keep cross-section visual
// continuity (Laplace is always blue, MC-dropout always green, etc.).
// =============================================================================

export const paletteBNN = {
  point: '#7f7f7f', //  gray   — point-estimate baseline
  laplace: '#1f77b4', // blue   — §3
  dropout: '#2ca02c', // green  — §4
  ensemble: '#ff7f0e', // orange — §5
  sgld: '#9467bd', //   purple — §6
  sghmc: '#d62728', //  red    — §7
  nngp: '#17becf', //   teal   — §9
  data: '#000000', //   black  — data points
  posterior: '#bbbbbb', // light-gray — posterior reference
} as const;

export type BNNColorKey = keyof typeof paletteBNN;

// =============================================================================
// CORE INTERFACES (brief §5)
// =============================================================================

export interface MLPArchSpec {
  inputDim: number;
  hiddenDims: number[];
  outputDim: number;
  activation: 'relu';
  dropoutP?: number;
}

export interface TrainingSpec {
  lr: number;
  weightDecay: number;
  epochs: number;
  optimizer: 'adam';
  seed: number;
}

export interface TrainingData {
  X: Float32Array; // shape [n × inputDim] flat row-major
  y: Float32Array; // shape [n] in {0, 1}
  n: number;
}

export interface TrainedMLP {
  arch: MLPArchSpec;
  weights: Float32Array;
  pDim: number;
  /** Returns sigmoid probabilities for an input grid (flat row-major). */
  forward: (gridFlat: Float32Array, gridN: number) => Float32Array;
  /** Final epoch BCE loss (training-set average). */
  finalLoss: number;
}

// =============================================================================
// DETERMINISTIC RNG (mulberry32) — seed-reproducible across browsers
// =============================================================================

export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return function rng(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller using a seeded uniform — returns two i.i.d. N(0,1) draws. */
export function gaussianPair(rng: () => number): [number, number] {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const r = Math.sqrt(-2 * Math.log(u));
  return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
}

// =============================================================================
// MLP MACHINERY
// =============================================================================
// Hand-rolled MLP forward + backward + Adam, designed for small architectures
// (notebook §1 uses 2 → 32 → 32 → 32 → 1 with p = 2241). Float32 typed arrays
// for speed; seeded init for reproducibility.
// =============================================================================

export interface MLPLayout {
  /** Number of weight parameters per layer (excluding biases). */
  wSizes: number[];
  /** Number of bias parameters per layer. */
  bSizes: number[];
  /** Cumulative offset of each layer's W block in the flat parameter vector. */
  wOffsets: number[];
  /** Cumulative offset of each layer's b block. */
  bOffsets: number[];
  /** Total parameter dimension. */
  pDim: number;
  /** [in, h1, h2, ..., out] sizes per neuron-count. */
  layerSizes: number[];
}

export function mlpLayout(arch: MLPArchSpec): MLPLayout {
  const layerSizes = [arch.inputDim, ...arch.hiddenDims, arch.outputDim];
  const wSizes: number[] = [];
  const bSizes: number[] = [];
  const wOffsets: number[] = [];
  const bOffsets: number[] = [];
  let off = 0;
  for (let k = 0; k < layerSizes.length - 1; k++) {
    const inSz = layerSizes[k];
    const outSz = layerSizes[k + 1];
    wSizes.push(inSz * outSz);
    bSizes.push(outSz);
    wOffsets.push(off);
    off += inSz * outSz;
  }
  for (let k = 0; k < bSizes.length; k++) {
    bOffsets.push(off);
    off += bSizes[k];
  }
  return { wSizes, bSizes, wOffsets, bOffsets, pDim: off, layerSizes };
}

/** He initialization (gaussian, scaled by sqrt(2/fanIn)) — standard for ReLU. */
export function mlpInit(arch: MLPArchSpec, seed: number): Float32Array {
  const layout = mlpLayout(arch);
  const w = new Float32Array(layout.pDim);
  const rng = mulberry32(seed);
  for (let k = 0; k < layout.wSizes.length; k++) {
    const inSz = layout.layerSizes[k];
    const scale = Math.sqrt(2 / inSz);
    const off = layout.wOffsets[k];
    for (let i = 0; i < layout.wSizes[k]; i += 2) {
      const [g0, g1] = gaussianPair(rng);
      w[off + i] = g0 * scale;
      if (i + 1 < layout.wSizes[k]) w[off + i + 1] = g1 * scale;
    }
  }
  // biases default to zero (already 0 from Float32Array)
  return w;
}

/**
 * Forward pass over a batch (flat row-major X, shape [n × inputDim]) returning
 * sigmoid probabilities (flat [n], outputDim assumed 1) plus optional cached
 * activations for backprop.
 */
export interface MLPForwardCache {
  acts: Float32Array[];
  preActs: Float32Array[];
}

export function mlpForward(
  X: Float32Array,
  n: number,
  weights: Float32Array,
  arch: MLPArchSpec,
  layout: MLPLayout,
  cache: MLPForwardCache | null,
  dropoutMasks: Float32Array[] | null,
): Float32Array {
  if (arch.outputDim !== 1) {
    throw new Error('mlpForward currently supports outputDim=1 (binary BCE) only');
  }
  let activations = X;
  if (cache) {
    cache.acts = [X];
    cache.preActs = [];
  }
  for (let k = 0; k < layout.wSizes.length; k++) {
    const inSz = layout.layerSizes[k];
    const outSz = layout.layerSizes[k + 1];
    const wOff = layout.wOffsets[k];
    const bOff = layout.bOffsets[k];
    const z = new Float32Array(n * outSz);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < outSz; j++) {
        let acc = weights[bOff + j];
        for (let m = 0; m < inSz; m++) {
          acc += activations[i * inSz + m] * weights[wOff + m * outSz + j];
        }
        z[i * outSz + j] = acc;
      }
    }
    if (cache) cache.preActs.push(z);
    const isLast = k === layout.wSizes.length - 1;
    const a = new Float32Array(n * outSz);
    if (isLast) {
      // sigmoid for binary BCE
      for (let i = 0; i < n * outSz; i++) {
        a[i] = 1 / (1 + Math.exp(-z[i]));
      }
    } else {
      // ReLU + optional dropout (training)
      for (let i = 0; i < n * outSz; i++) {
        a[i] = z[i] > 0 ? z[i] : 0;
      }
      if (dropoutMasks && dropoutMasks[k]) {
        const m = dropoutMasks[k];
        for (let i = 0; i < n * outSz; i++) a[i] *= m[i % outSz];
      }
    }
    if (cache) cache.acts.push(a);
    activations = a;
  }
  return activations;
}

/**
 * One BCE training step on a minibatch. Returns gradient (Float32Array, pDim)
 * and the per-batch BCE loss.
 */
function mlpBackward(
  X: Float32Array,
  y: Float32Array,
  n: number,
  weights: Float32Array,
  arch: MLPArchSpec,
  layout: MLPLayout,
  weightDecay: number,
  dropoutMasks: Float32Array[] | null,
): { grad: Float32Array; loss: number } {
  const cache: MLPForwardCache = { acts: [], preActs: [] };
  const probs = mlpForward(X, n, weights, arch, layout, cache, dropoutMasks);
  const acts = cache.acts;
  const preActs = cache.preActs;
  // BCE loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.min(Math.max(probs[i], BCE_EPSILON), 1 - BCE_EPSILON);
    loss += -y[i] * Math.log(p) - (1 - y[i]) * Math.log(1 - p);
  }
  loss /= n;
  // Backprop
  const grad = new Float32Array(layout.pDim);
  // Output-layer gradient: dL/dz = (probs - y) / n  (sigmoid + BCE)
  let delta = new Float32Array(n);
  for (let i = 0; i < n; i++) delta[i] = (probs[i] - y[i]) / n;
  for (let k = layout.wSizes.length - 1; k >= 0; k--) {
    const inSz = layout.layerSizes[k];
    const outSz = layout.layerSizes[k + 1];
    const wOff = layout.wOffsets[k];
    const bOff = layout.bOffsets[k];
    const a = acts[k];
    // bias grad = sum_i delta_{i,j}
    for (let j = 0; j < outSz; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += delta[i * outSz + j];
      grad[bOff + j] = sum;
    }
    // weight grad = a^T delta + weightDecay * w
    for (let m = 0; m < inSz; m++) {
      for (let j = 0; j < outSz; j++) {
        let sum = 0;
        for (let i = 0; i < n; i++) sum += a[i * inSz + m] * delta[i * outSz + j];
        grad[wOff + m * outSz + j] = sum + weightDecay * weights[wOff + m * outSz + j];
      }
    }
    if (k > 0) {
      const newDelta = new Float32Array(n * inSz);
      const z = preActs[k - 1];
      const dropMask = dropoutMasks && dropoutMasks[k - 1] ? dropoutMasks[k - 1] : null;
      for (let i = 0; i < n; i++) {
        for (let m = 0; m < inSz; m++) {
          let sum = 0;
          for (let j = 0; j < outSz; j++) {
            sum += delta[i * outSz + j] * weights[wOff + m * outSz + j];
          }
          // ReLU' on previous preact
          const reluD = z[i * inSz + m] > 0 ? 1 : 0;
          let d = sum * reluD;
          if (dropMask) d *= dropMask[m];
          newDelta[i * inSz + m] = d;
        }
      }
      delta = newDelta;
    }
  }
  return { grad, loss };
}

/** One Adam step in place. */
export function adamStep(
  weights: Float32Array,
  grad: Float32Array,
  m: Float32Array,
  v: Float32Array,
  t: number,
  lr: number,
  beta1 = 0.9,
  beta2 = 0.999,
  eps = ADAM_EPSILON,
): void {
  const corr1 = 1 - Math.pow(beta1, t);
  const corr2 = 1 - Math.pow(beta2, t);
  for (let i = 0; i < weights.length; i++) {
    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
    const mHat = m[i] / corr1;
    const vHat = v[i] / corr2;
    weights[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
  }
}

/** Train one MLP via full-batch Adam (no minibatch — Two Moons n=300 fits). */
export function mlpTrain(
  arch: MLPArchSpec,
  training: TrainingSpec,
  data: TrainingData,
  dropoutMasksProvider?: (epoch: number) => Float32Array[] | null,
): TrainedMLP {
  const layout = mlpLayout(arch);
  const weights = mlpInit(arch, training.seed);
  const m = new Float32Array(layout.pDim);
  const v = new Float32Array(layout.pDim);
  let lastLoss = 0;
  for (let e = 1; e <= training.epochs; e++) {
    const masks = dropoutMasksProvider ? dropoutMasksProvider(e) : null;
    const { grad, loss } = mlpBackward(
      data.X,
      data.y,
      data.n,
      weights,
      arch,
      layout,
      training.weightDecay,
      masks,
    );
    adamStep(weights, grad, m, v, e, training.lr);
    lastLoss = loss;
  }
  const forward = (gridFlat: Float32Array, gridN: number): Float32Array =>
    mlpForward(gridFlat, gridN, weights, arch, layout, null, null);
  return { arch, weights, pDim: layout.pDim, forward, finalLoss: lastLoss };
}

// =============================================================================
// §5 PRIMITIVE 1: deepEnsembleTraining (notebook cells 4, 12)
// =============================================================================

export interface DeepEnsembleResult {
  weights: Float32Array[];
  finalLosses: number[];
  predictOnGrid: (gridFlat: Float32Array, gridN: number) => Float32Array[];
}

export function deepEnsembleTraining(
  arch: MLPArchSpec,
  training: TrainingSpec,
  K: number,
  data: TrainingData,
): DeepEnsembleResult {
  const trained: TrainedMLP[] = [];
  for (let k = 0; k < K; k++) {
    const t = { ...training, seed: training.seed + k };
    trained.push(mlpTrain(arch, t, data));
  }
  return {
    weights: trained.map((m) => m.weights),
    finalLosses: trained.map((m) => m.finalLoss),
    predictOnGrid: (gridFlat, gridN) => trained.map((m) => m.forward(gridFlat, gridN)),
  };
}

// =============================================================================
// §5 PRIMITIVE 2: mcDropoutInference (notebook cell 10)
// =============================================================================

export interface MCDropoutResult {
  trainedModel: TrainedMLP;
  /** Returns T forward passes with fresh dropout masks. */
  predict: (gridFlat: Float32Array, gridN: number, T: number, seed: number) => Float32Array[];
}

export function mcDropoutInference(
  arch: MLPArchSpec,
  training: TrainingSpec,
  data: TrainingData,
): MCDropoutResult {
  if (!arch.dropoutP || arch.dropoutP <= 0) {
    throw new Error('mcDropoutInference requires arch.dropoutP > 0');
  }
  const dropP = arch.dropoutP;
  const layout = mlpLayout(arch);
  // Train with stochastic per-epoch dropout masks
  const masksRng = mulberry32(training.seed * 31 + 1);
  const masksProvider = (_e: number): Float32Array[] => {
    const masks: Float32Array[] = [];
    for (let k = 0; k < arch.hiddenDims.length; k++) {
      const sz = arch.hiddenDims[k];
      const mk = new Float32Array(sz);
      for (let i = 0; i < sz; i++) {
        // inverted dropout: keep with prob (1-p), scale by 1/(1-p)
        mk[i] = masksRng() < dropP ? 0 : 1 / (1 - dropP);
      }
      masks.push(mk);
    }
    return masks;
  };
  const trained = mlpTrain(arch, training, data, masksProvider);
  const predict = (
    gridFlat: Float32Array,
    gridN: number,
    T: number,
    seed: number,
  ): Float32Array[] => {
    const tRng = mulberry32(seed);
    const out: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const masks: Float32Array[] = [];
      for (let k = 0; k < arch.hiddenDims.length; k++) {
        const sz = arch.hiddenDims[k];
        const mk = new Float32Array(sz);
        for (let i = 0; i < sz; i++) {
          mk[i] = tRng() < dropP ? 0 : 1 / (1 - dropP);
        }
        masks.push(mk);
      }
      out.push(mlpForward(gridFlat, gridN, trained.weights, arch, layout, null, masks));
    }
    return out;
  };
  return { trainedModel: trained, predict };
}

// =============================================================================
// §3 PRIMITIVES: laplaceApproxBNN, lastLayerLaplace, kfacLaplace
// =============================================================================
// Full-Hessian Laplace at p=2241 is O(p²) finite-difference calls — too slow
// for in-browser. lastLayerLaplace computes the analytic Hessian over the
// last-layer weights only (BCE provides closed form: H = X^T diag(p(1-p)) X
// where X is the final-hidden-layer feature matrix). For v2 viz at the
// "full-hessian" dropdown, components consume precomputed JSON from
// notebooks/bayesian-neural-networks/precompute_laplace_grid.py.
// =============================================================================

export interface LaplaceResult {
  /** Trained MLP at the MAP. */
  map: TrainedMLP;
  /** Cholesky of the (last-layer or last-block) Hessian. */
  hessianCholesky: number[][];
  /** Effective rank — number of last-layer params (or full p for full Laplace). */
  pEffective: number;
  /** Condition number of the Hessian block. */
  conditionNumber: number;
  /** Sample S sets of last-layer weight perturbations from the Gaussian. */
  sampleWeights: (S: number, seed: number) => Float32Array[];
}

/**
 * Estimate κ(A) = λ_max(A) / λ_min(A) for a small SPD matrix A (Cholesky L
 * already computed). Uses power iteration on A for λ_max and on A^{-1} via
 * Cholesky solves for λ_min. ~10 iterations suffice for two-significant-digit
 * accuracy on the BNN last-layer block (~33×33).
 */
function estimateConditionNumberSPD(A: number[][], L: number[][]): number {
  const dim = A.length;
  const ITERS = 12;
  // Power iteration on A → dominant eigenvalue
  let v = new Array<number>(dim).fill(0);
  v[0] = 1;
  let lambdaMax = 0;
  for (let it = 0; it < ITERS; it++) {
    const Av = new Array<number>(dim).fill(0);
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) Av[i] += A[i][j] * v[j];
    }
    const norm = Math.sqrt(Av.reduce((s, x) => s + x * x, 0));
    if (norm < EIGENVALUE_FLOOR) break;
    for (let i = 0; i < dim; i++) v[i] = Av[i] / norm;
    lambdaMax = norm;
  }
  // Inverse power iteration via L L^T solves → smallest eigenvalue
  v = new Array<number>(dim).fill(0);
  v[0] = 1;
  let lambdaMin = 0;
  for (let it = 0; it < ITERS; it++) {
    // Solve L y = v (forward sub)
    const y = new Array<number>(dim).fill(0);
    for (let i = 0; i < dim; i++) {
      let acc = v[i];
      for (let j = 0; j < i; j++) acc -= L[i][j] * y[j];
      y[i] = acc / L[i][i];
    }
    // Solve L^T x = y (back sub)
    const x = new Array<number>(dim).fill(0);
    for (let i = dim - 1; i >= 0; i--) {
      let acc = y[i];
      for (let j = i + 1; j < dim; j++) acc -= L[j][i] * x[j];
      x[i] = acc / L[i][i];
    }
    const norm = Math.sqrt(x.reduce((s, xi) => s + xi * xi, 0));
    if (norm < EIGENVALUE_FLOOR) break;
    for (let i = 0; i < dim; i++) v[i] = x[i] / norm;
    lambdaMin = 1 / norm;
  }
  return lambdaMax / Math.max(lambdaMin, EIGENVALUE_FLOOR);
}

/**
 * Last-layer Laplace approximation given a *pre-trained* MAP. Computes the
 * Hessian H = Φ^T diag(p(1-p)) Φ + λI in closed form over the final-layer
 * weights only, where Φ is the final-hidden-layer feature matrix and p is
 * the predictive probability. Cholesky-factor H, then draw N(0, H^{-1})
 * perturbations to the last-layer weights for sampling.
 *
 * Decoupling Hessian from training is what enables interactive prior-scale
 * (τ²) sliders in the §3 viz: the MAP is computed once, then re-Hessian'd
 * cheaply (~ms) on every τ² tick.
 */
export function lastLayerLaplaceFromMAP(
  map: TrainedMLP,
  data: TrainingData,
  weightDecay: number,
): LaplaceResult {
  const arch = map.arch;
  const layout = mlpLayout(arch);
  const lastIdx = layout.wSizes.length - 1;
  const phiCols = layout.layerSizes[lastIdx];
  const phi: number[][] = [];
  const pVec: number[] = [];
  const cache: MLPForwardCache = { acts: [], preActs: [] };
  const probs = mlpForward(data.X, data.n, map.weights, arch, layout, cache, null);
  const lastActs = cache.acts[lastIdx];
  for (let i = 0; i < data.n; i++) {
    const row: number[] = [];
    for (let j = 0; j < phiCols; j++) row.push(lastActs[i * phiCols + j]);
    row.push(1.0);
    phi.push(row);
    const p = Math.max(Math.min(probs[i], 1 - BCE_EPSILON), BCE_EPSILON);
    pVec.push(p * (1 - p));
  }
  const dim = phiCols + 1;
  const H: number[][] = [];
  for (let a = 0; a < dim; a++) {
    const row = new Array<number>(dim).fill(0);
    H.push(row);
  }
  for (let a = 0; a < dim; a++) {
    for (let b = a; b < dim; b++) {
      let s = 0;
      for (let i = 0; i < data.n; i++) s += phi[i][a] * pVec[i] * phi[i][b];
      if (a === b) s += weightDecay;
      H[a][b] = s;
      H[b][a] = s;
    }
  }
  const L = choleskyFactor(H);
  const conditionNumber = estimateConditionNumberSPD(H, L);
  const lastWOff = layout.wOffsets[lastIdx];
  const lastBOff = layout.bOffsets[lastIdx];
  const sampleWeights = (S: number, seed: number): Float32Array[] => {
    const rng = mulberry32(seed);
    const out: Float32Array[] = [];
    for (let s = 0; s < S; s++) {
      const z = new Float32Array(dim);
      for (let i = 0; i < dim; i += 2) {
        const [g0, g1] = gaussianPair(rng);
        z[i] = g0;
        if (i + 1 < dim) z[i + 1] = g1;
      }
      const x = new Float32Array(dim);
      for (let i = dim - 1; i >= 0; i--) {
        let acc = z[i];
        for (let j = i + 1; j < dim; j++) acc -= L[j][i] * x[j];
        x[i] = acc / L[i][i];
      }
      const wsample = new Float32Array(map.weights);
      for (let j = 0; j < phiCols; j++) {
        wsample[lastWOff + j] = map.weights[lastWOff + j] + x[j];
      }
      wsample[lastBOff] = map.weights[lastBOff] + x[phiCols];
      out.push(wsample);
    }
    return out;
  };
  return { map, hessianCholesky: L, pEffective: dim, conditionNumber, sampleWeights };
}

/**
 * Last-layer Laplace approximation. Convenience wrapper that trains the MAP
 * and then calls lastLayerLaplaceFromMAP. Use the underlying primitive
 * directly when you want to recompute the Hessian with a varying weightDecay
 * without retraining.
 */
export function lastLayerLaplace(
  arch: MLPArchSpec,
  training: TrainingSpec,
  data: TrainingData,
): LaplaceResult {
  const map = mlpTrain(arch, training, data);
  return lastLayerLaplaceFromMAP(map, data, training.weightDecay);
}

/**
 * Full-Hessian Laplace via finite differences. O(p²) gradient calls — slow
 * (minutes for p=2241). For v2 viz, the "full-hessian" dropdown consumes
 * precomputed JSON from notebooks/bayesian-neural-networks/
 * precompute_laplace_grid.py rather than calling this in browser. This
 * implementation exists for completeness and small-architecture testing.
 */
export function laplaceApproxBNN(
  arch: MLPArchSpec,
  training: TrainingSpec,
  data: TrainingData,
  deltaStabilization: number = HESSIAN_STABILIZATION,
): LaplaceResult {
  // For v2 viz this is the "load precomputed" path. Throwing rather than
  // computing in-browser makes the precompute requirement explicit. For
  // small-architecture test cases (e.g., 1-hidden-layer MLP), use
  // lastLayerLaplace as an analytic stand-in.
  void arch;
  void training;
  void data;
  void deltaStabilization;
  throw new Error(
    'laplaceApproxBNN: full-Hessian Laplace is precomputed offline. ' +
      'Load from /sample-data/bayesian-neural-networks/laplace.json or use ' +
      'lastLayerLaplace() for in-browser interactive computation.',
  );
}

/**
 * KFAC Laplace: Kronecker-factored approximate curvature. Block-diagonal
 * approximation per layer where each block is A ⊗ G (input covariance ⊗
 * gradient covariance). Like full Laplace, ships from Python precompute for
 * v2 viz. Stub here for API surface.
 */
export function kfacLaplace(
  arch: MLPArchSpec,
  training: TrainingSpec,
  data: TrainingData,
): LaplaceResult {
  void arch;
  void training;
  void data;
  throw new Error(
    'kfacLaplace: KFAC blocks are precomputed offline. Load from ' +
      '/sample-data/bayesian-neural-networks/laplace.json#kfac.',
  );
}

// =============================================================================
// §6–7 PRIMITIVE: sgMCMCBNNTraining (in-browser)
// =============================================================================
// Full-batch SGLD and SGHMC chains on Two Moons. Per the §2.4 dimensions
// (n = 300, p = 2241), each step costs one mlpBackward call (~700k flops);
// at 500 post-burn samples + 200 burn-in, total ~700 steps × 700k = 490 MFLOP
// per chain (~2–4 s in pure JS). Acceptable for client:visible hydration.
//
// SGLD step (Welling & Teh 2011):
//   θ ← θ − (η/2)(n · ∇NLL_avg(θ) + λ · θ) + ξ,   ξ ~ N(0, η I)
// SGHMC step (Chen, Fox & Guestrin 2014):
//   r ← (1 − c) r − η (n · ∇NLL_avg(θ) + λ · θ) + ξ,   ξ ~ N(0, 2 η c I)
//   θ ← θ + η r
// where λ = training.weightDecay (≡ Gaussian prior precision 1/τ²).
//
// Minibatching is supported via spec.batchSize but defaults to full-batch
// (batchSize ≥ data.n). The minibatch gradient is rescaled by (n/b) to
// estimate the full-data NLL gradient unbiasedly.
// =============================================================================

export interface SGMCMCSpec {
  method: 'SGLD' | 'SGHMC';
  eta: number;
  batchSize: number;
  burnIn: number;
  samples: number;
  thin: number;
  friction?: number;
  warmStart?: Float32Array;
  seed: number;
}

export interface SGMCMCResult {
  weights: Float32Array[];
  /** Single-component trace of one weight across the full chain (post-burn-in). */
  weightTrace: number[];
  /** Autocorrelation function of weightTrace at lags 0..maxLag. */
  autocorrelation: number[];
}

function autocorrelation(x: number[], maxLag: number): number[] {
  const n = x.length;
  if (n === 0) return [];
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let var0 = 0;
  for (let i = 0; i < n; i++) var0 += (x[i] - mean) * (x[i] - mean);
  if (var0 === 0) return new Array(maxLag + 1).fill(0);
  const out: number[] = new Array(maxLag + 1).fill(0);
  for (let lag = 0; lag <= maxLag; lag++) {
    let s = 0;
    for (let i = 0; i < n - lag; i++) s += (x[i] - mean) * (x[i + lag] - mean);
    out[lag] = s / var0;
  }
  return out;
}

export function sgMCMCBNNTraining(
  arch: MLPArchSpec,
  training: TrainingSpec,
  data: TrainingData,
  spec: SGMCMCSpec,
): SGMCMCResult {
  const layout = mlpLayout(arch);
  const weights = spec.warmStart
    ? new Float32Array(spec.warmStart)
    : mlpInit(arch, spec.seed);
  const momentum = new Float32Array(layout.pDim);
  const rng = mulberry32(spec.seed * 1009 + 13);
  const lambda = training.weightDecay; // Gaussian prior precision 1/τ²
  const useFullBatch = spec.batchSize >= data.n;
  const totalSteps = spec.burnIn + spec.samples * spec.thin;
  const samples: Float32Array[] = [];
  const trace: number[] = [];
  const traceComponent = 0; // first weight, arbitrary monitor
  // Permutation buffer for minibatching
  const perm = new Int32Array(data.n);
  for (let i = 0; i < data.n; i++) perm[i] = i;
  let permPos = data.n;

  const Xb = useFullBatch ? data.X : new Float32Array(spec.batchSize * arch.inputDim);
  const yb = useFullBatch ? data.y : new Float32Array(spec.batchSize);

  // Reusable Gaussian sampler buffer (Box–Muller)
  let gBuf: number | null = null;
  const gauss = (): number => {
    if (gBuf !== null) {
      const v = gBuf;
      gBuf = null;
      return v;
    }
    const [a, b] = gaussianPair(rng);
    gBuf = b;
    return a;
  };

  for (let step = 0; step < totalSteps; step++) {
    let X: Float32Array;
    let y: Float32Array;
    let nUsed: number;
    if (useFullBatch) {
      X = data.X;
      y = data.y;
      nUsed = data.n;
    } else {
      // Refill the permutation when exhausted (Fisher–Yates)
      if (permPos + spec.batchSize > data.n) {
        for (let i = data.n - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          const tmp = perm[i];
          perm[i] = perm[j];
          perm[j] = tmp;
        }
        permPos = 0;
      }
      for (let i = 0; i < spec.batchSize; i++) {
        const src = perm[permPos + i];
        Xb[i * arch.inputDim + 0] = data.X[src * arch.inputDim + 0];
        Xb[i * arch.inputDim + 1] = data.X[src * arch.inputDim + 1];
        yb[i] = data.y[src];
      }
      permPos += spec.batchSize;
      X = Xb;
      y = yb;
      nUsed = spec.batchSize;
    }
    // gradAvg = (1/nUsed) ∇NLL_b ; we add prior term separately
    const { grad: gradAvg } = mlpBackward(X, y, nUsed, weights, arch, layout, 0, null);
    // Posterior negative-log gradient estimate at this minibatch:
    //   ∇U(θ) ≈ (n / nUsed) · b · gradAvg + λ θ = n · gradAvg + λ θ
    // (since gradAvg = (1/nUsed) ∇NLL_b ⇒ b · gradAvg = ∇NLL_b ⇒ scale by n/b → n · gradAvg)
    if (spec.method === 'SGLD') {
      const sqrtEta = Math.sqrt(spec.eta);
      for (let i = 0; i < layout.pDim; i++) {
        const gradFull = data.n * gradAvg[i] + lambda * weights[i];
        weights[i] -= 0.5 * spec.eta * gradFull;
        weights[i] += sqrtEta * gauss();
      }
    } else {
      const c = spec.friction ?? 0.05;
      const sqrtNoiseScale = Math.sqrt(2 * spec.eta * c);
      for (let i = 0; i < layout.pDim; i++) {
        const gradFull = data.n * gradAvg[i] + lambda * weights[i];
        momentum[i] = (1 - c) * momentum[i] - spec.eta * gradFull + sqrtNoiseScale * gauss();
      }
      for (let i = 0; i < layout.pDim; i++) {
        weights[i] += spec.eta * momentum[i];
      }
    }
    if (step >= spec.burnIn) {
      trace.push(weights[traceComponent]);
      const idxAfterBurn = step - spec.burnIn;
      if (idxAfterBurn % spec.thin === 0 && samples.length < spec.samples) {
        samples.push(new Float32Array(weights));
      }
    }
  }
  const acf = autocorrelation(trace, Math.min(50, Math.max(5, Math.floor(trace.length / 4))));
  return { weights: samples, weightTrace: trace, autocorrelation: acf };
}

// =============================================================================
// §8 PRIMITIVE: bnnCalibrationDiagnostic (notebook cell 18)
// =============================================================================

export interface CalibrationMetrics {
  ece: number;
  brier: number;
  nll: number;
  accuracy: number;
  reliabilityBins: { binConf: number; binAcc: number; binCount: number }[];
}

export function bnnCalibrationDiagnostic(
  testProbs: Float32Array,
  testLabels: Float32Array,
  nBins: number = 15,
): CalibrationMetrics {
  const n = testLabels.length;
  if (testProbs.length !== n) {
    throw new Error('bnnCalibrationDiagnostic: testProbs and testLabels length mismatch');
  }
  // ECE via uniform bins on max-prob
  const bins: { confSum: number; accSum: number; count: number }[] = [];
  for (let b = 0; b < nBins; b++) bins.push({ confSum: 0, accSum: 0, count: 0 });
  let brier = 0;
  let nll = 0;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.min(Math.max(testProbs[i], BCE_EPSILON), 1 - BCE_EPSILON);
    const y = testLabels[i];
    const conf = Math.max(p, 1 - p); // top-class confidence (binary)
    const pred = p >= 0.5 ? 1 : 0;
    const isCorrect = pred === y ? 1 : 0;
    correct += isCorrect;
    brier += (p - y) * (p - y);
    nll += -y * Math.log(p) - (1 - y) * Math.log(1 - p);
    let bIdx = Math.floor(conf * nBins);
    if (bIdx >= nBins) bIdx = nBins - 1;
    bins[bIdx].confSum += conf;
    bins[bIdx].accSum += isCorrect;
    bins[bIdx].count += 1;
  }
  let ece = 0;
  const reliabilityBins: { binConf: number; binAcc: number; binCount: number }[] = [];
  for (let b = 0; b < nBins; b++) {
    const c = bins[b].count;
    if (c === 0) {
      reliabilityBins.push({ binConf: (b + 0.5) / nBins, binAcc: 0, binCount: 0 });
      continue;
    }
    const binConf = bins[b].confSum / c;
    const binAcc = bins[b].accSum / c;
    ece += (c / n) * Math.abs(binConf - binAcc);
    reliabilityBins.push({ binConf, binAcc, binCount: c });
  }
  return {
    ece,
    brier: brier / n,
    nll: nll / n,
    accuracy: correct / n,
    reliabilityBins,
  };
}

// =============================================================================
// §9 PRIMITIVE: nngpArcCosineKernel (notebook cell 20)
// =============================================================================
// Cho & Saul 2009 closed-form arc-cosine kernel of order 1 (ReLU). For a
// single-hidden-layer MLP with weights ~ N(0, σ_w² / inDim) and bias
// ~ N(0, σ_b²), the infinite-width prior covariance is:
//
//   K_1(x, x') = (σ_w² / 2π) ||x|| ||x'|| (sin θ + (π − θ) cos θ) + σ_b²
//
// where θ = arccos(<x_hat, x_hat'>) is the angle between unit-normalized inputs.
// =============================================================================

export function nngpArcCosineKernel(
  X: number[][],
  Xprime: number[][],
  sigmaW2: number,
  sigmaB2: number,
): number[][] {
  const n = X.length;
  const m = Xprime.length;
  const K: number[][] = [];
  // Precompute norms
  const normsX = X.map((row) => Math.sqrt(row.reduce((s, v) => s + v * v, 0)));
  const normsXp = Xprime.map((row) => Math.sqrt(row.reduce((s, v) => s + v * v, 0)));
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(m);
    for (let j = 0; j < m; j++) {
      const ni = normsX[i];
      const nj = normsXp[j];
      let dot = 0;
      for (let d = 0; d < X[i].length; d++) dot += X[i][d] * Xprime[j][d];
      const cosTheta = ni * nj > 0 ? Math.min(Math.max(dot / (ni * nj), -1), 1) : 0;
      const theta = Math.acos(cosTheta);
      const term = Math.sin(theta) + (Math.PI - theta) * Math.cos(theta);
      row[j] = (sigmaW2 / (2 * Math.PI)) * ni * nj * term + sigmaB2;
    }
    K.push(row);
  }
  return K;
}

// =============================================================================
// DATA SYNTHESIS: makeMoonsData (§§1, 2, 4, 5 default training set)
// =============================================================================
// Two-moons dataset with Gaussian noise. The geometry matches
// sklearn.datasets.make_moons:
//   top moon (y=0):    (cos t, sin t)            for t ∈ [0, π]
//   bottom moon (y=1): (1 − cos t, 0.5 − sin t)  for t ∈ [0, π]
//   per-coordinate noise: N(0, noise²)
//
// Two intentional simplifications relative to sklearn:
//   - angles are evenly spaced along [0, π] rather than uniformly random.
//     This makes seed-reproducible verification cheaper and gives the noise=0
//     check (samples lie exactly on the parametric curves) a clean equality.
//   - samples come back in deterministic class order (n/2 of class 0 followed
//     by n/2 of class 1). sklearn shuffles. Callers that need shuffled data
//     can shuffle in place; viz components iterate all samples per panel so
//     ordering is irrelevant.
// =============================================================================

export function makeMoonsData(n: number, noise: number, seed: number): TrainingData {
  const rng = mulberry32(seed);
  const X = new Float32Array(n * 2);
  const y = new Float32Array(n);
  const half = Math.floor(n / 2);
  let buf: [number, number] | null = null;
  const gauss = (): number => {
    if (buf) {
      const v = buf[1];
      buf = null;
      return v;
    }
    const [a, b] = gaussianPair(rng);
    buf = [a, b];
    return a;
  };
  for (let i = 0; i < n; i++) {
    const isUpper = i < half;
    const idxInMoon = isUpper ? i : i - half;
    const denom = isUpper ? Math.max(half - 1, 1) : Math.max(n - half - 1, 1);
    const t = (idxInMoon / denom) * Math.PI;
    const baseX = isUpper ? Math.cos(t) : 1 - Math.cos(t);
    const baseY = isUpper ? Math.sin(t) : 0.5 - Math.sin(t);
    X[i * 2 + 0] = baseX + noise * gauss();
    X[i * 2 + 1] = baseY + noise * gauss();
    y[i] = isUpper ? 0 : 1;
  }
  return { X, y, n };
}

// =============================================================================
// PCA: pcaProject2D (§2 loss-landscape viz)
// =============================================================================
// Top-2 PC projection of K weight vectors of dimension p. Uses the K×K Gram
// matrix Xc Xc^T rather than the p×p covariance, since K ≪ p (typical: K=30,
// p=2241). Power iteration with deflation; deterministic via a seeded RNG.
//
// Returned scores[i] = (√λ₁·v₁[i], √λ₂·v₂[i]) — the projection of sample i
// onto the top-2 PCs in input space. PC vectors in p-space are not returned;
// callers needing them can compute pc_k = (Xc^T v_k) / √λ_k.
// =============================================================================

export interface PCAProject2DResult {
  /** Mean vector subtracted from each input row, length p. */
  mean: Float32Array;
  /** Top two eigenvalues of the K×K Gram matrix (descending). */
  eigenvalues: [number, number];
  /** 2D scores: scores[i] = projection of X[i] onto (PC1, PC2). */
  scores: Array<[number, number]>;
}

function powerIterationTopEigSym(
  M: number[][],
  rng: () => number,
  maxIters = 200,
  tol = 1e-9,
): { eigenvalue: number; eigenvector: number[] } {
  const K = M.length;
  const v = new Array<number>(K);
  const Mv = new Array<number>(K);
  for (let i = 0; i < K; i++) v[i] = rng() - 0.5;
  let norm = 0;
  for (let i = 0; i < K; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < K; i++) v[i] /= norm;
  let lambda = 0;
  for (let iter = 0; iter < maxIters; iter++) {
    // Mv ← M v  (in-place into the hoisted buffer)
    for (let i = 0; i < K; i++) {
      let s = 0;
      const Mi = M[i];
      for (let j = 0; j < K; j++) s += Mi[j] * v[j];
      Mv[i] = s;
    }
    let mvNorm = 0;
    for (let i = 0; i < K; i++) mvNorm += Mv[i] * Mv[i];
    mvNorm = Math.sqrt(mvNorm);
    if (mvNorm === 0) break;
    // Rayleigh quotient λ = vᵀ M v ; with v normalized, λ ≈ ‖Mv‖ · sign(vᵀ Mv).
    // Compute it from the unscaled product to avoid an extra pass.
    let lambdaNew = 0;
    for (let i = 0; i < K; i++) lambdaNew += (Mv[i] / mvNorm) * Mv[i];
    // v ← Mv / ‖Mv‖  (in-place into v)
    for (let i = 0; i < K; i++) v[i] = Mv[i] / mvNorm;
    if (Math.abs(lambdaNew - lambda) < tol * Math.max(1, Math.abs(lambdaNew))) {
      lambda = lambdaNew;
      break;
    }
    lambda = lambdaNew;
  }
  return { eigenvalue: lambda, eigenvector: v };
}

export function pcaProject2D(X: Float32Array[], seed = 1): PCAProject2DResult {
  const K = X.length;
  if (K < 2) throw new Error('pcaProject2D requires at least 2 input vectors');
  const p = X[0].length;
  const mean = new Float32Array(p);
  for (let k = 0; k < K; k++) for (let i = 0; i < p; i++) mean[i] += X[k][i];
  for (let i = 0; i < p; i++) mean[i] /= K;
  const Xc: Float32Array[] = X.map((x) => {
    const out = new Float32Array(p);
    for (let i = 0; i < p; i++) out[i] = x[i] - mean[i];
    return out;
  });
  const G: number[][] = [];
  for (let i = 0; i < K; i++) {
    const row = new Array<number>(K).fill(0);
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let d = 0; d < p; d++) s += Xc[i][d] * Xc[j][d];
      row[j] = s;
    }
    G.push(row);
  }
  for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) G[i][j] = G[j][i];
  const rng = mulberry32(seed);
  const eig1 = powerIterationTopEigSym(G, rng);
  const lam1 = Math.max(eig1.eigenvalue, 0);
  for (let i = 0; i < K; i++)
    for (let j = 0; j < K; j++) G[i][j] -= lam1 * eig1.eigenvector[i] * eig1.eigenvector[j];
  const eig2 = powerIterationTopEigSym(G, rng);
  const lam2 = Math.max(eig2.eigenvalue, 0);
  const sqrtL1 = Math.sqrt(lam1);
  const sqrtL2 = Math.sqrt(lam2);
  const scores: Array<[number, number]> = [];
  for (let i = 0; i < K; i++) {
    scores.push([sqrtL1 * eig1.eigenvector[i], sqrtL2 * eig2.eigenvector[i]]);
  }
  return { mean, eigenvalues: [lam1, lam2], scores };
}

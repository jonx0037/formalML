// =============================================================================
// meta-learning.ts
//
// Shared math/types/sampler/primitives module for the T5 meta-learning topic.
// Mirrors notebooks/meta-learning/01_meta_learning.ipynb in TypeScript, focusing
// on closed-form primitives that can be verified against the notebook's printed
// numerical outputs. The notebook's PyTorch network training is too expensive
// to replay in-browser at slider-commit speed; the heavy figures (05, 08, 10,
// 13, 15, 16, 17) consume precomputed JSON instead. This module covers:
//
//   1. PRNG and task samplers (cells 3, 7, 8): mulberry32 → sinusoid/GP/protonet
//      tasks. Used by TaskFamilyExplorer to live-resample.
//
//   2. MAML / FOMAML / Reptile primitives on quadratic targets (cells 12-18,
//      19-26): closed-form gradient and trajectory expressions verifying the
//      §2.6 and §3.2 formulas. The 1D Gaussian case of Theorem 6.1 (cell 49)
//      is also implemented exactly.
//
//   3. Neural Process forward pass (cell 34): pure functional CNP encoder /
//      decoder evaluation given fixed weights, plus the latent reparametrization
//      sample formula. Sufficient to render Figure-09-style predictives from
//      a precomputed weight payload.
//
//   4. Prototypical Networks (cell 44): class-mean prototypes and softmax-over-
//      squared-distances predictive. The Bregman-mean optimality check is the
//      core verification.
//
//   5. Amit-Meir bound (cell 57): the closed-form (7.2) bound with the across-
//      task and within-task decomposition.
//
//   6. FMO convergence constants (Lemma 8.6): smoothness constant L_F (8.4).
//
// Conventions match riemann-hmc.ts / bayesian-ml.ts:
//   - vectors: number[]; matrices: number[][] row-major
//   - RNGs: () => number uniform in [0, 1)
//   - pure functions, no React / D3 / DOM
//   - seed: 20260518 (matches notebook SEED)
//
// Brief: docs/plans/formalml-meta-learning-handoff-brief.md
// Notebook: notebooks/meta-learning/01_meta_learning.ipynb
// =============================================================================

import { mulberry32, gaussianPair } from './bayesian-ml';

export { mulberry32, gaussianPair };

// -----------------------------------------------------------------------------
// Notebook seed (kept here so verification suite can reproduce numerically).
// -----------------------------------------------------------------------------
export const META_SEED = 20260518;

// Notebook palette (cell 1).
export const META_PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e'];

// -----------------------------------------------------------------------------
// Standard normal draw via gaussianPair (one normal per call, throwing away
// the second). Convenient when consumers don't want to manage the pair cache.
// -----------------------------------------------------------------------------
export function drawNormal(rng: () => number): number {
  return gaussianPair(rng)[0];
}

// =============================================================================
// 1. Task samplers (cells 3, 7, 8)
// =============================================================================

export interface SinusoidTask {
  A: number;
  phi: number;
  xSupport: number[];
  ySupport: number[];
  xDense: number[];
  yDense: number[];
}

/**
 * Sample one sinusoidal task: y = A sin(x + phi), with A ~ U[0.1, 5], phi ~ U[0, 2π].
 * K support points uniform in xRange; n_dense linspace points for the truth curve.
 *
 * Matches cell 3 `sample_sinusoid_task`.
 */
export function sampleSinusoidTask(
  rng: () => number,
  options: {
    K?: number;
    xRange?: [number, number];
    nDense?: number;
    aRange?: [number, number];
    phiRange?: [number, number];
  } = {},
): SinusoidTask {
  const K = options.K ?? 5;
  const [xLo, xHi] = options.xRange ?? [-5, 5];
  const nDense = options.nDense ?? 200;
  const [aLo, aHi] = options.aRange ?? [0.1, 5.0];
  const [phiLo, phiHi] = options.phiRange ?? [0, 2 * Math.PI];

  const A = aLo + (aHi - aLo) * rng();
  const phi = phiLo + (phiHi - phiLo) * rng();
  const xSupport: number[] = new Array(K);
  const ySupport: number[] = new Array(K);
  for (let i = 0; i < K; i++) {
    xSupport[i] = xLo + (xHi - xLo) * rng();
    ySupport[i] = A * Math.sin(xSupport[i] + phi);
  }
  const xDense: number[] = new Array(nDense);
  const yDense: number[] = new Array(nDense);
  const step = (xHi - xLo) / (nDense - 1);
  for (let i = 0; i < nDense; i++) {
    xDense[i] = xLo + i * step;
    yDense[i] = A * Math.sin(xDense[i] + phi);
  }
  return { A, phi, xSupport, ySupport, xDense, yDense };
}

/**
 * Evaluate the RBF kernel matrix K(x1, x2) with lengthscale ell. Returns an
 * n1 × n2 matrix. Matches cell 7's `rbf_kernel`.
 *
 * When called with `x1 === x2` (same array reference — the Gram-matrix case in
 * `sampleGpTask` and elsewhere), computes only the strict lower triangle and
 * mirrors to the upper, with the diagonal pinned to K[i][i] = exp(0) = 1.
 * Halves the Math.exp count on n × n symmetric calls. The reference-equality
 * fast path is a no-op for the general n1 × n2 case.
 */
export function rbfKernel(x1: number[], x2: number[], ell: number): number[][] {
  const n1 = x1.length;
  const inv2ell2 = 1.0 / (2.0 * ell * ell);
  const K: number[][] = new Array(n1);

  if (x1 === x2) {
    for (let i = 0; i < n1; i++) K[i] = new Array(n1);
    for (let i = 0; i < n1; i++) {
      K[i][i] = 1.0;
      for (let j = 0; j < i; j++) {
        const d = x1[i] - x1[j];
        const val = Math.exp(-d * d * inv2ell2);
        K[i][j] = val;
        K[j][i] = val;
      }
    }
    return K;
  }

  const n2 = x2.length;
  for (let i = 0; i < n1; i++) {
    const row = new Array(n2);
    for (let j = 0; j < n2; j++) {
      const d = x1[i] - x2[j];
      row[j] = Math.exp(-d * d * inv2ell2);
    }
    K[i] = row;
  }
  return K;
}

/**
 * Cholesky factorization (lower-triangular L such that L @ L.T = A). Matches
 * the numpy.linalg.cholesky semantics used in cell 7. Throws on non-PD input.
 */
export function choleskyLower(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = new Array(n);
  for (let i = 0; i < n; i++) L[i] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum <= 0) throw new Error(`choleskyLower: non-PD at i=${i}, sum=${sum}`);
        L[i][i] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  return L;
}

/**
 * Sample a GP-prior function value at x given lengthscale, returning the
 * dense linspace evaluation. Matches cell 7's `sample_gp_task`. Adds a
 * small jitter to the Cholesky for numerical stability.
 */
export function sampleGpTask(
  rng: () => number,
  ell: number,
  options: { xRange?: [number, number]; nDense?: number; jitter?: number } = {},
): { x: number[]; f: number[] } {
  const [xLo, xHi] = options.xRange ?? [-3, 3];
  const nDense = options.nDense ?? 200;
  const jitter = options.jitter ?? 1e-6;
  const x: number[] = new Array(nDense);
  const step = (xHi - xLo) / (nDense - 1);
  for (let i = 0; i < nDense; i++) x[i] = xLo + i * step;
  const K = rbfKernel(x, x, ell);
  for (let i = 0; i < nDense; i++) K[i][i] += jitter;
  const L = choleskyLower(K);
  const z: number[] = new Array(nDense);
  for (let i = 0; i < nDense; i++) z[i] = drawNormal(rng);
  const f: number[] = new Array(nDense).fill(0);
  for (let i = 0; i < nDense; i++) {
    let acc = 0;
    for (let j = 0; j <= i; j++) acc += L[i][j] * z[j];
    f[i] = acc;
  }
  return { x, f };
}

export interface ProtoNetTask {
  Xs: number[][]; // support points (n_classes * k_shot) × 2
  ys: number[]; // support labels (n_classes * k_shot)
  Xq: number[][]; // query points (n_classes * m_query) × 2
  yq: number[]; // query labels
  means: number[][]; // per-class means (n_classes × 2)
}

/**
 * Sample a 2D K-way N-shot prototypical task. Class means are placed on a
 * disk of radius `radius` with a randomized rotational offset; each class
 * gets k_shot support and m_query query points, isotropic Gaussian noise σ.
 * Matches cell 8's `sample_protonet_task`.
 */
export function sampleProtoNetTask(
  rng: () => number,
  options: {
    nClasses?: number;
    kShot?: number;
    mQuery?: number;
    sigma?: number;
    radius?: number;
  } = {},
): ProtoNetTask {
  const nClasses = options.nClasses ?? 5;
  const kShot = options.kShot ?? 5;
  const mQuery = options.mQuery ?? 15;
  const sigma = options.sigma ?? 0.35;
  const radius = options.radius ?? 2.5;
  const base = 2 * Math.PI * rng();
  const means: number[][] = new Array(nClasses);
  for (let c = 0; c < nClasses; c++) {
    const ang = base + (2 * Math.PI * c) / nClasses;
    means[c] = [radius * Math.cos(ang), radius * Math.sin(ang)];
  }
  const Xs: number[][] = [];
  const ys: number[] = [];
  const Xq: number[][] = [];
  const yq: number[] = [];
  for (let c = 0; c < nClasses; c++) {
    const [mx, my] = means[c];
    for (let i = 0; i < kShot; i++) {
      Xs.push([mx + sigma * drawNormal(rng), my + sigma * drawNormal(rng)]);
      ys.push(c);
    }
    for (let j = 0; j < mQuery; j++) {
      Xq.push([mx + sigma * drawNormal(rng), my + sigma * drawNormal(rng)]);
      yq.push(c);
    }
  }
  return { Xs, ys, Xq, yq, means };
}

// =============================================================================
// 2. MAML / FOMAML / Reptile primitives on quadratic targets
// =============================================================================

/**
 * Theorem 6.1 (1D Gaussian case): N-step GD iterate on a quadratic
 * −log p(D|θ) = (λ/2)(θ − μ_MLE)² + const, starting from θ₀, step size α.
 *
 * Returns θ^{(N)} = μ_MLE + (1 − αλ)^N (θ₀ − μ_MLE).
 *
 * Used to verify the MAML-as-MAP statement and to render Figure-6-style
 * trajectories in the 1D pedagogical viz.
 */
export function quadraticGdIterate(
  theta0: number,
  muMle: number,
  lambda: number,
  alpha: number,
  N: number,
): number {
  const decay = Math.pow(1 - alpha * lambda, N);
  return muMle + decay * (theta0 - muMle);
}

/**
 * The MAP estimate (6.4) under a 1D Gaussian-quadratic likelihood × Gaussian
 * hyperprior θ ~ N(θ₀, σ_p²): θ_MAP = (λ μ_MLE + θ₀/σ_p²) / (λ + 1/σ_p²).
 */
export function quadraticMap(
  theta0: number,
  muMle: number,
  lambda: number,
  sigmaP: number,
): number {
  const tauP = 1 / (sigmaP * sigmaP);
  return (lambda * muMle + tauP * theta0) / (lambda + tauP);
}

/**
 * The implicit hyperprior precision (6.5) under which N-step GD equals MAP:
 *   σ_p^{-2} = λ (1−αλ)^N / (1 − (1−αλ)^N).
 *
 * Returns +Inf at N=0 (no adaptation = infinite precision = MAP at θ₀).
 */
export function implicitPriorPrecision(lambda: number, alpha: number, N: number): number {
  if (N <= 0) return Infinity;
  const decay = Math.pow(1 - alpha * lambda, N);
  if (decay <= 1e-15) return 0;
  return (lambda * decay) / (1 - decay);
}

/**
 * The MAML outer-loop gradient (2.6) at θ₀ = 0 for a 1D quadratic support and
 * a 1D quadratic query: returns the N=1 case
 *   ∇_θ₀ L^Q(θ^(1)) = (1 − α λ_S)(θ^(1) − μ_Q) λ_Q
 * with θ^(1) = θ₀ − α λ_S (θ₀ − μ_S).
 *
 * Lets us verify the (I − αH) corrector structure.
 */
export function mamlOuterGradient1D(
  theta0: number,
  muS: number,
  muQ: number,
  lambdaS: number,
  lambdaQ: number,
  alpha: number,
): number {
  const theta1 = theta0 - alpha * lambdaS * (theta0 - muS);
  const innerJacobian = 1 - alpha * lambdaS;
  const queryGrad = lambdaQ * (theta1 - muQ);
  return innerJacobian * queryGrad;
}

/**
 * The FOMAML meta-gradient (3.1) at θ₀ for the same 1D quadratic setup:
 * just the query-loss gradient at the adapted point, dropping the (I − αH)
 * Jacobian.
 */
export function fomamlOuterGradient1D(
  theta0: number,
  muS: number,
  muQ: number,
  lambdaS: number,
  lambdaQ: number,
  alpha: number,
): number {
  const theta1 = theta0 - alpha * lambdaS * (theta0 - muS);
  return lambdaQ * (theta1 - muQ);
}

/**
 * The FOMAML bias (3.2) at θ₀ in 1D: −α λ_S × λ_Q × (θ^(1) − μ_Q).
 */
export function fomamlBias1D(
  theta0: number,
  muS: number,
  muQ: number,
  lambdaS: number,
  lambdaQ: number,
  alpha: number,
): number {
  const maml = mamlOuterGradient1D(theta0, muS, muQ, lambdaS, lambdaQ, alpha);
  const fomaml = fomamlOuterGradient1D(theta0, muS, muQ, lambdaS, lambdaQ, alpha);
  return maml - fomaml;
}

/**
 * Reptile direction-of-travel (3.3) at θ₀: returns θ^(N) − θ₀ for N-step GD
 * on a 1D quadratic. With the convention from Nichol et al. that Reptile
 * runs SGD on the combined support+query, we expose the per-call inner-loss
 * curvature λ and target μ.
 */
export function reptileDirection1D(
  theta0: number,
  mu: number,
  lambda: number,
  alpha: number,
  N: number,
): number {
  const thetaN = quadraticGdIterate(theta0, mu, lambda, alpha, N);
  return thetaN - theta0;
}

/**
 * Reptile leading-order implicit-objective constant (3.5): the coefficient
 * α(N−1)/2 on the gradient-norm-squared regularizer.
 *
 * Returns 0 at N=1 (Reptile reduces to plain SGD on expected support loss).
 */
export function reptileImplicitCoeff(alpha: number, N: number): number {
  if (N <= 1) return 0;
  return (alpha * (N - 1)) / 2;
}

// =============================================================================
// 3. Conditional Neural Process forward pass (cell 34)
// =============================================================================

/**
 * Apply a stack of dense layers: x → relu(W₁ x + b₁) → relu(W₂ x + b₂) → ...
 * The last layer does NOT apply ReLU (used for both encoder and decoder heads).
 * weights[i] is [W_i, b_i] with W_i a (out × in) row-major matrix.
 *
 * Pure forward — no autograd, no batching. Used to evaluate a precomputed CNP
 * payload at viz time.
 */
export function mlpForward(
  x: number[],
  weights: Array<{ W: number[][]; b: number[] }>,
): number[] {
  let h = x.slice();
  for (let layer = 0; layer < weights.length; layer++) {
    const { W, b } = weights[layer];
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let acc = b[i];
      const row = W[i];
      for (let j = 0; j < row.length; j++) acc += row[j] * h[j];
      out[i] = layer === weights.length - 1 ? acc : Math.max(0, acc);
    }
    h = out;
  }
  return h;
}

/**
 * CNP encoder forward: per-point encode (x_i, y_i) → h_i, then mean-pool over
 * the context set. Returns the aggregate representation r.
 *
 * Matches cell 34's `NPEncoder.forward`.
 */
export function cnpEncoderForward(
  xContext: number[][], // n_ctx × x_dim
  yContext: number[][], // n_ctx × y_dim
  encoderWeights: Array<{ W: number[][]; b: number[] }>,
): number[] {
  const nCtx = xContext.length;
  if (nCtx === 0) throw new Error('cnpEncoderForward: empty context set');
  const rDim = encoderWeights[encoderWeights.length - 1].W.length;
  const r: number[] = new Array(rDim).fill(0);
  for (let i = 0; i < nCtx; i++) {
    const xy = xContext[i].concat(yContext[i]);
    const h = mlpForward(xy, encoderWeights);
    for (let d = 0; d < rDim; d++) r[d] += h[d];
  }
  for (let d = 0; d < rDim; d++) r[d] /= nCtx;
  return r;
}

/**
 * CNP decoder forward at one target: concat (x_target, r), pass through the
 * decoder MLP, split last layer into (mu, log_sigma), clamp log_sigma to
 * [−5, 2] for numerical stability (matches cell 34).
 *
 * Returns { mu, logSigma } each of length y_dim.
 */
export function cnpDecoderForward(
  xTarget: number[],
  r: number[],
  decoderWeights: Array<{ W: number[][]; b: number[] }>,
): { mu: number[]; logSigma: number[] } {
  const yDim = decoderWeights[decoderWeights.length - 1].W.length / 2;
  const xr = xTarget.concat(r);
  const out = mlpForward(xr, decoderWeights);
  const mu = out.slice(0, yDim);
  const logSigma = out.slice(yDim, 2 * yDim).map((v) => Math.max(-5, Math.min(2, v)));
  return { mu, logSigma };
}

/**
 * Latent NP reparametrization sample: given encoder Gaussian parameters
 * (mu_c, log_sigma_c) over latent z, draw z = mu_c + exp(log_sigma_c) * ε.
 * Used to render the §4.5 multi-sample posterior predictive band.
 */
export function latentNpSample(
  muC: number[],
  logSigmaC: number[],
  rng: () => number,
): number[] {
  const z = new Array(muC.length);
  for (let i = 0; i < muC.length; i++) {
    z[i] = muC[i] + Math.exp(logSigmaC[i]) * drawNormal(rng);
  }
  return z;
}

/**
 * Diagonal Gaussian KL: KL( N(mu1, diag σ1²) || N(mu2, diag σ2²) ).
 *
 * Used in the Latent NP -ELBO of §4.3 and the Amit-Meir Gaussian-Gaussian
 * KL formula of §7.5. Closed-form:
 *   KL = ½ Σ_i [ (σ1_i² + (mu1_i − mu2_i)²) / σ2_i² − 1 + log(σ2_i²/σ1_i²) ]
 */
export function diagonalGaussianKL(
  mu1: number[],
  sigma1: number[],
  mu2: number[],
  sigma2: number[],
): number {
  if (mu1.length !== mu2.length || mu1.length !== sigma1.length || mu1.length !== sigma2.length) {
    throw new Error('diagonalGaussianKL: shape mismatch');
  }
  let kl = 0;
  for (let i = 0; i < mu1.length; i++) {
    const s1Sq = sigma1[i] * sigma1[i];
    const s2Sq = sigma2[i] * sigma2[i];
    const d = mu1[i] - mu2[i];
    kl += 0.5 * ((s1Sq + d * d) / s2Sq - 1 + Math.log(s2Sq / s1Sq));
  }
  return kl;
}

// =============================================================================
// 4. Prototypical Networks (cells 44, 46)
// =============================================================================

/**
 * Compute class-mean prototypes from labelled support embeddings.
 *
 * supportEmbeddings: n × emb_dim
 * supportLabels: n
 * nClasses: K
 *
 * Returns K × emb_dim matrix of prototypes; per-class missing classes raise.
 * Matches cell 44's prototype computation.
 */
export function protonetClassPrototypes(
  supportEmbeddings: number[][],
  supportLabels: number[],
  nClasses: number,
): number[][] {
  const n = supportEmbeddings.length;
  if (n === 0) throw new Error('protonetClassPrototypes: empty support set');
  if (supportLabels.length !== n) throw new Error('protonetClassPrototypes: label/embedding count mismatch');
  const embDim = supportEmbeddings[0].length;
  const prototypes: number[][] = new Array(nClasses);
  const counts: number[] = new Array(nClasses).fill(0);
  for (let k = 0; k < nClasses; k++) prototypes[k] = new Array(embDim).fill(0);
  for (let i = 0; i < n; i++) {
    const lbl = supportLabels[i];
    if (lbl < 0 || lbl >= nClasses) throw new Error(`protonetClassPrototypes: out-of-range label ${lbl}`);
    counts[lbl]++;
    for (let d = 0; d < embDim; d++) prototypes[lbl][d] += supportEmbeddings[i][d];
  }
  for (let k = 0; k < nClasses; k++) {
    if (counts[k] === 0) throw new Error(`protonetClassPrototypes: class ${k} has no support`);
    for (let d = 0; d < embDim; d++) prototypes[k][d] /= counts[k];
  }
  return prototypes;
}

/**
 * Squared Euclidean distance between two embedding vectors.
 */
export function squaredDistance(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

/**
 * Softmax over negative squared distances: (5.3).
 *
 * Returns the per-class probability vector of length K.
 */
export function nearestPrototypeSoftmax(
  embedding: number[],
  prototypes: number[][],
): number[] {
  const K = prototypes.length;
  const negD: number[] = new Array(K);
  for (let k = 0; k < K; k++) negD[k] = -squaredDistance(embedding, prototypes[k]);
  let mx = negD[0];
  for (let k = 1; k < K; k++) if (negD[k] > mx) mx = negD[k];
  let sum = 0;
  const probs: number[] = new Array(K);
  for (let k = 0; k < K; k++) {
    probs[k] = Math.exp(negD[k] - mx);
    sum += probs[k];
  }
  for (let k = 0; k < K; k++) probs[k] /= sum;
  return probs;
}

// =============================================================================
// 5. Amit-Meir meta PAC-Bayes bound (cell 57, Theorem 7.1)
// =============================================================================

export interface AmitMeirComponents {
  across: number; // sqrt((KL_meta + log(4T/δ)) / (2(T-1)))
  within: number; // sqrt((KL_within + log(4Tn/δ)) / (2(n-1)))
  rHat: number; // empirical meta-risk
  total: number; // rHat + across + within
}

/**
 * The Amit-Meir bound (7.2) decomposed into its three pieces. Inputs:
 *   T       — number of observed tasks
 *   n       — within-task sample count
 *   klMeta  — KL(Q || P_0) at the meta level
 *   klWithin — averaged within-task KL
 *   delta   — confidence parameter
 *   rHat    — empirical meta-risk (optional, default 0.10 to match cell 57)
 *
 * Matches `amit_meir_bound` in cell 57. The √((·)/2(T-1)) and √((·)/2(n-1))
 * denominators are the Maurer-Pontil refinements (pac-bayes-bounds §7.1).
 */
export function amitMeirBound(
  T: number,
  n: number,
  klMeta: number,
  klWithin: number,
  delta: number,
  rHat = 0.1,
): AmitMeirComponents {
  if (T < 2) throw new Error(`amitMeirBound: need T >= 2, got ${T}`);
  if (n < 2) throw new Error(`amitMeirBound: need n >= 2, got ${n}`);
  const across = Math.sqrt((klMeta + Math.log((4 * T) / delta)) / (2 * (T - 1)));
  const within = Math.sqrt((klWithin + Math.log((4 * T * n) / delta)) / (2 * (n - 1)));
  return { across, within, rHat, total: rHat + across + within };
}

// =============================================================================
// 6. FMO convergence constants (Lemma 8.6)
// =============================================================================

/**
 * Meta-objective smoothness constant (8.4):
 *   L_F = (1 + αL)² L + α ρ G_Q
 *
 * Matches Lemma 1 in §8.3. Pure formula evaluation; no notebook reference cell
 * since the §8 viz uses this directly.
 */
export function fmoSmoothnessConstant(
  L: number,
  rho: number,
  alpha: number,
  G_Q: number,
): number {
  return (1 + alpha * L) * (1 + alpha * L) * L + alpha * rho * G_Q;
}

/**
 * Meta-gradient variance bound (8.5):
 *   E[||g̃_k − ∇F||²] ≤ σ_g² + α² σ_H² G_Q² + α² σ_g² σ_H²
 */
export function fmoVarianceBound(
  sigmaG: number,
  sigmaH: number,
  alpha: number,
  G_Q: number,
): number {
  const a2 = alpha * alpha;
  const sg2 = sigmaG * sigmaG;
  const sH2 = sigmaH * sigmaH;
  return sg2 + a2 * sH2 * G_Q * G_Q + a2 * sg2 * sH2;
}

/**
 * The C / √K convergence-rate constant for the meta-loss running minimum
 * envelope (cell 64):
 *   envelope = running_min[-1] * sqrt(K_max) / sqrt(K).
 *
 * Used by Figure 13 to overlay the theoretical envelope on the empirical
 * meta-loss curve.
 */
export function convergenceEnvelopeConstant(
  runningMinAtKmax: number,
  Kmax: number,
): number {
  return runningMinAtKmax * Math.sqrt(Kmax);
}

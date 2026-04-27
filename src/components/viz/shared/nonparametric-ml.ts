// =============================================================================
// nonparametric-ml.ts
//
// Shared math module for the T4 Nonparametric & Distribution-Free track.
// First consumer: ConformalPredictionExplorer / CQRExplorer / APSForClassification
// / ExchangeabilityBreakdown viz components in the conformal-prediction topic.
//
// All numerical algorithms are direct translations of the verified Python
// notebook at notebooks/conformal-prediction/01_conformal_prediction.ipynb.
// Where the Python uses np.random / sklearn, this module substitutes a seeded
// Mulberry32 PRNG, Box-Muller Gaussian sampling, and small hand-rolled solvers.
// Numerical agreement is verified by src/components/viz/shared/__tests__/
// verify-nonparametric-ml.ts against the notebook's printed outputs.
// =============================================================================

// -----------------------------------------------------------------------------
// Seeded PRNG and Gaussian sampler
// -----------------------------------------------------------------------------

/**
 * Mulberry32 — small, fast, deterministic PRNG. Produces uniform [0, 1) doubles.
 * Returns a closure so each call advances the same internal state.
 */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Box-Muller Gaussian sampler. Returns standard-normal draws given a uniform RNG.
 * Caches the second of each Box-Muller pair for the next call.
 */
export function gaussianSampler(rng: () => number): () => number {
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = rng();
    if (u1 < 1e-300) u1 = 1e-300; // avoid log(0)
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

// -----------------------------------------------------------------------------
// Synthetic data generators (match notebook §1, §6, §7, §8)
// -----------------------------------------------------------------------------

/**
 * Heteroscedastic 1D regression. Matches notebook synth_heteroscedastic:
 *   x ~ Uniform(-2, 2);  sigma(x) = 0.3 + 0.6 * |x|
 *   y = 0.5 * x + sin(1.5 * x) + N(0, sigma(x))
 */
export function synthHeteroscedastic(
  n: number,
  rng: () => number,
): { x: Float64Array; y: Float64Array } {
  const gauss = gaussianSampler(rng);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const xi = -2 + 4 * rng();
    const sigma = 0.3 + 0.6 * Math.abs(xi);
    x[i] = xi;
    y[i] = 0.5 * xi + Math.sin(1.5 * xi) + sigma * gauss();
  }
  return { x, y };
}

/**
 * Three Gaussian blobs in 2D for classification demos. Matches notebook synth_3class:
 *   centers = [(-1.5, -0.5), (1.5, -0.5), (0, 1.5)]
 *   each class has n // 3 points; X is row-major shape (3 * (n // 3), 2).
 * Returns y as Int32Array of class labels {0, 1, 2}.
 */
export function synth3Class(
  n: number,
  rng: () => number,
  sigma: number = 1.0,
): { X: Float64Array; y: Int32Array } {
  const gauss = gaussianSampler(rng);
  const centers: ReadonlyArray<readonly [number, number]> = [
    [-1.5, -0.5],
    [1.5, -0.5],
    [0, 1.5],
  ];
  const nPer = Math.floor(n / 3);
  const total = nPer * 3;
  const X = new Float64Array(total * 2);
  const y = new Int32Array(total);
  for (let k = 0; k < 3; k++) {
    const [cx, cy] = centers[k];
    for (let i = 0; i < nPer; i++) {
      const idx = k * nPer + i;
      X[idx * 2] = cx + sigma * gauss();
      X[idx * 2 + 1] = cy + sigma * gauss();
      y[idx] = k;
    }
  }
  return { X, y };
}

/**
 * Adversarial spiked-variance distribution from the Foygel-Barber 2021 impossibility proof.
 * Matches notebook synth_spike:
 *   x ~ Uniform(-1, 1)
 *   sigma(x) = sqrt(sigma_outside^2 + M^2) if |x - x_center| < eps/2 else sigma_outside
 *   y = 0.5 * x + N(0, sigma(x))
 * Returns inSpike as Uint8Array (0/1) flagging spike-region points.
 */
export function synthSpike(
  n: number,
  rng: () => number,
  eps: number,
  sigmaOutside: number = 0.4,
  M: number = 2.5,
  xCenter: number = 0.0,
): { x: Float64Array; y: Float64Array; inSpike: Uint8Array } {
  const gauss = gaussianSampler(rng);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const inSpike = new Uint8Array(n);
  const sigmaInside = Math.sqrt(sigmaOutside * sigmaOutside + M * M);
  for (let i = 0; i < n; i++) {
    const xi = -1 + 2 * rng();
    const inside = Math.abs(xi - xCenter) < eps / 2 ? 1 : 0;
    const sigma = inside ? sigmaInside : sigmaOutside;
    x[i] = xi;
    y[i] = 0.5 * xi + sigma * gauss();
    inSpike[i] = inside;
  }
  return { x, y, inSpike };
}

// -----------------------------------------------------------------------------
// Numerical helpers
// -----------------------------------------------------------------------------

/** Polynomial features [1, x, x^2, ..., x^degree]. */
function polyFeatures(x: number, degree: number, includeBias: boolean = true): number[] {
  const out: number[] = [];
  if (includeBias) out.push(1);
  let xk = x;
  for (let k = 1; k <= degree; k++) {
    out.push(xk);
    xk *= x;
  }
  return out;
}

/**
 * Solve Ax = b for x where A is symmetric positive-definite.
 * Uses Cholesky decomposition; in-place on A. Sized for tiny matrices (≤ 8x8 here).
 */
function solveCholesky(A: number[][], b: number[]): number[] {
  const n = b.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum <= 0) throw new Error('Matrix is not positive definite');
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  // Forward solve L y = b
  const yv = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let k = 0; k < i; k++) sum -= L[i][k] * yv[k];
    yv[i] = sum / L[i][i];
  }
  // Back solve L^T x = y
  const xv = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = yv[i];
    for (let k = i + 1; k < n; k++) sum -= L[k][i] * xv[k];
    xv[i] = sum / L[i][i];
  }
  return xv;
}

// -----------------------------------------------------------------------------
// Base predictors: ridge and quantile regression on degree-3 polynomial features
// -----------------------------------------------------------------------------

/**
 * Closed-form ridge regression on degree-3 polynomial features.
 * Matches notebook fit_predict_ridge with default lambda = 0.5.
 * Solves (Φ^T Φ + λI) β = Φ^T y via Cholesky on the (4 × 4) normal equations.
 */
export function fitPredictRidge(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xEval: Float64Array,
  lambda: number = 0.5,
): Float64Array {
  const degree = 3;
  const p = degree + 1; // includes bias
  const n = xTrain.length;
  // Build Φ^T Φ + λI and Φ^T y
  const ATA: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const ATy: number[] = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    const phi = polyFeatures(xTrain[i], degree, true);
    for (let r = 0; r < p; r++) {
      ATy[r] += phi[r] * yTrain[i];
      for (let c = 0; c <= r; c++) {
        ATA[r][c] += phi[r] * phi[c];
      }
    }
  }
  // Add λI to the diagonal; mirror upper triangle
  for (let r = 0; r < p; r++) {
    ATA[r][r] += lambda;
    for (let c = r + 1; c < p; c++) ATA[r][c] = ATA[c][r];
  }
  const beta = solveCholesky(ATA, ATy);
  const m = xEval.length;
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const phi = polyFeatures(xEval[i], degree, true);
    let s = 0;
    for (let k = 0; k < p; k++) s += phi[k] * beta[k];
    out[i] = s;
  }
  return out;
}

/**
 * Internal: solve QR via smoothed check loss + Nesterov AGD on degree-d polynomial
 * features. Returns un-scaled β coefficients of length (degree + 1), where
 *   pred(x) = β[0] + β[1]·x + β[2]·x² + ... + β[degree]·x^degree.
 *
 * Smoothed check loss: ρ̃_τ(r) = τ r + h·log(1 + exp(-r/h)).
 * Recovers the LP solution in the h → 0 limit; using h > 0 makes the gradient
 * Lipschitz so AGD converges at the standard 1/k² rate.
 *
 * `hFactor` controls smoothing: h = max(hFactor·std(y), 1e-3). Smaller hFactor
 * gives a sharper, less-biased estimate (closer to LP) at the cost of slower
 * convergence (smaller AGD step size). The default 0.05 is calibrated for the
 * tail-τ uses inside CQR; multi-τ uses (where adjacent levels need to be
 * cleanly distinguishable) want a smaller hFactor with a higher maxIter.
 *
 * Internally optimises β in *scaled* feature space (each polynomial column
 * normalized by its sample std), then unscales before returning. Without
 * scaling, x^degree dominates the gradient and convergence is slow.
 */
/**
 * Build degree-d polynomial features and standardise each non-bias column by
 * its sample std. Returns a row-major Float64Array of size (n × p) plus the
 * column-std vector and the row-major max-row-norm². Computing this matrix is
 * ~25% of a single solve; multi-τ workflows reuse the same `xTrain` across all
 * τ levels, so we factor this out so the caller can amortise it once across K
 * solves (PR #57 review feedback, comment 3142933354).
 */
function _buildScaledFeatures(
  xTrain: Float64Array,
  degree: number,
): {
  PhiScaled: Float64Array; // shape (n, p) row-major
  colStd: Float64Array; // length p; colStd[0] = 1 (bias unchanged)
  n: number;
  p: number;
  maxRowSq: number; // max ||row||² over all rows; used in the Lipschitz estimate
} {
  const p = degree + 1;
  const n = xTrain.length;
  const PhiRaw = new Float64Array(n * p);
  for (let i = 0; i < n; i++) {
    PhiRaw[i * p] = 1;
    let xk = xTrain[i];
    for (let c = 1; c < p; c++) {
      PhiRaw[i * p + c] = xk;
      xk *= xTrain[i];
    }
  }
  const colStd = new Float64Array(p);
  colStd[0] = 1;
  for (let c = 1; c < p; c++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += PhiRaw[i * p + c];
    mean /= n;
    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const d = PhiRaw[i * p + c] - mean;
      varSum += d * d;
    }
    const sd = Math.sqrt(varSum / Math.max(n - 1, 1));
    colStd[c] = sd > 1e-12 ? sd : 1;
  }
  const PhiScaled = new Float64Array(n * p);
  let maxRowSq = 0;
  for (let i = 0; i < n; i++) {
    let rowSq = 1; // bias contributes 1²
    PhiScaled[i * p] = 1;
    for (let c = 1; c < p; c++) {
      const v = PhiRaw[i * p + c] / colStd[c];
      PhiScaled[i * p + c] = v;
      rowSq += v * v;
    }
    if (rowSq > maxRowSq) maxRowSq = rowSq;
  }
  return { PhiScaled, colStd, n, p, maxRowSq };
}

/**
 * Run the smoothed-check-loss + Nesterov AGD on pre-scaled features. Returns
 * the *scaled* β as a Float64Array(p); caller is responsible for unscaling.
 *
 * Performance details (PR #57 review feedback, comment 3142933356):
 *  - All buffers (`grad`, `momentum`, `prevBeta`, `newBeta`) are
 *    `Float64Array(p)` allocated *once* outside the iter loop and reused.
 *  - The 1/n factor is hoisted out of the per-sample gradient accumulator;
 *    instead we accumulate `Σ_i x_i · grad_pred_i` and divide by n once per
 *    iteration.
 *  - Row offsets into `PhiScaled` are pre-computed (`baseI = i*p`).
 */
function _solveQuantileScaled(
  PhiScaled: Float64Array, // shape (n, p) row-major
  yTrain: Float64Array,
  tau: number,
  n: number,
  p: number,
  maxRowSq: number,
  lambda: number,
  hFactor: number,
  maxIter: number,
): Float64Array {
  // Smoothing parameter from y-scale.
  let yMean = 0;
  for (let i = 0; i < n; i++) yMean += yTrain[i];
  yMean /= n;
  let yVar = 0;
  for (let i = 0; i < n; i++) {
    const d = yTrain[i] - yMean;
    yVar += d * d;
  }
  const yStd = Math.sqrt(yVar / Math.max(n - 1, 1));
  const h = Math.max(hFactor * yStd, 1e-3);

  // Pre-allocated buffers (all reused across iterations).
  const beta = new Float64Array(p);
  const momentum = new Float64Array(p);
  const prevBeta = new Float64Array(p);
  const newBeta = new Float64Array(p);
  const grad = new Float64Array(p);
  // Initialise bias to median of y for stability.
  const ySorted = Float64Array.from(yTrain).sort();
  beta[0] = ySorted[Math.floor(n / 2)];
  for (let c = 0; c < p; c++) {
    momentum[c] = beta[c];
    prevBeta[c] = beta[c];
  }

  const L = maxRowSq / (4 * h * n) + lambda + 1e-6;
  const stepSize = 1 / L;
  const invH = 1 / h;
  const invN = 1 / n;
  let tPrev = 1;

  for (let iter = 0; iter < maxIter; iter++) {
    // Zero gradient.
    for (let c = 0; c < p; c++) grad[c] = 0;

    // Accumulate per-sample contributions (without the 1/n factor; applied below).
    for (let i = 0; i < n; i++) {
      const baseI = i * p;
      let pred = 0;
      for (let c = 0; c < p; c++) pred += PhiScaled[baseI + c] * momentum[c];
      const z = (yTrain[i] - pred) * invH;
      const sigmoidZ =
        z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
      // grad_pred = -(τ − (1 − sigmoid(z))) = (1 − τ) − sigmoid(z)
      const grad_pred = 1 - tau - sigmoidZ;
      for (let c = 0; c < p; c++) grad[c] += PhiScaled[baseI + c] * grad_pred;
    }
    // Apply 1/n + L2 ridge in one pass.
    grad[0] *= invN;
    for (let c = 1; c < p; c++) grad[c] = grad[c] * invN + lambda * momentum[c];

    // Gradient step + Nesterov momentum + convergence check (single pass).
    const tCurr = (1 + Math.sqrt(1 + 4 * tPrev * tPrev)) / 2;
    const w = (tPrev - 1) / tCurr;
    let maxDelta = 0;
    for (let c = 0; c < p; c++) {
      const nb = momentum[c] - stepSize * grad[c];
      newBeta[c] = nb;
      const delta = nb - prevBeta[c];
      momentum[c] = nb + w * delta;
      const ad = delta < 0 ? -delta : delta;
      if (ad > maxDelta) maxDelta = ad;
    }
    for (let c = 0; c < p; c++) prevBeta[c] = newBeta[c];
    tPrev = tCurr;
    if (maxDelta < 1e-7 && iter > 10) break;
  }
  // Return a copy so subsequent solves on the same buffers don't clobber it.
  return Float64Array.from(prevBeta);
}

/** Unscale β returned by `_solveQuantileScaled` to the original feature basis. */
function _unscaleBeta(betaScaled: Float64Array, colStd: Float64Array, p: number): Float64Array {
  const out = new Float64Array(p);
  out[0] = betaScaled[0];
  for (let c = 1; c < p; c++) out[c] = betaScaled[c] / colStd[c];
  return out;
}

function _solveQuantileBetaUnscaled(
  xTrain: Float64Array,
  yTrain: Float64Array,
  tau: number,
  degree: number,
  lambda: number,
  hFactor: number = 0.005,
  maxIter: number = 2000,
): Float64Array {
  const { PhiScaled, colStd, n, p, maxRowSq } = _buildScaledFeatures(xTrain, degree);
  const betaScaled = _solveQuantileScaled(
    PhiScaled,
    yTrain,
    tau,
    n,
    p,
    maxRowSq,
    lambda,
    hFactor,
    maxIter,
  );
  return _unscaleBeta(betaScaled, colStd, p);
}

/** Evaluate degree-d polynomial pred(x) = β[0] + β[1]·x + β[2]·x² + ... at xEval. */
function _polyPredict(beta: Float64Array, xEval: Float64Array, degree: number): Float64Array {
  const m = xEval.length;
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let pred = beta[0];
    let xk = xEval[i];
    for (let c = 1; c <= degree; c++) {
      pred += beta[c] * xk;
      xk *= xEval[i];
    }
    out[i] = pred;
  }
  return out;
}

/**
 * Quantile regression at level tau, solved via smoothed check loss + accelerated
 * gradient descent (Nesterov). Matches notebook fit_predict_quantile in spirit:
 *   loss(r) = tau * r + h * log(1 + exp(-r/h)) + λ ||β||²,  h ≈ 0.05 * std(y)
 * The smoothing parameter h shrinks the non-smooth corner of ρ_τ but recovers
 * the LP solution in the h → 0 limit. Output is QR predictions on xEval.
 *
 * Polynomial features include the bias term; matches the Python's
 * PolynomialFeatures(degree=3, include_bias=False) + free intercept by appending
 * a leading 1 explicitly. (Net effect: 4-dim feature vector.)
 */
export function fitPredictQuantile(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xEval: Float64Array,
  tau: number,
  lambda: number = 0.01,
): Float64Array {
  const degree = 3;
  const beta = _solveQuantileBetaUnscaled(xTrain, yTrain, tau, degree, lambda);
  return _polyPredict(beta, xEval, degree);
}

/**
 * Bootstrap CI for a single coefficient of the linear-QR estimator.
 * Resamples (X, Y) with replacement B times; refits the smoothed-check QR;
 * returns the empirical (alpha/2, 1-alpha/2) quantiles of the chosen coefficient
 * (default index 1, i.e., the slope on x in the un-scaled feature basis).
 *
 * Used by the BootstrapQuantileCI viz component and by §5 of the topic page.
 */
/**
 * Single bootstrap iteration: resample (X, Y) with replacement, refit the
 * smoothed-check QR, return the chosen coefficient. Exported so callers (e.g.
 * the BootstrapQuantileCI viz component) can drive the bootstrap manually —
 * yielding to the main thread between draws — instead of paying the full
 * `bootstrapQuantileCI` synchronous cost in one shot. PR #57 review feedback
 * (comment 3142933355).
 */
export function bootstrapQuantileSample(
  xData: Float64Array,
  yData: Float64Array,
  tau: number,
  rng: () => number,
  coefIndex: number = 1,
  degree: number = 3,
  lambda: number = 0.01,
): number {
  const n = xData.length;
  const xb = new Float64Array(n);
  const yb = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(rng() * n);
    xb[i] = xData[idx];
    yb[i] = yData[idx];
  }
  const beta = _solveQuantileBetaUnscaled(xb, yb, tau, degree, lambda);
  return beta[coefIndex];
}

export function bootstrapQuantileCI(
  x: Float64Array,
  y: Float64Array,
  tau: number,
  B: number,
  alpha: number,
  rng: () => number,
  coefIndex: number = 1,
  degree: number = 3,
  lambda: number = 0.01,
): {
  coefDraws: Float64Array;
  ciLower: number;
  ciUpper: number;
  empiricalMean: number;
  empiricalStd: number;
} {
  const coefDraws = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    coefDraws[b] = bootstrapQuantileSample(x, y, tau, rng, coefIndex, degree, lambda);
  }
  let mean = 0;
  for (let b = 0; b < B; b++) mean += coefDraws[b];
  mean /= B;
  let varSum = 0;
  for (let b = 0; b < B; b++) {
    const d = coefDraws[b] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / Math.max(B - 1, 1));
  const sorted = Float64Array.from(coefDraws).sort();
  const lowerIdx = Math.max(0, Math.floor((alpha / 2) * B));
  const upperIdx = Math.min(B - 1, Math.ceil((1 - alpha / 2) * B) - 1);
  return {
    coefDraws,
    ciLower: sorted[lowerIdx],
    ciUpper: sorted[upperIdx],
    empiricalMean: mean,
    empiricalStd: std,
  };
}

/**
 * Fit linear QR at K levels simultaneously on the same training data.
 * Returns a (K, nEval) row-major Float64Array Q with Q[k, j] = q̂_{taus[k]}(xEval[j]).
 *
 * Used by the QRFitExplorer viz component (multi-tau mode) and by §6 of the
 * topic page (rearrangement demonstration).
 */
export function fitPredictMultipleQuantiles(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xEval: Float64Array,
  taus: Float64Array,
  degree: number = 3,
  lambda: number = 0.01,
): Float64Array {
  const K = taus.length;
  const nEval = xEval.length;
  const Q = new Float64Array(K * nEval);
  // Build features ONCE: training data and degree are constant across τ levels.
  // PR #57 review feedback (comment 3142933354): previously this rebuilt the
  // (n × p) feature matrix and the column-std vector for every τ — pure waste.
  const { PhiScaled, colStd, n, p, maxRowSq } = _buildScaledFeatures(xTrain, degree);
  for (let k = 0; k < K; k++) {
    const betaScaled = _solveQuantileScaled(
      PhiScaled,
      yTrain,
      taus[k],
      n,
      p,
      maxRowSq,
      lambda,
      0.005,
      2000,
    );
    const beta = _unscaleBeta(betaScaled, colStd, p);
    const preds = _polyPredict(beta, xEval, degree);
    for (let j = 0; j < nEval; j++) Q[k * nEval + j] = preds[j];
  }
  return Q;
}

/**
 * CFV-G 2010 rearrangement: sort along the τ-axis at each evaluation point.
 * Pure post-processing on already-fitted multi-quantile predictions; enforces
 * monotonicity in τ without re-fitting. Input Q has shape (K, nEval) row-major
 * (as returned by fitPredictMultipleQuantiles); output has the same shape.
 *
 * Used by the QRFitExplorer viz component (rearrangement toggle, §6).
 */
export function rearrangedQuantilePredictions(
  Q: Float64Array,
  K: number,
  nEval: number,
): Float64Array {
  const Qtilde = new Float64Array(K * nEval);
  const col = new Float64Array(K);
  for (let j = 0; j < nEval; j++) {
    for (let k = 0; k < K; k++) col[k] = Q[k * nEval + j];
    col.sort();
    for (let k = 0; k < K; k++) Qtilde[k * nEval + j] = col[k];
  }
  return Qtilde;
}

// -----------------------------------------------------------------------------
// Conformal procedures
// -----------------------------------------------------------------------------

export interface ConformalInterval {
  lower: Float64Array;
  upper: Float64Array;
  qHat: number;
  calScores: Float64Array;
}

type FitPredictFn = (
  xTrain: Float64Array,
  yTrain: Float64Array,
  xEval: Float64Array,
) => Float64Array;

/**
 * Split (inductive) conformal prediction interval. Matches notebook §2 Definition 2:
 *   - Fit predictor on training set
 *   - Calibration scores S_i = |Y_i - μ̂(X_i)|
 *   - Threshold q̂_{1-α} = ceil((1-α)(n_cal+1))-th smallest of S
 *   - Prediction interval: [μ̂(X_test) - q̂, μ̂(X_test) + q̂]
 */
export function splitConformalInterval(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xCal: Float64Array,
  yCal: Float64Array,
  xTest: Float64Array,
  alpha: number,
  fitPredictFn: FitPredictFn,
): ConformalInterval {
  const yCalPred = fitPredictFn(xTrain, yTrain, xCal);
  const yTestPred = fitPredictFn(xTrain, yTrain, xTest);
  const nCal = xCal.length;
  const calScores = new Float64Array(nCal);
  for (let i = 0; i < nCal; i++) calScores[i] = Math.abs(yCal[i] - yCalPred[i]);
  // k = ceil((1-α)(n_cal+1)), capped at n_cal
  let k = Math.ceil((1 - alpha) * (nCal + 1));
  if (k > nCal) k = nCal;
  if (k < 1) k = 1;
  const sorted = Float64Array.from(calScores).sort();
  const qHat = sorted[k - 1];
  const m = xTest.length;
  const lower = new Float64Array(m);
  const upper = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    lower[i] = yTestPred[i] - qHat;
    upper[i] = yTestPred[i] + qHat;
  }
  return { lower, upper, qHat, calScores };
}

/**
 * Jackknife+ prediction interval (Barber-Candès-Ramdas-Tibshirani 2021).
 * Matches notebook §5 Definition 3. Fits n leave-one-out predictors.
 *   R_i = |Y_i - μ̂_{-i}(X_i)|
 *   lower endpoint: floor(α(n+1))-th smallest of {μ̂_{-i}(X_test) - R_i}
 *   upper endpoint: ceil((1-α)(n+1))-th smallest of {μ̂_{-i}(X_test) + R_i}
 */
export function jackknifePlusInterval(
  x: Float64Array,
  y: Float64Array,
  xTest: Float64Array,
  alpha: number,
  fitPredictFn: FitPredictFn,
): { lower: Float64Array; upper: Float64Array } {
  const n = x.length;
  const nTest = xTest.length;
  const R = new Float64Array(n);
  // muTest[i, t] = μ̂_{-i}(xTest[t]); store flat row-major (n × nTest)
  const muTest = new Float64Array(n * nTest);
  for (let i = 0; i < n; i++) {
    const xLoo = new Float64Array(n - 1);
    const yLoo = new Float64Array(n - 1);
    let j = 0;
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      xLoo[j] = x[k];
      yLoo[j] = y[k];
      j++;
    }
    // LOO residual at i
    const yiPred = fitPredictFn(xLoo, yLoo, new Float64Array([x[i]]));
    R[i] = Math.abs(y[i] - yiPred[0]);
    // LOO predictions at all test points
    const testPreds = fitPredictFn(xLoo, yLoo, xTest);
    for (let t = 0; t < nTest; t++) muTest[i * nTest + t] = testPreds[t];
  }
  return jackknifeQuantileEndpoints(muTest, R, n, nTest, alpha);
}

/**
 * K-fold CV+ prediction interval. Same coverage bound as jackknife+ (1 - 2α)
 * but K predictor fits instead of n. Matches notebook §5.
 */
export function cvPlusInterval(
  x: Float64Array,
  y: Float64Array,
  xTest: Float64Array,
  alpha: number,
  fitPredictFn: FitPredictFn,
  K: number,
  rng: () => number,
): { lower: Float64Array; upper: Float64Array } {
  const n = x.length;
  const nTest = xTest.length;
  // Random fold assignment (mod K of a random permutation)
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
  }
  const fold = new Int32Array(n);
  for (let i = 0; i < n; i++) fold[perm[i]] = i % K;
  const R = new Float64Array(n);
  const muTest = new Float64Array(n * nTest);
  for (let k = 0; k < K; k++) {
    // Train mask: not in fold k
    const trainIdx: number[] = [];
    const heldIdx: number[] = [];
    for (let i = 0; i < n; i++) (fold[i] === k ? heldIdx : trainIdx).push(i);
    if (heldIdx.length === 0) continue;
    const xTrain = new Float64Array(trainIdx.length);
    const yTrain = new Float64Array(trainIdx.length);
    for (let j = 0; j < trainIdx.length; j++) {
      xTrain[j] = x[trainIdx[j]];
      yTrain[j] = y[trainIdx[j]];
    }
    // Residuals on held-out fold
    const xHeld = new Float64Array(heldIdx.length);
    for (let j = 0; j < heldIdx.length; j++) xHeld[j] = x[heldIdx[j]];
    const heldPred = fitPredictFn(xTrain, yTrain, xHeld);
    for (let j = 0; j < heldIdx.length; j++) {
      R[heldIdx[j]] = Math.abs(y[heldIdx[j]] - heldPred[j]);
    }
    // Test predictions for this fold's predictor (assigned to held-out indices)
    const testPred = fitPredictFn(xTrain, yTrain, xTest);
    for (const i of heldIdx) {
      for (let t = 0; t < nTest; t++) muTest[i * nTest + t] = testPred[t];
    }
  }
  return jackknifeQuantileEndpoints(muTest, R, n, nTest, alpha);
}

/**
 * Shared quantile-endpoint computation for jackknife+ and CV+:
 *   lower = floor(α(n+1))-th smallest of {μ_{-i}(x_test) - R_i}
 *   upper = ceil((1-α)(n+1))-th smallest of {μ_{-i}(x_test) + R_i}
 */
function jackknifeQuantileEndpoints(
  muTest: Float64Array,
  R: Float64Array,
  n: number,
  nTest: number,
  alpha: number,
): { lower: Float64Array; upper: Float64Array } {
  let lowerRank = Math.max(1, Math.floor(alpha * (n + 1)));
  let upperRank = Math.min(n, Math.ceil((1 - alpha) * (n + 1)));
  const lower = new Float64Array(nTest);
  const upper = new Float64Array(nTest);
  const colLo = new Float64Array(n);
  const colHi = new Float64Array(n);
  for (let t = 0; t < nTest; t++) {
    for (let i = 0; i < n; i++) {
      colLo[i] = muTest[i * nTest + t] - R[i];
      colHi[i] = muTest[i * nTest + t] + R[i];
    }
    const sortedLo = Float64Array.from(colLo).sort();
    const sortedHi = Float64Array.from(colHi).sort();
    lower[t] = sortedLo[lowerRank - 1];
    upper[t] = sortedHi[upperRank - 1];
  }
  return { lower, upper };
}

/**
 * Conformalized Quantile Regression interval (Romano-Patterson-Candès 2019).
 * Matches notebook §6 Definition 4. Fit τ=α/2 and τ=1-α/2 quantile regressions
 * on training set; calibration score E_i = max(q̂_lo(X_i) - Y_i, Y_i - q̂_hi(X_i));
 * threshold Q̂_{1-α} = ceil((1-α)(n_cal+1))-th smallest of E.
 * Returns CQR interval [q̂_lo(X_test) - Q̂, q̂_hi(X_test) + Q̂].
 */
export function cqrInterval(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xCal: Float64Array,
  yCal: Float64Array,
  xTest: Float64Array,
  alpha: number,
): { lower: Float64Array; upper: Float64Array; qHat: number } {
  const tauLo = alpha / 2;
  const tauHi = 1 - alpha / 2;
  const qLoCal = fitPredictQuantile(xTrain, yTrain, xCal, tauLo);
  const qHiCal = fitPredictQuantile(xTrain, yTrain, xCal, tauHi);
  const nCal = xCal.length;
  const E = new Float64Array(nCal);
  for (let i = 0; i < nCal; i++) {
    E[i] = Math.max(qLoCal[i] - yCal[i], yCal[i] - qHiCal[i]);
  }
  let k = Math.ceil((1 - alpha) * (nCal + 1));
  if (k > nCal) k = nCal;
  if (k < 1) k = 1;
  const sortedE = Float64Array.from(E).sort();
  const qHat = sortedE[k - 1];
  const qLoTest = fitPredictQuantile(xTrain, yTrain, xTest, tauLo);
  const qHiTest = fitPredictQuantile(xTrain, yTrain, xTest, tauHi);
  const m = xTest.length;
  const lower = new Float64Array(m);
  const upper = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    lower[i] = qLoTest[i] - qHat;
    upper[i] = qHiTest[i] + qHat;
  }
  return { lower, upper, qHat };
}

/**
 * Importance-weighted split conformal under known covariate shift
 * (Tibshirani-Barber-Candès-Ramdas 2019). Each calibration point i carries
 * importance weight w_i = p_test(X_i) / p_train(X_i). The threshold is the
 * (1 - α) weighted quantile of {S_i ∪ {+∞}} where the test point's own weight
 * sits in the denominator. With uniform weights this reduces to standard split
 * conformal — verified by test 5.5.
 */
export function weightedSplitConformal(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xCal: Float64Array,
  yCal: Float64Array,
  xTest: Float64Array,
  alpha: number,
  weightFn: (x: number) => number,
  fitPredictFn: FitPredictFn,
): ConformalInterval {
  const yCalPred = fitPredictFn(xTrain, yTrain, xCal);
  const yTestPred = fitPredictFn(xTrain, yTrain, xTest);
  const nCal = xCal.length;
  const calScores = new Float64Array(nCal);
  const wCal = new Float64Array(nCal);
  let wSumCal = 0;
  for (let i = 0; i < nCal; i++) {
    calScores[i] = Math.abs(yCal[i] - yCalPred[i]);
    wCal[i] = weightFn(xCal[i]);
    wSumCal += wCal[i];
  }
  const m = xTest.length;
  const lower = new Float64Array(m);
  const upper = new Float64Array(m);
  // Per-test-point weighted quantile (test point's own weight matters)
  // Sort calibration scores and accumulate normalised weights until
  // cumulative ≥ 1 - α, with "+∞" virtual score carrying weight w_test.
  const order = new Int32Array(nCal);
  for (let i = 0; i < nCal; i++) order[i] = i;
  order.sort((a, b) => calScores[a] - calScores[b]);
  // qHat is genuinely per-test-point under covariate shift (the test point's own
  // importance weight enters the denominator). We surface only the LAST test
  // point's threshold for diagnostic parity with splitConformalInterval; callers
  // needing every per-point threshold should pass single-element xTest arrays.
  let qHat = Number.NaN;
  for (let t = 0; t < m; t++) {
    const wTest = weightFn(xTest[t]);
    const denom = wSumCal + wTest;
    let cum = 0;
    qHat = Number.POSITIVE_INFINITY;
    const target = 1 - alpha;
    for (let j = 0; j < nCal; j++) {
      cum += wCal[order[j]] / denom;
      if (cum >= target) {
        qHat = calScores[order[j]];
        break;
      }
    }
    lower[t] = yTestPred[t] - qHat;
    upper[t] = yTestPred[t] + qHat;
  }
  return { lower, upper, qHat, calScores };
}

// -----------------------------------------------------------------------------
// Adaptive Prediction Sets (APS) for classification
// -----------------------------------------------------------------------------

/**
 * Sort each row of probs in descending order, returning the permutation indices.
 * sortIdx[i*K + j] = which original class is the j-th largest in row i.
 */
function argsortRowsDesc(probs: Float64Array, n: number, K: number): Int32Array {
  const sortIdx = new Int32Array(n * K);
  const tmp = new Array(K);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < K; k++) tmp[k] = k;
    const offset = i * K;
    tmp.sort((a, b) => probs[offset + b] - probs[offset + a]);
    for (let k = 0; k < K; k++) sortIdx[offset + k] = tmp[k];
  }
  return sortIdx;
}

/**
 * Deterministic APS nonconformity score (Romano-Sesia-Candès 2020).
 * Matches notebook §7 Definition 5:
 *   s(x, y) = sum_{j=1}^{rho(y;x)-1} π̂(c_(j) | x)
 * i.e., cumulative softmax mass of all classes ranked strictly above the true label.
 *
 * @param probs  row-major (n × K) softmax probabilities
 * @param y      true class labels (length n)
 * @returns      length-n nonconformity scores
 */
export function apsScoreDeterministic(
  probs: Float64Array,
  y: Int32Array,
  n: number,
  K: number,
): Float64Array {
  const sortIdx = argsortRowsDesc(probs, n, K);
  const scores = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const offset = i * K;
    // Find rank of y[i] in the descending order
    let rank = -1;
    for (let j = 0; j < K; j++) {
      if (sortIdx[offset + j] === y[i]) {
        rank = j;
        break;
      }
    }
    // Cumulative mass of classes strictly above y[i] (ranks 0 .. rank-1)
    let cum = 0;
    for (let j = 0; j < rank; j++) cum += probs[offset + sortIdx[offset + j]];
    scores[i] = cum;
  }
  return scores;
}

/**
 * Deterministic APS prediction set. Matches notebook §7:
 *   include class c iff cumulative-mass-of-strictly-higher-ranked classes ≤ threshold.
 *
 * @param probs      row-major (n × K) softmax probabilities
 * @param threshold  empirical-quantile threshold from calibration
 * @returns          row-major (n × K) Uint8Array bool mask: 1 = in set
 */
export function apsPredictionSetDeterministic(
  probs: Float64Array,
  threshold: number,
  n: number,
  K: number,
): Uint8Array {
  const sortIdx = argsortRowsDesc(probs, n, K);
  const inSet = new Uint8Array(n * K);
  for (let i = 0; i < n; i++) {
    const offset = i * K;
    let cum = 0;
    for (let j = 0; j < K; j++) {
      const c = sortIdx[offset + j];
      // Class c is in the set iff cumulative mass of strictly-higher classes ≤ threshold.
      // The top class always passes (cum = 0).
      if (cum <= threshold) inSet[offset + c] = 1;
      cum += probs[offset + c];
    }
  }
  return inSet;
}

// -----------------------------------------------------------------------------
// Multinomial logistic regression (small Newton's method on (K-1) classes)
// Used by the APS region map in the viz components and by verification test 5.4.
// Reference class is K-1 (last class); softmax probabilities returned for all K.
// -----------------------------------------------------------------------------

/**
 * Fit a multinomial logistic regression on 2D features and return softmax
 * probabilities at evaluation points. Uses Newton-Raphson with L2 regularisation
 * on the (K-1) sets of coefficients (last class = reference; coefficients = 0).
 * Sufficient for the small problems APS visualises (n ≤ ~1000, K = 3).
 */
export function fitPredictLogisticRegression2D(
  XTrain: Float64Array, // shape (nTrain, 2) row-major
  yTrain: Int32Array,
  XEval: Float64Array, // shape (nEval, 2) row-major
  K: number,
  l2: number = 1e-4,
  maxIter: number = 50,
): Float64Array {
  const d = 3; // bias + 2 features
  const nTrain = yTrain.length;
  const nEval = XEval.length / 2;
  const Kfree = K - 1; // (K-1) classes have free coefficients
  const totalDim = d * Kfree;
  // Coefficients laid out flat: beta[c*d + j]
  const beta = new Array(totalDim).fill(0);
  // Build feature matrix Φ with bias column
  function feat(idx: number, isEval: boolean): [number, number, number] {
    const X = isEval ? XEval : XTrain;
    return [1, X[idx * 2], X[idx * 2 + 1]];
  }
  for (let iter = 0; iter < maxIter; iter++) {
    // Compute gradient and Hessian
    // Gradient: ∇_β_c L = Σ_i (p_{ic} - y_{ic}) φ_i  + l2 β_c    for c < K-1
    // Hessian (block c, c'): Σ_i p_{ic} (1{c=c'} - p_{ic'}) φ_i φ_i^T  + l2 I
    const grad = new Array(totalDim).fill(0);
    const H: number[][] = Array.from({ length: totalDim }, () => new Array(totalDim).fill(0));
    for (let i = 0; i < nTrain; i++) {
      const phi = feat(i, false);
      // Compute logits for classes 0..K-2; reference K-1 has logit 0
      const logits = new Array(K).fill(0);
      let maxLogit = 0;
      for (let c = 0; c < Kfree; c++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += beta[c * d + j] * phi[j];
        logits[c] = s;
        if (s > maxLogit) maxLogit = s;
      }
      // Softmax (numerically stable: subtract max)
      let expSum = 0;
      const expL = new Array(K);
      for (let c = 0; c < K; c++) {
        expL[c] = Math.exp(logits[c] - maxLogit);
        expSum += expL[c];
      }
      const probI = new Array(K);
      for (let c = 0; c < K; c++) probI[c] = expL[c] / expSum;
      // Gradient contribution: Σ_i (p_ic - y_ic) φ
      for (let c = 0; c < Kfree; c++) {
        const indicator = yTrain[i] === c ? 1 : 0;
        const r = probI[c] - indicator;
        for (let j = 0; j < d; j++) grad[c * d + j] += r * phi[j];
      }
      // Hessian: blocks (c, c')
      for (let c = 0; c < Kfree; c++) {
        for (let cp = 0; cp <= c; cp++) {
          const w = probI[c] * ((c === cp ? 1 : 0) - probI[cp]);
          for (let j = 0; j < d; j++) {
            for (let jp = 0; jp <= j; jp++) {
              const v = w * phi[j] * phi[jp];
              H[c * d + j][cp * d + jp] += v;
              if (j !== jp) H[c * d + jp][cp * d + j] += v;
            }
          }
        }
      }
    }
    // Add L2 regulariser
    for (let p = 0; p < totalDim; p++) {
      grad[p] += l2 * beta[p];
      H[p][p] += l2;
    }
    // Mirror Hessian's lower triangle into upper triangle
    for (let r = 0; r < totalDim; r++) {
      for (let c = r + 1; c < totalDim; c++) H[r][c] = H[c][r];
    }
    // Newton step: solve H δ = grad, then β -= δ
    let delta: number[];
    try {
      delta = solveCholesky(H, grad);
    } catch (_e) {
      // Hessian not PD (rare with l2 > 0); fall back to gradient step
      delta = grad.map((g) => g * 0.01);
    }
    let maxStep = 0;
    for (let p = 0; p < totalDim; p++) {
      beta[p] -= delta[p];
      const ad = Math.abs(delta[p]);
      if (ad > maxStep) maxStep = ad;
    }
    if (maxStep < 1e-7) break;
  }
  // Predict softmax at XEval
  const probsOut = new Float64Array(nEval * K);
  for (let i = 0; i < nEval; i++) {
    const phi = feat(i, true);
    const logits = new Array(K).fill(0);
    let maxLogit = 0;
    for (let c = 0; c < Kfree; c++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += beta[c * d + j] * phi[j];
      logits[c] = s;
      if (s > maxLogit) maxLogit = s;
    }
    let expSum = 0;
    const expL = new Array(K);
    for (let c = 0; c < K; c++) {
      expL[c] = Math.exp(logits[c] - maxLogit);
      expSum += expL[c];
    }
    for (let c = 0; c < K; c++) probsOut[i * K + c] = expL[c] / expSum;
  }
  return probsOut;
}

// ============================================================================
// PREDICTION-INTERVALS TOPIC HELPERS
// Source: notebooks/prediction-intervals/01_prediction_intervals.ipynb (cells 4, 6, 8, 10, 12, 14)
// Added: 2026-04-27
// ============================================================================

/** Color palette matching the notebook's matplotlib theme (also used by formalML site theme). */
export const palettePI = {
  blue: '#2563EB',
  red: '#DC2626',
  green: '#059669',
  amber: '#D97706',
  purple: '#7C3AED',
  slate: '#475569',
  teal: '#0F6E56',
  lightBlue: '#DBEAFE',
  lightRed: '#FEE2E2',
  lightGreen: '#D1FAE5',
  lightAmber: '#FEF3C7',
  lightPurple: '#EDE9FE',
  lightSlate: '#F1F5F9',
} as const;

// ─── Data generators ─────────────────────────────────────────────────────────

/**
 * Running Example 1 — heteroscedastic Gaussian.
 * Y | X = x ~ N(sin(x), σ(x)²) with σ(x) = 0.2 + slope·|x|/3, X ~ Uniform(-3, 3).
 * Default slope = 0.6 reproduces the canonical RE1; slope = 0 is homoscedastic
 * (used in §5's Theorem 5.2 sweep). Notebook source: generate_heteroscedastic
 * in cell 4 and generate_heteroscedastic_strength in cell 12.
 */
export function generateHeteroscedastic(
  n: number,
  rng: () => number,
  opts: { slope?: number } = {},
): { X: Float64Array; Y: Float64Array; sigma: Float64Array } {
  const slope = opts.slope ?? 0.6;
  const gauss = gaussianSampler(rng);
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  const sigma = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = -3 + 6 * rng();
    const s = 0.2 + (slope * Math.abs(x)) / 3;
    X[i] = x;
    sigma[i] = s;
    Y[i] = Math.sin(x) + s * gauss();
  }
  return { X, Y, sigma };
}

/** Sum of k independent Z² draws — chi-squared with df k. */
function _chiSquared(k: number, gauss: () => number): number {
  let s = 0;
  for (let i = 0; i < k; i++) {
    const z = gauss();
    s += z * z;
  }
  return s;
}

/**
 * Running Example 2 — heavy-tailed location-shift.
 * Y | X = x ~ μ(x) + sigmaScale·t₃ with μ(x) = 0.4·cos(πx), sigmaScale = 0.6, X ~ Uniform(-2, 2).
 * t₃ simulated as Z / √(W/3) with Z ~ N(0,1) and W ~ χ²₃ (sum of three Z²).
 * Notebook source: generate_heavy_tailed_location in cell 10.
 */
export function generateHeavyTailedLocation(
  n: number,
  rng: () => number,
  sigmaScale: number = 0.6,
): { X: Float64Array; Y: Float64Array; eps: Float64Array } {
  const gauss = gaussianSampler(rng);
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  const eps = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = -2 + 4 * rng();
    const z = gauss();
    const w = _chiSquared(3, gauss);
    const e = sigmaScale * (z / Math.sqrt(w / 3));
    X[i] = x;
    eps[i] = e;
    Y[i] = 0.4 * Math.cos(Math.PI * x) + e;
  }
  return { X, Y, eps };
}

/**
 * Skewed location-shift — symmetry-violation diagnostic for §4.5.
 * ε = sigmaScale · (χ²₃ - 3) / √6  → mean zero, right-skewed, F ≠ -F.
 * Notebook source: generate_skewed_location in cell 10.
 */
export function generateSkewedLocation(
  n: number,
  rng: () => number,
  sigmaScale: number = 0.6,
): { X: Float64Array; Y: Float64Array; eps: Float64Array } {
  const gauss = gaussianSampler(rng);
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  const eps = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = -2 + 4 * rng();
    const c = _chiSquared(3, gauss);
    const e = (sigmaScale * (c - 3)) / Math.sqrt(6);
    X[i] = x;
    eps[i] = e;
    Y[i] = 0.4 * Math.cos(Math.PI * x) + e;
  }
  return { X, Y, eps };
}

// ─── Core construction primitives ───────────────────────────────────────────

/**
 * Conformal (1-α)-quantile per Definition 7.
 * Returns the ⌈(1-α)(n+1)⌉-th order statistic of `scores`. +∞ if that index exceeds n
 * (vacuous interval — the honest finite-sample answer when α is too small for n).
 * Notebook source: conformal_quantile in cell 6.
 */
export function conformalQuantile(scores: ArrayLike<number>, alpha: number): number {
  const n = scores.length;
  const k = Math.ceil((1 - alpha) * (n + 1));
  if (k > n) return Infinity;
  const sorted = Array.from(scores).sort((a, b) => a - b);
  return sorted[k - 1];
}

/**
 * Walsh averages (rᵢ + rⱼ)/2 for i ≤ j (includes i = j diagonal).
 * Returns a flat unsorted Float64Array of length M = n(n+1)/2.
 * Notebook source: walsh_averages in cell 10. Conceptually identical to walshSorted
 * inside WalshAveragesExplorer.tsx but unsorted (callers sort when needed).
 */
export function walshAveragesPI(r: ArrayLike<number>): Float64Array {
  const n = r.length;
  const M = (n * (n + 1)) / 2;
  const out = new Float64Array(M);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const ri = r[i];
    for (let j = i; j < n; j++) {
      out[k++] = (ri + r[j]) / 2;
    }
  }
  return out;
}

/**
 * Hodges-Lehmann critical index for prediction-interval test inversion (Definition 11).
 * Closed form: w = ⌊M·α/2⌋ where M = n_cal(n_cal+1)/2 — the count of Walsh averages produced
 * by `walshAveragesPI(r_cal)` for a calibration set of size n_cal.
 *
 * The resulting interval [A_(w+1), A_(M-w)] in sorted Walsh-average order statistics has
 * coverage (M - 2w)/M ≥ 1 - α for the centre of symmetry.
 *
 * NOTE: The original notebook (cell 10) uses `n = nCal + 1` to compute M, motivated by the
 * augmented-sample (n_cal + 1 residuals) framing of Theorem 3's rank-symmetry argument. But
 * downstream `hl_interval` indexes into the *calibration-only* Walsh averages (length
 * n_cal(n_cal+1)/2), creating a small but real discrepancy where the upper-endpoint index
 * `A[M - 1 - w]` reads from a position offset by ~n_cal slots from where the closed-form
 * theory prescribes. We use the calibration-only M here so the closed-form's α-fraction
 * trim semantics apply to the array we actually slice. Empirical impact: HL widths shift by
 * ~0.5–1% versus the notebook output; coverage shifts by ≤0.5pp. See PR #59 review thread.
 */
export function hlCriticalIndexPI(
  nCal: number,
  alpha: number,
): { w: number; M: number } {
  const M = (nCal * (nCal + 1)) / 2;
  let w = Math.floor((M * alpha) / 2);
  if (2 * w >= M) w = Math.floor((M - 1) / 2);
  return { w, M };
}

// ─── Construction wrappers (match notebook signatures) ─────────────────────

export interface PIConstructionResult {
  /** Lower endpoints on Xte, length |Xte|. */
  lo: Float64Array;
  /** Upper endpoints on Xte, length |Xte|. */
  hi: Float64Array;
  /** Wall-clock fit + predict time in milliseconds. */
  fitTimeMs: number;
}

export interface SplitConformalResultPI extends PIConstructionResult {
  /** Conformal threshold q̂_{1-α}. */
  qHat: number;
  /** Base predictor μ̂ evaluated on Xte. */
  muTest: Float64Array;
  /** Calibration scores |Y_cal − μ̂(X_cal)|. */
  calScores: Float64Array;
}

/**
 * Split-conformal prediction interval with degree-3 polynomial-ridge base predictor.
 * Reproduces split_conformal_interval (notebook cell 6).
 * Verification (notebook): RE1, n_train = n_cal = 500, n_test = 5000, α = 0.1
 *   → q̂ ≈ 0.970, marginal coverage ≈ 0.921, conditional range ≈ 0.205.
 */
export function splitConformalIntervalPI(
  Xtr: ArrayLike<number>,
  Ytr: ArrayLike<number>,
  Xcal: ArrayLike<number>,
  Ycal: ArrayLike<number>,
  Xte: ArrayLike<number>,
  alpha: number,
  ridgeLambda: number = 0.1,
): SplitConformalResultPI {
  const t0 = performance.now();
  const xTr = Float64Array.from(Xtr);
  const yTr = Float64Array.from(Ytr);
  const nCal = Xcal.length;
  const nTe = Xte.length;
  const xEval = new Float64Array(nCal + nTe);
  for (let i = 0; i < nCal; i++) xEval[i] = Xcal[i];
  for (let i = 0; i < nTe; i++) xEval[nCal + i] = Xte[i];
  const muEval = fitPredictRidge(xTr, yTr, xEval, ridgeLambda);
  const muCal = muEval.subarray(0, nCal);
  const muTest = Float64Array.from(muEval.subarray(nCal));
  const calScores = new Float64Array(nCal);
  for (let i = 0; i < nCal; i++) calScores[i] = Math.abs(Ycal[i] - muCal[i]);
  const qHat = conformalQuantile(calScores, alpha);
  const lo = new Float64Array(nTe);
  const hi = new Float64Array(nTe);
  for (let i = 0; i < nTe; i++) {
    lo[i] = muTest[i] - qHat;
    hi[i] = muTest[i] + qHat;
  }
  return {
    lo,
    hi,
    qHat,
    muTest,
    calScores,
    fitTimeMs: performance.now() - t0,
  };
}

export interface PureQrResultPI extends PIConstructionResult {
  qLoTest: Float64Array;
  qHiTest: Float64Array;
}

/**
 * Pure-QR prediction interval (Definition 8). No calibration step.
 * Reproduces pure_qr_interval (notebook cell 8).
 * Verification (notebook): RE1, n_train = 1000, n_test = 5000, α = 0.1
 *   → marginal ≈ 0.904, mean width ≈ 1.662, conditional range ≈ 0.150.
 */
export function pureQrIntervalPI(
  Xtr: ArrayLike<number>,
  Ytr: ArrayLike<number>,
  Xte: ArrayLike<number>,
  alpha: number,
  qrLambda: number = 0.001,
): PureQrResultPI {
  const t0 = performance.now();
  const xTr = Float64Array.from(Xtr);
  const yTr = Float64Array.from(Ytr);
  const xTe = Float64Array.from(Xte);
  const qLo = fitPredictQuantile(xTr, yTr, xTe, alpha / 2, qrLambda);
  const qHi = fitPredictQuantile(xTr, yTr, xTe, 1 - alpha / 2, qrLambda);
  return {
    lo: Float64Array.from(qLo),
    hi: Float64Array.from(qHi),
    qLoTest: qLo,
    qHiTest: qHi,
    fitTimeMs: performance.now() - t0,
  };
}

export interface CqrResultPI extends PIConstructionResult {
  /** Conformal CQR threshold Q̂_{1-α}. */
  Q: number;
  /** Lower QR curve τ = α/2 evaluated on Xte. */
  qLoTest: Float64Array;
  /** Upper QR curve τ = 1 − α/2 evaluated on Xte. */
  qHiTest: Float64Array;
  /** CQR scores E_i = max(q̂_lo(X_i) − Y_i, Y_i − q̂_hi(X_i)). */
  calScores: Float64Array;
}

/**
 * CQR (Conformalized Quantile Regression) prediction interval (Definition 12).
 * Reproduces cqr_interval (notebook cell 12).
 * Verification (notebook): RE1, n_train = n_cal = 500, n_test = 5000, α = 0.1
 *   → CQR marginal ≈ 0.888, mean width ≈ 1.669, Q̂ ≈ -0.039, width ratio CQR/SC ≈ 0.964.
 *   Theorem 5.2 prediction at slope=0.6: E[σ(X)]/σ_+ = 0.5/0.8 = 0.625 (asymptotic limit).
 */
export function cqrIntervalPI(
  Xtr: ArrayLike<number>,
  Ytr: ArrayLike<number>,
  Xcal: ArrayLike<number>,
  Ycal: ArrayLike<number>,
  Xte: ArrayLike<number>,
  alpha: number,
  qrLambda: number = 0.001,
): CqrResultPI {
  const t0 = performance.now();
  const xTr = Float64Array.from(Xtr);
  const yTr = Float64Array.from(Ytr);
  const nCal = Xcal.length;
  const nTe = Xte.length;
  const xEval = new Float64Array(nCal + nTe);
  for (let i = 0; i < nCal; i++) xEval[i] = Xcal[i];
  for (let i = 0; i < nTe; i++) xEval[nCal + i] = Xte[i];
  const qLoEval = fitPredictQuantile(xTr, yTr, xEval, alpha / 2, qrLambda);
  const qHiEval = fitPredictQuantile(xTr, yTr, xEval, 1 - alpha / 2, qrLambda);
  const calScores = new Float64Array(nCal);
  for (let i = 0; i < nCal; i++) {
    const a = qLoEval[i] - Ycal[i];
    const b = Ycal[i] - qHiEval[i];
    calScores[i] = a > b ? a : b;
  }
  const Q = conformalQuantile(calScores, alpha);
  const qLoTest = Float64Array.from(qLoEval.subarray(nCal));
  const qHiTest = Float64Array.from(qHiEval.subarray(nCal));
  const lo = new Float64Array(nTe);
  const hi = new Float64Array(nTe);
  for (let i = 0; i < nTe; i++) {
    lo[i] = qLoTest[i] - Q;
    hi[i] = qHiTest[i] + Q;
  }
  return {
    lo,
    hi,
    Q,
    qLoTest,
    qHiTest,
    calScores,
    fitTimeMs: performance.now() - t0,
  };
}

export interface HlResultPI extends PIConstructionResult {
  /** Lower Walsh-average order statistic A_(w+1). */
  ALo: number;
  /** Upper Walsh-average order statistic A_(M−w). */
  AHi: number;
  /** Base predictor μ̂ evaluated on Xte. */
  muTest: Float64Array;
  /** All Walsh averages of calibration residuals, sorted ascending. Length M = n_cal(n_cal+1)/2. */
  walshSorted: Float64Array;
  /** Critical index w from hlCriticalIndexPI. */
  w: number;
  /** Number of Walsh averages M. */
  M: number;
}

/**
 * Hodges-Lehmann test-inversion prediction interval (Definition 11).
 * Reproduces hl_interval (notebook cell 10).
 * Verification (notebook): RE2, n_train = n_cal = 500, n_test = 5000, α = 0.1
 *   → A pair ≈ (-1.139, 1.433), HL band half-width ≈ 1.286, marginal ≈ 0.862, width ≈ 2.572.
 *
 * NOTE: HL coverage on RE2 single-draw is 0.862, well below 0.9. The §6 batch comparison
 * shows HL undercovering across all four scenarios (≈ 0.76-0.83). The Theorem 3 finite-
 * sample guarantee is conditional-on-test-point; the per-test-point batch average over
 * a fixed calibration set is a different statistic. See §6 narrative for the discussion.
 */
export function hlIntervalPI(
  Xtr: ArrayLike<number>,
  Ytr: ArrayLike<number>,
  Xcal: ArrayLike<number>,
  Ycal: ArrayLike<number>,
  Xte: ArrayLike<number>,
  alpha: number,
  ridgeLambda: number = 0.1,
): HlResultPI {
  const t0 = performance.now();
  const xTr = Float64Array.from(Xtr);
  const yTr = Float64Array.from(Ytr);
  const nCal = Xcal.length;
  const nTe = Xte.length;
  const xEval = new Float64Array(nCal + nTe);
  for (let i = 0; i < nCal; i++) xEval[i] = Xcal[i];
  for (let i = 0; i < nTe; i++) xEval[nCal + i] = Xte[i];
  const muEval = fitPredictRidge(xTr, yTr, xEval, ridgeLambda);
  const muCal = muEval.subarray(0, nCal);
  const muTest = Float64Array.from(muEval.subarray(nCal));
  const rCal = new Float64Array(nCal);
  for (let i = 0; i < nCal; i++) rCal[i] = Ycal[i] - muCal[i];
  const A = walshAveragesPI(rCal);
  A.sort();
  const { w, M } = hlCriticalIndexPI(nCal, alpha);
  const ALo = A[w];
  const AHi = A[M - 1 - w];
  const lo = new Float64Array(nTe);
  const hi = new Float64Array(nTe);
  for (let i = 0; i < nTe; i++) {
    lo[i] = muTest[i] + ALo;
    hi[i] = muTest[i] + AHi;
  }
  return {
    lo,
    hi,
    ALo,
    AHi,
    muTest,
    walshSorted: A,
    w,
    M,
    fitTimeMs: performance.now() - t0,
  };
}

// ─── §6 reference table (300-rep MC averages from notebook cell 14) ────────

/** Verified §6.2 4×4 table values from the notebook (n_rep = 300, n_train = n_cal = 500,
 *  n_test = 2000, α = 0.1). Used by ConstructionComparisonExplorer to render the live-vs-
 *  notebook comparison panel. Single source of truth for the topic's empirical claims. */
export const PI_REFERENCE_TABLE = {
  A: {
    split: { marg: 0.899, width: 1.660, cond: 0.056, ms: 0.4 },
    qr:    { marg: 0.896, width: 1.651, cond: 0.071, ms: 45.1 },
    cqr:   { marg: 0.900, width: 1.680, cond: 0.083, ms: 16.6 },
    hl:    { marg: 0.757, width: 1.180, cond: 0.080, ms: 4.3 },
  },
  B: {
    split: { marg: 0.899, width: 1.768, cond: 0.242, ms: 0.4 },
    qr:    { marg: 0.897, width: 1.657, cond: 0.110, ms: 44.3 },
    cqr:   { marg: 0.900, width: 1.686, cond: 0.115, ms: 16.6 },
    hl:    { marg: 0.789, width: 1.257, cond: 0.384, ms: 4.2 },
  },
  C: {
    split: { marg: 0.900, width: 2.978, cond: 0.058, ms: 0.4 },
    qr:    { marg: 0.896, width: 2.953, cond: 0.072, ms: 42.6 },
    cqr:   { marg: 0.901, width: 3.067, cond: 0.085, ms: 16.1 },
    hl:    { marg: 0.817, width: 2.240, cond: 0.084, ms: 4.2 },
  },
  D: {
    split: { marg: 0.899, width: 1.146, cond: 0.064, ms: 0.4 },
    qr:    { marg: 0.895, width: 1.133, cond: 0.072, ms: 44.3 },
    cqr:   { marg: 0.901, width: 1.183, cond: 0.077, ms: 16.6 },
    hl:    { marg: 0.827, width: 0.928, cond: 0.083, ms: 4.2 },
  },
} as const;

export type PIScenarioKey = keyof typeof PI_REFERENCE_TABLE;
export type PIConstructionKey = keyof typeof PI_REFERENCE_TABLE['A'];

// ─── §6 four scenarios ─────────────────────────────────────────────────────

/** Scenario A — homoscedastic Gaussian. Y | X = x ~ N(sin(x), 0.5²), X ~ Uniform(-3, 3). */
export function scenarioAPI(
  n: number,
  rng: () => number,
): { X: Float64Array; Y: Float64Array } {
  const gauss = gaussianSampler(rng);
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = -3 + 6 * rng();
    X[i] = x;
    Y[i] = Math.sin(x) + 0.5 * gauss();
  }
  return { X, Y };
}

/** Scenario B = Running Example 1 — heteroscedastic Gaussian. */
export function scenarioBPI(
  n: number,
  rng: () => number,
): { X: Float64Array; Y: Float64Array } {
  const { X, Y } = generateHeteroscedastic(n, rng);
  return { X, Y };
}

/** Scenario C = Running Example 2 — heavy-tailed location-shift. */
export function scenarioCPI(
  n: number,
  rng: () => number,
): { X: Float64Array; Y: Float64Array } {
  const { X, Y } = generateHeavyTailedLocation(n, rng);
  return { X, Y };
}

/** Scenario D — contaminated noise (95/5 mixture of N(·, 0.3²) and N(·, 2²)). */
export function scenarioDPI(
  n: number,
  rng: () => number,
): { X: Float64Array; Y: Float64Array } {
  const gauss = gaussianSampler(rng);
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = -3 + 6 * rng();
    const sigma = rng() < 0.05 ? 2.0 : 0.3;
    X[i] = x;
    Y[i] = Math.sin(x) + sigma * gauss();
  }
  return { X, Y };
}

// ─── End prediction-intervals helpers ──────────────────────────────────────

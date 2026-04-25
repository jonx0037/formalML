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
  const p = degree + 1; // bias + x + x^2 + x^3
  const n = xTrain.length;
  // Build feature matrix and standardise its columns to similar magnitudes.
  // (Without scaling, x^3 dominates the gradient and convergence is slow.)
  const Phi: number[][] = new Array(n);
  for (let i = 0; i < n; i++) Phi[i] = polyFeatures(xTrain[i], degree, true);
  // Compute per-column standard deviations (skip bias column at index 0).
  const colStd: number[] = new Array(p).fill(1);
  for (let c = 1; c < p; c++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += Phi[i][c];
    mean /= n;
    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const d = Phi[i][c] - mean;
      varSum += d * d;
    }
    const sd = Math.sqrt(varSum / Math.max(n - 1, 1));
    colStd[c] = sd > 1e-12 ? sd : 1;
  }
  // Scale features; β_scaled is what we optimise.
  const PhiScaled: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    PhiScaled[i] = new Array(p);
    PhiScaled[i][0] = Phi[i][0]; // bias unchanged
    for (let c = 1; c < p; c++) PhiScaled[i][c] = Phi[i][c] / colStd[c];
  }
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
  const h = Math.max(0.05 * yStd, 1e-3);
  // Accelerated gradient descent on β_scaled.
  // Loss: (1/n) Σ smoothed_check(y_i - x_i^T β; tau, h) + (λ/2) ||β||²  (skip bias in penalty)
  // Smoothed check ρ_τ(r) = τ r + h log(1 + exp(-r/h))   (= max(τr, (τ-1)r) + softplus correction)
  // Gradient w.r.t. r: τ - 1 / (1 + exp(r/h))    (sigmoid-shaped)
  const beta = new Array(p).fill(0);
  // Initialise bias to median of y for stability.
  const ySorted = Array.from(yTrain).sort((a, b) => a - b);
  beta[0] = ySorted[Math.floor(n / 2)];
  let momentum = beta.slice();
  // Lipschitz estimate of smoothed-check + L2 ridge:
  //  - ∇_β r_i = -x_i, gradient of smoothed-check w.r.t. r is bounded by 1/(2h) (sigmoid slope at 0)
  //  - max ||x_i||² ~ p*max(|x|^6) ≈ p · 64 for x in [-2, 2]; we conservatively use the estimate.
  //  L = max ||x||² / (4 h n) + λ. With scaled features, ||x_scaled||² ≈ p ≈ 4, so L ~ 1/(h n) + λ.
  let maxRowSq = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < p; c++) s += PhiScaled[i][c] * PhiScaled[i][c];
    if (s > maxRowSq) maxRowSq = s;
  }
  const L = maxRowSq / (4 * h * n) + lambda + 1e-6;
  const stepSize = 1 / L;
  const maxIter = 800;
  let tPrev = 1;
  let prevBeta = beta.slice();
  for (let iter = 0; iter < maxIter; iter++) {
    // Gradient at momentum point
    const grad = new Array(p).fill(0);
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let c = 0; c < p; c++) pred += PhiScaled[i][c] * momentum[c];
      const r = yTrain[i] - pred;
      // ρ_τ(r) = τr + h·log(1 + exp(-r/h))
      // ∂ρ/∂r = τ − 1/(1 + exp(r/h));   approaches τ as r→+∞ and τ−1 as r→−∞.
      // Numerically stable form (avoid overflow when r/h is large):
      const z = r / h;
      const sigmoidZ = z >= 0
        ? 1 / (1 + Math.exp(-z))
        : Math.exp(z) / (1 + Math.exp(z));
      const grad_r = tau - (1 - sigmoidZ); // = τ − 1/(1 + exp(r/h)) = τ − 1 + sigmoid(r/h)
      const grad_pred = -grad_r; // ∂ρ̃/∂pred = -∂ρ̃/∂r
      // Add to gradient w.r.t. β
      for (let c = 0; c < p; c++) grad[c] += (PhiScaled[i][c] * grad_pred) / n;
    }
    // L2 ridge on non-bias coefficients
    for (let c = 1; c < p; c++) grad[c] += lambda * momentum[c];
    // Step
    const newBeta = new Array(p);
    for (let c = 0; c < p; c++) newBeta[c] = momentum[c] - stepSize * grad[c];
    // Nesterov momentum update
    const tCurr = (1 + Math.sqrt(1 + 4 * tPrev * tPrev)) / 2;
    const w = (tPrev - 1) / tCurr;
    for (let c = 0; c < p; c++) momentum[c] = newBeta[c] + w * (newBeta[c] - prevBeta[c]);
    // Convergence check (max coefficient delta)
    let maxDelta = 0;
    for (let c = 0; c < p; c++) {
      const d = Math.abs(newBeta[c] - prevBeta[c]);
      if (d > maxDelta) maxDelta = d;
    }
    prevBeta = newBeta;
    tPrev = tCurr;
    if (maxDelta < 1e-7 && iter > 10) break;
  }
  // Predict: rescale β back to original feature space
  const m = xEval.length;
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const phi = polyFeatures(xEval[i], degree, true);
    let pred = prevBeta[0]; // bias
    for (let c = 1; c < p; c++) pred += (phi[c] / colStd[c]) * prevBeta[c];
    out[i] = pred;
  }
  return out;
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

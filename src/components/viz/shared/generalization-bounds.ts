// =============================================================================
// generalization-bounds.ts
//
// Shared math primitives for the generalization-bounds topic's viz components.
// Ported from notebooks/generalization-bounds/01_generalization_bounds.ipynb §§2–11.
// In-browser TS for all the §3–§11 computations (closed-form bounds + light Monte
// Carlo with commit-on-release sliders); the §10 MarginBoundDemo and §12
// VacuousnessDemo consume precomputed JSON from src/data/sampleData/.
//
// All exports are pure functions — deterministic outputs for a given seed via
// mulberry32 (re-exported from normalizing-flows.ts so viz components don't
// need to know which sister module owns it).
//
// Source-of-truth notebook: notebooks/generalization-bounds/01_generalization_bounds.ipynb
// =============================================================================

import { mulberry32 } from './normalizing-flows';

export { mulberry32 };

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export type Sample = { X: Float64Array; Y: Int8Array };
export type RademacherEstimate = { mean: number; se: number };

// -----------------------------------------------------------------------------
// §3 Closed-form bounds (no MC)
// -----------------------------------------------------------------------------

/** Theorem 1 (finite-class generalization bound). */
export function finiteClassBound(N: number, n: number, delta: number, M: number = 1): number {
  return M * Math.sqrt(Math.log((2 * N) / delta) / (2 * n));
}

/** Corollary 1: sample complexity n ≥ M² log(2|H|/δ) / (2ε²). */
export function finiteClassSampleComplexity(N: number, epsilon: number, delta: number, M: number = 1): number {
  return (M * M * Math.log((2 * N) / delta)) / (2 * epsilon * epsilon);
}

/** Massart's finite-class Rademacher upper bound for binary {-1, +1} classes. */
export function massartBound(N: number, n: number): number {
  return Math.sqrt((2 * Math.log(N)) / n);
}

/** §7 Corollary 3 (canonical Rademacher generalization bound). */
export function canonicalBound(rad: number, n: number, delta: number, M: number = 1): number {
  return rad + 3 * M * Math.sqrt(Math.log(2 / delta) / (2 * n));
}

// -----------------------------------------------------------------------------
// §4 DKW envelope + ECDF / KS distance
// -----------------------------------------------------------------------------

/** DKW envelope: with probability ≥ 1-δ, sup_t |F_n(t) - F(t)| ≤ √(log(2/δ)/(2n)). */
export function dkwEnvelope(n: number, delta: number): number {
  return Math.sqrt(Math.log(2 / delta) / (2 * n));
}

/** Empirical CDF F̂_n at a query grid; returns step-function values in [0, 1]. */
export function empiricalCDF(samples: Float64Array, queryGrid: Float64Array): Float64Array {
  const sorted = samples.slice().sort();
  const n = sorted.length;
  const out = new Float64Array(queryGrid.length);
  for (let i = 0; i < queryGrid.length; i++) {
    const t = queryGrid[i];
    let lo = 0;
    let hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sorted[mid] <= t) lo = mid + 1;
      else hi = mid;
    }
    out[i] = lo / n;
  }
  return out;
}

/** KS distance sup_t |F̂_n(t) - F(t)|. */
export function ksDistance(samples: Float64Array, trueCdf: (t: number) => number): number {
  const sorted = samples.slice().sort();
  const n = sorted.length;
  let maxDev = 0;
  for (let i = 0; i < n; i++) {
    const Fi = trueCdf(sorted[i]);
    const upper = Math.abs((i + 1) / n - Fi);
    const lower = Math.abs(Fi - i / n);
    if (upper > maxDev) maxDev = upper;
    if (lower > maxDev) maxDev = lower;
  }
  return maxDev;
}

// -----------------------------------------------------------------------------
// §2 Threshold-classifier toy problem
// -----------------------------------------------------------------------------

/** Sample n iid (X, Y) with X ~ U[0,1], Y = 1[X >= tauStar] XOR Bernoulli(eta). */
export function sampleThresholdProblem(
  n: number,
  eta: number,
  tauStar: number,
  rng: () => number,
): Sample {
  const X = new Float64Array(n);
  const Y = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    const x = rng();
    X[i] = x;
    const yClean = x >= tauStar ? 1 : 0;
    const flip = rng() < eta ? 1 : 0;
    Y[i] = ((yClean ^ flip) & 1) as 0 | 1;
  }
  return { X, Y };
}

/** ERM over threshold classifiers; returns the tau in [0,1] minimizing empirical 0-1 risk. */
export function ermThreshold(X: Float64Array, Y: Int8Array): number {
  const n = X.length;
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  const arr = order.slice().sort((a, b) => X[a] - X[b]);
  const Xs = new Float64Array(n);
  const Ys = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    Xs[i] = X[arr[i]];
    Ys[i] = Y[arr[i]];
  }
  // Candidate thresholds: 0, midpoints between consecutive sorted X, 1.
  const candidates = new Float64Array(n + 1);
  candidates[0] = 0;
  for (let i = 0; i < n - 1; i++) candidates[i + 1] = (Xs[i] + Xs[i + 1]) / 2;
  candidates[n] = 1;
  // Prefix sums of Ys to count errors per candidate in O(1).
  const prefY = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) prefY[i + 1] = prefY[i] + Ys[i];
  let bestErr = Infinity;
  let bestTau = 0.5;
  for (let k = 0; k <= n; k++) {
    const c = candidates[k];
    let lo = 0;
    let hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (Xs[mid] >= c) hi = mid;
      else lo = mid + 1;
    }
    const onesRight = prefY[n] - prefY[lo];
    const zerosRight = n - lo - onesRight;
    const onesLeft = prefY[lo];
    const err = zerosRight + onesLeft;
    if (err < bestErr) {
      bestErr = err;
      bestTau = c;
    }
  }
  return bestTau;
}

/** Closed-form true risk of h_tau under X~U[0,1] with label-flip rate eta. */
export function trueRiskThreshold(tau: number, eta: number, tauStar: number): number {
  const width = Math.abs(tau - tauStar);
  return width * (1 - eta) + (1 - width) * eta;
}

// -----------------------------------------------------------------------------
// §5 Empirical Rademacher complexity (Monte Carlo)
// -----------------------------------------------------------------------------

/**
 * Minimal class-evaluation matrix for the threshold class.
 *
 * Threshold class produces exactly n+1 distinct labelings on a sample of size n.
 * Returns a flat (n+1)*n Int8Array in row-major order with entries in {0, 1}.
 */
export function thresholdClassMatrixMinimal(X: Float64Array): Int8Array {
  const n = X.length;
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  const arr = order.slice().sort((a, b) => X[a] - X[b]);
  const H = new Int8Array((n + 1) * n);
  for (let k = 0; k <= n; k++) {
    const rowOffset = k * n;
    for (let i = k; i < n; i++) H[rowOffset + arr[i]] = 1;
  }
  return H;
}

/**
 * Monte-Carlo estimate of empirical Rademacher complexity.
 *
 * classMatrix: flat (numHypotheses * n) Int8Array in row-major order, entries in {0, 1}.
 * Internally converted to {-1, +1} for the Rademacher inner product.
 */
export function empiricalRademacherMC(
  classMatrix: Int8Array,
  numHypotheses: number,
  n: number,
  B: number,
  rng: () => number,
): RademacherEstimate {
  if (classMatrix.length !== numHypotheses * n) {
    throw new Error(`classMatrix length ${classMatrix.length} != ${numHypotheses}*${n}`);
  }
  // classMatrix entries are in {0, 1}; the {-1, +1} version is h' = 2h − 1.
  // The Rademacher inner product becomes Σ σ_i (2 h_i − 1) = 2 (Σ σ_i h_i) − Σ σ_i,
  // where Σ σ_i is constant for all hypotheses in a replicate.  Computing the
  // dot product directly against the {0, 1} matrix avoids allocating a parallel
  // {-1, +1} Float64Array of size |H|·n — significant GC relief in MC loops.
  const sigma = new Int8Array(n);
  let sum = 0;
  let sumSq = 0;
  for (let b = 0; b < B; b++) {
    let sSigma = 0;
    for (let i = 0; i < n; i++) {
      const s = rng() < 0.5 ? -1 : 1;
      sigma[i] = s;
      sSigma += s;
    }
    let supVal = -Infinity;
    for (let k = 0; k < numHypotheses; k++) {
      const rowOff = k * n;
      let dot = 0;
      for (let i = 0; i < n; i++) {
        if (classMatrix[rowOff + i]) dot += sigma[i];
      }
      const v = (2 * dot - sSigma) / n;
      if (v > supVal) supVal = v;
    }
    sum += supVal;
    sumSq += supVal * supVal;
  }
  const mean = sum / B;
  const variance = Math.max(0, sumSq / B - mean * mean);
  const se = Math.sqrt(variance / B);
  return { mean, se };
}

// -----------------------------------------------------------------------------
// §6 Symmetrization chain stats
// -----------------------------------------------------------------------------

/**
 * Per-replicate suprema for §6's symmetrization-chain visualization on thresholds.
 *
 * For each of B replicates: draw S, S' iid U[0,1]^n and σ ∈ {-1,+1}^n; compute
 *   raw   = sup_τ (Pf − P_n f)         on S
 *   ghost = sup_τ (P_{n'} f − P_n f)   on (S, S')
 *   sym   = sup_τ (1/n) Σ σ_i (h(X'_i) − h(X_i))
 *   rad   = sup_τ (1/n) Σ σ_i h_τ(X_i)   on S alone, {-1, +1}-coded
 */
export function symmetrizationStats(
  n: number,
  B: number,
  rng: () => number,
): {
  rawSup: Float64Array;
  ghostPairSup: Float64Array;
  sigmaSymSup: Float64Array;
  empRad: Float64Array;
} {
  const rawSup = new Float64Array(B);
  const ghostPairSup = new Float64Array(B);
  const sigmaSymSup = new Float64Array(B);
  const empRad = new Float64Array(B);
  const sigma = new Int8Array(n);

  for (let b = 0; b < B; b++) {
    const X = new Float64Array(n);
    const Xp = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      X[i] = rng();
      Xp[i] = rng();
    }
    for (let i = 0; i < n; i++) sigma[i] = rng() < 0.5 ? -1 : 1;
    // Threshold grid on S: [0, X_(1), ..., X_(n)]
    const sortedX = X.slice().sort();
    const grid = new Float64Array(n + 1);
    grid[0] = 0;
    for (let i = 0; i < n; i++) grid[i + 1] = sortedX[i];

    let rawMax = -Infinity;
    let ghostMax = -Infinity;
    let symMax = -Infinity;
    let radMax = -Infinity;
    for (let k = 0; k <= n; k++) {
      const tau = grid[k];
      let cntS = 0;
      let cntSp = 0;
      let symAcc = 0;
      let radAcc = 0;
      for (let i = 0; i < n; i++) {
        const hX01 = X[i] >= tau ? 1 : 0;
        const hXp01 = Xp[i] >= tau ? 1 : 0;
        cntS += hX01;
        cntSp += hXp01;
        symAcc += sigma[i] * (hXp01 - hX01);
        radAcc += sigma[i] * (X[i] >= tau ? 1 : -1);
      }
      const pnS = cntS / n;
      const pnSp = cntSp / n;
      const pf = 1 - tau;
      const rawVal = pf - pnS;
      const ghostVal = pnSp - pnS;
      const symVal = symAcc / n;
      const radVal = radAcc / n;
      if (rawVal > rawMax) rawMax = rawVal;
      if (ghostVal > ghostMax) ghostMax = ghostVal;
      if (symVal > symMax) symMax = symVal;
      if (radVal > radMax) radMax = radVal;
    }
    rawSup[b] = rawMax;
    ghostPairSup[b] = ghostMax;
    sigmaSymSup[b] = symMax;
    empRad[b] = radMax;
  }
  return { rawSup, ghostPairSup, sigmaSymSup, empRad };
}

// -----------------------------------------------------------------------------
// §8 Dudley's entropy integral (linear classes)
// -----------------------------------------------------------------------------

/** log N(ε, F, L_2) ≤ d log(3B/ε) for linear classes ||w|| ≤ B in R^d. */
export function linearCoveringLogN(epsilon: number, d: number, B: number): number {
  if (epsilon >= 3 * B) return 0;
  return d * Math.log((3 * B) / epsilon);
}

/**
 * Dudley's entropy integral via trapezoidal rule on [Dn/mSteps, Dn].
 * Computes (12/√n) ∫ √(log N(ε)) dε.
 */
export function dudleyIntegral(
  n: number,
  d: number,
  B: number,
  Dn: number,
  mSteps: number = 100,
): number {
  if (mSteps < 2) throw new Error('mSteps must be >= 2');
  const eps0 = Dn / mSteps;
  const h = (Dn - eps0) / (mSteps - 1);
  let total = 0;
  for (let i = 0; i < mSteps; i++) {
    const eps = eps0 + i * h;
    const logN = linearCoveringLogN(eps, d, B);
    const f = Math.sqrt(Math.max(0, logN));
    const weight = i === 0 || i === mSteps - 1 ? 0.5 : 1;
    total += weight * f;
  }
  return (12 / Math.sqrt(n)) * total * h;
}

// -----------------------------------------------------------------------------
// §9 Local Rademacher complexity (fixed point)
// -----------------------------------------------------------------------------

/**
 * Empirical Rademacher complexity of the local class {h_τ : P_n (h_τ − h_*)² ≤ r}
 * for the threshold class, evaluated on a fresh sample of size n.
 */
export function localRademacherThresholdClass(
  n: number,
  r: number,
  B: number,
  rng: () => number,
  tauStar: number = 0.5,
): number {
  const X = new Float64Array(n);
  for (let i = 0; i < n; i++) X[i] = rng();
  X.sort();
  const grid = new Float64Array(n + 1);
  grid[0] = 0;
  for (let i = 0; i < n; i++) grid[i + 1] = X[i];

  const eligible: number[] = [];
  for (let k = 0; k <= n; k++) {
    const tau = grid[k];
    let cntDis = 0;
    for (let i = 0; i < n; i++) {
      const hTau = X[i] >= tau ? 1 : 0;
      const hStar = X[i] >= tauStar ? 1 : 0;
      if (hTau !== hStar) cntDis++;
    }
    if (cntDis / n <= r) eligible.push(k);
  }
  if (eligible.length === 0) return 0;

  // Inline the Rademacher MC: for each eligible tau, the indicator h_tau(X_i) is
  // (X_i ≥ tau).  Use the 2·dot − sSigma identity to avoid allocating the
  // |eligible| × n class matrix.  Saves O(eN·n) Int8Array allocation per call,
  // which matters because localRademacherFixedPoint bisects 14× and the n
  // threshold-grid times the eligibility-pass keeps eN close to n+1 in practice.
  const sigma = new Int8Array(n);
  let sum = 0;
  for (let b = 0; b < B; b++) {
    let sSigma = 0;
    for (let i = 0; i < n; i++) {
      const s = rng() < 0.5 ? -1 : 1;
      sigma[i] = s;
      sSigma += s;
    }
    let supVal = -Infinity;
    for (let kIdx = 0; kIdx < eligible.length; kIdx++) {
      const tau = grid[eligible[kIdx]];
      let dot = 0;
      for (let i = 0; i < n; i++) if (X[i] >= tau) dot += sigma[i];
      const v = (2 * dot - sSigma) / n;
      if (v > supVal) supVal = v;
    }
    sum += supVal;
  }
  return sum / B;
}

/** Bisection on r = R̂_S(F_r) + log(1/δ)/n. Returns r*. */
export function localRademacherFixedPoint(
  n: number,
  delta: number,
  rng: () => number,
  B: number = 300,
  iterations: number = 20,
): number {
  const logTerm = Math.log(1 / delta) / n;
  let lo = logTerm;
  let hi = 1.0;
  for (let i = 0; i < iterations; i++) {
    const mid = (lo + hi) / 2;
    const rhs = localRademacherThresholdClass(n, mid, B, rng) + logTerm;
    if (mid > rhs) hi = mid;
    else lo = mid;
  }
  return (lo + hi) / 2;
}

// -----------------------------------------------------------------------------
// §11 Algorithmic stability — ridge regression
// -----------------------------------------------------------------------------

/** Cholesky-based ridge solve: (X^T X + n λ I) w = X^T y. */
export function ridgeSolve(
  X: Float64Array,
  y: Float64Array,
  n: number,
  d: number,
  lambda: number,
): Float64Array {
  const M = new Float64Array(d * d);
  const b = new Float64Array(d);
  // X^T X is symmetric — compute the upper triangle (j <= i) and mirror.
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[k * d + i] * X[k * d + j];
      M[i * d + j] = s;
      M[j * d + i] = s;
    }
    M[i * d + i] += n * lambda;
    let yb = 0;
    for (let k = 0; k < n; k++) yb += X[k * d + i] * y[k];
    b[i] = yb;
  }
  const L = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i * d + j];
      for (let k = 0; k < j; k++) s -= L[i * d + k] * L[j * d + k];
      if (i === j) {
        if (s <= 0) throw new Error(`Cholesky: non-SPD at (${i},${j}), s=${s}`);
        L[i * d + j] = Math.sqrt(s);
      } else {
        L[i * d + j] = s / L[j * d + j];
      }
    }
  }
  const z = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * d + k] * z[k];
    z[i] = s / L[i * d + i];
  }
  const w = new Float64Array(d);
  for (let i = d - 1; i >= 0; i--) {
    let s = z[i];
    for (let k = i + 1; k < d; k++) s -= L[k * d + i] * w[k];
    w[i] = s / L[i * d + i];
  }
  return w;
}

/**
 * Estimate β-stability of ridge regression via sample-swap perturbation.
 *
 * `unif` is a U[0,1] RNG (for index picking); `gauss` is a N(0,1) sampler
 * (for synthetic features and noise).
 */
export function ridgeStabilityBeta(
  n: number,
  d: number,
  lambda: number,
  nSwaps: number,
  unif: () => number,
  gauss: () => number,
  signalScale: number = 1.0,
  noiseScale: number = 0.5,
  nTest: number = 200,
): number {
  const X = new Float64Array(n * d);
  const y = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let yk = 0;
    for (let j = 0; j < d; j++) {
      const v = gauss();
      X[k * d + j] = v;
      yk += (v * signalScale) / Math.sqrt(d);
    }
    y[k] = yk + noiseScale * gauss();
  }
  const w = ridgeSolve(X, y, n, d, lambda);

  const Xtest = new Float64Array(nTest * d);
  const yTest = new Float64Array(nTest);
  for (let k = 0; k < nTest; k++) {
    let yk = 0;
    for (let j = 0; j < d; j++) {
      const v = gauss();
      Xtest[k * d + j] = v;
      yk += (v * signalScale) / Math.sqrt(d);
    }
    yTest[k] = yk + noiseScale * gauss();
  }
  const lossesOrig = new Float64Array(nTest);
  for (let k = 0; k < nTest; k++) {
    let pred = 0;
    for (let j = 0; j < d; j++) pred += Xtest[k * d + j] * w[j];
    lossesOrig[k] = (yTest[k] - pred) ** 2;
  }

  let maxDiff = 0;
  const Xp = new Float64Array(n * d);
  const yp = new Float64Array(n);
  for (let swap = 0; swap < nSwaps; swap++) {
    Xp.set(X);
    yp.set(y);
    const i = Math.min(n - 1, Math.floor(unif() * n));
    let yPrime = 0;
    for (let j = 0; j < d; j++) {
      const v = gauss();
      Xp[i * d + j] = v;
      yPrime += (v * signalScale) / Math.sqrt(d);
    }
    yp[i] = yPrime + noiseScale * gauss();
    const wp = ridgeSolve(Xp, yp, n, d, lambda);
    for (let k = 0; k < nTest; k++) {
      let pred = 0;
      for (let j = 0; j < d; j++) pred += Xtest[k * d + j] * wp[j];
      const lossNew = (yTest[k] - pred) ** 2;
      const diff = Math.abs(lossesOrig[k] - lossNew);
      if (diff > maxDiff) maxDiff = diff;
    }
  }
  return maxDiff;
}

/** Bousquet–Elisseeff deviation term: β + (2nβ + M) √(log(1/δ)/(2n)). */
export function bousquetElisseeffDeviation(beta: number, n: number, delta: number, M: number = 1): number {
  return beta + (2 * n * beta + M) * Math.sqrt(Math.log(1 / delta) / (2 * n));
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/** Box-Muller standard-normal sampler from a uniform RNG. */
export function gaussianFrom(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

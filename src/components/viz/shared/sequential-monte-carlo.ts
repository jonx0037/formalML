// =============================================================================
// sequential-monte-carlo.ts
//
// Shared math/types/palette/sampler module for the T5 sequential-monte-carlo
// topic. Mirrors notebooks/sequential-monte-carlo/01_sequential_monte_carlo.ipynb
// across the SMC skeleton, the four resampling schemes, the propagation
// kernels, the bootstrap and auxiliary particle filters, the adaptive
// temperature schedule, the unbiased log-evidence estimator, and the banana
// log-density.
//
// Three independent groups of functions:
//
//   1. Core SMC primitives (§§3, 4, 5, 7, 9, 10):
//        - Log-space helpers (logSumExp, logSpaceESS, normalizeLogW)
//        - Four resampling schemes via searchsorted-on-cumsum (never
//          random.choice — project-wide hard rule)
//        - Geometric-path incremental log-weight
//        - Adaptive-schedule bisection
//        - Cloud mean and covariance
//        - IMH cloud-fit primitives (proposal, acceptance log-ratio)
//        - RWM step
//
//   2. Target log-densities used across §§3-11:
//        - 1D Gaussian, 1D mixture, 2D anisotropic Gaussian
//        - Banana (§9, §10) — matches RMHMC topic's §10 demo
//        - Bayesian Poisson rate (§7)
//        - Gaussian-mixture model (§8)
//        - Stochastic-volatility step (§6)
//
//   3. End-to-end SMC harnesses (used by viz + verify suite):
//        - smcGaussianBridge — §3.4 reference run
//        - smcSamplerGeneric — generic geometric-annealing sampler
//        - bootstrapParticleFilter — §6.4 SV demo
//        - auxiliaryParticleFilter — §6.4 APF
//        - smcAdaptiveSampler — §9 ESS-driven adaptive schedule
//
// Conventions:
//   - All matrices are number[][] row-major (matches gaussian-processes.ts).
//   - All vectors are number[].
//   - All RNGs are () => number returning uniforms in [0, 1), seedable via
//     mulberry32. Standard normals via gaussianPair.
//   - Pure functions, no React / D3 / DOM dependencies.
//   - Log-space discipline enforced — all weight updates additive in log
//     space, normalize via logSumExp, ESS via -logSumExp(2 * log_w_norm).
//
// Brief: docs/plans/formalml-sequential-monte-carlo-handoff-brief.md
// Notebook: notebooks/sequential-monte-carlo/01_sequential_monte_carlo.ipynb
// =============================================================================

import { mulberry32, gaussianPair } from './bayesian-ml';
import {
  choleskyFactor,
  matVec,
  solveLowerTriangular,
  solveUpperTriangularT,
  addDiagonal,
} from './gaussian-processes';

export { mulberry32, gaussianPair };

// =============================================================================
// PALETTE — matches the notebook's matplotlib PALETTE
// =============================================================================

export const paletteSMC = {
  cloud: '#2c5282', //    particle cloud / IS estimate
  target: '#c53030', //   target / truth
  proposal: '#4a5568', // proposal / baseline
  accent: '#dd6b20', //   secondary
  muted: '#a0aec0', //    gridlines, secondary text
  // scheme-specific tints used by ResamplingSchemeComparator
  multinomial: '#4a5568',
  residual: '#dd6b20',
  stratified: '#2c5282',
  systematic: '#c53030',
} as const;

export type SMCColorKey = keyof typeof paletteSMC;

export const SMC_SEED = 20260518;

// =============================================================================
// 1. RNG + GAUSSIAN HELPERS
// =============================================================================

/** Stateful gaussian sampler that keeps the second value of each Box-Muller
 * pair in reserve, so adjacent calls cost a single uniform on average. */
export function makeGaussian(rng: () => number): () => number {
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    const [a, b] = gaussianPair(rng);
    cached = b;
    return a;
  };
}

/** Draw n iid standard normals as a fresh number[]. */
export function drawNormals(rng: () => number, n: number): number[] {
  const g = makeGaussian(rng);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = g();
  return out;
}

// =============================================================================
// 2. LOG-SPACE HELPERS
// =============================================================================

/** logSumExp on a number[]. Returns -Infinity on empty input.
 * Uses the shift-then-exp trick: max + log Σ exp(x - max). */
export function logSumExp(arr: readonly number[]): number {
  if (arr.length === 0) return -Infinity;
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  if (!isFinite(m)) return m;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += Math.exp(arr[i] - m);
  return m + Math.log(s);
}

/** Normalize a log-weight array so Σ exp(log_w_norm) = 1.
 * Returns a fresh array. */
export function normalizeLogW(logW: readonly number[]): number[] {
  const lse = logSumExp(logW);
  const out = new Array<number>(logW.length);
  for (let i = 0; i < logW.length; i++) out[i] = logW[i] - lse;
  return out;
}

/** Effective sample size from log-weights: 1 / Σ w_i^2.
 * Equivalent to exp(-logSumExp(2 * log_w_norm)). */
export function logSpaceESS(logW: readonly number[]): number {
  const lwNorm = normalizeLogW(logW);
  const twice = new Array<number>(lwNorm.length);
  for (let i = 0; i < lwNorm.length; i++) twice[i] = 2 * lwNorm[i];
  return Math.exp(-logSumExp(twice));
}

/** Running log-Ẑ_t = logSumExp(log_w) - log N (Theorem 4 estimator). */
export function logZFromLogW(logW: readonly number[]): number {
  return logSumExp(logW) - Math.log(logW.length);
}

// =============================================================================
// 3. RESAMPLING SCHEMES
//
// All four schemes go through searchsorted on a cumulative-sum array. The
// project-wide hard rule is: never use a categorical sampler that allocates
// per call — always searchsorted on a hoisted cumw buffer.
// =============================================================================

/** Searchsorted: for each value u in queries, return min{j : cum[j] >= u}.
 * Returns indices clamped to [0, cum.length - 1]. */
export function searchSorted(cum: readonly number[], queries: readonly number[]): number[] {
  const n = cum.length;
  const out = new Array<number>(queries.length);
  for (let q = 0; q < queries.length; q++) {
    const u = queries[q];
    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cum[mid] < u) lo = mid + 1;
      else hi = mid;
    }
    out[q] = lo;
  }
  return out;
}

/** Cumulative sum of a normalized weight array. */
function cumsumNormalized(wNorm: readonly number[]): number[] {
  const n = wNorm.length;
  const out = new Array<number>(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += wNorm[i];
    out[i] = s;
  }
  // numerical guard — clamp final to 1.0
  out[n - 1] = 1;
  return out;
}

/** Multinomial resampling — N iid uniforms over [0,1) mapped through cumw. */
export function multinomialResample(
  wNorm: readonly number[],
  rng: () => number,
  N?: number
): number[] {
  const n = N ?? wNorm.length;
  const cum = cumsumNormalized(wNorm);
  const u = new Array<number>(n);
  for (let i = 0; i < n; i++) u[i] = rng();
  return searchSorted(cum, u);
}

/** Systematic resampling (Kitagawa 1996): one base uniform shifted by k/N.
 * Variance-minimal among the four schemes for bounded test functions. */
export function systematicResample(
  wNorm: readonly number[],
  rng: () => number,
  N?: number
): number[] {
  const n = N ?? wNorm.length;
  const cum = cumsumNormalized(wNorm);
  const u0 = rng();
  const u = new Array<number>(n);
  for (let k = 0; k < n; k++) u[k] = (u0 + k) / n;
  return searchSorted(cum, u);
}

/** Stratified resampling (Kitagawa 1996): one uniform per stratum k/N. */
export function stratifiedResample(
  wNorm: readonly number[],
  rng: () => number,
  N?: number
): number[] {
  const n = N ?? wNorm.length;
  const cum = cumsumNormalized(wNorm);
  const u = new Array<number>(n);
  for (let k = 0; k < n; k++) u[k] = (rng() + k) / n;
  return searchSorted(cum, u);
}

/** Residual resampling (Liu-Chen 1998): deterministic ⌊N w_i⌋ offspring plus
 * a multinomial stage over the residuals (N w_i - ⌊N w_i⌋). */
export function residualResample(
  wNorm: readonly number[],
  rng: () => number,
  N?: number
): number[] {
  const n = N ?? wNorm.length;
  const k = wNorm.length;
  const counts = new Array<number>(k);
  let detCount = 0;
  for (let i = 0; i < k; i++) {
    counts[i] = Math.floor(n * wNorm[i]);
    detCount += counts[i];
  }
  const out = new Array<number>(n);
  let pos = 0;
  for (let i = 0; i < k; i++) {
    for (let r = 0; r < counts[i]; r++) out[pos++] = i;
  }
  const nRes = n - detCount;
  if (nRes <= 0) return out;
  // residual weights
  const res = new Array<number>(k);
  let rSum = 0;
  for (let i = 0; i < k; i++) {
    const r = n * wNorm[i] - counts[i];
    res[i] = r;
    rSum += r;
  }
  if (rSum <= 0) {
    // pathological: residuals all zero (exact integer counts). Fill with last.
    for (let r = 0; r < nRes; r++) out[pos++] = k - 1;
    return out;
  }
  for (let i = 0; i < k; i++) res[i] /= rSum;
  const cumRes = cumsumNormalized(res);
  const queries = new Array<number>(nRes);
  for (let i = 0; i < nRes; i++) queries[i] = rng();
  const resIdx = searchSorted(cumRes, queries);
  for (let i = 0; i < nRes; i++) out[pos++] = resIdx[i];
  return out;
}

export type ResamplingScheme = 'multinomial' | 'systematic' | 'stratified' | 'residual';

export function resample(
  scheme: ResamplingScheme,
  wNorm: readonly number[],
  rng: () => number,
  N?: number
): number[] {
  switch (scheme) {
    case 'multinomial':
      return multinomialResample(wNorm, rng, N);
    case 'systematic':
      return systematicResample(wNorm, rng, N);
    case 'stratified':
      return stratifiedResample(wNorm, rng, N);
    case 'residual':
      return residualResample(wNorm, rng, N);
  }
}

// =============================================================================
// 4. GEOMETRIC-PATH REWEIGHT (Proposition 3)
// =============================================================================

/** Closed-form incremental log-weight along a geometric annealing path:
 *   log α_t = (β_{t+1} - β_t) · log p(y | θ).
 * Prior cancels exactly — see brief §7.2 Proposition 3 proof. */
export function geometricPathIncrementalLogWeight(
  deltaBeta: number,
  logLik: readonly number[]
): number[] {
  const n = logLik.length;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = deltaBeta * logLik[i];
  return out;
}

// =============================================================================
// 5. ADAPTIVE TEMPERATURE SCHEDULE (Del Moral-Doucet-Jasra 2012)
//
// Bisect for β_{t+1} such that the projected ESS at β_{t+1} equals
// τ_adapt · N. ESS is monotone-decreasing in β so bisection always converges.
// =============================================================================

/** Projected ESS at β, given current particle log-weights and the cached
 * log-likelihood values at the current cloud positions. */
export function projectedEss(
  beta: number,
  betaCurrent: number,
  logW: readonly number[],
  logLik: readonly number[]
): number {
  const n = logW.length;
  const proj = new Array<number>(n);
  const delta = beta - betaCurrent;
  for (let i = 0; i < n; i++) proj[i] = logW[i] + delta * logLik[i];
  return logSpaceESS(proj);
}

/** Bisect β in [β_t, 1] to find the next temperature that hits target ESS.
 * Returns the new beta and the realized ESS at that beta. */
export function findNextBeta(
  betaCurrent: number,
  logW: readonly number[],
  logLik: readonly number[],
  targetEss: number,
  tol = 1e-3,
  maxIter = 40
): { beta: number; ess: number; iters: number } {
  const essAtOne = projectedEss(1.0, betaCurrent, logW, logLik);
  if (essAtOne >= targetEss) {
    // We can jump straight to β = 1 without falling below target.
    return { beta: 1.0, ess: essAtOne, iters: 0 };
  }
  let lo = betaCurrent;
  let hi = 1.0;
  let iter = 0;
  while (hi - lo > tol && iter < maxIter) {
    const mid = 0.5 * (lo + hi);
    const e = projectedEss(mid, betaCurrent, logW, logLik);
    if (e >= targetEss) lo = mid;
    else hi = mid;
    iter++;
  }
  const beta = 0.5 * (lo + hi);
  const ess = projectedEss(beta, betaCurrent, logW, logLik);
  return { beta, ess, iters: iter };
}

// =============================================================================
// 6. CLOUD STATISTICS (used by IMH cloud-fit, by viz)
// =============================================================================

/** Weighted mean of a 1D particle cloud. */
export function cloudMean1D(theta: readonly number[], wNorm: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < theta.length; i++) s += wNorm[i] * theta[i];
  return s;
}

/** Weighted standard deviation of a 1D particle cloud (population, not sample). */
export function cloudStd1D(
  theta: readonly number[],
  wNorm: readonly number[],
  mean?: number
): number {
  const mu = mean ?? cloudMean1D(theta, wNorm);
  let s = 0;
  for (let i = 0; i < theta.length; i++) {
    const d = theta[i] - mu;
    s += wNorm[i] * d * d;
  }
  return Math.sqrt(s);
}

/** d-dim weighted mean. theta is N row-vectors of length d. */
export function cloudMean(theta: readonly (readonly number[])[], wNorm: readonly number[]): number[] {
  const N = theta.length;
  if (N === 0) return [];
  const d = theta[0].length;
  const mu = new Array<number>(d).fill(0);
  for (let i = 0; i < N; i++) {
    const w = wNorm[i];
    const row = theta[i];
    for (let j = 0; j < d; j++) mu[j] += w * row[j];
  }
  return mu;
}

/** d-dim weighted covariance. Returns a d×d row-major matrix. */
export function cloudCov(
  theta: readonly (readonly number[])[],
  wNorm: readonly number[],
  mean?: readonly number[]
): number[][] {
  const N = theta.length;
  if (N === 0) return [];
  const d = theta[0].length;
  const mu = mean ?? cloudMean(theta, wNorm);
  const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (let i = 0; i < N; i++) {
    const w = wNorm[i];
    const row = theta[i];
    for (let j = 0; j < d; j++) {
      const dj = row[j] - mu[j];
      for (let k = j; k < d; k++) {
        const dk = row[k] - mu[k];
        cov[j][k] += w * dj * dk;
      }
    }
  }
  for (let j = 0; j < d; j++) for (let k = 0; k < j; k++) cov[j][k] = cov[k][j];
  return cov;
}

// =============================================================================
// 7. IMH CLOUD-FIT PROPOSAL (the PyMC pm.sample_smc default)
//
// At each SMC step we fit a Gaussian to the current weighted cloud (mean,
// covariance + ridge for stability) and propose IMH moves from it. The
// acceptance log-ratio is
//   log α = log π_t(θ') - log π_t(θ) + log q(θ) - log q(θ')
// where q(·) is the multivariate Gaussian density at the cloud's (μ̂, Σ̂).
// =============================================================================

/** Multivariate Gaussian log-density at theta, given a precomputed Cholesky
 * factor L of Σ. Closed-form via solveLowerTriangular(L, theta - mu). */
export function mvnLogPdfChol(
  theta: readonly number[],
  mu: readonly number[],
  L: readonly number[][],
  logDet: number
): number {
  const d = theta.length;
  const diff = new Array<number>(d);
  for (let i = 0; i < d; i++) diff[i] = theta[i] - mu[i];
  const y = solveLowerTriangular(L as number[][], diff);
  let q = 0;
  for (let i = 0; i < d; i++) q += y[i] * y[i];
  return -0.5 * (d * Math.log(2 * Math.PI) + logDet + q);
}

/** Sample one d-dim Gaussian: x = mu + L · z with z ~ N(0, I). */
export function mvnSampleChol(
  mu: readonly number[],
  L: readonly number[][],
  gaussRng: () => number
): number[] {
  const d = mu.length;
  const z = new Array<number>(d);
  for (let i = 0; i < d; i++) z[i] = gaussRng();
  const x = matVec(L as number[][], z);
  for (let i = 0; i < d; i++) x[i] += mu[i];
  return x;
}

export interface ImhCloudFit {
  mu: number[];
  L: number[][]; // Cholesky factor of Σ + εI
  logDet: number;
  d: number;
}

/** Fit a multivariate Gaussian to the weighted cloud, with ridge ε on the
 * covariance for Cholesky stability. */
export function fitImhProposal(
  theta: readonly (readonly number[])[],
  wNorm: readonly number[],
  ridge = 1e-6
): ImhCloudFit {
  const mu = cloudMean(theta, wNorm);
  const cov = cloudCov(theta, wNorm, mu);
  const Sigma = addDiagonal(cov, ridge);
  const L = choleskyFactor(Sigma);
  let logDet = 0;
  for (let i = 0; i < L.length; i++) logDet += 2 * Math.log(L[i][i]);
  return { mu, L, logDet, d: mu.length };
}

// =============================================================================
// 8. RWM (random-walk Metropolis) STEP
// =============================================================================

/** One vectorized RWM step on N particles in d=1. Returns new theta and per-
 * particle accept flags. logTarget is evaluated on a fresh proposal. */
export function rwmStep1D(
  theta: readonly number[],
  sigma: number,
  logTarget: (t: number) => number,
  rng: () => number,
  gaussRng: () => number
): { theta: number[]; accepted: boolean[] } {
  const n = theta.length;
  const out = new Array<number>(n);
  const acc = new Array<boolean>(n);
  for (let i = 0; i < n; i++) {
    const prop = theta[i] + sigma * gaussRng();
    const logAcc = logTarget(prop) - logTarget(theta[i]);
    if (Math.log(rng()) < logAcc) {
      out[i] = prop;
      acc[i] = true;
    } else {
      out[i] = theta[i];
      acc[i] = false;
    }
  }
  return { theta: out, accepted: acc };
}

/** Multiple RWM sweeps. */
export function rwmSweeps1D(
  theta: readonly number[],
  sigma: number,
  logTarget: (t: number) => number,
  sweeps: number,
  rng: () => number,
  gaussRng: () => number
): { theta: number[]; acceptRate: number } {
  let cur: number[] = theta.slice();
  let accCount = 0;
  let totalCount = 0;
  for (let s = 0; s < sweeps; s++) {
    const step = rwmStep1D(cur, sigma, logTarget, rng, gaussRng);
    cur = step.theta;
    for (let i = 0; i < step.accepted.length; i++) {
      if (step.accepted[i]) accCount++;
      totalCount++;
    }
  }
  return { theta: cur, acceptRate: accCount / Math.max(1, totalCount) };
}

/** Vectorized d-dim RWM step. logTarget evaluates a d-dim point. */
export function rwmStepDDim(
  theta: readonly (readonly number[])[],
  sigma: number,
  logTarget: (t: readonly number[]) => number,
  rng: () => number,
  gaussRng: () => number
): { theta: number[][]; accepted: boolean[] } {
  const n = theta.length;
  const d = theta[0].length;
  const out: number[][] = new Array(n);
  const acc = new Array<boolean>(n);
  for (let i = 0; i < n; i++) {
    const cur = theta[i];
    const prop = new Array<number>(d);
    for (let j = 0; j < d; j++) prop[j] = cur[j] + sigma * gaussRng();
    const logAcc = logTarget(prop) - logTarget(cur);
    if (Math.log(rng()) < logAcc) {
      out[i] = prop;
      acc[i] = true;
    } else {
      out[i] = cur.slice();
      acc[i] = false;
    }
  }
  return { theta: out, accepted: acc };
}

/** Multiple d-dim RWM sweeps. */
export function rwmSweepsDDim(
  theta: readonly (readonly number[])[],
  sigma: number,
  logTarget: (t: readonly number[]) => number,
  sweeps: number,
  rng: () => number,
  gaussRng: () => number
): { theta: number[][]; acceptRate: number } {
  let cur: number[][] = theta.map((row) => row.slice());
  let accCount = 0;
  let totalCount = 0;
  for (let s = 0; s < sweeps; s++) {
    const step = rwmStepDDim(cur, sigma, logTarget, rng, gaussRng);
    cur = step.theta;
    for (let i = 0; i < step.accepted.length; i++) {
      if (step.accepted[i]) accCount++;
      totalCount++;
    }
  }
  return { theta: cur, acceptRate: accCount / Math.max(1, totalCount) };
}

/** One IMH cloud-fit propagation sweep. Refits the Gaussian proposal to the
 * weighted cloud, then runs one IMH step per particle. */
export function imhCloudFitSweep(
  theta: readonly (readonly number[])[],
  wNorm: readonly number[],
  logTarget: (t: readonly number[]) => number,
  rng: () => number,
  gaussRng: () => number,
  ridge = 1e-6
): { theta: number[][]; acceptRate: number; fit: ImhCloudFit } {
  const fit = fitImhProposal(theta, wNorm, ridge);
  const n = theta.length;
  const out: number[][] = new Array(n);
  let accCount = 0;
  for (let i = 0; i < n; i++) {
    const prop = mvnSampleChol(fit.mu, fit.L, gaussRng);
    const logTargetCur = logTarget(theta[i]);
    const logTargetProp = logTarget(prop);
    const logProposalCur = mvnLogPdfChol(theta[i], fit.mu, fit.L, fit.logDet);
    const logProposalProp = mvnLogPdfChol(prop, fit.mu, fit.L, fit.logDet);
    const logAcc = logTargetProp - logTargetCur + logProposalCur - logProposalProp;
    if (Math.log(rng()) < logAcc) {
      out[i] = prop;
      accCount++;
    } else {
      out[i] = theta[i].slice();
    }
  }
  return { theta: out, acceptRate: accCount / Math.max(1, n), fit };
}

// =============================================================================
// 9. TARGET LOG-DENSITIES used in §§3, 5, 7, 8, 9, 10, 11
// =============================================================================

/** N(mu, sigma^2) log-pdf. */
export function gaussianLogPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return -0.5 * z * z - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
}

/** d-dim isotropic Gaussian log-pdf — diag covariance with scalar sigma. */
export function isotropicGaussianLogPdf(
  x: readonly number[],
  mu: readonly number[],
  sigma: number
): number {
  const d = x.length;
  let s = 0;
  for (let i = 0; i < d; i++) {
    const z = (x[i] - mu[i]) / sigma;
    s += z * z;
  }
  return -0.5 * s - d * Math.log(sigma) - 0.5 * d * Math.log(2 * Math.PI);
}

/** 2D anisotropic Gaussian log-pdf with diagonal covariance diag(s1^2, s2^2). */
export function anisotropic2DLogPdf(
  x: readonly number[],
  mu: readonly number[],
  s1: number,
  s2: number
): number {
  const z1 = (x[0] - mu[0]) / s1;
  const z2 = (x[1] - mu[1]) / s2;
  return -0.5 * (z1 * z1 + z2 * z2) - Math.log(s1) - Math.log(s2) - Math.log(2 * Math.PI);
}

/** Banana distribution log-density (Haario-Saksman-Tamminen 2001, simplified):
 *     log π(θ) = -θ_1^2 / 2 - (θ_2 - θ_1^2)^2 / 2.
 * Normalized 2D density (Z = 1 via change-of-variables); see brief §9.5.
 * Visually consistent with the RMHMC topic's §10 banana demo. */
export function bananaLogPdf(theta: readonly number[]): number {
  const t1 = theta[0];
  const t2 = theta[1];
  const u = t2 - t1 * t1;
  return -0.5 * t1 * t1 - 0.5 * u * u - Math.log(2 * Math.PI);
}

/** Geometric-path target on a Gaussian bridge: π_0 = N(mu0, sigma0^2) to
 * π_T = N(muT, sigmaT^2), unnormalized log γ_β(θ). Used by §3.4 skeleton
 * and the verify-suite end-to-end harness. */
export function gaussianBridgeLogGamma(
  theta: number,
  beta: number,
  mu0: number,
  sigma0: number,
  muT: number,
  sigmaT: number
): number {
  return (1 - beta) * gaussianLogPdf(theta, mu0, sigma0) + beta * gaussianLogPdf(theta, muT, sigmaT);
}

/** 2D anisotropic Gaussian bridge log γ_β: π_0 = N(0, I), π_T = N(mu_T, diag(s1², s2²)). */
export function anisotropic2DBridgeLogGamma(
  theta: readonly number[],
  beta: number,
  muT: readonly number[],
  s1: number,
  s2: number
): number {
  return (
    (1 - beta) * isotropicGaussianLogPdf(theta, [0, 0], 1) +
    beta * anisotropic2DLogPdf(theta, muT, s1, s2)
  );
}

// =============================================================================
// 10. END-TO-END HARNESSES
// =============================================================================

export interface SmcSkeletonOptions {
  N: number;
  T: number;
  mu0?: number;
  sigma0?: number;
  muT?: number;
  sigmaT?: number;
  resampleThreshold?: number;
  rwmStep?: number;
  rwmSweeps?: number;
  scheme?: ResamplingScheme;
  seed?: number;
}

export interface SmcSkeletonResult {
  betaGrid: number[];
  thetaHistory: number[][]; // [T+1][N]
  logWHistory: number[][]; // [T+1][N]
  essHistory: number[]; // [T+1]
  logZHistory: number[]; // [T+1]
  resampleMarks: boolean[]; // [T+1]
  acceptRateHistory: number[]; // [T+1]
  terminalMean: number;
  terminalStd: number;
  logZHat: number;
}

/** Reference §3.4 SMC-skeleton run on a Gaussian bridge. Default parameters
 * match the notebook's cell 7: π_0 = N(0, 4), π_T = N(5, 1), T = 10, N = 200,
 * RWM step 0.5 with 3 sweeps, systematic resampling at τ = 0.5.
 *
 * Used by SMCSkeletonExplorer for the live in-browser run, by
 * smcGaussianBridge in the verify suite for end-to-end assertions, and as the
 * algorithmic reference for §5 (PropagationKernelExplorer).
 */
export function smcGaussianBridge(opts: SmcSkeletonOptions = { N: 200, T: 10 }): SmcSkeletonResult {
  const N = opts.N;
  const T = opts.T;
  const mu0 = opts.mu0 ?? 0.0;
  const sigma0 = opts.sigma0 ?? 2.0;
  const muT = opts.muT ?? 5.0;
  const sigmaT = opts.sigmaT ?? 1.0;
  const tau = opts.resampleThreshold ?? 0.5;
  const rwmSigma = opts.rwmStep ?? 0.5;
  const sweeps = opts.rwmSweeps ?? 3;
  const scheme = opts.scheme ?? 'systematic';
  const seed = opts.seed ?? SMC_SEED + 3;
  const rng = mulberry32(seed);
  const gauss = makeGaussian(rng);

  const betaGrid = new Array<number>(T + 1);
  for (let t = 0; t <= T; t++) betaGrid[t] = t / T;

  let theta = new Array<number>(N);
  for (let i = 0; i < N; i++) theta[i] = mu0 + sigma0 * gauss();
  let logW = new Array<number>(N).fill(0);

  const thetaHistory: number[][] = Array.from({ length: T + 1 }, () => new Array<number>(N));
  const logWHistory: number[][] = Array.from({ length: T + 1 }, () => new Array<number>(N));
  const essHistory = new Array<number>(T + 1);
  const logZHistory = new Array<number>(T + 1).fill(0);
  const resampleMarks = new Array<boolean>(T + 1).fill(false);
  const acceptRateHistory = new Array<number>(T + 1).fill(0);
  for (let i = 0; i < N; i++) thetaHistory[0][i] = theta[i];
  essHistory[0] = N;

  for (let t = 0; t < T; t++) {
    const betaNext = betaGrid[t + 1];
    const betaCur = betaGrid[t];
    // reweight
    for (let i = 0; i < N; i++) {
      logW[i] += gaussianBridgeLogGamma(theta[i], betaNext, mu0, sigma0, muT, sigmaT)
        - gaussianBridgeLogGamma(theta[i], betaCur, mu0, sigma0, muT, sigmaT);
    }
    logZHistory[t + 1] = logZFromLogW(logW);
    const logWNorm = normalizeLogW(logW);
    const ess = logSpaceESS(logW);

    // resample if ESS below threshold
    if (ess < tau * N) {
      const wNorm = logWNorm.map(Math.exp);
      const idx = resample(scheme, wNorm, rng, N);
      const newTheta = new Array<number>(N);
      for (let i = 0; i < N; i++) newTheta[i] = theta[idx[i]];
      theta = newTheta;
      logW = new Array<number>(N).fill(logZHistory[t + 1]);
      resampleMarks[t + 1] = true;
    }

    // propagate via RWM-on-π_{t+1}
    const logTarget = (x: number) =>
      gaussianBridgeLogGamma(x, betaNext, mu0, sigma0, muT, sigmaT);
    const swept = rwmSweeps1D(theta, rwmSigma, logTarget, sweeps, rng, gauss);
    theta = swept.theta;
    acceptRateHistory[t + 1] = swept.acceptRate;

    for (let i = 0; i < N; i++) {
      thetaHistory[t + 1][i] = theta[i];
      logWHistory[t + 1][i] = logW[i];
    }
    essHistory[t + 1] = logSpaceESS(logW);
  }

  // terminal weighted moments
  const lwT = logWHistory[T];
  const wT = normalizeLogW(lwT).map(Math.exp);
  const terminalMean = cloudMean1D(thetaHistory[T], wT);
  const terminalStd = cloudStd1D(thetaHistory[T], wT, terminalMean);

  return {
    betaGrid,
    thetaHistory,
    logWHistory,
    essHistory,
    logZHistory,
    resampleMarks,
    acceptRateHistory,
    terminalMean,
    terminalStd,
    logZHat: logZHistory[T],
  };
}

export type PropagationKernel =
  | { kind: 'rwm'; sigma: number; sweeps: number }
  | { kind: 'imh'; ridge?: number };

export interface SmcAnisotropicOptions {
  N: number;
  T: number;
  muT: readonly [number, number];
  s1: number;
  s2: number;
  resampleThreshold?: number;
  kernel: PropagationKernel;
  scheme?: ResamplingScheme;
  seed?: number;
}

export interface SmcAnisotropicResult {
  betaGrid: number[];
  thetaHistory: number[][][]; // [T+1][N][2]
  logWHistory: number[][]; // [T+1][N]
  essHistory: number[];
  logZHistory: number[];
  resampleMarks: boolean[];
  acceptRateHistory: number[]; // [T+1]
  terminalCloud: number[][]; // [N][2]
  terminalWeights: number[]; // [N], normalized
  logZHat: number;
}

/** Generic 2D SMC sampler on the anisotropic Gaussian bridge with pluggable
 * propagation kernel (RWM with adjustable sweeps, or IMH cloud-fit). */
export function smcAnisotropic2D(opts: SmcAnisotropicOptions): SmcAnisotropicResult {
  const { N, T, muT, s1, s2, kernel } = opts;
  const tau = opts.resampleThreshold ?? 0.5;
  const scheme = opts.scheme ?? 'systematic';
  const seed = opts.seed ?? SMC_SEED + 5;
  const rng = mulberry32(seed);
  const gauss = makeGaussian(rng);

  const betaGrid = new Array<number>(T + 1);
  for (let t = 0; t <= T; t++) betaGrid[t] = t / T;

  let theta: number[][] = Array.from({ length: N }, () => [gauss(), gauss()]);
  let logW = new Array<number>(N).fill(0);

  const thetaHistory: number[][][] = Array.from({ length: T + 1 }, () =>
    Array.from({ length: N }, () => [0, 0]),
  );
  const logWHistory: number[][] = Array.from({ length: T + 1 }, () => new Array<number>(N).fill(0));
  const essHistory = new Array<number>(T + 1).fill(N);
  const logZHistory = new Array<number>(T + 1).fill(0);
  const resampleMarks = new Array<boolean>(T + 1).fill(false);
  const acceptRateHistory = new Array<number>(T + 1).fill(1);
  for (let i = 0; i < N; i++) {
    thetaHistory[0][i][0] = theta[i][0];
    thetaHistory[0][i][1] = theta[i][1];
  }

  for (let t = 0; t < T; t++) {
    const betaNext = betaGrid[t + 1];
    const betaCur = betaGrid[t];
    // reweight
    for (let i = 0; i < N; i++) {
      logW[i] +=
        anisotropic2DBridgeLogGamma(theta[i], betaNext, muT, s1, s2) -
        anisotropic2DBridgeLogGamma(theta[i], betaCur, muT, s1, s2);
    }
    logZHistory[t + 1] = logZFromLogW(logW);
    const ess = logSpaceESS(logW);
    if (ess < tau * N) {
      const wNorm = normalizeLogW(logW).map(Math.exp);
      const idx = resample(scheme, wNorm, rng, N);
      const newTheta = new Array<number[]>(N);
      for (let i = 0; i < N; i++) newTheta[i] = theta[idx[i]].slice();
      theta = newTheta;
      logW = new Array<number>(N).fill(logZHistory[t + 1]);
      resampleMarks[t + 1] = true;
    }
    // propagate
    const logTarget = (x: readonly number[]) =>
      anisotropic2DBridgeLogGamma(x, betaNext, muT, s1, s2);
    if (kernel.kind === 'rwm') {
      const swept = rwmSweepsDDim(theta, kernel.sigma, logTarget, kernel.sweeps, rng, gauss);
      theta = swept.theta;
      acceptRateHistory[t + 1] = swept.acceptRate;
    } else {
      const wNorm = normalizeLogW(logW).map(Math.exp);
      const swept = imhCloudFitSweep(theta, wNorm, logTarget, rng, gauss, kernel.ridge ?? 1e-6);
      theta = swept.theta;
      acceptRateHistory[t + 1] = swept.acceptRate;
    }
    for (let i = 0; i < N; i++) {
      thetaHistory[t + 1][i][0] = theta[i][0];
      thetaHistory[t + 1][i][1] = theta[i][1];
      logWHistory[t + 1][i] = logW[i];
    }
    essHistory[t + 1] = logSpaceESS(logW);
  }
  const lwT = logWHistory[T];
  const wT = normalizeLogW(lwT).map(Math.exp);
  return {
    betaGrid,
    thetaHistory,
    logWHistory,
    essHistory,
    logZHistory,
    resampleMarks,
    acceptRateHistory,
    terminalCloud: theta,
    terminalWeights: wT,
    logZHat: logZHistory[T],
  };
}

export interface AdaptiveSmcOptions {
  N: number;
  d: number;
  logPrior: (theta: readonly number[]) => number;
  logLik: (theta: readonly number[]) => number;
  drawPrior: (gauss: () => number, rng: () => number) => number[];
  /** "adaptive": ESS-driven bisection; otherwise a fixed schedule array of betas. */
  schedule: 'adaptive' | { fixedBetas: readonly number[] };
  tauAdapt?: number; // for adaptive: target ESS fraction (default 0.9)
  resampleThreshold?: number; // default 0.5
  maxSteps?: number; // safety cap on adaptive paths (default 200)
  kernel?: 'imh' | { kind: 'rwm'; sigma: number; sweeps: number };
  scheme?: ResamplingScheme;
  ridge?: number;
  seed?: number;
}

export interface AdaptiveSmcResult {
  betaTrace: number[];
  thetaHistory: number[][][]; // per step, N particles in R^d
  logWHistory: number[][];
  essHistory: number[];
  logZHistory: number[];
  resampleMarks: boolean[];
  acceptRateHistory: number[];
  terminalCloud: number[][];
  terminalWeights: number[];
  logZHat: number;
  T: number;
}

/** Generic adaptive SMC sampler — works for banana (d=2), GMM (d=3),
 * and other small-dimensional targets. The schedule is either fixed
 * (array of betas) or adaptive (ESS-bisection at each step). */
export function smcAdaptiveSampler(opts: AdaptiveSmcOptions): AdaptiveSmcResult {
  const N = opts.N;
  const d = opts.d;
  const tauAdapt = opts.tauAdapt ?? 0.9;
  const tau = opts.resampleThreshold ?? 0.5;
  const maxSteps = opts.maxSteps ?? 200;
  const kernel = opts.kernel ?? 'imh';
  const scheme = opts.scheme ?? 'systematic';
  const ridge = opts.ridge ?? 1e-6;
  const seed = opts.seed ?? SMC_SEED;
  const rng = mulberry32(seed);
  const gauss = makeGaussian(rng);

  let theta: number[][] = Array.from({ length: N }, () => opts.drawPrior(gauss, rng));
  let logW = new Array<number>(N).fill(0);
  let logLikCache = theta.map((p) => opts.logLik(p));

  const betaTrace = [0];
  const thetaHistory: number[][][] = [theta.map((row) => row.slice())];
  const logWHistory: number[][] = [logW.slice()];
  const essHistory = [N];
  const logZHistory = [0];
  const resampleMarks = [false];
  const acceptRateHistory = [1];

  let beta = 0;
  let step = 0;
  while (beta < 1 - 1e-9 && step < maxSteps) {
    let nextBeta: number;
    if (opts.schedule === 'adaptive') {
      const res = findNextBeta(beta, logW, logLikCache, tauAdapt * N, 1e-4, 60);
      nextBeta = Math.max(res.beta, beta + 1e-4); // force progress
    } else {
      const fixed = opts.schedule.fixedBetas;
      if (step + 1 < fixed.length) nextBeta = fixed[step + 1];
      else nextBeta = 1;
    }
    if (nextBeta > 1) nextBeta = 1;
    const dBeta = nextBeta - beta;
    // reweight
    for (let i = 0; i < N; i++) logW[i] += dBeta * logLikCache[i];
    const logZ = logZFromLogW(logW);
    logZHistory.push(logZ);
    const ess = logSpaceESS(logW);

    let resampled = false;
    if (ess < tau * N) {
      const wNorm = normalizeLogW(logW).map(Math.exp);
      const idx = resample(scheme, wNorm, rng, N);
      const newTheta = new Array<number[]>(N);
      const newLogLik = new Array<number>(N);
      for (let i = 0; i < N; i++) {
        newTheta[i] = theta[idx[i]].slice();
        newLogLik[i] = logLikCache[idx[i]];
      }
      theta = newTheta;
      logLikCache = newLogLik;
      logW = new Array<number>(N).fill(logZ);
      resampled = true;
    }

    // propagate with kernel that targets π_{β = nextBeta}
    const logTarget = (x: readonly number[]) =>
      opts.logPrior(x) + nextBeta * opts.logLik(x);
    let accRate = 1;
    if (kernel === 'imh') {
      const wNorm = normalizeLogW(logW).map(Math.exp);
      const swept = imhCloudFitSweep(theta, wNorm, logTarget, rng, gauss, ridge);
      theta = swept.theta;
      accRate = swept.acceptRate;
    } else {
      const swept = rwmSweepsDDim(theta, kernel.sigma, logTarget, kernel.sweeps, rng, gauss);
      theta = swept.theta;
      accRate = swept.acceptRate;
    }
    // re-cache log-likelihood at new positions
    for (let i = 0; i < N; i++) logLikCache[i] = opts.logLik(theta[i]);

    beta = nextBeta;
    step++;
    betaTrace.push(beta);
    thetaHistory.push(theta.map((row) => row.slice()));
    logWHistory.push(logW.slice());
    essHistory.push(logSpaceESS(logW));
    resampleMarks.push(resampled);
    acceptRateHistory.push(accRate);
  }
  const T = step;
  const wT = normalizeLogW(logW).map(Math.exp);
  return {
    betaTrace,
    thetaHistory,
    logWHistory,
    essHistory,
    logZHistory,
    resampleMarks,
    acceptRateHistory,
    terminalCloud: theta,
    terminalWeights: wT,
    logZHat: logZHistory[logZHistory.length - 1],
    T,
  };
}

// =============================================================================
// 11. STOCHASTIC-VOLATILITY HELPERS (§6.4 ParticleFilterExplorer)
//
// Model: x_t = phi x_{t-1} + sigma eps_t,    eps_t ~ N(0, 1)
//        y_t = exp(x_t / 2) eta_t,           eta_t ~ N(0, 1)
// Stationary init: x_0 ~ N(0, sigma^2 / (1 - phi^2)).
// =============================================================================

export interface SVDataset {
  T: number;
  xTrue: number[];
  y: number[];
}

/** Simulate a stochastic-volatility trajectory. */
export function simulateSV(
  T: number,
  phi: number,
  sigma: number,
  seed: number
): SVDataset {
  const rng = mulberry32(seed);
  const gauss = makeGaussian(rng);
  const sigmaStat = sigma / Math.sqrt(Math.max(1e-12, 1 - phi * phi));
  const xTrue = new Array<number>(T);
  const y = new Array<number>(T);
  let xPrev = sigmaStat * gauss();
  for (let t = 0; t < T; t++) {
    const x = phi * xPrev + sigma * gauss();
    const yt = Math.exp(x / 2) * gauss();
    xTrue[t] = x;
    y[t] = yt;
    xPrev = x;
  }
  return { T, xTrue, y };
}

/** SV emission log-density log p(y_t | x_t). */
export function svEmissionLogPdf(y: number, x: number): number {
  // y ~ N(0, exp(x))
  const sigma = Math.exp(x / 2);
  return gaussianLogPdf(y, 0, sigma);
}

export interface ParticleFilterResult {
  T: number;
  N: number;
  mean: number[]; // [T] filtered E[x_t | y_{1:t}]
  q025: number[]; // 2.5% quantile
  q975: number[]; // 97.5% quantile
  ess: number[]; // [T]
  logZ: number; // log p(y_{1:T})
}

/** Bootstrap particle filter on the stochastic-volatility model. */
export function bootstrapParticleFilter(
  data: SVDataset,
  N: number,
  phi: number,
  sigma: number,
  tau: number,
  seed: number,
  scheme: ResamplingScheme = 'systematic'
): ParticleFilterResult {
  const T = data.T;
  const rng = mulberry32(seed);
  const gauss = makeGaussian(rng);
  const sigmaStat = sigma / Math.sqrt(Math.max(1e-12, 1 - phi * phi));

  let x = new Array<number>(N);
  for (let i = 0; i < N; i++) x[i] = sigmaStat * gauss();
  let logW = new Array<number>(N).fill(0);

  const mean = new Array<number>(T);
  const q025 = new Array<number>(T);
  const q975 = new Array<number>(T);
  const ess = new Array<number>(T);
  let logZ = 0;

  for (let t = 0; t < T; t++) {
    // propagate via prior dynamics
    for (let i = 0; i < N; i++) x[i] = phi * x[i] + sigma * gauss();
    // reweight by emission likelihood
    const logLik = new Array<number>(N);
    for (let i = 0; i < N; i++) logLik[i] = svEmissionLogPdf(data.y[t], x[i]);
    for (let i = 0; i < N; i++) logW[i] += logLik[i];
    // marginal log-likelihood update: log p(y_t | y_{1:t-1}) ≈ logsumexp(logW) - logsumexp(prev)
    // simpler: incrementally accumulate logZ = sum_t log mean(exp(logLik))
    logZ += logSumExp(logLik) - Math.log(N);
    // diagnostics
    const wNorm = normalizeLogW(logW).map(Math.exp);
    const m = cloudMean1D(x, wNorm);
    mean[t] = m;
    // quantiles via weighted sort
    const sortIdx = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
    let cumw = 0;
    let q025v = x[sortIdx[0]];
    let q975v = x[sortIdx[N - 1]];
    for (let k = 0; k < N; k++) {
      const idx = sortIdx[k];
      const prev = cumw;
      cumw += wNorm[idx];
      if (prev < 0.025 && cumw >= 0.025) q025v = x[idx];
      if (prev < 0.975 && cumw >= 0.975) {
        q975v = x[idx];
        break;
      }
    }
    q025[t] = q025v;
    q975[t] = q975v;
    ess[t] = logSpaceESS(logW);
    // resample if needed
    if (ess[t] < tau * N) {
      const idx = resample(scheme, wNorm, rng, N);
      const newX = new Array<number>(N);
      for (let i = 0; i < N; i++) newX[i] = x[idx[i]];
      x = newX;
      logW = new Array<number>(N).fill(0);
    }
  }
  return { T, N, mean, q025, q975, ess, logZ };
}

// =============================================================================
// 12. GMM EVIDENCE (§8.4 EvidenceUnbiasednessExplorer)
//
// 2-component Gaussian mixture y_i ~ p N(mu1, 1) + (1-p) N(mu2, 1)
// Priors: mu1, mu2 ~ N(0, 4); p ~ Beta(2, 2). Posterior is bimodal (label
// switching). We use SMC-samplers on geometric path from prior to posterior.
//
// Parameter vector: theta = [mu1, mu2, p] with p reparameterized via
// logit-Beta proposal handled separately in the viz.
// =============================================================================

/** Log-prior on [mu1, mu2, p]: mu1,mu2 ~ N(0, 4), p ~ Beta(2, 2). */
export function gmmLogPrior(theta: readonly number[]): number {
  const [mu1, mu2, p] = theta;
  if (p <= 0 || p >= 1) return -Infinity;
  return (
    gaussianLogPdf(mu1, 0, 2) +
    gaussianLogPdf(mu2, 0, 2) +
    // Beta(2,2) log-pdf: 6 p (1-p)
    Math.log(6) + Math.log(p) + Math.log(1 - p)
  );
}

/** Per-observation log-likelihood log p(y_i | mu1, mu2, p) for the 2-comp GMM. */
export function gmmLogLikOne(yi: number, mu1: number, mu2: number, p: number): number {
  const l1 = Math.log(p) + gaussianLogPdf(yi, mu1, 1);
  const l2 = Math.log(1 - p) + gaussianLogPdf(yi, mu2, 1);
  return logSumExp([l1, l2]);
}

/** Full GMM log-likelihood at theta given observed y. */
export function gmmLogLik(theta: readonly number[], y: readonly number[]): number {
  const [mu1, mu2, p] = theta;
  if (p <= 0 || p >= 1) return -Infinity;
  let s = 0;
  for (let i = 0; i < y.length; i++) s += gmmLogLikOne(y[i], mu1, mu2, p);
  return s;
}

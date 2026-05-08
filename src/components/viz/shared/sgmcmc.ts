// =============================================================================
// sgmcmc.ts
//
// Shared math/types/palette module for the T5 stochastic-gradient-mcmc topic.
// Mirrors notebooks/stochastic-gradient-mcmc/01_stochastic_gradient_mcmc.ipynb
// cells 2 (diagnostics + toy potentials), 4 (complete-recipe phase portraits),
// 6 (Brownian paths), 8 (OU density), 10 (Langevin on GMM), 12 (BLR + SGLD),
// 14 (anisotropic Gaussian + SGHMC), 16 (synthetic-noise SGLD for VZT),
// 18 (SVRG-LD + ZV-SGLD), 20 (funnel SGLD vs RSGLD), and 24 (R-hat).
//
// Cell 26 (NUTS via PyMC + ESS/sec head-to-head) is NOT ported — that comparison
// is genuinely non-conjugate at deep-learning scale and is precomputed in
// notebooks/stochastic-gradient-mcmc/precompute_head_to_head.py.
//
// Conventions:
//   - All chain runners return Float32Array[] (length n_steps × d). For 1D
//     chains we still wrap in length-1 buffers for uniformity.
//   - All RNGs are () => number returning uniforms in [0, 1). Seedable via
//     mulberry32 from bayesian-ml.ts. Standard normals via gaussianPair (also
//     bayesian-ml.ts) wrapped in a stateful gauss() helper.
//   - Functions are pure (no global state) and have no React/D3 dependencies.
// =============================================================================

import { gaussianPair, mulberry32 } from './bayesian-ml';

// =============================================================================
// PALETTE
// =============================================================================
// Mirrors the notebook's COLOR_* constants in cell 1. Anchors for cross-section
// visual continuity: SGLD always purple, SGHMC always teal, RSGLD always pink,
// the VZT bound always orange, NUTS always red.
// =============================================================================

export const paletteSGMCMC = {
  target: '#2c3e50', //   near-black — analytical posterior / target
  brownian: '#7f7f7f', //  gray       — §3 Brownian paths
  langevin: '#1f77b4', //  blue       — §§4–5 continuous-time Langevin
  sgld: '#9467bd', //      purple     — §6 SGLD
  sghmc: '#17becf', //     teal       — §7 SGHMC
  rsgld: '#e377c2', //     pink       — §10 Riemann-manifold SGLD
  rmhmc: '#bcbd22', //     olive      — §10 Riemann-manifold HMC
  nuts: '#d62728', //      red        — §12 NUTS reference
  biasBound: '#ff7f0e', // orange     — §8 VZT bias bound
} as const;

export type SGMCMCColorKey = keyof typeof paletteSGMCMC;

// =============================================================================
// RNG HELPERS
// =============================================================================

/** A standard-normal sampler built on top of a uniform RNG. Stateful: caches
 *  the second Gaussian from each Box–Muller pair to halve calls to the RNG. */
export function makeGauss(rng: () => number): () => number {
  let buf: number | null = null;
  return () => {
    if (buf !== null) {
      const v = buf;
      buf = null;
      return v;
    }
    const [a, b] = gaussianPair(rng);
    buf = b;
    return a;
  };
}

/** Sample n indices uniformly without replacement from {0, ..., N-1}.
 *  Partial Fisher–Yates, O(n). Allocates a fresh Int32Array each call —
 *  for hot loops (chain runners with mini-batch gradients), use makeSampler
 *  below to hoist the pool and output buffer out of the loop. */
export function sampleWithoutReplacement(N: number, n: number, rng: () => number): Int32Array {
  if (n > N) throw new Error(`sampleWithoutReplacement: n=${n} > N=${N}`);
  const pool = new Int32Array(N);
  for (let i = 0; i < N; i++) pool[i] = i;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(rng() * (N - i));
    const tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
    out[i] = pool[i];
  }
  return out;
}

/** Build a hoisted-buffer sampler for repeated calls with the same (N, B).
 *  The returned function reuses a single `pool` of length N and a single
 *  `out` of length B across calls, undoing the partial Fisher–Yates swaps
 *  in O(B) at the end of each call so `pool` stays in `[0..N-1]` order.
 *  The returned Int32Array is the same buffer every call — consume it
 *  before the next call. Use this inside chain runners to avoid the
 *  per-step allocation pressure of `sampleWithoutReplacement`. */
export function makeSampler(N: number, B: number): (rng: () => number) => Int32Array {
  if (B > N) throw new Error(`makeSampler: B=${B} > N=${N}`);
  const pool = new Int32Array(N);
  for (let i = 0; i < N; i++) pool[i] = i;
  const out = new Int32Array(B);
  const swapJ = new Int32Array(B);
  return (rng: () => number): Int32Array => {
    for (let i = 0; i < B; i++) {
      const j = i + Math.floor(rng() * (N - i));
      swapJ[i] = j;
      const tmp = pool[i];
      pool[i] = pool[j];
      pool[j] = tmp;
      out[i] = pool[i];
    }
    // Undo swaps in reverse so `pool` is restored to [0..N-1] for the next call.
    for (let i = B - 1; i >= 0; i--) {
      const j = swapJ[i];
      const tmp = pool[i];
      pool[i] = pool[j];
      pool[j] = tmp;
    }
    return out;
  };
}

// =============================================================================
// DIAGNOSTICS
// =============================================================================
// Mirrors notebook cell 2's autocorr / integrated_autocorr_time /
// effective_sample_size, and cell 24's gelman_rubin_rhat.
// =============================================================================

/** Sample autocorrelation of a 1D chain at lags 0..maxLag.
 *  Defaults: maxLag = min(n/4, 500). Returns acf[0] = 1 by construction. */
export function autocorr(x: ArrayLike<number>, maxLag?: number): Float32Array {
  const n = x.length;
  if (n === 0) return new Float32Array(0);
  const lag = maxLag ?? Math.min(Math.floor(n / 4), 500);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let var0 = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    var0 += d * d;
  }
  var0 /= n;
  if (var0 === 0) return new Float32Array(lag + 1);
  const out = new Float32Array(lag + 1);
  for (let k = 0; k <= lag; k++) {
    let s = 0;
    for (let i = 0; i < n - k; i++) s += (x[i] - mean) * (x[i + k] - mean);
    out[k] = s / (n * var0);
  }
  return out;
}

/** Sokal's automated-windowing IAT estimator: τ_IAT = 1 + 2Σ_{k=1}^M ρ(k),
 *  with M chosen as the smallest index where M ≥ c·τ. */
export function integratedAutocorrTime(x: ArrayLike<number>, c: number = 5.0): number {
  const acf = autocorr(x);
  if (acf.length <= 1) return 1.0;
  const tauCum = new Float32Array(acf.length - 1);
  let s = 0;
  for (let i = 0; i < tauCum.length; i++) {
    s += acf[i + 1];
    tauCum[i] = 1 + 2 * s;
  }
  for (let m = 0; m < tauCum.length; m++) {
    if (m + 1 >= c * tauCum[m]) return tauCum[m];
  }
  return tauCum[tauCum.length - 1];
}

/** ESS = N / IAT. Accepts a 1D trace (number[]/Float32Array) or a multi-coord
 *  chain (Float32Array[]) — for the latter, returns the average per-coord ESS. */
export function effectiveSampleSize(chain: ArrayLike<number> | Float32Array[]): number {
  if (Array.isArray(chain) && chain.length > 0 && chain[0] instanceof Float32Array) {
    const c = chain as Float32Array[];
    const d = c[0].length;
    let total = 0;
    for (let j = 0; j < d; j++) {
      const trace = new Float32Array(c.length);
      for (let i = 0; i < c.length; i++) trace[i] = c[i][j];
      total += c.length / Math.max(integratedAutocorrTime(trace), 1.0);
    }
    return total / d;
  }
  const c = chain as ArrayLike<number>;
  return c.length / Math.max(integratedAutocorrTime(c), 1.0);
}

/** Gelman–Rubin R̂ over multiple chains of equal length. Convergence: R̂ < 1.01.
 *  Each chain is a 1D trace; pass the same coordinate from multiple chains. */
export function gelmanRubinRhat(chains: ArrayLike<number>[]): number {
  const m = chains.length;
  if (m < 2) return NaN;
  const n = chains[0].length;
  // Per-chain mean and variance (unbiased, ddof=1).
  const chainMeans = new Float32Array(m);
  const chainVars = new Float32Array(m);
  for (let j = 0; j < m; j++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += chains[j][i];
    mean /= n;
    chainMeans[j] = mean;
    let v = 0;
    for (let i = 0; i < n; i++) {
      const d = chains[j][i] - mean;
      v += d * d;
    }
    chainVars[j] = v / (n - 1);
  }
  let grandMean = 0;
  for (let j = 0; j < m; j++) grandMean += chainMeans[j];
  grandMean /= m;
  let B = 0;
  for (let j = 0; j < m; j++) {
    const d = chainMeans[j] - grandMean;
    B += d * d;
  }
  B *= n / (m - 1);
  let W = 0;
  for (let j = 0; j < m; j++) W += chainVars[j];
  W /= m;
  if (W <= 0) return NaN;
  const varHat = ((n - 1) / n) * W + B / n;
  return Math.sqrt(varHat / W);
}

/** Extract one coordinate's trace from a (possibly thinned) chain. */
export function extractCoord(chain: Float32Array[], j: number): Float32Array {
  const out = new Float32Array(chain.length);
  for (let i = 0; i < chain.length; i++) out[i] = chain[i][j];
  return out;
}

// =============================================================================
// CHAIN RUNNERS
// =============================================================================
// Exactly the §§5–10 update rules from the brief, ported from the notebook
// cells 12, 14, 18, 20.
// =============================================================================

// -----------------------------------------------------------------------------
// Chain-buffer convention
// -----------------------------------------------------------------------------
// Every chain runner allocates ONE flat `Float32Array(nSteps * d)` and returns
// `Float32Array[]` whose entries are typed-array VIEWS (subarray) into that
// single buffer. The `Float32Array[]` shape is preserved for caller ergonomics
// (`chain[n][j]`, `chain.length`, `chain.slice(burn)`) but the per-step
// allocation cost drops from O(nSteps) Float32Array objects to O(1) — only
// the backing buffer and the wrapper array of views are allocated.
// -----------------------------------------------------------------------------

function allocChainViews(nSteps: number, d: number): { buf: Float32Array; views: Float32Array[] } {
  const buf = new Float32Array(nSteps * d);
  const views: Float32Array[] = new Array(nSteps);
  for (let n = 0; n < nSteps; n++) views[n] = buf.subarray(n * d, (n + 1) * d);
  return { buf, views };
}

/** Generic Euler–Maruyama on overdamped Langevin (§5–6).
 *  Update: θ_{n+1} = θ_n - η · gradFn(θ_n) + √(2η) · ξ_n, ξ ~ N(0, I).
 *  Use a closure for `gradFn` if you want fresh minibatch sampling per step. */
export function sgldChain(
  gradFn: (theta: Float32Array) => Float32Array,
  theta0: Float32Array,
  nSteps: number,
  eta: number,
  rng: () => number,
): Float32Array[] {
  const d = theta0.length;
  const gauss = makeGauss(rng);
  const sqrt2eta = Math.sqrt(2 * eta);
  const theta = new Float32Array(theta0);
  const { buf, views } = allocChainViews(nSteps, d);
  for (let n = 0; n < nSteps; n++) {
    const g = gradFn(theta);
    for (let i = 0; i < d; i++) theta[i] -= eta * g[i] - sqrt2eta * gauss();
    buf.set(theta, n * d);
  }
  return views;
}

/** SGLD with a step-size schedule (Welling–Teh §6.3). schedule(n) returns η_n. */
export function sgldChainScheduled(
  gradFn: (theta: Float32Array) => Float32Array,
  theta0: Float32Array,
  nSteps: number,
  schedule: (n: number) => number,
  rng: () => number,
): Float32Array[] {
  const d = theta0.length;
  const gauss = makeGauss(rng);
  const theta = new Float32Array(theta0);
  const { buf, views } = allocChainViews(nSteps, d);
  for (let n = 0; n < nSteps; n++) {
    const eta = schedule(n);
    const sqrt2eta = Math.sqrt(2 * eta);
    const g = gradFn(theta);
    for (let i = 0; i < d; i++) theta[i] -= eta * g[i] - sqrt2eta * gauss();
    buf.set(theta, n * d);
  }
  return views;
}

/** Underdamped Langevin / SGHMC (§7.4). Mass matrix is the identity (M = I);
 *  friction matrix is C·I. Update:
 *    r_{n+1} = (1 - η·C) r_n - η · gradFn(θ_n) + √(2ηC) · ξ
 *    θ_{n+1} = θ_n + η · r_{n+1}
 */
export function sghmcChain(
  gradFn: (theta: Float32Array) => Float32Array,
  theta0: Float32Array,
  nSteps: number,
  eta: number,
  C: number,
  rng: () => number,
): Float32Array[] {
  const d = theta0.length;
  const gauss = makeGauss(rng);
  const theta = new Float32Array(theta0);
  // Initialize momentum from the kinetic-energy stationary, N(0, M) = N(0, I).
  const r = new Float32Array(d);
  for (let i = 0; i < d; i++) r[i] = gauss();
  const oneMinusEtaC = 1 - eta * C;
  const sqrtNoise = Math.sqrt(2 * eta * C);
  const { buf, views } = allocChainViews(nSteps, d);
  for (let n = 0; n < nSteps; n++) {
    const g = gradFn(theta);
    for (let i = 0; i < d; i++) {
      r[i] = oneMinusEtaC * r[i] - eta * g[i] + sqrtNoise * gauss();
    }
    for (let i = 0; i < d; i++) theta[i] += eta * r[i];
    buf.set(theta, n * d);
  }
  return views;
}

/** Riemann-manifold Langevin with diagonal metric G⁻¹(θ) (§10.3).
 *  Update:
 *    θ_{n+1} = θ_n + η · [-G⁻¹(θ) · gradFn(θ) + ∇·G⁻¹(θ)] + √(2η) · √G⁻¹ · ξ
 *  For the funnel demo, divGInvFn returns zeros (verified in §10.4). */
export function rsgldDiagonalChain(
  gradFn: (theta: Float32Array) => Float32Array,
  gInvDiagFn: (theta: Float32Array) => Float32Array,
  divGInvFn: (theta: Float32Array) => Float32Array,
  theta0: Float32Array,
  nSteps: number,
  eta: number,
  rng: () => number,
): Float32Array[] {
  const d = theta0.length;
  const gauss = makeGauss(rng);
  const sqrt2eta = Math.sqrt(2 * eta);
  const theta = new Float32Array(theta0);
  const { buf, views } = allocChainViews(nSteps, d);
  for (let n = 0; n < nSteps; n++) {
    const gInv = gInvDiagFn(theta);
    const corr = divGInvFn(theta);
    const g = gradFn(theta);
    for (let i = 0; i < d; i++) {
      const drift = -gInv[i] * g[i] + corr[i];
      theta[i] += eta * drift + sqrt2eta * Math.sqrt(gInv[i]) * gauss();
    }
    buf.set(theta, n * d);
  }
  return views;
}

// =============================================================================
// TOY POTENTIALS / TARGETS
// =============================================================================

/** Two well-separated 2D Gaussians: 0.5 · N([-2, 0], 0.5·I) + 0.5 · N([2, 0], 0.5·I).
 *  Used in §§5 as the running 2D target for Langevin demos. */
export function gmm2DGradAndLogDensity(): {
  gradU: (theta: Float32Array) => Float32Array;
  logDensity: (v: number, x: number) => number;
  mus: number[][];
  sigma2: number;
} {
  const mus = [
    [-2.0, 0.0],
    [2.0, 0.0],
  ];
  const sigma2 = 0.5;
  // Diagonal covariance — inverse and log det are scalars.
  const covInv = 1 / sigma2;
  const logNorm = -Math.log(2 * Math.PI) - Math.log(sigma2);
  const logW = Math.log(0.5);

  function logDensity(v: number, x: number): number {
    let mx = -Infinity;
    const ek = new Float64Array(mus.length);
    for (let k = 0; k < mus.length; k++) {
      const dv = v - mus[k][0];
      const dx = x - mus[k][1];
      ek[k] = -0.5 * covInv * (dv * dv + dx * dx) + logNorm + logW;
      if (ek[k] > mx) mx = ek[k];
    }
    let s = 0;
    for (let k = 0; k < mus.length; k++) s += Math.exp(ek[k] - mx);
    return mx + Math.log(s);
  }

  function gradU(theta: Float32Array): Float32Array {
    // Soft-min over components: weight by posterior responsibility.
    const ek = new Float64Array(mus.length);
    let mx = -Infinity;
    for (let k = 0; k < mus.length; k++) {
      const dv = theta[0] - mus[k][0];
      const dx = theta[1] - mus[k][1];
      ek[k] = -0.5 * covInv * (dv * dv + dx * dx) + logNorm + logW;
      if (ek[k] > mx) mx = ek[k];
    }
    let z = 0;
    for (let k = 0; k < mus.length; k++) {
      ek[k] = Math.exp(ek[k] - mx);
      z += ek[k];
    }
    let g0 = 0,
      g1 = 0;
    for (let k = 0; k < mus.length; k++) {
      const w = ek[k] / z;
      g0 += w * (theta[0] - mus[k][0]);
      g1 += w * (theta[1] - mus[k][1]);
    }
    const out = new Float32Array(2);
    out[0] = covInv * g0;
    out[1] = covInv * g1;
    return out;
  }

  return { gradU, logDensity, mus, sigma2 };
}

/** Anisotropic 2D Gaussian N(0, diag(1, κ)). Standard target for §7's
 *  SGLD vs SGHMC mixing comparison. Returns gradU(θ) = Σ⁻¹ θ. */
export function anisotropic2DGradAndLogDensity(kappa: number): {
  gradU: (theta: Float32Array) => Float32Array;
  logDensity: (theta1: number, theta2: number) => number;
  kappa: number;
} {
  const inv1 = 1.0;
  const inv2 = 1.0 / kappa;
  const logNorm = -Math.log(2 * Math.PI) - 0.5 * Math.log(kappa);

  function gradU(theta: Float32Array): Float32Array {
    const out = new Float32Array(2);
    out[0] = inv1 * theta[0];
    out[1] = inv2 * theta[1];
    return out;
  }
  function logDensity(theta1: number, theta2: number): number {
    return logNorm - 0.5 * (inv1 * theta1 * theta1 + inv2 * theta2 * theta2);
  }
  return { gradU, logDensity, kappa };
}

/** Neal's funnel: v ~ N(0, σ_v²), x | v ~ N(0, exp(v)). Mirrors §10.1. */
export function funnelGradAndLogDensity(sigmaV: number = 3.0): {
  gradU: (theta: Float32Array) => Float32Array;
  logDensity: (v: number, x: number) => number;
  sigmaV: number;
} {
  const inv2 = 1 / (sigmaV * sigmaV);
  const logSigmaV = Math.log(sigmaV);
  const halfLog2Pi = 0.5 * Math.log(2 * Math.PI);

  function logDensity(v: number, x: number): number {
    // log N(v; 0, σ_v²) + log N(x; 0, exp(v))
    const lpV = -halfLog2Pi - logSigmaV - 0.5 * inv2 * v * v;
    const lpX = -halfLog2Pi - 0.5 * v - 0.5 * x * x * Math.exp(-v);
    return lpV + lpX;
  }
  function gradU(theta: Float32Array): Float32Array {
    const v = theta[0],
      x = theta[1];
    const expNegV = Math.exp(-v);
    const out = new Float32Array(2);
    out[0] = v * inv2 + 0.5 - 0.5 * x * x * expNegV;
    out[1] = x * expNegV;
    return out;
  }
  return { gradU, logDensity, sigmaV };
}

/** Diagonal Riemann metric for the funnel: G⁻¹(v, x) = diag(1, e^v). The
 *  divergence ∇·G⁻¹ vanishes identically (verified in §10.4). */
export function funnelDiagMetric(): {
  gInvDiag: (theta: Float32Array) => Float32Array;
  divGInv: (theta: Float32Array) => Float32Array;
} {
  function gInvDiag(theta: Float32Array): Float32Array {
    const out = new Float32Array(2);
    out[0] = 1.0;
    out[1] = Math.exp(theta[0]);
    return out;
  }
  function divGInv(_theta: Float32Array): Float32Array {
    return new Float32Array(2); // [0, 0]
  }
  return { gInvDiag, divGInv };
}

// =============================================================================
// BAYESIAN LINEAR REGRESSION (§§6, 8, 9 running BLR target)
// =============================================================================

export interface BLRSpec {
  /** [n × 2] design matrix in row-major flat form (intercept + slope). */
  XDesign: Float32Array;
  /** Length-n response. */
  yData: Float32Array;
  N: number;
  sigmaNoise: number;
  tauPrior: number;
}

export interface BLRPosterior {
  muPost: Float32Array; // length 2
  /** [2 × 2] flat row-major. */
  sigmaPost: Float32Array;
  /** [2 × 2] flat row-major. */
  sigmaPostInv: Float32Array;
}

/** Generate the §6 toy BLR dataset with a fixed seed. Default arguments
 *  match the notebook (N = 100, true θ = (1, 2), σ_noise = 0.5, τ_prior = 5). */
export function makeBLRDataset(
  seed: number,
  N: number = 100,
  trueTheta: [number, number] = [1.0, 2.0],
  sigmaNoise: number = 0.5,
  tauPrior: number = 5.0,
): BLRSpec & { trueTheta: [number, number] } {
  const rng = mulberry32(seed);
  const gauss = makeGauss(rng);
  const X = new Float32Array(N * 2);
  const y = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const xi = -2 + 4 * rng(); // U(-2, 2)
    X[i * 2 + 0] = 1;
    X[i * 2 + 1] = xi;
    y[i] = trueTheta[0] + trueTheta[1] * xi + sigmaNoise * gauss();
  }
  return { XDesign: X, yData: y, N, sigmaNoise, tauPrior, trueTheta };
}

/** Closed-form posterior of the §6 BLR. Returns μ_post, Σ_post, Σ_post⁻¹. */
export function blrPosterior(spec: BLRSpec): BLRPosterior {
  const { XDesign, yData, N, sigmaNoise, tauPrior } = spec;
  const inv2 = 1 / (sigmaNoise * sigmaNoise);
  // X^T X is 2x2:
  let xtx00 = 0,
    xtx01 = 0,
    xtx11 = 0;
  for (let i = 0; i < N; i++) {
    const a = XDesign[i * 2 + 0];
    const b = XDesign[i * 2 + 1];
    xtx00 += a * a;
    xtx01 += a * b;
    xtx11 += b * b;
  }
  const tauInv2 = 1 / (tauPrior * tauPrior);
  const A00 = tauInv2 + inv2 * xtx00;
  const A01 = inv2 * xtx01;
  const A11 = tauInv2 + inv2 * xtx11;
  // Σ_post⁻¹ = [[A00, A01], [A01, A11]]; invert 2x2.
  const det = A00 * A11 - A01 * A01;
  const S00 = A11 / det;
  const S01 = -A01 / det;
  const S11 = A00 / det;
  // X^T y:
  let xty0 = 0,
    xty1 = 0;
  for (let i = 0; i < N; i++) {
    xty0 += XDesign[i * 2 + 0] * yData[i];
    xty1 += XDesign[i * 2 + 1] * yData[i];
  }
  const b0 = inv2 * xty0;
  const b1 = inv2 * xty1;
  // μ_post = Σ_post · b
  const mu = new Float32Array(2);
  mu[0] = S00 * b0 + S01 * b1;
  mu[1] = S01 * b0 + S11 * b1;
  const sig = new Float32Array(4);
  sig[0] = S00;
  sig[1] = S01;
  sig[2] = S01;
  sig[3] = S11;
  const sigInv = new Float32Array(4);
  sigInv[0] = A00;
  sigInv[1] = A01;
  sigInv[2] = A01;
  sigInv[3] = A11;
  return { muPost: mu, sigmaPost: sig, sigmaPostInv: sigInv };
}

/** Full-data gradient of U(θ) = ½τ⁻²‖θ‖² + ½σ⁻² ‖Xθ - y‖².
 *  Reuses a single 2-element Float32Array across calls to avoid per-step
 *  allocations inside chain runners (the caller copies values out before
 *  the next gradient evaluation). */
export function blrGradFull(spec: BLRSpec): (theta: Float32Array) => Float32Array {
  const { XDesign, yData, N, sigmaNoise, tauPrior } = spec;
  const inv2 = 1 / (sigmaNoise * sigmaNoise);
  const tauInv2 = 1 / (tauPrior * tauPrior);
  const out = new Float32Array(2);
  return (theta: Float32Array) => {
    let g0 = tauInv2 * theta[0];
    let g1 = tauInv2 * theta[1];
    for (let i = 0; i < N; i++) {
      const a = XDesign[i * 2 + 0];
      const b = XDesign[i * 2 + 1];
      const r = a * theta[0] + b * theta[1] - yData[i];
      g0 += inv2 * a * r;
      g1 += inv2 * b * r;
    }
    out[0] = g0;
    out[1] = g1;
    return out;
  };
}

/** Mini-batch gradient of U(θ) using the (N/B)-rescaling of (6.2).
 *  Hoists a sampler (Fisher–Yates pool reused across calls) and a 2-element
 *  output buffer to avoid the allocation pressure of per-step
 *  `sampleWithoutReplacement` and `new Float32Array(2)` inside long chains. */
export function blrGradMinibatch(
  spec: BLRSpec,
  B: number,
  rng: () => number,
): (theta: Float32Array) => Float32Array {
  const { XDesign, yData, N, sigmaNoise, tauPrior } = spec;
  const inv2 = 1 / (sigmaNoise * sigmaNoise);
  const tauInv2 = 1 / (tauPrior * tauPrior);
  const scale = N / B;
  const sampler = makeSampler(N, B);
  const out = new Float32Array(2);
  return (theta: Float32Array) => {
    const idx = sampler(rng);
    let g0 = tauInv2 * theta[0];
    let g1 = tauInv2 * theta[1];
    for (let k = 0; k < B; k++) {
      const i = idx[k];
      const a = XDesign[i * 2 + 0];
      const b = XDesign[i * 2 + 1];
      const r = a * theta[0] + b * theta[1] - yData[i];
      g0 += scale * inv2 * a * r;
      g1 += scale * inv2 * b * r;
    }
    out[0] = g0;
    out[1] = g1;
    return out;
  };
}

/** Synthetic-noise SGLD gradient (§8.5): full gradient + N(0, σ_g²/B · I).
 *  Used to decouple η from B in the VZT bias verification panel. */
export function blrGradSyntheticNoise(
  spec: BLRSpec,
  sigmaG: number,
  Bsynth: number,
  rng: () => number,
): (theta: Float32Array) => Float32Array {
  const fullGrad = blrGradFull(spec);
  const gauss = makeGauss(rng);
  const noiseScale = sigmaG / Math.sqrt(Bsynth);
  return (theta: Float32Array) => {
    const g = fullGrad(theta);
    g[0] += noiseScale * gauss();
    g[1] += noiseScale * gauss();
    return g;
  };
}

// =============================================================================
// CLOSED-FORM TARGETS / DENSITIES
// =============================================================================

/** Ornstein–Uhlenbeck transition density at time t starting from x₀ = 0:
 *    p(x, t) = N(x; 0, σ²(1 - e^{-2αt})/(2α))
 *  Use t → ∞ to recover the stationary N(0, σ²/(2α)). */
export function ouDensity(x: number, t: number, alpha: number, sigma2: number): number {
  if (t <= 0) {
    // δ at 0: not a density; return 0 except when x==0.
    return x === 0 ? Infinity : 0;
  }
  const v = (sigma2 * (1 - Math.exp(-2 * alpha * t))) / (2 * alpha);
  if (v <= 0) return 0;
  return Math.exp(-(x * x) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
}

/** Standard 1D Brownian motion path on [0, T] at uniform spacing dt. */
export function brownianPath(
  T: number,
  dt: number,
  rng: () => number,
): { t: Float32Array; w: Float32Array } {
  const n = Math.floor(T / dt) + 1;
  const t = new Float32Array(n);
  const w = new Float32Array(n);
  const gauss = makeGauss(rng);
  const sqrtDt = Math.sqrt(dt);
  for (let i = 1; i < n; i++) {
    t[i] = i * dt;
    w[i] = w[i - 1] + sqrtDt * gauss();
  }
  return { t, w };
}

/** Specialized helper for the §3 viz histogram panel: simulate K independent
 *  Brownian paths to the terminal time T (step size dt) and return only the
 *  K terminal values W_T. Avoids allocating 2K full-path Float32Arrays when
 *  the caller only needs the end-point. */
export function brownianTerminalValues(
  T: number,
  dt: number,
  K: number,
  rng: () => number,
): Float32Array {
  const out = new Float32Array(K);
  const gauss = makeGauss(rng);
  const sqrtDt = Math.sqrt(dt);
  const nSteps = Math.floor(T / dt);
  for (let k = 0; k < K; k++) {
    let w = 0;
    for (let i = 0; i < nSteps; i++) w += sqrtDt * gauss();
    out[k] = w;
  }
  return out;
}

// =============================================================================
// UTILITIES FOR VIZ PANELS
// =============================================================================

/** Compute the running mean of a 1D trace for the §6 / §9 convergence panels. */
export function runningMean(x: ArrayLike<number>): Float32Array {
  const n = x.length;
  const out = new Float32Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += x[i];
    out[i] = s / (i + 1);
  }
  return out;
}

/** Histogram a 1D trace into nBins equal-width bins on [lo, hi]. Returns
 *  density per bin: each bin's value is (count_in_bin) / (total · binWidth)
 *  where total is `x.length`. Samples outside [lo, hi) are dropped, so when
 *  the trace has tail mass beyond the window, sum(density) · binWidth equals
 *  the in-window fraction (≤ 1) — not 1. Pick [lo, hi] wide enough to cover
 *  the bulk of the distribution if you want the histogram to integrate to 1. */
export function histogram(
  x: ArrayLike<number>,
  lo: number,
  hi: number,
  nBins: number,
): { centers: Float32Array; density: Float32Array } {
  const w = (hi - lo) / nBins;
  const counts = new Float32Array(nBins);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (v < lo || v >= hi) continue;
    const k = Math.min(nBins - 1, Math.floor((v - lo) / w));
    counts[k]++;
  }
  const total = x.length;
  const centers = new Float32Array(nBins);
  const density = new Float32Array(nBins);
  for (let k = 0; k < nBins; k++) {
    centers[k] = lo + (k + 0.5) * w;
    density[k] = total > 0 ? counts[k] / (total * w) : 0;
  }
  return { centers, density };
}

/** Quick Welling–Teh polynomial schedule η_n = a / (b + n)^γ for §6.3 demos. */
export function wellingTehSchedule(
  a: number = 8e-3,
  b: number = 50,
  gamma: number = 0.55,
): (n: number) => number {
  return (n: number) => a / Math.pow(b + n, gamma);
}

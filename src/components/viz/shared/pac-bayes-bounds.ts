// =============================================================================
// pac-bayes-bounds.ts
//
// Shared math primitives for the pac-bayes-bounds topic's viz components.
// Ported from notebooks/pac-bayes-bounds/01_pac_bayes_bounds.ipynb §§1–11.
// In-browser TS for all the §§1–10 viz (closed-form bounds + Gibbs-on-grid +
// bisection KL inversion); the §11 DznRReproduction consumes precomputed JSON
// from public/sample-data/pac-bayes-bounds/dziugaite_roy.json.
//
// All exports are pure functions — deterministic given a seeded RNG.
// mulberry32 and gaussianFrom are re-exported from the predecessor topic's
// shared module (generalization-bounds.ts) so viz components only need to
// import this module.
//
// Source-of-truth notebook: notebooks/pac-bayes-bounds/01_pac_bayes_bounds.ipynb
// =============================================================================

import { gaussianFrom, mulberry32 } from './generalization-bounds';

export { gaussianFrom, mulberry32 };

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/** Standard-Gaussian-X / sign-Y sample for the §1.5 running example. */
export type NormalThresholdSample = { X: Float64Array; Y: Int8Array };

/** Three-way decomposition of the Catoni-bound RHS at fixed λ. */
export type CatoniDecomposition = {
  empiricalRisk: number;
  linearizationPenalty: number; // λ / (8n)
  klOverLambda: number;         // (KL + log(1/δ)) / λ
  total: number;
};

// -----------------------------------------------------------------------------
// Standard-normal CDF (Abramowitz–Stegun erf 7.1.26 approx; max abs error ≈ 1.5e-7).
// -----------------------------------------------------------------------------

/** Φ(x) — standard normal CDF. */
export function normalCdf(x: number): number {
  // erf(x) via A&S 7.1.26
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * ax);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const erf = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1 + sign * erf);
}

// -----------------------------------------------------------------------------
// §1.5 / §2 Running example — Gaussian-X threshold classification
// -----------------------------------------------------------------------------

/**
 * Sample n iid (X, Y) with X ~ N(0, 1), Y = sign(X - μ*) ⊕ Bernoulli(η).
 * Labels are encoded as Int8Array in {-1, +1}.
 */
export function sampleNormalThresholdProblem(
  n: number,
  eta: number,
  muStar: number,
  rng: () => number,
): NormalThresholdSample {
  const X = new Float64Array(n);
  const Y = new Int8Array(n);
  const gauss = gaussianFrom(rng);
  for (let i = 0; i < n; i++) {
    const x = gauss();
    X[i] = x;
    const yClean = x >= muStar ? 1 : -1;
    const flip = rng() < eta ? -1 : 1;
    Y[i] = (yClean * flip) as -1 | 1;
  }
  return { X, Y };
}

/**
 * Build the 101-point threshold grid T = {-2, -1.96, ..., 1.96, 2}
 * used throughout §§1–10 of the brief.
 */
export function thresholdGrid(min: number = -2, max: number = 2, count: number = 101): Float64Array {
  const grid = new Float64Array(count);
  const step = (max - min) / (count - 1);
  for (let i = 0; i < count; i++) grid[i] = min + i * step;
  return grid;
}

/** Empirical 0/1 risk on the threshold grid: returns Float64Array of risks parallel to grid. */
export function empiricalRiskOnGrid(
  X: Float64Array,
  Y: Int8Array,
  grid: Float64Array,
): Float64Array {
  const n = X.length;
  const k = grid.length;
  const risks = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    const t = grid[j];
    let errors = 0;
    for (let i = 0; i < n; i++) {
      const yhat = X[i] >= t ? 1 : -1;
      if (yhat !== Y[i]) errors++;
    }
    risks[j] = errors / n;
  }
  return risks;
}

/** ERM over the threshold grid: returns (tau, riskHat). */
export function ermOnGrid(grid: Float64Array, risks: Float64Array): { tau: number; risk: number } {
  let best = 0;
  for (let j = 1; j < grid.length; j++) {
    if (risks[j] < risks[best]) best = j;
  }
  return { tau: grid[best], risk: risks[best] };
}

/**
 * Closed-form true 0/1 risk for h_t under X ~ N(0, 1) with label-flip rate η:
 *   R(h_t) = η + (1 - 2η) · |Φ(t) - Φ(μ*)|
 * — the probability of being in the "wrong half" between t and μ*, modulated by noise.
 */
export function trueRiskOnGrid(grid: Float64Array, eta: number, muStar: number): Float64Array {
  const PhiStar = normalCdf(muStar);
  const out = new Float64Array(grid.length);
  for (let j = 0; j < grid.length; j++) {
    const Phi = normalCdf(grid[j]);
    out[j] = eta + (1 - 2 * eta) * Math.abs(Phi - PhiStar);
  }
  return out;
}

// -----------------------------------------------------------------------------
// §2.5 Posterior builders + KL on discrete grids
// -----------------------------------------------------------------------------

/**
 * Build a Gaussian-shaped posterior on the threshold grid centered at `center`,
 * bandwidth `sigmaQ`, against a discrete prior `P` of the same length.
 *   Q[i] ∝ P[i] · exp(-((T[i] - center)² / (2·σ_Q²))).
 * Returns Q renormalized to sum to 1.
 */
export function gaussianOnGridPosterior(
  grid: Float64Array,
  center: number,
  sigmaQ: number,
  prior: Float64Array,
): Float64Array {
  const k = grid.length;
  const logQ = new Float64Array(k);
  const inv2s2 = 1 / (2 * sigmaQ * sigmaQ);
  for (let i = 0; i < k; i++) {
    const d = grid[i] - center;
    logQ[i] = Math.log(prior[i]) - d * d * inv2s2;
  }
  return normalizeLog(logQ);
}

/** KL divergence between two discrete distributions on the same support: Σ Q log(Q/P). */
export function klDiscrete(Q: Float64Array, P: Float64Array): number {
  let kl = 0;
  for (let i = 0; i < Q.length; i++) {
    if (Q[i] > 0) kl += Q[i] * Math.log(Q[i] / P[i]);
  }
  return kl;
}

/** Σ Q[i] · f[i] — the workhorse "expected X under Q" computation. */
export function expectedUnderQ(Q: Float64Array, f: Float64Array): number {
  let s = 0;
  for (let i = 0; i < Q.length; i++) s += Q[i] * f[i];
  return s;
}

// -----------------------------------------------------------------------------
// §3 / §6 Gibbs distribution + change-of-measure variational functional
// -----------------------------------------------------------------------------

/**
 * Gibbs distribution Q*_λ(h) ∝ P(h) · exp(-λ · risks[h]) on a finite grid.
 * Log-stable via subtract-max trick (handles λ up to ~10⁴ safely in double precision).
 */
export function gibbsOnGrid(prior: Float64Array, risks: Float64Array, lambda: number): Float64Array {
  const k = prior.length;
  const logQ = new Float64Array(k);
  for (let i = 0; i < k; i++) logQ[i] = Math.log(prior[i]) - lambda * risks[i];
  return normalizeLog(logQ);
}

/** log-sum-exp on Float64Array, subtract-max stable. */
export function logSumExp(arr: Float64Array): number {
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
  if (!Number.isFinite(max)) return max;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += Math.exp(arr[i] - max);
  return max + Math.log(s);
}

/**
 * Variational functional g(Q) = E_Q[f] − KL(Q ∥ P). At Q = Q* (Gibbs tilt) the
 * functional equals log E_P[e^f] (Donsker–Varadhan envelope).
 */
export function variationalFunctional(Q: Float64Array, P: Float64Array, f: Float64Array): number {
  return expectedUnderQ(Q, f) - klDiscrete(Q, P);
}

/** log E_P[e^f] — universal envelope of the variational functional. */
export function logExpEnvelope(P: Float64Array, f: Float64Array): number {
  const logQ = new Float64Array(P.length);
  for (let i = 0; i < P.length; i++) logQ[i] = Math.log(P[i]) + f[i];
  return logSumExp(logQ);
}

// -----------------------------------------------------------------------------
// §4–§5 Binary-KL + brentq-style inversion
// -----------------------------------------------------------------------------

/**
 * Binary (Bernoulli) KL divergence kl(p ∥ q) = p log(p/q) + (1-p) log((1-p)/(1-q)).
 * Handles the boundary cases p ∈ {0, 1} robustly (returns −log(1-q) and −log(q)
 * respectively, per the limit p · log(p) → 0).
 */
export function binaryKL(p: number, q: number): number {
  if (q <= 0 || q >= 1) return Number.POSITIVE_INFINITY;
  if (p <= 0) return -Math.log(1 - q);
  if (p >= 1) return -Math.log(q);
  return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
}

/**
 * Invert kl(p̂ ∥ q) = c for q ∈ [p̂, 1) — the §5.4 upper-confidence root.
 * The function q ↦ kl(p̂ ∥ q) is strictly increasing on [p̂, 1), so bisection
 * converges monotonically to machine precision in ~52 iterations.
 *
 * Vacuous-regime detection: if c ≥ −log(1 − p̂), no finite q solves the equation
 * — return 1.
 */
export function klInversionUpper(pHat: number, c: number): number {
  if (c <= 0) return pHat;
  if (pHat >= 1) return 1;
  if (pHat <= 0) {
    // kl(0 ∥ q) = −log(1−q); solving = c gives q = 1 − e^(−c).
    return 1 - Math.exp(-c);
  }
  const cVacuous = -Math.log(1 - pHat);
  if (c >= cVacuous) return 1;
  let lo = pHat;
  let hi = 1 - 1e-15;
  for (let iter = 0; iter < 80; iter++) {
    const mid = 0.5 * (lo + hi);
    if (binaryKL(pHat, mid) < c) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-14) break;
  }
  return 0.5 * (lo + hi);
}

// -----------------------------------------------------------------------------
// §4–§7 Certificate functionals
// -----------------------------------------------------------------------------

/**
 * McAllester (Maurer-tightened) bound on E_Q[R(h)]:
 *   E_Q[R] ≤ E_Q[R̂_S] + √((KL(Q ∥ P) + log(2√n/δ)) / (2n)).
 */
export function mcAllesterCertificate(
  empRiskExpUnderQ: number,
  klQP: number,
  n: number,
  delta: number,
): number {
  const slack = Math.sqrt((klQP + Math.log((2 * Math.sqrt(n)) / delta)) / (2 * n));
  return empRiskExpUnderQ + slack;
}

/** McAllester's slack term in isolation (square-root component above E_Q[R̂_S]). */
export function mcAllesterSlack(klQP: number, n: number, delta: number): number {
  return Math.sqrt((klQP + Math.log((2 * Math.sqrt(n)) / delta)) / (2 * n));
}

/**
 * Seeger / kl-form certificate: upper-confidence inversion of
 *   kl(E_Q[R̂_S] ∥ E_Q[R]) ≤ (KL(Q ∥ P) + log(2√n/δ)) / n.
 */
export function seegerCertificate(
  empRiskExpUnderQ: number,
  klQP: number,
  n: number,
  delta: number,
): number {
  const c = (klQP + Math.log((2 * Math.sqrt(n)) / delta)) / n;
  return klInversionUpper(empRiskExpUnderQ, c);
}

/**
 * Catoni linear bound at fixed temperature λ:
 *   E_Q[R] ≤ E_Q[R̂_S] + λ/(8n) + (KL(Q ∥ P) + log(1/δ)) / λ.
 */
export function catoniCertificate(
  empRiskExpUnderQ: number,
  klQP: number,
  n: number,
  delta: number,
  lambda: number,
): number {
  return (
    empRiskExpUnderQ + lambda / (8 * n) + (klQP + Math.log(1 / delta)) / lambda
  );
}

/**
 * Catoni-optimal λ* (closed form from §6.4 ∂/∂λ = 0):
 *   λ* = √(8n · (KL(Q ∥ P) + log(1/δ))).
 * Plugging back collapses the RHS to McAllester-shape with log(1/δ) in place of
 * log(2√n/δ) — a (½ log n)-nat tightening.
 */
export function catoniOptimalLambda(klQP: number, n: number, delta: number): number {
  return Math.sqrt(8 * n * (klQP + Math.log(1 / delta)));
}

/** Catoni-optimized certificate at λ*. */
export function catoniOptimizedCertificate(
  empRiskExpUnderQ: number,
  klQP: number,
  n: number,
  delta: number,
): number {
  return empRiskExpUnderQ + Math.sqrt((klQP + Math.log(1 / delta)) / (2 * n));
}

/** Catoni-bound three-way decomposition at fixed λ (for §6.5b visualization). */
export function catoniDecomposition(
  empRiskExpUnderQ: number,
  klQP: number,
  n: number,
  delta: number,
  lambda: number,
): CatoniDecomposition {
  const linearizationPenalty = lambda / (8 * n);
  const klOverLambda = (klQP + Math.log(1 / delta)) / lambda;
  return {
    empiricalRisk: empRiskExpUnderQ,
    linearizationPenalty,
    klOverLambda,
    total: empRiskExpUnderQ + linearizationPenalty + klOverLambda,
  };
}

// -----------------------------------------------------------------------------
// §7 Tolstikhin–Seldin empirical-Bernstein vs Catoni-Hoeffding slack
// -----------------------------------------------------------------------------

/** Catoni-Hoeffding slack at optimal λ — same shape as Catoni-optimized minus E_Q[R̂_S]. */
export function catoniHoeffdingSlack(klQP: number, n: number, delta: number): number {
  return Math.sqrt((klQP + Math.log(1 / delta)) / (2 * n));
}

/**
 * Tolstikhin–Seldin empirical-Bernstein PAC-Bayes slack (brief §7.2). With
 * variance V_n = p̂(1 − p̂) for Bernoulli losses and union-grid size K:
 *   slack = √(2 V_n (KL + log(K/δ)) / n)  +  7 (KL + log(K/δ)) / (3(n − 1)).
 */
export function tolstikhinSeldinSlack(
  pHat: number,
  klQP: number,
  n: number,
  delta: number,
  K: number,
): number {
  const Vn = pHat * (1 - pHat);
  const numerator = klQP + Math.log(K / delta);
  return (
    Math.sqrt((2 * Vn * numerator) / n) + (7 * numerator) / (3 * (n - 1))
  );
}

// -----------------------------------------------------------------------------
// §8 Gaussian–Gaussian KL (Proposition 1; diagonal-isotropic case)
// -----------------------------------------------------------------------------

/**
 * KL(Q ∥ P) for Q = N(μ_Q, σ²_Q I_d), P = N(0, σ²_P I_d) — diagonal-isotropic
 * scalar form (brief §8.2 specialization). Vectorized via reduced norm-squared.
 *
 *   KL = ‖μ_Q‖² / (2 σ²_P) + (d/2) · [σ²_Q/σ²_P − 1 − log(σ²_Q/σ²_P)].
 */
export function gaussianKLIsotropic(
  muQNormSq: number,
  sigmaQ2: number,
  sigmaP2: number,
  d: number,
): number {
  const rho = sigmaQ2 / sigmaP2;
  return muQNormSq / (2 * sigmaP2) + (d / 2) * (rho - 1 - Math.log(rho));
}

/**
 * Full diagonal-Gaussian KL with per-coordinate σ_{Q,i}² (brief §8.2 / §11):
 *   KL = ‖μ_Q‖²/(2σ²_P) + ½ Σ_i [σ²_{Q,i}/σ²_P − 1 − log(σ²_{Q,i}/σ²_P)].
 */
export function gaussianKLPerCoord(
  muQ: Float64Array,
  sigmaQ2: Float64Array,
  sigmaP2: number,
): number {
  let muSq = 0;
  let varSum = 0;
  for (let i = 0; i < muQ.length; i++) {
    muSq += muQ[i] * muQ[i];
    const rho = sigmaQ2[i] / sigmaP2;
    varSum += rho - 1 - Math.log(rho);
  }
  return muSq / (2 * sigmaP2) + 0.5 * varSum;
}

// -----------------------------------------------------------------------------
// §10 Union-bound baseline (re-exported convenience; reuses generalization-bounds.ts)
// -----------------------------------------------------------------------------

/**
 * Finite-class union-bound certificate (brief §10.1):
 *   sup_h R(h) − R̂_S(h) ≤ √(log(2|H|/δ) / (2n)).
 * Provided here as a co-located helper so PAC-Bayes viz don't reach into
 * generalization-bounds.ts directly.
 */
export function finiteClassUnionBound(N: number, n: number, delta: number): number {
  return Math.sqrt(Math.log((2 * N) / delta) / (2 * n));
}

// -----------------------------------------------------------------------------
// Internal — log-array → probability-array normalizer (subtract-max stable).
// -----------------------------------------------------------------------------

function normalizeLog(logQ: Float64Array): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < logQ.length; i++) if (logQ[i] > max) max = logQ[i];
  const Q = new Float64Array(logQ.length);
  let s = 0;
  for (let i = 0; i < logQ.length; i++) {
    Q[i] = Math.exp(logQ[i] - max);
    s += Q[i];
  }
  for (let i = 0; i < logQ.length; i++) Q[i] /= s;
  return Q;
}

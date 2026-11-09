// =============================================================================
// reverseKLFit.ts — single-Gaussian projections of a 1-D Gaussian mixture.
//
// Used by ForwardVsReverseKL (variational-inference) and
// ForwardReverseKLExplorer (kl-divergence).
//
//   forward KL  argmin_q KL(p ∥ q)  — closed form: match the mixture's mean
//                                     and variance (moment matching).
//   reverse KL  argmin_q KL(q ∥ p)  — no closed form; minimized numerically
//                                     over (μ, log σ).
//
// The reverse-KL objective is not convex in (μ, σ): for a two-mode target it
// has a "covering" minimum between the modes and a "mode-locked" minimum on
// each mode, and which one is global depends on the separation. For an equal
// mixture of N(±Δ/2, 1) the covering Gaussian wins until Δ ≈ 5.8 (at Δ = 4 it
// is N(0, 2.05²) with KL 0.227, against 0.633 for a fit locked on one mode).
// So the minimizer searches globally: a coarse (μ, log σ) grid, then compass
// refinement started from every component mean, from the moment-matched
// center and from the best grid cell, keeping every distinct local minimum.
// =============================================================================

export interface MixtureComponent {
  w: number;
  mu: number;
  sigma: number;
}

export interface GaussianFit {
  mu: number;
  sigma: number;
  /** KL(q ∥ p) at this fit. */
  kl: number;
}

export interface ReverseKLResult {
  /** Global minimizer of KL(q ∥ p) over single Gaussians q. */
  best: GaussianFit;
  /** Every distinct local minimum found, sorted by KL (best first). */
  minima: GaussianFit[];
}

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

/** log p(x) for the mixture, by log-sum-exp (no underflow in far tails). */
export function mixtureLogPdf(x: number, comps: MixtureComponent[]): number {
  let m = -Infinity;
  const terms = new Array<number>(comps.length);
  for (let j = 0; j < comps.length; j++) {
    const c = comps[j];
    const z = (x - c.mu) / c.sigma;
    terms[j] = Math.log(c.w) - 0.5 * z * z - Math.log(c.sigma) - LOG_SQRT_2PI;
    if (terms[j] > m) m = terms[j];
  }
  let s = 0;
  for (const t of terms) s += Math.exp(t - m);
  return m + Math.log(s);
}

export function mixturePdf(x: number, comps: MixtureComponent[]): number {
  return Math.exp(mixtureLogPdf(x, comps));
}

/** Forward-KL projection: the mixture's mean and standard deviation. */
export function forwardKLFit(comps: MixtureComponent[]): { mu: number; sigma: number } {
  let mean = 0;
  for (const c of comps) mean += c.w * c.mu;
  let variance = 0;
  for (const c of comps) variance += c.w * (c.sigma * c.sigma + (c.mu - mean) ** 2);
  return { mu: mean, sigma: Math.sqrt(variance) };
}

// Quadrature nodes for E_{z~N(0,1)}[f(z)]: trapezoid on [−Z_MAX, Z_MAX].
// The integrand is Gaussian-weighted and smooth, so the trapezoid rule is
// accurate to ~1e-12 here.
const Z_MAX = 9;
const N_Z = 241;
const Z_NODES: Float64Array = new Float64Array(N_Z);
const Z_WEIGHTS: Float64Array = new Float64Array(N_Z);
(() => {
  const h = (2 * Z_MAX) / (N_Z - 1);
  for (let i = 0; i < N_Z; i++) {
    const z = -Z_MAX + i * h;
    Z_NODES[i] = z;
    const edge = i === 0 || i === N_Z - 1 ? 0.5 : 1;
    Z_WEIGHTS[i] = edge * h * Math.exp(-0.5 * z * z - LOG_SQRT_2PI);
  }
})();

/**
 * KL(q ∥ p) for q = N(μ, σ²): −H(q) − E_q[log p], with the Gaussian entropy
 * H(q) = ½ log(2πeσ²) in closed form and E_q[log p] by grid quadrature.
 */
export function reverseKL(mu: number, sigma: number, comps: MixtureComponent[]): number {
  let crossEntropy = 0;
  for (let i = 0; i < N_Z; i++) {
    crossEntropy -= Z_WEIGHTS[i] * mixtureLogPdf(mu + sigma * Z_NODES[i], comps);
  }
  const entropy = 0.5 + Math.log(sigma) + LOG_SQRT_2PI;
  return crossEntropy - entropy;
}

// Compass (pattern) search on (μ, s = log σ) from one start.
function refine(
  mu0: number,
  s0: number,
  comps: MixtureComponent[],
  muStep0: number,
): GaussianFit {
  let mu = mu0;
  let s = s0;
  let f = reverseKL(mu, Math.exp(s), comps);
  let dMu = muStep0;
  let dS = 0.25;
  while (dMu > 1e-7 || dS > 1e-7) {
    let moved = false;
    const moves: [number, number][] = [
      [dMu, 0], [-dMu, 0], [0, dS], [0, -dS],
    ];
    for (const [a, b] of moves) {
      const fNew = reverseKL(mu + a, Math.exp(s + b), comps);
      if (fNew < f - 1e-15) {
        mu += a;
        s += b;
        f = fNew;
        moved = true;
        break;
      }
    }
    if (!moved) {
      dMu *= 0.5;
      dS *= 0.5;
    }
  }
  return { mu, sigma: Math.exp(s), kl: f };
}

/**
 * Global reverse-KL projection of a Gaussian mixture onto single Gaussians.
 * Ties between symmetric minima (an equal two-mode mixture) resolve to the
 * larger μ, so the default view locks onto the right-hand mode.
 */
export function fitReverseKL(comps: MixtureComponent[]): ReverseKLResult {
  const fwd = forwardKLFit(comps);
  let lo = Infinity;
  let hi = -Infinity;
  let sMin = Infinity;
  for (const c of comps) {
    lo = Math.min(lo, c.mu - 2 * c.sigma);
    hi = Math.max(hi, c.mu + 2 * c.sigma);
    sMin = Math.min(sMin, c.sigma);
  }
  const logLo = Math.log(0.25 * sMin);
  const logHi = Math.log(1.5 * fwd.sigma);

  // Coarse grid.
  const N_MU = 31;
  const N_S = 20;
  let gridBest = { mu: fwd.mu, s: Math.log(fwd.sigma), kl: Infinity };
  for (let a = 0; a < N_MU; a++) {
    const mu = lo + (a / (N_MU - 1)) * (hi - lo);
    for (let b = 0; b < N_S; b++) {
      const s = logLo + (b / (N_S - 1)) * (logHi - logLo);
      const kl = reverseKL(mu, Math.exp(s), comps);
      if (kl < gridBest.kl) gridBest = { mu, s, kl };
    }
  }

  // Local refinement from each mode, the center, and the best grid cell.
  const muStep = 0.25 * fwd.sigma;
  const starts: [number, number][] = [
    ...comps.map((c): [number, number] => [c.mu, Math.log(c.sigma)]),
    [fwd.mu, Math.log(fwd.sigma)],
    [gridBest.mu, gridBest.s],
  ];
  const found = starts.map(([m, s]) => refine(m, s, comps, muStep));

  // Deduplicate converged points.
  const minima: GaussianFit[] = [];
  for (const fit of found) {
    const dup = minima.find(
      (m) =>
        Math.abs(m.mu - fit.mu) < 1e-3 * fwd.sigma &&
        Math.abs(Math.log(m.sigma / fit.sigma)) < 1e-3,
    );
    if (dup) {
      if (fit.kl < dup.kl) Object.assign(dup, fit);
    } else minima.push({ ...fit });
  }
  minima.sort((p, q) => (Math.abs(p.kl - q.kl) < 1e-9 ? q.mu - p.mu : p.kl - q.kl));
  return { best: minima[0], minima };
}

/**
 * Whether a fit sits on a single component: centered within one σ_j of μ_j
 * with width at most 1.5 σ_j. Otherwise it spans several modes.
 */
export function lockedComponent(fit: GaussianFit, comps: MixtureComponent[]): number {
  for (let j = 0; j < comps.length; j++) {
    const c = comps[j];
    if (Math.abs(fit.mu - c.mu) < c.sigma && fit.sigma <= 1.5 * c.sigma) return j;
  }
  return -1;
}

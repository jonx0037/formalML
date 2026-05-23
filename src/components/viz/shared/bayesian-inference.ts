// =============================================================================
// bayesian-inference.ts
//
// Shared math/types/palette for the T5 Bayesian & Probabilistic ML umbrella.
// Mirrors notebooks/bayesian-inference/01_bayesian_inference.ipynb in
// TypeScript: Beta–Binomial conjugacy, Normal–Normal precision-weighting,
// posterior summaries (MAP / mean / median / central-CI / HPD), the
// closed-form Normal-vs-Normal Bayes factor of §10.3, hierarchical empirical
// Bayes for the §9 Beta–Binomial hierarchy, and 1-D quadrature for §11's
// Student-t non-conjugate demo.
//
// Tolerances in verify-bayesian-inference.ts reflect cross-RNG drift between
// NumPy's PCG64 and our seeded mulberry32 — exact MC values diverge, but the
// closed-form identities (posterior parameters, log-marginal at given tau,
// Theorem 6.1's tau_n^2 + sigma^2 identity) are bit-pinnable.
// =============================================================================

// =============================================================================
// PALETTE
// =============================================================================
// Matches the notebook color scheme (precompute_*.py preamble).

export const COLOR_VARIANT_A = '#1f4e79';     // deep blue   — prior / variant A
export const COLOR_VARIANT_B = '#c0504d';     // red         — posterior / variant B
export const COLOR_LIKELIHOOD = '#3a6e3a';    // forest      — likelihood / truth
export const COLOR_ACCENT = '#9467bd';        // purple      — posterior predictive
export const COLOR_DATA = '#7f7f7f';          // gray        — raw data
export const COLOR_BAND = '#ff7f0e';          // orange      — decision / highlight bands

// =============================================================================
// SPECIAL FUNCTIONS — lgamma, logBeta, regularized incomplete beta
// =============================================================================

const G_LANCZOS = 7;
const C_LANCZOS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/** Log of the gamma function via Lanczos (g=7). Accurate to ~15 decimal places. */
export function lgamma(x: number): number {
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  x -= 1;
  let a = C_LANCZOS[0];
  const t = x + G_LANCZOS + 0.5;
  for (let i = 1; i < G_LANCZOS + 2; i++) a += C_LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b). */
export function logBeta(a: number, b: number): number {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

const BETACF_EPS = 1e-12;
const BETACF_FPMIN = 1e-300;
const BETACF_MAXIT = 500;

/** Lentz's continued fraction for the regularized incomplete beta function.
 *  Used inside {@link betaCdf}. */
function betacf(a: number, b: number, x: number): number {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < BETACF_FPMIN) d = BETACF_FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= BETACF_MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < BETACF_FPMIN) d = BETACF_FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < BETACF_FPMIN) c = BETACF_FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < BETACF_FPMIN) d = BETACF_FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < BETACF_FPMIN) c = BETACF_FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < BETACF_EPS) return h;
  }
  return h;
}

// =============================================================================
// BETA DISTRIBUTION
// =============================================================================

export function betaPdf(theta: number, alpha: number, beta: number): number {
  if (theta <= 0 || theta >= 1) return 0;
  return Math.exp(betaLogPdf(theta, alpha, beta));
}

export function betaLogPdf(theta: number, alpha: number, beta: number): number {
  if (theta <= 0 || theta >= 1) return -Infinity;
  return (alpha - 1) * Math.log(theta) + (beta - 1) * Math.log(1 - theta) - logBeta(alpha, beta);
}

export function betaMean(alpha: number, beta: number): number {
  return alpha / (alpha + beta);
}

export function betaVariance(alpha: number, beta: number): number {
  const s = alpha + beta;
  return (alpha * beta) / (s * s * (s + 1));
}

/** Posterior mode of Beta(alpha, beta) — defined only when alpha, beta > 1. */
export function betaMode(alpha: number, beta: number): number {
  if (alpha <= 1 || beta <= 1) return Number.NaN;
  return (alpha - 1) / (alpha + beta - 2);
}

/** Regularized incomplete beta function I_x(a, b) = P(Beta(a,b) <= x). */
export function betaCdf(theta: number, alpha: number, beta: number): number {
  if (theta <= 0) return 0;
  if (theta >= 1) return 1;
  const logBt =
    lgamma(alpha + beta) -
    lgamma(alpha) -
    lgamma(beta) +
    alpha * Math.log(theta) +
    beta * Math.log(1 - theta);
  const bt = Math.exp(logBt);
  if (theta < (alpha + 1) / (alpha + beta + 2)) {
    return (bt * betacf(alpha, beta, theta)) / alpha;
  }
  return 1 - (bt * betacf(beta, alpha, 1 - theta)) / beta;
}

/** Inverse CDF (quantile) via bisection. Accurate to ~1e-10 in ~35 iters. */
export function betaQuantile(p: number, alpha: number, beta: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (betaCdf(mid, alpha, beta) < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12) break;
  }
  return 0.5 * (lo + hi);
}

/** Central credible interval at given level (e.g. 0.95 → [2.5%, 97.5%]). */
export function betaCentralCI(
  alpha: number,
  beta: number,
  level = 0.95,
): { lower: number; upper: number } {
  const tail = (1 - level) / 2;
  return { lower: betaQuantile(tail, alpha, beta), upper: betaQuantile(1 - tail, alpha, beta) };
}

/** Highest-posterior-density interval for Beta(alpha, beta).
 *
 * For a unimodal Beta (alpha, beta > 1) we solve the system
 *   pdf(lower) = pdf(upper),  cdf(upper) - cdf(lower) = level
 * by nested bisection: the outer bisection finds the lower endpoint that
 * yields total mass = level; the inner bisection finds the upper endpoint
 * (right of the mode) with equal pdf. For U-shaped or J-shaped posteriors
 * the HPD is multi-modal; we fall back to the central CI in that case.
 */
export function betaHPD(
  alpha: number,
  beta: number,
  level = 0.95,
): { lower: number; upper: number } {
  if (alpha <= 1 || beta <= 1) return betaCentralCI(alpha, beta, level);
  const mode = betaMode(alpha, beta);
  // Inner: given a candidate lower (in (0, mode)), find upper > mode where
  // pdf(upper) = pdf(lower) by bisection.
  function upperForLower(lower: number): number {
    const pdfLow = betaPdf(lower, alpha, beta);
    let lo = mode;
    let hi = 1 - 1e-12;
    for (let k = 0; k < 80; k++) {
      const mid = 0.5 * (lo + hi);
      if (betaPdf(mid, alpha, beta) > pdfLow) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }
  // Outer: mass(0) = 1; mass(mode) = 0. mass is monotone decreasing in lower
  // on (0, mode). Bisect for mass(lower) = level.
  function massForLower(lower: number): number {
    const upper = upperForLower(lower);
    return betaCdf(upper, alpha, beta) - betaCdf(lower, alpha, beta);
  }
  let lo = 1e-12;
  let hi = mode - 1e-12;
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (massForLower(mid) > level) lo = mid;
    else hi = mid;
  }
  const lower = 0.5 * (lo + hi);
  return { lower, upper: upperForLower(lower) };
}

// =============================================================================
// BETA–BINOMIAL CONJUGACY (Theorem 4.2)
// =============================================================================

/** Beta(alpha + s, beta + n - s) — pseudo-counts plus observed counts. */
export function betaBinomialUpdate(
  alpha: number,
  beta: number,
  n: number,
  s: number,
): { alpha: number; beta: number } {
  return { alpha: alpha + s, beta: beta + n - s };
}

/** log P(s | n, alpha, beta) — Beta-Binomial PMF on the log scale.
 *  Used by hierarchical EB and the §10 marginal-likelihood demos. */
export function betaBinomialLogPmf(
  s: number,
  n: number,
  alpha: number,
  beta: number,
): number {
  const logBinom = lgamma(n + 1) - lgamma(s + 1) - lgamma(n - s + 1);
  return logBinom + logBeta(alpha + s, beta + n - s) - logBeta(alpha, beta);
}

// =============================================================================
// NORMAL–NORMAL CONJUGACY (Theorem 4.3) AND POSTERIOR PREDICTIVE (Theorem 6.1)
// =============================================================================

export interface NormalNormalInput {
  mu0: number;
  tau0Sq: number;
  sigmaSq: number;
  yBar: number;
  n: number;
}

export interface NormalNormalPosterior {
  muN: number;
  tauNSq: number;
  /** Posterior precision = 1/tau0Sq + n/sigmaSq. */
  precision: number;
}

export function normalNormalUpdate(inp: NormalNormalInput): NormalNormalPosterior {
  const precision = 1 / inp.tau0Sq + inp.n / inp.sigmaSq;
  const tauNSq = 1 / precision;
  const muN = tauNSq * (inp.mu0 / inp.tau0Sq + (inp.n * inp.yBar) / inp.sigmaSq);
  return { muN, tauNSq, precision };
}

/** Posterior predictive variance for a single new observation under the
 *  Normal-Normal setup: tau_n^2 + sigma^2 (Theorem 6.1). */
export function normalNormalPosteriorPredictiveVar(
  posterior: NormalNormalPosterior,
  sigmaSq: number,
): number {
  return posterior.tauNSq + sigmaSq;
}

export function normalPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

export function normalLogPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return -0.5 * z * z - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
}

/** Inverse standard-normal CDF via Beasley–Springer–Moro. */
export function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
             1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
             6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
             -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
             3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number, r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
          ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

// =============================================================================
// STUDENT-T (§11 non-conjugate robust prior)
// =============================================================================

export function studentTPdf(x: number, nu: number, loc = 0, scale = 1): number {
  const z = (x - loc) / scale;
  const logCoef =
    lgamma((nu + 1) / 2) -
    lgamma(nu / 2) -
    0.5 * Math.log(nu * Math.PI) -
    Math.log(scale);
  return Math.exp(logCoef - ((nu + 1) / 2) * Math.log(1 + (z * z) / nu));
}

export function studentTLogPdf(x: number, nu: number, loc = 0, scale = 1): number {
  const z = (x - loc) / scale;
  return (
    lgamma((nu + 1) / 2) -
    lgamma(nu / 2) -
    0.5 * Math.log(nu * Math.PI) -
    Math.log(scale) -
    ((nu + 1) / 2) * Math.log(1 + (z * z) / nu)
  );
}

// =============================================================================
// 1-D NUMERICAL INTEGRATION + GRID POSTERIOR (§11)
// =============================================================================

/** Composite trapezoid rule over n+1 evenly-spaced abscissae in [a, b]. */
export function trapezoidIntegrate(
  fn: (x: number) => number,
  a: number,
  b: number,
  nIntervals = 1000,
): number {
  const h = (b - a) / nIntervals;
  let sum = 0.5 * (fn(a) + fn(b));
  for (let i = 1; i < nIntervals; i++) sum += fn(a + i * h);
  return sum * h;
}

export interface GridPosteriorInput {
  /** Unnormalized log-posterior on the grid (log-likelihood + log-prior). */
  logUnnormalized: (theta: number) => number;
  gridStart: number;
  gridEnd: number;
  nGrid?: number;
}

export interface GridPosteriorResult {
  thetaGrid: Float64Array;
  density: Float64Array;
  mean: number;
  variance: number;
  std: number;
  /** Central 95% credible interval. */
  centralCI95: { lower: number; upper: number };
}

/** Computes a grid-normalized posterior over a 1-D parameter, plus mean, std,
 *  and a 95% central credible interval. Used in §11 for the non-conjugate
 *  Student-t + Normal demonstration. */
export function gridPosterior(inp: GridPosteriorInput): GridPosteriorResult {
  const n = inp.nGrid ?? 1024;
  const theta = new Float64Array(n + 1);
  const density = new Float64Array(n + 1);
  const h = (inp.gridEnd - inp.gridStart) / n;
  // Compute unnormalized log-density and shift by max for numerical stability.
  let logMax = -Infinity;
  for (let i = 0; i <= n; i++) {
    theta[i] = inp.gridStart + i * h;
    density[i] = inp.logUnnormalized(theta[i]);
    if (density[i] > logMax) logMax = density[i];
  }
  // Exponentiate shifted log-density, then renormalize by trapezoid quadrature.
  let mass = 0;
  for (let i = 0; i <= n; i++) {
    density[i] = Math.exp(density[i] - logMax);
    const w = i === 0 || i === n ? 0.5 : 1;
    mass += w * density[i];
  }
  mass *= h;
  for (let i = 0; i <= n; i++) density[i] /= mass;
  // Mean and variance by trapezoid.
  let mean = 0;
  for (let i = 0; i <= n; i++) {
    const w = i === 0 || i === n ? 0.5 : 1;
    mean += w * theta[i] * density[i];
  }
  mean *= h;
  let variance = 0;
  for (let i = 0; i <= n; i++) {
    const w = i === 0 || i === n ? 0.5 : 1;
    const d = theta[i] - mean;
    variance += w * d * d * density[i];
  }
  variance *= h;
  // 95% central CI via cumulative trapezoid on the density.
  const cum = new Float64Array(n + 1);
  cum[0] = 0;
  for (let i = 1; i <= n; i++) {
    cum[i] = cum[i - 1] + 0.5 * (density[i] + density[i - 1]) * h;
  }
  function quantile(p: number): number {
    let lo = 0;
    let hi = n;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < p) lo = mid;
      else hi = mid;
    }
    // Linear interpolation between grid points.
    const w = (p - cum[lo]) / Math.max(cum[hi] - cum[lo], 1e-30);
    return theta[lo] + w * (theta[hi] - theta[lo]);
  }
  return {
    thetaGrid: theta,
    density,
    mean,
    variance,
    std: Math.sqrt(variance),
    centralCI95: { lower: quantile(0.025), upper: quantile(0.975) },
  };
}

// =============================================================================
// BAYES FACTOR — Normal-vs-Normal §10.3 closed form
// =============================================================================

/** log B_10(tau) = n^2 yBar^2 tau^2 / [2(1 + n tau^2)] - 0.5 log(1 + n tau^2).
 *  Derived in brief §10.3 by completing the square in the joint Gaussian for y. */
export function logBayesFactorNormalNormal(yBar: number, n: number, tau: number): number {
  const tauSq = tau * tau;
  const denom = 1 + n * tauSq;
  return (n * n * yBar * yBar * tauSq) / (2 * denom) - 0.5 * Math.log(denom);
}

/** Argmax over tau^2 of log B_10(tau). When yBar^2 > 1/n the interior maximum
 *  is tau*^2 = yBar^2 - 1/n; otherwise B_10 is maximized at tau -> 0 (no alt). */
export function bayesFactorOptimalTauSq(yBar: number, n: number): number {
  const t = yBar * yBar - 1 / n;
  return t > 0 ? t : 0;
}

// =============================================================================
// HIERARCHICAL BETA–BINOMIAL EMPIRICAL BAYES (§9.3)
// =============================================================================

export interface HierarchicalGroup {
  /** Group label for the viz. */
  label: string;
  /** Trial count. */
  n: number;
  /** Success count. */
  s: number;
}

export interface HierarchicalEBResult {
  /** Grand-mean estimate \hat\omega = \hat\alpha / (\hat\alpha + \hat\beta). */
  omegaHat: number;
  /** Concentration \hat\kappa = \hat\alpha + \hat\beta. */
  kappaHat: number;
  alphaHat: number;
  betaHat: number;
  /** Per-group posterior means after partial pooling. */
  perGroup: {
    label: string;
    n: number;
    s: number;
    mle: number;
    poolWeight: number;
    posteriorMean: number;
    posteriorAlpha: number;
    posteriorBeta: number;
  }[];
  /** Negative log marginal likelihood at the optimum (for diagnostics). */
  negLogMarginal: number;
}

/** Negative log marginal likelihood for the hierarchical Beta–Binomial
 *  parameterized by (omega, kappa) with alpha = kappa * omega, beta = kappa * (1 - omega).
 *  Each group contributes log p(s_g | n_g, alpha, beta) = the Beta-Binomial PMF. */
function negLogMarginal(omega: number, kappa: number, groups: HierarchicalGroup[]): number {
  if (omega <= 0 || omega >= 1 || kappa <= 0) return Infinity;
  const alpha = kappa * omega;
  const beta = kappa * (1 - omega);
  let total = 0;
  for (const g of groups) total += betaBinomialLogPmf(g.s, g.n, alpha, beta);
  return -total;
}

/** Empirical-Bayes fit for the §9 Beta–Binomial hierarchy via
 *  Nelder–Mead-like coordinate descent. Initializes omega at the
 *  raw grand mean and kappa via a moment-matching seed. */
export function hierarchicalBetaBinomialEB(
  groups: HierarchicalGroup[],
): HierarchicalEBResult {
  const totalS = groups.reduce((acc, g) => acc + g.s, 0);
  const totalN = groups.reduce((acc, g) => acc + g.n, 0);
  const rawMean = totalS / Math.max(totalN, 1);
  // Method-of-moments seed for kappa via the cross-group sample variance.
  const ps = groups.map((g) => g.s / Math.max(g.n, 1));
  const meanP = ps.reduce((a, b) => a + b, 0) / ps.length;
  const varP = ps.reduce((a, b) => a + (b - meanP) ** 2, 0) / ps.length;
  const kappaSeed = varP > 1e-9 ? Math.max(1, (meanP * (1 - meanP)) / varP - 1) : 100;
  let omega = rawMean;
  let kappa = kappaSeed;
  let f = negLogMarginal(omega, kappa, groups);
  // Coordinate descent with shrinking step sizes — robust without scipy.optimize.
  let omegaStep = 0.05;
  let kappaStep = Math.max(kappa * 0.5, 5);
  for (let iter = 0; iter < 300; iter++) {
    let improved = false;
    for (const dOm of [-omegaStep, omegaStep]) {
      const o2 = omega + dOm;
      if (o2 <= 0.001 || o2 >= 0.999) continue;
      const f2 = negLogMarginal(o2, kappa, groups);
      if (f2 < f - 1e-9) {
        omega = o2;
        f = f2;
        improved = true;
      }
    }
    for (const dK of [-kappaStep, kappaStep]) {
      const k2 = kappa + dK;
      if (k2 <= 0.1) continue;
      const f2 = negLogMarginal(omega, k2, groups);
      if (f2 < f - 1e-9) {
        kappa = k2;
        f = f2;
        improved = true;
      }
    }
    if (!improved) {
      if (omegaStep > 1e-6) omegaStep *= 0.5;
      if (kappaStep > 1e-3) kappaStep *= 0.5;
      if (omegaStep <= 1e-6 && kappaStep <= 1e-3) break;
    }
  }
  const alphaHat = kappa * omega;
  const betaHat = kappa * (1 - omega);
  const perGroup = groups.map((g) => {
    const poolWeight = kappa / (kappa + g.n);
    const postAlpha = alphaHat + g.s;
    const postBeta = betaHat + (g.n - g.s);
    return {
      label: g.label,
      n: g.n,
      s: g.s,
      mle: g.s / Math.max(g.n, 1),
      poolWeight,
      posteriorMean: postAlpha / (postAlpha + postBeta),
      posteriorAlpha: postAlpha,
      posteriorBeta: postBeta,
    };
  });
  return { omegaHat: omega, kappaHat: kappa, alphaHat, betaHat, perGroup, negLogMarginal: f };
}

// =============================================================================
// PRNG (mulberry32) + Gaussian / Beta / Gamma sampling
// =============================================================================
// We use mulberry32 instead of Math.random() so all MC viz reproduce across
// re-mounts within a single browser session. Cross-RNG drift vs NumPy's PCG64
// is the reason verify-bayesian-inference.ts tolerances are stated as ranges
// for MC quantities like P(B > A | data).

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function (): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample via Box–Muller. */
export function gaussianSample(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-300);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Marsaglia & Tsang Gamma(shape, 1) sampler. shape > 0. */
export function gammaSample(shape: number, rng: () => number): number {
  if (shape < 1) {
    return gammaSample(shape + 1, rng) * Math.pow(Math.max(rng(), 1e-300), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (let attempt = 0; attempt < 10000; attempt++) {
    let x: number;
    let v: number;
    do {
      x = gaussianSample(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
  return d; // fallback — should not happen
}

/** Beta(alpha, beta) sample via gamma ratio: X/(X+Y) ~ Beta when X ~ Gamma(alpha), Y ~ Gamma(beta). */
export function betaSample(alpha: number, beta: number, rng: () => number): number {
  const x = gammaSample(alpha, rng);
  const y = gammaSample(beta, rng);
  return x / (x + y);
}

// =============================================================================
// MONTE CARLO PROBABILITY ESTIMATORS
// =============================================================================

/** P(theta_B > theta_A | data) via Monte Carlo from the two posteriors —
 *  the §1.2 A/B-test deliverable. */
export function probBeatsAB(
  alphaA: number,
  betaA: number,
  alphaB: number,
  betaB: number,
  m: number,
  rng: () => number,
): number {
  let count = 0;
  for (let i = 0; i < m; i++) {
    const tA = betaSample(alphaA, betaA, rng);
    const tB = betaSample(alphaB, betaB, rng);
    if (tB > tA) count++;
  }
  return count / m;
}

/** P(switch correct | y) = P(theta_B - c > theta_A | y) for a given cost c. */
export function probSwitchCorrect(
  alphaA: number,
  betaA: number,
  alphaB: number,
  betaB: number,
  c: number,
  m: number,
  rng: () => number,
): number {
  let count = 0;
  for (let i = 0; i < m; i++) {
    const tA = betaSample(alphaA, betaA, rng);
    const tB = betaSample(alphaB, betaB, rng);
    if (tB - c > tA) count++;
  }
  return count / m;
}

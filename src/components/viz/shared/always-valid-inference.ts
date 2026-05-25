// =============================================================================
// Shared helpers for the Always-Valid Inference topic.
//
// Phase 1 ships only the seeded RNG + Gaussian / Bernoulli samplers used by the
// §1, §2 viz. Phase 4 extends with closed-form math (binaryKL, gaussianMixtureHW,
// subExponentialHW, lgamma/betaln, lambertW0, vovk/shafer calibrators) plus the
// numerical-regression suite in __tests__/verify-always-valid-inference.ts.
//
// All viz reuse mulberry32 instead of Math.random() so Monte Carlo trajectories
// reproduce across re-mounts within a single browser session. Cross-RNG drift vs
// NumPy's PCG64 means notebook-derived empirical values come within MC noise but
// are not bit-pinnable.
// =============================================================================

// -----------------------------------------------------------------------------
// PRNG — mulberry32, returning a callable rng() in [0, 1).
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Gaussian sampler via Box–Muller, lazy paired so each call returns one draw.
// -----------------------------------------------------------------------------

export function gaussianFrom(rng: () => number): () => number {
  let cached: number | null = null;
  return function (): number {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = 0;
    let u2 = 0;
    while (u1 === 0) u1 = rng();
    while (u2 === 0) u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

// -----------------------------------------------------------------------------
// z_{δ/2} approximation — Beasley–Springer–Moro for the inverse normal CDF.
// Adequate for δ ∈ [0.001, 0.5] which is the AVI viz slider range.
// -----------------------------------------------------------------------------

export function zHalf(delta: number): number {
  // returns z such that P(Z >= z) = delta/2
  const p = 1 - delta / 2;
  // Acklam's inverse-normal approximation, accurate to ~1e-9 over (0, 1).
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return (
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

// -----------------------------------------------------------------------------
// Single-λ sub-Gaussian half-width for the running mean at level δ.
//   HW(n, λ, σ) = log(1/δ)/(nλ) + λσ²/2
// Tight at one n; the Robbins mixture tracks the lower envelope across λ.
// -----------------------------------------------------------------------------

export function singleLambdaSGHW(
  n: number,
  lambda: number,
  delta: number,
  sigma = 1,
): number {
  return Math.log(1 / delta) / (n * lambda) + (lambda * sigma * sigma) / 2;
}

// -----------------------------------------------------------------------------
// Sub-exponential confidence-sequence half-width (5.9):
//   HW(n, ν, c, δ) = ν √(2 log(√n / δ) / n) + c log(1/δ) / n
// The first term is the sub-Gaussian-like main; the second is the tail
// correction. Setting c = 0 recovers sub-Gaussian. Used for Poisson, χ², etc.
// -----------------------------------------------------------------------------

export function subExponentialHW(
  n: number,
  nu: number,
  c: number,
  delta: number,
): number {
  const main = nu * Math.sqrt((2 * Math.log(Math.sqrt(n) / delta)) / n);
  const corr = (c * Math.log(1 / delta)) / n;
  return main + corr;
}

// -----------------------------------------------------------------------------
// Gaussian-mixture (Robbins) confidence-sequence half-width — Theorem 4.2.
//   HW_mix(n, ρ², σ, δ) = √( (2(1 + n ρ² σ²) / (n² ρ²)) · log(√(1 + n ρ² σ²) / δ) )
// -----------------------------------------------------------------------------

export function gaussianMixtureHW(
  n: number,
  rhoSq: number,
  delta: number,
  sigma = 1,
): number {
  const inner = 1 + n * rhoSq * sigma * sigma;
  const denom = n * n * rhoSq;
  return Math.sqrt(((2 * inner) / denom) * Math.log(Math.sqrt(inner) / delta));
}

// -----------------------------------------------------------------------------
// LIL-shaped half-width — σ √(2 log log n / n). Only meaningful for n > e^e ≈
// 15, but we return 0 below that to keep plot generators safe.
// -----------------------------------------------------------------------------

export function lilHW(n: number, sigma = 1): number {
  if (n < 16) return 0;
  return sigma * Math.sqrt((2 * Math.log(Math.log(n))) / n);
}

// -----------------------------------------------------------------------------
// Binary KL: kl(p || q) = p log(p/q) + (1-p) log((1-p)/(1-q)).
// Returns NaN if p or q are outside (0, 1).
// -----------------------------------------------------------------------------

export function binaryKL(p: number, q: number): number {
  if (p <= 0 || p >= 1 || q <= 0 || q >= 1) return NaN;
  return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
}

// -----------------------------------------------------------------------------
// log Γ(x) — Lanczos approximation, accurate to ~1e-14 for x > 0.5.
// Reflection for x < 0.5: log Γ(x) = log(π) − log(sin(πx)) − log Γ(1 − x).
// -----------------------------------------------------------------------------

export function lgamma(x: number): number {
  if (x < 0.5) {
    // reflection
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  // Lanczos coefficients (g = 7, n = 9)
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  const xm = x - 1;
  let a = c[0];
  for (let i = 1; i < 9; i++) a += c[i] / (xm + i);
  const t = xm + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (xm + 0.5) * Math.log(t) - t + Math.log(a);
}

// log B(a, b) = log Γ(a) + log Γ(b) − log Γ(a+b).
export function betaln(a: number, b: number): number {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

// -----------------------------------------------------------------------------
// Beta-mixture e-value for the sub-Bernoulli construction at hypothesized
// mean μ, given S_n = sum and n trials. The Beta(a, b) prior on the
// Bernoulli parameter induces the marginal-likelihood ratio (5.14):
//   log E_n = log B(a+S_n, b+n−S_n) − log B(a, b) − S_n log μ − (n−S_n) log(1−μ)
// Uniform prior is the default (a = b = 1).
// -----------------------------------------------------------------------------

export function logBetaMixtureEvalue(
  S_n: number,
  n: number,
  mu: number,
  a = 1,
  b = 1,
): number {
  if (mu <= 0 || mu >= 1) return NaN;
  const num = betaln(a + S_n, b + n - S_n) - betaln(a, b);
  const den = S_n * Math.log(mu) + (n - S_n) * Math.log(1 - mu);
  return num - den;
}

// -----------------------------------------------------------------------------
// PrPl-EB asymptotic half-width — √(2 σ̂² log(2/δ) / n). Used as a comparator
// to the sub-Bernoulli Beta-mixture in §6.
// -----------------------------------------------------------------------------

export function prplEbHW(n: number, sigmaHatSq: number, delta: number): number {
  return Math.sqrt((2 * sigmaHatSq * Math.log(2 / delta)) / n);
}

// -----------------------------------------------------------------------------
// Sub-Bernoulli Beta-mixture CS endpoints at expected mean μ_obs and δ.
// Returns [lo, hi] such that logBetaMixtureEvalue(n*μ_obs, n, μ') ≤ log(1/δ)
// for μ' ∈ [lo, hi]. Uses bisection. For visualization at "expected
// trajectory" S_n = n·μ_obs.
// -----------------------------------------------------------------------------

export function subBernoulliCSEndpoints(
  n: number,
  muObs: number,
  delta: number,
  a = 1,
  b = 1,
  tol = 1e-5,
): [number, number] {
  const S = n * muObs;
  const logT = Math.log(1 / delta);
  // f(μ) = logBetaMixtureEvalue(S, n, μ) − logT. f(μ_obs) is the minimum and
  // is ≤ 0 for any δ ≤ 1 (provided the e-process at μ_obs hasn't crossed
  // 1/δ — which it generally hasn't at the realized trajectory). Endpoints
  // are the two roots of f bracketing μ_obs. If μ_obs ≤ tol the lower root
  // is forced to 0; if f(lo_min) > 0 we bisect for the left endpoint.
  const f = (mu: number): number => logBetaMixtureEvalue(S, n, mu, a, b) - logT;

  let left = 0;
  if (muObs > 1e-6) {
    let lo = 1e-6;
    let hi = muObs;
    if (f(lo) > 0) {
      while (hi - lo > tol) {
        const mid = 0.5 * (lo + hi);
        if (f(mid) >= 0) lo = mid;
        else hi = mid;
      }
      left = 0.5 * (lo + hi);
    }
  }

  let right = 1;
  if (muObs < 1 - 1e-6) {
    let lo = muObs;
    let hi = 1 - 1e-6;
    if (f(hi) > 0) {
      while (hi - lo > tol) {
        const mid = 0.5 * (lo + hi);
        if (f(mid) >= 0) hi = mid;
        else lo = mid;
      }
      right = 0.5 * (lo + hi);
    }
  }
  return [left, right];
}

// -----------------------------------------------------------------------------
// Inverse standard-normal CDF — Acklam approximation, ~1e-9 accuracy over (0, 1).
// Returns z such that Φ(z) = p. Useful when callers need a quantile other than
// the symmetric two-sided one served by zHalf (e.g., z_{1−α} for SPRT n_NP
// formulas).
// -----------------------------------------------------------------------------

export function inverseNormalCDF(p: number): number {
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return (
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

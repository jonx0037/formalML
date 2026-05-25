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

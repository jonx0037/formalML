// =============================================================================
// mixed-effects.ts
//
// Shared math/types/palette module for the T5 Bayesian & Probabilistic ML
// `mixed-effects` topic. Centralizes the six-classroom DGP and BLUP-shrinkage
// algebra so multiple viz components (PoolingSpectrumExplorer, A1;
// BLUPShrinkageDial, A2; planned MarginalCovarianceHeatmap, A3) all consume
// the same numbers.
//
// Numerical algorithms mirror the topic's notebook setup at line 232–249 of
// src/content/topics/mixed-effects.mdx — group sizes (4, 6, 8, 10, 12, 20),
// truths (α=50, β=5, τ=5, σ=8), seed 20260429. Exact byte-equality with the
// notebook's NumPy PRNG isn't guaranteed because we use mulberry32 here
// instead of NumPy's PCG64; the qualitative structure (group sizes, score
// range, shared slope) carries the pedagogy. The default τ²/σ²=0.07 in
// downstream viz reproduces the topic's REML shrinkage factors at line 249
// (0.222, 0.298, 0.359, 0.412, 0.451, 0.588) to two decimals.
// =============================================================================

export const GROUP_SIZES = [4, 6, 8, 10, 12, 20] as const;
export const ALPHA_TRUE = 50;
export const BETA_TRUE = 5;
export const TAU_TRUE = 5;
export const SIGMA_TRUE = 8;
export const SEED = 20260429;
export const X_DOMAIN: [number, number] = [0, 5];

// Six color-blind-friendly hues for the six classrooms.
export const PALETTE_CLASSROOMS = [
  '#1f77b4', // blue
  '#ff7f0e', // orange
  '#2ca02c', // green
  '#d62728', // red
  '#9467bd', // purple
  '#8c564b', // brown
] as const;

// Mulberry32 — small, fast, deterministic 32-bit PRNG.
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

// Box–Muller standard-normal sampler from a uniform PRNG.
export function gaussianSampler(rng: () => number): () => number {
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = rng();
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

export interface ClassroomData {
  j: number;        // classroom index 0..5
  nj: number;       // group size
  x: Float64Array;  // prep-hours
  y: Float64Array;  // exam scores
  xMean: number;    // x̄_j
  yMean: number;    // ȳ_j
}

export interface SixClassrooms {
  classrooms: ClassroomData[];
  N: number;                   // total observations
  betaWithin: number;          // shared within-group slope (no-pooling estimate)
  alphaPool: number;           // population-mean intercept under shared slope
  sigma2Hat: number;           // pooled within-group residual variance estimate
  alphasNoPool: Float64Array;  // per-classroom OLS intercepts holding shared slope
}

// Generate the six-classroom synthetic dataset and pre-compute the
// no-pooling and complete-pooling intercepts under a shared within-group
// slope. Deterministic given the module-level seed.
export function generateSixClassrooms(): SixClassrooms {
  const rng = mulberry32(SEED);
  const sample = gaussianSampler(rng);
  const J = GROUP_SIZES.length;

  // Sample classroom random intercepts a_j ~ N(0, τ²).
  const aTrue = new Float64Array(J);
  for (let j = 0; j < J; j++) aTrue[j] = TAU_TRUE * sample();

  const classrooms: ClassroomData[] = [];
  let N = 0;
  for (let j = 0; j < J; j++) {
    const nj = GROUP_SIZES[j];
    const x = new Float64Array(nj);
    const y = new Float64Array(nj);
    for (let i = 0; i < nj; i++) {
      x[i] = X_DOMAIN[0] + rng() * (X_DOMAIN[1] - X_DOMAIN[0]);
      y[i] = ALPHA_TRUE + BETA_TRUE * x[i] + aTrue[j] + SIGMA_TRUE * sample();
    }
    let xs = 0;
    let ys = 0;
    for (let i = 0; i < nj; i++) {
      xs += x[i];
      ys += y[i];
    }
    classrooms.push({
      j,
      nj,
      x,
      y,
      xMean: xs / nj,
      yMean: ys / nj,
    });
    N += nj;
  }

  // Within-group OLS: β̂ = Σⱼᵢ (x − x̄ⱼ)(y − ȳⱼ) / Σⱼᵢ (x − x̄ⱼ)².
  let sxy = 0;
  let sxx = 0;
  for (const c of classrooms) {
    for (let i = 0; i < c.nj; i++) {
      const dx = c.x[i] - c.xMean;
      const dy = c.y[i] - c.yMean;
      sxy += dx * dy;
      sxx += dx * dx;
    }
  }
  const betaWithin = sxy / sxx;

  // No-pooling intercepts: α̂ⱼ = ȳⱼ − β̂·x̄ⱼ.
  const alphasNoPool = new Float64Array(J);
  for (let j = 0; j < J; j++) {
    const c = classrooms[j];
    alphasNoPool[j] = c.yMean - betaWithin * c.xMean;
  }

  // Population-mean intercept under the shared slope.
  let xBar = 0;
  let yBar = 0;
  for (const c of classrooms) {
    for (let i = 0; i < c.nj; i++) {
      xBar += c.x[i];
      yBar += c.y[i];
    }
  }
  xBar /= N;
  yBar /= N;
  const alphaPool = yBar - betaWithin * xBar;

  // Pooled within-group residual variance σ̂² = Σ (yᵢⱼ − α̂ⱼ − β̂·xᵢⱼ)² / (N − J − 1).
  let rss = 0;
  for (let j = 0; j < J; j++) {
    const c = classrooms[j];
    for (let i = 0; i < c.nj; i++) {
      const r = c.y[i] - alphasNoPool[j] - betaWithin * c.x[i];
      rss += r * r;
    }
  }
  const sigma2Hat = rss / (N - J - 1);

  return { classrooms, N, betaWithin, alphaPool, sigma2Hat, alphasNoPool };
}

// BLUP shrinkage factor λⱼ = τ²·nⱼ / (τ²·nⱼ + σ²) = ρ·nⱼ / (ρ·nⱼ + 1) where
// ρ = τ²/σ². The closed-form Proposition 2 result from §3.3.
export function blupShrinkage(nj: number, tauSqOverSigmaSq: number): number {
  const rho = tauSqOverSigmaSq;
  return (rho * nj) / (rho * nj + 1);
}

// Convex-combination form of the partial-pooled intercept: eq. (3.16).
export function partialPooledIntercept(
  alphaNoPool: number,
  alphaPool: number,
  lambda: number,
): number {
  return lambda * alphaNoPool + (1 - lambda) * alphaPool;
}

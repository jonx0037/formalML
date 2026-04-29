// =============================================================================
// gaussian-processes.ts
//
// Shared math/types/palette module for the T5 Bayesian & Probabilistic ML
// `gaussian-processes` topic. Provides numerical primitives for:
//   - kernel zoo (SE, Matérn-1/2/3/2/5/2, periodic, ARD-SE)
//   - Cholesky factorization and triangular solves
//   - GP regression predictive distribution (Theorem 3.2)
//   - posterior joint sampling
//   - log marginal likelihood + analytic gradient (Propositions 5.1–5.2)
//   - multi-restart L-BFGS-B for hyperparameter optimization
//   - sparse approximations: Nyström, SVGP (Titsias 2009 closed form), RFF
//
// All numerical algorithms are direct translations of the verified Python
// notebook at notebooks/gaussian-processes/01_gaussian_processes.ipynb (NumPy +
// SciPy + matplotlib only — no JAX). Where the Python uses np.random / scipy.
// linalg, this module substitutes a seeded Mulberry32 PRNG, Box-Muller Gaussian
// sampling, and small hand-rolled solvers. Numerical agreement is verified
// against the notebook's printed outputs by
// src/components/viz/shared/__tests__/verify-gaussian-processes.ts.
//
// PRNG note: Mulberry32 is independent of NumPy's PCG64; bit-exact
// reproduction of notebook RNG draws is not possible. For deterministic
// verification we hard-code the notebook's training-data outputs (see
// src/data/gaussian-processes-data.ts) and check downstream computations
// against them.
// =============================================================================

// -----------------------------------------------------------------------------
// Color palette — mirrors the notebook setup cell
// -----------------------------------------------------------------------------

/**
 * Color palette mirroring `notebooks/gaussian-processes/01_gaussian_processes.
 * ipynb` setup-cell constants. Keeping the TypeScript palette in lockstep with
 * the notebook means React enhancements stay visually continuous with the v1
 * static PNGs.
 */
export const paletteGP = {
  posterior: '#1f77b4', // blue   — GP posterior mean
  truth: '#d62728',     // red    — ground-truth function
  nystrom: '#ff7f0e',   // orange — Nyström / random-feature approximation
  svgp: '#2ca02c',      // green  — sparse variational GP (inducing points)
  data: '#7f7f7f',      // gray   — training data points
  reference: '#000000', // black  — reference lines, ±2σ band edges
} as const;

export const paletteSamples = [
  paletteGP.posterior,
  paletteGP.truth,
  paletteGP.nystrom,
  paletteGP.svgp,
  '#9467bd',
] as const;

export type GPColorKey = keyof typeof paletteGP;

// -----------------------------------------------------------------------------
// Seeded PRNG — Mulberry32, Box-Muller for Gaussian draws
// -----------------------------------------------------------------------------

/** Mulberry32: small, fast, deterministic uniform [0, 1) PRNG. */
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

/** Box-Muller standard-normal sampler given a uniform RNG. Caches the second
 * draw of each pair for the next call. */
export function gaussianSampler(rng: () => number): () => number {
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = rng();
    if (u1 < 1e-300) u1 = 1e-300;
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

// -----------------------------------------------------------------------------
// Matrix primitives
//
// Conventions:
//   - Matrices stored as `number[][]` (row-major, jagged 2D arrays).
//   - All allocations happen inside helpers so callers don't need to pre-size.
//   - Functions accept and return new matrices; never mutate inputs.
// -----------------------------------------------------------------------------

/** Construct an n×n identity matrix scaled by `s`. */
export function identityMatrix(n: number, s = 1): number[][] {
  const M: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = new Array(n).fill(0);
    row[i] = s;
    M.push(row);
  }
  return M;
}

/** Add `s * I_n` to A in place; returns the same array. Used to add jitter. */
export function addDiagonal(A: number[][], s: number): number[][] {
  for (let i = 0; i < A.length; i++) A[i][i] += s;
  return A;
}

/** Matrix-matrix multiply A B (no shape check beyond what the loops enforce). */
export function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const k = A[0].length;
  const n = B[0].length;
  const C: number[][] = [];
  for (let i = 0; i < m; i++) {
    const Ai = A[i];
    const Crow = new Array<number>(n).fill(0);
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < n; j++) Crow[j] += a * Bp[j];
    }
    C.push(Crow);
  }
  return C;
}

/** Matrix-vector multiply A v. */
export function matVec(A: number[][], v: number[]): number[] {
  const m = A.length;
  const n = v.length;
  const out = new Array<number>(m).fill(0);
  for (let i = 0; i < m; i++) {
    const Ai = A[i];
    let s = 0;
    for (let j = 0; j < n; j++) s += Ai[j] * v[j];
    out[i] = s;
  }
  return out;
}

/** Transpose. */
export function transpose(A: number[][]): number[][] {
  const m = A.length;
  const n = A[0].length;
  const T: number[][] = [];
  for (let j = 0; j < n; j++) {
    const row = new Array<number>(m);
    for (let i = 0; i < m; i++) row[i] = A[i][j];
    T.push(row);
  }
  return T;
}

/** Outer product u v^T as an m×n matrix. */
export function outer(u: number[], v: number[]): number[][] {
  const m = u.length;
  const n = v.length;
  const M: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row = new Array<number>(n);
    const ui = u[i];
    for (let j = 0; j < n; j++) row[j] = ui * v[j];
    M.push(row);
  }
  return M;
}

/** Vector dot product. */
export function dot(u: number[], v: number[]): number {
  let s = 0;
  for (let i = 0; i < u.length; i++) s += u[i] * v[i];
  return s;
}

// -----------------------------------------------------------------------------
// Cholesky factorization (lower triangular L such that L L^T = A)
//
// Standard right-looking algorithm; throws on non-PSD input. Caller is
// responsible for adding sufficient jitter for numerical stability — the
// notebook adds `(σ_n^2 + 1e-8) * I` before factorizing.
// -----------------------------------------------------------------------------

/**
 * Cholesky factorization of a symmetric positive-definite matrix A.
 * Returns L (lower-triangular, with strict-upper entries set to 0) such that
 * L L^T = A.
 *
 * Throws if a non-positive diagonal pivot appears (caller should add jitter).
 */
export function choleskyFactor(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = [];
  for (let i = 0; i < n; i++) L.push(new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let s = A[j][j];
    for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    if (s <= 0) {
      throw new Error(
        `choleskyFactor: non-PSD pivot at (${j}, ${j}) = ${s.toExponential(3)}; add more jitter`,
      );
    }
    const ljj = Math.sqrt(s);
    L[j][j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let t = A[i][j];
      for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
      L[i][j] = t / ljj;
    }
  }
  return L;
}

/** Solve L y = b for lower-triangular L (forward substitution). */
export function solveLowerTriangular(L: number[][], b: number[]): number[] {
  const n = L.length;
  const y = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    const Li = L[i];
    for (let k = 0; k < i; k++) s -= Li[k] * y[k];
    y[i] = s / Li[i];
  }
  return y;
}

/** Solve L^T x = y for lower-triangular L (backward substitution).
 *  Equivalent to scipy.linalg.solve_triangular(L.T, y, lower=False). */
export function solveUpperTriangularT(L: number[][], y: number[]): number[] {
  const n = L.length;
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/** Solve L Y = B column-by-column, returning Y (same shape as B). */
export function solveLowerTriangularMatrix(L: number[][], B: number[][]): number[][] {
  const n = L.length;
  const m = B[0].length;
  const Y: number[][] = [];
  for (let i = 0; i < n; i++) Y.push(new Array<number>(m).fill(0));
  for (let j = 0; j < m; j++) {
    for (let i = 0; i < n; i++) {
      let s = B[i][j];
      const Li = L[i];
      for (let k = 0; k < i; k++) s -= Li[k] * Y[k][j];
      Y[i][j] = s / Li[i];
    }
  }
  return Y;
}

/** log(det(A)) via Cholesky factor: 2 ∑ log(L_ii). */
export function choleskyLogDet(L: number[][]): number {
  let s = 0;
  for (let i = 0; i < L.length; i++) s += Math.log(L[i][i]);
  return 2 * s;
}

/** Compute A^{-1} from a Cholesky factor L (used in §5 gradient).
 *  Equivalent to scipy.linalg.cho_solve((L, True), I). */
export function choleskyInverse(L: number[][]): number[][] {
  const n = L.length;
  const I = identityMatrix(n);
  // Solve L Y = I, then L^T A^{-1} = Y → backward substitute column by column.
  const Y = solveLowerTriangularMatrix(L, I);
  const Ainv: number[][] = [];
  for (let i = 0; i < n; i++) Ainv.push(new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    const yCol = new Array<number>(n);
    for (let i = 0; i < n; i++) yCol[i] = Y[i][j];
    const xCol = solveUpperTriangularT(L, yCol);
    for (let i = 0; i < n; i++) Ainv[i][j] = xCol[i];
  }
  return Ainv;
}

// -----------------------------------------------------------------------------
// Kernel zoo
//
// 1D inputs are passed as `number[]`; 2D ARD inputs are passed as `number[][]`
// where each row is a sample. `KernelFn1D` is the unified 1D signature; ARD
// has its own signature in `kernelSEARD`.
// -----------------------------------------------------------------------------

/** Squared-exponential / RBF kernel on 1D inputs. */
export function kernelSE(
  X1: number[], X2: number[], sigmaF: number, lengthscale: number,
): number[][] {
  const n1 = X1.length;
  const n2 = X2.length;
  const sf2 = sigmaF * sigmaF;
  const L2 = lengthscale * lengthscale;
  const K: number[][] = [];
  for (let i = 0; i < n1; i++) {
    const row = new Array<number>(n2);
    const x1i = X1[i];
    for (let j = 0; j < n2; j++) {
      const d = x1i - X2[j];
      row[j] = sf2 * Math.exp(-(d * d) / (2 * L2));
    }
    K.push(row);
  }
  return K;
}

/** Matérn-1/2 (a.k.a. exponential) kernel on 1D inputs. Samples are continuous
 *  but nowhere mean-square differentiable — Brownian-motion-like. */
export function kernelMatern12(
  X1: number[], X2: number[], sigmaF: number, lengthscale: number,
): number[][] {
  const n1 = X1.length;
  const n2 = X2.length;
  const sf2 = sigmaF * sigmaF;
  const K: number[][] = [];
  for (let i = 0; i < n1; i++) {
    const row = new Array<number>(n2);
    const x1i = X1[i];
    for (let j = 0; j < n2; j++) {
      const r = Math.abs(x1i - X2[j]);
      row[j] = sf2 * Math.exp(-r / lengthscale);
    }
    K.push(row);
  }
  return K;
}

/** Matérn-3/2 kernel: once mean-square differentiable. */
export function kernelMatern32(
  X1: number[], X2: number[], sigmaF: number, lengthscale: number,
): number[][] {
  const n1 = X1.length;
  const n2 = X2.length;
  const sf2 = sigmaF * sigmaF;
  const sqrt3 = Math.sqrt(3);
  const K: number[][] = [];
  for (let i = 0; i < n1; i++) {
    const row = new Array<number>(n2);
    const x1i = X1[i];
    for (let j = 0; j < n2; j++) {
      const r = Math.abs(x1i - X2[j]);
      const arg = sqrt3 * r / lengthscale;
      row[j] = sf2 * (1 + arg) * Math.exp(-arg);
    }
    K.push(row);
  }
  return K;
}

/** Matérn-5/2 kernel: twice mean-square differentiable. */
export function kernelMatern52(
  X1: number[], X2: number[], sigmaF: number, lengthscale: number,
): number[][] {
  const n1 = X1.length;
  const n2 = X2.length;
  const sf2 = sigmaF * sigmaF;
  const sqrt5 = Math.sqrt(5);
  const K: number[][] = [];
  for (let i = 0; i < n1; i++) {
    const row = new Array<number>(n2);
    const x1i = X1[i];
    for (let j = 0; j < n2; j++) {
      const r = Math.abs(x1i - X2[j]);
      const arg = sqrt5 * r / lengthscale;
      row[j] = sf2 * (1 + arg + (arg * arg) / 3) * Math.exp(-arg);
    }
    K.push(row);
  }
  return K;
}

/** MacKay 1998 periodic kernel: every sample is exactly `period`-periodic. */
export function kernelPeriodic(
  X1: number[], X2: number[], sigmaF: number, lengthscale: number, period: number,
): number[][] {
  const n1 = X1.length;
  const n2 = X2.length;
  const sf2 = sigmaF * sigmaF;
  const L2 = lengthscale * lengthscale;
  const K: number[][] = [];
  for (let i = 0; i < n1; i++) {
    const row = new Array<number>(n2);
    const x1i = X1[i];
    for (let j = 0; j < n2; j++) {
      const r = Math.abs(x1i - X2[j]);
      const s = Math.sin(Math.PI * r / period);
      row[j] = sf2 * Math.exp(-2 * s * s / L2);
    }
    K.push(row);
  }
  return K;
}

/** ARD-SE kernel: one lengthscale per input dimension. Inputs are 2D arrays
 *  (rows are samples, columns are features). */
export function kernelSEARD(
  X1: number[][], X2: number[][], sigmaF: number, lengthscales: number[],
): number[][] {
  const n1 = X1.length;
  const n2 = X2.length;
  const d = lengthscales.length;
  const sf2 = sigmaF * sigmaF;
  // Pre-square inverse lengthscales for efficiency.
  const invL2 = lengthscales.map((l) => 1 / (l * l));
  const K: number[][] = [];
  for (let i = 0; i < n1; i++) {
    const x1i = X1[i];
    const row = new Array<number>(n2);
    for (let j = 0; j < n2; j++) {
      const x2j = X2[j];
      let s = 0;
      for (let k = 0; k < d; k++) {
        const dk = x1i[k] - x2j[k];
        s += dk * dk * invL2[k];
      }
      row[j] = sf2 * Math.exp(-0.5 * s);
    }
    K.push(row);
  }
  return K;
}

// 1D kernel signature for generic GP routines below.
export type Kernel1DFn = (
  X1: number[], X2: number[], ...params: number[]
) => number[][];

/**
 * Convenience adapter: takes a kernel name and a parameter object, returns a
 * closure with the (X1, X2) signature that GP routines accept. Used by
 * components that switch between kernels at runtime.
 */
export type KernelName1D = 'se' | 'matern12' | 'matern32' | 'matern52' | 'periodic';

export interface KernelParams1D {
  sigmaF: number;
  lengthscale: number;
  period?: number; // periodic only
}

export function kernelByName1D(
  name: KernelName1D, params: KernelParams1D,
): (X1: number[], X2: number[]) => number[][] {
  switch (name) {
    case 'se':
      return (X1, X2) => kernelSE(X1, X2, params.sigmaF, params.lengthscale);
    case 'matern12':
      return (X1, X2) => kernelMatern12(X1, X2, params.sigmaF, params.lengthscale);
    case 'matern32':
      return (X1, X2) => kernelMatern32(X1, X2, params.sigmaF, params.lengthscale);
    case 'matern52':
      return (X1, X2) => kernelMatern52(X1, X2, params.sigmaF, params.lengthscale);
    case 'periodic':
      if (params.period === undefined) {
        throw new Error('kernelByName1D: periodic kernel requires params.period');
      }
      return (X1, X2) =>
        kernelPeriodic(X1, X2, params.sigmaF, params.lengthscale, params.period!);
  }
}

// -----------------------------------------------------------------------------
// GP regression — closed-form predictive distribution (Theorem 3.2)
// -----------------------------------------------------------------------------

export interface GPPredictResult {
  /** Predictive mean μ_* = K_*^T (K + σ_n² I)^{-1} y. */
  mean: number[];
  /** Predictive covariance Σ_* = K_** - K_*^T (K + σ_n² I)^{-1} K_*. */
  cov: number[][];
  /** Predictive standard deviation, diag(Σ_*)^{1/2}. Convenience for ±2σ band. */
  sd: number[];
  /** Cholesky factor of (K + σ_n² I); reused in §5 for marginal-likelihood gradients. */
  L: number[][];
}

/**
 * GP regression predictive distribution via the Cholesky route.
 * Direct port of the notebook's `gp_predict` (cell 8).
 *
 * @param XTrain  Training inputs, shape (n,)
 * @param yTrain  Training observations, shape (n,)
 * @param XTest   Test inputs, shape (m,)
 * @param kernelFn  Closure (X1, X2) => kernel matrix (use `kernelByName1D`)
 * @param sigmaN  Observation-noise standard deviation
 * @param jitter  Diagonal stability term added to (K + σ_n² I); default 1e-8
 */
export function gpPredict(
  XTrain: number[],
  yTrain: number[],
  XTest: number[],
  kernelFn: (X1: number[], X2: number[]) => number[][],
  sigmaN: number,
  jitter = 1e-8,
): GPPredictResult {
  const n = XTrain.length;
  const m = XTest.length;
  const K = kernelFn(XTrain, XTrain);
  const KStar = kernelFn(XTrain, XTest);     // shape (n, m)
  const KStarStar = kernelFn(XTest, XTest);  // shape (m, m)

  // L L^T = K + (σ_n² + jitter) I
  const A = K.map((row) => row.slice());
  addDiagonal(A, sigmaN * sigmaN + jitter);
  const L = choleskyFactor(A);

  // α = A^{-1} y via two triangular solves
  const beta = solveLowerTriangular(L, yTrain);
  const alpha = solveUpperTriangularT(L, beta);

  // Predictive mean: μ_* = K_*^T α
  const mean = new Array<number>(m).fill(0);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += KStar[i][j] * alpha[i];
    mean[j] = s;
  }

  // Predictive covariance: V = L^{-1} K_*; Σ_* = K_** - V^T V
  const V = solveLowerTriangularMatrix(L, KStar); // shape (n, m)
  const cov: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row = new Array<number>(m);
    const KSSi = KStarStar[i];
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += V[k][i] * V[k][j];
      row[j] = KSSi[j] - s;
    }
    cov.push(row);
  }

  // Predictive standard deviation (clamp tiny negatives from rounding).
  const sd = new Array<number>(m);
  for (let i = 0; i < m; i++) sd[i] = Math.sqrt(Math.max(cov[i][i], 0));

  return { mean, cov, sd, L };
}

/**
 * Diagonal-only GP regression predictive distribution.
 *
 * Same as `gpPredict` but skips the O(m^2) materialization of the full m×m
 * predictive covariance matrix and the O(m^2) test-test kernel `K_**`.
 * Returns `{mean, sd, L}` only — pointwise standard deviation suffices for
 * ±2σ band rendering, which is what most viz components need.
 *
 * Cost: O(n^3 + n^2 m + nm) — same Cholesky and forward solve as `gpPredict`,
 * but the cov→sd step drops from O(m^2 n) to O(mn) and the test-test kernel
 * drops from O(m^2) entries to O(m) diagonal calls.
 */
export function gpPredictDiag(
  XTrain: number[],
  yTrain: number[],
  XTest: number[],
  kernelFn: (X1: number[], X2: number[]) => number[][],
  sigmaN: number,
  jitter = 1e-8,
): { mean: number[]; sd: number[]; L: number[][] } {
  const n = XTrain.length;
  const m = XTest.length;
  const K = kernelFn(XTrain, XTrain);
  const KStar = kernelFn(XTrain, XTest); // shape (n, m)

  const A = K.map((row) => row.slice());
  addDiagonal(A, sigmaN * sigmaN + jitter);
  const L = choleskyFactor(A);

  const beta = solveLowerTriangular(L, yTrain);
  const alpha = solveUpperTriangularT(L, beta);

  const mean = new Array<number>(m).fill(0);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += KStar[i][j] * alpha[i];
    mean[j] = s;
  }

  // V = L^{-1} K_* (n × m); pointwise variance is K_**[j,j] - ‖V[:,j]‖²
  const V = solveLowerTriangularMatrix(L, KStar);
  const sd = new Array<number>(m);
  for (let j = 0; j < m; j++) {
    // K_**[j,j] = k(XTest[j], XTest[j]) — single-point kernel call avoids
    // materializing the full m×m K_**.
    const xj = [XTest[j]];
    const kjj = kernelFn(xj, xj)[0][0];
    let v2 = 0;
    for (let k = 0; k < n; k++) v2 += V[k][j] * V[k][j];
    sd[j] = Math.sqrt(Math.max(kjj - v2, 0));
  }

  return { mean, sd, L };
}

/**
 * Joint posterior samples from N(post_mean, post_cov) via Cholesky.
 * Returns an array of `nSamples` paths, each of length `m`.
 */
export function sampleFromPosterior(
  postMean: number[],
  postCov: number[][],
  nSamples: number,
  rng: () => number,
  jitter = 1e-8,
): number[][] {
  const m = postMean.length;
  const Cov = postCov.map((row) => row.slice());
  addDiagonal(Cov, jitter);
  const Lpost = choleskyFactor(Cov);
  const gauss = gaussianSampler(rng);

  const samples: number[][] = [];
  for (let s = 0; s < nSamples; s++) {
    const z = new Array<number>(m);
    for (let i = 0; i < m; i++) z[i] = gauss();
    // path = post_mean + L_post @ z
    const path = new Array<number>(m);
    for (let i = 0; i < m; i++) {
      let acc = postMean[i];
      const Li = Lpost[i];
      for (let k = 0; k <= i; k++) acc += Li[k] * z[k];
      path[i] = acc;
    }
    samples.push(path);
  }
  return samples;
}

/**
 * Sample `nSamples` paths from a centered Gaussian process prior with
 * covariance `K`. Useful for §1 panel (c) and §2 prior galleries.
 */
export function sampleGPPrior(
  K: number[][],
  nSamples: number,
  rng: () => number,
  jitter = 1e-8,
): number[][] {
  const n = K.length;
  const Kj = K.map((row) => row.slice());
  addDiagonal(Kj, jitter);
  const L = choleskyFactor(Kj);
  const gauss = gaussianSampler(rng);
  const samples: number[][] = [];
  for (let s = 0; s < nSamples; s++) {
    const z = new Array<number>(n);
    for (let i = 0; i < n; i++) z[i] = gauss();
    const path = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      const Li = L[i];
      for (let k = 0; k <= i; k++) acc += Li[k] * z[k];
      path[i] = acc;
    }
    samples.push(path);
  }
  return samples;
}

// -----------------------------------------------------------------------------
// Marginal likelihood (Proposition 5.1) and analytic gradient (Proposition 5.2)
//
// Parameterized via log-hyperparameters (log σ_f, log ℓ, log σ_n) so that the
// optimizer operates on an unconstrained R^3 and positivity is automatic.
// SE-kernel only — Matérn variants and ARD are exposed via separate routines
// below if needed.
// -----------------------------------------------------------------------------

/**
 * Negative log marginal likelihood with SE kernel (Proposition 5.1):
 *   -log p(y | X, η) = ½ y^T A^{-1} y + ½ log det A + (n/2) log(2π)
 * where A = K(η) + σ_n² I.
 *
 * Returns +∞ (1e8) if the Cholesky fails (non-PSD) — convex region of the
 * log-likelihood landscape always admits a valid Cholesky, so this only fires
 * on degenerate parameter values that L-BFGS-B will retreat from.
 */
export function negLogMarginalSE(
  logParams: [number, number, number], // [log σ_f, log ℓ, log σ_n]
  X: number[],
  y: number[],
  jitter = 1e-8,
): number {
  const [logSf, logEll, logSn] = logParams;
  const sigmaF = Math.exp(logSf);
  const ell = Math.exp(logEll);
  const sigmaN = Math.exp(logSn);
  const n = y.length;
  const K = kernelSE(X, X, sigmaF, ell);
  const A = K.map((row) => row.slice());
  addDiagonal(A, sigmaN * sigmaN + jitter);
  let L: number[][];
  try {
    L = choleskyFactor(A);
  } catch {
    return 1e8;
  }
  const beta = solveLowerTriangular(L, y);
  const dataFit = 0.5 * dot(beta, beta);
  let logDet = 0;
  for (let i = 0; i < n; i++) logDet += Math.log(L[i][i]);
  return dataFit + logDet + 0.5 * n * Math.log(2 * Math.PI);
}

/**
 * Negative log marginal likelihood with SE kernel, plus analytic gradient
 * (Proposition 5.2):
 *   ∂/∂η_k = -½ tr[(α α^T - A^{-1}) ∂A/∂η_k]
 * where α = A^{-1} y. Implemented via the trace identity
 *   tr(B M) = ∑_{ij} B_ij M_ji = ∑_{ij} B_ij M_ij  (when M is symmetric)
 * which avoids materializing the n×n product.
 *
 * Partial derivatives wrt log-hyperparameters:
 *   ∂A/∂(log σ_f) = 2 K
 *   ∂A/∂(log ℓ)   = K * sqdist / ℓ²
 *   ∂A/∂(log σ_n) = 2 σ_n² I
 */
export function negLogMarginalSEWithGrad(
  logParams: [number, number, number],
  X: number[],
  y: number[],
  jitter = 1e-8,
): { nll: number; grad: [number, number, number] } {
  const [logSf, logEll, logSn] = logParams;
  const sigmaF = Math.exp(logSf);
  const ell = Math.exp(logEll);
  const sigmaN = Math.exp(logSn);
  const n = y.length;
  const ell2 = ell * ell;

  // Precompute squared-distance matrix (used both in K and in ∂A/∂ log ℓ).
  const sqdist: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      const d = X[i] - X[j];
      row[j] = d * d;
    }
    sqdist.push(row);
  }
  const K: number[][] = [];
  const sf2 = sigmaF * sigmaF;
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n);
    const sqi = sqdist[i];
    for (let j = 0; j < n; j++) row[j] = sf2 * Math.exp(-sqi[j] / (2 * ell2));
    K.push(row);
  }
  const A = K.map((row) => row.slice());
  addDiagonal(A, sigmaN * sigmaN + jitter);

  let L: number[][];
  try {
    L = choleskyFactor(A);
  } catch {
    return { nll: 1e8, grad: [0, 0, 0] };
  }
  const beta = solveLowerTriangular(L, y);
  const alpha = solveUpperTriangularT(L, beta);

  const nll =
    0.5 * dot(beta, beta) +
    (() => {
      let s = 0;
      for (let i = 0; i < n; i++) s += Math.log(L[i][i]);
      return s;
    })() +
    0.5 * n * Math.log(2 * Math.PI);

  // inner = α α^T - A^{-1}; we accumulate -½ tr(inner · ∂A/∂η_k) directly,
  // which matches the notebook's `grad = -0.5 * np.sum(inner * dA_dη_k)`.
  const Ainv = choleskyInverse(L);

  let gSf = 0;
  let gEll = 0;
  let gSn = 0;
  for (let i = 0; i < n; i++) {
    const ai = alpha[i];
    const Ki = K[i];
    const Ainvi = Ainv[i];
    const sqi = sqdist[i];
    for (let j = 0; j < n; j++) {
      const innerIj = ai * alpha[j] - Ainvi[j];
      // ∂A/∂(log σ_f) = 2 K  →  -½ · inner_ij · 2 K_ij = -inner_ij · K_ij
      gSf -= innerIj * Ki[j];
      // ∂A/∂(log ℓ)   = K · sqdist / ℓ²  →  -½ · inner_ij · K_ij · sqdist_ij / ℓ²
      gEll -= 0.5 * innerIj * Ki[j] * sqi[j] / ell2;
    }
    // ∂A/∂(log σ_n) = 2 σ_n² I (only diagonal)  →  -σ_n² · inner_ii
    const innerII = ai * ai - Ainvi[i];
    gSn -= sigmaN * sigmaN * innerII;
  }

  return { nll, grad: [gSf, gEll, gSn] };
}

// -----------------------------------------------------------------------------
// Multi-restart L-BFGS-B style optimizer
//
// We use a minimal port of L-BFGS with a backtracking line search; box
// constraints are unnecessary because we operate in unconstrained log-space.
// The hyperparameter recovery problem on the §3 dataset is well-conditioned at
// the truth, so a few-iteration L-BFGS converges reliably from random
// log-uniform restarts in [-2, 2].
// -----------------------------------------------------------------------------

export interface LBFGSResult {
  /** Optimum point. */
  x: number[];
  /** Objective value at the optimum. */
  f: number;
  /** Per-iteration objective trace, for §5 panel (c) restart curves. */
  history: number[];
  /** Number of iterations taken. */
  iter: number;
}

/**
 * Minimal L-BFGS with backtracking line search (Armijo). Operates on R^d,
 * unbounded. `f` returns `{nll, grad}` jointly.
 *
 * `m` is the L-BFGS memory size; default 10 is enough for d=3.
 *
 * The optional `fVal` callback returns NLL only; when provided, the line
 * search uses it instead of `f` so failed backtracking steps don't pay the
 * O(n^3) cost of computing the analytic gradient. The full `f` is only
 * called once the line search has accepted a step.
 */
export function lbfgs(
  f: (x: number[]) => { nll: number; grad: number[] },
  x0: number[],
  options?: {
    maxIter?: number;
    gradTol?: number;
    memory?: number;
    fVal?: (x: number[]) => number;
  },
): LBFGSResult {
  const maxIter = options?.maxIter ?? 100;
  const gradTol = options?.gradTol ?? 1e-6;
  const m = options?.memory ?? 10;
  const d = x0.length;
  const fVal = options?.fVal ?? ((x: number[]) => f(x).nll);

  let x = x0.slice();
  let { nll, grad } = f(x);
  const history: number[] = [nll];

  // L-BFGS memory: pairs (s_k, y_k) and rho_k = 1 / (y_k^T s_k).
  const sList: number[][] = [];
  const yList: number[][] = [];
  const rhoList: number[] = [];

  let iter = 0;
  for (; iter < maxIter; iter++) {
    const gnorm = Math.sqrt(grad.reduce((acc, gi) => acc + gi * gi, 0));
    if (gnorm < gradTol) break;

    // Two-loop recursion to compute search direction d_k = -H_k g_k.
    const q = grad.slice();
    const alphas: number[] = new Array(sList.length).fill(0);
    for (let i = sList.length - 1; i >= 0; i--) {
      alphas[i] = rhoList[i] * dot(sList[i], q);
      for (let j = 0; j < d; j++) q[j] -= alphas[i] * yList[i][j];
    }
    // Initial Hessian: γ_k I where γ_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1}).
    let gamma = 1;
    if (sList.length > 0) {
      const sLast = sList[sList.length - 1];
      const yLast = yList[yList.length - 1];
      gamma = dot(sLast, yLast) / dot(yLast, yLast);
    }
    const r = q.map((qi) => gamma * qi);
    for (let i = 0; i < sList.length; i++) {
      const beta = rhoList[i] * dot(yList[i], r);
      for (let j = 0; j < d; j++) r[j] += sList[i][j] * (alphas[i] - beta);
    }
    // Search direction
    const dir = r.map((ri) => -ri);

    // Armijo backtracking. Use the cheap value-only callback during the
    // search; we only need the gradient once a step is accepted.
    const c1 = 1e-4;
    const fxg = dot(grad, dir);
    let step = 1;
    let xNew = x.map((xi, j) => xi + step * dir[j]);
    let fNew = fVal(xNew);
    let backtrack = 0;
    while (
      (!Number.isFinite(fNew) || fNew > nll + c1 * step * fxg) &&
      backtrack < 30
    ) {
      step *= 0.5;
      xNew = x.map((xi, j) => xi + step * dir[j]);
      fNew = fVal(xNew);
      backtrack++;
    }
    if (backtrack >= 30) break;

    // Step accepted. Now compute the gradient at xNew (full f-call).
    const fGradResult = f(xNew);
    const gNew = fGradResult.grad;
    // Use the gradient-call's NLL — it should match fNew within numerical
    // precision; preferring it keeps NLL and grad mutually consistent.
    fNew = fGradResult.nll;

    // Update memory: s_k = x_new - x; y_k = grad_new - grad
    const sk = xNew.map((xn, j) => xn - x[j]);
    const yk = gNew.map((gn, j) => gn - grad[j]);
    const sy = dot(sk, yk);
    if (sy > 1e-10) {
      sList.push(sk);
      yList.push(yk);
      rhoList.push(1 / sy);
      if (sList.length > m) {
        sList.shift();
        yList.shift();
        rhoList.shift();
      }
    }

    x = xNew;
    nll = fNew;
    grad = gNew;
    history.push(nll);
  }

  return { x, f: nll, history, iter };
}

/**
 * Multi-restart L-BFGS for SE-kernel marginal-likelihood maximization.
 * Returns the best-restart result plus per-restart histories for §5 panel (c).
 *
 * Each restart initializes log-hyperparameters uniformly in [-2, 2]³.
 */
export function fitSEMarginalLikelihood(
  X: number[],
  y: number[],
  rng: () => number,
  nRestarts = 5,
): {
  best: { sigmaF: number; lengthscale: number; sigmaN: number; logLikelihood: number };
  restarts: Array<{
    init: [number, number, number];
    x: [number, number, number];
    sigmaF: number;
    lengthscale: number;
    sigmaN: number;
    logLikelihood: number;
    /** Per-iteration *log-marginal-likelihood* trace for the panel-(c) overlay. */
    history: number[];
  }>;
} {
  const restarts: Array<{
    init: [number, number, number];
    x: [number, number, number];
    sigmaF: number;
    lengthscale: number;
    sigmaN: number;
    logLikelihood: number;
    history: number[];
  }> = [];

  for (let r = 0; r < nRestarts; r++) {
    const init: [number, number, number] = [
      -2 + 4 * rng(),
      -2 + 4 * rng(),
      -2 + 4 * rng(),
    ];
    const result = lbfgs(
      (x) => {
        const r2 = negLogMarginalSEWithGrad([x[0], x[1], x[2]], X, y);
        return { nll: r2.nll, grad: r2.grad };
      },
      init.slice(),
      {
        maxIter: 120,
        gradTol: 1e-6,
        // Cheap value-only path for the line search — skips the analytic
        // gradient's O(n^3) inverse on failed backtracking steps.
        fVal: (x) => negLogMarginalSE([x[0], x[1], x[2]], X, y),
      },
    );
    const x: [number, number, number] = [result.x[0], result.x[1], result.x[2]];
    restarts.push({
      init,
      x,
      sigmaF: Math.exp(x[0]),
      lengthscale: Math.exp(x[1]),
      sigmaN: Math.exp(x[2]),
      logLikelihood: -result.f,
      history: result.history.map((h) => -h),
    });
  }

  // Best by log-marginal-likelihood (max).
  let bestIdx = 0;
  for (let i = 1; i < restarts.length; i++) {
    if (restarts[i].logLikelihood > restarts[bestIdx].logLikelihood) bestIdx = i;
  }
  const best = restarts[bestIdx];
  return {
    best: {
      sigmaF: best.sigmaF,
      lengthscale: best.lengthscale,
      sigmaN: best.sigmaN,
      logLikelihood: best.logLikelihood,
    },
    restarts,
  };
}

// -----------------------------------------------------------------------------
// Sparse approximations (§6)
//
// Three approaches that all reduce the O(n³) Cholesky cost:
//   - Nyström: rank-m kernel surrogate via inducing points (Woodbury identity)
//   - SVGP:    Titsias 2009 closed-form q(u) for conjugate Gaussian likelihood
//   - RFF:     Bayesian linear regression in random Fourier features (Bochner)
// -----------------------------------------------------------------------------

/** Nyström approximation: rank-m surrogate K̂ = K_nm K_mm^{-1} K_nm^T.
 *  Predictive mean uses Woodbury; predictive variance is intentionally omitted
 *  (Nyström variance is structurally unreliable per §6 brief). */
export function gpPredictNystrom(
  XTrain: number[],
  yTrain: number[],
  XInducing: number[],
  XTest: number[],
  kernelFn: (X1: number[], X2: number[]) => number[][],
  sigmaN: number,
  jitter = 1e-6,
): { mean: number[] } {
  const m = XInducing.length;
  const Kmm = kernelFn(XInducing, XInducing);
  const Knm = kernelFn(XTrain, XInducing);     // shape (n, m)
  const Ktm = kernelFn(XTest, XInducing);      // shape (n_test, m)

  // Kmm + jitter I, Cholesky for solves
  const Amm = Kmm.map((row) => row.slice());
  addDiagonal(Amm, jitter);
  const Lmm = choleskyFactor(Amm);

  // Compute α_m = (σ_n² K_mm + K_nm^T K_nm)^{-1} K_nm^T y  via Cholesky on
  // M = σ_n² K_mm + K_nm^T K_nm.
  // K_nm^T y has shape (m,)
  const KnmT_y = new Array<number>(m).fill(0);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < XTrain.length; i++) s += Knm[i][j] * yTrain[i];
    KnmT_y[j] = s;
  }
  // M = σ_n² K_mm + K_nm^T K_nm  (m × m)
  const M: number[][] = [];
  for (let i = 0; i < m; i++) M.push(new Array<number>(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let s = sigmaN * sigmaN * Kmm[i][j];
      for (let k = 0; k < XTrain.length; k++) s += Knm[k][i] * Knm[k][j];
      M[i][j] = s;
    }
  }
  addDiagonal(M, jitter);
  const Lm = choleskyFactor(M);
  const aBeta = solveLowerTriangular(Lm, KnmT_y);
  const alphaM = solveUpperTriangularT(Lm, aBeta);

  // Predictive mean: K_tm α_m
  const mean = new Array<number>(XTest.length).fill(0);
  for (let i = 0; i < XTest.length; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += Ktm[i][j] * alphaM[j];
    mean[i] = s;
  }
  return { mean };
}

/** SVGP / Titsias 2009 closed-form predictive distribution for conjugate
 *  Gaussian likelihood. The closed-form q(u) is the global ELBO maximum; no
 *  SGD loop is needed in this case. */
export function gpPredictSVGP(
  XTrain: number[],
  yTrain: number[],
  XInducing: number[],
  XTest: number[],
  kernelFn: (X1: number[], X2: number[]) => number[][],
  sigmaN: number,
  jitter = 1e-6,
): { mean: number[]; sd: number[] } {
  const n = XTrain.length;
  const m = XInducing.length;
  const mTest = XTest.length;
  const Kmm = kernelFn(XInducing, XInducing);
  const Knm = kernelFn(XTrain, XInducing); // (n, m)
  const Ktm = kernelFn(XTest, XInducing);  // (m_test, m)

  // Diagonal of K_** via single-point kernel calls — O(m_test) instead of O(m_test^2).
  const KttDiag = new Array<number>(mTest);
  for (let i = 0; i < mTest; i++) {
    const xi = [XTest[i]];
    KttDiag[i] = kernelFn(xi, xi)[0][0];
  }

  // Hensman et al. 2013, Eq. 8–10:
  //   Σ_inv = K_mm + σ_n^{-2} K_nm^T K_nm
  //   q(u) = N(u; m_q, K_mm Σ_inv^{-1} K_mm)
  //   m_q  = σ_n^{-2} K_mm Σ_inv^{-1} K_nm^T y
  //   Predictive μ_t = K_tm K_mm^{-1} m_q
  //                  = K_tm K_mm^{-1} K_mm Σ_inv^{-1} (K_nm^T y / σ_n²)
  //                  = K_tm a,    with  a = Σ_inv^{-1} (K_nm^T y / σ_n²)
  //   Predictive σ_t² = K_tt - K_tm K_mm^{-1} K_mt + K_tm Σ_inv^{-1} K_mt
  // (function-level variance; callers add σ_n² for observation-level intervals).

  const sn2 = sigmaN * sigmaN;
  const SigmaInv: number[][] = [];
  for (let i = 0; i < m; i++) SigmaInv.push(new Array<number>(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let s = Kmm[i][j];
      for (let k = 0; k < n; k++) s += Knm[k][i] * Knm[k][j] / sn2;
      SigmaInv[i][j] = s;
    }
  }
  addDiagonal(SigmaInv, jitter);
  const Lsig = choleskyFactor(SigmaInv);

  // a = Σ_inv^{-1} (K_nm^T y / σ_n²)
  const KnmT_y = new Array<number>(m).fill(0);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Knm[i][j] * yTrain[i];
    KnmT_y[j] = s / sn2;
  }
  const beta1 = solveLowerTriangular(Lsig, KnmT_y);
  const a = solveUpperTriangularT(Lsig, beta1);

  // Cholesky of K_mm + jitter I for the K_mm^{-1} K_mt term in the variance.
  const Kmm_jit = Kmm.map((row) => row.slice());
  addDiagonal(Kmm_jit, jitter);
  const LmmK = choleskyFactor(Kmm_jit);

  // Predictive mean: μ_t = K_tm K_mm^{-1} (K_mm a) = K_tm a — the round trip
  // through K_mm^{-1} cancels K_mm exactly, so we use `a` directly.
  const mean = new Array<number>(mTest).fill(0);
  for (let i = 0; i < mTest; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += Ktm[i][j] * a[j];
    mean[i] = s;
  }

  // Predictive variance, batched. Build K_mt = K_tm^T (m × m_test), then
  //   V_mm  = L_mmK^{-1} K_mt   →  diag(K_tm K_mm^{-1} K_mt)[i]  = ‖V_mm[:,i]‖²
  //   V_sig = L_sig^{-1} K_mt   →  diag(K_tm Σ_inv^{-1} K_mt)[i] = ‖V_sig[:,i]‖²
  // One matrix solve per term, instead of m_test independent vector solves.
  const Kmt: number[][] = [];
  for (let j = 0; j < m; j++) {
    const row = new Array<number>(mTest);
    for (let i = 0; i < mTest; i++) row[i] = Ktm[i][j];
    Kmt.push(row);
  }
  const Vmm = solveLowerTriangularMatrix(LmmK, Kmt);  // m × m_test
  const Vsig = solveLowerTriangularMatrix(Lsig, Kmt); // m × m_test

  const sd = new Array<number>(mTest).fill(0);
  for (let i = 0; i < mTest; i++) {
    let term1 = 0;
    let term2 = 0;
    for (let j = 0; j < m; j++) {
      term1 += Vmm[j][i] * Vmm[j][i];
      term2 += Vsig[j][i] * Vsig[j][i];
    }
    const variance = KttDiag[i] - term1 + term2;
    // Function-level standard deviation, matching `gpPredict` and `rffPredict`.
    sd[i] = Math.sqrt(Math.max(variance, 0));
  }

  return { mean, sd };
}

/** Random Fourier features for the SE kernel (Rahimi-Recht 2007).
 *  Predictive mean and variance via Bayesian linear regression in feature
 *  space. Cost O(n D² + D³) instead of O(n³).
 *
 *  ω_i ~ N(0, ℓ^{-2})  (spectral density of SE kernel)
 *  φ(x)_i = √(2σ_f² / D) cos(ω_i^T x + b_i),   b_i ~ Uniform[0, 2π) */
export function rffPredict(
  XTrain: number[],
  yTrain: number[],
  XTest: number[],
  sigmaF: number,
  lengthscale: number,
  sigmaN: number,
  D: number,
  rng: () => number,
  jitter = 1e-6,
): { mean: number[]; sd: number[] } {
  const gauss = gaussianSampler(rng);
  const n = XTrain.length;
  // Sample frequencies ω_i ~ N(0, 1/ℓ²) for the 1D case.
  const omega = new Array<number>(D);
  for (let i = 0; i < D; i++) omega[i] = gauss() / lengthscale;
  // Sample phases b_i ~ Uniform[0, 2π).
  const b = new Array<number>(D);
  for (let i = 0; i < D; i++) b[i] = 2 * Math.PI * rng();

  const scale = Math.sqrt(2 * sigmaF * sigmaF / D);

  function featureMap(X: number[]): number[][] {
    const Phi: number[][] = [];
    for (let k = 0; k < X.length; k++) {
      const row = new Array<number>(D);
      const xk = X[k];
      for (let i = 0; i < D; i++) row[i] = scale * Math.cos(omega[i] * xk + b[i]);
      Phi.push(row);
    }
    return Phi;
  }

  const PhiTrain = featureMap(XTrain);   // (n, D)
  const PhiTest = featureMap(XTest);     // (n_test, D)

  // Bayesian linear regression posterior:
  //   weights ~ N(μ_w, Σ_w)
  //   Σ_w^{-1} = (1/σ_n²) Phi^T Phi + I_D
  //   μ_w = (1/σ_n²) Σ_w Phi^T y
  // Predictive: f(x*) ~ N(φ(x*)^T μ_w, σ_n² + φ(x*)^T Σ_w φ(x*)). We return
  // function-level sd (excluding σ_n²).
  const sn2 = sigmaN * sigmaN;
  const SigmaWInv: number[][] = [];
  for (let i = 0; i < D; i++) SigmaWInv.push(new Array<number>(D).fill(0));
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += PhiTrain[k][i] * PhiTrain[k][j];
      SigmaWInv[i][j] = s / sn2;
    }
    SigmaWInv[i][i] += 1; // unit prior precision in feature space
  }
  addDiagonal(SigmaWInv, jitter);
  const Lw = choleskyFactor(SigmaWInv);

  // Phi^T y
  const PhiT_y = new Array<number>(D).fill(0);
  for (let i = 0; i < D; i++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += PhiTrain[k][i] * yTrain[k];
    PhiT_y[i] = s / sn2;
  }
  const betaW = solveLowerTriangular(Lw, PhiT_y);
  const muW = solveUpperTriangularT(Lw, betaW);

  // Predictions
  const mean = new Array<number>(XTest.length).fill(0);
  const sd = new Array<number>(XTest.length).fill(0);
  for (let i = 0; i < XTest.length; i++) {
    const phi = PhiTest[i];
    let m = 0;
    for (let j = 0; j < D; j++) m += phi[j] * muW[j];
    mean[i] = m;
    // φ^T Σ_w φ: solve SigmaWInv u = φ, then compute φ^T u
    const b1 = solveLowerTriangular(Lw, phi);
    const sigW_phi = solveUpperTriangularT(Lw, b1);
    let v = 0;
    for (let j = 0; j < D; j++) v += phi[j] * sigW_phi[j];
    sd[i] = Math.sqrt(Math.max(v, 0));
  }
  return { mean, sd };
}

// Robust PCA (Principal Component Pursuit) solved in the browser.
//
// Each (rank, corruption) preset draws a 30×25 test matrix X = L + S with a
// seeded PRNG, then runs the same ADMM as the article's `robust_pca_admm`:
//   λ = 1/√max(m, n),  μ = mn / (4‖X‖₁),
//   L ← SVT_{1/μ}(X − S + Y/μ),  S ← soft_{λ/μ}(X − L + Y/μ),
//   Y ← Y + μ(X − L − S),  stop when ‖X − L − S‖_F / ‖X‖_F < 1e-7 (≤ 500 iters).
// The readouts (recovered rank, support, relative error, iterations) come from
// that solve. Results are memoized per preset.

// ─── Types ───

export interface RobustPCAResult {
  rank: number;
  corruption: number;
  rows: number;
  cols: number;
  /** Observed matrix X = L + S. */
  matrixX: number[][];
  /** Ground-truth low-rank and sparse components. */
  matrixLTrue: number[][];
  matrixSTrue: number[][];
  /** ADMM estimates L̂ and Ŝ. */
  matrixLHat: number[][];
  matrixSHat: number[][];
  /** Leading singular values (descending). */
  singularValuesX: number[];
  singularValuesLTrue: number[];
  singularValuesLHat: number[];
  /** Number of singular values that survive the final SVT step (rank of L̂). */
  recoveredRank: number;
  nnzSTrue: number;
  nnzSHat: number;
  /** Entries in supp(S) ∩ supp(Ŝ). */
  supportMatched: number;
  /** ‖L̂ − L‖_F / ‖L‖_F. */
  relativeError: number;
  iterations: number;
  converged: boolean;
  lambda: number;
  mu: number;
}

// ─── Constants ───

export const ROWS = 30;
export const COLS = 25;
const TOP_K_SV = 15;
const TOL = 1e-7;
const MAX_ITER = 500;

// ─── Seeded PRNG (mulberry32 + Box–Muller) ───

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalSampler(uniform: () => number): () => number {
  return () => {
    const u1 = Math.max(uniform(), 1e-300);
    const u2 = uniform();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

// ─── Dense helpers (row-major Float64Array, m × n) ───

type Mat = Float64Array;

function toRows(A: Mat, m: number, n: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < m; i++) out.push(Array.from(A.subarray(i * n, (i + 1) * n)));
  return out;
}

function frob(A: Mat): number {
  let s = 0;
  for (let i = 0; i < A.length; i++) s += A[i] * A[i];
  return Math.sqrt(s);
}

/**
 * One-sided (Hestenes) Jacobi SVD of an m × n matrix with m ≥ n.
 * Returns W = A·V (whose columns are σ_k u_k), the column norms σ, and V (n × n,
 * column k stored at V[k*n .. k*n + n)). An optional orthogonal warm start V0
 * speeds up repeated SVDs of slowly changing matrices, as in the ADMM loop.
 */
function jacobiSVD(
  A: Mat,
  m: number,
  n: number,
  V0?: Float64Array,
): { W: Float64Array; sigma: Float64Array; V: Float64Array } {
  // W stored column-major: W[k*m + i].
  const W = new Float64Array(m * n);
  const V = new Float64Array(n * n);
  if (V0) {
    V.set(V0);
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < m; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += A[i * n + j] * V0[k * n + j];
        W[k * m + i] = s;
      }
    }
  } else {
    for (let k = 0; k < n; k++) {
      V[k * n + k] = 1;
      for (let i = 0; i < m; i++) W[k * m + i] = A[i * n + k];
    }
  }

  const eps = 1e-12;
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        const op = p * m;
        const oq = q * m;
        for (let i = 0; i < m; i++) {
          const wp = W[op + i];
          const wq = W[oq + i];
          alpha += wp * wp;
          beta += wq * wq;
          gamma += wp * wq;
        }
        if (gamma === 0 || Math.abs(gamma) <= eps * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let i = 0; i < m; i++) {
          const wp = W[op + i];
          const wq = W[oq + i];
          W[op + i] = c * wp - s * wq;
          W[oq + i] = s * wp + c * wq;
        }
        const vp = p * n;
        const vq = q * n;
        for (let j = 0; j < n; j++) {
          const a = V[vp + j];
          const b = V[vq + j];
          V[vp + j] = c * a - s * b;
          V[vq + j] = s * a + c * b;
        }
      }
    }
    if (!rotated) break;
  }

  const sigma = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i < m; i++) s += W[k * m + i] * W[k * m + i];
    sigma[k] = Math.sqrt(s);
  }
  return { W, sigma, V };
}

function singularValues(A: Mat, m: number, n: number, k: number): number[] {
  const { sigma } = jacobiSVD(A, m, n);
  return Array.from(sigma)
    .sort((a, b) => b - a)
    .slice(0, k);
}

// ─── Principal Component Pursuit via ADMM ───

export interface ADMMOutput {
  L: Mat;
  S: Mat;
  rank: number;
  iterations: number;
  converged: boolean;
  lambda: number;
  mu: number;
}

/** The article's `robust_pca_admm`, for an m × n row-major X with m ≥ n. */
export function robustPCAADMM(X: Mat, m: number, n: number, tol = TOL, maxIter = MAX_ITER): ADMMOutput {
  const size = m * n;
  const lambda = 1 / Math.sqrt(Math.max(m, n));
  let absSum = 0;
  for (let i = 0; i < size; i++) absSum += Math.abs(X[i]);
  const mu = (m * n) / (4 * absSum);
  const normX = frob(X);

  const L = new Float64Array(size);
  const S = new Float64Array(size);
  const Y = new Float64Array(size);
  const Z = new Float64Array(size);
  let Vwarm: Float64Array | undefined;
  let rank = 0;
  let iterations = 0;
  let converged = false;

  for (let it = 0; it < maxIter; it++) {
    iterations = it + 1;

    // L-update: singular value thresholding of X − S + Y/μ at 1/μ.
    for (let i = 0; i < size; i++) Z[i] = X[i] - S[i] + Y[i] / mu;
    const { W, sigma, V } = jacobiSVD(Z, m, n, Vwarm);
    Vwarm = V;
    L.fill(0);
    rank = 0;
    const tau = 1 / mu;
    for (let k = 0; k < n; k++) {
      if (sigma[k] <= tau) continue;
      rank++;
      const scale = (sigma[k] - tau) / sigma[k];
      for (let i = 0; i < m; i++) {
        const wi = W[k * m + i] * scale;
        if (wi === 0) continue;
        for (let j = 0; j < n; j++) L[i * n + j] += wi * V[k * n + j];
      }
    }

    // S-update: entrywise soft-thresholding of X − L + Y/μ at λ/μ.
    const thr = lambda / mu;
    for (let i = 0; i < size; i++) {
      const z = X[i] - L[i] + Y[i] / mu;
      const a = Math.abs(z) - thr;
      S[i] = a > 0 ? (z > 0 ? a : -a) : 0;
    }

    // Dual ascent and stopping rule.
    let r2 = 0;
    for (let i = 0; i < size; i++) {
      const r = X[i] - L[i] - S[i];
      Y[i] += mu * r;
      r2 += r * r;
    }
    if (Math.sqrt(r2) / normX < tol) {
      converged = true;
      break;
    }
  }

  return { L, S, rank, iterations, converged, lambda, mu };
}

// ─── Test-matrix generation ───

/** L = U Vᵀ with U (m × r), V (n × r) i.i.d. N(0, 1); spikes ±U(5, 15) on a Bernoulli(ρ) mask. */
export function generateProblem(
  m: number,
  n: number,
  rank: number,
  corruption: number,
  seed: number,
): { X: Mat; L: Mat; S: Mat } {
  const uniform = mulberry32(seed);
  const normal = normalSampler(uniform);
  const U = Array.from({ length: m * rank }, normal);
  const Vt = Array.from({ length: n * rank }, normal);
  const L = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < rank; k++) s += U[i * rank + k] * Vt[j * rank + k];
      L[i * n + j] = s;
    }
  }
  const S = new Float64Array(m * n);
  for (let i = 0; i < m * n; i++) {
    if (uniform() < corruption) {
      const sign = uniform() < 0.5 ? -1 : 1;
      S[i] = sign * (5 + 10 * uniform());
    }
  }
  const X = new Float64Array(m * n);
  for (let i = 0; i < m * n; i++) X[i] = L[i] + S[i];
  return { X, L, S };
}

function computeResult(rank: number, corruption: number): RobustPCAResult {
  const m = ROWS;
  const n = COLS;
  const seed = 1000 * rank + Math.round(corruption * 100);
  const { X, L, S } = generateProblem(m, n, rank, corruption, seed);
  const sol = robustPCAADMM(X, m, n);

  let nnzSTrue = 0;
  let nnzSHat = 0;
  let supportMatched = 0;
  let errL = 0;
  for (let i = 0; i < m * n; i++) {
    const t = S[i] !== 0;
    const h = sol.S[i] !== 0;
    if (t) nnzSTrue++;
    if (h) nnzSHat++;
    if (t && h) supportMatched++;
    const d = sol.L[i] - L[i];
    errL += d * d;
  }

  return {
    rank,
    corruption,
    rows: m,
    cols: n,
    matrixX: toRows(X, m, n),
    matrixLTrue: toRows(L, m, n),
    matrixSTrue: toRows(S, m, n),
    matrixLHat: toRows(sol.L, m, n),
    matrixSHat: toRows(sol.S, m, n),
    singularValuesX: singularValues(X, m, n, TOP_K_SV),
    singularValuesLTrue: singularValues(L, m, n, TOP_K_SV),
    singularValuesLHat: singularValues(sol.L, m, n, TOP_K_SV),
    recoveredRank: sol.rank,
    nnzSTrue,
    nnzSHat,
    supportMatched,
    relativeError: Math.sqrt(errL) / frob(L),
    iterations: sol.iterations,
    converged: sol.converged,
    lambda: sol.lambda,
    mu: sol.mu,
  };
}

// ─── Export (memoized per preset) ───

const cache = new Map<string, RobustPCAResult>();

/** Solve PCP for one (rank, corruption) preset; repeated calls hit the cache. */
export function getRobustPCAResult(rank: number, corruption: number): RobustPCAResult {
  const key = `${rank}|${corruption}`;
  let hit = cache.get(key);
  if (!hit) {
    hit = computeResult(rank, corruption);
    cache.set(key, hit);
  }
  return hit;
}


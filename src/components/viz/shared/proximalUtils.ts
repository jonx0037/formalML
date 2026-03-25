/**
 * Shared math utilities for proximal method visualizations.
 * All exports are pure functions — no module-level state.
 *
 * For vector utilities (dot, norm2, matVec) and quadratic construction
 * (makeQuadratic, nesterovStep), import from gradientDescentUtils.ts.
 */

// ── Types ──────────────────────────────────────────────────────────────

export interface LassoProblem {
  A: number[][];
  xTrue: number[];
  b: number[];
  support: number[];
  n: number;
  p: number;
  L: number; // Lipschitz constant of the smooth part
}

// ── Proximal operators ─────────────────────────────────────────────────

/**
 * Soft-thresholding: prox of λ‖·‖₁, applied componentwise.
 * S_λ(v)_i = sign(v_i) · max(|v_i| - λ, 0)
 */
export function softThreshold(x: number[], lambda: number): number[] {
  return x.map((xi) => Math.sign(xi) * Math.max(Math.abs(xi) - lambda, 0));
}

/** Scalar soft-thresholding (convenience for 1D viz). */
export function softThresholdScalar(v: number, lambda: number): number {
  return Math.sign(v) * Math.max(Math.abs(v) - lambda, 0);
}

/**
 * Projection onto the box [lo, hi]^n.
 * This is the prox of the indicator function ι_{[lo,hi]^n}.
 */
export function projectBox(x: number[], lo: number, hi: number): number[] {
  return x.map((xi) => Math.max(lo, Math.min(hi, xi)));
}

// ── Lasso problem utilities ────────────────────────────────────────────

/**
 * Gradient of the smooth part of Lasso: g(x) = (1/2)‖Ax - b‖².
 * Returns ∇g(x) = Aᵀ(Ax - b).
 */
export function lassoGradient(A: number[][], x: number[], b: number[]): number[] {
  const n = A.length;
  const p = x.length;

  // Compute residual r = Ax - b
  const r = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = -b[i];
    for (let j = 0; j < p; j++) {
      sum += A[i][j] * x[j];
    }
    r[i] = sum;
  }

  // Compute Aᵀr
  const grad = new Array(p);
  for (let j = 0; j < p; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += A[i][j] * r[i];
    }
    grad[j] = sum;
  }

  return grad;
}

/**
 * Compute ‖Ax - b‖² / 2 + λ‖x‖₁ (full Lasso objective).
 */
export function lassoObjective(
  A: number[][],
  x: number[],
  b: number[],
  lambda: number,
): number {
  const n = A.length;
  const p = x.length;
  let smoothPart = 0;
  for (let i = 0; i < n; i++) {
    let ax = 0;
    for (let j = 0; j < p; j++) {
      ax += A[i][j] * x[j];
    }
    smoothPart += (ax - b[i]) ** 2;
  }
  let l1Part = 0;
  for (let j = 0; j < p; j++) {
    l1Part += Math.abs(x[j]);
  }
  return 0.5 * smoothPart + lambda * l1Part;
}

/**
 * Generate a reproducible Lasso problem for visualizations.
 * Uses a simple seeded PRNG (xoshiro128) for deterministic results.
 *
 * Returns A (n×p), xTrue (p), b (n), support indices, and L (Lipschitz constant).
 */
export function generateLassoProblem(
  n: number,
  p: number,
  sparsity: number,
  seed: number,
): LassoProblem {
  const rng = seededRng(seed);

  // Generate A with entries ~ N(0, 1/√n)
  const scale = 1 / Math.sqrt(n);
  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < p; j++) {
      row.push(rng.randn() * scale);
    }
    A.push(row);
  }

  // Sparse ground truth
  const xTrue = new Array(p).fill(0);
  const support: number[] = [];
  const step = Math.floor(p / sparsity);
  for (let s = 0; s < sparsity; s++) {
    const idx = Math.min(s * step + Math.floor(step / 2), p - 1);
    support.push(idx);
    xTrue[idx] = (rng.randn() > 0 ? 1 : -1) * (1.0 + rng.rand() * 2.0);
  }

  // b = Ax_true + noise
  const b: number[] = [];
  for (let i = 0; i < n; i++) {
    let ax = 0;
    for (let j = 0; j < p; j++) {
      ax += A[i][j] * xTrue[j];
    }
    b.push(ax + 0.05 * rng.randn());
  }

  // Estimate L = largest eigenvalue of AᵀA via power iteration
  const L = estimateLipschitz(A, n, p);

  return { A, xTrue, b, support, n, p, L };
}

// ── ADMM solver ────────────────────────────────────────────────────────

export interface ADMMResult {
  z: number[];
  objectives: number[];
  primalResiduals: number[];
  dualResiduals: number[];
}

/**
 * Solve Lasso via ADMM: min (1/2)‖Ax-b‖² + λ‖x‖₁
 * using consensus splitting x = z.
 */
export function solveADMM(
  A: number[][],
  b: number[],
  lambda: number,
  rho: number,
  maxIter: number,
): ADMMResult {
  const n = A.length;
  const p = A[0].length;

  // Precompute (AᵀA + ρI)⁻¹ and Aᵀb
  const AtA = computeAtA(A, n, p);
  const Atb = computeAtb(A, b, n, p);

  // Add ρI and invert
  const M = AtA.map((row) => row.slice());
  for (let j = 0; j < p; j++) M[j][j] += rho;
  const Minv = invertMatrix(M, p);

  const x = new Array(p).fill(0);
  const z = new Array(p).fill(0);
  const u = new Array(p).fill(0);

  const objectives: number[] = [];
  const primalResiduals: number[] = [];
  const dualResiduals: number[] = [];

  for (let k = 0; k < maxIter; k++) {
    // x-update: x = (AᵀA + ρI)⁻¹(Aᵀb + ρ(z - u))
    const rhs = new Array(p);
    for (let j = 0; j < p; j++) {
      rhs[j] = Atb[j] + rho * (z[j] - u[j]);
    }
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let l = 0; l < p; l++) sum += Minv[j][l] * rhs[l];
      x[j] = sum;
    }

    // z-update: z = S_{λ/ρ}(x + u)
    const zOld = z.slice();
    const xu = x.map((xi, i) => xi + u[i]);
    const zNew = softThreshold(xu, lambda / rho);
    for (let j = 0; j < p; j++) z[j] = zNew[j];

    // u-update: u = u + x - z
    for (let j = 0; j < p; j++) u[j] += x[j] - z[j];

    objectives.push(lassoObjective(A, z, b, lambda));
    primalResiduals.push(
      Math.sqrt(x.reduce((s, xi, i) => s + (xi - z[i]) ** 2, 0)),
    );
    dualResiduals.push(
      rho * Math.sqrt(z.reduce((s, zi, i) => s + (zi - zOld[i]) ** 2, 0)),
    );
  }

  return { z, objectives, primalResiduals, dualResiduals };
}

// ── ISTA / FISTA solvers ───────────────────────────────────────────────

export interface ISTAResult {
  x: number[];
  objectives: number[];
}

/**
 * Run ISTA (proximal gradient) on a Lasso problem.
 */
export function solveISTA(
  A: number[][],
  b: number[],
  lambda: number,
  L: number,
  maxIter: number,
): ISTAResult {
  const p = A[0].length;
  const eta = 1 / L;
  const x = new Array(p).fill(0);
  const objectives: number[] = [lassoObjective(A, x, b, lambda)];

  for (let k = 0; k < maxIter; k++) {
    const grad = lassoGradient(A, x, b);
    const xHalf = x.map((xi, i) => xi - eta * grad[i]);
    const xNew = softThreshold(xHalf, eta * lambda);
    for (let i = 0; i < p; i++) x[i] = xNew[i];
    objectives.push(lassoObjective(A, x, b, lambda));
  }

  return { x, objectives };
}

/**
 * Run FISTA (accelerated proximal gradient) on a Lasso problem.
 */
export function solveFISTA(
  A: number[][],
  b: number[],
  lambda: number,
  L: number,
  maxIter: number,
): ISTAResult {
  const p = A[0].length;
  const eta = 1 / L;
  let x = new Array(p).fill(0);
  let xPrev = x.slice();
  let tk = 1;
  const objectives: number[] = [lassoObjective(A, x, b, lambda)];

  for (let k = 0; k < maxIter; k++) {
    const tNext = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2;
    const momentum = (tk - 1) / tNext;
    const y = x.map((xi, i) => xi + momentum * (xi - xPrev[i]));

    const grad = lassoGradient(A, y, b);
    const yHalf = y.map((yi, i) => yi - eta * grad[i]);
    xPrev = x.slice();
    x = softThreshold(yHalf, eta * lambda);
    tk = tNext;

    objectives.push(lassoObjective(A, x, b, lambda));
  }

  return { x, objectives };
}

// ── Internal helpers ───────────────────────────────────────────────────

/** Simple seeded PRNG (xoshiro128**) for reproducible visualizations. */
function seededRng(seed: number) {
  let s0 = seed | 0 || 1;
  let s1 = (seed * 1664525 + 1013904223) | 0;
  let s2 = (s1 * 1664525 + 1013904223) | 0;
  let s3 = (s2 * 1664525 + 1013904223) | 0;

  function next(): number {
    const result = (s1 * 5) | 0;
    const t = (s1 << 9) | 0;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = (s3 << 11) | (s3 >>> 21);
    return (result >>> 0) / 4294967296;
  }

  return {
    rand: () => next(),
    randn: () => {
      // Box-Muller transform
      const u1 = next() || 1e-10;
      const u2 = next();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    },
  };
}

/** Estimate largest eigenvalue of AᵀA via power iteration. */
function estimateLipschitz(A: number[][], n: number, p: number): number {
  let v = new Array(p).fill(0).map((_, i) => (i === 0 ? 1 : 0));
  for (let iter = 0; iter < 30; iter++) {
    // w = AᵀA v
    const Av = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) Av[i] += A[i][j] * v[j];
    }
    const w = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      for (let i = 0; i < n; i++) w[j] += A[i][j] * Av[i];
    }
    const norm = Math.sqrt(w.reduce((s, wi) => s + wi * wi, 0));
    if (norm < 1e-15) break;
    v = w.map((wi) => wi / norm);
  }
  // Rayleigh quotient
  const Av = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) Av[i] += A[i][j] * v[j];
  }
  return Av.reduce((s, ai) => s + ai * ai, 0);
}

/** Compute AᵀA for a matrix A. */
function computeAtA(A: number[][], n: number, p: number): number[][] {
  const result: number[][] = [];
  for (let j = 0; j < p; j++) {
    const row = new Array(p).fill(0);
    for (let l = 0; l < p; l++) {
      for (let i = 0; i < n; i++) {
        row[l] += A[i][j] * A[i][l];
      }
    }
    result.push(row);
  }
  return result;
}

/** Compute Aᵀb. */
function computeAtb(
  A: number[][],
  b: number[],
  n: number,
  p: number,
): number[] {
  const result = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) {
      result[j] += A[i][j] * b[i];
    }
  }
  return result;
}

/** Invert a p×p matrix via Gauss-Jordan (for small p in viz). */
function invertMatrix(M: number[][], p: number): number[][] {
  // Augmented matrix [M | I]
  const aug: number[][] = [];
  for (let i = 0; i < p; i++) {
    const row = M[i].slice();
    for (let j = 0; j < p; j++) row.push(i === j ? 1 : 0);
    aug.push(row);
  }

  for (let col = 0; col < p; col++) {
    // Partial pivoting
    let maxRow = col;
    for (let row = col + 1; row < p; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-15) continue;
    for (let j = 0; j < 2 * p; j++) aug[col][j] /= pivot;

    for (let row = 0; row < p; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * p; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  // Extract inverse
  return aug.map((row) => row.slice(p));
}

/**
 * Shared math utilities for gradient descent visualizations.
 * All exports are pure functions — no module-level state.
 */

// ── Types ──────────────────────────────────────────────────────────────

export interface QuadraticProblem {
  A: number[][];
  L: number;
  mu: number;
  kappa: number;
  f: (x: number[]) => number;
  grad: (x: number[]) => number[];
}

export interface GDTrajectory {
  path: number[][];
  objectives: number[];
}

// ── Quadratic construction ─────────────────────────────────────────────

/**
 * Construct a 2×2 quadratic f(x) = ½ xᵀAx with given condition number κ,
 * rotated by angle theta (default 30° for visual interest).
 *
 * A = Rᵀ · diag(1, κ) · R where R is a rotation matrix.
 * Eigenvalues: μ = 1 (smallest), L = κ (largest).
 */
export function makeQuadratic(kappa: number, theta: number = Math.PI / 6): QuadraticProblem {
  const c = Math.cos(theta);
  const s = Math.sin(theta);

  // A = Rᵀ diag(1, κ) R
  const a00 = c * c + kappa * s * s;
  const a01 = (1 - kappa) * c * s;
  const a11 = s * s + kappa * c * c;
  const A = [
    [a00, a01],
    [a01, a11],
  ];

  const mu = 1;
  const L = kappa;

  const f = (x: number[]): number =>
    0.5 * (A[0][0] * x[0] * x[0] + 2 * A[0][1] * x[0] * x[1] + A[1][1] * x[1] * x[1]);

  const grad = (x: number[]): number[] => [
    A[0][0] * x[0] + A[0][1] * x[1],
    A[0][1] * x[0] + A[1][1] * x[1],
  ];

  return { A, L, mu, kappa, f, grad };
}

// ── Gradient descent trajectory ────────────────────────────────────────

/** Run gradient descent on a QuadraticProblem, returning the full trajectory. */
export function runGDTrajectory(
  problem: QuadraticProblem,
  x0: number[],
  eta: number,
  maxIter: number,
): GDTrajectory {
  const path: number[][] = [x0.slice()];
  const objectives: number[] = [problem.f(x0)];
  let x = x0.slice();

  for (let k = 0; k < maxIter; k++) {
    const g = problem.grad(x);
    x = [x[0] - eta * g[0], x[1] - eta * g[1]];
    path.push(x.slice());
    objectives.push(problem.f(x));

    // Divergence guard
    if (Math.abs(x[0]) > 200 || Math.abs(x[1]) > 200) break;
    // Convergence guard
    if (g[0] * g[0] + g[1] * g[1] < 1e-20) break;
  }

  return { path, objectives };
}

// ── Armijo backtracking line search ────────────────────────────────────

/**
 * Armijo backtracking: find step size α such that
 *   f(x + α·d) ≤ f(x) + c·α·gᵀd
 * where d is the search direction (typically -g).
 */
export function armijoBacktracking(
  f: (x: number[]) => number,
  grad: number[],
  x: number[],
  d: number[],
  c: number = 0.1,
  beta: number = 0.5,
  alpha0: number = 1.0,
): number {
  const fx = f(x);
  const gTd = grad[0] * d[0] + grad[1] * d[1];
  let alpha = alpha0;

  for (let i = 0; i < 50; i++) {
    const xNew = [x[0] + alpha * d[0], x[1] + alpha * d[1]];
    if (f(xNew) <= fx + c * alpha * gTd) return alpha;
    alpha *= beta;
  }

  return alpha;
}

// ── Nesterov accelerated gradient step ─────────────────────────────────

/**
 * One step of Nesterov's accelerated gradient method.
 * Returns the new iterate x_{k+1}, extrapolation point y_{k+1}, and sequence value t_{k+1}.
 */
export function nesterovStep(
  gradF: (x: number[]) => number[],
  x: number[],
  xPrev: number[],
  t: number,
  eta: number,
): { xNew: number[]; yNew: number[]; tNew: number } {
  const tNew = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
  const momentum = (t - 1) / tNew;

  // Extrapolation: y = x + momentum * (x - x_prev)
  const y = [x[0] + momentum * (x[0] - xPrev[0]), x[1] + momentum * (x[1] - xPrev[1])];

  // Gradient step from the extrapolated point
  const g = gradF(y);
  const xNew = [y[0] - eta * g[0], y[1] - eta * g[1]];

  return { xNew, yNew: y, tNew };
}

// ── Simplex projection (sort-and-threshold) ────────────────────────────

/**
 * Project v onto the probability simplex {x ≥ 0, Σx = 1}
 * using the sort-and-threshold algorithm.
 */
export function projectSimplex(v: number[]): number[] {
  const n = v.length;
  const sorted = v.slice().sort((a, b) => b - a); // descending
  let cumSum = 0;
  let rho = 0;

  for (let j = 0; j < n; j++) {
    cumSum += sorted[j];
    if (sorted[j] + (1 - cumSum) / (j + 1) > 0) {
      rho = j;
    }
  }

  let cumSumRho = 0;
  for (let j = 0; j <= rho; j++) cumSumRho += sorted[j];
  const tau = (cumSumRho - 1) / (rho + 1);

  return v.map((vi) => Math.max(vi - tau, 1e-10));
}

// ── Exponentiated gradient (mirror descent with neg-entropy) ───────────

/**
 * One step of the exponentiated gradient algorithm:
 *   x_{k+1,i} ∝ x_{k,i} · exp(−η · g_i), then normalize.
 * This is mirror descent with the negative entropy mirror map on the simplex.
 */
export function expGradientStep(x: number[], grad: number[], eta: number): number[] {
  const unnormalized = x.map((xi, i) => xi * Math.exp(-eta * grad[i]));
  const sum = unnormalized.reduce((s, v) => s + v, 0);
  return unnormalized.map((v) => Math.max(v / sum, 1e-10));
}

// ── Vector utilities ───────────────────────────────────────────────────

export function dot(a: number[], b: number[]): number {
  return a.reduce((s, ai, i) => s + ai * b[i], 0);
}

export function norm2(v: number[]): number {
  return Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0));
}

export function matVec(A: number[][], x: number[]): number[] {
  return A.map((row) => row.reduce((s, aij, j) => s + aij * x[j], 0));
}

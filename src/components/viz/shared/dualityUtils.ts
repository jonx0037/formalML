/**
 * Shared math utilities for Lagrangian duality visualizations.
 * All exports are pure functions — no module-level state.
 */

import { norm2 } from './gradientDescentUtils';

// Re-export vector utilities used by multiple duality viz components
export { dot, norm2, matVec } from './gradientDescentUtils';

// ── Types ──────────────────────────────────────────────────────────────

export interface KKTStatus {
  stationarity: boolean;
  primalFeasibility: boolean;
  dualFeasibility: boolean;
  complementarySlackness: boolean;
  allSatisfied: boolean;
  /** Numerical residual for stationarity ‖∇L‖ */
  stationarityResidual: number;
  /** Per-constraint values: fᵢ(x) */
  constraintValues: number[];
  /** Dual variables */
  lambdas: number[];
}

export interface SVMSolution {
  w: number[];
  b: number;
  alpha: number[];
  svIndices: number[];
  margin: number;
}

// ── Quadratic evaluation ─────────────────────────────────────────────

/**
 * Evaluate f(x) = 0.5 * xᵀAx + qᵀx for 2D.
 */
export function evaluateQuadratic2D(
  A: number[][],
  q: number[],
  x: number[],
): number {
  return (
    0.5 * (A[0][0] * x[0] * x[0] + 2 * A[0][1] * x[0] * x[1] + A[1][1] * x[1] * x[1]) +
    q[0] * x[0] +
    q[1] * x[1]
  );
}

/**
 * Gradient ∇f(x) = Ax + q for 2D.
 */
export function gradientQuadratic2D(
  A: number[][],
  q: number[],
  x: number[],
): number[] {
  return [
    A[0][0] * x[0] + A[0][1] * x[1] + q[0],
    A[0][1] * x[0] + A[1][1] * x[1] + q[1],
  ];
}

// ── Lagrangian evaluation ────────────────────────────────────────────

/**
 * Evaluate the Lagrangian L(x, λ, ν) = f₀(x) + Σ λᵢ fᵢ(x) + Σ νⱼ hⱼ(x)
 */
export function evaluateLagrangian(
  f0Val: number,
  fiVals: number[],
  hjVals: number[],
  lambda: number[],
  nu: number[],
): number {
  let L = f0Val;
  for (let i = 0; i < lambda.length; i++) {
    L += lambda[i] * fiVals[i];
  }
  for (let j = 0; j < nu.length; j++) {
    L += nu[j] * hjVals[j];
  }
  return L;
}

// ── KKT condition checker ────────────────────────────────────────────

/**
 * Check KKT conditions at a given point.
 *
 * @param gradF0 — gradient of the objective at x
 * @param gradFi — gradients of inequality constraints at x (array of gradients)
 * @param lambdas — dual variables for inequality constraints
 * @param fi — constraint values fᵢ(x)
 * @param tol — tolerance for numerical checks
 */
export function checkKKT(
  gradF0: number[],
  gradFi: number[][],
  lambdas: number[],
  fi: number[],
  tol: number = 0.15,
): KKTStatus {
  const n = gradF0.length;

  // Stationarity: ∇f₀ + Σ λᵢ ∇fᵢ = 0
  const gradL = gradF0.slice();
  for (let i = 0; i < lambdas.length; i++) {
    for (let d = 0; d < n; d++) {
      gradL[d] += lambdas[i] * gradFi[i][d];
    }
  }
  const stationarityResidual = norm2(gradL);
  const stationarity = stationarityResidual < tol;

  // Primal feasibility: fᵢ(x) ≤ 0
  const primalFeasibility = fi.every((v) => v <= tol);

  // Dual feasibility: λᵢ ≥ 0
  const dualFeasibility = lambdas.every((l) => l >= -tol);

  // Complementary slackness: λᵢ fᵢ(x) = 0
  const complementarySlackness = lambdas.every(
    (l, i) => Math.abs(l * fi[i]) < tol,
  );

  return {
    stationarity,
    primalFeasibility,
    dualFeasibility,
    complementarySlackness,
    allSatisfied: stationarity && primalFeasibility && dualFeasibility && complementarySlackness,
    stationarityResidual,
    constraintValues: fi,
    lambdas,
  };
}

// ── Simple SVM dual solver ───────────────────────────────────────────

/**
 * Solve the hard-margin SVM dual via projected gradient ascent.
 *
 * Dual: max Σαᵢ − ½ α'Qα  subject to α ≥ 0, y'α = 0
 * where Q_{ij} = yᵢyⱼ(xᵢ·xⱼ)
 *
 * Uses projected gradient ascent with equality constraint enforcement
 * via alternating projection. For n ≤ 30, converges in ~300 iterations.
 */
export function solveSimpleSVM(X: number[][], y: number[]): SVMSolution {
  const n = X.length;
  const d = X[0].length;

  // Build Q matrix: Q_{ij} = y_i y_j x_i^T x_j
  const Q: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let innerProduct = 0;
      for (let k = 0; k < d; k++) {
        innerProduct += X[i][k] * X[j][k];
      }
      Q[i][j] = y[i] * y[j] * innerProduct;
      Q[j][i] = Q[i][j];
    }
  }

  // Initialize alpha
  const alpha = new Array(n).fill(0);
  const lr = 0.002;
  const maxIter = 500;

  for (let iter = 0; iter < maxIter; iter++) {
    // Gradient of dual objective: ∂/∂α = 1 − Qα
    const Qa = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Qa[i] += Q[i][j] * alpha[j];
      }
    }

    // Gradient ascent step
    for (let i = 0; i < n; i++) {
      alpha[i] += lr * (1 - Qa[i]);
    }

    // Project: α ≥ 0
    for (let i = 0; i < n; i++) {
      alpha[i] = Math.max(alpha[i], 0);
    }

    // Enforce y'α = 0 via correction
    let yAlpha = 0;
    let yNormSq = 0;
    for (let i = 0; i < n; i++) {
      yAlpha += y[i] * alpha[i];
      yNormSq += y[i] * y[i];
    }
    const correction = yAlpha / yNormSq;
    for (let i = 0; i < n; i++) {
      alpha[i] -= correction * y[i];
      alpha[i] = Math.max(alpha[i], 0);
    }
  }

  // Recover w* = Σ αᵢ yᵢ xᵢ
  const w = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < d; k++) {
      w[k] += alpha[i] * y[i] * X[i][k];
    }
  }

  // Find support vectors (α > threshold)
  const svThreshold = 1e-4;
  const svIndices: number[] = [];
  for (let i = 0; i < n; i++) {
    if (alpha[i] > svThreshold) svIndices.push(i);
  }

  // Recover b* from support vectors
  let b = 0;
  if (svIndices.length > 0) {
    let bSum = 0;
    for (const i of svIndices) {
      let wxI = 0;
      for (let k = 0; k < d; k++) {
        wxI += w[k] * X[i][k];
      }
      bSum += y[i] - wxI;
    }
    b = bSum / svIndices.length;
  }

  const wNorm = Math.sqrt(w.reduce((s, wi) => s + wi * wi, 0));
  const margin = wNorm > 1e-10 ? 2 / wNorm : Infinity;

  return { w, b, alpha, svIndices, margin };
}

// ── Seeded PRNG ──────────────────────────────────────────────────────

/**
 * Simple seeded PRNG (mulberry32) for deterministic data generation.
 */
export function seededRandom(seed: number): () => number {
  let t = seed;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let v = t;
    v = Math.imul(v ^ (v >>> 15), v | 1);
    v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate a linearly separable 2D dataset for SVM visualization.
 */
export function generateSVMDataset(
  nPerClass: number = 10,
  seed: number = 42,
): { X: number[][]; y: number[] } {
  const rng = seededRandom(seed);
  const X: number[][] = [];
  const y: number[] = [];

  // Class +1: centered around (1.5, 1.5)
  for (let i = 0; i < nPerClass; i++) {
    X.push([1.0 + rng() * 1.5, 0.8 + rng() * 1.5]);
    y.push(1);
  }

  // Class -1: centered around (-1.0, -1.0)
  for (let i = 0; i < nPerClass; i++) {
    X.push([-0.5 - rng() * 1.5, -0.7 - rng() * 1.5]);
    y.push(-1);
  }

  return { X, y };
}

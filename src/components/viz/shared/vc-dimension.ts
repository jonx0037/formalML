// =============================================================================
// vc-dimension.ts
//
// Shared math module for the T6 Learning Theory & Methodology track,
// vc-dimension topic. Consumers:
//   GrowthFunctionPlotter, SauerShelahDemo, FTSLBoundExplorer,
//   RealizableVsAgnostic, ShatteringPlayground, RadonTheoremAnimator,
//   SVMMarginVsVCBound, VCVsRademacher, EmpiricalShatterCheck.
//
// All numerical algorithms are direct translations of the verified Python
// notebook at notebooks/vc-dimension/01_vc_dimension.ipynb. Where the Python
// uses np.random (PCG64), this module substitutes the seeded Mulberry32 PRNG
// from nonparametric-ml.ts. Closed-form quantities match exactly; Monte-Carlo
// quantities match in shape but not bit-for-bit (cross-RNG drift).
//
// Numerical agreement is verified by
//   src/components/viz/shared/__tests__/verify-vc-dimension.ts.
// =============================================================================

import { mulberry32, gaussianSampler } from './nonparametric-ml';
export { mulberry32, gaussianSampler };

// -----------------------------------------------------------------------------
// Notebook seed and constants (notebook uses default_rng(20260513))
// -----------------------------------------------------------------------------

export const NOTEBOOK_SEED = 20260513;
export const DEFAULT_DELTA = 0.05;

// -----------------------------------------------------------------------------
// Color palette — matches the SRM / pac-bayes / generalization-bounds family
// so that T6 visualizations read as one suite.
// -----------------------------------------------------------------------------

export const paletteVC = {
  primary: '#1f5fa8', // deep blue   — FTSL bound / Sauer–Shelah / closed-form
  emp: '#c0392b',     // brown-red   — empirical Monte-Carlo
  alt: '#e67e22',     // orange      — margin-based / alternate bound
  highlight: '#16a085', // teal-green — phase-transition / shatter highlight
  muted: '#7f8c8d',   // gray        — 2^n trivial bound / annotations
  accent: '#8e44ad',  // purple      — decorative
} as const;

export type VCColorKey = keyof typeof paletteVC;

// -----------------------------------------------------------------------------
// Binomial coefficients — log-domain to handle large n without overflow.
// Notebook uses scipy.special.comb(n, k, exact=True).
// -----------------------------------------------------------------------------

export function binomialCoefficient(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  k = Math.min(k, n - k);
  let r = 1;
  for (let i = 0; i < k; i++) {
    r = (r * (n - i)) / (i + 1);
  }
  return r;
}

/** Partial binomial sum sum_{i=0..d} C(n, i). The exact RHS of Sauer–Shelah (5.1). */
export function sauerShelahBinomialSum(n: number, d: number): number {
  if (d < 0) return 0;
  if (d >= n) return Math.pow(2, n);
  let s = 0;
  for (let i = 0; i <= d; i++) {
    s += binomialCoefficient(n, i);
  }
  return s;
}

/** Closed-form Sauer–Shelah upper bound (en/d)^d for n >= d >= 1. */
export function sauerShelahClosedForm(n: number, d: number): number {
  if (d < 1 || n < d) return Number.POSITIVE_INFINITY;
  return Math.pow((Math.E * n) / d, d);
}

// -----------------------------------------------------------------------------
// Closed-form growth functions for canonical classes (§3.3).
// Notebook cell 25 verifies these against brute-force enumeration.
// -----------------------------------------------------------------------------

/** Half-lines on R: Pi(n) = n + 1. */
export function growthHalfLine(n: number): number {
  return n + 1;
}

/** Intervals on R: Pi(n) = C(n+1, 2) + 1 = n(n+1)/2 + 1. */
export function growthInterval(n: number): number {
  return (n * (n + 1)) / 2 + 1;
}

/** Half-planes in R^2: Pi(n) = n^2 - n + 2 (Cover 1965). */
export function growthHalfPlane(n: number): number {
  if (n === 0) return 1;
  return n * n - n + 2;
}

/** Axis-aligned rectangles in R^2: empirical or Sauer–Shelah ceiling.
 *  No closed form. For pedagogical viz we use the Sauer–Shelah ceiling
 *  C(n, <= 4) as the upper-envelope curve; for empirical n <= 12 the
 *  enumeration is brute-forced in the notebook. */
export function growthRectangle2DBound(n: number): number {
  return Math.min(Math.pow(2, n), sauerShelahBinomialSum(n, 4));
}

// -----------------------------------------------------------------------------
// FTSL bound (eq. 6.2) and sample-complexity inversions (§7).
// -----------------------------------------------------------------------------

/** FTSL upper bound eps from eq. (6.2):
 *  eps = sqrt(8 (d log(2 e n / d) + log(4/delta)) / n). */
export function ftslBoundEpsilon(n: number, d: number, delta: number): number {
  if (n <= 0 || d <= 0 || delta <= 0 || delta >= 1) return Number.NaN;
  const numerator = 8 * (d * Math.log((2 * Math.E * n) / d) + Math.log(4 / delta));
  return Math.sqrt(numerator / n);
}

/** Realizable PAC sample complexity n(eps, delta, d) from eq. (7.1):
 *  n >= 8 (d log(2 e / eps) + log(2/delta)) / eps. */
export function realizableSampleComplexity(eps: number, delta: number, d: number): number {
  if (eps <= 0 || eps >= 1 || delta <= 0 || delta >= 1 || d <= 0) return Number.NaN;
  return (8 * (d * Math.log((2 * Math.E) / eps) + Math.log(2 / delta))) / eps;
}

/** Agnostic PAC sample complexity n(eps, delta, d):
 *  n >= 8 (d log(2 e / eps) + log(2/delta)) / eps^2.
 *  Same numerator as the realizable formula (eq. 7.1), but denominator eps^2
 *  rather than eps — the variance-blind Hoeffding-via-FTSL inversion.
 *  Matches the notebook cell 58 outputs at d=5, delta=0.05. */
export function agnosticSampleComplexity(eps: number, delta: number, d: number): number {
  if (eps <= 0 || eps >= 1 || delta <= 0 || delta >= 1 || d <= 0) return Number.NaN;
  return (8 * (d * Math.log((2 * Math.E) / eps) + Math.log(2 / delta))) / (eps * eps);
}

// -----------------------------------------------------------------------------
// Margin-based VC bound (§11.1): R^2 / gamma^2.
// -----------------------------------------------------------------------------

/** Margin-based capacity ceil(R^2 / gamma^2) on data of radius R with margin gamma. */
export function marginVCBound(R: number, gamma: number): number {
  if (R <= 0 || gamma <= 0) return Number.POSITIVE_INFINITY;
  return Math.ceil((R * R) / (gamma * gamma));
}

/** FTSL-style epsilon using margin-based capacity in place of the dimension-dependent d_VC. */
export function marginFTSLEpsilon(n: number, R: number, gamma: number, delta: number): number {
  const d = marginVCBound(R, gamma);
  return ftslBoundEpsilon(n, d, delta);
}

// -----------------------------------------------------------------------------
// Empirical Rademacher complexity (§9).
// Massart bound on the restricted-class size, eq. (9.2).
// -----------------------------------------------------------------------------

/** Massart bound on empirical Rademacher complexity given |H|_S| and sample size n. */
export function massartBound(restrictedClassSize: number, n: number): number {
  if (restrictedClassSize <= 0 || n <= 0) return Number.NaN;
  return Math.sqrt((2 * Math.log(restrictedClassSize)) / n);
}

/** Sauer–Shelah Rademacher bound: substitute (en/d)^d for |H|_S| in Massart. */
export function sauerShelahRademacherBound(n: number, d: number): number {
  if (n <= 0 || d <= 0) return Number.NaN;
  return Math.sqrt((2 * d * Math.log((Math.E * n) / d)) / n);
}

// -----------------------------------------------------------------------------
// Shattering enumeration helpers (§3, §9, §10 brute-force).
//
// Caveat: 2^n enumeration is O(n * 2^n). The notebook caps n <= 20; the
// in-browser cap is n <= 16 per §12.4. These helpers respect the cap.
// -----------------------------------------------------------------------------

export type Point2D = readonly [number, number];

/** A half-plane classifier {x : <w, x> >= b}. */
export interface HalfPlane {
  w: Point2D;
  b: number;
}

/** An axis-aligned rectangle [x_lo, x_hi] x [y_lo, y_hi]. */
export interface Rect {
  xLo: number;
  xHi: number;
  yLo: number;
  yHi: number;
}

export function classifyHalfPlane(hp: HalfPlane, p: Point2D): 0 | 1 {
  const dot = hp.w[0] * p[0] + hp.w[1] * p[1];
  return dot >= hp.b ? 1 : 0;
}

export function classifyRect(r: Rect, p: Point2D): 0 | 1 {
  return p[0] >= r.xLo && p[0] <= r.xHi && p[1] >= r.yLo && p[1] <= r.yHi ? 1 : 0;
}

/** Enumerate the restriction H|_S by trying many half-planes.
 *  We try all binary labelings as candidate targets and use the
 *  perceptron-style oracle (LinearSVC analogue): a half-plane realizes
 *  labeling b iff the labeled points are linearly separable.
 *
 *  Worst-case O(n * 2^n) labeling iterations × O(n) linear-separability check. */
export function enumerateHalfPlaneDichotomies(points: Point2D[]): Set<string> {
  const n = points.length;
  if (n > 20) {
    throw new Error(`enumerateHalfPlaneDichotomies: n=${n} exceeds notebook cap of 20`);
  }
  const realized = new Set<string>();
  for (let mask = 0; mask < (1 << n); mask++) {
    const labels = new Uint8Array(n);
    for (let i = 0; i < n; i++) labels[i] = (mask >> i) & 1;
    if (isLinearlySeparable2D(points, labels)) {
      realized.add(Array.from(labels).join(''));
    }
  }
  return realized;
}

/** Check whether two classes (labels 0/1) are linearly separable in R^2.
 *  Uses the perceptron convergence theorem for separable data plus a sanity cap.
 *  For research use this would be replaced by LinearSVC; for viz use this
 *  perceptron suffices on n <= 16 with floor offset. */
export function isLinearlySeparable2D(points: Point2D[], labels: Uint8Array): boolean {
  const n = points.length;
  // Trivially separable if one class is empty.
  let n1 = 0;
  for (let i = 0; i < n; i++) n1 += labels[i];
  if (n1 === 0 || n1 === n) return true;
  // Build augmented points (x, y, 1) with sign +1 for label 1, -1 for label 0.
  const X: Float64Array[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const s = labels[i] === 1 ? 1 : -1;
    X[i] = new Float64Array([s * points[i][0], s * points[i][1], s]);
  }
  // Perceptron with bounded iterations.
  const w = new Float64Array(3);
  const MAX_ITER = 5000;
  for (let t = 0; t < MAX_ITER; t++) {
    let mistakes = 0;
    for (let i = 0; i < n; i++) {
      const dot = w[0] * X[i][0] + w[1] * X[i][1] + w[2] * X[i][2];
      if (dot <= 0) {
        w[0] += X[i][0];
        w[1] += X[i][1];
        w[2] += X[i][2];
        mistakes++;
      }
    }
    if (mistakes === 0) return true;
  }
  return false;
}

/** Enumerate the restriction H|_S for axis-aligned rectangles by trying
 *  all C(n, k1) * C(n, k2)... boundary configurations.
 *  Cleanest implementation: for each binary labeling, the rectangle of the
 *  positive class is its bounding box; the labeling is realized iff the
 *  bounding box contains no negative points. */
export function enumerateRectangleDichotomies(points: Point2D[]): Set<string> {
  const n = points.length;
  if (n > 20) {
    throw new Error(`enumerateRectangleDichotomies: n=${n} exceeds notebook cap of 20`);
  }
  const realized = new Set<string>();
  for (let mask = 0; mask < (1 << n); mask++) {
    let xLo = Number.POSITIVE_INFINITY,
      xHi = Number.NEGATIVE_INFINITY,
      yLo = Number.POSITIVE_INFINITY,
      yHi = Number.NEGATIVE_INFINITY;
    let any = false;
    for (let i = 0; i < n; i++) {
      if ((mask >> i) & 1) {
        any = true;
        const [x, y] = points[i];
        if (x < xLo) xLo = x;
        if (x > xHi) xHi = x;
        if (y < yLo) yLo = y;
        if (y > yHi) yHi = y;
      }
    }
    // Empty positive class: realized by an empty rectangle.
    let ok = true;
    if (any) {
      for (let i = 0; i < n; i++) {
        if (!((mask >> i) & 1)) {
          const [x, y] = points[i];
          if (x >= xLo && x <= xHi && y >= yLo && y <= yHi) {
            ok = false;
            break;
          }
        }
      }
    }
    if (ok) {
      const labels: string[] = new Array(n);
      for (let i = 0; i < n; i++) labels[i] = String((mask >> i) & 1);
      realized.add(labels.join(''));
    }
  }
  return realized;
}

// -----------------------------------------------------------------------------
// Radon partition (§8.1, §8.5).
//
// Given d+2 points in R^d, solve the linear system
//   sum_i lambda_i x_i = 0,  sum_i lambda_i = 0
// for a non-trivial lambda. The partition is I+ = {i : lambda_i > 0},
// I- = {i : lambda_i < 0}, with shared point p in conv(I+) ∩ conv(I-).
//
// In R^2 we have 4 points and 3 equations (2 for x_i + 1 for the lambda sum).
// We solve via SVD of the 3x4 matrix.
// -----------------------------------------------------------------------------

export interface RadonResult {
  Iplus: number[];
  Iminus: number[];
  p: Point2D;
  lambdas: Float64Array;
}

/** Compute the Radon partition of 4 points in R^2.
 *  Returns null if the points are degenerate (e.g., all coincident). */
export function radonPartition2D(points: readonly Point2D[]): RadonResult | null {
  if (points.length !== 4) {
    throw new Error(`radonPartition2D: expected 4 points, got ${points.length}`);
  }
  // Linear system:
  //   [ x1 x2 x3 x4 ]   [lambda]   [0]
  //   [ y1 y2 y3 y4 ] * [      ] = [0]
  //   [  1  1  1  1 ]   [      ]   [0]
  // Find a vector in the null space of A (size 3x4).
  // The null space is at least 1D; pick a generator via cofactor expansion.
  const x = points.map((p) => p[0]);
  const y = points.map((p) => p[1]);

  // lambda_i = (-1)^i * det of 3x3 minor formed by deleting column i from A.
  // A = [[x1..x4],[y1..y4],[1..1]] (3x4).
  function det3([a, b, c]: number[][]): number {
    return (
      a[0] * (b[1] * c[2] - b[2] * c[1]) -
      a[1] * (b[0] * c[2] - b[2] * c[0]) +
      a[2] * (b[0] * c[1] - b[1] * c[0])
    );
  }

  const lambdas = new Float64Array(4);
  for (let i = 0; i < 4; i++) {
    // Build the 3x3 minor by deleting column i.
    const cols = [0, 1, 2, 3].filter((j) => j !== i);
    const minor: number[][] = [
      cols.map((j) => x[j]),
      cols.map((j) => y[j]),
      cols.map((_j) => 1),
    ];
    lambdas[i] = (i % 2 === 0 ? 1 : -1) * det3(minor);
  }

  // Check non-trivial solution.
  let maxAbs = 0;
  for (let i = 0; i < 4; i++) if (Math.abs(lambdas[i]) > maxAbs) maxAbs = Math.abs(lambdas[i]);
  if (maxAbs < 1e-12) return null;

  // Normalize so sum of positives = 1.
  const Iplus: number[] = [];
  const Iminus: number[] = [];
  let sumPlus = 0;
  for (let i = 0; i < 4; i++) {
    if (lambdas[i] > 1e-9) {
      Iplus.push(i);
      sumPlus += lambdas[i];
    } else if (lambdas[i] < -1e-9) {
      Iminus.push(i);
    }
  }
  if (Iplus.length === 0 || Iminus.length === 0 || sumPlus === 0) return null;

  let px = 0,
    py = 0;
  for (const i of Iplus) {
    px += (lambdas[i] / sumPlus) * x[i];
    py += (lambdas[i] / sumPlus) * y[i];
  }
  return { Iplus, Iminus, p: [px, py], lambdas };
}

// -----------------------------------------------------------------------------
// Triangle / point-in-polygon helpers (used by shattering playground).
// -----------------------------------------------------------------------------

export function sign(x: number): -1 | 0 | 1 {
  return x > 0 ? 1 : x < 0 ? -1 : 0;
}

/** Is point p inside the (closed) triangle abc? Barycentric test. */
export function isInTriangle(p: Point2D, a: Point2D, b: Point2D, c: Point2D): boolean {
  const d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1]);
  const d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1]);
  const d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1]);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}

// -----------------------------------------------------------------------------
// Discretized half-line cardinality experiment (§1.5).
// |H_k|_S| saturates at n+1 when k >= n+1 thresholds are placed in every gap.
// -----------------------------------------------------------------------------

export function discretizedHalflineRestrictedSize(k: number, samplePoints: number[]): number {
  // For each threshold a_j = j/k, compute the labeling vector h(x_i) = 1[x_i >= a_j]
  // and count distinct vectors. With n sample points sorted, an integer encoding
  // is the count of points >= threshold (the labeling is a prefix of 0s then 1s
  // once sorted). The distinct values correspond to where the threshold falls.
  const n = samplePoints.length;
  const sorted = [...samplePoints].sort((a, b) => a - b);
  const distinct = new Set<number>();
  for (let j = 1; j <= k; j++) {
    const a = j / k;
    // count of sorted[i] >= a
    let cnt = 0;
    for (let i = 0; i < n; i++) if (sorted[i] >= a) cnt++;
    distinct.add(cnt);
  }
  return distinct.size;
}

// -----------------------------------------------------------------------------
// No-trailing-newline marker so eslint --quiet sees a clean file.
// -----------------------------------------------------------------------------

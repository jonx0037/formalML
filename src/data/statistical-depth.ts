// =============================================================================
// statistical-depth.ts
//
// Depth functions, sample generators, and the Iris feature subset used by the
// Statistical Depth topic in the T4 Nonparametric & Distribution-Free track.
//
// All algorithms are direct translations of the verified Python notebook at
// notebooks/statistical-depth/01_statistical_depth.ipynb. Where the Python uses
// NumPy's PCG64 (np.random.default_rng), this module substitutes a seeded
// Mulberry32 PRNG from the shared nonparametric-ml module — so deterministic
// claims (unit-square corners, exact-arithmetic depths) match the notebook
// exactly while stochastic claims (depth at origin of a random Gaussian
// sample) match within an explicit tolerance documented in
// src/components/viz/shared/__tests__/verify-statistical-depth.ts.
//
// Three implementations referenced in the brief stay inline at their viz
// component sites rather than living here, per the topic-specific notes:
//
//   - simplicial_depth_2d   ($n \le 100$ in DepthFunctionComparison)
//   - exact_halfspace_depth_3d  ($d \le 3$, $n \le 200$ in RuntimeScalingExplorer)
//   - modified_band_depth   ($n \le 50$ in FunctionalDepthVisualizer)
//
// Each is an enumeration ($\binom{n}{3}$, $\binom{n}{2}$, $\binom{n}{2}$ T-grid
// integrand respectively); promoting them to efficient incremental algorithms
// is well beyond the scope of this topic shipment.
// =============================================================================

import { mulberry32, gaussianSampler } from '../components/viz/shared/nonparametric-ml';

export type Point2D = [number, number];

// -----------------------------------------------------------------------------
// Constants — match notebook §0 setup
// -----------------------------------------------------------------------------

export const SEED = 42;
export const SAMPLE_SIZE = 200;

/**
 * Per-depth-function colour palette. Recurs across §§2–5 figures.
 */
export const DEPTH_COLORS = {
  tukey: '#1f77b4',
  mahalanobis: '#d62728',
  simplicial: '#2ca02c',
  projection: '#ff7f0e',
  spatial: '#9467bd',
  data: '#7f7f7f',
  outlier: '#000000',
  query: '#e377c2',
} as const;

// -----------------------------------------------------------------------------
// Sample generators — match notebook §1.4 running examples
// -----------------------------------------------------------------------------

// Cholesky factor of Σ = [[2, 1], [1, 1]] used by Sample A and Sample B.
//   L = [[√2, 0], [1/√2, 1/√2]];   L L^T = Σ ✓
const SIGMA_L11 = Math.SQRT2;
const SIGMA_L21 = 1 / Math.SQRT2;
const SIGMA_L22 = 1 / Math.SQRT2;

/**
 * Sample A — bivariate Gaussian with tilted covariance Σ = [[2, 1], [1, 1]].
 */
export function sampleGaussian(n: number, seed: number): Point2D[] {
  const rng = mulberry32(seed);
  const gauss = gaussianSampler(rng);
  const out: Point2D[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const z1 = gauss();
    const z2 = gauss();
    out[i] = [SIGMA_L11 * z1, SIGMA_L21 * z1 + SIGMA_L22 * z2];
  }
  return out;
}

/**
 * Sample B — bivariate Cauchy with the same scale matrix as Sample A.
 * Uses the (Z, W) construction: Z ~ N(0, Σ), W = Z'^2 ~ χ²₁,
 * Cauchy(0, Σ) = Z / √W.
 */
export function sampleCauchy(n: number, seed: number): Point2D[] {
  const rng = mulberry32(seed);
  const gauss = gaussianSampler(rng);
  const out: Point2D[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const z1 = gauss();
    const z2 = gauss();
    const zp = gauss();
    const sqW = Math.sqrt(Math.max(zp * zp, 1e-300));
    const x = SIGMA_L11 * z1;
    const y = SIGMA_L21 * z1 + SIGMA_L22 * z2;
    out[i] = [x / sqW, y / sqW];
  }
  return out;
}

/**
 * Sample C — (1 − ε) Sample-A-style Gaussian + ε outliers at `muOut`.
 * Default contamination is ε = 0.10 with outliers at (8, 8) — strong enough
 * that the moment-based centres visibly drift in the §1 figure.
 */
export function sampleContaminated(
  n: number,
  seed: number,
  eps = 0.10,
  muOut: Point2D = [8, 8],
): { points: Point2D[]; nOut: number } {
  const nOut = Math.floor(eps * n);
  const nIn = n - nOut;
  const inliers = sampleGaussian(nIn, seed);

  const rngOut = mulberry32(seed + 1000);
  const gaussOut = gaussianSampler(rngOut);
  const outliers: Point2D[] = new Array(nOut);
  for (let i = 0; i < nOut; i++) {
    outliers[i] = [muOut[0] + gaussOut(), muOut[1] + gaussOut()];
  }

  const all = [...inliers, ...outliers];
  // Fisher–Yates shuffle with a fresh stream so seed semantics stay stable.
  const rngShuffle = mulberry32(seed + 2000);
  for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(rngShuffle() * (i + 1));
    const tmp = all[i];
    all[i] = all[j];
    all[j] = tmp;
  }
  return { points: all, nOut };
}

/**
 * Banana sample — quadratic-bend distribution used in §3.5 to expose the
 * elliptical commitment of Mahalanobis depth. x₁ ~ N(0, 1.5²);
 * x₂ ~ N(0.5 x₁² − 1.5, 0.4²).
 */
export function sampleBanana(n: number, seed: number): Point2D[] {
  const rng = mulberry32(seed);
  const gauss = gaussianSampler(rng);
  const out: Point2D[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const x1 = gauss() * 1.5;
    const x2 = 0.5 * x1 * x1 - 1.5 + 0.4 * gauss();
    out[i] = [x1, x2];
  }
  return out;
}

// -----------------------------------------------------------------------------
// Geometric median — Weiszfeld's iteration. Used by §1 motivation figure.
// -----------------------------------------------------------------------------

export function weiszfeld(X: Point2D[], maxIter = 200, tol = 1e-8): Point2D {
  let yx = 0;
  let yy = 0;
  for (const [x, y] of X) {
    yx += x;
    yy += y;
  }
  yx /= X.length;
  yy /= X.length;
  for (let iter = 0; iter < maxIter; iter++) {
    let sumWX = 0;
    let sumWY = 0;
    let sumW = 0;
    for (const [x, y] of X) {
      const dx = x - yx;
      const dy = y - yy;
      let d = Math.sqrt(dx * dx + dy * dy);
      if (d < 1e-12) d = 1e-12;
      const w = 1 / d;
      sumWX += w * x;
      sumWY += w * y;
      sumW += w;
    }
    const yxNew = sumWX / sumW;
    const yyNew = sumWY / sumW;
    const dx = yxNew - yx;
    const dy = yyNew - yy;
    yx = yxNew;
    yy = yyNew;
    if (Math.sqrt(dx * dx + dy * dy) < tol) break;
  }
  return [yx, yy];
}

/**
 * Componentwise median — the §1 motivation figure's red square marker.
 */
export function componentwiseMedian(X: Point2D[]): Point2D {
  const xs = X.map(([x]) => x).sort((a, b) => a - b);
  const ys = X.map(([, y]) => y).sort((a, b) => a - b);
  return [median1D(xs), median1D(ys)];
}

function median1D(sorted: number[]): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n % 2 === 1) return sorted[(n - 1) >> 1];
  return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

/**
 * Sample mean — the §1 motivation figure's blue diamond marker.
 */
export function sampleMean(X: Point2D[]): Point2D {
  let sx = 0;
  let sy = 0;
  for (const [x, y] of X) {
    sx += x;
    sy += y;
  }
  return [sx / X.length, sy / X.length];
}

// -----------------------------------------------------------------------------
// Tukey halfspace depth (2D) — Rousseeuw–Ruts O(n log n) sweep
// -----------------------------------------------------------------------------

/**
 * Exact 2D Tukey halfspace depth via the rotating-halfplane sweep. Returns
 * HD_n(query) ∈ [0, 1]. Coincident sample points (within `eps` of `query`)
 * lie on every closed halfspace through the query and contribute additively
 * to the depth count.
 */
export function tukeyDepth2D(query: Point2D, X: Point2D[], eps = 1e-10): number {
  const n = X.length;
  if (n === 0) return 0;

  const angles: number[] = [];
  let nCoincident = 0;
  for (const [x, y] of X) {
    const dx = x - query[0];
    const dy = y - query[1];
    if (Math.sqrt(dx * dx + dy * dy) < eps) {
      nCoincident++;
    } else {
      angles.push(Math.atan2(dy, dx));
    }
  }
  const nNz = angles.length;
  if (nNz === 0) return 1;

  angles.sort((a, b) => a - b);
  const a = new Float64Array(2 * nNz);
  for (let i = 0; i < nNz; i++) {
    a[i] = angles[i];
    a[i + nNz] = angles[i] + 2 * Math.PI;
  }

  let minCount = nNz;
  let j = 0;
  for (let i = 0; i < nNz; i++) {
    while (j < 2 * nNz && a[j] < a[i] + Math.PI - 1e-12) j++;
    const count = j - i;
    if (count < minCount) minCount = count;
    if (nNz - count < minCount) minCount = nNz - count;
  }

  return (minCount + nCoincident) / n;
}

// -----------------------------------------------------------------------------
// Mahalanobis depth — 1 / (1 + (x−μ)' Σ⁻¹ (x−μ))
// -----------------------------------------------------------------------------

export interface MahalanobisParams {
  mu: Point2D;
  /** Σ⁻¹ packed as [Σ⁻¹₀₀, Σ⁻¹₀₁, Σ⁻¹₁₁] (symmetric 2×2). */
  sigmaInv: [number, number, number];
}

/**
 * Sample mean and inverse sample covariance for the Mahalanobis depth's
 * elliptical centre and metric.
 */
export function fitMahalanobis(X: Point2D[]): MahalanobisParams {
  const N = X.length;
  let mx = 0;
  let my = 0;
  for (const [x, y] of X) {
    mx += x;
    my += y;
  }
  mx /= N;
  my /= N;
  let s00 = 0;
  let s01 = 0;
  let s11 = 0;
  for (const [x, y] of X) {
    const dx = x - mx;
    const dy = y - my;
    s00 += dx * dx;
    s01 += dx * dy;
    s11 += dy * dy;
  }
  // Unbiased (N − 1) divisor matches numpy's default cov.
  const denom = N - 1;
  s00 /= denom;
  s01 /= denom;
  s11 /= denom;
  const det = s00 * s11 - s01 * s01;
  return {
    mu: [mx, my],
    sigmaInv: [s11 / det, -s01 / det, s00 / det],
  };
}

export function mahalanobisDepth(query: Point2D, params: MahalanobisParams): number {
  const dx = query[0] - params.mu[0];
  const dy = query[1] - params.mu[1];
  const [s00, s01, s11] = params.sigmaInv;
  const md = dx * (s00 * dx + s01 * dy) + dy * (s01 * dx + s11 * dy);
  return 1 / (1 + md);
}

// -----------------------------------------------------------------------------
// Projection depth — sup over directions of standardized univariate residual
// -----------------------------------------------------------------------------

/**
 * Projection depth via K random unit directions in R^d. For each direction u,
 * project the sample and the query onto u, measure |proj_q − median| / MAD,
 * and take the worst case over directions. Depth = 1 / (1 + outlyingness).
 *
 * Cost is O(K · n) per query; scales linearly in dimension. Default K = 200
 * matches notebook §3.3 / §4.5.
 */
export function projectionDepth(
  query: number[],
  X: number[][],
  K = 200,
  dirSeed = 0,
): number {
  const d = query.length;
  const n = X.length;
  if (n === 0) return 0;

  const rng = mulberry32(dirSeed);
  const gauss = gaussianSampler(rng);

  const projX = new Float64Array(n);
  let maxOutlyingness = 0;

  for (let k = 0; k < K; k++) {
    // Draw a fresh unit direction u ∈ S^{d-1}.
    const u = new Array(d);
    let nrm = 0;
    for (let dd = 0; dd < d; dd++) {
      const g = gauss();
      u[dd] = g;
      nrm += g * g;
    }
    nrm = Math.sqrt(nrm);
    if (nrm < 1e-12) continue;
    for (let dd = 0; dd < d; dd++) u[dd] /= nrm;

    let projQ = 0;
    for (let dd = 0; dd < d; dd++) projQ += query[dd] * u[dd];

    for (let i = 0; i < n; i++) {
      let p = 0;
      const Xi = X[i];
      for (let dd = 0; dd < d; dd++) p += Xi[dd] * u[dd];
      projX[i] = p;
    }

    // Median and MAD of the projected sample.
    const sorted = Array.from(projX).sort((a, b) => a - b);
    const med = median1D(sorted);
    const absDev = sorted.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
    let mad = median1D(absDev);
    if (mad < 1e-12) mad = 1e-12;

    const outlyingness = Math.abs(projQ - med) / mad;
    if (outlyingness > maxOutlyingness) maxOutlyingness = outlyingness;
  }
  return 1 / (1 + maxOutlyingness);
}

/**
 * Convenience wrapper for 2D queries — accepts `Point2D` directly.
 */
export function projectionDepth2D(
  query: Point2D,
  X: Point2D[],
  K = 200,
  dirSeed = 0,
): number {
  const X2 = X.map(([x, y]) => [x, y]);
  return projectionDepth([query[0], query[1]], X2, K, dirSeed);
}

// -----------------------------------------------------------------------------
// Spatial / L¹ depth — 1 − ‖mean unit-direction vector‖
// -----------------------------------------------------------------------------

export function spatialDepth(query: Point2D, X: Point2D[]): number {
  let sumX = 0;
  let sumY = 0;
  let kept = 0;
  for (const [x, y] of X) {
    const dx = x - query[0];
    const dy = y - query[1];
    const norm = Math.sqrt(dx * dx + dy * dy);
    if (norm < 1e-12) continue;
    sumX += dx / norm;
    sumY += dy / norm;
    kept++;
  }
  if (kept === 0) return 1;
  const meanX = sumX / kept;
  const meanY = sumY / kept;
  return 1 - Math.sqrt(meanX * meanX + meanY * meanY);
}

// -----------------------------------------------------------------------------
// Depth-contour grid — common scaffolding for §2 and §3.5 figures
// -----------------------------------------------------------------------------

export interface DepthGrid {
  xs: Float64Array;
  ys: Float64Array;
  Z: Float64Array; // row-major, size = xs.length * ys.length
  nrows: number;
  ncols: number;
}

/**
 * Evaluate `depthFn` on a regular 2D grid covering `X` plus padding.
 * Cost is gridSize² depth evaluations — keep gridSize small (30–50) for
 * stochastic depth functions, larger (60–80) for fast deterministic ones.
 */
export function depthContourGrid(
  X: Point2D[],
  depthFn: (q: Point2D) => number,
  gridSize = 70,
  pad = 1.5,
): DepthGrid {
  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;
  for (const [x, y] of X) {
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  xmin -= pad;
  xmax += pad;
  ymin -= pad;
  ymax += pad;

  const xs = new Float64Array(gridSize);
  const ys = new Float64Array(gridSize);
  for (let i = 0; i < gridSize; i++) {
    xs[i] = xmin + ((xmax - xmin) * i) / (gridSize - 1);
    ys[i] = ymin + ((ymax - ymin) * i) / (gridSize - 1);
  }

  const Z = new Float64Array(gridSize * gridSize);
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      Z[i * gridSize + j] = depthFn([xs[j], ys[i]]);
    }
  }
  return { xs, ys, Z, nrows: gridSize, ncols: gridSize };
}

/**
 * Centroid of grid cells achieving (within one grid step of) the maximum
 * depth — the empirical Tukey-median estimate from §2.5.
 */
export function tukeyMedianFromGrid(grid: DepthGrid): Point2D {
  let zMax = -Infinity;
  for (let k = 0; k < grid.Z.length; k++) {
    if (grid.Z[k] > zMax) zMax = grid.Z[k];
  }
  const threshold = zMax - 1.0 / Math.max(grid.nrows, grid.ncols);
  let sumX = 0;
  let sumY = 0;
  let count = 0;
  for (let i = 0; i < grid.nrows; i++) {
    for (let j = 0; j < grid.ncols; j++) {
      if (grid.Z[i * grid.ncols + j] >= threshold - 1e-10) {
        sumX += grid.xs[j];
        sumY += grid.ys[i];
        count++;
      }
    }
  }
  return count > 0 ? [sumX / count, sumY / count] : [0, 0];
}

// -----------------------------------------------------------------------------
// Iris dataset — Versicolor vs. Virginica, petal-length × petal-width
// -----------------------------------------------------------------------------
//
// Extracted via sklearn.datasets.load_iris(); features [petal length, petal
// width] in cm. 50 samples per class. The harder Iris pair (linear
// separability ≈ 96%) gives the §5.1 DD-classifier a non-trivial decision
// boundary. Petal features chosen per Fisher's classical analysis as the
// most discriminating 2D subspace.

export const IRIS_VERSICOLOR: Point2D[] = [
  [4.7, 1.4], [4.5, 1.5], [4.9, 1.5], [4.0, 1.3], [4.6, 1.5],
  [4.5, 1.3], [4.7, 1.6], [3.3, 1.0], [4.6, 1.3], [3.9, 1.4],
  [3.5, 1.0], [4.2, 1.5], [4.0, 1.0], [4.7, 1.4], [3.6, 1.3],
  [4.4, 1.4], [4.5, 1.5], [4.1, 1.0], [4.5, 1.5], [3.9, 1.1],
  [4.8, 1.8], [4.0, 1.3], [4.9, 1.5], [4.7, 1.2], [4.3, 1.3],
  [4.4, 1.4], [4.8, 1.4], [5.0, 1.7], [4.5, 1.5], [3.5, 1.0],
  [3.8, 1.1], [3.7, 1.0], [3.9, 1.2], [5.1, 1.6], [4.5, 1.5],
  [4.5, 1.6], [4.7, 1.5], [4.4, 1.3], [4.1, 1.3], [4.0, 1.3],
  [4.4, 1.2], [4.6, 1.4], [4.0, 1.2], [3.3, 1.0], [4.2, 1.3],
  [4.2, 1.2], [4.2, 1.3], [4.3, 1.3], [3.0, 1.1], [4.1, 1.3],
];

export const IRIS_VIRGINICA: Point2D[] = [
  [6.0, 2.5], [5.1, 1.9], [5.9, 2.1], [5.6, 1.8], [5.8, 2.2],
  [6.6, 2.1], [4.5, 1.7], [6.3, 1.8], [5.8, 1.8], [6.1, 2.5],
  [5.1, 2.0], [5.3, 1.9], [5.5, 2.1], [5.0, 2.0], [5.1, 2.4],
  [5.3, 2.3], [5.5, 1.8], [6.7, 2.2], [6.9, 2.3], [5.0, 1.5],
  [5.7, 2.3], [4.9, 2.0], [6.7, 2.0], [4.9, 1.8], [5.7, 2.1],
  [6.0, 1.8], [4.8, 1.8], [4.9, 1.8], [5.6, 2.1], [5.8, 1.6],
  [6.1, 1.9], [6.4, 2.0], [5.6, 2.2], [5.1, 1.5], [5.6, 1.4],
  [6.1, 2.3], [5.6, 2.4], [5.5, 1.8], [4.8, 1.8], [5.4, 2.1],
  [5.6, 2.4], [5.1, 2.3], [5.1, 1.9], [5.9, 2.3], [5.7, 2.5],
  [5.2, 2.3], [5.0, 1.9], [5.2, 2.0], [5.4, 2.3], [5.1, 1.8],
];

// -----------------------------------------------------------------------------
// Lazy accessors for the canonical sample triple — used by viz components
// -----------------------------------------------------------------------------

let cachedSampleA: Point2D[] | null = null;
let cachedSampleB: Point2D[] | null = null;
let cachedSampleC: { points: Point2D[]; nOut: number } | null = null;

/** Sample A — Gaussian, n = 200, seed offset 0 from SEED. */
export function getSampleA(): Point2D[] {
  if (cachedSampleA === null) cachedSampleA = sampleGaussian(SAMPLE_SIZE, SEED);
  return cachedSampleA;
}

/** Sample B — Cauchy, n = 200, seed offset 1 from SEED. */
export function getSampleB(): Point2D[] {
  if (cachedSampleB === null) cachedSampleB = sampleCauchy(SAMPLE_SIZE, SEED + 1);
  return cachedSampleB;
}

/** Sample C — contaminated Gaussian, n = 200, seed offset 2 from SEED. */
export function getSampleC(): { points: Point2D[]; nOut: number } {
  if (cachedSampleC === null) cachedSampleC = sampleContaminated(SAMPLE_SIZE, SEED + 2);
  return cachedSampleC;
}

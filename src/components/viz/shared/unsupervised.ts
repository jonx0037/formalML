// =============================================================================
// unsupervised.ts
//
// Shared math primitives for the T3 (Unsupervised & Generative) track.
// Introduced by the clustering topic; extended later by density-ratio-estimation.
//
// Source-of-truth notebook: notebooks/clustering/01_clustering.ipynb
// Brief: docs/plans/formalml-clustering-handoff-brief.md §12.5
//
// All exports are pure functions — no module-level state, deterministic
// outputs for a given seed. Long-running grid sweeps (e.g. basinOfAttractionMap
// at 80×80) skip trajectory history to keep memory flat.
// =============================================================================

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export type KernelName = 'gaussian' | 'epanechnikov' | 'biweight' | 'triweight';

export interface MeanShiftOptions {
  /** Maximum mean-shift iterations per query point. Default 100. */
  maxIter?: number;
  /** Per-iteration max-step convergence tolerance (Euclidean). Default 1e-4. */
  tol?: number;
  /** If false, only finalPositions and per-query iteration counts are returned. Default true. */
  returnHistory?: boolean;
  /** Kernel profile. Default 'gaussian'. */
  kernel?: KernelName;
}

export interface MeanShiftResult {
  /** (T+1) × m × d trajectory history (only present when returnHistory is true). */
  trajectories: number[][][] | null;
  /** Final position per query — shape m × d. */
  finalPositions: number[][];
  /** Iteration count at which each query stopped (≤ maxIter). */
  iterations: number[];
  /** Whether each query converged within maxIter. */
  converged: boolean[];
}

export interface ModeFinderOptions {
  dedupTol?: number;
  kernel?: KernelName;
  maxIter?: number;
  tol?: number;
}

export interface ModeFinderResult {
  modes: number[][];
  labels: number[];
}

export interface BandwidthSelectorOptions {
  mode?: 'silverman' | 'scree';
  /** When mode === 'scree', sweep across this log-spaced grid. */
  hGrid?: number[];
  dedupTol?: number;
}

export interface BandwidthSelectorResult {
  silverman: number;
  /** Present when mode === 'scree'. */
  recommended?: number;
  plateauRange?: [number, number];
  modeCounts?: number[];
}

export interface BasinGridSpec {
  x: [number, number];
  y: [number, number];
  nx: number;
  ny: number;
}

export interface BasinMapResult {
  /** ny rows × nx cols of integer mode labels in [0, modes.length-1]. */
  labels: number[][];
  /** Mode set aligned to sample-as-queries modeFinder output (so labels match scatter). */
  modes: number[][];
  /** Flat grid coordinates row-major (ny × nx, d=2). */
  grid: number[][];
}

// -----------------------------------------------------------------------------
// Implementation stubs — filled in Task 2.
// -----------------------------------------------------------------------------

/**
 * Vectorized Gaussian-kernel mean-shift trajectory computation.
 *
 * Brief §3.3, notebook §3.3 `mean_shift_trajectories`. Implements the
 * sample-weighted-mean update
 *   x_{t+1} = (Σ g(u_i(x_t)) x_i) / (Σ g(u_i(x_t)))
 * with the shadow profile g(s) selected by opts.kernel.
 *
 * Numerical-stability note: per-row max-subtract before exponentiation
 * (Gaussian only). Compact-support kernels (Epanechnikov / biweight /
 * triweight) zero out non-neighborhood weights directly.
 */
export function meanShift(
  X: number[][],
  queries: number[][],
  h: number,
  opts?: MeanShiftOptions,
): MeanShiftResult {
  throw new Error('unimplemented');
}

/**
 * Kernel-selectable mean-shift wrapper (brief §6).
 *
 * Convenience alias of `meanShift` with the kernel parameter promoted out of
 * the options object for call-site clarity. The four supported kernel families
 * (Gaussian default; Epanechnikov; biweight; triweight) all share the radial
 * profile-function framework from brief §3.1.
 */
export function meanShiftKernel(
  X: number[][],
  queries: number[][],
  h: number,
  kernel: KernelName,
  opts?: Omit<MeanShiftOptions, 'kernel'>,
): MeanShiftResult {
  throw new Error('unimplemented');
}

/**
 * Sample-as-queries mean-shift wrapper returning deduplicated modes plus
 * per-point cluster labels.
 *
 * Brief §7.2 mode-merging policy: greedy-union pass with tolerance `dedupTol`
 * (default 1e-3). Two endpoints within `dedupTol` Euclidean distance map to
 * the same mode.
 */
export function modeFinder(
  X: number[][],
  h: number,
  opts?: ModeFinderOptions,
): ModeFinderResult {
  throw new Error('unimplemented');
}

/**
 * Bandwidth selector for clustering — Silverman default plus optional
 * scree-criterion sweep.
 *
 * Silverman (brief §5.4):
 *   h_S = σ̂ · (4 / ((d+2) · n))^{1/(d+4)}
 * where σ̂ is the mean of per-coordinate standard deviations.
 *
 * Scree mode (brief §5.4 / §5.5): sweeps `hGrid` (default 50 log-spaced points
 * on [0.05, 2.0]), finds the longest plateau where countDistinctModes is
 * stable, returns the plateau midpoint as `recommended`.
 */
export function bandwidthSelectorForMeanShift(
  X: number[][],
  opts?: BandwidthSelectorOptions,
): BandwidthSelectorResult {
  throw new Error('unimplemented');
}

/**
 * Grid-as-queries basin-of-attraction computation for the §7.5 signature viz.
 *
 * For each cell of the grid defined by `gridSpec`, runs a mean-shift
 * trajectory from the cell centroid and assigns the cell to the closest
 * mode in the sample-as-queries mode set (so basin colors match scatter
 * colors deterministically, brief §7.5).
 *
 * Pass `returnHistory: false` to `meanShift` to keep memory flat at the
 * 80×80 default resolution.
 */
export function basinOfAttractionMap(
  X: number[][],
  h: number,
  gridSpec: BasinGridSpec,
  opts?: ModeFinderOptions,
): BasinMapResult {
  throw new Error('unimplemented');
}

/**
 * Scalar mode-count M(h) (brief §5). Wraps `modeFinder` and returns
 * `result.modes.length`.
 */
export function countDistinctModes(X: number[][], h: number, dedupTol = 1e-3): number {
  throw new Error('unimplemented');
}

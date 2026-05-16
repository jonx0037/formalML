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
  /** Maximum mean-shift iterations per query point. Default 300. */
  maxIter?: number;
  /** Per-iteration max-step convergence tolerance (Euclidean). Default 1e-6. */
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
// Kernel profiles. Returns g(s) where s = ||u||² and the radial kernel is
// K(u) = c_{k,d} · k(s) with shadow profile g(s) = -k'(s).
//
// The constants (e.g. the leading ½ on Gaussian) cancel in the weighted-mean
// update so we omit them for speed; the closed-form mean-shift step uses
// w_i = g(s_i) up to a constant common factor.
// -----------------------------------------------------------------------------

function shadowWeight(kernel: KernelName, sNormalized: number): number {
  // sNormalized := ||q - x||² / h²  (so the kernel's support is sNormalized ≤ 1 for compact kernels)
  switch (kernel) {
    case 'gaussian':
      return Math.exp(-0.5 * sNormalized);
    case 'epanechnikov':
      return sNormalized <= 1 ? 1 : 0;
    case 'biweight':
      return sNormalized <= 1 ? 1 - sNormalized : 0;
    case 'triweight': {
      if (sNormalized > 1) return 0;
      const t = 1 - sNormalized;
      return t * t;
    }
  }
}

/**
 * KDE value $\hat f_h(\mathbf{q})$ at a single point using the Gaussian kernel.
 * For verification of monotone-ascent: comparison across iterations needs only
 * relative magnitude, so the per-bandwidth normalization is constant and we
 * use the un-normalized log-sum-exp $\log \sum_i \exp(-\tfrac{1}{2} \|q-x_i\|^2 / h^2)$
 * as a monotone surrogate. The actual KDE differs only by a positive constant
 * factor $1 / (n h^d (2\pi)^{d/2})$.
 */
export function gaussianKdeLogSurrogate(
  X: number[][],
  q: number[],
  h: number,
): number {
  const n = X.length;
  const d = q.length;
  const h2 = h * h;
  let maxNeg = -Infinity;
  const logs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s2 = 0;
    for (let j = 0; j < d; j++) {
      const dv = q[j] - X[i][j];
      s2 += dv * dv;
    }
    logs[i] = -0.5 * s2 / h2;
    if (logs[i] > maxNeg) maxNeg = logs[i];
  }
  let sumExp = 0;
  for (let i = 0; i < n; i++) sumExp += Math.exp(logs[i] - maxNeg);
  return maxNeg + Math.log(sumExp);
}

// -----------------------------------------------------------------------------
// meanShift — vectorized mean-shift trajectory computation
// Brief §3.3, notebook §3.3 `mean_shift_trajectories`.
// -----------------------------------------------------------------------------

/**
 * Vectorized mean-shift trajectory computation.
 *
 * Update rule (brief §3.3):
 *   x_{t+1}^q = (Σ_i g_i · x_i) / (Σ_i g_i),   g_i = g(||x_t^q - x_i||² / h²)
 *
 * where g is the shadow profile selected by opts.kernel (Gaussian default).
 * For Gaussian kernel we use the standard log-domain max-subtract for
 * numerical stability; compact-support kernels avoid the exp altogether.
 */
export function meanShift(
  X: number[][],
  queries: number[][],
  h: number,
  opts: MeanShiftOptions = {},
): MeanShiftResult {
  const n = X.length;
  if (n === 0) throw new Error('meanShift: X must be non-empty');
  const d = X[0].length;
  const m = queries.length;
  const maxIter = opts.maxIter ?? 300;
  const tol = opts.tol ?? 1e-6;
  const kernel: KernelName = opts.kernel ?? 'gaussian';
  const returnHistory = opts.returnHistory ?? true;
  const h2 = h * h;

  // Validate input shapes.
  for (let q = 0; q < m; q++) {
    if (queries[q].length !== d) {
      throw new Error(`meanShift: queries[${q}] dim=${queries[q].length}, expected d=${d}`);
    }
  }

  // Working buffers: current positions, next positions, weight scratch.
  const current: number[][] = queries.map((row) => row.slice());
  const next: number[][] = Array.from({ length: m }, () => new Array<number>(d).fill(0));
  const iterations = new Array<number>(m).fill(0);
  const converged = new Array<boolean>(m).fill(false);
  const trajectories: number[][][] | null = returnHistory ? [] : null;
  if (trajectories) trajectories.push(current.map((row) => row.slice()));

  // Per-query active flag — stop iterating once a query has converged.
  const active = new Array<boolean>(m).fill(true);

  // Scratch buffers hoisted out of both the t- and q-loops. With m up to ~6400
  // for an 80×80 basin grid and maxIter up to 300, per-iteration allocation
  // would burn ~2M Float64Array(n) allocations per call.
  const num = new Float64Array(d);
  const negHalfNorm = kernel === 'gaussian' ? new Float64Array(n) : null;

  for (let t = 0; t < maxIter; t++) {
    let anyActive = false;
    let maxStepSquared = 0;

    for (let q = 0; q < m; q++) {
      if (!active[q]) {
        // Carry forward.
        for (let j = 0; j < d; j++) next[q][j] = current[q][j];
        continue;
      }
      anyActive = true;
      const cur = current[q];

      // Compute weights and weighted-mean accumulator in one pass.
      let wSum = 0;
      num.fill(0);

      if (kernel === 'gaussian' && negHalfNorm) {
        // Pass 1: compute squared distances and track max for log-domain shift.
        let maxNeg = -Infinity;
        for (let i = 0; i < n; i++) {
          let s2 = 0;
          for (let j = 0; j < d; j++) {
            const dv = cur[j] - X[i][j];
            s2 += dv * dv;
          }
          const e = -0.5 * s2 / h2;
          negHalfNorm[i] = e;
          if (e > maxNeg) maxNeg = e;
        }
        // Pass 2: exp(e - max), accumulate.
        for (let i = 0; i < n; i++) {
          const w = Math.exp(negHalfNorm[i] - maxNeg);
          wSum += w;
          for (let j = 0; j < d; j++) num[j] += w * X[i][j];
        }
      } else {
        // Compact-support kernels: zero contribution outside the bandwidth ball.
        for (let i = 0; i < n; i++) {
          let s2 = 0;
          for (let j = 0; j < d; j++) {
            const dv = cur[j] - X[i][j];
            s2 += dv * dv;
          }
          const sNorm = s2 / h2;
          const w = shadowWeight(kernel, sNorm);
          if (w > 0) {
            wSum += w;
            for (let j = 0; j < d; j++) num[j] += w * X[i][j];
          }
        }
      }

      if (wSum > 0) {
        // Standard update.
        let stepSq = 0;
        for (let j = 0; j < d; j++) {
          const newVal = num[j] / wSum;
          const delta = newVal - cur[j];
          stepSq += delta * delta;
          next[q][j] = newVal;
        }
        if (stepSq > maxStepSquared) maxStepSquared = stepSq;
        if (stepSq < tol * tol) {
          active[q] = false;
          converged[q] = true;
          iterations[q] = t + 1;
        }
      } else {
        // Empty neighborhood (compact kernel) — stop the trajectory.
        for (let j = 0; j < d; j++) next[q][j] = cur[j];
        active[q] = false;
        converged[q] = false;
        iterations[q] = t + 1;
      }
    }

    // Swap current ↔ next.
    for (let q = 0; q < m; q++) {
      for (let j = 0; j < d; j++) current[q][j] = next[q][j];
    }
    if (trajectories) trajectories.push(current.map((row) => row.slice()));

    if (!anyActive) break;
  }

  // Fill in iterations for queries that hit maxIter without converging.
  for (let q = 0; q < m; q++) {
    if (iterations[q] === 0) iterations[q] = maxIter;
  }

  return {
    trajectories,
    finalPositions: current.map((row) => row.slice()),
    iterations,
    converged,
  };
}

/**
 * Kernel-selectable mean-shift wrapper (brief §6).
 *
 * Convenience alias of `meanShift` with the kernel parameter promoted out of
 * the options object for call-site clarity.
 */
export function meanShiftKernel(
  X: number[][],
  queries: number[][],
  h: number,
  kernel: KernelName,
  opts: Omit<MeanShiftOptions, 'kernel'> = {},
): MeanShiftResult {
  return meanShift(X, queries, h, { ...opts, kernel });
}

// -----------------------------------------------------------------------------
// Greedy-union mode dedup (brief §7.2)
// -----------------------------------------------------------------------------

function greedyDedup(
  endpoints: number[][],
  tol: number,
): { modes: number[][]; labels: number[] } {
  const modes: number[][] = [];
  const labels: number[] = [];
  const tol2 = tol * tol;
  for (const p of endpoints) {
    let matched = -1;
    for (let m = 0; m < modes.length; m++) {
      let s2 = 0;
      for (let j = 0; j < p.length; j++) {
        const dv = p[j] - modes[m][j];
        s2 += dv * dv;
      }
      if (s2 < tol2) {
        matched = m;
        break;
      }
    }
    if (matched < 0) {
      modes.push(p.slice());
      labels.push(modes.length - 1);
    } else {
      labels.push(matched);
    }
  }
  return { modes, labels };
}

// -----------------------------------------------------------------------------
// modeFinder — sample-as-queries wrapper
// -----------------------------------------------------------------------------

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
  opts: ModeFinderOptions = {},
): ModeFinderResult {
  const result = meanShift(X, X, h, {
    maxIter: opts.maxIter ?? 300,
    tol: opts.tol ?? 1e-6,
    kernel: opts.kernel ?? 'gaussian',
    returnHistory: false,
  });
  return greedyDedup(result.finalPositions, opts.dedupTol ?? 1e-3);
}

// -----------------------------------------------------------------------------
// bandwidthSelectorForMeanShift — Silverman + scree
// -----------------------------------------------------------------------------

function meanCoordinateStd(X: number[][]): number {
  const n = X.length;
  const d = X[0].length;
  let total = 0;
  for (let j = 0; j < d; j++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += X[i][j];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) {
      const dv = X[i][j] - mean;
      v += dv * dv;
    }
    total += Math.sqrt(v / Math.max(1, n - 1));
  }
  return total / d;
}

/**
 * Silverman bandwidth (brief §5.4):
 *   h_S = σ̂ · (4 / ((d+2) · n))^{1/(d+4)}
 *
 * Plus optional scree mode that sweeps an h-grid and picks the longest plateau.
 */
export function bandwidthSelectorForMeanShift(
  X: number[][],
  opts: BandwidthSelectorOptions = {},
): BandwidthSelectorResult {
  const n = X.length;
  const d = X[0].length;
  const sigma = meanCoordinateStd(X);
  const exponent = 1 / (d + 4);
  const silverman = sigma * Math.pow(4 / ((d + 2) * n), exponent);

  if (opts.mode !== 'scree') return { silverman };

  const hGrid = opts.hGrid ?? logSpace(0.05, 2.0, 50);
  const dedupTol = opts.dedupTol ?? 1e-3;
  const modeCounts: number[] = hGrid.map((h) => countDistinctModes(X, h, dedupTol));

  // Find longest plateau of constant mode-count.
  let bestStart = 0;
  let bestLen = 0;
  let runStart = 0;
  for (let i = 1; i <= hGrid.length; i++) {
    const breaking = i === hGrid.length || modeCounts[i] !== modeCounts[runStart];
    if (breaking) {
      const runLen = i - runStart;
      if (runLen > bestLen) {
        bestLen = runLen;
        bestStart = runStart;
      }
      runStart = i;
    }
  }
  const plateauRange: [number, number] = [hGrid[bestStart], hGrid[bestStart + bestLen - 1]];
  const recommended = Math.sqrt(plateauRange[0] * plateauRange[1]); // geometric midpoint on log scale

  return { silverman, recommended, plateauRange, modeCounts };
}

function logSpace(lo: number, hi: number, n: number): number[] {
  const out = new Array<number>(n);
  const logLo = Math.log(lo);
  const logHi = Math.log(hi);
  for (let i = 0; i < n; i++) out[i] = Math.exp(logLo + ((logHi - logLo) * i) / (n - 1));
  return out;
}

// -----------------------------------------------------------------------------
// basinOfAttractionMap — grid-as-queries
// -----------------------------------------------------------------------------

/**
 * Grid-as-queries basin computation. Labels are aligned to the sample-as-queries
 * mode set so basin colors match scatter colors deterministically (brief §7.5).
 */
export function basinOfAttractionMap(
  X: number[][],
  h: number,
  gridSpec: BasinGridSpec,
  opts: ModeFinderOptions = {},
): BasinMapResult {
  // Step 1: compute the sample-as-queries mode set.
  const dataResult = modeFinder(X, h, opts);
  const refModes = dataResult.modes;

  // Step 2: build the grid query matrix.
  const { x, y, nx, ny } = gridSpec;
  const dx = (x[1] - x[0]) / (nx - 1);
  const dy = (y[1] - y[0]) / (ny - 1);
  const grid: number[][] = [];
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      grid.push([x[0] + c * dx, y[0] + r * dy]);
    }
  }

  // Step 3: run mean-shift from each grid cell, no history kept.
  const meanShiftResult = meanShift(X, grid, h, {
    maxIter: opts.maxIter ?? 300,
    tol: opts.tol ?? 1e-6,
    kernel: opts.kernel ?? 'gaussian',
    returnHistory: false,
  });

  // Step 4: assign each grid endpoint to its nearest reference mode.
  const labels: number[][] = [];
  for (let r = 0; r < ny; r++) {
    const row: number[] = [];
    for (let c = 0; c < nx; c++) {
      const ep = meanShiftResult.finalPositions[r * nx + c];
      let bestIdx = 0;
      let bestDist2 = Infinity;
      for (let m = 0; m < refModes.length; m++) {
        let s2 = 0;
        for (let j = 0; j < ep.length; j++) {
          const dv = ep[j] - refModes[m][j];
          s2 += dv * dv;
        }
        if (s2 < bestDist2) {
          bestDist2 = s2;
          bestIdx = m;
        }
      }
      row.push(bestIdx);
    }
    labels.push(row);
  }

  return { labels, modes: refModes, grid };
}

// -----------------------------------------------------------------------------
// countDistinctModes — scalar M(h)
// -----------------------------------------------------------------------------

/**
 * Scalar mode-count M(h) — wraps modeFinder and returns modes.length.
 * Brief §5; notebook §5.5 `count_modes`.
 */
export function countDistinctModes(X: number[][], h: number, dedupTol = 1e-3): number {
  return modeFinder(X, h, { dedupTol }).modes.length;
}

// Pre-computed Robust PCA decompositions for heatmap visualizations.
// Each entry holds a 30×25 observed matrix X = L + S, the low-rank and
// sparse components, singular values, and recovery diagnostics.
//
// All matrices are generated deterministically via index-based hashing.

// ─── Types ───

export interface RobustPCAResult {
  rank: number;
  corruption: number;
  matrixX: number[][];
  matrixL: number[][];
  matrixS: number[][];
  singularValuesX: number[];
  singularValuesL: number[];
  recoveredRank: number;
  nnzS: number;
  relativeError: number;
}

// ─── Constants ───

const ROWS = 30;
const COLS = 25;
const RANKS = [1, 2, 3, 5, 8];
const CORRUPTIONS = [0.05, 0.10, 0.15, 0.20, 0.30];
const TOP_K_SV = 15;

// ─── Deterministic pseudo-random functions ───

/** Returns a value in [0, 1). */
function rand(seed: number): number {
  // Use a simple LCG-style hash.
  let s = ((seed * 2654435761) >>> 0) ^ 0xdeadbeef;
  s = ((s * 1597334677) >>> 0);
  return (s >>> 0) / 4294967296;
}

/** Returns a value roughly in [-1, 1] with a bell-curve-ish distribution. */
function randn(seed: number): number {
  // Approximate normal via sum of uniforms (Irwin-Hall, n=4).
  return (rand(seed) + rand(seed + 1) + rand(seed + 2) + rand(seed + 3) - 2) * 1.0;
}

/** Round to 4 decimal places. */
function r4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

// ─── Matrix generation ───

/**
 * Generate a rank-r low-rank matrix L = U * V^T where U is ROWS×r and V is COLS×r.
 * Each entry of U, V is drawn from a seeded pseudo-normal distribution.
 */
function generateLowRank(rank: number, baseSeed: number): number[][] {
  // Generate U (ROWS × rank)
  const U: number[][] = [];
  for (let i = 0; i < ROWS; i++) {
    const row: number[] = [];
    for (let k = 0; k < rank; k++) {
      row.push(randn(baseSeed + i * rank + k));
    }
    U.push(row);
  }
  // Generate V (COLS × rank)
  const V: number[][] = [];
  for (let j = 0; j < COLS; j++) {
    const row: number[] = [];
    for (let k = 0; k < rank; k++) {
      row.push(randn(baseSeed + 10000 + j * rank + k));
    }
    V.push(row);
  }
  // L = U * V^T
  const L: number[][] = [];
  for (let i = 0; i < ROWS; i++) {
    const row: number[] = [];
    for (let j = 0; j < COLS; j++) {
      let val = 0;
      for (let k = 0; k < rank; k++) {
        val += U[i][k] * V[j][k];
      }
      row.push(r4(val));
    }
    L.push(row);
  }
  return L;
}

/**
 * Generate a sparse corruption matrix S with approximately (corruption * ROWS * COLS)
 * nonzero entries. Nonzero values are large relative to L entries.
 */
function generateSparse(corruption: number, baseSeed: number): number[][] {
  const S: number[][] = [];
  const totalEntries = ROWS * COLS;

  // Determine which positions are nonzero using a deterministic hash.
  // We threshold rand() < corruption for each entry.
  for (let i = 0; i < ROWS; i++) {
    const row: number[] = [];
    for (let j = 0; j < COLS; j++) {
      const idx = i * COLS + j;
      if (rand(baseSeed + idx) < corruption) {
        // Large value with random sign
        const sign = rand(baseSeed + idx + totalEntries) < 0.5 ? -1 : 1;
        const magnitude = 3 + rand(baseSeed + idx + 2 * totalEntries) * 5;
        row.push(r4(sign * magnitude));
      } else {
        row.push(0);
      }
    }
    S.push(row);
  }
  return S;
}

/** Add two matrices element-wise. */
function matAdd(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => r4(v + B[i][j])));
}

/** Frobenius norm of a matrix. */
function frobNorm(M: number[][]): number {
  let sum = 0;
  for (const row of M) {
    for (const v of row) {
      sum += v * v;
    }
  }
  return Math.sqrt(sum);
}

/** Count nonzero entries. */
function nnz(M: number[][]): number {
  let count = 0;
  for (const row of M) {
    for (const v of row) {
      if (v !== 0) count++;
    }
  }
  return count;
}

/**
 * Approximate top-k singular values of a matrix using power iteration on M*M^T.
 * This is a simplified, deterministic approximation suitable for visualization data.
 */
function approxSingularValues(M: number[][], k: number, seed: number): number[] {
  const m = M.length;
  const n = M[0].length;

  // Compute M * M^T (m × m) for eigenvalue estimation.
  // For our small matrices (30×25) this is fine.
  const MMt: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      let val = 0;
      for (let p = 0; p < n; p++) {
        val += M[i][p] * M[j][p];
      }
      row.push(val);
    }
    MMt.push(row);
  }

  // Power iteration with deflation to get top-k eigenvalues of M*M^T.
  const eigenvalues: number[] = [];
  const deflated = MMt.map((row) => [...row]);

  for (let ki = 0; ki < Math.min(k, m); ki++) {
    // Initialize vector deterministically.
    let v: number[] = Array.from({ length: m }, (_, i) =>
      randn(seed + ki * 1000 + i),
    );
    // Normalize.
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    v = v.map((x) => x / norm);

    // 30 iterations of power method.
    for (let iter = 0; iter < 30; iter++) {
      const w: number[] = new Array(m).fill(0);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
          w[i] += deflated[i][j] * v[j];
        }
      }
      norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
      if (norm < 1e-12) break;
      v = w.map((x) => x / norm);
    }

    // Rayleigh quotient = eigenvalue of M*M^T = sigma^2.
    let eigenvalue = 0;
    for (let i = 0; i < m; i++) {
      let mv = 0;
      for (let j = 0; j < m; j++) {
        mv += deflated[i][j] * v[j];
      }
      eigenvalue += v[i] * mv;
    }
    eigenvalue = Math.max(0, eigenvalue);
    eigenvalues.push(r4(Math.sqrt(eigenvalue)));

    // Deflate: remove this eigenvector's contribution.
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        deflated[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }

  return eigenvalues;
}

/**
 * Simulate Robust PCA recovery quality.
 * For low rank + low corruption: good recovery (small relative error).
 * For high rank or high corruption: worse recovery.
 */
function simulateRecovery(
  rank: number,
  corruption: number,
): { recoveredRank: number; relativeError: number } {
  // Incoherence condition: recovery degrades when rank * corruption is large.
  // Threshold ~ 0.15 is roughly the boundary of exact recovery for RPCA.
  const difficulty = rank * corruption;

  let relativeError: number;
  if (difficulty < 0.1) {
    relativeError = 0.001 + rand(rank * 100 + Math.round(corruption * 1000)) * 0.01;
  } else if (difficulty < 0.3) {
    relativeError = 0.01 + difficulty * 0.15 + rand(rank * 200 + Math.round(corruption * 1000)) * 0.02;
  } else if (difficulty < 0.8) {
    relativeError = 0.05 + difficulty * 0.2 + rand(rank * 300 + Math.round(corruption * 1000)) * 0.05;
  } else {
    relativeError = 0.15 + difficulty * 0.15 + rand(rank * 400 + Math.round(corruption * 1000)) * 0.1;
  }

  // Recovered rank: correct for easy cases, overestimated for hard cases.
  let recoveredRank = rank;
  if (difficulty > 0.5) {
    recoveredRank = rank + Math.floor(difficulty * 2);
  } else if (difficulty > 0.2) {
    recoveredRank = rank + (rand(rank * 500 + Math.round(corruption * 1000)) < 0.3 ? 1 : 0);
  }

  return { recoveredRank, relativeError: r4(relativeError) };
}

// ─── Generate all (rank, corruption) combinations ───

function generateAllResults(): RobustPCAResult[] {
  const results: RobustPCAResult[] = [];

  for (const rank of RANKS) {
    for (const corruption of CORRUPTIONS) {
      const baseSeedL = rank * 10000 + Math.round(corruption * 10000);
      const baseSeedS = baseSeedL + 50000;

      const L = generateLowRank(rank, baseSeedL);
      const S = generateSparse(corruption, baseSeedS);
      const X = matAdd(L, S);

      const svSeed = baseSeedL + 80000;
      const singularValuesX = approxSingularValues(X, TOP_K_SV, svSeed);
      const singularValuesL = approxSingularValues(L, TOP_K_SV, svSeed + 20000);

      const { recoveredRank, relativeError } = simulateRecovery(rank, corruption);

      results.push({
        rank,
        corruption,
        matrixX: X,
        matrixL: L,
        matrixS: S,
        singularValuesX,
        singularValuesL,
        recoveredRank,
        nnzS: nnz(S),
        relativeError,
      });
    }
  }

  return results;
}

// ─── Export ───

let robustPCAResultsCache: RobustPCAResult[] | null = null;

/** Lazily generate all (rank, corruption) results on first access. */
export function getRobustPCAResults(): RobustPCAResult[] {
  if (robustPCAResultsCache === null) {
    robustPCAResultsCache = generateAllResults();
  }
  return robustPCAResultsCache;
}

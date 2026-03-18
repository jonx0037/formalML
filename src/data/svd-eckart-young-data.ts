// Pre-computed SVD data for the Eckart–Young Explorer.
// The matrix is constructed FROM its SVD factors (DCT-II basis vectors + designed
// singular values), so the decomposition is exact by construction.

export const ROWS = 16;
export const COLS = 12;

// ─── Singular values: 4 dominant + rapid decay to noise floor ───

export const singularValues: number[] = [
  50, 30, 18, 10, 4, 2, 1, 0.5, 0.25, 0.12, 0.06, 0.03,
];

// ─── DCT-II basis (orthonormal) ───

function dctBasisValue(N: number, k: number, i: number): number {
  const scale = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
  return scale * Math.cos((Math.PI * (2 * i + 1) * k) / (2 * N));
}

// Column-basis permutation for Vt rows — interleaves frequencies
// so the matrix has interesting 2D block structure in the heatmap.
const vtOrder: number[] = [0, 3, 1, 5, 2, 7, 4, 9, 6, 11, 8, 10];

// ─── Access functions ───

/** U[i][k]: the (i,k) entry of the left singular matrix (ROWS × COLS). */
export function getU(i: number, k: number): number {
  return dctBasisValue(ROWS, k, i);
}

/** Vt[k][j]: the (k,j) entry of the right singular matrix (COLS × COLS). */
export function getVt(k: number, j: number): number {
  return dctBasisValue(COLS, vtOrder[k], j);
}

// ─── Matrix reconstruction ───

/** Compute element (i,j) of the rank-r approximation A_r. */
export function computeElement(i: number, j: number, rank: number): number {
  let val = 0;
  const r = Math.min(rank, COLS);
  for (let k = 0; k < r; k++) {
    val += singularValues[k] * getU(i, k) * getVt(k, j);
  }
  return val;
}

/** Compute the full rank-r approximation matrix (ROWS × COLS). */
export function computeMatrix(rank: number): number[][] {
  return Array.from({ length: ROWS }, (_, i) =>
    Array.from({ length: COLS }, (_, j) => computeElement(i, j, rank)),
  );
}

// ─── Pre-computed original matrix (full rank) ───

export const originalMatrix: number[][] = computeMatrix(COLS);

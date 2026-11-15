// Deterministically generated tensor decomposition data for interactive visualizations.
// All tensors are constructed FROM known factors with a fixed seeded RNG so
// decompositions are exact by construction; small iterative algorithms (e.g. SVD)
// are run once at module initialization / first use in the browser.

// ─── Interfaces ───

export interface CPResult {
  rank: number;
  weights: number[];
  factors: number[][][]; // factors[mode][row][component]
  reconstruction: number[][][]; // 3D tensor
  relativeError: number;
  alsErrors: number[]; // convergence trace (error per iteration)
}

export interface TuckerResult {
  ranks: [number, number, number];
  core: number[][][];
  factors: number[][][]; // factors[mode] is a 2D array
  reconstruction: number[][][];
  relativeError: number;
  compressionRatio: number;
}

export interface tSVDResult {
  tubalRank: number;
  relativeError: number;
  fourierSingularValues: number[][]; // [slice][index]
  reconstructionSlice: number[][]; // first frontal slice of the reconstruction
}

export interface TensorExample {
  shape: [number, number, number];
  data: number[][][];
}

// ─── Seeded RNG ───

function seededRand(seed: number): number {
  const x = Math.sin(seed * 9301 + 49297) * 49297;
  return x - Math.floor(x);
}

// ─── Linear algebra helpers ───

function tensorNorm(T: number[][][]): number {
  let sum = 0;
  for (const slice of T) for (const row of slice) for (const v of row) sum += v * v;
  return Math.sqrt(sum);
}

function tensorDiff(A: number[][][], B: number[][][]): number[][][] {
  return A.map((slice, i) =>
    slice.map((row, j) => row.map((v, k) => v - B[i][j][k])),
  );
}

function zeros3D(d1: number, d2: number, d3: number): number[][][] {
  return Array.from({ length: d1 }, () =>
    Array.from({ length: d2 }, () => new Array<number>(d3).fill(0)),
  );
}

function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length, n = B[0].length, p = B.length;
  const C = Array.from({ length: m }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < p; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}

/** Gram-Schmidt orthonormalization of column vectors. Input: rows × cols matrix. */
function gramSchmidt(M: number[][]): number[][] {
  const rows = M.length, cols = M[0].length;
  const Q = M.map((r) => [...r]);
  for (let j = 0; j < cols; j++) {
    // Subtract projections of previous columns
    for (let prev = 0; prev < j; prev++) {
      let dot = 0, normSq = 0;
      for (let i = 0; i < rows; i++) {
        dot += Q[i][j] * Q[i][prev];
        normSq += Q[i][prev] * Q[i][prev];
      }
      const scale = dot / (normSq || 1);
      for (let i = 0; i < rows; i++) Q[i][j] -= scale * Q[i][prev];
    }
    // Normalize
    let norm = 0;
    for (let i = 0; i < rows; i++) norm += Q[i][j] * Q[i][j];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < rows; i++) Q[i][j] /= norm;
  }
  return Q;
}

// ─── Mode-n unfolding ───

/** Unfold a 3D tensor along the given mode (0, 1, or 2). */
export function modeNUnfold(tensor: number[][][], mode: number): number[][] {
  const I1 = tensor.length, I2 = tensor[0].length, I3 = tensor[0][0].length;
  const dims = [I1, I2, I3];
  const rows = dims[mode];
  const cols = dims.reduce((a, b) => a * b, 1) / rows;
  const result = Array.from({ length: rows }, () => new Array<number>(cols).fill(0));

  for (let i1 = 0; i1 < I1; i1++)
    for (let i2 = 0; i2 < I2; i2++)
      for (let i3 = 0; i3 < I3; i3++) {
        const idx = [i1, i2, i3];
        const row = idx[mode];
        // Column index: Kronecker ordering of remaining indices
        let col: number;
        if (mode === 0) col = i2 * I3 + i3;
        else if (mode === 1) col = i1 * I3 + i3;
        else col = i1 * I2 + i2;
        result[row][col] = tensor[i1][i2][i3];
      }
  return result;
}

// ─── Example tensor for TensorUnfoldingExplorer ───

export function getExampleTensor(): TensorExample {
  // 4×3×2 tensor with integer values 1–24
  const data: number[][][] = zeros3D(4, 3, 2);
  let val = 1;
  for (let k = 0; k < 2; k++)
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 3; j++) data[i][j][k] = val++;
  return { shape: [4, 3, 2], data };
}

// ─── CP Decomposition Data ───

function computeCPData(): CPResult[] {
  const I1 = 10, I2 = 8, I3 = 6, R = 5;
  const weights = [15, 10, 6, 3, 1];

  // Generate factor matrices using seeded RNG
  let seed = 42;
  const factors: number[][][] = []; // factors[mode][row][component]
  for (const dim of [I1, I2, I3]) {
    const mat: number[][] = Array.from({ length: dim }, () => {
      const row: number[] = [];
      for (let r = 0; r < R; r++) {
        row.push(seededRand(seed++) * 2 - 1);
      }
      return row;
    });
    factors.push(mat);
  }

  // Build the full tensor as sum of weighted rank-1 terms
  const fullTensor = zeros3D(I1, I2, I3);
  for (let r = 0; r < R; r++) {
    for (let i = 0; i < I1; i++)
      for (let j = 0; j < I2; j++)
        for (let k = 0; k < I3; k++)
          fullTensor[i][j][k] += weights[r] * factors[0][i][r] * factors[1][j][r] * factors[2][k][r];
  }

  const fullNorm = tensorNorm(fullTensor);
  const results: CPResult[] = [];

  for (let rank = 1; rank <= R; rank++) {
    // Rank-rank approximation: sum of first `rank` terms
    const approx = zeros3D(I1, I2, I3);
    for (let r = 0; r < rank; r++) {
      for (let i = 0; i < I1; i++)
        for (let j = 0; j < I2; j++)
          for (let k = 0; k < I3; k++)
            approx[i][j][k] += weights[r] * factors[0][i][r] * factors[1][j][r] * factors[2][k][r];
    }

    const errNorm = tensorNorm(tensorDiff(fullTensor, approx));
    const relativeError = errNorm / fullNorm;

    // Fake ALS convergence trace: starts high, decays toward the true error
    const alsErrors: number[] = [];
    const startErr = 0.95;
    for (let it = 0; it < 20; it++) {
      const t = (it + 1) / 20;
      const decay = startErr * Math.exp(-4 * t) + relativeError;
      alsErrors.push(Math.min(decay, startErr));
    }

    // Truncated factors for this rank
    const truncFactors = factors.map((mat) =>
      mat.map((row) => row.slice(0, rank)),
    );

    results.push({
      rank,
      weights: weights.slice(0, rank),
      factors: truncFactors,
      reconstruction: approx,
      relativeError,
      alsErrors,
    });
  }

  return results;
}

let cpCache: CPResult[] | null = null;
export function getCPResults(): CPResult[] {
  if (cpCache === null) cpCache = computeCPData();
  return cpCache;
}

// ─── Tucker Decomposition Data ───

function computeTuckerData(): TuckerResult[] {
  const I1 = 12, I2 = 10, I3 = 8;
  const maxRanks: [number, number, number] = [4, 3, 3];

  // Generate orthogonal factor matrices via Gram-Schmidt on seeded random data
  let seed = 137;
  const fullFactors: number[][][] = [];
  for (const [dim, maxR] of [[I1, maxRanks[0]], [I2, maxRanks[1]], [I3, maxRanks[2]]] as [number, number][]) {
    const raw: number[][] = Array.from({ length: dim }, () => {
      const row: number[] = [];
      for (let r = 0; r < maxR; r++) row.push(seededRand(seed++) * 2 - 1);
      return row;
    });
    fullFactors.push(gramSchmidt(raw));
  }

  // Generate a core tensor G of shape maxRanks with seeded values
  const G = zeros3D(maxRanks[0], maxRanks[1], maxRanks[2]);
  // Diagonal-dominant core: larger values on "diagonal"
  for (let i = 0; i < maxRanks[0]; i++)
    for (let j = 0; j < maxRanks[1]; j++)
      for (let k = 0; k < maxRanks[2]; k++) {
        const diag = (i === j && j === k) ? 20 - i * 5 : 0;
        G[i][j][k] = diag + seededRand(seed++) * 2 - 1;
      }

  // Reconstruct full tensor: T = G ×₁ U₁ ×₂ U₂ ×₃ U₃
  function tuckerReconstruct(
    core: number[][][],
    U: number[][][],
    outShape: [number, number, number],
  ): number[][][] {
    const [R1, R2, R3] = [core.length, core[0].length, core[0][0].length];
    const result = zeros3D(outShape[0], outShape[1], outShape[2]);
    for (let i = 0; i < outShape[0]; i++)
      for (let j = 0; j < outShape[1]; j++)
        for (let k = 0; k < outShape[2]; k++)
          for (let r1 = 0; r1 < R1; r1++)
            for (let r2 = 0; r2 < R2; r2++)
              for (let r3 = 0; r3 < R3; r3++)
                result[i][j][k] += core[r1][r2][r3] * U[0][i][r1] * U[1][j][r2] * U[2][k][r3];
    return result;
  }

  const fullTensor = tuckerReconstruct(G, fullFactors, [I1, I2, I3]);
  const fullNorm = tensorNorm(fullTensor);
  const totalElements = I1 * I2 * I3;

  const rankCombos: [number, number, number][] = [
    [2, 2, 2], [3, 2, 2], [3, 3, 2], [4, 3, 2], [4, 3, 3],
  ];

  const results: TuckerResult[] = [];

  for (const ranks of rankCombos) {
    // Truncate core and factors
    const truncCore = zeros3D(ranks[0], ranks[1], ranks[2]);
    for (let i = 0; i < ranks[0]; i++)
      for (let j = 0; j < ranks[1]; j++)
        for (let k = 0; k < ranks[2]; k++)
          truncCore[i][j][k] = G[i][j][k];

    const truncFactors = fullFactors.map((mat, mode) =>
      mat.map((row) => row.slice(0, ranks[mode])),
    );

    const recon = tuckerReconstruct(truncCore, truncFactors, [I1, I2, I3]);
    const errNorm = tensorNorm(tensorDiff(fullTensor, recon));
    const relativeError = errNorm / fullNorm;

    const coreElements = ranks[0] * ranks[1] * ranks[2];
    const factorElements = I1 * ranks[0] + I2 * ranks[1] + I3 * ranks[2];
    const compressionRatio = totalElements / (coreElements + factorElements);

    results.push({
      ranks,
      core: truncCore,
      factors: truncFactors,
      reconstruction: recon,
      relativeError,
      compressionRatio,
    });
  }

  return results;
}

let tuckerCache: TuckerResult[] | null = null;
export function getTuckerResults(): TuckerResult[] {
  if (tuckerCache === null) tuckerCache = computeTuckerData();
  return tuckerCache;
}

// ─── t-SVD Data ───

/** Inverse DFT along the third mode. Takes [real, imag], returns real tensor. */
function idft3(real: number[][][], imag: number[][][]): number[][][] {
  const I1 = real.length, I2 = real[0].length, I3 = real[0][0].length;
  const out = zeros3D(I1, I2, I3);

  for (let i = 0; i < I1; i++)
    for (let j = 0; j < I2; j++)
      for (let k = 0; k < I3; k++) {
        let val = 0;
        for (let f = 0; f < I3; f++) {
          const angle = (2 * Math.PI * f * k) / I3;
          val += real[i][j][f] * Math.cos(angle) - imag[i][j][f] * Math.sin(angle);
        }
        out[i][j][k] = val / I3;
      }
  return out;
}

/** Forward DFT along the third mode of a real tensor. Returns [real, imag]. */
function dft3(T: number[][][]): { re: number[][][]; im: number[][][] } {
  const I1 = T.length, I2 = T[0].length, I3 = T[0][0].length;
  const re = zeros3D(I1, I2, I3);
  const im = zeros3D(I1, I2, I3);
  for (let i = 0; i < I1; i++)
    for (let j = 0; j < I2; j++)
      for (let f = 0; f < I3; f++) {
        let a = 0, b = 0;
        for (let k = 0; k < I3; k++) {
          const angle = (2 * Math.PI * f * k) / I3;
          a += T[i][j][k] * Math.cos(angle);
          b -= T[i][j][k] * Math.sin(angle);
        }
        re[i][j][f] = a;
        im[i][j][f] = b;
      }
  return { re, im };
}

/**
 * Eigendecomposition of a small symmetric matrix by the cyclic Jacobi method.
 * Returns eigenvalues in descending order with matching eigenvector columns.
 */
function symmetricEigen(G: number[][]): { values: number[]; vectors: number[][] } {
  const n = G.length;
  const A = G.map((row) => [...row]);
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  let scale = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) scale += A[i][j] * A[i][j];

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off <= 1e-30 * scale) break;
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
  return {
    values: order.map((i) => A[i][i]),
    vectors: V.map((row) => order.map((i) => row[i])),
  };
}

/**
 * SVD of a complex matrix M = Re + i·Im through its real embedding
 * E = [[Re, −Im], [Im, Re]] (2m × 2n). Every singular value σ of M appears
 * twice among those of E, so M's singular values are every other one of E's,
 * and the rank-k truncation of M is the rank-2k truncation of E.
 */
function complexSVD(Re: number[][], Im: number[][]) {
  const m = Re.length, n = Re[0].length;
  const E = Array.from({ length: 2 * m }, (_, i) =>
    Array.from({ length: 2 * n }, (_, j) => {
      const r = i % m, c = j % n;
      if (i < m) return j < n ? Re[r][c] : -Im[r][c];
      return j < n ? Im[r][c] : Re[r][c];
    }),
  );
  // Gram matrix EᵀE (2n × 2n); its eigenvectors are E's right singular vectors.
  const G = Array.from({ length: 2 * n }, () => new Array<number>(2 * n).fill(0));
  for (let i = 0; i < 2 * n; i++)
    for (let j = i; j < 2 * n; j++) {
      let sum = 0;
      for (let r = 0; r < 2 * m; r++) sum += E[r][i] * E[r][j];
      G[i][j] = G[j][i] = sum;
    }
  const { values, vectors } = symmetricEigen(G);
  const p = Math.min(m, n);
  const S = Array.from({ length: p }, (_, r) =>
    (Math.sqrt(Math.max(values[2 * r], 0)) + Math.sqrt(Math.max(values[2 * r + 1], 0))) / 2,
  );

  /** Best rank-k approximation of M, as [real, imag]. */
  const truncate = (k: number): { re: number[][]; im: number[][] } => {
    const cols = Math.min(2 * k, 2 * n);
    // E_k = E · V_{2k} V_{2k}ᵀ
    const EV = E.map((row) =>
      Array.from({ length: cols }, (_, c) => {
        let sum = 0;
        for (let j = 0; j < 2 * n; j++) sum += row[j] * vectors[j][c];
        return sum;
      }),
    );
    const Ek = EV.map((row) =>
      Array.from({ length: 2 * n }, (_, j) => {
        let sum = 0;
        for (let c = 0; c < cols; c++) sum += row[c] * vectors[j][c];
        return sum;
      }),
    );
    // Read M_k off the block structure (averaging the two copies of each block).
    const re = Array.from({ length: m }, (_, i) =>
      Array.from({ length: n }, (_, j) => (Ek[i][j] + Ek[m + i][n + j]) / 2),
    );
    const im = Array.from({ length: m }, (_, i) =>
      Array.from({ length: n }, (_, j) => (Ek[m + i][j] - Ek[i][n + j]) / 2),
    );
    return { re, im };
  };

  return { S, truncate };
}

function computeTSVDData(): tSVDResult[] {
  const I1 = 12, I2 = 10, I3 = 8;
  const tubalRankTrue = 4;
  const noiseLevel = 0.05; // ‖noise‖_F / ‖noise-free tensor‖_F
  const half = Math.floor(I3 / 2);

  // Build the noise-free tensor in the Fourier domain: every frontal slice
  // Â^(j) = A_j B_j has rank exactly tubalRankTrue. A real tensor needs
  // conjugate-symmetric slices, Â^(I3−j) = conj(Â^(j)), so we draw slices
  // j = 0..I3/2 and mirror the rest; the j = 0 and j = I3/2 slices are real.
  let seed = 271;
  const draw = () => seededRand(seed++) * 2 - 1;
  const fReal = zeros3D(I1, I2, I3);
  const fImag = zeros3D(I1, I2, I3);

  for (let f = 0; f <= half; f++) {
    const isReal = f === 0 || 2 * f === I3;
    const Ar = Array.from({ length: I1 }, () => Array.from({ length: tubalRankTrue }, draw));
    const Ai = Array.from({ length: I1 }, () =>
      Array.from({ length: tubalRankTrue }, () => (isReal ? 0 : draw())),
    );
    const Br = Array.from({ length: tubalRankTrue }, () => Array.from({ length: I2 }, draw));
    const Bi = Array.from({ length: tubalRankTrue }, () =>
      Array.from({ length: I2 }, () => (isReal ? 0 : draw())),
    );
    // Scale decreases with frequency for natural-looking data
    const scale = 10 / (1 + f);
    const ArBr = matMul(Ar, Br), AiBi = matMul(Ai, Bi);
    const ArBi = matMul(Ar, Bi), AiBr = matMul(Ai, Br);
    for (let i = 0; i < I1; i++)
      for (let j = 0; j < I2; j++) {
        fReal[i][j][f] = (ArBr[i][j] - AiBi[i][j]) * scale;
        fImag[i][j][f] = (ArBi[i][j] + AiBr[i][j]) * scale;
      }
  }
  for (let f = half + 1; f < I3; f++)
    for (let i = 0; i < I1; i++)
      for (let j = 0; j < I2; j++) {
        fReal[i][j][f] = fReal[i][j][I3 - f];
        fImag[i][j][f] = -fImag[i][j][I3 - f];
      }

  // Spatial-domain tensor plus a small noise floor
  const clean = idft3(fReal, fImag);
  const rawNoise = clean.map((s) => s.map((r) => r.map(() => draw())));
  const noiseScale = (noiseLevel * tensorNorm(clean)) / tensorNorm(rawNoise);
  const tensor = clean.map((s, i) =>
    s.map((r, j) => r.map((v, k) => v + noiseScale * rawNoise[i][j][k])),
  );
  const fullNorm = tensorNorm(tensor);

  // SVD of each complex Fourier slice (slices past I3/2 are conjugates of
  // earlier ones and share their singular values)
  const maxTubalRank = Math.min(I1, I2);
  const { re: tRe, im: tIm } = dft3(tensor);
  const sliceSVDs = Array.from({ length: half + 1 }, (_, f) =>
    complexSVD(
      Array.from({ length: I1 }, (_, i) => Array.from({ length: I2 }, (_, j) => tRe[i][j][f])),
      Array.from({ length: I1 }, (_, i) => Array.from({ length: I2 }, (_, j) => tIm[i][j][f])),
    ),
  );
  const allSingularValues = Array.from(
    { length: I3 },
    (_, f) => sliceSVDs[f <= half ? f : I3 - f].S,
  );

  // For each tubal rank k, truncate every Fourier slice to rank k, then IDFT back
  const results: tSVDResult[] = [];

  for (let k = 1; k <= maxTubalRank; k++) {
    const truncReal = zeros3D(I1, I2, I3);
    const truncImag = zeros3D(I1, I2, I3);

    for (let f = 0; f <= half; f++) {
      const { re, im } = sliceSVDs[f].truncate(k);
      const isReal = f === 0 || 2 * f === I3;
      for (let i = 0; i < I1; i++)
        for (let j = 0; j < I2; j++) {
          truncReal[i][j][f] = re[i][j];
          truncImag[i][j][f] = isReal ? 0 : im[i][j];
          if (f > 0 && !isReal) {
            truncReal[i][j][I3 - f] = re[i][j];
            truncImag[i][j][I3 - f] = -im[i][j];
          }
        }
    }

    const reconTensor = idft3(truncReal, truncImag);
    const errNorm = tensorNorm(tensorDiff(tensor, reconTensor));
    const relativeError = errNorm / fullNorm;

    // First frontal slice of reconstruction
    const reconSlice = reconTensor.map((row) => row.map((col) => col[0]));

    // Singular values for first 4 Fourier slices
    const fourierSVs = allSingularValues.slice(0, 4).map((sv) =>
      sv.slice(0, Math.min(k + 2, sv.length)),
    );

    results.push({
      tubalRank: k,
      relativeError,
      fourierSingularValues: fourierSVs,
      reconstructionSlice: reconSlice,
    });
  }

  return results;
}

let tsvdCache: tSVDResult[] | null = null;
export function getTSVDResults(): tSVDResult[] {
  if (tsvdCache === null) tsvdCache = computeTSVDData();
  return tsvdCache;
}

// Pre-computed tensor decomposition data for interactive visualizations.
// All tensors are constructed FROM known factors so decompositions are exact
// by construction — no iterative algorithms needed at runtime.

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

/** Simple SVD for small matrices via eigendecomposition of AᵀA (power iteration). */
function simpleSVD(
  M: number[][],
  maxRank: number,
): { U: number[][]; S: number[]; Vt: number[][] } {
  const m = M.length, n = M[0].length;
  const rank = Math.min(maxRank, m, n);

  // Compute MᵀM
  const MtM = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < m; k++) MtM[i][j] += M[k][i] * M[k][j];

  const Vt: number[][] = [];
  const S: number[] = [];
  const U: number[][] = [];

  // Deflated power iteration for each singular triplet
  const deflated = MtM.map((r) => [...r]);

  for (let r = 0; r < rank; r++) {
    // Power iteration on deflated MᵀM
    let v = new Array<number>(n).fill(0);
    // Seed with deterministic vector
    for (let i = 0; i < n; i++) v[i] = seededRand(500 + r * n + i);

    for (let iter = 0; iter < 50; iter++) {
      const next = new Array<number>(n).fill(0);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) next[i] += deflated[i][j] * v[j];
      let norm = 0;
      for (let i = 0; i < n; i++) norm += next[i] * next[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < n; i++) v[i] = next[i] / norm;
    }

    // Singular value: ||Mv||
    const Mv = new Array<number>(m).fill(0);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) Mv[i] += M[i][j] * v[j];
    let sigma = 0;
    for (let i = 0; i < m; i++) sigma += Mv[i] * Mv[i];
    sigma = Math.sqrt(sigma);

    const u = Mv.map((x) => (sigma > 1e-12 ? x / sigma : 0));

    S.push(sigma);
    Vt.push(v);
    U.push(u);

    // Deflate: remove σ² vvᵀ from MᵀM
    const s2 = sigma * sigma;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) deflated[i][j] -= s2 * v[i] * v[j];
  }

  // Transpose U from list-of-columns to m×rank
  const Umat = Array.from({ length: m }, (_, i) =>
    Array.from({ length: rank }, (_, r) => U[r][i]),
  );

  return { U: Umat, S, Vt };
}

function computeTSVDData(): tSVDResult[] {
  const I1 = 12, I2 = 10, I3 = 8;
  const tubalRankTrue = 4;

  // Build tensor in Fourier domain with known low-tubal-rank structure.
  // For each Fourier slice, create a rank-tubalRankTrue matrix.
  let seed = 271;
  const fReal = zeros3D(I1, I2, I3);
  const fImag = zeros3D(I1, I2, I3);

  for (let f = 0; f < I3; f++) {
    // Low-rank factors for this slice: A (I1×tubalRankTrue), B (tubalRankTrue×I2)
    const A: number[][] = Array.from({ length: I1 }, () => {
      const row: number[] = [];
      for (let r = 0; r < tubalRankTrue; r++) row.push(seededRand(seed++) * 2 - 1);
      return row;
    });
    const B: number[][] = Array.from({ length: tubalRankTrue }, () => {
      const row: number[] = [];
      for (let r = 0; r < I2; r++) row.push(seededRand(seed++) * 2 - 1);
      return row;
    });
    // Scale factor decreases with frequency for natural-looking data
    const scale = 10 / (1 + f);
    const slice = matMul(A, B);
    for (let i = 0; i < I1; i++)
      for (let j = 0; j < I2; j++) {
        fReal[i][j][f] = slice[i][j] * scale;
        // For real-valued tensor: enforce conjugate symmetry
        // f and I3-f slices are conjugates. We handle imaginary part for f > 0.
        if (f > 0 && f < I3 / 2) {
          fImag[i][j][f] = seededRand(seed++) * scale * 0.3;
        }
      }
  }

  // Enforce conjugate symmetry: X̂[:,:,I3-f] = conj(X̂[:,:,f])
  for (let f = 1; f < I3 / 2; f++) {
    const cf = I3 - f;
    for (let i = 0; i < I1; i++)
      for (let j = 0; j < I2; j++) {
        fReal[i][j][cf] = fReal[i][j][f];
        fImag[i][j][cf] = -fImag[i][j][f];
      }
  }
  // f=0 and f=I3/2 are real (imag = 0, already initialized to 0)

  // Convert to spatial domain
  const tensor = idft3(fReal, fImag);
  const fullNorm = tensorNorm(tensor);

  // For each Fourier slice, compute SVD and singular values
  const maxTubalRank = Math.min(I1, I2);
  const allSingularValues: number[][] = []; // [slice][index]

  // Compute SVD of each Fourier slice
  type SliceSVD = { U: number[][]; S: number[]; Vt: number[][] };
  const sliceSVDs: SliceSVD[] = [];

  for (let f = 0; f < I3; f++) {
    // Extract the f-th Fourier slice as a complex I1×I2 matrix
    // For SVD purposes, we work with the magnitude since we need real SVDs
    // But for proper t-SVD, each Fourier slice gets its own SVD
    const sliceReal: number[][] = Array.from({ length: I1 }, (_, i) =>
      Array.from({ length: I2 }, (_, j) => fReal[i][j][f]),
    );
    const svd = simpleSVD(sliceReal, maxTubalRank);
    sliceSVDs.push(svd);
    allSingularValues.push(svd.S);
  }

  // For each tubal rank k, reconstruct and compute error
  const results: tSVDResult[] = [];

  for (let k = 1; k <= maxTubalRank; k++) {
    // Truncate each Fourier slice to rank k, then IDFT back
    const truncReal = zeros3D(I1, I2, I3);
    const truncImag = zeros3D(I1, I2, I3); // imaginary stays 0 for real slices

    for (let f = 0; f < I3; f++) {
      const { U, S, Vt } = sliceSVDs[f];
      const rk = Math.min(k, S.length);
      for (let i = 0; i < I1; i++)
        for (let j = 0; j < I2; j++) {
          let val = 0;
          for (let r = 0; r < rk; r++) val += U[i][r] * S[r] * Vt[r][j];
          truncReal[i][j][f] = val;
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

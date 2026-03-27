// ============================================================================
// Graph Theory utilities for the Graph Theory track.
// Covers graph construction, Laplacian matrices, eigendecomposition,
// Cheeger constant computation, dataset generators, and clustering.
// ============================================================================

// === Types ===

export interface Graph {
  n: number;
  adjacency: number[][];
  labels?: string[];
}

export interface EigenResult {
  eigenvalues: number[];
  eigenvectors: number[][];
}

export interface GraphSpectrum {
  laplacian: number[][];
  normalizedLaplacian: number[][];
  eigen: EigenResult;
  normalizedEigen: EigenResult;
  fiedlerValue: number;
  fiedlerVector: number[];
  numComponents: number;
}

export interface CheegerResult {
  cheegerConstant: number;
  optimalPartition: [number[], number[]];
  cutEdges: [number, number][];
}

// === Seeded PRNG (LCG) ===

/** Linear congruential generator for reproducible randomness. */
export function createRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return (s >>> 0) / 4294967296;
  };
}

// === Graph Construction ===

function emptyAdjacency(n: number): number[][] {
  return Array.from({ length: n }, () => new Array(n).fill(0));
}

/** Path graph P_n. */
export function pathGraph(n: number): Graph {
  const adj = emptyAdjacency(n);
  for (let i = 0; i < n - 1; i++) {
    adj[i][i + 1] = 1;
    adj[i + 1][i] = 1;
  }
  return { n, adjacency: adj };
}

/** Cycle graph C_n. */
export function cycleGraph(n: number): Graph {
  const adj = emptyAdjacency(n);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    adj[i][j] = 1;
    adj[j][i] = 1;
  }
  return { n, adjacency: adj };
}

/** Complete graph K_n. */
export function completeGraph(n: number): Graph {
  const adj = emptyAdjacency(n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      adj[i][j] = 1;
      adj[j][i] = 1;
    }
  }
  return { n, adjacency: adj };
}

/** Star graph S_n (one hub node 0 + n-1 leaves). */
export function starGraph(n: number): Graph {
  const adj = emptyAdjacency(n);
  for (let i = 1; i < n; i++) {
    adj[0][i] = 1;
    adj[i][0] = 1;
  }
  return { n, adjacency: adj };
}

/** Barbell graph: two K_k cliques connected by a single bridge edge. */
export function barbellGraph(k: number): Graph {
  const n = 2 * k;
  const adj = emptyAdjacency(n);
  // First clique: nodes 0..k-1
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      adj[i][j] = 1;
      adj[j][i] = 1;
    }
  }
  // Second clique: nodes k..2k-1
  for (let i = k; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      adj[i][j] = 1;
      adj[j][i] = 1;
    }
  }
  // Bridge edge
  adj[k - 1][k] = 1;
  adj[k][k - 1] = 1;
  return { n, adjacency: adj };
}

/** Grid graph m × m. */
export function gridGraph(m: number): Graph {
  const n = m * m;
  const adj = emptyAdjacency(n);
  for (let r = 0; r < m; r++) {
    for (let c = 0; c < m; c++) {
      const i = r * m + c;
      if (c < m - 1) {
        adj[i][i + 1] = 1;
        adj[i + 1][i] = 1;
      }
      if (r < m - 1) {
        adj[i][i + m] = 1;
        adj[i + m][i] = 1;
      }
    }
  }
  return { n, adjacency: adj };
}

/** Petersen graph (10 nodes, 15 edges). */
export function petersenGraph(): Graph {
  const n = 10;
  const adj = emptyAdjacency(n);
  // Outer cycle: 0-1-2-3-4-0
  for (let i = 0; i < 5; i++) {
    const j = (i + 1) % 5;
    adj[i][j] = 1;
    adj[j][i] = 1;
  }
  // Inner pentagram: 5-7-9-6-8-5
  const inner = [
    [5, 7], [7, 9], [9, 6], [6, 8], [8, 5],
  ];
  for (const [a, b] of inner) {
    adj[a][b] = 1;
    adj[b][a] = 1;
  }
  // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
  for (let i = 0; i < 5; i++) {
    adj[i][i + 5] = 1;
    adj[i + 5][i] = 1;
  }
  return { n, adjacency: adj };
}

/** Erdős–Rényi random graph G(n, p) with seeded PRNG. */
export function erdosRenyiGraph(n: number, p: number, seed: number): Graph {
  const rng = createRng(seed);
  const adj = emptyAdjacency(n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (rng() < p) {
        adj[i][j] = 1;
        adj[j][i] = 1;
      }
    }
  }
  return { n, adjacency: adj };
}

/** Random d-regular graph via configuration model (approximate, for small n). */
export function randomRegularGraph(n: number, d: number, seed: number): Graph {
  const rng = createRng(seed);
  const adj = emptyAdjacency(n);

  // Attempt pairing via stubs
  for (let attempt = 0; attempt < 100; attempt++) {
    // Reset
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) adj[i][j] = 0;

    const stubs: number[] = [];
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < d; k++) stubs.push(i);
    }

    // Fisher-Yates shuffle
    for (let i = stubs.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [stubs[i], stubs[j]] = [stubs[j], stubs[i]];
    }

    let valid = true;
    for (let i = 0; i < stubs.length; i += 2) {
      const u = stubs[i];
      const v = stubs[i + 1];
      if (u === v || adj[u][v] === 1) {
        valid = false;
        break;
      }
      adj[u][v] = 1;
      adj[v][u] = 1;
    }
    if (valid) return { n, adjacency: adj };
  }

  // Fallback: return whatever we have
  return { n, adjacency: adj };
}

/** Two cliques of size k connected by a single bridge edge. */
export function twoCliquesBridge(k: number): Graph {
  return barbellGraph(k);
}

/** k-nearest-neighbor similarity graph from 2D points. */
export function knnGraph(
  points: [number, number][],
  k: number
): Graph {
  const n = points.length;
  const adj = emptyAdjacency(n);

  // Compute all pairwise distances
  const dists: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      const d = Math.sqrt(dx * dx + dy * dy);
      dists[i][j] = d;
      dists[j][i] = d;
    }
  }

  // Find median distance for Gaussian kernel bandwidth
  const allDists: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      allDists.push(dists[i][j]);
    }
  }
  allDists.sort((a, b) => a - b);
  const sigma = allDists[Math.floor(allDists.length / 2)];
  const sigma2 = 2 * sigma * sigma;

  // For each point, find k nearest neighbors and set Gaussian weight
  for (let i = 0; i < n; i++) {
    const indices = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => dists[i][a] - dists[i][b]);
    for (let ki = 0; ki < Math.min(k, indices.length); ki++) {
      const j = indices[ki];
      const w = Math.exp(-(dists[i][j] * dists[i][j]) / sigma2);
      // Symmetrize
      adj[i][j] = Math.max(adj[i][j], w);
      adj[j][i] = Math.max(adj[j][i], w);
    }
  }

  return { n, adjacency: adj };
}

/** ε-ball similarity graph from 2D points. */
export function epsilonBallGraph(
  points: [number, number][],
  epsilon: number
): Graph {
  const n = points.length;
  const adj = emptyAdjacency(n);

  // Compute median distance for Gaussian kernel
  const allDists: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      allDists.push(Math.sqrt(dx * dx + dy * dy));
    }
  }
  allDists.sort((a, b) => a - b);
  const sigma = allDists[Math.floor(allDists.length / 2)];
  const sigma2 = 2 * sigma * sigma;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d <= epsilon) {
        const w = Math.exp(-(d * d) / sigma2);
        adj[i][j] = w;
        adj[j][i] = w;
      }
    }
  }

  return { n, adjacency: adj };
}

// === Matrix Operations ===

/** Degree matrix D = diag(row sums of A). Returns full n×n diagonal matrix. */
export function degreeMatrix(A: number[][]): number[][] {
  const n = A.length;
  const D = emptyAdjacency(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += A[i][j];
    D[i][i] = sum;
  }
  return D;
}

/** Degree vector (just the diagonal of D). */
export function degrees(A: number[][]): number[] {
  const n = A.length;
  return Array.from({ length: n }, (_, i) => {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += A[i][j];
    return sum;
  });
}

/** Unnormalized Laplacian L = D - A. */
export function laplacian(A: number[][]): number[][] {
  const n = A.length;
  const L = emptyAdjacency(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += A[i][j];
      if (i !== j) L[i][j] = -A[i][j];
    }
    L[i][i] = sum;
  }
  return L;
}

/** Normalized Laplacian L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}. */
export function normalizedLaplacian(A: number[][]): number[][] {
  const n = A.length;
  const deg = degrees(A);
  const L = emptyAdjacency(n);
  for (let i = 0; i < n; i++) {
    if (deg[i] === 0) continue;
    L[i][i] = 1;
    for (let j = 0; j < n; j++) {
      if (i !== j && A[i][j] !== 0 && deg[j] > 0) {
        L[i][j] = -A[i][j] / Math.sqrt(deg[i] * deg[j]);
      }
    }
  }
  return L;
}

/** Random walk Laplacian L_rw = D^{-1} L = I - D^{-1} A. */
export function randomWalkLaplacian(A: number[][]): number[][] {
  const n = A.length;
  const deg = degrees(A);
  const L = emptyAdjacency(n);
  for (let i = 0; i < n; i++) {
    if (deg[i] === 0) continue;
    L[i][i] = 1;
    for (let j = 0; j < n; j++) {
      if (i !== j && A[i][j] !== 0) {
        L[i][j] = -A[i][j] / deg[i];
      }
    }
  }
  return L;
}

/** Laplacian quadratic form x^T L x. */
export function quadraticForm(L: number[][], x: number[]): number {
  let sum = 0;
  const n = x.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      sum += x[i] * L[i][j] * x[j];
    }
  }
  return sum;
}

// === Eigendecomposition ===

/**
 * Jacobi eigenvalue algorithm for real symmetric matrices.
 * Returns eigenvalues sorted ascending and corresponding eigenvectors (columns).
 * Suitable for matrices up to ~150×150 in real time.
 */
export function jacobiEigen(
  M: number[][],
  maxIter: number = 200,
  tol: number = 1e-10
): EigenResult {
  const n = M.length;

  // Work on a copy
  const A: number[][] = M.map((row) => [...row]);

  // Initialize eigenvector matrix as identity
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  for (let iter = 0; iter < maxIter * n; iter++) {
    // Find the largest off-diagonal element
    let maxVal = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const absVal = Math.abs(A[i][j]);
        if (absVal > maxVal) {
          maxVal = absVal;
          p = i;
          q = j;
        }
      }
    }

    if (maxVal < tol) break;

    // Compute rotation angle
    const apq = A[p][q];
    const diff = A[q][q] - A[p][p];
    let t: number;
    if (Math.abs(diff) < tol) {
      t = apq > 0 ? 1 : -1;
    } else {
      const phi = diff / (2 * apq);
      t = 1 / (Math.abs(phi) + Math.sqrt(phi * phi + 1));
      if (phi < 0) t = -t;
    }

    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    const tau = s / (1 + c);

    // Update matrix A
    const app = A[p][p];
    const aqq = A[q][q];
    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let r = 0; r < n; r++) {
      if (r === p || r === q) continue;
      const arp = A[r][p];
      const arq = A[r][q];
      A[r][p] = arp - s * (arq + tau * arp);
      A[p][r] = A[r][p];
      A[r][q] = arq + s * (arp - tau * arq);
      A[q][r] = A[r][q];
    }

    // Update eigenvectors
    for (let r = 0; r < n; r++) {
      const vrp = V[r][p];
      const vrq = V[r][q];
      V[r][p] = vrp - s * (vrq + tau * vrp);
      V[r][q] = vrq + s * (vrp - tau * vrq);
    }
  }

  // Extract eigenvalues and sort ascending
  const eigenvalues = Array.from({ length: n }, (_, i) => A[i][i]);
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => eigenvalues[a] - eigenvalues[b]);

  const sortedEigenvalues = indices.map((i) => eigenvalues[i]);
  const sortedEigenvectors: number[][] = indices.map((i) =>
    Array.from({ length: n }, (_, r) => V[r][i])
  );

  return { eigenvalues: sortedEigenvalues, eigenvectors: sortedEigenvectors };
}

// === Spectral Analysis ===

/** Number of connected components (count eigenvalues ≈ 0). */
export function countComponents(eigenvalues: number[], tol: number = 1e-6): number {
  let count = 0;
  for (const ev of eigenvalues) {
    if (Math.abs(ev) < tol) count++;
  }
  return Math.max(1, count);
}

/** Full spectral analysis of a graph. */
export function analyzeSpectrum(graph: Graph): GraphSpectrum {
  const L = laplacian(graph.adjacency);
  const Lnorm = normalizedLaplacian(graph.adjacency);
  const eigen = jacobiEigen(L);
  const normalizedEigen = jacobiEigen(Lnorm);

  const numComponents = countComponents(eigen.eigenvalues);

  // Fiedler value and vector (second smallest eigenvalue/eigenvector)
  const fiedlerValue = graph.n > 1 ? eigen.eigenvalues[1] : 0;
  const fiedlerVector = graph.n > 1 ? eigen.eigenvectors[1] : [0];

  return {
    laplacian: L,
    normalizedLaplacian: Lnorm,
    eigen,
    normalizedEigen,
    fiedlerValue,
    fiedlerVector,
    numComponents,
  };
}

// === Cheeger Constant ===

/**
 * Compute the Cheeger constant h(G) for the normalized Laplacian by enumeration.
 * For small graphs (n ≤ 14), enumerates all subsets S with |S| ≤ n/2.
 * For larger graphs, uses the Fiedler vector partition as an approximation.
 */
export function cheegerConstant(graph: Graph): CheegerResult {
  const { n, adjacency: A } = graph;
  const deg = degrees(A);
  const totalVol = deg.reduce((s, d) => s + d, 0);

  if (n === 0 || totalVol === 0) {
    return {
      cheegerConstant: 0,
      optimalPartition: [[], []],
      cutEdges: [],
    };
  }

  // For large graphs, use Fiedler vector approximation
  if (n > 14) {
    return cheegerFiedlerApprox(graph);
  }

  // Enumerate all non-empty subsets S with |S| ≤ n/2
  let bestH = Infinity;
  let bestS: number[] = [];
  const totalSubsets = 1 << n;

  for (let mask = 1; mask < totalSubsets; mask++) {
    const S: number[] = [];
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) S.push(i);
    }
    if (S.length > n / 2) continue;

    // Compute vol(S)
    let volS = 0;
    for (const i of S) volS += deg[i];
    if (volS === 0) continue;

    // Compute cut(S, S̄)
    let cutSize = 0;
    for (const i of S) {
      for (let j = 0; j < n; j++) {
        if (!(mask & (1 << j)) && A[i][j] > 0) {
          cutSize += A[i][j];
        }
      }
    }

    const volSbar = totalVol - volS;
    const minVol = Math.min(volS, volSbar);
    if (minVol === 0) continue;

    const h = cutSize / minVol;
    if (h < bestH) {
      bestH = h;
      bestS = S;
    }
  }

  const Sbar = Array.from({ length: n }, (_, i) => i).filter(
    (i) => !bestS.includes(i)
  );

  // Find cut edges
  const cutEdges: [number, number][] = [];
  for (const i of bestS) {
    for (const j of Sbar) {
      if (A[i][j] > 0) cutEdges.push([i, j]);
    }
  }

  return {
    cheegerConstant: bestH === Infinity ? 0 : bestH,
    optimalPartition: [bestS, Sbar],
    cutEdges,
  };
}

/** Approximate Cheeger constant using the Fiedler vector threshold cut. */
function cheegerFiedlerApprox(graph: Graph): CheegerResult {
  const { n, adjacency: A } = graph;
  const deg = degrees(A);
  const totalVol = deg.reduce((s, d) => s + d, 0);

  const spectrum = analyzeSpectrum(graph);
  const fiedler = spectrum.fiedlerVector;

  // Sort vertices by Fiedler vector value
  const sorted = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => fiedler[a] - fiedler[b]
  );

  // Try all threshold cuts and pick the best
  let bestH = Infinity;
  let bestK = 1;

  for (let k = 1; k < n; k++) {
    const S = sorted.slice(0, k);
    const SSet = new Set(S);

    let volS = 0;
    for (const i of S) volS += deg[i];
    if (volS === 0) continue;

    let cutSize = 0;
    for (const i of S) {
      for (let j = 0; j < n; j++) {
        if (!SSet.has(j) && A[i][j] > 0) cutSize += A[i][j];
      }
    }

    const volSbar = totalVol - volS;
    const minVol = Math.min(volS, volSbar);
    if (minVol === 0) continue;

    const h = cutSize / minVol;
    if (h < bestH) {
      bestH = h;
      bestK = k;
    }
  }

  const bestS = sorted.slice(0, bestK);
  const Sbar = sorted.slice(bestK);

  const SSet = new Set(bestS);
  const cutEdges: [number, number][] = [];
  for (const i of bestS) {
    for (const j of Sbar) {
      if (A[i][j] > 0) cutEdges.push([i, j]);
    }
  }

  return {
    cheegerConstant: bestH === Infinity ? 0 : bestH,
    optimalPartition: [bestS, Sbar],
    cutEdges,
  };
}

/** Compute the Fiedler vector partition (sign-based split). */
export function fiedlerPartition(graph: Graph): CheegerResult {
  const { n, adjacency: A } = graph;
  const spectrum = analyzeSpectrum(graph);
  const fiedler = spectrum.fiedlerVector;

  const S: number[] = [];
  const Sbar: number[] = [];
  for (let i = 0; i < n; i++) {
    if (fiedler[i] >= 0) {
      S.push(i);
    } else {
      Sbar.push(i);
    }
  }

  // Compute Cheeger ratio for this partition
  const deg = degrees(A);
  const totalVol = deg.reduce((s, d) => s + d, 0);
  let volS = 0;
  for (const i of S) volS += deg[i];
  let cutSize = 0;
  const SSet = new Set(S);
  for (const i of S) {
    for (const j of Sbar) {
      if (A[i][j] > 0) cutSize += A[i][j];
    }
  }
  const minVol = Math.min(volS, totalVol - volS);

  const cutEdges: [number, number][] = [];
  for (const i of S) {
    for (const j of Sbar) {
      if (A[i][j] > 0) cutEdges.push([i, j]);
    }
  }

  return {
    cheegerConstant: minVol > 0 ? cutSize / minVol : 0,
    optimalPartition: [S, Sbar],
    cutEdges,
  };
}

// === Dataset Generators ===

/** Two moons dataset with adjustable noise. */
export function twoMoons(
  n: number,
  noise: number,
  seed: number
): [number, number][] {
  const rng = createRng(seed);
  const half = Math.floor(n / 2);
  const points: [number, number][] = [];

  for (let i = 0; i < half; i++) {
    const angle = (Math.PI * i) / half;
    const x = Math.cos(angle) + noise * (rng() - 0.5);
    const y = Math.sin(angle) + noise * (rng() - 0.5);
    points.push([x, y]);
  }
  for (let i = 0; i < n - half; i++) {
    const angle = (Math.PI * i) / (n - half);
    const x = 1 - Math.cos(angle) + noise * (rng() - 0.5);
    const y = 1 - Math.sin(angle) - 0.5 + noise * (rng() - 0.5);
    points.push([x, y]);
  }

  return points;
}

/** Two concentric circles with adjustable noise. */
export function twoCircles(
  n: number,
  noise: number,
  seed: number
): [number, number][] {
  const rng = createRng(seed);
  const half = Math.floor(n / 2);
  const points: [number, number][] = [];

  // Outer circle (radius 1)
  for (let i = 0; i < half; i++) {
    const angle = (2 * Math.PI * i) / half;
    const x = Math.cos(angle) + noise * (rng() - 0.5);
    const y = Math.sin(angle) + noise * (rng() - 0.5);
    points.push([x, y]);
  }
  // Inner circle (radius 0.4)
  for (let i = 0; i < n - half; i++) {
    const angle = (2 * Math.PI * i) / (n - half);
    const x = 0.4 * Math.cos(angle) + noise * (rng() - 0.5);
    const y = 0.4 * Math.sin(angle) + noise * (rng() - 0.5);
    points.push([x, y]);
  }

  return points;
}

/** Three Gaussian blobs. */
export function threeBlobs(
  n: number,
  noise: number,
  seed: number
): [number, number][] {
  const rng = createRng(seed);
  const points: [number, number][] = [];
  const centers: [number, number][] = [
    [-1, -1],
    [1, -1],
    [0, 1],
  ];
  const perCluster = Math.floor(n / 3);

  // Box-Muller transform for Gaussian samples
  function gaussPair(): [number, number] {
    const u1 = rng();
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10)));
    return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
  }

  for (let c = 0; c < 3; c++) {
    const count = c < 2 ? perCluster : n - 2 * perCluster;
    for (let i = 0; i < count; i++) {
      const [g1, g2] = gaussPair();
      points.push([
        centers[c][0] + noise * 0.5 * g1,
        centers[c][1] + noise * 0.5 * g2,
      ]);
    }
  }

  return points;
}

/** Spiral dataset with two interleaving arms. */
export function spiral(
  n: number,
  noise: number,
  seed: number
): [number, number][] {
  const rng = createRng(seed);
  const half = Math.floor(n / 2);
  const points: [number, number][] = [];

  for (let i = 0; i < half; i++) {
    const t = (3 * Math.PI * i) / half;
    const r = t / (3 * Math.PI);
    const x = r * Math.cos(t) + noise * 0.1 * (rng() - 0.5);
    const y = r * Math.sin(t) + noise * 0.1 * (rng() - 0.5);
    points.push([x, y]);
  }
  for (let i = 0; i < n - half; i++) {
    const t = (3 * Math.PI * i) / (n - half);
    const r = t / (3 * Math.PI);
    const x = -r * Math.cos(t) + noise * 0.1 * (rng() - 0.5);
    const y = -r * Math.sin(t) + noise * 0.1 * (rng() - 0.5);
    points.push([x, y]);
  }

  return points;
}

// === Clustering ===

/** k-means clustering with k-means++ initialization.
 *  @param points - each row is a data point
 *  @param k - number of clusters
 *  @returns cluster assignment array */
export function kMeans(
  points: number[][],
  k: number,
  maxIter: number = 20,
  seed: number = 42
): number[] {
  const rng = createRng(seed);
  const n = points.length;

  if (n === 0 || k <= 0) return [];
  if (k >= n) return Array.from({ length: n }, (_, i) => i);

  const dim = points[0].length;

  // k-means++ initialization
  const centroids: number[][] = [];
  const firstIdx = Math.floor(rng() * n);
  centroids.push([...points[firstIdx]]);

  for (let c = 1; c < k; c++) {
    const dists = points.map((p) => {
      let minD = Infinity;
      for (const cent of centroids) {
        let d = 0;
        for (let d2 = 0; d2 < dim; d2++) d += (p[d2] - cent[d2]) ** 2;
        minD = Math.min(minD, d);
      }
      return minD;
    });
    const totalDist = dists.reduce((s, d) => s + d, 0);
    let r = rng() * totalDist;
    let idx = 0;
    for (let i = 0; i < n; i++) {
      r -= dists[i];
      if (r <= 0) {
        idx = i;
        break;
      }
    }
    centroids.push([...points[idx]]);
  }

  // Lloyd's iterations
  let assignments = new Array(n).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign
    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < k; c++) {
        let d = 0;
        for (let d2 = 0; d2 < dim; d2++) {
          d += (points[i][d2] - centroids[c][d2]) ** 2;
        }
        if (d < bestDist) {
          bestDist = d;
          bestC = c;
        }
      }
      assignments[i] = bestC;
    }

    // Update centroids
    const counts = new Array(k).fill(0);
    const sums: number[][] = Array.from({ length: k }, () =>
      new Array(dim).fill(0)
    );
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let d = 0; d < dim; d++) sums[c][d] += points[i][d];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (let d = 0; d < dim; d++) {
          centroids[c][d] = sums[c][d] / counts[c];
        }
      }
    }
  }

  return assignments;
}

// === Graph Utility Helpers ===

/** Check if a graph is connected (via BFS). */
export function isConnected(graph: Graph): boolean {
  if (graph.n <= 1) return true;
  const visited = new Set<number>();
  const queue = [0];
  visited.add(0);
  while (queue.length > 0) {
    const u = queue.shift()!;
    for (let v = 0; v < graph.n; v++) {
      if (graph.adjacency[u][v] > 0 && !visited.has(v)) {
        visited.add(v);
        queue.push(v);
      }
    }
  }
  return visited.size === graph.n;
}

/** List all edges as [source, target] pairs (each edge once, source < target). */
export function getEdges(graph: Graph): [number, number][] {
  const edges: [number, number][] = [];
  for (let i = 0; i < graph.n; i++) {
    for (let j = i + 1; j < graph.n; j++) {
      if (graph.adjacency[i][j] > 0) {
        edges.push([i, j]);
      }
    }
  }
  return edges;
}

/** Check if removing an edge disconnects the graph (bridge detection).
 *  Pure function — operates on a cloned adjacency matrix to avoid mutating the input. */
export function isBridge(graph: Graph, u: number, v: number): boolean {
  if (graph.adjacency[u][v] === 0) return false;
  const clonedAdj = graph.adjacency.map((row) => row.slice());
  clonedAdj[u][v] = 0;
  clonedAdj[v][u] = 0;
  return !isConnected({ n: graph.n, adjacency: clonedAdj, labels: graph.labels });
}

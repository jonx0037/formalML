// =============================================================================
// clustering-data.ts
//
// Shared data generators, helpers, and palette for the clustering topic's
// viz components. Mulberry32 + Box-Muller drop-ins for sklearn.datasets.
// =============================================================================

export type Pt = [number, number];

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianRng(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

export interface Dataset {
  X: Pt[];
  y: number[];
  label: string;
  xDomain: [number, number];
  yDomain: [number, number];
}

export function makeMoons(n = 200, noise = 0.1, seed = 42): Dataset {
  const rng = mulberry32(seed);
  const g = gaussianRng(rng);
  const X: Pt[] = [];
  const y: number[] = [];
  const nU = Math.floor(n / 2);
  for (let i = 0; i < nU; i++) {
    const th = (Math.PI * i) / Math.max(1, nU - 1);
    X.push([Math.cos(th) + noise * g(), Math.sin(th) + noise * g()]);
    y.push(0);
  }
  for (let i = 0; i < n - nU; i++) {
    const th = (Math.PI * i) / Math.max(1, n - nU - 1);
    X.push([1 - Math.cos(th) + noise * g(), -Math.sin(th) + 0.5 + noise * g()]);
    y.push(1);
  }
  return { X, y, label: 'moons', xDomain: [-1.5, 2.6], yDomain: [-0.8, 1.4] };
}

export function makeBlobs(
  centers: Pt[],
  stds: number[],
  nPer: number[],
  seed = 42,
  xDomain: [number, number] = [-5, 5],
  yDomain: [number, number] = [-3, 5],
  label = 'blobs',
): Dataset {
  const rng = mulberry32(seed);
  const g = gaussianRng(rng);
  const X: Pt[] = [];
  const y: number[] = [];
  for (let k = 0; k < centers.length; k++) {
    for (let i = 0; i < nPer[k]; i++) {
      X.push([centers[k][0] + stds[k] * g(), centers[k][1] + stds[k] * g()]);
      y.push(k);
    }
  }
  return { X, y, label, xDomain, yDomain };
}

export function makeCircles(n = 200, factor = 0.5, noise = 0.05, seed = 42): Dataset {
  const rng = mulberry32(seed);
  const g = gaussianRng(rng);
  const X: Pt[] = [];
  const y: number[] = [];
  const nOuter = Math.floor(n / 2);
  for (let i = 0; i < nOuter; i++) {
    const th = (2 * Math.PI * i) / nOuter;
    X.push([Math.cos(th) + noise * g(), Math.sin(th) + noise * g()]);
    y.push(0);
  }
  for (let i = 0; i < n - nOuter; i++) {
    const th = (2 * Math.PI * i) / (n - nOuter);
    X.push([factor * Math.cos(th) + noise * g(), factor * Math.sin(th) + noise * g()]);
    y.push(1);
  }
  return { X, y, label: 'circles', xDomain: [-1.3, 1.3], yDomain: [-1.3, 1.3] };
}

export const BLOB3 = (): Dataset => makeBlobs(
  [[-3, 0], [0, 2.5], [3, -0.5]],
  [0.4, 0.6, 0.8],
  [134, 133, 133],
  42,
  [-5, 5],
  [-3, 5],
  '3-blob',
);

export const BLOB4 = (): Dataset => makeBlobs(
  [[-7, -5], [-2, 6], [3, -4], [8, 7]],
  [0.5, 1.0, 1.5, 2.0],
  [100, 100, 100, 100],
  42,
  [-12, 14],
  [-9, 12],
  '4-blob',
);

// -----------------------------------------------------------------------------
// Cluster-quality metrics (permutation-invariant)
// -----------------------------------------------------------------------------

export function ari(yTrue: number[], yPred: number[]): number {
  const n = yTrue.length;
  const rows = new Map<number, Map<number, number>>();
  const rowSums = new Map<number, number>();
  const colSums = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const r = yTrue[i], c = yPred[i];
    let inner = rows.get(r);
    if (!inner) { inner = new Map(); rows.set(r, inner); }
    inner.set(c, (inner.get(c) || 0) + 1);
    rowSums.set(r, (rowSums.get(r) || 0) + 1);
    colSums.set(c, (colSums.get(c) || 0) + 1);
  }
  const c2 = (k: number) => (k * (k - 1)) / 2;
  let sIJ = 0;
  for (const m of rows.values()) for (const v of m.values()) sIJ += c2(v);
  let sR = 0; for (const v of rowSums.values()) sR += c2(v);
  let sC = 0; for (const v of colSums.values()) sC += c2(v);
  const t = c2(n);
  const exp = (sR * sC) / t;
  const max = (sR + sC) / 2;
  if (max - exp === 0) return 1;
  return (sIJ - exp) / (max - exp);
}

export function silhouette(X: Pt[], labels: number[]): number {
  const n = X.length;
  let total = 0;
  let scored = 0;
  for (let i = 0; i < n; i++) {
    const li = labels[i];
    const byCluster = new Map<number, number[]>();
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = X[i][0] - X[j][0];
      const dy = X[i][1] - X[j][1];
      const d = Math.sqrt(dx * dx + dy * dy);
      let arr = byCluster.get(labels[j]);
      if (!arr) { arr = []; byCluster.set(labels[j], arr); }
      arr.push(d);
    }
    const aArr = byCluster.get(li);
    if (!aArr || aArr.length === 0) continue;
    const a = aArr.reduce((s, x) => s + x, 0) / aArr.length;
    let bMin = Infinity;
    for (const [lab, arr] of byCluster) {
      if (lab === li) continue;
      const mean = arr.reduce((s, x) => s + x, 0) / arr.length;
      if (mean < bMin) bMin = mean;
    }
    if (!Number.isFinite(bMin)) continue;
    total += (bMin - a) / Math.max(a, bMin);
    scored++;
  }
  return scored > 0 ? total / scored : 0;
}

// -----------------------------------------------------------------------------
// K-means with k-means++ initialization
// -----------------------------------------------------------------------------

export function kmeans(X: Pt[], K: number, seed = 17, maxIter = 100): { labels: number[]; centers: Pt[] } {
  const rng = mulberry32(seed);
  const n = X.length;
  const centers: Pt[] = [[X[Math.floor(rng() * n)][0], X[Math.floor(rng() * n)][1]]];
  const dSq = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const dx = X[j][0] - centers[0][0];
    const dy = X[j][1] - centers[0][1];
    dSq[j] = dx * dx + dy * dy;
  }
  while (centers.length < K) {
    let total = 0;
    for (let j = 0; j < n; j++) total += dSq[j];
    let target = rng() * total;
    let pick = n - 1;
    for (let j = 0; j < n; j++) {
      target -= dSq[j];
      if (target <= 0) { pick = j; break; }
    }
    const c: Pt = [X[pick][0], X[pick][1]];
    centers.push(c);
    for (let j = 0; j < n; j++) {
      const dx = X[j][0] - c[0];
      const dy = X[j][1] - c[1];
      const s = dx * dx + dy * dy;
      if (s < dSq[j]) dSq[j] = s;
    }
  }
  const labels = new Array<number>(n).fill(0);
  for (let it = 0; it < maxIter; it++) {
    let changed = false;
    for (let j = 0; j < n; j++) {
      let best = 0;
      let bestD = Infinity;
      for (let m = 0; m < K; m++) {
        const dx = X[j][0] - centers[m][0];
        const dy = X[j][1] - centers[m][1];
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = m; }
      }
      if (labels[j] !== best) { labels[j] = best; changed = true; }
    }
    if (!changed) break;
    for (let m = 0; m < K; m++) centers[m] = [0, 0];
    const counts = new Array<number>(K).fill(0);
    for (let j = 0; j < n; j++) {
      counts[labels[j]]++;
      centers[labels[j]][0] += X[j][0];
      centers[labels[j]][1] += X[j][1];
    }
    for (let m = 0; m < K; m++) if (counts[m] > 0) {
      centers[m][0] /= counts[m];
      centers[m][1] /= counts[m];
    }
  }
  return { labels, centers };
}

// -----------------------------------------------------------------------------
// Cluster color palette
// -----------------------------------------------------------------------------

export const CLUSTER_COLORS = ['#0F6E56', '#534AB7', '#D97706', '#0EA5E9', '#DC2626', '#7C3AED', '#059669', '#7C2D12'];

// -----------------------------------------------------------------------------
// Gaussian KDE evaluation for contour rendering
// -----------------------------------------------------------------------------

export function gaussianKdeGrid(
  X: Pt[],
  h: number,
  xLo: number, xHi: number, nx: number,
  yLo: number, yHi: number, ny: number,
): { values: number[][]; xVals: number[]; yVals: number[] } {
  const n = X.length;
  const norm = 1 / (n * 2 * Math.PI * h * h);
  const h2 = h * h;
  const xVals: number[] = [];
  for (let c = 0; c < nx; c++) xVals.push(xLo + ((xHi - xLo) * c) / (nx - 1));
  const yVals: number[] = [];
  for (let r = 0; r < ny; r++) yVals.push(yLo + ((yHi - yLo) * r) / (ny - 1));

  const values: number[][] = [];
  for (let r = 0; r < ny; r++) {
    const row: number[] = [];
    const qy = yVals[r];
    for (let c = 0; c < nx; c++) {
      const qx = xVals[c];
      let s = 0;
      for (let i = 0; i < n; i++) {
        const dx = qx - X[i][0];
        const dy = qy - X[i][1];
        s += Math.exp(-0.5 * (dx * dx + dy * dy) / h2);
      }
      row.push(norm * s);
    }
    values.push(row);
  }
  return { values, xVals, yVals };
}

// -----------------------------------------------------------------------------
// Mode locator via max-filter on a grid (brute-force, used for §2.4 reference)
// -----------------------------------------------------------------------------

export function findGridModes(
  grid: number[][], xVals: number[], yVals: number[],
  threshold = 0,
): Pt[] {
  const ny = grid.length;
  const nx = grid[0].length;
  const out: Pt[] = [];
  for (let r = 1; r < ny - 1; r++) {
    for (let c = 1; c < nx - 1; c++) {
      const v = grid[r][c];
      if (v < threshold) continue;
      let isMax = true;
      for (let dr = -1; dr <= 1 && isMax; dr++) {
        for (let dc = -1; dc <= 1 && isMax; dc++) {
          if (dr === 0 && dc === 0) continue;
          if (grid[r + dr][c + dc] > v) isMax = false;
        }
      }
      if (isMax) out.push([xVals[c], yVals[r]]);
    }
  }
  return out;
}

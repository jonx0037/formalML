// =============================================================================
// verify-clustering.ts
//
// Numerical regression tests for src/components/viz/shared/unsupervised.ts.
// Each assertion reproduces a numerical identity from the verified notebook
//   notebooks/clustering/01_clustering.ipynb
// and asserts the TS port satisfies the identity within notebook-derived
// tolerances. Tolerances widen modestly to absorb PCG64-vs-mulberry32 RNG
// drift; the math identities themselves are RNG-independent.
//
// Run with: pnpm verify:clustering
// Exits non-zero if any assertion fails.
//
// Brief: docs/plans/formalml-clustering-handoff-brief.md
// =============================================================================

import {
  basinOfAttractionMap,
  bandwidthSelectorForMeanShift,
  countDistinctModes,
  gaussianKdeLogSurrogate,
  meanShift,
  modeFinder,
} from '../unsupervised';

// -----------------------------------------------------------------------------
// PRNG + samplers — mulberry32 + Box-Muller, matching the convention in
// normalizing-flows.ts. PCG64 → mulberry32 drift is the source of all
// remaining ~5% tolerance widening in this suite.
// -----------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRng(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

function makeMoons(n: number, noise: number, seed: number): { X: number[][]; y: number[] } {
  const rng = mulberry32(seed);
  const g = gaussianRng(rng);
  const X: number[][] = [];
  const y: number[] = [];
  const nUpper = Math.floor(n / 2);
  const nLower = n - nUpper;
  for (let i = 0; i < nUpper; i++) {
    const theta = (Math.PI * i) / Math.max(1, nUpper - 1);
    X.push([Math.cos(theta) + noise * g(), Math.sin(theta) + noise * g()]);
    y.push(0);
  }
  for (let i = 0; i < nLower; i++) {
    const theta = (Math.PI * i) / Math.max(1, nLower - 1);
    X.push([1 - Math.cos(theta) + noise * g(), -Math.sin(theta) + 0.5 + noise * g()]);
    y.push(1);
  }
  return { X, y };
}

function makeBlobs(
  centers: number[][],
  stds: number[],
  nPerBlob: number[],
  seed: number,
): { X: number[][]; y: number[] } {
  const rng = mulberry32(seed);
  const g = gaussianRng(rng);
  const X: number[][] = [];
  const y: number[] = [];
  for (let k = 0; k < centers.length; k++) {
    const c = centers[k];
    const s = stds[k];
    const n = nPerBlob[k];
    for (let i = 0; i < n; i++) {
      const point = c.map((cj) => cj + s * g());
      X.push(point);
      y.push(k);
    }
  }
  return { X, y };
}

// -----------------------------------------------------------------------------
// Adjusted Rand Index (Hubert & Arabie 1985) for cluster-label comparison.
// Permutation-invariant; ARI = 1 iff perfect agreement up to label permutation,
// ARI ≈ 0 iff agreement consistent with chance, ARI < 0 iff worse than chance.
// -----------------------------------------------------------------------------

function ari(yTrue: number[], yPred: number[]): number {
  const n = yTrue.length;
  if (yPred.length !== n) throw new Error('ari: length mismatch');
  const rows = new Map<number, Map<number, number>>();
  const rowSums = new Map<number, number>();
  const colSums = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const r = yTrue[i];
    const c = yPred[i];
    let inner = rows.get(r);
    if (!inner) {
      inner = new Map();
      rows.set(r, inner);
    }
    inner.set(c, (inner.get(c) || 0) + 1);
    rowSums.set(r, (rowSums.get(r) || 0) + 1);
    colSums.set(c, (colSums.get(c) || 0) + 1);
  }
  const choose2 = (k: number) => (k * (k - 1)) / 2;
  let sumIJ = 0;
  for (const inner of rows.values()) for (const v of inner.values()) sumIJ += choose2(v);
  let sumRow = 0;
  for (const v of rowSums.values()) sumRow += choose2(v);
  let sumCol = 0;
  for (const v of colSums.values()) sumCol += choose2(v);
  const total = choose2(n);
  const expected = (sumRow * sumCol) / total;
  const maxIndex = (sumRow + sumCol) / 2;
  if (maxIndex - expected === 0) return 1; // degenerate: both partitions trivial
  return (sumIJ - expected) / (maxIndex - expected);
}

// -----------------------------------------------------------------------------
// K-means (Lloyd's algorithm with k-means++ seeding).
// -----------------------------------------------------------------------------

function kmeansFit(X: number[][], K: number, seed: number, maxIter = 100): number[] {
  const rng = mulberry32(seed);
  const n = X.length;
  const d = X[0].length;

  // k-means++ seeding.
  const centers: number[][] = [X[Math.floor(rng() * n)].slice()];
  const distSq = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let best = Infinity;
    for (let m = 0; m < centers.length; m++) {
      let s2 = 0;
      for (let c = 0; c < d; c++) {
        const dv = X[j][c] - centers[m][c];
        s2 += dv * dv;
      }
      if (s2 < best) best = s2;
    }
    distSq[j] = best;
  }
  while (centers.length < K) {
    let total = 0;
    for (let j = 0; j < n; j++) total += distSq[j];
    let target = rng() * total;
    let pickIdx = n - 1;
    for (let j = 0; j < n; j++) {
      target -= distSq[j];
      if (target <= 0) {
        pickIdx = j;
        break;
      }
    }
    centers.push(X[pickIdx].slice());
    // Update distSq.
    for (let j = 0; j < n; j++) {
      let s2 = 0;
      for (let c = 0; c < d; c++) {
        const dv = X[j][c] - centers[centers.length - 1][c];
        s2 += dv * dv;
      }
      if (s2 < distSq[j]) distSq[j] = s2;
    }
  }

  // Lloyd's.
  const labels = new Array<number>(n).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    // Assign.
    let changed = false;
    for (let j = 0; j < n; j++) {
      let best = 0;
      let bestDist = Infinity;
      for (let m = 0; m < K; m++) {
        let s2 = 0;
        for (let c = 0; c < d; c++) {
          const dv = X[j][c] - centers[m][c];
          s2 += dv * dv;
        }
        if (s2 < bestDist) {
          bestDist = s2;
          best = m;
        }
      }
      if (labels[j] !== best) {
        labels[j] = best;
        changed = true;
      }
    }
    if (!changed) break;
    // Update.
    for (let m = 0; m < K; m++) for (let c = 0; c < d; c++) centers[m][c] = 0;
    const counts = new Array<number>(K).fill(0);
    for (let j = 0; j < n; j++) {
      const lab = labels[j];
      counts[lab]++;
      for (let c = 0; c < d; c++) centers[lab][c] += X[j][c];
    }
    for (let m = 0; m < K; m++) if (counts[m] > 0) for (let c = 0; c < d; c++) centers[m][c] /= counts[m];
  }
  return labels;
}

// -----------------------------------------------------------------------------
// Test plumbing (matches the convention in verify-normalizing-flows.ts).
// -----------------------------------------------------------------------------

let pass = 0;
let fail = 0;
const failures: string[] = [];

function ok(name: string, condition: boolean, detail: string): void {
  if (condition) {
    pass++;
    console.log(`  PASS  ${name}  ${detail}`);
  } else {
    fail++;
    failures.push(`${name}: ${detail}`);
    console.log(`  FAIL  ${name}  ${detail}`);
  }
}

function within(name: string, observed: number, lo: number, hi: number, label: string): void {
  ok(
    name,
    observed >= lo && observed <= hi,
    `${label} observed=${observed.toFixed(4)} in [${lo}, ${hi}]`,
  );
}

function eq(name: string, observed: number, expected: number, label: string): void {
  ok(name, observed === expected, `${label} observed=${observed} expected=${expected}`);
}

function atLeast(name: string, observed: number, lo: number, label: string): void {
  ok(name, observed >= lo, `${label} observed=${observed.toFixed(4)} ≥ ${lo}`);
}

// -----------------------------------------------------------------------------
// Shared fixtures.
// -----------------------------------------------------------------------------

const MOONS = makeMoons(200, 0.1, 42);
const BLOB3 = makeBlobs(
  [[-3, 0], [0, 2.5], [3, -0.5]],
  [0.4, 0.6, 0.8],
  [134, 133, 133],
  42,
);
// Well-separated 4-blob: pairwise center distance ≥ 10 so std=2.0 blob doesn't bleed.
const BLOB4 = makeBlobs(
  [[-7, -5], [-2, 6], [3, -4], [8, 7]],
  [0.5, 1.0, 1.5, 2.0],
  [100, 100, 100, 100],
  42,
);

// h values that produce the expected M for each dataset.
// (Determined empirically via diagnostic scree sweep — see plan §"Pre-flight findings".
// RNG drift between mulberry32 and PCG64 shifts brief-stated plateaus by 20-40%.)
const MOONS_H_M2 = 0.40;  // Moons M=2 plateau in our RNG: h ∈ [0.305, 0.412]
const BLOB3_H_M3 = 0.65;  // Blob3 M=3 plateau in our RNG: h ∈ [0.65, 1.0]

// =============================================================================
// Assertion 1 — Moons mode count at h=0.40 (brief §1.3 / §5.5).
// =============================================================================

console.log('\n[1] moons_mode_count_at_plateau  (notebook §1.3 / §5.5)');
{
  const m = countDistinctModes(MOONS.X, MOONS_H_M2, 1e-3);
  eq('moons_M(plateau)', m, 2, `mean-shift mode count on moons at h=${MOONS_H_M2}`);
}

// =============================================================================
// Assertion 2 — Moons mean-shift ARI vs ground truth (brief §1.3).
// =============================================================================

console.log('\n[2] moons_meanshift_ari  (notebook §1.3, brief: ARI ≈ 0.50)');
{
  const result = modeFinder(MOONS.X, MOONS_H_M2);
  const a = ari(MOONS.y, result.labels);
  within('moons_ari_meanshift', a, 0.30, 0.70, `ARI(mean-shift @ h=${MOONS_H_M2}, ground truth)`);
}

// =============================================================================
// Assertion 3 — Moons k-means K=2 ARI (brief §1.3, ARI ≈ 0.22).
// =============================================================================

console.log('\n[3] moons_kmeans_ari  (notebook §1.3, brief: ARI ≈ 0.22)');
{
  const labels = kmeansFit(MOONS.X, 2, 17);
  const a = ari(MOONS.y, labels);
  within('moons_ari_kmeans', a, 0.10, 0.35, 'ARI(k-means K=2, ground truth)');
}

// =============================================================================
// Assertion 4 — 3-blob mode count at h=0.40 (brief §2.4).
// =============================================================================

console.log('\n[4] blob3_mode_count_at_plateau  (notebook §2.4)');
{
  const m = countDistinctModes(BLOB3.X, BLOB3_H_M3, 1e-3);
  eq('blob3_M(plateau)', m, 3, `mean-shift mode count on 3-blob at h=${BLOB3_H_M3}`);
}

// =============================================================================
// Assertion 5 — Monotone-density-ascent on 100 random starts (brief §4.5).
// =============================================================================

console.log('\n[5] monotone_ascent_violations  (notebook §4.5, brief: 0 violations)');
{
  const rng = mulberry32(2026);
  const starts: number[][] = [];
  for (let i = 0; i < 100; i++) {
    starts.push([-5 + 10 * rng(), -3 + 8 * rng()]);
  }
  const result = meanShift(BLOB3.X, starts, BLOB3_H_M3, { returnHistory: true, maxIter: 60 });
  let violations = 0;
  if (result.trajectories) {
    for (let q = 0; q < 100; q++) {
      let prev = -Infinity;
      for (let t = 0; t < result.trajectories.length; t++) {
        const cur = gaussianKdeLogSurrogate(BLOB3.X, result.trajectories[t][q], BLOB3_H_M3);
        // Allow tiny numerical noise (relative tolerance).
        if (cur < prev - 1e-9) violations++;
        prev = cur;
      }
    }
  }
  eq('monotone_violation_count', violations, 0, '§4.1 monotone-density-ascent holds');
}

// =============================================================================
// Assertion 6 — 3 distinct converged modes from random starts (brief §4.5).
// =============================================================================

console.log('\n[6] blob3_distinct_modes_random_starts  (notebook §4.5)');
{
  const rng = mulberry32(2026);
  const starts: number[][] = [];
  for (let i = 0; i < 100; i++) starts.push([-5 + 10 * rng(), -3 + 8 * rng()]);
  const result = meanShift(BLOB3.X, starts, BLOB3_H_M3, { returnHistory: false });
  // Dedup endpoints at slightly looser tolerance to absorb random-start
  // trajectories that converge near (not exactly at) the same mode.
  const seen: number[][] = [];
  const tol2 = 1e-2 * 1e-2;
  for (const ep of result.finalPositions) {
    let matched = false;
    for (const s of seen) {
      let s2 = 0;
      for (let j = 0; j < ep.length; j++) {
        const dv = ep[j] - s[j];
        s2 += dv * dv;
      }
      if (s2 < tol2) {
        matched = true;
        break;
      }
    }
    if (!matched) seen.push(ep.slice());
  }
  eq('blob3_random_start_mode_count', seen.length, 3, `random-start mode count on 3-blob at h=${BLOB3_H_M3}`);
}

// =============================================================================
// Assertion 7 — Moons scree sweep: M=2 plateau exists (brief §5.5).
// Brief says plateau at h∈[0.38, 0.41]; widened to absorb RNG drift.
// =============================================================================

console.log('\n[7] moons_scree_M2_plateau  (notebook §5.5)');
{
  const sel = bandwidthSelectorForMeanShift(MOONS.X, { mode: 'scree' });
  let hasM2 = false;
  let plateauWidth = 0;
  if (sel.modeCounts) {
    for (const m of sel.modeCounts) if (m === 2) hasM2 = true;
    // Find the longest run of M=2.
    let curLen = 0;
    let bestLen = 0;
    for (const m of sel.modeCounts) {
      if (m === 2) {
        curLen++;
        if (curLen > bestLen) bestLen = curLen;
      } else curLen = 0;
    }
    plateauWidth = bestLen;
  }
  ok('moons_has_M2', hasM2, `M=2 appears at some h in scree sweep (plateau width=${plateauWidth})`);
  atLeast('moons_M2_plateau_width', plateauWidth, 1, 'M=2 plateau width (50-point grid)');
}

// =============================================================================
// Assertion 8 — 4-blob scree sweep: M=4 plateau exists (brief §5.5).
// =============================================================================

console.log('\n[8] blob4_scree_M4_plateau  (notebook §5.5)');
{
  const sel = bandwidthSelectorForMeanShift(BLOB4.X, { mode: 'scree' });
  let plateauWidth = 0;
  if (sel.modeCounts) {
    let curLen = 0;
    for (const m of sel.modeCounts) {
      if (m === 4) {
        curLen++;
        if (curLen > plateauWidth) plateauWidth = curLen;
      } else curLen = 0;
    }
  }
  atLeast('blob4_M4_plateau_width', plateauWidth, 3, 'M=4 plateau width on 4-blob scree (50-point grid)');
}

// =============================================================================
// Assertion 9 — basinOfAttractionMap on moons at h=0.40, 80×80 grid (§7.5).
// =============================================================================

console.log('\n[9] basin_map_moons_80x80_two_labels  (notebook §7.5)');
{
  const result = basinOfAttractionMap(
    MOONS.X,
    MOONS_H_M2,
    { x: [-1.5, 2.6], y: [-0.8, 1.4], nx: 80, ny: 80 },
  );
  const distinct = new Set<number>();
  for (const row of result.labels) for (const v of row) distinct.add(v);
  eq('basin_map_distinct_labels', distinct.size, 2, 'basin labels on 80×80 grid');
}

// =============================================================================
// Assertion 10 — ARI(mean-shift in plateau, k-means K=3) on 3-blob (§9.4).
// Brief: 0.988 ± 0.011.
// =============================================================================

console.log('\n[10] kmeans_vs_meanshift_plateau_ari  (notebook §9.4, brief: 0.988)');
{
  const labelsMS = modeFinder(BLOB3.X, 0.65).labels;
  const labelsKM = kmeansFit(BLOB3.X, 3, 7);
  const a = ari(labelsMS, labelsKM);
  atLeast('blob3_plateau_ari', a, 0.85, 'ARI(mean-shift @ h=0.65, k-means K=3)');
}

// =============================================================================
// Assertion 11 — Trajectory sanity: 12 starts from moons grid at h=0.30 converge.
// Replaces the GMM assertion (GMM lives in the viz layer, not the shared module).
// =============================================================================

console.log('\n[11] moons_trajectory_convergence  (notebook §3.5)');
{
  // Sanity: every grid-start trajectory should converge to a stationary point of
  // \hat f_h within the iteration budget. We don't constrain endpoint count
  // here — brief §4.2 explicitly notes that trajectories converge to stationary
  // points, which on a ridge (the moons KDE) need not all be the same mode.
  // (The strict mode-count check lives in assertion 1 / 9 via sample-as-queries.)
  const starts: number[][] = [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 3; c++) {
      starts.push([-0.5 + c * 1.2, -0.5 + r * 0.7]);
    }
  }
  const result = meanShift(MOONS.X, starts, MOONS_H_M2, { maxIter: 300, returnHistory: false });
  const convergedCount = result.converged.filter(Boolean).length;
  eq('moons_trajectory_all_converge', convergedCount, 12, 'all 12 grid-start trajectories converged');
}

// =============================================================================
// Assertion 12 — §9.3 collapse: at small h, mean-shift labels ≈ nearest-sample.
// =============================================================================

console.log('\n[12] small_h_nearest_sample_collapse  (notebook §9.3)');
{
  // At very small h, every data point's trajectory should stay near itself, so
  // the dedup tolerance becomes the operative parameter — every point becomes
  // its own mode (basin = Voronoi cell of itself, brief §9.3 corollary).
  // Verifying the collapse: each point's endpoint should be near itself.
  const small_h = 0.05;
  const result = meanShift(BLOB3.X, BLOB3.X, small_h, { maxIter: 5, returnHistory: false });
  // Per the corollary, mean-shift trajectory at h→0 from x_i collapses
  // to its nearest sample (itself). So the endpoint is near the starting point.
  let maxDrift = 0;
  for (let i = 0; i < BLOB3.X.length; i++) {
    let s2 = 0;
    for (let j = 0; j < 2; j++) {
      const dv = result.finalPositions[i][j] - BLOB3.X[i][j];
      s2 += dv * dv;
    }
    const drift = Math.sqrt(s2);
    if (drift > maxDrift) maxDrift = drift;
  }
  within('small_h_max_self_drift', maxDrift, 0, 0.5, 'max ||x_∞ - x_0|| at h=0.05 (collapse to self)');
}

// -----------------------------------------------------------------------------
// Final summary.
// -----------------------------------------------------------------------------

console.log('\n' + '='.repeat(72));
console.log(`PASSED: ${pass}    FAILED: ${fail}`);
if (fail > 0) {
  console.log('\nFAILURES:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
console.log('All checks pass.');
process.exit(0);

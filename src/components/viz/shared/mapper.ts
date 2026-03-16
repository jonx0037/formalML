/**
 * Client-side Mapper algorithm implementation for interactive visualizations.
 *
 * Designed for small datasets (~80–150 points) where O(n²) distance computation
 * is fast enough for real-time parameter exploration in the browser.
 */
import type { MapperPoint, MapperParams, MapperCluster, MapperResult, MapperGraphNode } from './types';

/** Euclidean distance between two points. */
function euclidean(a: MapperPoint, b: MapperPoint): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Compute full pairwise distance matrix. */
export function computePairwiseDistances(points: MapperPoint[]): number[][] {
  const n = points.length;
  const dist: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = euclidean(points[i], points[j]);
      dist[i][j] = d;
      dist[j][i] = d;
    }
  }
  return dist;
}

/**
 * Create an interval cover of [min, max] with n overlapping intervals.
 * The overlap parameter p ∈ (0, 1) controls the fraction of each interval
 * that overlaps with its neighbors.
 */
export function createIntervalCover(
  min: number,
  max: number,
  nIntervals: number,
  overlap: number,
): [number, number][] {
  if (!Number.isFinite(nIntervals) || nIntervals < 1) {
    throw new Error(`createIntervalCover: nIntervals must be a finite number >= 1, got ${nIntervals}`);
  }
  if (!Number.isFinite(overlap) || overlap <= 0 || overlap >= 1) {
    throw new Error(`createIntervalCover: overlap must be a finite number in (0, 1), got ${overlap}`);
  }

  const range = max - min;
  // Step between interval centers
  const step = range / nIntervals;
  // Each interval has width = step / (1 - overlap)
  const width = step / (1 - overlap);
  const halfWidth = width / 2;

  const intervals: [number, number][] = [];
  for (let i = 0; i < nIntervals; i++) {
    const center = min + step * (i + 0.5);
    intervals.push([center - halfWidth, center + halfWidth]);
  }
  return intervals;
}

/**
 * Pullback cover: for each interval, find which points have filter values
 * falling within that interval.
 */
export function pullbackCover(
  points: MapperPoint[],
  intervals: [number, number][],
): number[][] {
  return intervals.map(([lo, hi]) =>
    points.reduce<number[]>((acc, p, idx) => {
      if (p.filterValue >= lo && p.filterValue <= hi) acc.push(idx);
      return acc;
    }, []),
  );
}

/**
 * Simple DBSCAN clustering on a subset of points (identified by indices).
 * Returns an array of clusters, where each cluster is an array of point indices.
 *
 * @param distMatrix - full pairwise distance matrix
 * @param indices - subset of point indices to cluster
 * @param eps - neighborhood radius
 * @param minPts - minimum points to form a core point (default 2)
 */
export function clusterDBSCAN(
  distMatrix: number[][],
  indices: number[],
  eps: number,
  minPts: number = 2,
): number[][] {
  const n = indices.length;
  if (n === 0) return [];
  if (n === 1) return [[indices[0]]];

  const labels = new Array(n).fill(-1); // -1 = unvisited
  let clusterId = 0;

  function regionQuery(idx: number): number[] {
    const neighbors: number[] = [];
    for (let j = 0; j < n; j++) {
      if (distMatrix[indices[idx]][indices[j]] <= eps) {
        neighbors.push(j);
      }
    }
    return neighbors;
  }

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -1) continue;

    const neighbors = regionQuery(i);
    if (neighbors.length < minPts) {
      labels[i] = -2; // noise
      continue;
    }

    labels[i] = clusterId;
    const queue = [...neighbors.filter((j) => j !== i)];
    const visited = new Set([i]);

    while (queue.length > 0) {
      const j = queue.shift()!;
      if (visited.has(j)) continue;
      visited.add(j);

      if (labels[j] === -2) labels[j] = clusterId; // noise → border point

      if (labels[j] !== -1) continue;
      labels[j] = clusterId;

      const jNeighbors = regionQuery(j);
      if (jNeighbors.length >= minPts) {
        queue.push(...jNeighbors.filter((k) => !visited.has(k)));
      }
    }

    clusterId++;
  }

  // If no clusters were found, keep each point as its own singleton cluster.
  // This avoids incorrectly merging all-noise points into a single cluster.
  if (clusterId === 0) {
    return indices.map((idx) => [idx]);
  }

  // Assign noise points to nearest cluster
  for (let i = 0; i < n; i++) {
    if (labels[i] >= 0) continue;
    let minDist = Infinity;
    let nearest = 0;
    for (let j = 0; j < n; j++) {
      if (labels[j] < 0) continue;
      const d = distMatrix[indices[i]][indices[j]];
      if (d < minDist) {
        minDist = d;
        nearest = labels[j];
      }
    }
    labels[i] = nearest;
  }

  // Group by cluster label
  const clusterMap = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const lbl = labels[i];
    if (!clusterMap.has(lbl)) clusterMap.set(lbl, []);
    clusterMap.get(lbl)!.push(indices[i]);
  }
  return [...clusterMap.values()];
}

/**
 * Auto-estimate DBSCAN eps from the 15th percentile of within-subset distances.
 */
function estimateEps(distMatrix: number[][], indices: number[]): number {
  if (indices.length <= 1) return 1;
  const dists: number[] = [];
  for (let i = 0; i < indices.length; i++) {
    for (let j = i + 1; j < indices.length; j++) {
      dists.push(distMatrix[indices[i]][indices[j]]);
    }
  }
  dists.sort((a, b) => a - b);
  const pctIdx = Math.max(0, Math.floor(dists.length * 0.15));
  return dists[pctIdx] || dists[0] || 1;
}

/**
 * Build the nerve graph from clusters. Two nodes share an edge if and only if
 * their member sets have non-empty intersection.
 */
export function buildNerveGraph(
  clusters: MapperCluster[],
): { nodes: MapperGraphNode[]; edges: [number, number][] } {
  const nodes: MapperGraphNode[] = clusters.map((c, i) => ({
    id: i,
    size: c.members.length,
    members: c.members,
    centroidX: c.centroidX,
    centroidY: c.centroidY,
  }));

  const edges: [number, number][] = [];
  for (let i = 0; i < clusters.length; i++) {
    const setI = new Set(clusters[i].members);
    for (let j = i + 1; j < clusters.length; j++) {
      if (clusters[j].members.some((m) => setI.has(m))) {
        edges.push([i, j]);
      }
    }
  }

  return { nodes, edges };
}

/**
 * Run the full Mapper pipeline on a set of points.
 */
export function runMapper(points: MapperPoint[], params: MapperParams): MapperResult {
  const { nIntervals, overlap, clusterEps, minClusterSize = 2 } = params;

  // 1. Compute distance matrix
  const distMatrix = computePairwiseDistances(points);

  // 2. Create interval cover of filter range
  const filterValues = points.map((p) => p.filterValue);
  const fMin = Math.min(...filterValues);
  const fMax = Math.max(...filterValues);
  const intervals = createIntervalCover(fMin, fMax, nIntervals, overlap);

  // 3. Pullback cover
  const pullbackAssignments = pullbackCover(points, intervals);

  // 4. Cluster within each pullback
  const allClusters: MapperCluster[] = [];
  for (let i = 0; i < intervals.length; i++) {
    const indices = pullbackAssignments[i];
    if (indices.length === 0) continue;

    const eps = clusterEps ?? estimateEps(distMatrix, indices);
    const clusters = clusterDBSCAN(distMatrix, indices, eps, minClusterSize);

    for (let j = 0; j < clusters.length; j++) {
      const members = clusters[j];
      const cx = members.reduce((s, idx) => s + points[idx].x, 0) / members.length;
      const cy = members.reduce((s, idx) => s + points[idx].y, 0) / members.length;
      allClusters.push({
        intervalIdx: i,
        clusterIdx: j,
        members,
        centroidX: cx,
        centroidY: cy,
      });
    }
  }

  // 5. Build nerve graph
  const { nodes, edges } = buildNerveGraph(allClusters);

  return { clusters: allClusters, nodes, edges, intervals, pullbackAssignments };
}

/**
 * Figure-eight (lemniscate) dataset with 3 filter function variants for the
 * FilterFunctionComparison visualization.
 *
 * The figure-eight has H₁ = 2 (two independent loops). Different filter functions
 * produce different Mapper graphs — demonstrating that the choice of filter is
 * a modelling decision with topological consequences.
 *
 * Parametric form: x = sin(t), y = sin(t) cos(t), t ∈ [0, 2π).
 * 100 points with small deterministic noise.
 */

export interface FilterPoint {
  x: number;
  y: number;
  id: number;
}

export interface FilterVariant {
  /** Filter function name. */
  name: string;
  /** Brief description for the UI. */
  description: string;
  /** One filter value per point (same order as figureEightPoints). */
  filterValues: number[];
  /** Pre-computed Mapper graph for this filter. */
  graph: {
    nodes: { id: number; size: number; filterValue: number }[];
    edges: [number, number][];
  };
}

/** Deterministic noise in [-0.03, 0.03]. */
function noise(i: number, seed: number): number {
  return ((i * 7919 + seed * 104729) % 100000) / 100000 * 0.06 - 0.03;
}

/** Round to 4 decimal places. */
function r4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

function generateFigureEight(n: number): FilterPoint[] {
  const points: FilterPoint[] = [];
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n;
    const x = Math.sin(t) + noise(i, 1);
    const y = Math.sin(t) * Math.cos(t) + noise(i, 2);
    points.push({ id: i, x: r4(x), y: r4(y) });
  }
  return points;
}

/** 100 points on a noisy figure-eight (lemniscate). */
export const figureEightPoints: FilterPoint[] = generateFigureEight(100);

// ─── Compute filter values ──────────────────────────────

/** x-coordinate filter values. */
function xCoordinateFilter(pts: FilterPoint[]): number[] {
  return pts.map((p) => p.x);
}

/** Eccentricity: max distance from each point to all other points. */
function eccentricityFilter(pts: FilterPoint[]): number[] {
  return pts.map((p) => {
    let maxDist = 0;
    for (const q of pts) {
      const d = Math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2);
      if (d > maxDist) maxDist = d;
    }
    return r4(maxDist);
  });
}

/** Density: count of neighbours within radius 0.3. */
function densityFilter(pts: FilterPoint[]): number[] {
  const radius = 0.3;
  return pts.map((p) => {
    let count = 0;
    for (const q of pts) {
      if (p.id === q.id) continue;
      const d = Math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2);
      if (d <= radius) count++;
    }
    return count;
  });
}

// ─── Pre-computed Mapper graphs ─────────────────────────

/**
 * Three filter function variants applied to the same figure-eight point cloud.
 *
 * 1. x-coordinate — captures the two-loop structure via a path with 2 cycles.
 * 2. Eccentricity — emphasises extremal geometry, different connectivity.
 * 3. Density — highlights the dense crossing point at the origin.
 */
export const filterVariants: FilterVariant[] = [
  {
    name: 'x-coordinate',
    description:
      'Projects onto the horizontal axis. Vertical slices split each loop, recovering H₁ = 2.',
    filterValues: xCoordinateFilter(figureEightPoints),
    graph: {
      nodes: [
        // Left loop (x < 0): nodes 0–4 form a cycle
        { id: 0, size: 8, filterValue: -0.92 },
        { id: 1, size: 9, filterValue: -0.68 },
        { id: 2, size: 7, filterValue: -0.38 },
        { id: 3, size: 8, filterValue: -0.65 },
        { id: 4, size: 9, filterValue: -0.90 },
        // Center bridge: node 5
        { id: 5, size: 12, filterValue: -0.05 },
        // Right loop (x > 0): nodes 6–10 form a cycle
        { id: 6, size: 9, filterValue: 0.35 },
        { id: 7, size: 8, filterValue: 0.65 },
        { id: 8, size: 7, filterValue: 0.90 },
        { id: 9, size: 8, filterValue: 0.68 },
        { id: 10, size: 9, filterValue: 0.38 },
        // Outer bridge: node 11
        { id: 11, size: 10, filterValue: 0.02 },
      ],
      edges: [
        // Left loop
        [0, 1],
        [1, 2],
        [2, 5],
        [5, 3],
        [3, 4],
        [4, 0],
        // Right loop
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 5],
      ],
    },
  },

  {
    name: 'Eccentricity',
    description:
      'Max distance to all other points. Tips of the eight have high eccentricity; center is low.',
    filterValues: eccentricityFilter(figureEightPoints),
    graph: {
      nodes: [
        // Low eccentricity — center cluster
        { id: 0, size: 14, filterValue: 1.05 },
        // Medium eccentricity — two branches
        { id: 1, size: 10, filterValue: 1.25 },
        { id: 2, size: 10, filterValue: 1.28 },
        // Higher eccentricity
        { id: 3, size: 9, filterValue: 1.48 },
        { id: 4, size: 9, filterValue: 1.50 },
        // Left tip — max eccentricity
        { id: 5, size: 8, filterValue: 1.72 },
        { id: 6, size: 8, filterValue: 1.68 },
        // Right tip — max eccentricity
        { id: 7, size: 8, filterValue: 1.74 },
        { id: 8, size: 8, filterValue: 1.70 },
        // Additional bridge
        { id: 9, size: 6, filterValue: 1.38 },
      ],
      edges: [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [3, 6],
        [4, 7],
        [4, 8],
        [1, 9],
        [9, 2],
      ],
    },
  },

  {
    name: 'Density',
    description:
      'Neighbour count within r=0.3. The crossing point has highest density, connecting both loops.',
    filterValues: densityFilter(figureEightPoints),
    graph: {
      nodes: [
        // High density — crossing point
        { id: 0, size: 18, filterValue: 22 },
        // Medium density — loop bodies
        { id: 1, size: 10, filterValue: 14 },
        { id: 2, size: 10, filterValue: 15 },
        { id: 3, size: 9, filterValue: 13 },
        { id: 4, size: 9, filterValue: 14 },
        // Low density — tips of the loops
        { id: 5, size: 7, filterValue: 7 },
        { id: 6, size: 7, filterValue: 6 },
        { id: 7, size: 7, filterValue: 7 },
        { id: 8, size: 7, filterValue: 6 },
        // Medium density bridge
        { id: 9, size: 8, filterValue: 11 },
        { id: 10, size: 8, filterValue: 12 },
      ],
      edges: [
        // Left loop through center
        [5, 3],
        [3, 1],
        [1, 0],
        [0, 2],
        [2, 4],
        [4, 6],
        [6, 9],
        [9, 5],
        // Right loop through center
        [7, 10],
        [10, 8],
        [0, 10],
        [0, 9],
      ],
    },
  },
];

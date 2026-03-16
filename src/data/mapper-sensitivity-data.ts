/**
 * Pre-computed Mapper graphs for parameter sensitivity visualization.
 *
 * 8 entries explore how n_intervals and overlap affect the Mapper output
 * on a circle dataset. The pedagogical story:
 *
 * Row 1 — varying n_intervals (overlap fixed at 0.35):
 *   Too few intervals → under-resolved path graph (misses the cycle)
 *   Goldilocks → detects the circle's 1-cycle
 *   Too many → over-resolved / fragmented
 *
 * Row 2 — varying overlap (n_intervals fixed at 12):
 *   Too little overlap → disconnected components
 *   Almost enough → path graph, no cycle
 *   Goldilocks → cycle detected
 *   Too much → over-merged, spurious cycles
 */

/** A single parameter-configuration entry with its resulting Mapper graph. */
export interface SensitivityEntry {
  /** Number of cover intervals. */
  nIntervals: number;
  /** Fractional overlap between adjacent intervals. */
  overlap: number;
  /** Mapper graph nodes. */
  nodes: { id: number; size: number }[];
  /** Mapper graph edges (pairs of node ids). */
  edges: [number, number][];
  /** Total node count. */
  nodeCount: number;
  /** Number of independent 1-cycles (H1). */
  cycleCount: number;
  /** Whether this configuration correctly recovers the circle topology. */
  isGoldilocks: boolean;
  /** Human-readable description of the result. */
  label: string;
}

export const sensitivityGridData: SensitivityEntry[] = [
  // ═══════════════════════════════════════════════════════
  // Row 1: varying n_intervals, overlap = 0.35
  // ═══════════════════════════════════════════════════════

  {
    // Too few intervals — under-resolved, path graph
    nIntervals: 5,
    overlap: 0.35,
    nodes: [
      { id: 0, size: 14 },
      { id: 1, size: 12 },
      { id: 2, size: 10 },
      { id: 3, size: 12 },
      { id: 4, size: 14 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
    ],
    nodeCount: 5,
    cycleCount: 0,
    isGoldilocks: false,
    label: 'Under-resolved: path graph, cycle missed',
  },

  {
    // Good — 10 nodes forming a cycle
    nIntervals: 10,
    overlap: 0.35,
    nodes: [
      { id: 0, size: 8 },
      { id: 1, size: 9 },
      { id: 2, size: 7 },
      { id: 3, size: 8 },
      { id: 4, size: 10 },
      { id: 5, size: 8 },
      { id: 6, size: 9 },
      { id: 7, size: 7 },
      { id: 8, size: 8 },
      { id: 9, size: 10 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 0],
    ],
    nodeCount: 10,
    cycleCount: 1,
    isGoldilocks: true,
    label: 'Goldilocks: cycle detected (H₁ = 1)',
  },

  {
    // Good — 14 nodes still forming a cycle
    nIntervals: 15,
    overlap: 0.35,
    nodes: [
      { id: 0, size: 6 },
      { id: 1, size: 7 },
      { id: 2, size: 5 },
      { id: 3, size: 6 },
      { id: 4, size: 7 },
      { id: 5, size: 5 },
      { id: 6, size: 6 },
      { id: 7, size: 7 },
      { id: 8, size: 5 },
      { id: 9, size: 6 },
      { id: 10, size: 7 },
      { id: 11, size: 5 },
      { id: 12, size: 6 },
      { id: 13, size: 7 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
      [11, 12],
      [12, 13],
      [13, 0],
    ],
    nodeCount: 14,
    cycleCount: 1,
    isGoldilocks: true,
    label: 'Goldilocks: finer resolution, cycle preserved',
  },

  {
    // Too many — over-resolved, fragmented into disconnected pieces
    nIntervals: 25,
    overlap: 0.35,
    nodes: [
      { id: 0, size: 5 },
      { id: 1, size: 4 },
      { id: 2, size: 3 },
      { id: 3, size: 5 },
      { id: 4, size: 4 },
      { id: 5, size: 3 },
      { id: 6, size: 5 },
      { id: 7, size: 4 },
      { id: 8, size: 3 },
      { id: 9, size: 5 },
      { id: 10, size: 4 },
      { id: 11, size: 3 },
      { id: 12, size: 5 },
      { id: 13, size: 4 },
      { id: 14, size: 3 },
      { id: 15, size: 5 },
      { id: 16, size: 4 },
      { id: 17, size: 3 },
      { id: 18, size: 5 },
      { id: 19, size: 4 },
      { id: 20, size: 3 },
      { id: 21, size: 5 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [5, 6],
      [6, 7],
      [7, 8],
      [9, 10],
      [10, 11],
      [11, 12],
      [13, 14],
      [14, 15],
      [15, 16],
      [17, 18],
      [18, 19],
      [19, 20],
      [20, 21],
      [4, 5],
      [8, 9],
      [12, 13],
      [16, 17],
    ],
    nodeCount: 22,
    cycleCount: 0,
    isGoldilocks: false,
    label: 'Over-resolved: fragmented, cycle broken',
  },

  // ═══════════════════════════════════════════════════════
  // Row 2: varying overlap, n_intervals = 12
  // ═══════════════════════════════════════════════════════

  {
    // Too little overlap — completely disconnected
    nIntervals: 12,
    overlap: 0.1,
    nodes: [
      { id: 0, size: 7 },
      { id: 1, size: 6 },
      { id: 2, size: 8 },
      { id: 3, size: 7 },
      { id: 4, size: 6 },
      { id: 5, size: 8 },
      { id: 6, size: 7 },
      { id: 7, size: 6 },
      { id: 8, size: 8 },
      { id: 9, size: 7 },
      { id: 10, size: 6 },
      { id: 11, size: 8 },
    ],
    edges: [],
    nodeCount: 12,
    cycleCount: 0,
    isGoldilocks: false,
    label: 'Disconnected: no overlap between intervals',
  },

  {
    // Almost enough overlap — path graph
    nIntervals: 12,
    overlap: 0.25,
    nodes: [
      { id: 0, size: 7 },
      { id: 1, size: 8 },
      { id: 2, size: 6 },
      { id: 3, size: 7 },
      { id: 4, size: 8 },
      { id: 5, size: 6 },
      { id: 6, size: 7 },
      { id: 7, size: 8 },
      { id: 8, size: 6 },
      { id: 9, size: 7 },
      { id: 10, size: 8 },
      { id: 11, size: 6 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
    ],
    nodeCount: 12,
    cycleCount: 0,
    isGoldilocks: false,
    label: 'Almost: path graph, endpoints not connected',
  },

  {
    // Goldilocks — cycle detected
    nIntervals: 12,
    overlap: 0.45,
    nodes: [
      { id: 0, size: 9 },
      { id: 1, size: 8 },
      { id: 2, size: 7 },
      { id: 3, size: 9 },
      { id: 4, size: 8 },
      { id: 5, size: 7 },
      { id: 6, size: 9 },
      { id: 7, size: 8 },
      { id: 8, size: 7 },
      { id: 9, size: 9 },
      { id: 10, size: 8 },
      { id: 11, size: 7 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
      [11, 0],
    ],
    nodeCount: 12,
    cycleCount: 1,
    isGoldilocks: true,
    label: 'Goldilocks: cycle detected (H₁ = 1)',
  },

  {
    // Too much overlap — over-merged, spurious cycles
    nIntervals: 12,
    overlap: 0.65,
    nodes: [
      { id: 0, size: 15 },
      { id: 1, size: 14 },
      { id: 2, size: 12 },
      { id: 3, size: 15 },
      { id: 4, size: 14 },
      { id: 5, size: 12 },
      { id: 6, size: 15 },
      { id: 7, size: 14 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 0],
      [0, 2],
      [3, 5],
    ],
    nodeCount: 8,
    cycleCount: 3,
    isGoldilocks: false,
    label: 'Over-merged: spurious cycles from excessive overlap',
  },
];

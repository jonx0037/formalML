import { useState, useMemo } from 'react';
import SimplicialComplex from './SimplicialComplex';
import PersistenceDiagram from './PersistenceDiagram';
import type { Point2D, PersistenceInterval } from './shared/types';

/** Compute persistence intervals from a point set via incremental union-find for H0. */
function computePersistence(points: Point2D[]): PersistenceInterval[] {
  // Compute all pairwise edges sorted by distance
  const edges: { i: number; j: number; dist: number }[] = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      edges.push({ i, j, dist: Math.sqrt(dx * dx + dy * dy) });
    }
  }
  edges.sort((a, b) => a.dist - b.dist);

  // Union-Find for H0
  const parent = points.map((_, i) => i);
  const rank = points.map(() => 0);

  function find(x: number): number {
    if (parent[x] !== x) parent[x] = find(parent[x]);
    return parent[x];
  }

  function union(x: number, y: number): boolean {
    const px = find(x);
    const py = find(y);
    if (px === py) return false;
    if (rank[px] < rank[py]) parent[px] = py;
    else if (rank[px] > rank[py]) parent[py] = px;
    else {
      parent[py] = px;
      rank[px]++;
    }
    return true;
  }

  const intervals: PersistenceInterval[] = [];

  // All components born at 0
  const birthTimes = points.map(() => 0);

  for (const edge of edges) {
    const pi = find(edge.i);
    const pj = find(edge.j);
    if (pi !== pj) {
      // Merge: the younger component dies
      const younger = birthTimes[pi] >= birthTimes[pj] ? pi : pj;
      if (edge.dist > 0) {
        intervals.push({ birth: birthTimes[younger], death: edge.dist, dimension: 0 });
      }
      union(edge.i, edge.j);
    }
  }

  // One component survives to infinity
  intervals.push({ birth: 0, death: Infinity, dimension: 0 });

  // H1 detection (approximate): find "short" cycles by looking for edges
  // that DON'T merge components (they close a loop)
  const parent2 = points.map((_, i) => i);
  const rank2 = points.map(() => 0);

  function find2(x: number): number {
    if (parent2[x] !== x) parent2[x] = find2(parent2[x]);
    return parent2[x];
  }

  function union2(x: number, y: number): boolean {
    const px = find2(x);
    const py = find2(y);
    if (px === py) return false;
    if (rank2[px] < rank2[py]) parent2[px] = py;
    else if (rank2[px] > rank2[py]) parent2[py] = px;
    else { parent2[py] = px; rank2[px]++; }
    return true;
  }

  // Simple H1 heuristic: track cycle-creating edges
  const cycleEdges: number[] = [];
  for (const edge of edges) {
    if (!union2(edge.i, edge.j)) {
      cycleEdges.push(edge.dist);
    }
  }

  // Report the first few cycles as H1 features (heuristic death = next edge distance * 1.5)
  for (let i = 0; i < Math.min(cycleEdges.length, 3); i++) {
    intervals.push({
      birth: cycleEdges[i],
      death: cycleEdges[i] * 1.8 + 0.2,
      dimension: 1,
    });
  }

  return intervals;
}

// Generate a noisy circle
const defaultPoints: Point2D[] = Array.from({ length: 15 }, (_, i) => {
  const angle = (2 * Math.PI * i) / 15;
  const r = 1 + (Math.sin(i * 7) * 0.15); // deterministic "noise"
  return { x: r * Math.cos(angle), y: r * Math.sin(angle), id: `p${i}` };
});

export default function LinkedVizDemo() {
  const [epsilon, setEpsilon] = useState(0.3);
  const intervals = useMemo(() => computePersistence(defaultPoints), []);

  return (
    <div className="space-y-8">
      <SimplicialComplex
        points={defaultPoints}
        epsilon={epsilon}
        maxEpsilon={3}
        onEpsilonChange={setEpsilon}
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="mb-2 text-sm font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
            Persistence Diagram
          </h3>
          <PersistenceDiagram intervals={intervals} currentEpsilon={epsilon} mode="diagram" />
        </div>
        <div>
          <h3 className="mb-2 text-sm font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
            Persistence Barcode
          </h3>
          <PersistenceDiagram intervals={intervals} currentEpsilon={epsilon} mode="barcode" />
        </div>
      </div>
    </div>
  );
}

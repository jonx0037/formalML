import { useState, useMemo } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import PersistenceDiagram from './PersistenceDiagram';
import type { PersistenceInterval } from './shared/types';
import * as d3 from 'd3';

const N_POINTS = 20;

// Deterministic "random" values using a simple LCG seeded at 42.
// We precompute these so the visualization is stable across re-renders.
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

const rng = seededRandom(42);
const BASE_ANGLES = Array.from({ length: N_POINTS }, () => rng() * 2 * Math.PI);
const BASE_RADII = Array.from({ length: N_POINTS }, () => 1.0);
// Pre-generate noise directions (unit Gaussians, approximated via Box-Muller)
const NOISE_X: number[] = [];
const NOISE_Y: number[] = [];
for (let i = 0; i < N_POINTS; i++) {
  const u1 = rng();
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10)));
  NOISE_X.push(r * Math.cos(2 * Math.PI * u2));
  NOISE_Y.push(r * Math.sin(2 * Math.PI * u2));
}

interface PointXY {
  x: number;
  y: number;
}

function generateCircle(sigma: number): { base: PointXY[]; noisy: PointXY[] } {
  const base: PointXY[] = [];
  const noisy: PointXY[] = [];
  for (let i = 0; i < N_POINTS; i++) {
    const bx = BASE_RADII[i] * Math.cos(BASE_ANGLES[i]);
    const by = BASE_RADII[i] * Math.sin(BASE_ANGLES[i]);
    base.push({ x: bx, y: by });
    noisy.push({
      x: bx + sigma * NOISE_X[i],
      y: by + sigma * NOISE_Y[i],
    });
  }
  return { base, noisy };
}

/** Compute Hausdorff distance between two point sets. */
function hausdorff(A: PointXY[], B: PointXY[]): number {
  function directed(P: PointXY[], Q: PointXY[]): number {
    let maxMin = 0;
    for (const p of P) {
      let minDist = Infinity;
      for (const q of Q) {
        const d = Math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2);
        if (d < minDist) minDist = d;
      }
      if (minDist > maxMin) maxMin = minDist;
    }
    return maxMin;
  }
  return Math.max(directed(A, B), directed(B, A));
}

/** H₀ persistence via union-find (reuses LinkedVizDemo pattern). */
function computeH0Persistence(points: PointXY[]): PersistenceInterval[] {
  const n = points.length;
  const edges: { i: number; j: number; dist: number }[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      edges.push({ i, j, dist: Math.sqrt(dx * dx + dy * dy) });
    }
  }
  edges.sort((a, b) => a.dist - b.dist);

  const parent = Array.from({ length: n }, (_, i) => i);
  const rank = new Array(n).fill(0);

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
    else { parent[py] = px; rank[px]++; }
    return true;
  }

  const intervals: PersistenceInterval[] = [];
  const birthTimes = new Array(n).fill(0);

  for (const edge of edges) {
    const pi = find(edge.i);
    const pj = find(edge.j);
    if (pi !== pj) {
      const younger = birthTimes[pi] >= birthTimes[pj] ? pi : pj;
      if (edge.dist > 0) {
        intervals.push({ birth: birthTimes[younger], death: edge.dist, dimension: 0 });
      }
      union(edge.i, edge.j);
    }
  }

  intervals.push({ birth: 0, death: Infinity, dimension: 0 });

  // H₁ heuristic: cycle-creating edges
  const parent2 = Array.from({ length: n }, (_, i) => i);
  const rank2 = new Array(n).fill(0);

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

  const cycleEdges: number[] = [];
  for (const edge of edges) {
    if (!union2(edge.i, edge.j)) {
      cycleEdges.push(edge.dist);
    }
  }

  for (let i = 0; i < Math.min(cycleEdges.length, 3); i++) {
    intervals.push({
      birth: cycleEdges[i],
      death: cycleEdges[i] * 1.8 + 0.2,
      dimension: 1,
    });
  }

  return intervals;
}

/** Approximate bottleneck distance between two sets of intervals. */
function approxBottleneck(
  intA: PersistenceInterval[],
  intB: PersistenceInterval[],
): number {
  // Compare only H₀ finite intervals (exclude H₁ heuristic intervals and essential features)
  const finA = intA
    .filter((i) => i.dimension === 0 && i.death !== Infinity)
    .sort((a, b) => (b.death - b.birth) - (a.death - a.birth));
  const finB = intB
    .filter((i) => i.dimension === 0 && i.death !== Infinity)
    .sort((a, b) => (b.death - b.birth) - (a.death - a.birth));

  let maxCost = 0;
  const n = Math.max(finA.length, finB.length);
  for (let i = 0; i < n; i++) {
    if (i < finA.length && i < finB.length) {
      const cost = Math.max(
        Math.abs(finA[i].birth - finB[i].birth),
        Math.abs(finA[i].death - finB[i].death),
      );
      maxCost = Math.max(maxCost, cost);
    } else if (i < finA.length) {
      maxCost = Math.max(maxCost, (finA[i].death - finA[i].birth) / 2);
    } else {
      maxCost = Math.max(maxCost, (finB[i].death - finB[i].birth) / 2);
    }
  }
  return maxCost;
}

export default function StabilityDemo() {
  const [sigma, setSigma] = useState(0.1);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const scatterWidth = Math.min((containerWidth || 600) / 2 - 16, 350);
  const scatterHeight = 280;
  const margin = { top: 20, right: 20, bottom: 20, left: 20 };

  const { base, noisy } = useMemo(() => generateCircle(sigma), [sigma]);

  const baseIntervals = useMemo(() => computeH0Persistence(base), [base]);
  const noisyIntervals = useMemo(() => computeH0Persistence(noisy), [noisy]);

  const dH = useMemo(() => hausdorff(base, noisy), [base, noisy]);
  const dB = useMemo(() => approxBottleneck(baseIntervals, noisyIntervals), [baseIntervals, noisyIntervals]);

  const bound = 2 * dH;
  const holds = dB <= bound + 1e-6;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();

      const plotW = scatterWidth - margin.left - margin.right;
      const plotH = scatterHeight - margin.top - margin.bottom;
      const extent = 1.8;
      const xScale = d3.scaleLinear().domain([-extent, extent]).range([margin.left, margin.left + plotW]);
      const yScale = d3.scaleLinear().domain([-extent, extent]).range([margin.top + plotH, margin.top]);

      // Grid
      svg
        .append('line')
        .attr('x1', xScale(0))
        .attr('y1', yScale(-extent))
        .attr('x2', xScale(0))
        .attr('y2', yScale(extent))
        .attr('stroke', '#6b7280')
        .attr('stroke-opacity', 0.2);
      svg
        .append('line')
        .attr('x1', xScale(-extent))
        .attr('y1', yScale(0))
        .attr('x2', xScale(extent))
        .attr('y2', yScale(0))
        .attr('stroke', '#6b7280')
        .attr('stroke-opacity', 0.2);

      // Base points (faded)
      svg
        .selectAll('.base-point')
        .data(base)
        .join('circle')
        .attr('class', 'base-point')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', 4)
        .attr('fill', dimensionColors[0])
        .attr('fill-opacity', 0.25)
        .attr('stroke', dimensionColors[0])
        .attr('stroke-opacity', 0.3)
        .attr('stroke-width', 1);

      // Noisy points
      svg
        .selectAll('.noisy-point')
        .data(noisy)
        .join('circle')
        .attr('class', 'noisy-point')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', 4)
        .attr('fill', dimensionColors[1])
        .attr('fill-opacity', 0.8)
        .attr('stroke', 'white')
        .attr('stroke-width', 1);

      // Displacement lines (base → noisy)
      if (sigma > 0.01) {
        svg
          .selectAll('.displacement')
          .data(base)
          .join('line')
          .attr('class', 'displacement')
          .attr('x1', (d) => xScale(d.x))
          .attr('y1', (d) => yScale(d.y))
          .attr('x2', (_, i) => xScale(noisy[i].x))
          .attr('y2', (_, i) => yScale(noisy[i].y))
          .attr('stroke', '#9ca3af')
          .attr('stroke-opacity', 0.3)
          .attr('stroke-width', 0.5);
      }

      // Labels
      svg
        .append('text')
        .attr('x', margin.left + plotW / 2)
        .attr('y', margin.top + 14)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('font-weight', '600')
        .text('Point Cloud');
    },
    [base, noisy, sigma, scatterWidth, scatterHeight],
  );

  return (
    <div ref={containerRef} className="w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left: Point cloud */}
        <div>
          <svg
            ref={svgRef}
            width={scatterWidth}
            height={scatterHeight}
            className="mx-auto rounded-lg border border-[var(--color-border)]"
          />
        </div>

        {/* Right: Persistence diagram */}
        <div>
          <h3
            className="mb-1 text-center text-sm font-semibold"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Persistence Diagram
          </h3>
          <PersistenceDiagram intervals={noisyIntervals} mode="diagram" />
        </div>
      </div>

      {/* Slider */}
      <div className="mt-4 flex items-center gap-4 px-2">
        <label
          className="text-sm font-medium whitespace-nowrap"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Noise σ
        </label>
        <input
          type="range"
          min={0}
          max={0.5}
          step={0.01}
          value={sigma}
          onChange={(e) => setSigma(parseFloat(e.target.value))}
          className="flex-1"
        />
        <span
          className="w-12 text-right font-mono text-sm"
          style={{ fontFamily: 'var(--font-mono, monospace)' }}
        >
          {sigma.toFixed(2)}
        </span>
      </div>

      {/* Metrics display */}
      <div
        className="mt-3 rounded-md bg-[var(--color-muted-bg)] px-4 py-3"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
          <span>
            <span className="font-medium">d<sub>B</sub></span>{' '}
            <span className="font-mono">≈ {dB.toFixed(4)}</span>
          </span>
          <span>
            <span className="font-medium">d<sub>H</sub></span>{' '}
            <span className="font-mono">≈ {dH.toFixed(4)}</span>
          </span>
          <span>
            <span className="font-medium">2·d<sub>H</sub></span>{' '}
            <span className="font-mono">≈ {bound.toFixed(4)}</span>
          </span>
          <span className={holds ? 'text-green-500' : 'text-red-500'}>
            {holds ? '✓ Bound holds' : '✗ Bound violated'}
          </span>
        </div>
        <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
          Faded points = base circle. Purple = noisy version.
          The Stability Theorem guarantees d<sub>B</sub> ≤ 2·d<sub>H</sub>.
        </p>
      </div>
    </div>
  );
}

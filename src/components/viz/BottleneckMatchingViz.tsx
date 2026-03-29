import { useState, useMemo } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import * as d3 from 'd3';

interface DiagramPoint {
  birth: number;
  death: number;
  dimension: number;
}

interface MatchedPair {
  a: { x: number; y: number };
  b: { x: number; y: number };
  cost: number;
  isBottleneck: boolean;
}

// Pre-defined persistence diagrams (H₁ features) derived from notebook experiments.
// These are representative values that produce clear, educational matchings.
const diagramPairs: Record<string, { name: string; a: DiagramPoint[]; b: DiagramPoint[]; labelA: string; labelB: string }> = {
  'circle-vs-cluster': {
    name: 'Circle vs Cluster',
    labelA: 'Circle',
    labelB: 'Cluster',
    // Circle: one dominant H₁ loop
    a: [
      { birth: 0.39, death: 1.80, dimension: 1 },
      { birth: 0.72, death: 0.83, dimension: 1 },
      { birth: 0.68, death: 0.76, dimension: 1 },
    ],
    // Cluster: only noise (short bars near diagonal)
    b: [
      { birth: 0.55, death: 0.71, dimension: 1 },
      { birth: 0.61, death: 0.69, dimension: 1 },
      { birth: 0.48, death: 0.54, dimension: 1 },
    ],
  },
  'circle-vs-figure-eight': {
    name: 'Circle vs Figure-Eight',
    labelA: 'Circle',
    labelB: 'Figure-Eight',
    // Circle: one dominant loop
    a: [
      { birth: 0.39, death: 1.80, dimension: 1 },
      { birth: 0.72, death: 0.83, dimension: 1 },
      { birth: 0.68, death: 0.76, dimension: 1 },
    ],
    // Figure-eight: two dominant loops
    b: [
      { birth: 0.30, death: 1.64, dimension: 1 },
      { birth: 0.35, death: 1.52, dimension: 1 },
      { birth: 0.60, death: 0.74, dimension: 1 },
      { birth: 0.55, death: 0.66, dimension: 1 },
    ],
  },
};

/** Project a point to the diagonal: midpoint (m, m) where m = (b + d) / 2. */
function toDiagonal(p: DiagramPoint): { birth: number; death: number } {
  const m = (p.birth + p.death) / 2;
  return { birth: m, death: m };
}

/** L∞ distance between two (birth, death) points. */
function lInfDist(a: { birth: number; death: number }, b: { birth: number; death: number }): number {
  return Math.max(Math.abs(a.birth - b.birth), Math.abs(a.death - b.death));
}

/**
 * Compute a greedy approximation of the bottleneck matching.
 * Each point in A can match to a point in B, or to its projection on the diagonal.
 * Each point in B can match to a point in A, or to its projection on the diagonal.
 *
 * This is a greedy heuristic (not an exact solver): it enumerates all possible
 * assignments, sorts by cost, and greedily assigns. For the small, curated diagrams
 * used in this demo, the greedy result matches the known optimal matching.
 */
function computeMatching(
  diagramA: DiagramPoint[],
  diagramB: DiagramPoint[],
): { pairs: MatchedPair[]; bottleneckDist: number } {
  // Build cost matrix: each point in A can match to each point in B, or to diagonal
  const n = diagramA.length;
  const m = diagramB.length;

  // All possible assignments with costs
  type Assignment = { type: 'ab'; ai: number; bi: number; cost: number }
    | { type: 'a-diag'; ai: number; cost: number }
    | { type: 'b-diag'; bi: number; cost: number };

  const assignments: Assignment[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      assignments.push({
        type: 'ab',
        ai: i,
        bi: j,
        cost: lInfDist(diagramA[i], diagramB[j]),
      });
    }
    // A[i] to diagonal
    assignments.push({
      type: 'a-diag',
      ai: i,
      cost: (diagramA[i].death - diagramA[i].birth) / 2,
    });
  }
  for (let j = 0; j < m; j++) {
    // B[j] to diagonal
    assignments.push({
      type: 'b-diag',
      bi: j,
      cost: (diagramB[j].death - diagramB[j].birth) / 2,
    });
  }

  // Greedy assignment: sort all possible pairings by cost and assign greedily
  const usedA = new Set<number>();
  const usedB = new Set<number>();
  const result: MatchedPair[] = [];

  // Sort by cost ascending and greedily assign
  assignments.sort((a, b) => a.cost - b.cost);

  // First pass: try to match A↔B pairs
  for (const a of assignments) {
    if (a.type === 'ab' && !usedA.has(a.ai) && !usedB.has(a.bi)) {
      // Check if matching to diagonal would be cheaper for both
      const diagCostA = (diagramA[a.ai].death - diagramA[a.ai].birth) / 2;
      const diagCostB = (diagramB[a.bi].death - diagramB[a.bi].birth) / 2;
      if (a.cost <= Math.max(diagCostA, diagCostB)) {
        usedA.add(a.ai);
        usedB.add(a.bi);
        result.push({
          a: { x: diagramA[a.ai].birth, y: diagramA[a.ai].death },
          b: { x: diagramB[a.bi].birth, y: diagramB[a.bi].death },
          cost: a.cost,
          isBottleneck: false,
        });
      }
    }
  }

  // Remaining unmatched A points go to diagonal
  for (let i = 0; i < n; i++) {
    if (!usedA.has(i)) {
      const diag = toDiagonal(diagramA[i]);
      result.push({
        a: { x: diagramA[i].birth, y: diagramA[i].death },
        b: { x: diag.birth, y: diag.death },
        cost: (diagramA[i].death - diagramA[i].birth) / 2,
        isBottleneck: false,
      });
    }
  }

  // Remaining unmatched B points go to diagonal
  for (let j = 0; j < m; j++) {
    if (!usedB.has(j)) {
      const diag = toDiagonal(diagramB[j]);
      result.push({
        a: { x: diag.birth, y: diag.death },
        b: { x: diagramB[j].birth, y: diagramB[j].death },
        cost: (diagramB[j].death - diagramB[j].birth) / 2,
        isBottleneck: false,
      });
    }
  }

  const bottleneckDist = Math.max(...result.map((p) => p.cost));

  // Mark the bottleneck pair
  for (const pair of result) {
    if (Math.abs(pair.cost - bottleneckDist) < 1e-6) {
      pair.isBottleneck = true;
      break; // only mark one
    }
  }

  return { pairs: result, bottleneckDist };
}

export default function BottleneckMatchingViz() {
  const [selectedPair, setSelectedPair] = useState<string>('circle-vs-cluster');
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 600;
  const height = 380;
  const margin = { top: 40, right: 30, bottom: 50, left: 50 };

  const data = diagramPairs[selectedPair];

  const { pairs, bottleneckDist } = useMemo(
    () => computeMatching(data.a, data.b),
    [data],
  );

  // Compute axis bounds from all points
  const maxVal = useMemo(() => {
    const allPoints = [...data.a, ...data.b];
    const vals = allPoints.flatMap((p) => [p.birth, p.death]);
    return Math.max(...vals) * 1.15 || 1;
  }, [data.a, data.b]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();

      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const xScale = d3.scaleLinear().domain([0, maxVal]).range([margin.left, margin.left + plotWidth]);
      const yScale = d3.scaleLinear().domain([0, maxVal]).range([margin.top + plotHeight, margin.top]);

      // Diagonal line
      svg
        .append('line')
        .attr('x1', xScale(0))
        .attr('y1', yScale(0))
        .attr('x2', xScale(maxVal))
        .attr('y2', yScale(maxVal))
        .attr('stroke', '#6b7280')
        .attr('stroke-dasharray', '4,4')
        .attr('stroke-opacity', 0.5);

      // Diagonal label
      svg
        .append('text')
        .attr('x', xScale(maxVal * 0.85))
        .attr('y', yScale(maxVal * 0.85) - 8)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', '#6b7280')
        .text('Δ (diagonal)');

      // Axes
      svg
        .append('g')
        .attr('transform', `translate(0,${margin.top + plotHeight})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('font-size', '11px');

      svg
        .append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('font-size', '11px');

      // Axis labels
      svg
        .append('text')
        .attr('x', margin.left + plotWidth / 2)
        .attr('y', height - 6)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .text('Birth');

      svg
        .append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -(margin.top + plotHeight / 2))
        .attr('y', 14)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .text('Death');

      // Matching lines
      const matchGroup = svg.append('g').attr('class', 'matchings');

      for (const pair of pairs) {
        matchGroup
          .append('line')
          .attr('x1', xScale(pair.a.x))
          .attr('y1', yScale(pair.a.y))
          .attr('x2', xScale(pair.b.x))
          .attr('y2', yScale(pair.b.y))
          .attr('stroke', pair.isBottleneck ? '#ef4444' : '#9ca3af')
          .attr('stroke-width', pair.isBottleneck ? 2.5 : 1.5)
          .attr('stroke-dasharray', pair.isBottleneck ? 'none' : '4,3')
          .attr('stroke-opacity', pair.isBottleneck ? 0.9 : 0.5);
      }

      // Diagram A points (filled circles)
      svg
        .selectAll('.point-a')
        .data(data.a)
        .join('circle')
        .attr('class', 'point-a')
        .attr('cx', (d) => xScale(d.birth))
        .attr('cy', (d) => yScale(d.death))
        .attr('r', 6)
        .attr('fill', (d) => dimensionColors[d.dimension])
        .attr('stroke', 'white')
        .attr('stroke-width', 1.5)
        .attr('fill-opacity', 0.9);

      // Diagram B points (outlined diamonds)
      const diamondSize = 7;
      svg
        .selectAll('.point-b')
        .data(data.b)
        .join('path')
        .attr('class', 'point-b')
        .attr('d', (d) => {
          const cx = xScale(d.birth);
          const cy = yScale(d.death);
          return `M${cx},${cy - diamondSize}L${cx + diamondSize},${cy}L${cx},${cy + diamondSize}L${cx - diamondSize},${cy}Z`;
        })
        .attr('fill', (d) => dimensionColors[d.dimension])
        .attr('stroke', 'white')
        .attr('stroke-width', 1.5)
        .attr('fill-opacity', 0.9);

      // Legend
      const legendX = margin.left + 8;
      const legendY = margin.top + 8;
      const legend = svg.append('g').attr('transform', `translate(${legendX}, ${legendY})`);

      // Diagram A legend
      legend.append('circle').attr('cx', 0).attr('cy', 0).attr('r', 5).attr('fill', dimensionColors[1]);
      legend
        .append('text')
        .attr('x', 12)
        .attr('y', 4)
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .text(data.labelA + ' (●)');

      // Diagram B legend
      legend
        .append('path')
        .attr('d', `M0,20L5,25L0,30L-5,25Z`)
        .attr('fill', dimensionColors[1]);
      legend
        .append('text')
        .attr('x', 12)
        .attr('y', 29)
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .text(data.labelB + ' (◆)');

      // Bottleneck line legend
      legend
        .append('line')
        .attr('x1', -5)
        .attr('y1', 50)
        .attr('x2', 5)
        .attr('y2', 50)
        .attr('stroke', '#ef4444')
        .attr('stroke-width', 2.5);
      legend
        .append('text')
        .attr('x', 12)
        .attr('y', 54)
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', '#ef4444')
        .text('Bottleneck match');
    },
    [data, pairs, width, height, maxVal],
  );

  return (
    <div ref={containerRef} className="w-full">
      <div className="mb-4 flex flex-wrap gap-2">
        {Object.entries(diagramPairs).map(([key, pair]) => (
          <button
            key={key}
            onClick={() => setSelectedPair(key)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              selectedPair === key
                ? 'bg-[var(--color-accent)] text-white'
                : 'bg-[var(--color-muted-bg)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border)]'
            }`}
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            {pair.name}
          </button>
        ))}
      </div>

      <svg role="img" aria-label="Bottleneck matching viz visualization"
        ref={svgRef}
        width={width}
        height={height}
        className="rounded-lg border border-[var(--color-border)]"
      />

      <div className="mt-3 rounded-md bg-[var(--color-muted-bg)] px-4 py-3" style={{ fontFamily: 'var(--font-sans)' }}>
        <p className="text-sm">
          <span className="font-semibold">Bottleneck distance:</span>{' '}
          <span className="font-mono text-[var(--color-accent)]">
            d<sub>B</sub> = {bottleneckDist.toFixed(4)}
          </span>
          <span className="ml-3 text-[var(--color-text-secondary)]">
            — the <span style={{ color: '#ef4444' }}>red line</span> shows the worst-cost match
          </span>
        </p>
        <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
          Circles (●) are {data.labelA} H₁ features. Diamonds (◆) are {data.labelB} H₁ features.
          Dashed lines show other matches (including to diagonal).
        </p>
      </div>
    </div>
  );
}

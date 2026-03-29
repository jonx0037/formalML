import { useState, useMemo, useCallback } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import type { Point2D, Simplex } from './shared/types';
import * as d3 from 'd3';

interface SimplicialComplexProps {
  points: Point2D[];
  epsilon: number;
  maxEpsilon?: number;
  showSlider?: boolean;
  highlightSimplices?: number[];
  colorScheme?: 'dimension' | 'birth-time';
  onEpsilonChange?: (eps: number) => void;
}

/** Compute pairwise distances between points. */
function pairwiseDistances(points: Point2D[]): Map<string, number> {
  const dists = new Map<string, number>();
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      dists.set(`${points[i].id}-${points[j].id}`, Math.sqrt(dx * dx + dy * dy));
    }
  }
  return dists;
}

/** Compute the Vietoris-Rips complex at a given epsilon. */
function vietorisRips(points: Point2D[], epsilon: number, distances: Map<string, number>): Simplex[] {
  const simplices: Simplex[] = [];

  // 0-simplices (vertices) — always present
  for (const p of points) {
    simplices.push({ vertices: [p.id], dimension: 0, birthTime: 0 });
  }

  // 1-simplices (edges)
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dist = distances.get(`${points[i].id}-${points[j].id}`) ?? Infinity;
      if (dist <= epsilon) {
        simplices.push({ vertices: [points[i].id, points[j].id], dimension: 1, birthTime: dist });
      }
    }
  }

  // 2-simplices (triangles) — three vertices all pairwise within epsilon
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      for (let k = j + 1; k < points.length; k++) {
        const dij = distances.get(`${points[i].id}-${points[j].id}`) ?? Infinity;
        const dik = distances.get(`${points[i].id}-${points[k].id}`) ?? Infinity;
        const djk = distances.get(`${points[j].id}-${points[k].id}`) ?? Infinity;
        const maxDist = Math.max(dij, dik, djk);
        if (maxDist <= epsilon) {
          simplices.push({
            vertices: [points[i].id, points[j].id, points[k].id],
            dimension: 2,
            birthTime: maxDist,
          });
        }
      }
    }
  }

  return simplices;
}

export default function SimplicialComplex({
  points,
  epsilon: initialEpsilon,
  maxEpsilon = 2,
  showSlider = true,
  highlightSimplices,
  colorScheme = 'dimension',
  onEpsilonChange,
}: SimplicialComplexProps) {
  const [epsilon, setEpsilon] = useState(initialEpsilon);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const width = containerWidth || 600;
  const height = 400;
  const margin = { top: 20, right: 20, bottom: 20, left: 20 };

  const distances = useMemo(() => pairwiseDistances(points), [points]);
  const simplices = useMemo(() => vietorisRips(points, epsilon, distances), [points, epsilon, distances]);

  const handleEpsilonChange = useCallback(
    (value: number) => {
      setEpsilon(value);
      onEpsilonChange?.(value);
    },
    [onEpsilonChange],
  );

  const pointMap = useMemo(() => new Map(points.map((p) => [p.id, p])), [points]);

  const xExtent = d3.extent(points, (p) => p.x) as [number, number];
  const yExtent = d3.extent(points, (p) => p.y) as [number, number];
  const pad = maxEpsilon * 0.3;

  const xScale = d3
    .scaleLinear()
    .domain([xExtent[0] - pad, xExtent[1] + pad])
    .range([margin.left, width - margin.right]);

  const yScale = d3
    .scaleLinear()
    .domain([yExtent[0] - pad, yExtent[1] + pad])
    .range([height - margin.bottom, margin.top]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();

      const shouldHighlight = (dim: number) => !highlightSimplices || highlightSimplices.includes(dim);

      // Draw 2-simplices (triangles)
      const triangles = simplices.filter((s) => s.dimension === 2);
      svg
        .selectAll('.triangle')
        .data(triangles)
        .join('polygon')
        .attr('class', 'triangle')
        .attr('points', (s) =>
          s.vertices
            .map((id) => {
              const p = pointMap.get(id)!;
              return `${xScale(p.x)},${yScale(p.y)}`;
            })
            .join(' '),
        )
        .attr('fill', dimensionColors[2])
        .attr('fill-opacity', shouldHighlight(2) ? 0.15 : 0.03)
        .attr('stroke', dimensionColors[2])
        .attr('stroke-opacity', shouldHighlight(2) ? 0.3 : 0.05)
        .attr('stroke-width', 1);

      // Draw 1-simplices (edges)
      const edges = simplices.filter((s) => s.dimension === 1);
      svg
        .selectAll('.edge')
        .data(edges)
        .join('line')
        .attr('class', 'edge')
        .attr('x1', (s) => xScale(pointMap.get(s.vertices[0])!.x))
        .attr('y1', (s) => yScale(pointMap.get(s.vertices[0])!.y))
        .attr('x2', (s) => xScale(pointMap.get(s.vertices[1])!.x))
        .attr('y2', (s) => yScale(pointMap.get(s.vertices[1])!.y))
        .attr('stroke', dimensionColors[1])
        .attr('stroke-opacity', shouldHighlight(1) ? 0.6 : 0.1)
        .attr('stroke-width', 1.5);

      // Draw 0-simplices (vertices)
      svg
        .selectAll('.vertex')
        .data(points)
        .join('circle')
        .attr('class', 'vertex')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', 4)
        .attr('fill', dimensionColors[0])
        .attr('fill-opacity', shouldHighlight(0) ? 1 : 0.2);

      // Draw epsilon-balls (translucent circles showing radius)
      svg
        .selectAll('.eps-ball')
        .data(points)
        .join('circle')
        .attr('class', 'eps-ball')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', xScale(xExtent[0] + epsilon / 2) - xScale(xExtent[0]))
        .attr('fill', dimensionColors[0])
        .attr('fill-opacity', 0.04)
        .attr('stroke', dimensionColors[0])
        .attr('stroke-opacity', 0.1)
        .attr('stroke-width', 0.5);
    },
    [simplices, points, epsilon, width, height, highlightSimplices],
  );

  return (
    <div ref={containerRef} className="w-full">
      <svg role="img" aria-label="Simplicial complex visualization" ref={svgRef} width={width} height={height} className="rounded-lg border border-[var(--color-border)]" />
      {showSlider && (
        <div className="mt-3 flex items-center gap-4">
          <label className="text-sm font-medium whitespace-nowrap" style={{ fontFamily: 'var(--font-sans)' }}>
            ε = {epsilon.toFixed(2)}
          </label>
          <input
            type="range"
            min={0}
            max={maxEpsilon}
            step={0.01}
            value={epsilon}
            onChange={(e) => handleEpsilonChange(parseFloat(e.target.value))}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
      )}
    </div>
  );
}

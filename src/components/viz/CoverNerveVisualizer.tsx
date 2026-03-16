import { useState, useMemo, useCallback } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import type { Point2D } from './shared/types';
import * as d3 from 'd3';

interface CoverNerveVisualizerProps {
  initialEpsilon?: number;
}

const DEFAULT_POINTS: Point2D[] = [
  { x: 0.0, y: 0.0, id: 'p0' },
  { x: 1.2, y: 0.3, id: 'p1' },
  { x: 0.5, y: 1.1, id: 'p2' },
  { x: 1.8, y: 1.0, id: 'p3' },
  { x: 0.8, y: -0.4, id: 'p4' },
];

const COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];

/** Euclidean distance between two points. */
function dist(a: Point2D, b: Point2D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Circumradius of a triangle with vertices a, b, c. Returns Infinity if collinear. */
function circumradius(a: Point2D, b: Point2D, c: Point2D): number {
  const ab = dist(a, b);
  const bc = dist(b, c);
  const ca = dist(c, a);
  // Area via cross product
  const area = Math.abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2;
  if (area < 1e-10) return Infinity;
  return (ab * bc * ca) / (4 * area);
}

/**
 * Radius of the smallest enclosing circle of three points.
 *
 * For three points, the minimal enclosing circle is:
 * - With radius maxEdge / 2 if the triangle is right, obtuse, or degenerate/collinear.
 * - Otherwise, the circumcircle of the triangle (circumradius).
 */
function minimalEnclosingCircleRadius(a: Point2D, b: Point2D, c: Point2D): number {
  const ab = dist(a, b);
  const bc = dist(b, c);
  const ca = dist(c, a);

  // Identify the longest side c and the other two sides a and b.
  let x = ab;
  let y = bc;
  let z = ca;

  // Ensure z is the longest side.
  if (x > y && x > z) {
    [x, z] = [z, x];
  } else if (y > x && y > z) {
    [y, z] = [z, y];
  }

  const aLen = x;
  const bLen = y;
  const cLen = z; // longest side

  // If angle opposite the longest side is >= 90 degrees (or points are collinear),
  // the minimal enclosing circle has radius cLen / 2.
  if (aLen * aLen + bLen * bLen <= cLen * cLen) {
    return cLen / 2;
  }

  // Acute triangle: minimal enclosing circle is the circumcircle.
  return circumradius(a, b, c);
}

interface NerveComplex {
  edges: [number, number][];
  triangles: [number, number, number][];
}

/** Compute the nerve complex: edges where balls overlap, triangles where triple intersections exist. */
function computeNerve(points: Point2D[], epsilon: number): NerveComplex {
  const n = points.length;
  const edges: [number, number][] = [];
  const triangles: [number, number, number][] = [];

  // Pairwise overlap: d(center_i, center_j) <= 2 * epsilon
  const threshold = 2 * epsilon;
  const overlaps = new Set<string>();

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (dist(points[i], points[j]) <= threshold) {
        edges.push([i, j]);
        overlaps.add(`${i}-${j}`);
      }
    }
  }

  // Triple intersection: all 3 pairwise overlap AND the smallest enclosing circle radius <= epsilon
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        if (
          overlaps.has(`${i}-${j}`) &&
          overlaps.has(`${i}-${k}`) &&
          overlaps.has(`${j}-${k}`)
        ) {
          const R = minimalEnclosingCircleRadius(points[i], points[j], points[k]);
          if (R <= epsilon) {
            triangles.push([i, j, k]);
          }
        }
      }
    }
  }

  return { edges, triangles };
}

/** Build D3 scales from points and dimensions. */
function buildScales(
  points: Point2D[],
  epsilon: number,
  width: number,
  height: number,
  margin: { top: number; right: number; bottom: number; left: number },
) {
  const xExtent = d3.extent(points, (p) => p.x) as [number, number];
  const yExtent = d3.extent(points, (p) => p.y) as [number, number];
  const pad = Math.max(epsilon, 0.5);

  const xScale = d3
    .scaleLinear()
    .domain([xExtent[0] - pad, xExtent[1] + pad])
    .range([margin.left, width - margin.right]);

  const yScale = d3
    .scaleLinear()
    .domain([yExtent[0] - pad, yExtent[1] + pad])
    .range([height - margin.bottom, margin.top]);

  return { xScale, yScale };
}

export default function CoverNerveVisualizer({
  initialEpsilon = 0.75,
}: CoverNerveVisualizerProps) {
  const [points, setPoints] = useState<Point2D[]>(DEFAULT_POINTS);
  const [epsilon, setEpsilon] = useState(initialEpsilon);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const panelWidth = useMemo(() => {
    if (!containerWidth) {
      return 300;
    }
    const gap = 16; // approximate horizontal gap/padding between panels
    const minWidth = 280;

    // For narrow viewports (stacked / single-column layout), use nearly full container width
    if (containerWidth < 600) {
      return Math.max(containerWidth - gap, minWidth);
    }

    // For wider viewports (two-column layout), use roughly half the container width
    return Math.max((containerWidth - gap) / 2, minWidth);
  }, [containerWidth]);
  const height = 360;
  const margin = { top: 24, right: 20, bottom: 20, left: 20 };

  const nerve = useMemo(() => computeNerve(points, epsilon), [points, epsilon]);

  const handleEpsilonChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setEpsilon(parseFloat(e.target.value));
  }, []);

  // ---- Left panel: The Cover ----
  const coverRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const { xScale, yScale } = buildScales(points, epsilon, panelWidth, height, margin);
      const epsilonPx = Math.abs(xScale(epsilon) - xScale(0));

      // Title
      svg
        .append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--color-text)')
        .attr('font-family', 'var(--font-sans)')
        .attr('font-size', 13)
        .attr('font-weight', 600)
        .text('The Cover');

      const g = svg.append('g');

      // Epsilon-balls: translucent fill + dashed border
      g.selectAll('.eps-ball')
        .data(points)
        .join('circle')
        .attr('class', 'eps-ball')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', epsilonPx)
        .attr('fill', (_, i) => COLORS[i])
        .attr('fill-opacity', 0.15)
        .attr('stroke', (_, i) => COLORS[i])
        .attr('stroke-opacity', 0.5)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3');

      // Draggable point circles
      const circles = g
        .selectAll('.point')
        .data(points)
        .join('circle')
        .attr('class', 'point')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', 6)
        .attr('fill', (_, i) => COLORS[i])
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .style('cursor', 'grab');

      // Labels U_i
      g.selectAll('.label')
        .data(points)
        .join('text')
        .attr('class', 'label')
        .attr('x', (p) => xScale(p.x))
        .attr('y', (p) => yScale(p.y) - 10)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--color-text)')
        .attr('font-family', 'var(--font-sans)')
        .attr('font-size', 11)
        .attr('font-weight', 500)
        .text((_, i) => `U${'\u2080\u2081\u2082\u2083\u2084'[i]}`);

      // Drag behavior
      const drag = d3
        .drag<SVGCircleElement, Point2D>()
        .on('start', function () {
          d3.select(this).style('cursor', 'grabbing');
        })
        .on('drag', function (event, d) {
          const newX = xScale.invert(event.x);
          const newY = yScale.invert(event.y);
          setPoints((prev) =>
            prev.map((p) => (p.id === d.id ? { ...p, x: newX, y: newY } : p)),
          );
        })
        .on('end', function () {
          d3.select(this).style('cursor', 'grab');
        });

      circles.call(drag);
    },
    [points, epsilon, panelWidth, height],
  );

  // ---- Right panel: The Nerve ----
  const nerveRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const { xScale, yScale } = buildScales(points, epsilon, panelWidth, height, margin);

      // Title
      svg
        .append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--color-text)')
        .attr('font-family', 'var(--font-sans)')
        .attr('font-size', 13)
        .attr('font-weight', 600)
        .text('The Nerve');

      const g = svg.append('g');

      // Filled triangles (triple intersections)
      g.selectAll('.triangle')
        .data(nerve.triangles)
        .join('polygon')
        .attr('class', 'triangle')
        .attr('points', ([i, j, k]) =>
          [points[i], points[j], points[k]]
            .map((p) => `${xScale(p.x)},${yScale(p.y)}`)
            .join(' '),
        )
        .attr('fill', '#6b7280')
        .attr('fill-opacity', 0.15)
        .attr('stroke', '#6b7280')
        .attr('stroke-opacity', 0.25)
        .attr('stroke-width', 0.5);

      // Edges (pairwise intersections)
      g.selectAll('.edge')
        .data(nerve.edges)
        .join('line')
        .attr('class', 'edge')
        .attr('x1', ([i]) => xScale(points[i].x))
        .attr('y1', ([i]) => yScale(points[i].y))
        .attr('x2', ([, j]) => xScale(points[j].x))
        .attr('y2', ([, j]) => yScale(points[j].y))
        .attr('stroke', 'var(--color-text)')
        .attr('stroke-opacity', 0.7)
        .attr('stroke-width', 2);

      // Vertices
      g.selectAll('.vertex')
        .data(points)
        .join('circle')
        .attr('class', 'vertex')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', 6)
        .attr('fill', (_, i) => COLORS[i])
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);

      // Labels v_i
      g.selectAll('.vlabel')
        .data(points)
        .join('text')
        .attr('class', 'vlabel')
        .attr('x', (p) => xScale(p.x))
        .attr('y', (p) => yScale(p.y) - 10)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--color-text)')
        .attr('font-family', 'var(--font-sans)')
        .attr('font-size', 11)
        .attr('font-weight', 500)
        .text((_, i) => `v${'\u2080\u2081\u2082\u2083\u2084'[i]}`);
    },
    [points, epsilon, panelWidth, height, nerve],
  );

  return (
    <div ref={containerRef} className="w-full">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <svg
          ref={coverRef}
          width={panelWidth}
          height={height}
          className="rounded-lg border border-[var(--color-border)]"
        />
        <svg
          ref={nerveRef}
          width={panelWidth}
          height={height}
          className="rounded-lg border border-[var(--color-border)]"
        />
      </div>
      <div className="mt-3 flex items-center gap-4">
        <label
          className="text-sm font-medium whitespace-nowrap"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          &epsilon; = {epsilon.toFixed(2)}
        </label>
        <input
          type="range"
          min={0}
          max={2.0}
          step={0.01}
          value={epsilon}
          onChange={handleEpsilonChange}
          className="w-full accent-[var(--color-accent)]"
        />
      </div>
    </div>
  );
}

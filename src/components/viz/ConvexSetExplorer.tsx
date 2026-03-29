import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  type Point,
  type ConvexSetDef,
  segmentMembership,
  computeConvexHull,
  clampPoint,
} from './shared/convexGeometry';
import * as d3 from 'd3';

/* ─── Set catalogue ─── */

interface SetEntry {
  label: string;
  def: ConvexSetDef;
  convex: boolean;
}

function centerPolygon(vertices: Point[]): Point[] {
  const cx = vertices.reduce((s, v) => s + v.x, 0) / vertices.length;
  const cy = vertices.reduce((s, v) => s + v.y, 0) / vertices.length;
  return vertices.map((v) => ({ x: v.x - cx, y: v.y - cy }));
}

function starVertices(outerR: number, innerR: number, points: number): Point[] {
  const verts: Point[] = [];
  for (let i = 0; i < points * 2; i++) {
    const angle = (Math.PI / 2) + (Math.PI * i) / points;
    const r = i % 2 === 0 ? outerR : innerR;
    verts.push({ x: r * Math.cos(angle), y: r * Math.sin(angle) });
  }
  return verts;
}

function hexagonVertices(r: number): Point[] {
  const verts: Point[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i;
    verts.push({ x: r * Math.cos(angle), y: r * Math.sin(angle) });
  }
  return verts;
}

const SET_CATALOGUE: SetEntry[] = [
  {
    label: 'Ellipse',
    def: { type: 'ellipse', cx: 0, cy: 0, a: 2.5, b: 1.5 },
    convex: true,
  },
  {
    label: 'Circle',
    def: { type: 'ellipse', cx: 0, cy: 0, a: 2, b: 2 },
    convex: true,
  },
  {
    label: 'Rectangle',
    def: { type: 'rectangle', xMin: -2.5, xMax: 2.5, yMin: -1.5, yMax: 1.5 },
    convex: true,
  },
  {
    label: 'Regular polygon (hexagon)',
    def: { type: 'polygon', vertices: hexagonVertices(2.2) },
    convex: true,
  },
  {
    label: 'L-shape',
    def: {
      type: 'polygon',
      vertices: centerPolygon([
        { x: 0, y: 0 },
        { x: 3, y: 0 },
        { x: 3, y: 1.2 },
        { x: 1.2, y: 1.2 },
        { x: 1.2, y: 3 },
        { x: 0, y: 3 },
      ]),
    },
    convex: false,
  },
  {
    label: 'Star',
    def: { type: 'polygon', vertices: starVertices(2.5, 1.0, 5) },
    convex: false,
  },
];

/* ─── Helpers ─── */

/** Convert segment membership data into contiguous runs for drawing. */
function membershipRuns(
  membership: { t: number; inside: boolean }[],
): { tStart: number; tEnd: number; inside: boolean }[] {
  if (membership.length === 0) return [];
  const runs: { tStart: number; tEnd: number; inside: boolean }[] = [];
  let current = membership[0].inside;
  let tStart = membership[0].t;

  for (let i = 1; i < membership.length; i++) {
    if (membership[i].inside !== current) {
      runs.push({ tStart, tEnd: membership[i - 1].t, inside: current });
      current = membership[i].inside;
      tStart = membership[i].t;
    }
  }
  runs.push({ tStart, tEnd: membership[membership.length - 1].t, inside: current });
  return runs;
}

/** Get SVG path data for a polygon set. */
function polygonPathData(vertices: Point[], xScale: d3.ScaleLinear<number, number>, yScale: d3.ScaleLinear<number, number>): string {
  return (
    vertices
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(v.x)},${yScale(v.y)}`)
      .join(' ') + ' Z'
  );
}

/* ─── Default draggable point positions ─── */

function defaultPoints(setDef: ConvexSetDef): [Point, Point] {
  switch (setDef.type) {
    case 'ellipse':
      return [
        { x: setDef.cx - setDef.a * 0.5, y: setDef.cy + setDef.b * 0.3 },
        { x: setDef.cx + setDef.a * 0.5, y: setDef.cy - setDef.b * 0.3 },
      ];
    case 'rectangle':
      return [
        { x: (setDef.xMin + setDef.xMax) / 2 - 1, y: (setDef.yMin + setDef.yMax) / 2 + 0.5 },
        { x: (setDef.xMin + setDef.xMax) / 2 + 1, y: (setDef.yMin + setDef.yMax) / 2 - 0.5 },
      ];
    case 'polygon': {
      const cx = setDef.vertices.reduce((s, v) => s + v.x, 0) / setDef.vertices.length;
      const cy = setDef.vertices.reduce((s, v) => s + v.y, 0) / setDef.vertices.length;
      return [
        { x: cx - 0.6, y: cy + 0.4 },
        { x: cx + 0.6, y: cy - 0.4 },
      ];
    }
  }
}

/* ─── Coordinate system constants ─── */

const VIEW_MIN = -4;
const VIEW_MAX = 4;
const VIEW_SIZE = VIEW_MAX - VIEW_MIN; // 8

const INSIDE_COLOR = '#22c55e';
const OUTSIDE_COLOR = '#ef4444';
const HULL_COLOR = '#3b82f6';
const POINT_A_COLOR = '#8b5cf6';
const POINT_B_COLOR = '#f59e0b';

/* ─── Component ─── */

export default function ConvexSetExplorer() {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [showHull, setShowHull] = useState(false);
  const [pointA, setPointA] = useState<Point>(() => defaultPoints(SET_CATALOGUE[0].def)[0]);
  const [pointB, setPointB] = useState<Point>(() => defaultPoints(SET_CATALOGUE[0].def)[1]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const entry = SET_CATALOGUE[selectedIdx];
  const setDef = entry.def;

  const handleSetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const idx = parseInt(e.target.value, 10);
      setSelectedIdx(idx);
      setShowHull(false);
      const [a, b] = defaultPoints(SET_CATALOGUE[idx].def);
      setPointA(a);
      setPointB(b);
    },
    [],
  );

  const handleHullToggle = useCallback(() => {
    setShowHull((prev) => !prev);
  }, []);

  // Compute membership along the segment
  const membership = useMemo(
    () => segmentMembership(pointA, pointB, setDef, 200),
    [pointA, pointB, setDef],
  );

  const runs = useMemo(() => membershipRuns(membership), [membership]);

  const allInside = useMemo(() => runs.every((r) => r.inside), [runs]);

  // Convex hull for non-convex polygon sets
  const hull = useMemo(() => {
    if (entry.convex || setDef.type !== 'polygon') return null;
    return computeConvexHull(setDef.vertices);
  }, [entry.convex, setDef]);

  // Responsive sizing
  const svgRef = useRef<SVGSVGElement>(null);

  const svgWidth = containerWidth ? Math.min(containerWidth, 600) : 0;
  const svgHeight = svgWidth; // square
  const margin = { top: 16, right: 16, bottom: 16, left: 16 };

  useEffect(() => {
    if (!svgRef.current || !svgWidth) return;
    const svg = d3.select(svgRef.current);

    const xScale = d3
      .scaleLinear()
      .domain([VIEW_MIN, VIEW_MAX])
      .range([margin.left, svgWidth - margin.right]);

    const yScale = d3
      .scaleLinear()
      .domain([VIEW_MIN, VIEW_MAX])
      .range([svgHeight - margin.bottom, margin.top]); // flip y
      svg.selectAll('*').remove();

      const g = svg.append('g');

      // ─── Draw the set ───

      if (setDef.type === 'ellipse') {
        g.append('ellipse')
          .attr('cx', xScale(setDef.cx))
          .attr('cy', yScale(setDef.cy))
          .attr('rx', Math.abs(xScale(setDef.cx + setDef.a) - xScale(setDef.cx)))
          .attr('ry', Math.abs(yScale(setDef.cy + setDef.b) - yScale(setDef.cy)))
          .style('fill', 'var(--color-accent)')
          .style('fill-opacity', '0.12')
          .style('stroke', 'var(--color-accent)')
          .style('stroke-opacity', '0.5')
          .style('stroke-width', '1.5');
      } else if (setDef.type === 'rectangle') {
        const x0 = xScale(setDef.xMin);
        const y0 = yScale(setDef.yMax); // yScale is flipped
        const w = xScale(setDef.xMax) - xScale(setDef.xMin);
        const h = yScale(setDef.yMin) - yScale(setDef.yMax);
        g.append('rect')
          .attr('x', x0)
          .attr('y', y0)
          .attr('width', w)
          .attr('height', h)
          .style('fill', 'var(--color-accent)')
          .style('fill-opacity', '0.12')
          .style('stroke', 'var(--color-accent)')
          .style('stroke-opacity', '0.5')
          .style('stroke-width', '1.5');
      } else if (setDef.type === 'polygon') {
        g.append('path')
          .attr('d', polygonPathData(setDef.vertices, xScale, yScale))
          .style('fill', 'var(--color-accent)')
          .style('fill-opacity', '0.12')
          .style('stroke', 'var(--color-accent)')
          .style('stroke-opacity', '0.5')
          .style('stroke-width', '1.5');
      }

      // ─── Convex hull overlay ───

      if (showHull && hull) {
        g.append('path')
          .attr('d', polygonPathData(hull, xScale, yScale))
          .style('fill', 'none')
          .style('stroke', HULL_COLOR)
          .style('stroke-width', '2')
          .style('stroke-dasharray', '6,4')
          .style('stroke-opacity', '0.7');
      }

      // ─── Draw segment runs (green/red) ───

      for (const run of runs) {
        const x1 = pointA.x + run.tStart * (pointB.x - pointA.x);
        const y1 = pointA.y + run.tStart * (pointB.y - pointA.y);
        const x2 = pointA.x + run.tEnd * (pointB.x - pointA.x);
        const y2 = pointA.y + run.tEnd * (pointB.y - pointA.y);

        g.append('line')
          .attr('x1', xScale(x1))
          .attr('y1', yScale(y1))
          .attr('x2', xScale(x2))
          .attr('y2', yScale(y2))
          .style('stroke', run.inside ? INSIDE_COLOR : OUTSIDE_COLOR)
          .style('stroke-width', '2.5')
          .style('stroke-linecap', 'round');
      }

      // ─── Draggable point A ───

      const circleA = g
        .append('circle')
        .datum({ id: 'A' })
        .attr('cx', xScale(pointA.x))
        .attr('cy', yScale(pointA.y))
        .attr('r', 7)
        .style('fill', POINT_A_COLOR)
        .style('stroke', '#fff')
        .style('stroke-width', '2')
        .style('cursor', 'grab');

      g.append('text')
        .attr('x', xScale(pointA.x) + 12)
        .attr('y', yScale(pointA.y) - 10)
        .attr('text-anchor', 'start')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 13)
        .attr('font-weight', 600)
        .text('x');

      // ─── Draggable point B ───

      const circleB = g
        .append('circle')
        .datum({ id: 'B' })
        .attr('cx', xScale(pointB.x))
        .attr('cy', yScale(pointB.y))
        .attr('r', 7)
        .style('fill', POINT_B_COLOR)
        .style('stroke', '#fff')
        .style('stroke-width', '2')
        .style('cursor', 'grab');

      g.append('text')
        .attr('x', xScale(pointB.x) + 12)
        .attr('y', yScale(pointB.y) - 10)
        .attr('text-anchor', 'start')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 13)
        .attr('font-weight', 600)
        .text('y');

      // ─── Drag behavior for point A ───

      const dragA = d3
        .drag<SVGCircleElement, { id: string }>()
        .on('start', function () {
          d3.select(this).style('cursor', 'grabbing');
        })
        .on('drag', function (event) {
          const newX = xScale.invert(event.x);
          const newY = yScale.invert(event.y);
          const clamped = clampPoint(
            { x: newX, y: newY },
            VIEW_MIN + 0.2,
            VIEW_MAX - 0.2,
            VIEW_MIN + 0.2,
            VIEW_MAX - 0.2,
          );
          setPointA(clamped);
        })
        .on('end', function () {
          d3.select(this).style('cursor', 'grab');
        });

      circleA.call(dragA);

      // ─── Drag behavior for point B ───

      const dragB = d3
        .drag<SVGCircleElement, { id: string }>()
        .on('start', function () {
          d3.select(this).style('cursor', 'grabbing');
        })
        .on('drag', function (event) {
          const newX = xScale.invert(event.x);
          const newY = yScale.invert(event.y);
          const clamped = clampPoint(
            { x: newX, y: newY },
            VIEW_MIN + 0.2,
            VIEW_MAX - 0.2,
            VIEW_MIN + 0.2,
            VIEW_MAX - 0.2,
          );
          setPointB(clamped);
        })
        .on('end', function () {
          d3.select(this).style('cursor', 'grab');
        });

      circleB.call(dragB);

      // ─── Status annotation ───

      const statusText = allInside
        ? 'Segment stays inside the set'
        : 'Segment exits the set — not convex!';
      const statusColor = allInside ? INSIDE_COLOR : OUTSIDE_COLOR;

      g.append('text')
        .attr('x', svgWidth / 2)
        .attr('y', svgHeight - 4)
        .attr('text-anchor', 'middle')
        .style('fill', statusColor)
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .text(statusText);
  }, [pointA, pointB, setDef, showHull, hull, runs, allInside, svgWidth, svgHeight]);

  if (!containerWidth) return <div ref={containerRef} style={{ minHeight: 400 }} />;

  return (
    <div ref={containerRef} className="w-full">
      <svg role="img" aria-label="Convex set explorer visualization"
        ref={svgRef}
        width={svgWidth}
        height={svgHeight}
        className="mx-auto block rounded-lg border border-[var(--color-border)]"
        style={{ background: 'var(--color-bg-subtle, transparent)' }}
      />

      {/* ─── Controls ─── */}
      <div
        className="mx-auto mt-3 flex flex-wrap items-center gap-4"
        style={{ maxWidth: svgWidth, fontFamily: 'var(--font-sans)' }}
      >
        <label className="flex items-center gap-2 text-sm font-medium">
          <span className="whitespace-nowrap">Set:</span>
          <select
            value={selectedIdx}
            onChange={handleSetChange}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-sm"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            {SET_CATALOGUE.map((s, i) => (
              <option key={s.label} value={i}>
                {s.label}
                {s.convex ? '' : ' (non-convex)'}
              </option>
            ))}
          </select>
        </label>

        {!entry.convex && (
          <label className="flex items-center gap-2 text-sm font-medium">
            <input
              type="checkbox"
              checked={showHull}
              onChange={handleHullToggle}
              className="accent-[var(--color-accent)]"
            />
            <span className="whitespace-nowrap">Show convex hull</span>
          </label>
        )}
      </div>

      {/* ─── Legend ─── */}
      <div
        className="mx-auto mt-2 flex flex-wrap items-center gap-4 text-xs"
        style={{ maxWidth: svgWidth, fontFamily: 'var(--font-sans)', color: 'var(--color-text-muted, #6b7280)' }}
      >
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-4 rounded-sm"
            style={{ background: INSIDE_COLOR }}
          />
          Inside
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-4 rounded-sm"
            style={{ background: OUTSIDE_COLOR }}
          />
          Outside
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-3 w-3 rounded-full"
            style={{ background: POINT_A_COLOR }}
          />
          x
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-3 w-3 rounded-full"
            style={{ background: POINT_B_COLOR }}
          />
          y
        </span>
        {showHull && !entry.convex && (
          <span className="flex items-center gap-1">
            <span
              className="inline-block h-0 w-4"
              style={{ borderTop: `2px dashed ${HULL_COLOR}` }}
            />
            Convex hull
          </span>
        )}
      </div>
    </div>
  );
}

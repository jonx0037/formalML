import { useResizeObserver } from './shared/useResizeObserver';
import {
  type Point,
  type EllipseDef,
  closestBoundaryPoint,
  dist,
  ellipseNormal,
  ellipseBoundary,
} from './shared/convexGeometry';
import * as d3 from 'd3';
import { useState, useEffect, useRef, useMemo } from 'react';

const SM_BREAKPOINT = 640;
const MARGIN = { top: 20, right: 20, bottom: 20, left: 20 };

const SET1_COLOR = '#2563eb';
const SET2_COLOR = '#dc2626';
const HYPERPLANE_COLOR = '#22c55e';
const NORMAL_COLOR = '#d97706';

export default function SeparatingHyperplaneExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [set1Center, setSet1Center] = useState<Point>({ x: -1.8, y: 0.2 });
  const [set2Center, setSet2Center] = useState<Point>({ x: 1.8, y: -0.2 });
  const [supportingMode, setSupportingMode] = useState(false);
  const [showHalfspaces, setShowHalfspaces] = useState(false);
  const [boundaryT, setBoundaryT] = useState(Math.PI / 3);

  const isNarrow = containerWidth < SM_BREAKPOINT;
  const svgHeight = isNarrow ? 300 : 400;
  const innerWidth = containerWidth - MARGIN.left - MARGIN.right;
  const innerHeight = svgHeight - MARGIN.top - MARGIN.bottom;

  // Scales: data coords [-4,4] x [-3,3] -> pixel coords
  const xScale = useMemo(
    () => d3.scaleLinear().domain([-4, 4]).range([0, innerWidth]),
    [innerWidth],
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain([-3, 3]).range([innerHeight, 0]),
    [innerHeight],
  );

  // ─── Separation mode computations ───
  const separationResult = useMemo(() => {
    if (supportingMode) return null;

    const set1: EllipseDef = {
      type: 'ellipse',
      cx: set1Center.x,
      cy: set1Center.y,
      a: 1.0,
      b: 0.8,
    };
    const set2: EllipseDef = {
      type: 'ellipse',
      cx: set2Center.x,
      cy: set2Center.y,
      a: 0.9,
      b: 1.1,
    };

    const boundary1 = ellipseBoundary(set1.cx, set1.cy, set1.a, set1.b, 100);
    const boundary2 = ellipseBoundary(set2.cx, set2.cy, set2.a, set2.b, 100);

    let minDist = Infinity;
    let closest1: Point = boundary1[0];
    let closest2: Point = boundary2[0];

    for (const p1 of boundary1) {
      for (const p2 of boundary2) {
        const d = dist(p1, p2);
        if (d < minDist) {
          minDist = d;
          closest1 = p1;
          closest2 = p2;
        }
      }
    }

    const overlapping = minDist < 0.05;

    if (overlapping) {
      return { overlapping: true as const, set1, set2 };
    }

    const midpoint: Point = {
      x: (closest1.x + closest2.x) / 2,
      y: (closest1.y + closest2.y) / 2,
    };

    // Normal direction: from closest1 to closest2
    const nx = closest2.x - closest1.x;
    const ny = closest2.y - closest1.y;
    const nLen = Math.sqrt(nx * nx + ny * ny);
    const normal: Point = { x: nx / nLen, y: ny / nLen };

    // Tangent direction (perpendicular to normal)
    const tangent: Point = { x: -normal.y, y: normal.x };

    return {
      overlapping: false as const,
      set1,
      set2,
      midpoint,
      normal,
      tangent,
      closest1,
      closest2,
    };
  }, [supportingMode, set1Center, set2Center]);

  // ─── Supporting mode computations ───
  const supportingResult = useMemo(() => {
    if (!supportingMode) return null;

    const a = 2.0;
    const b = 1.2;
    const cx = 0;
    const cy = 0;

    const boundaryPoint: Point = {
      x: cx + a * Math.cos(boundaryT),
      y: cy + b * Math.sin(boundaryT),
    };

    const normal = ellipseNormal(a, b, boundaryT);
    const tangent: Point = { x: -normal.y, y: normal.x };

    return {
      a,
      b,
      cx,
      cy,
      boundaryPoint,
      normal,
      tangent,
    };
  }, [supportingMode, boundaryT]);

  // ─── D3 rendering ───
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    if (!svgRef.current || innerWidth <= 0 || innerHeight <= 0) return;

    svg.selectAll('*').remove();

    // Defs for clip paths
    const defs = svg.append('defs');

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // ─── Supporting mode ───
    if (supportingMode && supportingResult) {
      const { a, b, cx, cy, boundaryPoint, normal, tangent } =
        supportingResult;

      // Draw ellipse fill
      g.append('ellipse')
        .attr('cx', xScale(cx))
        .attr('cy', yScale(cy))
        .attr('rx', xScale(cx + a) - xScale(cx))
        .attr('ry', yScale(cy) - yScale(cy + b))
        .style('fill', SET1_COLOR)
        .style('fill-opacity', 0.15)
        .style('stroke', SET1_COLOR)
        .style('stroke-width', 2);

      // Supporting hyperplane line
      const lineLen = 8;
      const lx1 = boundaryPoint.x - tangent.x * lineLen;
      const ly1 = boundaryPoint.y - tangent.y * lineLen;
      const lx2 = boundaryPoint.x + tangent.x * lineLen;
      const ly2 = boundaryPoint.y + tangent.y * lineLen;

      if (showHalfspaces) {
        // Halfspace shading for supporting mode
        // The normal points outward; shade the interior side (opposite normal) and exterior side
        const clipId1 = 'support-half-inner';
        const clipId2 = 'support-half-outer';

        // Build polygon vertices for each half
        const corners = [
          { x: -4, y: -3 },
          { x: 4, y: -3 },
          { x: 4, y: 3 },
          { x: -4, y: 3 },
        ];

        // Classify corners by which side of the hyperplane they fall on
        // The hyperplane passes through boundaryPoint with normal direction `normal`
        // dot(p - boundaryPoint, normal) > 0 => exterior side
        const exteriorPoly: Point[] = [];
        const interiorPoly: Point[] = [];

        // Use a polygon clipping approach: intersect rectangle with half-planes
        const rectPoly = [...corners];
        const splitPolygon = (
          poly: Point[],
          linePoint: Point,
          lineNormal: Point,
        ): { positive: Point[]; negative: Point[] } => {
          const positive: Point[] = [];
          const negative: Point[] = [];

          for (let i = 0; i < poly.length; i++) {
            const curr = poly[i];
            const next = poly[(i + 1) % poly.length];
            const dCurr =
              (curr.x - linePoint.x) * lineNormal.x +
              (curr.y - linePoint.y) * lineNormal.y;
            const dNext =
              (next.x - linePoint.x) * lineNormal.x +
              (next.y - linePoint.y) * lineNormal.y;

            if (dCurr >= 0) positive.push(curr);
            if (dCurr < 0) negative.push(curr);

            if ((dCurr > 0 && dNext < 0) || (dCurr < 0 && dNext > 0)) {
              const t = dCurr / (dCurr - dNext);
              const inter: Point = {
                x: curr.x + t * (next.x - curr.x),
                y: curr.y + t * (next.y - curr.y),
              };
              positive.push(inter);
              negative.push(inter);
            }
          }
          return { positive, negative };
        };

        const { positive: extPoly, negative: intPoly } = splitPolygon(
          rectPoly,
          boundaryPoint,
          normal,
        );

        if (extPoly.length >= 3) {
          g.append('polygon')
            .attr(
              'points',
              extPoly.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(' '),
            )
            .style('fill', SET2_COLOR)
            .style('fill-opacity', 0.08);
        }
        if (intPoly.length >= 3) {
          g.append('polygon')
            .attr(
              'points',
              intPoly.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(' '),
            )
            .style('fill', SET1_COLOR)
            .style('fill-opacity', 0.08);
        }
      }

      g.append('line')
        .attr('x1', xScale(lx1))
        .attr('y1', yScale(ly1))
        .attr('x2', xScale(lx2))
        .attr('y2', yScale(ly2))
        .style('stroke', HYPERPLANE_COLOR)
        .style('stroke-width', 2)
        .style('stroke-dasharray', '6,3');

      // Normal arrow from boundary point outward
      const arrowLen = 0.8;
      const arrowEnd: Point = {
        x: boundaryPoint.x + normal.x * arrowLen,
        y: boundaryPoint.y + normal.y * arrowLen,
      };

      // Arrow marker
      defs
        .append('marker')
        .attr('id', 'arrowhead-support')
        .attr('markerWidth', 10)
        .attr('markerHeight', 7)
        .attr('refX', 10)
        .attr('refY', 3.5)
        .attr('orient', 'auto')
        .append('polygon')
        .attr('points', '0 0, 10 3.5, 0 7')
        .style('fill', NORMAL_COLOR);

      g.append('line')
        .attr('x1', xScale(boundaryPoint.x))
        .attr('y1', yScale(boundaryPoint.y))
        .attr('x2', xScale(arrowEnd.x))
        .attr('y2', yScale(arrowEnd.y))
        .style('stroke', NORMAL_COLOR)
        .style('stroke-width', 2.5)
        .attr('marker-end', 'url(#arrowhead-support)');

      // Draggable boundary point
      const dragCircle = g
        .append('circle')
        .attr('cx', xScale(boundaryPoint.x))
        .attr('cy', yScale(boundaryPoint.y))
        .attr('r', 8)
        .style('fill', HYPERPLANE_COLOR)
        .style('stroke', 'white')
        .style('stroke-width', 2)
        .style('cursor', 'grab');

      const dragBehavior = d3
        .drag<SVGCircleElement, unknown>()
        .on('start', function () {
          d3.select(this).style('cursor', 'grabbing');
        })
        .on('drag', function (event) {
          const mouseX = xScale.invert(event.x);
          const mouseY = yScale.invert(event.y);
          const newT = Math.atan2(mouseY / b, mouseX / a);
          setBoundaryT(newT);
        })
        .on('end', function () {
          d3.select(this).style('cursor', 'grab');
        });

      dragCircle.call(dragBehavior);

      // Label
      g.append('text')
        .attr('x', xScale(0))
        .attr('y', yScale(2.6))
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Supporting hyperplane — drag the point along the boundary');

      return;
    }

    // ─── Separation mode ───
    if (!separationResult) return;

    const { set1, set2 } = separationResult;

    // Halfspace shading (drawn first, behind everything)
    if (
      showHalfspaces &&
      !separationResult.overlapping &&
      !separationResult.overlapping
    ) {
      const { midpoint, normal } = separationResult as {
        midpoint: Point;
        normal: Point;
        overlapping: false;
      };

      const corners = [
        { x: -4, y: -3 },
        { x: 4, y: -3 },
        { x: 4, y: 3 },
        { x: -4, y: 3 },
      ];

      const splitPolygon = (
        poly: Point[],
        linePoint: Point,
        lineNormal: Point,
      ): { positive: Point[]; negative: Point[] } => {
        const positive: Point[] = [];
        const negative: Point[] = [];

        for (let i = 0; i < poly.length; i++) {
          const curr = poly[i];
          const next = poly[(i + 1) % poly.length];
          const dCurr =
            (curr.x - linePoint.x) * lineNormal.x +
            (curr.y - linePoint.y) * lineNormal.y;
          const dNext =
            (next.x - linePoint.x) * lineNormal.x +
            (next.y - linePoint.y) * lineNormal.y;

          if (dCurr >= 0) positive.push(curr);
          if (dCurr < 0) negative.push(curr);

          if ((dCurr > 0 && dNext < 0) || (dCurr < 0 && dNext > 0)) {
            const t = dCurr / (dCurr - dNext);
            const inter: Point = {
              x: curr.x + t * (next.x - curr.x),
              y: curr.y + t * (next.y - curr.y),
            };
            positive.push(inter);
            negative.push(inter);
          }
        }
        return { positive, negative };
      };

      const { positive, negative } = splitPolygon(corners, midpoint, normal);

      if (negative.length >= 3) {
        g.append('polygon')
          .attr(
            'points',
            negative.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(' '),
          )
          .style('fill', SET1_COLOR)
          .style('fill-opacity', 0.08);
      }
      if (positive.length >= 3) {
        g.append('polygon')
          .attr(
            'points',
            positive.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(' '),
          )
          .style('fill', SET2_COLOR)
          .style('fill-opacity', 0.08);
      }
    }

    // Draw ellipses
    const drawEllipseGroup = (
      setDef: EllipseDef,
      color: string,
      label: string,
      onDrag: (newCenter: Point) => void,
    ) => {
      const ellipseG = g
        .append('g')
        .style('cursor', 'grab');

      ellipseG
        .append('ellipse')
        .attr('cx', xScale(setDef.cx))
        .attr('cy', yScale(setDef.cy))
        .attr('rx', xScale(setDef.cx + setDef.a) - xScale(setDef.cx))
        .attr('ry', yScale(setDef.cy) - yScale(setDef.cy + setDef.b))
        .style('fill', color)
        .style('fill-opacity', 0.15)
        .style('stroke', color)
        .style('stroke-width', 2);

      ellipseG
        .append('text')
        .attr('x', xScale(setDef.cx))
        .attr('y', yScale(setDef.cy))
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .style('fill', color)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .style('pointer-events', 'none')
        .text(label);

      const drag = d3
        .drag<SVGGElement, unknown>()
        .on('start', function () {
          d3.select(this).style('cursor', 'grabbing');
        })
        .on('drag', function (event) {
          const newX = Math.max(-3.5, Math.min(3.5, xScale.invert(event.x)));
          const newY = Math.max(-2.5, Math.min(2.5, yScale.invert(event.y)));
          onDrag({ x: newX, y: newY });
        })
        .on('end', function () {
          d3.select(this).style('cursor', 'grab');
        });

      ellipseG.call(drag);
    };

    drawEllipseGroup(set1, SET1_COLOR, 'C₁', setSet1Center);
    drawEllipseGroup(set2, SET2_COLOR, 'C₂', setSet2Center);

    if (separationResult.overlapping) {
      // Overlap warning
      g.append('text')
        .attr('x', xScale(0))
        .attr('y', yScale(2.6))
        .attr('text-anchor', 'middle')
        .style('fill', SET2_COLOR)
        .style('font-size', '14px')
        .style('font-weight', '600')
        .text('Sets overlap — no separating hyperplane exists');
    } else {
      const { midpoint, normal, tangent, closest1, closest2 } =
        separationResult;

      // Closest pair dashed line
      g.append('line')
        .attr('x1', xScale(closest1.x))
        .attr('y1', yScale(closest1.y))
        .attr('x2', xScale(closest2.x))
        .attr('y2', yScale(closest2.y))
        .style('stroke', 'var(--color-muted)')
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4,3');

      // Separating hyperplane line
      const lineLen = 8;
      const lx1 = midpoint.x - tangent.x * lineLen;
      const ly1 = midpoint.y - tangent.y * lineLen;
      const lx2 = midpoint.x + tangent.x * lineLen;
      const ly2 = midpoint.y + tangent.y * lineLen;

      g.append('line')
        .attr('x1', xScale(lx1))
        .attr('y1', yScale(ly1))
        .attr('x2', xScale(lx2))
        .attr('y2', yScale(ly2))
        .style('stroke', HYPERPLANE_COLOR)
        .style('stroke-width', 2.5);

      // Normal arrow at midpoint
      const arrowLen = 0.7;
      const arrowEnd: Point = {
        x: midpoint.x + normal.x * arrowLen,
        y: midpoint.y + normal.y * arrowLen,
      };

      defs
        .append('marker')
        .attr('id', 'arrowhead-sep')
        .attr('markerWidth', 10)
        .attr('markerHeight', 7)
        .attr('refX', 10)
        .attr('refY', 3.5)
        .attr('orient', 'auto')
        .append('polygon')
        .attr('points', '0 0, 10 3.5, 0 7')
        .style('fill', NORMAL_COLOR);

      g.append('line')
        .attr('x1', xScale(midpoint.x))
        .attr('y1', yScale(midpoint.y))
        .attr('x2', xScale(arrowEnd.x))
        .attr('y2', yScale(arrowEnd.y))
        .style('stroke', NORMAL_COLOR)
        .style('stroke-width', 2.5)
        .attr('marker-end', 'url(#arrowhead-sep)');

      // Normal label
      g.append('text')
        .attr('x', xScale(arrowEnd.x) + 6)
        .attr('y', yScale(arrowEnd.y) - 4)
        .style('fill', NORMAL_COLOR)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('n');

      // Title
      g.append('text')
        .attr('x', xScale(0))
        .attr('y', yScale(2.6))
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Separating hyperplane — drag each set to explore');
    }
  }, [
    containerWidth,
    innerWidth,
    innerHeight,
    xScale,
    yScale,
    supportingMode,
    supportingResult,
    separationResult,
    showHalfspaces,
  ]);

  if (!containerWidth) {
    return <div ref={containerRef} style={{ minHeight: 400 }} />;
  }

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg
        ref={svgRef}
        width={containerWidth}
        height={svgHeight}
        style={{
          display: 'block',
          border: '1px solid var(--color-border)',
          borderRadius: 8,
          background: 'var(--color-surface, transparent)',
        }}
      />
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          marginTop: '0.75rem',
          fontSize: '0.875rem',
          color: 'var(--color-text)',
        }}
      >
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={supportingMode}
            onChange={(e) => setSupportingMode(e.target.checked)}
          />
          Supporting hyperplane mode
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showHalfspaces}
            onChange={(e) => setShowHalfspaces(e.target.checked)}
          />
          Show halfspaces
        </label>
      </div>
    </div>
  );
}

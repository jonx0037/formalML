import { useState, useMemo, useCallback } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import * as d3 from 'd3';

// Geometry constants for an equilateral triangle with side length s = 1.0
const SIDE = 1.0;
const CIRCUMRADIUS = 1 / Math.sqrt(3); // ≈ 0.5774
const EDGE_THRESHOLD = SIDE / 2; // ε ≥ 0.5 ⟹ 2ε ≥ 1.0
const VERTICES: [number, number][] = [
  [0, 0],
  [1, 0],
  [0.5, Math.sqrt(3) / 2],
];
const VERTEX_LABELS = ['v₀', 'v₁', 'v₂'];

/**
 * PhantomCycleExplorer
 *
 * Visualizes the divergence between the Čech and Vietoris-Rips complexes
 * on three points forming an equilateral triangle. In the "phantom window"
 * ε ∈ [0.500, 0.577], VR fills the triangle (killing the 1-cycle) while
 * Čech correctly preserves it — demonstrating that VR can introduce
 * phantom simplices not justified by the underlying geometry.
 */
export default function PhantomCycleExplorer() {
  const [epsilon, setEpsilon] = useState(0.3);
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const handleEpsilonChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setEpsilon(parseFloat(e.target.value));
  }, []);

  // Derived topology state
  const topology = useMemo(() => {
    const hasEdges = epsilon >= EDGE_THRESHOLD;
    const vrHasTriangle = hasEdges; // VR: triangle iff all pairwise edges exist
    const cechHasTriangle = epsilon >= CIRCUMRADIUS;
    const edgeCount = hasEdges ? 3 : 0;
    const vrTriangleCount = vrHasTriangle ? 1 : 0;
    const cechTriangleCount = cechHasTriangle ? 1 : 0;
    // Čech β₁: 1 when edges form a cycle but triangle hasn't filled it
    const cechBeta1 = hasEdges && !cechHasTriangle ? 1 : 0;
    // VR β₁: always 0 — edges and triangle appear simultaneously
    const vrBeta1 = 0;
    const inPhantomWindow = hasEdges && !cechHasTriangle;

    return {
      hasEdges,
      vrHasTriangle,
      cechHasTriangle,
      edgeCount,
      vrTriangleCount,
      cechTriangleCount,
      cechBeta1,
      vrBeta1,
      inPhantomWindow,
    };
  }, [epsilon]);

  // SVG dimensions
  const width = Math.min(containerWidth || 500, 500);
  const height = Math.max(300, width * 0.75);
  const margin = { top: 30, right: 30, bottom: 30, left: 30 };

  const xScale = d3
    .scaleLinear()
    .domain([-0.3, 1.3])
    .range([margin.left, width - margin.right]);

  const yScale = d3
    .scaleLinear()
    .domain([-0.3, 1.2])
    .range([height - margin.bottom, margin.top]);

  // Radius in SVG pixels for ε-balls
  const epsPixelRadius = Math.abs(xScale(epsilon) - xScale(0));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();

      const g = svg.append('g');

      // --- ε-balls (translucent circles) ---
      VERTICES.forEach(([vx, vy], i) => {
        // Filled ball
        g.append('circle')
          .attr('cx', xScale(vx))
          .attr('cy', yScale(vy))
          .attr('r', epsPixelRadius)
          .attr('fill', dimensionColors[i])
          .attr('fill-opacity', 0.12)
          .attr('stroke', dimensionColors[i])
          .attr('stroke-opacity', 0.35)
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '4 3');
      });

      // --- Edges ---
      if (topology.hasEdges) {
        const edgePairs: [number, number][] = [
          [0, 1],
          [1, 2],
          [0, 2],
        ];
        edgePairs.forEach(([i, j]) => {
          g.append('line')
            .attr('x1', xScale(VERTICES[i][0]))
            .attr('y1', yScale(VERTICES[i][1]))
            .attr('x2', xScale(VERTICES[j][0]))
            .attr('y2', yScale(VERTICES[j][1]))
            .style('stroke', 'var(--color-text)')
            .attr('stroke-opacity', 0.8)
            .attr('stroke-width', 2);
        });
      }

      // --- Triangle fill ---
      if (topology.vrHasTriangle || topology.cechHasTriangle) {
        const polyPoints = VERTICES.map(([vx, vy]) => `${xScale(vx)},${yScale(vy)}`).join(' ');

        if (topology.inPhantomWindow) {
          // VR has triangle but Čech doesn't — phantom triangle
          g.append('polygon')
            .attr('points', polyPoints)
            .attr('fill', '#9CA3AF') // gray
            .attr('fill-opacity', 0.2)
            .attr('stroke', '#9CA3AF')
            .attr('stroke-opacity', 0.4)
            .attr('stroke-width', 1);

          // "VR only" label at centroid
          const cx = xScale((VERTICES[0][0] + VERTICES[1][0] + VERTICES[2][0]) / 3);
          const cy = yScale((VERTICES[0][1] + VERTICES[1][1] + VERTICES[2][1]) / 3);
          g.append('text')
            .attr('x', cx)
            .attr('y', cy)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', '#6B7280')
            .attr('font-size', '12px')
            .style('font-family', 'var(--font-sans)')
            .attr('font-style', 'italic')
            .text('VR only');
        } else if (topology.cechHasTriangle) {
          // Both complexes have the triangle
          g.append('polygon')
            .attr('points', polyPoints)
            .attr('fill', '#3B82F6') // blue
            .attr('fill-opacity', 0.18)
            .attr('stroke', '#3B82F6')
            .attr('stroke-opacity', 0.4)
            .attr('stroke-width', 1);
        }
      }

      // --- Vertex points (on top) ---
      VERTICES.forEach(([vx, vy], i) => {
        g.append('circle')
          .attr('cx', xScale(vx))
          .attr('cy', yScale(vy))
          .attr('r', 5)
          .style('fill', 'var(--color-surface)')
          .style('stroke', 'var(--color-text)')
          .attr('stroke-width', 1.5);

        // Vertex labels
        const labelOffsetX = i === 0 ? -12 : i === 1 ? 12 : 0;
        const labelOffsetY = i === 2 ? -14 : 14;
        g.append('text')
          .attr('x', xScale(vx) + labelOffsetX)
          .attr('y', yScale(vy) + labelOffsetY)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('fill', 'var(--color-text)')
          .attr('font-size', '13px')
          .style('font-family', 'var(--font-sans)')
          .text(VERTEX_LABELS[i]);
      });
    },
    [epsilon, topology, width, height, epsPixelRadius, xScale, yScale],
  );

  // Determine diverging rows for highlighting
  const trianglesDiverge = topology.inPhantomWindow;
  const beta1Diverges = topology.inPhantomWindow;

  return (
    <div ref={containerRef} className="w-full">
      {/* Two-panel grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left panel: SVG visualization */}
        <div>
          <svg role="img" aria-label="Phantom cycle explorer visualization"
            ref={svgRef}
            width={width}
            height={height}
            className="rounded-lg border border-[var(--color-border)]"
          />
        </div>

        {/* Right panel: comparison table */}
        <div className="flex flex-col justify-center">
          <table
            className="w-full text-sm border-collapse rounded-lg overflow-hidden"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            <thead>
              <tr>
                <th
                  className="px-4 py-2 text-left border-b border-[var(--color-border)]"
                  style={{ color: 'var(--color-text)' }}
                >
                  Simplex
                </th>
                <th
                  className="px-4 py-2 text-center border-b border-[var(--color-border)]"
                  style={{ color: 'var(--color-text)' }}
                >
                  Cech
                </th>
                <th
                  className="px-4 py-2 text-center border-b border-[var(--color-border)]"
                  style={{ color: 'var(--color-text)' }}
                >
                  VR
                </th>
              </tr>
            </thead>
            <tbody>
              <TableRow label="Vertices" cech={3} vr={3} diverges={false} />
              <TableRow label="Edges" cech={topology.edgeCount} vr={topology.edgeCount} diverges={false} />
              <TableRow
                label="Triangles"
                cech={topology.cechTriangleCount}
                vr={topology.vrTriangleCount}
                diverges={trianglesDiverge}
              />
              <TableRow
                label={<>&beta;<sub>1</sub></>}
                cech={topology.cechBeta1}
                vr={topology.vrBeta1}
                diverges={beta1Diverges}
              />
            </tbody>
          </table>
        </div>
      </div>

      {/* Slider */}
      <div className="mt-5">
        <div className="flex items-center gap-4 mb-1">
          <label
            className="text-sm font-medium whitespace-nowrap"
            style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}
          >
            &epsilon; = {epsilon.toFixed(3)}
          </label>
          <input
            type="range"
            min={0}
            max={1.0}
            step={0.001}
            value={epsilon}
            onChange={handleEpsilonChange}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>

        {/* Tick marks */}
        <div className="relative w-full h-6" style={{ fontFamily: 'var(--font-sans)' }}>
          {/* VR fills tick at ε = 0.500 → 50% of range */}
          <div
            className="absolute text-xs"
            style={{ left: '50%', transform: 'translateX(-50%)', color: 'var(--color-text)', opacity: 0.6 }}
          >
            <span className="block text-center">|</span>
            <span className="block text-center whitespace-nowrap">&epsilon;=0.500 VR fills</span>
          </div>
          {/* Čech fills tick at ε ≈ 0.577 → 57.7% of range */}
          <div
            className="absolute text-xs"
            style={{
              left: `${(CIRCUMRADIUS / 1.0) * 100}%`,
              transform: 'translateX(-50%)',
              color: 'var(--color-text)',
              opacity: 0.6,
            }}
          >
            <span className="block text-center">|</span>
            <span className="block text-center whitespace-nowrap">&epsilon;=0.577 Čech fills</span>
          </div>
        </div>
      </div>

      {/* Phantom window banner */}
      {topology.inPhantomWindow && (
        <div
          className="mt-4 px-4 py-3 rounded-lg text-sm font-medium text-center"
          style={{
            background: 'rgba(217, 119, 6, 0.12)',
            border: '1px solid rgba(217, 119, 6, 0.35)',
            color: '#B45309',
            fontFamily: 'var(--font-sans)',
          }}
        >
          Phantom window: VR says no loop — Čech says loop exists
        </div>
      )}
    </div>
  );
}

/** A single row of the comparison table, with optional divergence highlighting. */
function TableRow({
  label,
  cech,
  vr,
  diverges,
}: {
  label: React.ReactNode;
  cech: number;
  vr: number;
  diverges: boolean;
}) {
  const highlightStyle = diverges
    ? { background: 'rgba(217, 119, 6, 0.10)', fontWeight: 600 as const }
    : {};

  return (
    <tr>
      <td
        className="px-4 py-2 border-b border-[var(--color-border)]"
        style={{ color: 'var(--color-text)' }}
      >
        {label}
      </td>
      <td
        className="px-4 py-2 text-center border-b border-[var(--color-border)]"
        style={{ color: 'var(--color-text)', ...highlightStyle }}
      >
        {cech}
      </td>
      <td
        className="px-4 py-2 text-center border-b border-[var(--color-border)]"
        style={{ color: 'var(--color-text)', ...highlightStyle }}
      >
        {vr}
      </td>
    </tr>
  );
}

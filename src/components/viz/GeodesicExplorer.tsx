import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  type Vec2,
  spherePoint,
  orthoProject,
  isVisible,
  sphereWireframe,
  solveGeodesicS2,
  solveGeodesicPoincare,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 380;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

type SurfaceType = 'sphere' | 'poincare' | 'flat';

const fmt = (x: number) => x.toFixed(2);

// ── Component ────────────────────────────────────────────────────────

export default function GeodesicExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [surface, setSurface] = useState<SurfaceType>('sphere');
  const [theta, setTheta] = useState(Math.PI / 3);
  const [phi, setPhi] = useState(0.0);
  const [px, setPx] = useState(0.0);
  const [py, setPy] = useState(0.0);
  const [dirAngle, setDirAngle] = useState(0.5);
  const [showAllRays, setShowAllRays] = useState(false);
  const [showGrid, setShowGrid] = useState(false);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const svgWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.58);
  const tangentWidth = isStacked ? containerWidth : containerWidth - svgWidth;

  // ── Geodesics computation ──────────────────────────────────────────
  const geodesics = useMemo(() => {
    const nRays = showAllRays ? 12 : 1;
    const rays: { x: number; y: number }[][] = [];
    const tMax = surface === 'sphere' ? Math.PI * 0.85 : 2.0;

    for (let r = 0; r < nRays; r++) {
      const angle = nRays === 1 ? dirAngle : dirAngle + (2 * Math.PI * r) / nRays;

      if (surface === 'sphere') {
        const dth = Math.cos(angle);
        const dph = Math.sin(angle) / (Math.sin(theta) + 1e-12);
        const pts = solveGeodesicS2(theta, phi, dth, dph, tMax, 200);
        rays.push(pts.map((p) => ({ x: p.theta, y: p.phi })));
      } else if (surface === 'poincare') {
        const speed = 0.3;
        const dx = speed * Math.cos(angle);
        const dy = speed * Math.sin(angle);
        const pts = solveGeodesicPoincare(px, py, dx, dy, tMax, 200);
        rays.push(pts.map((p) => ({ x: p.x, y: p.y })));
      } else {
        // Flat: straight lines
        const dx = Math.cos(angle);
        const dy = Math.sin(angle);
        const pts: { x: number; y: number }[] = [];
        for (let i = 0; i <= 100; i++) {
          const t = (i / 100) * 2;
          pts.push({ x: px + dx * t, y: py + dy * t });
        }
        rays.push(pts);
      }
    }
    return rays;
  }, [surface, theta, phi, px, py, dirAngle, showAllRays]);

  // ── Sphere / Flat Surface SVG ──────────────────────────────────────
  const surfaceRef = useD3<SVGSVGElement>(
    (svg) => {
      if (svgWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = svgWidth;
      const h = HEIGHT;
      const cx = w / 2;
      const cy = h / 2;

      if (surface === 'sphere') {
        const scale = Math.min(w, h) * 0.38;
        const rotY = -0.3;
        const rotX = -0.4;

        // Wireframe
        const { lines, backLines } = sphereWireframe(8, 12, rotY, rotX);
        const lineGen = d3.line<Vec2>().x((d) => cx + d.x * scale).y((d) => cy - d.y * scale);

        backLines.forEach((pts) => {
          svg.append('path')
            .attr('d', lineGen(pts))
            .attr('fill', 'none')
            .style('stroke', 'var(--color-text-muted, #999)')
            .style('stroke-width', '0.5')
            .style('opacity', '0.2');
        });
        lines.forEach((pts) => {
          svg.append('path')
            .attr('d', lineGen(pts))
            .attr('fill', 'none')
            .style('stroke', 'var(--color-text-muted, #999)')
            .style('stroke-width', '0.5')
            .style('opacity', '0.4');
        });

        // Normal coordinate grid
        if (showGrid) {
          const nGrid = 8;
          const gridMax = Math.PI * 0.4;
          for (let i = 0; i <= nGrid; i++) {
            const frac = (i / nGrid) * 2 - 1;
            const gridLine: Vec2[] = [];
            for (let j = 0; j <= 40; j++) {
              const s = (j / 40) * 2 - 1;
              const gth = theta + frac * gridMax * 0.3;
              const gph = phi + s * gridMax * 0.3 / (Math.sin(theta) + 1e-6);
              const p3 = spherePoint(gth, gph);
              if (isVisible(p3, rotY, rotX)) {
                gridLine.push(orthoProject(p3, rotY, rotX));
              }
            }
            if (gridLine.length > 1) {
              svg.append('path')
                .attr('d', lineGen(gridLine))
                .attr('fill', 'none')
                .style('stroke', AMBER)
                .style('stroke-width', '0.6')
                .style('opacity', '0.25');
            }
            // Cross lines
            const gridLine2: Vec2[] = [];
            for (let j = 0; j <= 40; j++) {
              const s = (j / 40) * 2 - 1;
              const gth = theta + s * gridMax * 0.3;
              const gph = phi + frac * gridMax * 0.3 / (Math.sin(theta) + 1e-6);
              const p3 = spherePoint(gth, gph);
              if (isVisible(p3, rotY, rotX)) {
                gridLine2.push(orthoProject(p3, rotY, rotX));
              }
            }
            if (gridLine2.length > 1) {
              svg.append('path')
                .attr('d', lineGen(gridLine2))
                .attr('fill', 'none')
                .style('stroke', AMBER)
                .style('stroke-width', '0.6')
                .style('opacity', '0.25');
            }
          }
        }

        // Geodesics
        geodesics.forEach((ray, ri) => {
          const projected: Vec2[] = [];
          for (const pt of ray) {
            const p3 = spherePoint(pt.x, pt.y);
            if (isVisible(p3, rotY, rotX)) {
              projected.push(orthoProject(p3, rotY, rotX));
            }
          }
          if (projected.length > 1) {
            svg.append('path')
              .attr('d', lineGen(projected))
              .attr('fill', 'none')
              .style('stroke', ri === 0 ? TEAL : AMBER)
              .style('stroke-width', ri === 0 ? '2.5' : '1.5')
              .style('opacity', ri === 0 ? '1' : '0.5');
          }
        });

        // Base point
        const p3Base = spherePoint(theta, phi);
        if (isVisible(p3Base, rotY, rotX)) {
          const proj = orthoProject(p3Base, rotY, rotX);
          svg.append('circle')
            .attr('cx', cx + proj.x * scale)
            .attr('cy', cy - proj.y * scale)
            .attr('r', 6)
            .style('fill', PURPLE)
            .style('stroke', '#fff')
            .style('stroke-width', '2');
        }

        // Label
        svg.append('text')
          .attr('x', 8)
          .attr('y', 18)
          .style('fill', 'var(--color-text-muted, #666)')
          .style('font-size', '11px')
          .text('Sphere S²');

      } else if (surface === 'poincare') {
        const scale = Math.min(w, h) * 0.38;

        // Unit disk boundary
        svg.append('circle')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', scale)
          .style('fill', 'none')
          .style('stroke', 'var(--color-text-muted, #999)')
          .style('stroke-width', '1.5');

        // Grid circles
        for (let r = 0.2; r < 1; r += 0.2) {
          svg.append('circle')
            .attr('cx', cx)
            .attr('cy', cy)
            .attr('r', r * scale)
            .style('fill', 'none')
            .style('stroke', 'var(--color-text-muted, #999)')
            .style('stroke-width', '0.4')
            .style('opacity', '0.3');
        }

        // Geodesics
        geodesics.forEach((ray, ri) => {
          const projected = ray
            .filter((p) => p.x * p.x + p.y * p.y < 0.99)
            .map((p) => ({ x: p.x, y: p.y }));
          if (projected.length > 1) {
            const pathGen = d3.line<{ x: number; y: number }>()
              .x((d) => cx + d.x * scale)
              .y((d) => cy - d.y * scale);
            svg.append('path')
              .attr('d', pathGen(projected))
              .attr('fill', 'none')
              .style('stroke', ri === 0 ? TEAL : AMBER)
              .style('stroke-width', ri === 0 ? '2.5' : '1.5')
              .style('opacity', ri === 0 ? '1' : '0.5');
          }
        });

        // Base point
        svg.append('circle')
          .attr('cx', cx + px * scale)
          .attr('cy', cy - py * scale)
          .attr('r', 6)
          .style('fill', PURPLE)
          .style('stroke', '#fff')
          .style('stroke-width', '2');

        svg.append('text')
          .attr('x', 8)
          .attr('y', 18)
          .style('fill', 'var(--color-text-muted, #666)')
          .style('font-size', '11px')
          .text('Poincaré Disk 𝔻²');

      } else {
        // Flat R²
        const scale = Math.min(w, h) * 0.15;

        // Grid
        for (let i = -3; i <= 3; i++) {
          svg.append('line')
            .attr('x1', cx + i * scale).attr('y1', cy - 3 * scale)
            .attr('x2', cx + i * scale).attr('y2', cy + 3 * scale)
            .style('stroke', 'var(--color-text-muted, #999)')
            .style('stroke-width', '0.4')
            .style('opacity', '0.3');
          svg.append('line')
            .attr('x1', cx - 3 * scale).attr('y1', cy + i * scale)
            .attr('x2', cx + 3 * scale).attr('y2', cy + i * scale)
            .style('stroke', 'var(--color-text-muted, #999)')
            .style('stroke-width', '0.4')
            .style('opacity', '0.3');
        }

        // Geodesics (straight lines)
        geodesics.forEach((ray, ri) => {
          const pathGen = d3.line<{ x: number; y: number }>()
            .x((d) => cx + d.x * scale)
            .y((d) => cy - d.y * scale);
          svg.append('path')
            .attr('d', pathGen(ray))
            .attr('fill', 'none')
            .style('stroke', ri === 0 ? TEAL : AMBER)
            .style('stroke-width', ri === 0 ? '2.5' : '1.5')
            .style('opacity', ri === 0 ? '1' : '0.5');
        });

        // Base point
        svg.append('circle')
          .attr('cx', cx + px * scale)
          .attr('cy', cy - py * scale)
          .attr('r', 6)
          .style('fill', PURPLE)
          .style('stroke', '#fff')
          .style('stroke-width', '2');

        svg.append('text')
          .attr('x', 8)
          .attr('y', 18)
          .style('fill', 'var(--color-text-muted, #666)')
          .style('font-size', '11px')
          .text('Flat ℝ²');
      }
    },
    [surface, theta, phi, px, py, geodesics, svgWidth, showGrid]
  );

  // ── Tangent Space SVG ──────────────────────────────────────────────
  const tangentRef = useD3<SVGSVGElement>(
    (svg) => {
      if (tangentWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = tangentWidth;
      const h = HEIGHT;
      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w, h) * 0.1;

      // Axes
      svg.append('line')
        .attr('x1', cx - scale * 3.5).attr('y1', cy)
        .attr('x2', cx + scale * 3.5).attr('y2', cy)
        .style('stroke', 'var(--color-text-muted, #999)')
        .style('stroke-width', '0.5');
      svg.append('line')
        .attr('x1', cx).attr('y1', cy - scale * 3.5)
        .attr('x2', cx).attr('y2', cy + scale * 3.5)
        .style('stroke', 'var(--color-text-muted, #999)')
        .style('stroke-width', '0.5');

      // Injectivity radius circle
      const injRad = surface === 'sphere' ? Math.PI : 3;
      const injScale = Math.min(injRad * scale, Math.min(w, h) * 0.42);
      svg.append('circle')
        .attr('cx', cx)
        .attr('cy', cy)
        .attr('r', injScale)
        .style('fill', 'none')
        .style('stroke', PURPLE)
        .style('stroke-width', '1')
        .style('stroke-dasharray', '4,3')
        .style('opacity', '0.6');

      svg.append('text')
        .attr('x', cx + injScale + 4)
        .attr('y', cy - 4)
        .style('fill', PURPLE)
        .style('font-size', '10px')
        .text(surface === 'sphere' ? 'inj = π' : 'inj = ∞');

      // Straight rays (pre-images of geodesics)
      const nRays = showAllRays ? 12 : 1;
      for (let r = 0; r < nRays; r++) {
        const angle = nRays === 1 ? dirAngle : dirAngle + (2 * Math.PI * r) / nRays;
        const rayLen = Math.min(injScale, Math.min(w, h) * 0.42);
        svg.append('line')
          .attr('x1', cx)
          .attr('y1', cy)
          .attr('x2', cx + Math.cos(angle) * rayLen)
          .attr('y2', cy - Math.sin(angle) * rayLen)
          .style('stroke', r === 0 ? TEAL : AMBER)
          .style('stroke-width', r === 0 ? '2' : '1.2')
          .style('opacity', r === 0 ? '1' : '0.5');
      }

      // Origin
      svg.append('circle')
        .attr('cx', cx)
        .attr('cy', cy)
        .attr('r', 4)
        .style('fill', PURPLE)
        .style('stroke', '#fff')
        .style('stroke-width', '1.5');

      // Labels
      svg.append('text')
        .attr('x', 8)
        .attr('y', 18)
        .style('fill', 'var(--color-text-muted, #666)')
        .style('font-size', '11px')
        .text('Tangent space T_pM');

      svg.append('text')
        .attr('x', cx + 4)
        .attr('y', cy + 14)
        .style('fill', 'var(--color-text-muted, #666)')
        .style('font-size', '10px')
        .text('0');
    },
    [surface, dirAngle, showAllRays, tangentWidth]
  );

  // ── Render ─────────────────────────────────────────────────────────
  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border p-3"
      style={{ borderColor: 'var(--color-border, #e5e7eb)', background: 'var(--color-bg-secondary, #f9fafb)' }}
    >
      <div className="mb-2 text-sm font-semibold" style={{ color: 'var(--color-text-primary, #111)' }}>
        Geodesic Explorer
      </div>

      <div style={{ display: 'flex', flexDirection: isStacked ? 'column' : 'row', gap: '4px' }}>
        <svg
          ref={surfaceRef}
          width={svgWidth}
          height={HEIGHT}
          style={{ background: 'var(--color-bg-primary, #fff)', borderRadius: '6px' }}
        />
        <svg
          ref={tangentRef}
          width={tangentWidth}
          height={HEIGHT}
          style={{ background: 'var(--color-bg-primary, #fff)', borderRadius: '6px' }}
        />
      </div>

      {/* Controls */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Surface:</span>
          <select
            value={surface}
            onChange={(e) => {
              const s = e.target.value as SurfaceType;
              setSurface(s);
              if (s === 'sphere') { setTheta(Math.PI / 3); setPhi(0); }
              else { setPx(0); setPy(0); }
            }}
            className="rounded border px-1.5 py-0.5 text-xs"
            style={{ borderColor: 'var(--color-border, #d1d5db)', background: 'var(--color-bg-primary, #fff)', color: 'var(--color-text-primary, #111)' }}
          >
            <option value="sphere">Sphere S²</option>
            <option value="poincare">Poincaré Disk 𝔻²</option>
            <option value="flat">Flat ℝ²</option>
          </select>
        </label>

        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Direction:</span>
          <input
            type="range"
            min={0}
            max={2 * Math.PI}
            step={0.1}
            value={dirAngle}
            onChange={(e) => setDirAngle(parseFloat(e.target.value))}
            className="w-20"
          />
          <span className="font-mono w-10">{fmt(dirAngle)}</span>
        </label>

        {surface === 'sphere' && (
          <label className="flex items-center gap-1.5">
            <span style={{ color: 'var(--color-text-muted, #666)' }}>θ:</span>
            <input
              type="range"
              min={0.2}
              max={Math.PI - 0.2}
              step={0.05}
              value={theta}
              onChange={(e) => setTheta(parseFloat(e.target.value))}
              className="w-20"
            />
            <span className="font-mono w-10">{fmt(theta)}</span>
          </label>
        )}

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showAllRays} onChange={(e) => setShowAllRays(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Show all rays</span>
        </label>

        {surface === 'sphere' && (
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
            <span style={{ color: 'var(--color-text-muted, #666)' }}>Normal grid</span>
          </label>
        )}
      </div>
    </div>
  );
}

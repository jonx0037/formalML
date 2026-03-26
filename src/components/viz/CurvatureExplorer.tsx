import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  type Vec2,
  torusPoint,
  orthoProject,
  isVisible,
  torusCurvature,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 380;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];

type SurfaceType = 'sphere' | 'poincare' | 'torus' | 'flat';

const fmt = (x: number) => x.toFixed(3);

// ── Component ────────────────────────────────────────────────────────

export default function CurvatureExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [surface, setSurface] = useState<SurfaceType>('torus');
  const [uParam, setUParam] = useState(0.0);
  const [vParam, setVParam] = useState(0.0);
  const [showTriangle, setShowTriangle] = useState(false);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const svgWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.6);
  const infoWidth = isStacked ? containerWidth : containerWidth - svgWidth;

  // ── Curvature at current point ─────────────────────────────────────
  const curvature = useMemo(() => {
    if (surface === 'sphere') return { K: 1, ric1: 1, ric2: 1, scalar: 2 };
    if (surface === 'poincare') return { K: -1, ric1: -1, ric2: -1, scalar: -2 };
    if (surface === 'flat') return { K: 0, ric1: 0, ric2: 0, scalar: 0 };
    // Torus
    const K = torusCurvature(vParam);
    return { K, ric1: K, ric2: K, scalar: 2 * K };
  }, [surface, vParam]);

  const angleExcess = useMemo(() => {
    // Angle excess of a small geodesic triangle ≈ K * A
    // Use a small area A = 0.1
    const A = 0.1;
    return curvature.K * A;
  }, [curvature.K]);

  // ── Curvature color scale ──────────────────────────────────────────
  const colorScale = useMemo(() => {
    return d3.scaleLinear<string>()
      .domain([-1.5, 0, 1.5])
      .range([PURPLE, '#94a3b8', TEAL])
      .clamp(true);
  }, []);

  // ── Surface SVG ────────────────────────────────────────────────────
  const surfaceRef = useD3<SVGSVGElement>(
    (svg) => {
      if (svgWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = svgWidth;
      const h = HEIGHT;
      const cx = w / 2;
      const cy = h / 2;

      if (surface === 'torus') {
        const scale = Math.min(w, h) * 0.13;
        const rotY = -0.3;
        const rotX = -0.5;

        // Draw curvature-colored torus patches
        const nU = 30;
        const nV = 20;
        for (let i = 0; i < nU; i++) {
          for (let j = 0; j < nV; j++) {
            const u = (i / nU) * 2 * Math.PI;
            const v = (j / nV) * 2 * Math.PI;
            const p = torusPoint(u, v);
            if (!isVisible(p, rotY, rotX)) continue;
            const proj = orthoProject(p, rotY, rotX);
            const K = torusCurvature(v);
            svg.append('circle')
              .attr('cx', cx + proj.x * scale)
              .attr('cy', cy - proj.y * scale)
              .attr('r', 3.5)
              .style('fill', colorScale(K))
              .style('opacity', '0.7');
          }
        }

        // Highlight current point
        const pt = torusPoint(uParam, vParam);
        if (isVisible(pt, rotY, rotX)) {
          const proj = orthoProject(pt, rotY, rotX);
          svg.append('circle')
            .attr('cx', cx + proj.x * scale)
            .attr('cy', cy - proj.y * scale)
            .attr('r', 7)
            .style('fill', colorScale(curvature.K))
            .style('stroke', '#fff')
            .style('stroke-width', '2');
        }

        // Triangle indicator
        if (showTriangle && isVisible(pt, rotY, rotX)) {
          const proj = orthoProject(pt, rotY, rotX);
          const triSize = 12;
          const tri = d3.line<[number, number]>()
            .x((d) => d[0]).y((d) => d[1]);
          const triPts: [number, number][] = [
            [cx + proj.x * scale, cy - proj.y * scale - triSize],
            [cx + proj.x * scale - triSize * 0.87, cy - proj.y * scale + triSize * 0.5],
            [cx + proj.x * scale + triSize * 0.87, cy - proj.y * scale + triSize * 0.5],
            [cx + proj.x * scale, cy - proj.y * scale - triSize],
          ];
          // Small triangle indicator at the current point
          svg.append('path')
            .attr('d', tri(triPts))
            .style('fill', 'none')
            .style('stroke', '#fff')
            .style('stroke-width', '2');
        }

        svg.append('text')
          .attr('x', 8).attr('y', 18)
          .style('fill', 'var(--color-text-muted, #666)')
          .style('font-size', '11px')
          .text('Torus (curvature varies)');

      } else if (surface === 'sphere') {
        const scale = Math.min(w, h) * 0.38;
        // Uniform positive curvature — all teal
        svg.append('circle')
          .attr('cx', cx).attr('cy', cy)
          .attr('r', scale)
          .style('fill', colorScale(1))
          .style('opacity', '0.15');
        svg.append('circle')
          .attr('cx', cx).attr('cy', cy)
          .attr('r', scale)
          .style('fill', 'none')
          .style('stroke', TEAL)
          .style('stroke-width', '1.5');

        svg.append('text')
          .attr('x', 8).attr('y', 18)
          .style('fill', 'var(--color-text-muted, #666)')
          .style('font-size', '11px')
          .text('Sphere S² (K = 1)');

      } else if (surface === 'poincare') {
        const scale = Math.min(w, h) * 0.38;
        svg.append('circle')
          .attr('cx', cx).attr('cy', cy)
          .attr('r', scale)
          .style('fill', colorScale(-1))
          .style('opacity', '0.15');
        svg.append('circle')
          .attr('cx', cx).attr('cy', cy)
          .attr('r', scale)
          .style('fill', 'none')
          .style('stroke', PURPLE)
          .style('stroke-width', '1.5');

        svg.append('text')
          .attr('x', 8).attr('y', 18)
          .style('fill', 'var(--color-text-muted, #666)')
          .style('font-size', '11px')
          .text('Poincaré Disk (K = −1)');

      } else {
        const scale = Math.min(w, h) * 0.15;
        for (let i = -3; i <= 3; i++) {
          svg.append('line')
            .attr('x1', cx + i * scale).attr('y1', cy - 3 * scale)
            .attr('x2', cx + i * scale).attr('y2', cy + 3 * scale)
            .style('stroke', '#94a3b8').style('stroke-width', '0.4').style('opacity', '0.3');
          svg.append('line')
            .attr('x1', cx - 3 * scale).attr('y1', cy + i * scale)
            .attr('x2', cx + 3 * scale).attr('y2', cy + i * scale)
            .style('stroke', '#94a3b8').style('stroke-width', '0.4').style('opacity', '0.3');
        }
        svg.append('text')
          .attr('x', 8).attr('y', 18)
          .style('fill', 'var(--color-text-muted, #666)')
          .style('font-size', '11px')
          .text('Flat ℝ² (K = 0)');
      }
    },
    [surface, uParam, vParam, svgWidth, showTriangle, colorScale, curvature.K]
  );

  // ── Render ─────────────────────────────────────────────────────────
  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border p-3"
      style={{ borderColor: 'var(--color-border, #e5e7eb)', background: 'var(--color-bg-secondary, #f9fafb)' }}
    >
      <div className="mb-2 text-sm font-semibold" style={{ color: 'var(--color-text-primary, #111)' }}>
        Curvature Explorer
      </div>

      <div style={{ display: 'flex', flexDirection: isStacked ? 'column' : 'row', gap: '8px' }}>
        <svg
          ref={surfaceRef}
          width={svgWidth}
          height={HEIGHT}
          style={{ background: 'var(--color-bg-primary, #fff)', borderRadius: '6px' }}
        />

        {/* Info panel */}
        <div style={{ width: isStacked ? '100%' : infoWidth, padding: '12px' }}>
          <div className="space-y-3">
            <div>
              <div className="mb-1 text-xs font-medium" style={{ color: TEAL }}>
                Sectional curvature K
              </div>
              <div className="font-mono text-lg font-bold" style={{ color: colorScale(curvature.K) }}>
                {fmt(curvature.K)}
              </div>
            </div>

            <div>
              <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-text-muted, #666)' }}>
                Ricci eigenvalues
              </div>
              <div className="font-mono text-xs">
                λ₁ = {fmt(curvature.ric1)}, λ₂ = {fmt(curvature.ric2)}
              </div>
            </div>

            <div>
              <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-text-muted, #666)' }}>
                Scalar curvature S
              </div>
              <div className="font-mono text-xs">{fmt(curvature.scalar)}</div>
            </div>

            <div className="text-xs" style={{ color: 'var(--color-text-muted, #666)' }}>
              In dim 2: K = S/2
            </div>

            {showTriangle && (
              <div>
                <div className="mb-1 text-xs font-medium" style={{ color: PURPLE }}>
                  Angle excess (small triangle)
                </div>
                <div className="font-mono text-xs">
                  Σαᵢ − π ≈ {fmt(angleExcess)} rad
                </div>
              </div>
            )}

            <div className="mt-2 flex items-center gap-2 text-xs">
              <span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 2, background: TEAL }} />
              <span>K &gt; 0</span>
              <span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 2, background: '#94a3b8' }} />
              <span>K = 0</span>
              <span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 2, background: PURPLE }} />
              <span>K &lt; 0</span>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Surface:</span>
          <select
            value={surface}
            onChange={(e) => setSurface(e.target.value as SurfaceType)}
            className="rounded border px-1.5 py-0.5 text-xs"
            style={{ borderColor: 'var(--color-border, #d1d5db)', background: 'var(--color-bg-primary, #fff)', color: 'var(--color-text-primary, #111)' }}
          >
            <option value="sphere">Sphere S² (K = 1)</option>
            <option value="poincare">Poincaré Disk (K = −1)</option>
            <option value="torus">Torus (K varies)</option>
            <option value="flat">Flat Plane (K = 0)</option>
          </select>
        </label>

        {surface === 'torus' && (
          <>
            <label className="flex items-center gap-1.5">
              <span style={{ color: 'var(--color-text-muted, #666)' }}>u:</span>
              <input type="range" min={0} max={2 * Math.PI} step={0.1} value={uParam}
                onChange={(e) => setUParam(parseFloat(e.target.value))} className="w-20" />
            </label>
            <label className="flex items-center gap-1.5">
              <span style={{ color: 'var(--color-text-muted, #666)' }}>v:</span>
              <input type="range" min={0} max={2 * Math.PI} step={0.1} value={vParam}
                onChange={(e) => setVParam(parseFloat(e.target.value))} className="w-20" />
              <span className="font-mono w-12">K={fmt(curvature.K)}</span>
            </label>
          </>
        )}

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showTriangle} onChange={(e) => setShowTriangle(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Show geodesic triangle</span>
        </label>
      </div>
    </div>
  );
}

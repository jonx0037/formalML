import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  type Vec2,
  spherePoint,
  ellipsoidPoint,
  torusPoint,
  orthoProject,
  isVisible,
  torusCurvature,
  totalCurvature,
  torusAreaElement,
  sphereAreaElement,
  ellipsoidAreaElement,
  ellipsoidCurvature,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 380;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];

type TopologyType = 'sphere' | 'torus' | 'doubleTorus';

const fmt = (x: number) => x.toFixed(3);

const eulerChar: Record<TopologyType, number> = { sphere: 2, torus: 0, doubleTorus: -2 };
const topologyLabel: Record<TopologyType, string> = {
  sphere: 'Sphere (χ = 2)',
  torus: 'Torus (χ = 0)',
  doubleTorus: 'Double Torus (χ = −2)',
};

// ── Component ────────────────────────────────────────────────────────

export default function GaussBonnetExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [topology, setTopology] = useState<TopologyType>('sphere');
  const [deformation, setDeformation] = useState(0);
  const [showColoring, setShowColoring] = useState(true);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const svgWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.6);
  const infoWidth = isStacked ? containerWidth : containerWidth - svgWidth;

  const chi = eulerChar[topology];
  const target = 2 * Math.PI * chi;

  // ── Compute total curvature ────────────────────────────────────────
  const computed = useMemo(() => {
    if (topology === 'sphere') {
      const a = 1;
      const b = 1 + deformation * 1.5; // Deform into an ellipsoid
      const integral = totalCurvature(
        (th) => ellipsoidCurvature(th, a, b),
        (th, ph) => ellipsoidAreaElement(th, ph, a, b),
        [0.01, Math.PI - 0.01],
        [0, 2 * Math.PI],
        60,
        60
      );
      return { integral, posContrib: integral, negContrib: 0 };
    }
    if (topology === 'torus') {
      const R = 2 + deformation * 0.8;
      const r = 0.8;
      let pos = 0;
      let neg = 0;
      const nU = 60;
      const nV = 60;
      const du = (2 * Math.PI) / nU;
      const dv = (2 * Math.PI) / nV;
      for (let i = 0; i < nU; i++) {
        const u = (i + 0.5) * du;
        for (let j = 0; j < nV; j++) {
          const v = (j + 0.5) * dv;
          const K = torusCurvature(v, R, r);
          const dA = torusAreaElement(u, v, R, r) * du * dv;
          if (K >= 0) pos += K * dA;
          else neg += K * dA;
        }
      }
      return { integral: pos + neg, posContrib: pos, negContrib: neg };
    }
    // Double torus: use the formula directly (χ = -2)
    return { integral: -4 * Math.PI, posContrib: 0, negContrib: -4 * Math.PI };
  }, [topology, deformation]);

  // ── Color scale ────────────────────────────────────────────────────
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
      const rotY = -0.3;
      const rotX = -0.5;

      if (topology === 'sphere') {
        const a = 1;
        const b = 1 + deformation * 1.5;
        const scale = Math.min(w, h) * 0.35;

        if (showColoring) {
          const nPts = 35;
          for (let i = 0; i < nPts; i++) {
            for (let j = 0; j < nPts * 2; j++) {
              const th = 0.1 + (i / nPts) * (Math.PI - 0.2);
              const ph = (j / (nPts * 2)) * 2 * Math.PI;
              const p3 = ellipsoidPoint(th, ph, a, b);
              if (!isVisible(p3, rotY, rotX)) continue;
              const proj = orthoProject(p3, rotY, rotX);
              const K = ellipsoidCurvature(th, a, b);
              svg.append('circle')
                .attr('cx', cx + proj.x * scale)
                .attr('cy', cy - proj.y * scale)
                .attr('r', 3)
                .style('fill', colorScale(K))
                .style('opacity', '0.6');
            }
          }
        }

        // Wireframe outline
        for (let i = 0; i <= 8; i++) {
          const th = 0.1 + (i / 8) * (Math.PI - 0.2);
          const pts: Vec2[] = [];
          for (let j = 0; j <= 60; j++) {
            const ph = (j / 60) * 2 * Math.PI;
            const p3 = ellipsoidPoint(th, ph, a, b);
            if (isVisible(p3, rotY, rotX)) {
              pts.push(orthoProject(p3, rotY, rotX));
            }
          }
          if (pts.length > 1) {
            const lineGen = d3.line<Vec2>().x((d) => cx + d.x * scale).y((d) => cy - d.y * scale);
            svg.append('path')
              .attr('d', lineGen(pts))
              .attr('fill', 'none')
              .style('stroke', 'var(--color-text-muted, #999)')
              .style('stroke-width', '0.4')
              .style('opacity', '0.3');
          }
        }

      } else if (topology === 'torus') {
        const R = 2 + deformation * 0.8;
        const r = 0.8;
        const scale = Math.min(w, h) * 0.12;

        if (showColoring) {
          const nPts = 30;
          for (let i = 0; i < nPts; i++) {
            for (let j = 0; j < nPts; j++) {
              const u = (i / nPts) * 2 * Math.PI;
              const v = (j / nPts) * 2 * Math.PI;
              const p3 = torusPoint(u, v, R, r);
              if (!isVisible(p3, rotY, rotX)) continue;
              const proj = orthoProject(p3, rotY, rotX);
              const K = torusCurvature(v, R, r);
              svg.append('circle')
                .attr('cx', cx + proj.x * scale)
                .attr('cy', cy - proj.y * scale)
                .attr('r', 3)
                .style('fill', colorScale(K))
                .style('opacity', '0.6');
            }
          }
        }
      } else {
        // Double torus: 2D schematic
        const scale = Math.min(w, h) * 0.25;

        // Draw two connected torus shapes schematically
        const drawHandle = (ox: number) => {
          svg.append('ellipse')
            .attr('cx', cx + ox).attr('cy', cy)
            .attr('rx', scale * 0.45).attr('ry', scale * 0.55)
            .style('fill', showColoring ? colorScale(-0.5) : 'none')
            .style('fill-opacity', showColoring ? 0.15 : 0)
            .style('stroke', PURPLE)
            .style('stroke-width', '1.5');
          // Hole
          svg.append('ellipse')
            .attr('cx', cx + ox).attr('cy', cy)
            .attr('rx', scale * 0.2).attr('ry', scale * 0.15)
            .style('fill', 'var(--color-bg-primary, #fff)')
            .style('stroke', 'var(--color-text-muted, #999)')
            .style('stroke-width', '1');
        };
        drawHandle(-scale * 0.35);
        drawHandle(scale * 0.35);

        svg.append('text')
          .attr('x', cx).attr('y', cy + scale * 0.8)
          .attr('text-anchor', 'middle')
          .style('fill', PURPLE)
          .style('font-size', '11px')
          .text('genus 2: χ = −2');
      }

      svg.append('text')
        .attr('x', 8).attr('y', 18)
        .style('fill', 'var(--color-text-muted, #666)')
        .style('font-size', '11px')
        .text(topologyLabel[topology]);
    },
    [topology, deformation, svgWidth, showColoring, colorScale]
  );

  // ── Render ─────────────────────────────────────────────────────────
  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border p-3"
      style={{ borderColor: 'var(--color-border, #e5e7eb)', background: 'var(--color-bg-secondary, #f9fafb)' }}
    >
      <div className="mb-2 text-sm font-semibold" style={{ color: 'var(--color-text-primary, #111)' }}>
        Gauss–Bonnet Explorer
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
                Euler characteristic χ(M)
              </div>
              <div className="font-mono text-lg font-bold" style={{ color: 'var(--color-text-primary, #111)' }}>
                {chi}
              </div>
            </div>

            <div>
              <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-text-muted, #666)' }}>
                Target: 2πχ(M)
              </div>
              <div className="font-mono text-sm">{fmt(target)}</div>
            </div>

            <div>
              <div className="mb-1 text-xs font-medium" style={{ color: PURPLE }}>
                Computed ∫K dA
              </div>
              <div className="font-mono text-sm font-bold" style={{ color: colorScale(computed.integral / (4 * Math.PI)) }}>
                {fmt(computed.integral)}
              </div>
            </div>

            {/* Bar chart: positive vs negative */}
            <div>
              <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-text-muted, #666)' }}>
                Curvature distribution
              </div>
              <div className="flex items-end gap-1" style={{ height: 40 }}>
                {computed.posContrib > 0 && (
                  <div
                    style={{
                      width: 24,
                      height: Math.min(40, Math.max(4, Math.abs(computed.posContrib) * 3)),
                      background: TEAL,
                      borderRadius: 2,
                      opacity: 0.7,
                    }}
                    title={`Positive: ${fmt(computed.posContrib)}`}
                  />
                )}
                {computed.negContrib < 0 && (
                  <div
                    style={{
                      width: 24,
                      height: Math.min(40, Math.max(4, Math.abs(computed.negContrib) * 3)),
                      background: PURPLE,
                      borderRadius: 2,
                      opacity: 0.7,
                    }}
                    title={`Negative: ${fmt(computed.negContrib)}`}
                  />
                )}
              </div>
              <div className="flex gap-2 mt-1 text-xs" style={{ color: 'var(--color-text-muted, #666)' }}>
                {computed.posContrib > 0 && <span style={{ color: TEAL }}>+{fmt(computed.posContrib)}</span>}
                {computed.negContrib < 0 && <span style={{ color: PURPLE }}>{fmt(computed.negContrib)}</span>}
              </div>
            </div>

            <div className="text-xs" style={{ color: 'var(--color-text-muted, #666)' }}>
              Total curvature = 2πχ regardless of deformation — a topological invariant.
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Surface:</span>
          <select
            value={topology}
            onChange={(e) => { setTopology(e.target.value as TopologyType); setDeformation(0); }}
            className="rounded border px-1.5 py-0.5 text-xs"
            style={{ borderColor: 'var(--color-border, #d1d5db)', background: 'var(--color-bg-primary, #fff)', color: 'var(--color-text-primary, #111)' }}
          >
            <option value="sphere">Sphere (χ = 2)</option>
            <option value="torus">Torus (χ = 0)</option>
            <option value="doubleTorus">Double Torus (χ = −2)</option>
          </select>
        </label>

        {topology !== 'doubleTorus' && (
          <label className="flex items-center gap-1.5">
            <span style={{ color: 'var(--color-text-muted, #666)' }}>Deformation:</span>
            <input
              type="range" min={0} max={1} step={0.05} value={deformation}
              onChange={(e) => setDeformation(parseFloat(e.target.value))}
              className="w-24"
            />
          </label>
        )}

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showColoring} onChange={(e) => setShowColoring(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Curvature coloring</span>
        </label>
      </div>
    </div>
  );
}

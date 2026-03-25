import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  type Vec2,
  type Vec3,
  spherePoint,
  torusPoint,
  paraboloidPoint,
  surfaceTangents,
  orthoProject,
  vec3Add,
  vec3Scale,
  vec3Normalize,
  vec3Cross,
  vec3Norm,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 340;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

type SurfaceType = 'sphere' | 'torus' | 'paraboloid';

interface SurfaceConfig {
  label: string;
  surface: (u: number, v: number) => Vec3;
  uRange: [number, number];
  vRange: [number, number];
  uSteps: number;
  vSteps: number;
  defaultU: number;
  defaultV: number;
  rotY: number;
  rotX: number;
  scale: number;
}

const SURFACES: Record<SurfaceType, SurfaceConfig> = {
  sphere: {
    label: 'Sphere S²',
    surface: spherePoint,
    uRange: [0.1, Math.PI - 0.1],
    vRange: [0, 2 * Math.PI],
    uSteps: 10,
    vSteps: 16,
    defaultU: 1.0,
    defaultV: 0.8,
    rotY: -0.3,
    rotX: -0.4,
    scale: 100,
  },
  torus: {
    label: 'Torus T²',
    surface: (u: number, v: number) => torusPoint(u, v, 2, 0.8),
    uRange: [0, 2 * Math.PI],
    vRange: [0, 2 * Math.PI],
    uSteps: 16,
    vSteps: 12,
    defaultU: 0.5,
    defaultV: 0.8,
    rotY: -0.3,
    rotX: -0.5,
    scale: 48,
  },
  paraboloid: {
    label: 'Paraboloid',
    surface: paraboloidPoint,
    uRange: [0, 1.5],
    vRange: [0, 2 * Math.PI],
    uSteps: 8,
    vSteps: 16,
    defaultU: 0.8,
    defaultV: 1.0,
    rotY: -0.4,
    rotX: -0.6,
    scale: 70,
  },
};

// ── Component ────────────────────────────────────────────────────────

export default function TangentSpaceExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');

  const [surfaceType, setSurfaceType] = useState<SurfaceType>('sphere');
  const [paramU, setParamU] = useState(SURFACES.sphere.defaultU);
  const [paramV, setParamV] = useState(SURFACES.sphere.defaultV);
  const [a1, setA1] = useState(1.0);
  const [a2, setA2] = useState(0.5);

  const cfg = SURFACES[surfaceType];
  const isStacked = containerWidth < SM_BREAKPOINT;
  const leftWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.55);
  const rightWidth = isStacked ? containerWidth : containerWidth - leftWidth - 12;

  // Compute tangent data
  const tangentData = useMemo(() => {
    const p = cfg.surface(paramU, paramV);
    const { du, dv } = surfaceTangents(cfg.surface, paramU, paramV);

    // Normalize for display
    const duNorm = vec3Norm(du);
    const dvNorm = vec3Norm(dv);
    const duNormalized = duNorm > 1e-8 ? vec3Scale(du, 1 / duNorm) : du;
    const dvNormalized = dvNorm > 1e-8 ? vec3Scale(dv, 1 / dvNorm) : dv;

    // Normal vector for tangent plane orientation
    const normal = vec3Normalize(vec3Cross(du, dv));

    // Combined tangent vector
    const combined = vec3Add(vec3Scale(duNormalized, a1), vec3Scale(dvNormalized, a2));

    return { p, du: duNormalized, dv: dvNormalized, normal, combined };
  }, [paramU, paramV, a1, a2, cfg]);

  // ── Left panel: surface with tangent plane ─────────────────────────

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (leftWidth < 80) return;

      const w = leftWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const cx = w / 2;
      const cy = h / 2;
      const sc = cfg.scale;

      const { rotY, rotX } = cfg;
      const proj = (p: Vec3): Vec2 => {
        const v = orthoProject(p, rotY, rotX);
        return { x: cx + v.x * sc, y: cy - v.y * sc };
      };

      // Draw wireframe (back lines first, then front)
      const drawWireframe = (
        uRange: [number, number],
        vRange: [number, number],
        uSteps: number,
        vSteps: number,
      ) => {
        const lineGen = d3.line<Vec2>().x((d) => d.x).y((d) => d.y);

        // Constant-u lines
        for (let i = 0; i <= uSteps; i++) {
          const u = uRange[0] + (i / uSteps) * (uRange[1] - uRange[0]);
          const pts: Vec2[] = [];
          for (let j = 0; j <= vSteps * 3; j++) {
            const v = vRange[0] + (j / (vSteps * 3)) * (vRange[1] - vRange[0]);
            pts.push(proj(cfg.surface(u, v)));
          }
          svg.append('path')
            .attr('d', lineGen(pts))
            .style('fill', 'none')
            .style('stroke', 'var(--color-muted-border)')
            .style('stroke-width', 0.6)
            .style('opacity', 0.5);
        }
        // Constant-v lines
        for (let j = 0; j <= vSteps; j++) {
          const v = vRange[0] + (j / vSteps) * (vRange[1] - vRange[0]);
          const pts: Vec2[] = [];
          for (let i = 0; i <= uSteps * 3; i++) {
            const u = uRange[0] + (i / (uSteps * 3)) * (uRange[1] - uRange[0]);
            pts.push(proj(cfg.surface(u, v)));
          }
          svg.append('path')
            .attr('d', lineGen(pts))
            .style('fill', 'none')
            .style('stroke', 'var(--color-muted-border)')
            .style('stroke-width', 0.6)
            .style('opacity', 0.5);
        }
      };

      drawWireframe(cfg.uRange, cfg.vRange, cfg.uSteps, cfg.vSteps);

      const { p, du, dv, combined } = tangentData;
      const pp = proj(p);

      // Tangent plane (translucent quad)
      const tangentScale = 0.6;
      const corners = [
        vec3Add(vec3Add(p, vec3Scale(du, tangentScale)), vec3Scale(dv, tangentScale)),
        vec3Add(vec3Add(p, vec3Scale(du, tangentScale)), vec3Scale(dv, -tangentScale)),
        vec3Add(vec3Add(p, vec3Scale(du, -tangentScale)), vec3Scale(dv, -tangentScale)),
        vec3Add(vec3Add(p, vec3Scale(du, -tangentScale)), vec3Scale(dv, tangentScale)),
      ];
      const cornerProj = corners.map(proj);

      svg.append('polygon')
        .attr('points', cornerProj.map((c) => `${c.x},${c.y}`).join(' '))
        .style('fill', 'var(--color-text)')
        .style('opacity', 0.08)
        .style('stroke', 'var(--color-text-muted)')
        .style('stroke-width', 0.5);

      // Define arrow markers once
      const defs = svg.append('defs');
      const colors = { teal: TEAL, purple: PURPLE, amber: AMBER };
      for (const [key, color] of Object.entries(colors)) {
        defs.append('marker')
          .attr('id', `arr-${key}-${instanceId}`)
          .attr('viewBox', '0 0 10 6')
          .attr('refX', 8).attr('refY', 3)
          .attr('markerWidth', 7).attr('markerHeight', 5)
          .attr('orient', 'auto')
          .append('path')
          .attr('d', 'M0,0 L10,3 L0,6 Z')
          .style('fill', color);
      }

      const drawArrow = (from: Vec2, to: Vec2, markerKey: string, color: string, width: number) => {
        svg.append('line')
          .attr('x1', from.x).attr('y1', from.y)
          .attr('x2', to.x).attr('y2', to.y)
          .style('stroke', color).style('stroke-width', width)
          .attr('marker-end', `url(#arr-${markerKey}-${instanceId})`);
      };

      // Basis vectors (project arrow endpoints properly)
      const duEndP = proj(vec3Add(p, vec3Scale(du, 0.7)));
      const dvEndP = proj(vec3Add(p, vec3Scale(dv, 0.7)));
      const combEndP = proj(vec3Add(p, vec3Scale(combined, 0.7)));

      drawArrow(pp, duEndP, 'teal', TEAL, 2);
      drawArrow(pp, dvEndP, 'purple', PURPLE, 2);
      drawArrow(pp, combEndP, 'amber', AMBER, 2.5);

      // Labels
      svg.append('text')
        .attr('x', duEndP.x + 6).attr('y', duEndP.y - 4)
        .style('fill', TEAL).style('font-size', '11px').style('font-family', 'var(--font-sans)')
        .text('∂/∂u');
      svg.append('text')
        .attr('x', dvEndP.x + 6).attr('y', dvEndP.y - 4)
        .style('fill', PURPLE).style('font-size', '11px').style('font-family', 'var(--font-sans)')
        .text('∂/∂v');
      svg.append('text')
        .attr('x', combEndP.x + 6).attr('y', combEndP.y + 12)
        .style('fill', AMBER).style('font-size', '11px').style('font-family', 'var(--font-sans)')
        .text('v');

      // Point
      svg.append('circle')
        .attr('cx', pp.x).attr('cy', pp.y).attr('r', 5)
        .style('fill', 'var(--color-text)')
        .style('stroke', 'var(--color-surface)')
        .style('stroke-width', 2);

      // Title
      svg.append('text')
        .attr('x', cx).attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-family', 'var(--font-sans)')
        .style('font-weight', '600')
        .text(`${cfg.label} with tangent plane at p`);
    },
    [leftWidth, tangentData, cfg, instanceId],
  );

  // ── Right panel: abstract tangent space T_pM ≅ R² ──────────────────

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (rightWidth < 80) return;

      const w = rightWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const plotW = w - 40;
      const plotH = h - 50;
      const ox = 20 + plotW / 2;
      const oy = 30 + plotH / 2;
      const sc = Math.min(plotW, plotH) / 5;

      // Grid
      for (let i = -2; i <= 2; i++) {
        svg.append('line')
          .attr('x1', ox + i * sc).attr('x2', ox + i * sc)
          .attr('y1', oy - 2.5 * sc).attr('y2', oy + 2.5 * sc)
          .style('stroke', 'var(--color-muted-border)').style('stroke-width', 0.3);
        svg.append('line')
          .attr('x1', ox - 2.5 * sc).attr('x2', ox + 2.5 * sc)
          .attr('y1', oy + i * sc).attr('y2', oy + i * sc)
          .style('stroke', 'var(--color-muted-border)').style('stroke-width', 0.3);
      }

      // Axes
      svg.append('line')
        .attr('x1', ox - 2.5 * sc).attr('x2', ox + 2.5 * sc)
        .attr('y1', oy).attr('y2', oy)
        .style('stroke', 'var(--color-text-muted)').style('stroke-width', 1);
      svg.append('line')
        .attr('x1', ox).attr('x2', ox)
        .attr('y1', oy - 2.5 * sc).attr('y2', oy + 2.5 * sc)
        .style('stroke', 'var(--color-text-muted)').style('stroke-width', 1);

      // Define arrow markers once
      const defs = svg.append('defs');
      const colors = { teal: TEAL, purple: PURPLE, amber: AMBER };
      for (const [key, color] of Object.entries(colors)) {
        defs.append('marker')
          .attr('id', `arr2-${key}-${instanceId}`)
          .attr('viewBox', '0 0 10 6')
          .attr('refX', 8).attr('refY', 3)
          .attr('markerWidth', 7).attr('markerHeight', 5)
          .attr('orient', 'auto')
          .append('path')
          .attr('d', 'M0,0 L10,3 L0,6 Z')
          .style('fill', color);
      }

      const drawArrow = (fx: number, fy: number, markerKey: string, color: string, width: number, label: string) => {
        svg.append('line')
          .attr('x1', ox).attr('y1', oy)
          .attr('x2', ox + fx * sc).attr('y2', oy - fy * sc)
          .style('stroke', color).style('stroke-width', width)
          .attr('marker-end', `url(#arr2-${markerKey}-${instanceId})`);

        svg.append('text')
          .attr('x', ox + fx * sc + 8).attr('y', oy - fy * sc - 4)
          .style('fill', color).style('font-size', '11px').style('font-family', 'var(--font-sans)')
          .text(label);
      };

      // Basis vectors (unit vectors in abstract space)
      drawArrow(1, 0, 'teal', TEAL, 2, 'e₁ = ∂/∂u');
      drawArrow(0, 1, 'purple', PURPLE, 2, 'e₂ = ∂/∂v');

      // Combined vector
      drawArrow(a1, a2, 'amber', AMBER, 2.5, `v = ${a1.toFixed(1)}e₁ + ${a2.toFixed(1)}e₂`);

      // Origin dot
      svg.append('circle')
        .attr('cx', ox).attr('cy', oy).attr('r', 3)
        .style('fill', 'var(--color-text)');

      // Title
      svg.append('text')
        .attr('x', w / 2).attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-family', 'var(--font-sans)')
        .style('font-weight', '600')
        .text('T_pM ≅ ℝ²');

      // Axis labels
      svg.append('text')
        .attr('x', ox + 2.5 * sc).attr('y', oy - 6)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-muted)').style('font-size', '10px').style('font-family', 'var(--font-sans)')
        .text('a¹');
      svg.append('text')
        .attr('x', ox + 8).attr('y', oy - 2.5 * sc + 4)
        .style('fill', 'var(--color-text-muted)').style('font-size', '10px').style('font-family', 'var(--font-sans)')
        .text('a²');
    },
    [rightWidth, a1, a2, instanceId],
  );

  // ── Render ─────────────────────────────────────────────────────────

  return (
    <div ref={containerRef} className="w-full my-8">
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-3`}>
        <svg ref={leftRef} />
        <svg ref={rightRef} />
      </div>

      {/* Controls */}
      <div className="mt-4 flex flex-wrap gap-4 items-center text-sm" style={{ fontFamily: 'var(--font-sans)' }}>
        <label className="flex items-center gap-2" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Surface:</span>
          <select
            value={surfaceType}
            onChange={(e) => {
              const s = e.target.value as SurfaceType;
              setSurfaceType(s);
              setParamU(SURFACES[s].defaultU);
              setParamV(SURFACES[s].defaultV);
            }}
            className="px-2 py-1 rounded"
            style={{
              background: 'var(--color-surface)',
              border: '1px solid var(--color-muted-border)',
              color: 'var(--color-text)',
            }}
          >
            {Object.entries(SURFACES).map(([key, val]) => (
              <option key={key} value={key}>{val.label}</option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2" style={{ color: TEAL }}>
          <span className="font-medium">a¹:</span>
          <input
            type="range" min="-2" max="2" step="0.1" value={a1}
            onChange={(e) => setA1(Number(e.target.value))}
            className="w-20 accent-[#0F6E56]"
          />
          <span style={{ fontFamily: 'var(--font-mono)', minWidth: '2.5em' }}>{a1.toFixed(1)}</span>
        </label>

        <label className="flex items-center gap-2" style={{ color: PURPLE }}>
          <span className="font-medium">a²:</span>
          <input
            type="range" min="-2" max="2" step="0.1" value={a2}
            onChange={(e) => setA2(Number(e.target.value))}
            className="w-20 accent-[#534AB7]"
          />
          <span style={{ fontFamily: 'var(--font-mono)', minWidth: '2.5em' }}>{a2.toFixed(1)}</span>
        </label>
      </div>

      {/* Parameter sliders for point position */}
      <div className="mt-2 flex flex-wrap gap-4 items-center text-sm" style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text-muted)' }}>
        <label className="flex items-center gap-2">
          <span>u:</span>
          <input
            type="range"
            min={cfg.uRange[0]} max={cfg.uRange[1]}
            step={0.05} value={paramU}
            onChange={(e) => setParamU(Number(e.target.value))}
            className="w-24"
          />
          <span style={{ fontFamily: 'var(--font-mono)', minWidth: '2.5em' }}>{paramU.toFixed(2)}</span>
        </label>

        <label className="flex items-center gap-2">
          <span>v:</span>
          <input
            type="range"
            min={cfg.vRange[0]} max={cfg.vRange[1]}
            step={0.05} value={paramV}
            onChange={(e) => setParamV(Number(e.target.value))}
            className="w-24"
          />
          <span style={{ fontFamily: 'var(--font-mono)', minWidth: '2.5em' }}>{paramV.toFixed(2)}</span>
        </label>
      </div>
    </div>
  );
}

import { useState, useMemo, useId } from 'react';
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
  parallelTransportS2,
  sphereMetric,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 380;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

type CurveType = 'greatCircle' | 'latitude60' | 'latitude45' | 'triangle';

interface CurveDef {
  label: string;
  curve: (t: number) => [number, number];
  curveDot: (t: number) => [number, number];
  isClosed: boolean;
}

const CURVES: Record<CurveType, CurveDef> = {
  greatCircle: {
    label: 'Great circle (equator)',
    curve: (t) => [Math.PI / 2, t * 2 * Math.PI],
    curveDot: (t) => [0, 2 * Math.PI],
    isClosed: true,
  },
  latitude60: {
    label: 'Latitude circle θ = π/3',
    curve: (t) => [Math.PI / 3, t * 2 * Math.PI],
    curveDot: (t) => [0, 2 * Math.PI],
    isClosed: true,
  },
  latitude45: {
    label: 'Latitude circle θ = π/4',
    curve: (t) => [Math.PI / 4, t * 2 * Math.PI],
    curveDot: (t) => [0, 2 * Math.PI],
    isClosed: true,
  },
  triangle: {
    label: 'Spherical triangle',
    // Triangle: NP → (π/2, 0) → (π/2, π/2) → NP
    curve: (t) => {
      if (t < 1 / 3) {
        // NP to equator at φ=0: θ goes from ε to π/2
        const s = t * 3;
        return [0.05 + s * (Math.PI / 2 - 0.05), 0];
      } else if (t < 2 / 3) {
        // Along equator from φ=0 to φ=π/2
        const s = (t - 1 / 3) * 3;
        return [Math.PI / 2, s * Math.PI / 2];
      } else {
        // Equator at φ=π/2 back to NP
        const s = (t - 2 / 3) * 3;
        return [Math.PI / 2 - s * (Math.PI / 2 - 0.05), Math.PI / 2];
      }
    },
    curveDot: (t) => {
      if (t < 1 / 3) {
        return [(Math.PI / 2 - 0.05) * 3, 0];
      } else if (t < 2 / 3) {
        return [0, (Math.PI / 2) * 3];
      } else {
        return [-(Math.PI / 2 - 0.05) * 3, 0];
      }
    },
    isClosed: true,
  },
};

// ── Component ────────────────────────────────────────────────────────

export default function ParallelTransportExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');

  const [curveType, setCurveType] = useState<CurveType>('latitude60');
  const [initialAngle, setInitialAngle] = useState(0);
  const [showHolonomy, setShowHolonomy] = useState(false);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const svgWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.55);
  const plotWidth = isStacked ? containerWidth : containerWidth - svgWidth;

  // ── Parallel transport computation ───────────────────────────────
  const transportData = useMemo(() => {
    const curveDef = CURVES[curveType];
    const startTheta = curveDef.curve(0)[0];
    const sinTh = Math.sin(startTheta);
    // Initial vector in tangent plane at starting point, angle controlled by slider
    const V0: [number, number] = [
      Math.cos(initialAngle),
      sinTh > 0.01 ? Math.sin(initialAngle) / sinTh : 0, // V^φ scaled by 1/sinθ for unit norm
    ];
    return parallelTransportS2(curveDef.curve, curveDef.curveDot, V0, 400);
  }, [curveType, initialAngle]);

  // ── Sphere SVG ───────────────────────────────────────────────────
  const sphereRef = useD3<SVGSVGElement>(
    (svg) => {
      if (svgWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = svgWidth;
      const h = HEIGHT;
      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w, h) * 0.38;
      const rotY = -0.3;
      const rotX = -0.4;

      // Wireframe
      const { lines, backLines } = sphereWireframe(8, 12, rotY, rotX);
      const lineGen = d3.line<Vec2>().x((d) => cx + d.x * scale).y((d) => cy - d.y * scale);

      svg.selectAll('.back-line')
        .data(backLines)
        .join('path')
        .attr('d', (d) => lineGen(d))
        .attr('fill', 'none')
        .style('stroke', 'var(--color-text-muted, #666)')
        .style('stroke-width', '0.5')
        .style('stroke-dasharray', '2,2')
        .style('opacity', '0.3');

      svg.selectAll('.front-line')
        .data(lines)
        .join('path')
        .attr('d', (d) => lineGen(d))
        .attr('fill', 'none')
        .style('stroke', 'var(--color-text-muted, #888)')
        .style('stroke-width', '0.7')
        .style('opacity', '0.5');

      // Draw the curve
      const curvePts: Vec2[] = [];
      const curveDef = CURVES[curveType];
      for (let i = 0; i <= 200; i++) {
        const t = i / 200;
        const [th, ph] = curveDef.curve(t);
        const p3 = spherePoint(th, ph);
        if (isVisible(p3, rotY, rotX)) {
          curvePts.push(orthoProject(p3, rotY, rotX));
        }
      }
      if (curvePts.length > 1) {
        svg.append('path')
          .datum(curvePts)
          .attr('d', lineGen)
          .attr('fill', 'none')
          .style('stroke', 'var(--color-text-primary, #333)')
          .style('stroke-width', '2');
      }

      // Draw transported vectors at intervals
      const nArrows = 12;
      const arrowScale = scale * 0.12;
      transportData.forEach((pt, idx) => {
        if (idx % Math.floor(transportData.length / nArrows) !== 0 && idx !== transportData.length - 1) return;

        const [th, ph] = pt.pos;
        const p3 = spherePoint(th, ph);
        if (!isVisible(p3, rotY, rotX)) return;
        const proj = orthoProject(p3, rotY, rotX);

        // Convert tangent vector (V^θ, V^φ) to 3D direction in ambient R^3
        const sinTh = Math.sin(th);
        const cosTh = Math.cos(th);
        const sinPh = Math.sin(ph);
        const cosPh = Math.cos(ph);

        // ∂/∂θ in R^3
        const eTh = { x: cosTh * cosPh, y: cosTh * sinPh, z: -sinTh };
        // ∂/∂φ in R^3
        const ePh = { x: -sinTh * sinPh, y: sinTh * cosPh, z: 0 };

        const vx = pt.V[0] * eTh.x + pt.V[1] * ePh.x;
        const vy = pt.V[0] * eTh.y + pt.V[1] * ePh.y;
        const vz = pt.V[0] * eTh.z + pt.V[1] * ePh.z;

        const tipProj = orthoProject(
          { x: p3.x + vx * 0.15, y: p3.y + vy * 0.15, z: p3.z + vz * 0.15 },
          rotY, rotX,
        );

        const frac = idx / transportData.length;
        let color: string;
        if (idx === 0) color = TEAL;
        else if (idx === transportData.length - 1) color = PURPLE;
        else color = AMBER;

        const opacity = idx === 0 || idx === transportData.length - 1 ? 1 : 0.4 + frac * 0.4;

        svg.append('line')
          .attr('x1', cx + proj.x * scale)
          .attr('y1', cy - proj.y * scale)
          .attr('x2', cx + tipProj.x * scale)
          .attr('y2', cy - tipProj.y * scale)
          .style('stroke', color)
          .style('stroke-width', idx === 0 || idx === transportData.length - 1 ? '2.5' : '1.5')
          .style('opacity', String(opacity));

        // Arrowhead
        if (idx === 0 || idx === transportData.length - 1) {
          svg.append('circle')
            .attr('cx', cx + tipProj.x * scale)
            .attr('cy', cy - tipProj.y * scale)
            .attr('r', 3)
            .attr('fill', color);
        }
      });

      // Holonomy annotation
      if (showHolonomy && CURVES[curveType].isClosed && transportData.length > 1) {
        const first = transportData[0];
        const last = transportData[transportData.length - 1];
        const angle1 = Math.atan2(first.V[1], first.V[0]);
        const angle2 = Math.atan2(last.V[1], last.V[0]);
        let holAngle = ((angle2 - angle1) * 180 / Math.PI + 360) % 360;
        if (holAngle > 180) holAngle -= 360;

        svg.append('text')
          .attr('x', 10).attr('y', h - 10)
          .text(`Holonomy: ${holAngle.toFixed(1)}°`)
          .style('fill', PURPLE)
          .style('font-size', '12px')
          .style('font-weight', '600');
      }
    },
    [curveType, transportData, showHolonomy, svgWidth],
  );

  // ── Component plots ──────────────────────────────────────────────
  const plotRef = useD3<SVGSVGElement>(
    (svg) => {
      if (plotWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = plotWidth - 16;
      const h = HEIGHT;
      const margin = { top: 20, right: 15, bottom: 30, left: 40 };
      const plotH = (h - margin.top - margin.bottom - 30) / 2;

      const tVals = transportData.map((d) => d.t);
      const xScale = d3.scaleLinear().domain([0, 1]).range([margin.left, w - margin.right]);

      // Top plot: V^θ and V^φ
      const vTheta = transportData.map((d) => d.V[0]);
      const vPhi = transportData.map((d) => d.V[1]);
      const allV = [...vTheta, ...vPhi];
      const vMin = d3.min(allV) ?? -1;
      const vMax = d3.max(allV) ?? 1;
      const vPad = (vMax - vMin) * 0.1 || 0.5;
      const yScale1 = d3.scaleLinear()
        .domain([vMin - vPad, vMax + vPad])
        .range([margin.top + plotH, margin.top]);

      // Axes
      svg.append('g')
        .attr('transform', `translate(0, ${margin.top + plotH})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat((d) => String(d)))
        .style('font-size', '9px')
        .style('color', 'var(--color-text-muted, #888)');

      svg.append('g')
        .attr('transform', `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(yScale1).ticks(4))
        .style('font-size', '9px')
        .style('color', 'var(--color-text-muted, #888)');

      // V^θ line
      const line1 = d3.line<number>()
        .x((_, i) => xScale(tVals[i]))
        .y((d) => yScale1(d));

      svg.append('path')
        .datum(vTheta)
        .attr('d', line1)
        .attr('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', '2');

      // V^φ line
      svg.append('path')
        .datum(vPhi)
        .attr('d', line1)
        .attr('fill', 'none')
        .style('stroke', PURPLE)
        .style('stroke-width', '2');

      // Legend
      svg.append('text').attr('x', margin.left + 5).attr('y', margin.top + 12)
        .text('Vᶿ').style('fill', TEAL).style('font-size', '11px').style('font-weight', '600');
      svg.append('text').attr('x', margin.left + 30).attr('y', margin.top + 12)
        .text('Vᵠ').style('fill', PURPLE).style('font-size', '11px').style('font-weight', '600');

      // Bottom plot: norm
      const normData = transportData.map((d) => d.normG);
      const normMin = d3.min(normData) ?? 0;
      const normMax = d3.max(normData) ?? 1;
      const normPad = (normMax - normMin) * 0.2 || 0.1;
      const yOffset = margin.top + plotH + 40;
      const yScale2 = d3.scaleLinear()
        .domain([Math.max(0, normMin - normPad), normMax + normPad])
        .range([yOffset + plotH, yOffset]);

      svg.append('g')
        .attr('transform', `translate(0, ${yOffset + plotH})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat((d) => String(d)))
        .style('font-size', '9px')
        .style('color', 'var(--color-text-muted, #888)');

      svg.append('g')
        .attr('transform', `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(yScale2).ticks(3))
        .style('font-size', '9px')
        .style('color', 'var(--color-text-muted, #888)');

      const line2 = d3.line<number>()
        .x((_, i) => xScale(tVals[i]))
        .y((d) => yScale2(d));

      svg.append('path')
        .datum(normData)
        .attr('d', line2)
        .attr('fill', 'none')
        .style('stroke', AMBER)
        .style('stroke-width', '2');

      svg.append('text').attr('x', margin.left + 5).attr('y', yOffset + 12)
        .text('‖V‖ᵍ (norm)').style('fill', AMBER).style('font-size', '11px').style('font-weight', '600');

      // X-axis label
      svg.append('text')
        .attr('x', w / 2).attr('y', h - 2)
        .attr('text-anchor', 'middle')
        .text('t (curve parameter)')
        .style('fill', 'var(--color-text-muted, #888)')
        .style('font-size', '10px');
    },
    [transportData, plotWidth],
  );

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border,#e5e7eb)] bg-[var(--color-bg-secondary,#f9fafb)] p-4">
      <div className="mb-3 text-sm font-semibold" style={{ color: 'var(--color-text-primary, #111)' }}>
        Parallel Transport Explorer
      </div>

      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-4`}>
        <div style={{ width: isStacked ? '100%' : `${svgWidth}px` }}>
          <svg ref={sphereRef} width={svgWidth} height={HEIGHT} />
        </div>
        <div style={{ width: isStacked ? '100%' : `${plotWidth}px` }}>
          <svg ref={plotRef} width={plotWidth} height={HEIGHT} />
        </div>
      </div>

      {/* Controls */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Curve:</span>
          <select
            value={curveType}
            onChange={(e) => setCurveType(e.target.value as CurveType)}
            className="rounded border px-1.5 py-0.5 text-xs"
            style={{ borderColor: 'var(--color-border, #d1d5db)', background: 'var(--color-bg-primary, #fff)', color: 'var(--color-text-primary, #111)' }}
          >
            {Object.entries(CURVES).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Initial angle:</span>
          <input
            type="range"
            min={-Math.PI}
            max={Math.PI}
            step={0.1}
            value={initialAngle}
            onChange={(e) => setInitialAngle(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="font-mono w-10">{(initialAngle * 180 / Math.PI).toFixed(0)}°</span>
        </label>

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showHolonomy} onChange={(e) => setShowHolonomy(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Show holonomy</span>
        </label>
      </div>
    </div>
  );
}

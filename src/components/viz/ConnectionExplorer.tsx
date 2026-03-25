import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  type Vec2,
  type ChristoffelSymbols,
  spherePoint,
  sphereChristoffel,
  orthoProject,
  isVisible,
  sphereWireframe,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 380;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

type MetricType = 'sphere' | 'poincare' | 'flat';

// ── Component ────────────────────────────────────────────────────────

export default function ConnectionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [metricType, setMetricType] = useState<MetricType>('sphere');
  const [theta, setTheta] = useState(1.0);
  const [phi, setPhi] = useState(0.8);
  const [showTorsionFree, setShowTorsionFree] = useState(false);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const svgWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.55);
  const tableWidth = isStacked ? containerWidth : containerWidth - svgWidth;

  // ── Christoffel symbols ──────────────────────────────────────────
  const christoffel = useMemo<ChristoffelSymbols>(() => {
    if (metricType === 'sphere') return sphereChristoffel(theta);
    // Flat and Poincaré: all zeros for display (Poincaré conformal → nonzero, but we simplify)
    if (metricType === 'flat') {
      return { gamma: [[[0, 0], [0, 0]], [[0, 0], [0, 0]]] };
    }
    // Poincaré disk Christoffel symbols at (x, y) = (0.3, 0.2) approx
    // For conformal metric g = λ²δ, Γ^k_{ij} = (δ^k_i ∂_j log λ + δ^k_j ∂_i log λ − δ_{ij} g^{kl} ∂_l log λ)
    const x = 0.3, y = 0.2;
    const r2 = x * x + y * y;
    const denom = 1 - r2;
    const dlx = 2 * x / denom; // ∂_x log λ
    const dly = 2 * y / denom; // ∂_y log λ
    return {
      gamma: [
        [[dlx, dly], [dly, -dlx]],   // Γ^x
        [[-dly, dlx], [dlx, dly]],    // Γ^y
      ],
    };
  }, [metricType, theta]);

  const fmt = (v: number) => {
    if (Math.abs(v) < 1e-6) return '0';
    if (Math.abs(v) < 0.001) return v.toExponential(1);
    return v.toFixed(3);
  };

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

      if (metricType === 'flat') {
        // Flat R^2 grid
        const gridN = 8;
        const gridScale = Math.min(w, h) * 0.35;
        for (let i = -gridN; i <= gridN; i++) {
          const frac = i / gridN;
          svg.append('line')
            .attr('x1', cx + frac * gridScale).attr('y1', cy - gridScale)
            .attr('x2', cx + frac * gridScale).attr('y2', cy + gridScale)
            .style('stroke', 'var(--color-text-muted, #888)')
            .style('stroke-width', '0.5')
            .style('opacity', '0.3');
          svg.append('line')
            .attr('x1', cx - gridScale).attr('y1', cy + frac * gridScale)
            .attr('x2', cx + gridScale).attr('y2', cy + frac * gridScale)
            .style('stroke', 'var(--color-text-muted, #888)')
            .style('stroke-width', '0.5')
            .style('opacity', '0.3');
        }

        // Point and basis vectors
        svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', 5)
          .attr('fill', TEAL).style('stroke', '#fff').style('stroke-width', '1.5');

        // e_x and e_y
        const arrowLen = 40;
        svg.append('line')
          .attr('x1', cx).attr('y1', cy)
          .attr('x2', cx + arrowLen).attr('y2', cy)
          .style('stroke', TEAL).style('stroke-width', '2');
        svg.append('line')
          .attr('x1', cx).attr('y1', cy)
          .attr('x2', cx).attr('y2', cy - arrowLen)
          .style('stroke', PURPLE).style('stroke-width', '2');

        svg.append('text').attr('x', cx + arrowLen + 5).attr('y', cy + 4)
          .text('∂/∂x').style('fill', TEAL).style('font-size', '11px');
        svg.append('text').attr('x', cx + 5).attr('y', cy - arrowLen - 5)
          .text('∂/∂y').style('fill', PURPLE).style('font-size', '11px');

        svg.append('text').attr('x', 10).attr('y', h - 10)
          .text('All Γᵏᵢⱼ = 0 (flat space)')
          .style('fill', 'var(--color-text-muted, #888)').style('font-size', '11px');
        return;
      }

      if (metricType === 'poincare') {
        // Poincaré disk
        const diskR = Math.min(w, h) * 0.4;
        svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', diskR)
          .attr('fill', 'none').style('stroke', 'var(--color-text-muted, #888)').style('stroke-width', '1.5');

        // Radial/angular grid
        for (let r = 0.2; r < 1; r += 0.2) {
          svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', diskR * r)
            .attr('fill', 'none').style('stroke', 'var(--color-text-muted, #888)')
            .style('stroke-width', '0.5').style('opacity', '0.3');
        }

        // Color overlay for Christoffel magnitude
        const gridN = 20;
        const maxGamma = 3;
        const colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxGamma]);

        for (let ix = 0; ix < gridN; ix++) {
          for (let iy = 0; iy < gridN; iy++) {
            const gx = -0.95 + (ix / gridN) * 1.9;
            const gy = -0.95 + (iy / gridN) * 1.9;
            if (gx * gx + gy * gy >= 0.9) continue;
            const r2 = gx * gx + gy * gy;
            const denom = 1 - r2;
            const mag = Math.sqrt((2 * gx / denom) ** 2 + (2 * gy / denom) ** 2);
            svg.append('rect')
              .attr('x', cx + gx * diskR - diskR / gridN / 2)
              .attr('y', cy - gy * diskR - diskR / gridN / 2)
              .attr('width', diskR * 1.9 / gridN)
              .attr('height', diskR * 1.9 / gridN)
              .attr('fill', colorScale(Math.min(mag, maxGamma)))
              .style('opacity', '0.3');
          }
        }

        svg.append('circle')
          .attr('cx', cx + 0.3 * diskR).attr('cy', cy - 0.2 * diskR).attr('r', 5)
          .attr('fill', TEAL).style('stroke', '#fff').style('stroke-width', '1.5');

        return;
      }

      // Sphere
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

      // Color overlay: Christoffel symbol magnitude
      const nGrid = 16;
      const maxMag = 1.2;
      const colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxMag]);

      for (let i = 0; i < nGrid; i++) {
        for (let j = 0; j < nGrid * 2; j++) {
          const th = 0.1 + (i / nGrid) * (Math.PI - 0.2);
          const ph = (j / (nGrid * 2)) * 2 * Math.PI;
          const p3 = spherePoint(th, ph);
          if (!isVisible(p3, rotY, rotX)) continue;
          const proj = orthoProject(p3, rotY, rotX);
          const chris = sphereChristoffel(th);
          // Include both symmetric components: Γ^θ_{φφ} and Γ^φ_{θφ} = Γ^φ_{φθ}
          const mag = Math.sqrt(
            chris.gamma[0][1][1] ** 2 + 2 * chris.gamma[1][0][1] ** 2
          );
          svg.append('circle')
            .attr('cx', cx + proj.x * scale)
            .attr('cy', cy - proj.y * scale)
            .attr('r', 4)
            .attr('fill', colorScale(Math.min(mag, maxMag)))
            .style('opacity', '0.4');
        }
      }

      // Current point with basis vectors
      const p3 = spherePoint(theta, phi);
      if (isVisible(p3, rotY, rotX)) {
        const proj = orthoProject(p3, rotY, rotX);
        const sinTh = Math.sin(theta);
        const cosTh = Math.cos(theta);
        const sinPh = Math.sin(phi);
        const cosPh = Math.cos(phi);

        const eTh = { x: cosTh * cosPh, y: cosTh * sinPh, z: -sinTh };
        const ePh = { x: -sinTh * sinPh, y: sinTh * cosPh, z: 0 };

        const arrowLen = 0.2;
        const tipTh = orthoProject(
          { x: p3.x + eTh.x * arrowLen, y: p3.y + eTh.y * arrowLen, z: p3.z + eTh.z * arrowLen },
          rotY, rotX,
        );
        const tipPh = orthoProject(
          { x: p3.x + ePh.x * arrowLen, y: p3.y + ePh.y * arrowLen, z: p3.z + ePh.z * arrowLen },
          rotY, rotX,
        );

        svg.append('line')
          .attr('x1', cx + proj.x * scale).attr('y1', cy - proj.y * scale)
          .attr('x2', cx + tipTh.x * scale).attr('y2', cy - tipTh.y * scale)
          .style('stroke', TEAL).style('stroke-width', '2.5');

        svg.append('line')
          .attr('x1', cx + proj.x * scale).attr('y1', cy - proj.y * scale)
          .attr('x2', cx + tipPh.x * scale).attr('y2', cy - tipPh.y * scale)
          .style('stroke', PURPLE).style('stroke-width', '2.5');

        // Torsion-free parallelogram
        if (showTorsionFree) {
          const eps = 0.08;
          const pA = proj;
          const pB = orthoProject(spherePoint(theta + eps, phi), rotY, rotX);
          const pC = orthoProject(spherePoint(theta, phi + eps), rotY, rotX);
          const pD = orthoProject(spherePoint(theta + eps, phi + eps), rotY, rotX);

          const pts = [pA, pB, pD, pC, pA];
          const parPath = d3.line<Vec2>().x((d) => cx + d.x * scale).y((d) => cy - d.y * scale);
          svg.append('path')
            .datum(pts)
            .attr('d', parPath)
            .attr('fill', AMBER)
            .style('fill-opacity', '0.15')
            .style('stroke', AMBER)
            .style('stroke-width', '1.5')
            .style('stroke-dasharray', '4,2');
        }

        svg.append('circle')
          .attr('cx', cx + proj.x * scale)
          .attr('cy', cy - proj.y * scale)
          .attr('r', 5)
          .attr('fill', TEAL)
          .style('stroke', '#fff')
          .style('stroke-width', '1.5');
      }
    },
    [metricType, theta, phi, showTorsionFree, svgWidth],
  );

  // ── Coordinate labels ────────────────────────────────────────────
  const coords = metricType === 'poincare' ? ['x', 'y'] : metricType === 'flat' ? ['x', 'y'] : ['θ', 'φ'];

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border,#e5e7eb)] bg-[var(--color-bg-secondary,#f9fafb)] p-4">
      <div className="mb-3 text-sm font-semibold" style={{ color: 'var(--color-text-primary, #111)' }}>
        Connection Explorer
      </div>

      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-4`}>
        <div style={{ width: isStacked ? '100%' : `${svgWidth}px` }}>
          <svg ref={sphereRef} width={svgWidth} height={HEIGHT} />
        </div>

        {/* Christoffel symbol table */}
        <div style={{ width: isStacked ? '100%' : `${tableWidth}px` }} className="flex flex-col justify-center gap-3">
          <div className="text-xs font-medium" style={{ color: 'var(--color-text-muted, #666)' }}>
            Christoffel symbols Γ<sup>k</sup><sub>ij</sub>
            {metricType === 'sphere' && ` at θ = ${fmt(theta)}`}
          </div>

          {[0, 1].map((k) => (
            <div key={k} className="rounded border p-2" style={{ borderColor: 'var(--color-border, #e5e7eb)', background: 'var(--color-bg-primary, #fff)' }}>
              <div className="mb-1 text-xs font-mono font-medium" style={{ color: k === 0 ? TEAL : PURPLE }}>
                Γ<sup>{coords[k]}</sup><sub>ij</sub>
              </div>
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr>
                    <th className="w-8"></th>
                    <th className="text-center" style={{ color: 'var(--color-text-muted, #888)' }}>{coords[0]}</th>
                    <th className="text-center" style={{ color: 'var(--color-text-muted, #888)' }}>{coords[1]}</th>
                  </tr>
                </thead>
                <tbody>
                  {[0, 1].map((i) => (
                    <tr key={i}>
                      <td className="text-right pr-2" style={{ color: 'var(--color-text-muted, #888)' }}>{coords[i]}</td>
                      {[0, 1].map((j) => {
                        const val = christoffel.gamma[k][i][j];
                        const isNonzero = Math.abs(val) > 1e-6;
                        return (
                          <td key={j} className="text-center px-2 py-0.5" style={{
                            color: isNonzero ? (k === 0 ? TEAL : PURPLE) : 'var(--color-text-muted, #ccc)',
                            fontWeight: isNonzero ? '600' : '400',
                          }}>
                            {fmt(val)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}

          <div className="text-xs" style={{ color: 'var(--color-text-muted, #666)' }}>
            {metricType === 'sphere' && 'Only Γᶿφφ and Γᵠθφ are nonzero on S². Heatmap shows |Γ| magnitude.'}
            {metricType === 'flat' && 'All Christoffel symbols vanish in flat space — the connection is trivial.'}
            {metricType === 'poincare' && 'Conformal metric Christoffel symbols grow near the boundary.'}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Metric:</span>
          <select
            value={metricType}
            onChange={(e) => setMetricType(e.target.value as MetricType)}
            className="rounded border px-1.5 py-0.5 text-xs"
            style={{ borderColor: 'var(--color-border, #d1d5db)', background: 'var(--color-bg-primary, #fff)', color: 'var(--color-text-primary, #111)' }}
          >
            <option value="sphere">Sphere S²</option>
            <option value="poincare">Poincaré Disk</option>
            <option value="flat">Flat ℝ²</option>
          </select>
        </label>

        {metricType === 'sphere' && (
          <>
            <label className="flex items-center gap-1.5">
              <span style={{ color: 'var(--color-text-muted, #666)' }}>θ:</span>
              <input
                type="range" min={0.15} max={Math.PI - 0.15} step={0.05}
                value={theta} onChange={(e) => setTheta(parseFloat(e.target.value))}
                className="w-20"
              />
              <span className="font-mono w-10">{fmt(theta)}</span>
            </label>

            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={showTorsionFree} onChange={(e) => setShowTorsionFree(e.target.checked)} />
              <span style={{ color: 'var(--color-text-muted, #666)' }}>Show torsion-free parallelogram</span>
            </label>
          </>
        )}
      </div>
    </div>
  );
}

import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  type Vec2,
  type Vec3,
  type MetricTensor,
  spherePoint,
  sphereMetric,
  poincareMetric,
  ellipsoidMetric,
  ellipsoidPoint,
  poincareConformalFactor,
  metricEigendecomp,
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

type ManifoldType = 'sphere' | 'poincare' | 'ellipsoid';

// ── Component ────────────────────────────────────────────────────────

export default function MetricTensorExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');

  const [manifold, setManifold] = useState<ManifoldType>('sphere');
  // Sphere/ellipsoid: (theta, phi); Poincaré: (x, y)
  const [theta, setTheta] = useState(1.2);
  const [phi, setPhi] = useState(0.8);
  const [px, setPx] = useState(0.3);
  const [py, setPy] = useState(0.2);
  const [showGrid, setShowGrid] = useState(false);
  const [showEigenvectors, setShowEigenvectors] = useState(true);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const svgWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.58);
  const infoWidth = isStacked ? containerWidth : containerWidth - svgWidth;

  // ── Metric computation ───────────────────────────────────────────
  const metric = useMemo<MetricTensor>(() => {
    if (manifold === 'sphere') return sphereMetric(theta);
    if (manifold === 'ellipsoid') return ellipsoidMetric(theta);
    return poincareMetric(px, py);
  }, [manifold, theta, px, py]);

  const eigen = useMemo(() => metricEigendecomp(metric.g), [metric]);

  // ── Sphere / Ellipsoid SVG ───────────────────────────────────────
  const sphereRef = useD3<SVGSVGElement>(
    (svg) => {
      if (manifold === 'poincare' || svgWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = svgWidth;
      const h = HEIGHT;
      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w, h) * 0.38;
      const rotY = -0.3;
      const rotX = -0.4;

      const isEllipsoid = manifold === 'ellipsoid';
      const surfaceFn = isEllipsoid
        ? (th: number, ph: number) => ellipsoidPoint(th, ph, 1, 0.6)
        : spherePoint;

      // Wireframe
      const { lines, backLines } = sphereWireframe(8, 12, rotY, rotX);
      const lineGen = d3.line<Vec2>().x((d) => cx + d.x * scale).y((d) => cy - d.y * scale);

      // Scale wireframe for ellipsoid
      const projectFn = (th: number, ph: number) => {
        const p3 = surfaceFn(th, ph);
        return orthoProject(p3, rotY, rotX);
      };

      if (isEllipsoid) {
        // Generate ellipsoid wireframe manually
        const elines: Vec2[][] = [];
        const ebackLines: Vec2[][] = [];
        for (let i = 0; i <= 8; i++) {
          const th = 0.05 + (i / 8) * (Math.PI - 0.1);
          const pts: Vec2[] = [];
          const bpts: Vec2[] = [];
          for (let j = 0; j <= 32; j++) {
            const ph = (j / 32) * 2 * Math.PI;
            const p3 = surfaceFn(th, ph);
            const proj = orthoProject(p3, rotY, rotX);
            if (isVisible(p3, rotY, rotX)) {
              if (bpts.length > 1) ebackLines.push([...bpts]);
              bpts.length = 0;
              pts.push(proj);
            } else {
              if (pts.length > 1) elines.push([...pts]);
              pts.length = 0;
              bpts.push(proj);
            }
          }
          if (pts.length > 1) elines.push(pts);
          if (bpts.length > 1) ebackLines.push(bpts);
        }
        for (let j = 0; j <= 12; j++) {
          const ph = (j / 12) * 2 * Math.PI;
          const pts: Vec2[] = [];
          const bpts: Vec2[] = [];
          for (let i = 0; i <= 32; i++) {
            const th = 0.05 + (i / 32) * (Math.PI - 0.1);
            const p3 = surfaceFn(th, ph);
            const proj = orthoProject(p3, rotY, rotX);
            if (isVisible(p3, rotY, rotX)) {
              if (bpts.length > 1) ebackLines.push([...bpts]);
              bpts.length = 0;
              pts.push(proj);
            } else {
              if (pts.length > 1) elines.push([...pts]);
              pts.length = 0;
              bpts.push(proj);
            }
          }
          if (pts.length > 1) elines.push(pts);
          if (bpts.length > 1) ebackLines.push(bpts);
        }

        svg.selectAll('.back-line')
          .data(ebackLines)
          .join('path')
          .attr('class', 'back-line')
          .attr('d', (d) => lineGen(d))
          .attr('fill', 'none')
          .style('stroke', 'var(--color-text-muted, #666)')
          .style('stroke-width', '0.5')
          .style('stroke-dasharray', '2,2')
          .style('opacity', '0.3');

        svg.selectAll('.front-line')
          .data(elines)
          .join('path')
          .attr('class', 'front-line')
          .attr('d', (d) => lineGen(d))
          .attr('fill', 'none')
          .style('stroke', 'var(--color-text-muted, #888)')
          .style('stroke-width', '0.7')
          .style('opacity', '0.5');
      } else {
        svg.selectAll('.back-line')
          .data(backLines)
          .join('path')
          .attr('class', 'back-line')
          .attr('d', (d) => lineGen(d))
          .attr('fill', 'none')
          .style('stroke', 'var(--color-text-muted, #666)')
          .style('stroke-width', '0.5')
          .style('stroke-dasharray', '2,2')
          .style('opacity', '0.3');

        svg.selectAll('.front-line')
          .data(lines)
          .join('path')
          .attr('class', 'front-line')
          .attr('d', (d) => lineGen(d))
          .attr('fill', 'none')
          .style('stroke', 'var(--color-text-muted, #888)')
          .style('stroke-width', '0.7')
          .style('opacity', '0.5');
      }

      // Metric ellipses at grid points
      if (showGrid) {
        const gridThetas = [0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5];
        const gridPhis = [0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6];

        for (const gt of gridThetas) {
          for (const gp of gridPhis) {
            const p3 = surfaceFn(gt, gp);
            if (!isVisible(p3, rotY, rotX)) continue;
            const proj = orthoProject(p3, rotY, rotX);
            const m = isEllipsoid ? ellipsoidMetric(gt) : sphereMetric(gt);
            const { eigenvalues } = metricEigendecomp(m.g);
            const rx = scale * 0.06 / Math.sqrt(Math.max(eigenvalues[0], 0.01));
            const ry = scale * 0.06 / Math.sqrt(Math.max(eigenvalues[1], 0.01));

            svg.append('ellipse')
              .attr('cx', cx + proj.x * scale)
              .attr('cy', cy - proj.y * scale)
              .attr('rx', Math.min(rx, 20))
              .attr('ry', Math.min(ry, 20))
              .attr('fill', 'none')
              .style('stroke', TEAL)
              .style('stroke-width', '1')
              .style('opacity', '0.5');
          }
        }
      }

      // Current point
      const p3 = surfaceFn(theta, phi);
      const vis = isVisible(p3, rotY, rotX);
      const proj = orthoProject(p3, rotY, rotX);

      // Metric ellipse at current point
      const { eigenvalues: ev, eigenvectors: evec } = eigen;
      const ellipseScale = scale * 0.1;
      const rx = ellipseScale / Math.sqrt(Math.max(ev[0], 0.01));
      const ry = ellipseScale / Math.sqrt(Math.max(ev[1], 0.01));
      const angle = Math.atan2(evec[0].y, evec[0].x) * (180 / Math.PI);

      if (vis) {
        svg.append('ellipse')
          .attr('cx', cx + proj.x * scale)
          .attr('cy', cy - proj.y * scale)
          .attr('rx', Math.min(rx, 40))
          .attr('ry', Math.min(ry, 40))
          .attr('transform', `rotate(${-angle}, ${cx + proj.x * scale}, ${cy - proj.y * scale})`)
          .attr('fill', TEAL)
          .style('fill-opacity', '0.15')
          .style('stroke', TEAL)
          .style('stroke-width', '2');

        // Eigenvector arrows
        if (showEigenvectors) {
          for (let i = 0; i < 2; i++) {
            const len = ellipseScale / Math.sqrt(Math.max(ev[i], 0.01));
            const dx = evec[i].x * Math.min(len, 35);
            const dy = evec[i].y * Math.min(len, 35);
            const color = i === 0 ? PURPLE : AMBER;
            const pcx = cx + proj.x * scale;
            const pcy = cy - proj.y * scale;
            svg.append('line')
              .attr('x1', pcx - dx).attr('y1', pcy + dy)
              .attr('x2', pcx + dx).attr('y2', pcy - dy)
              .style('stroke', color)
              .style('stroke-width', '2')
              .style('stroke-dasharray', '4,2');
          }
        }

        // Point marker
        svg.append('circle')
          .attr('cx', cx + proj.x * scale)
          .attr('cy', cy - proj.y * scale)
          .attr('r', 5)
          .attr('fill', TEAL)
          .style('stroke', '#fff')
          .style('stroke-width', '1.5')
          .style('cursor', 'grab');
      }

      // Drag behavior
      const drag = d3.drag<SVGSVGElement, unknown>()
        .on('drag', (event) => {
          const sx = (event.x - cx) / scale;
          const sy = -(event.y - cy) / scale;
          // Map screen coords back to approx (theta, phi) for the sphere
          const r = Math.sqrt(sx * sx + sy * sy);
          if (r > 0.98) return;
          const newTheta = Math.acos(Math.max(-0.95, Math.min(0.95, -sy / Math.max(r, 0.01) * r)));
          const newPhi = Math.atan2(sx, 0.3) + 0.8; // approximate
          setTheta(Math.max(0.15, Math.min(Math.PI - 0.15, Math.acos(Math.max(-1, Math.min(1, sy))))));
          setPhi(((Math.atan2(sx, 0.3) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI));
        });

      svg.call(drag);
    },
    [manifold, theta, phi, showGrid, showEigenvectors, svgWidth, eigen],
  );

  // ── Poincaré disk SVG ────────────────────────────────────────────
  const poincareRef = useD3<SVGSVGElement>(
    (svg) => {
      if (manifold !== 'poincare' || svgWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = svgWidth;
      const h = HEIGHT;
      const cx = w / 2;
      const cy = h / 2;
      const diskR = Math.min(w, h) * 0.4;

      // Boundary circle
      svg.append('circle')
        .attr('cx', cx).attr('cy', cy)
        .attr('r', diskR)
        .attr('fill', 'none')
        .style('stroke', 'var(--color-text-muted, #888)')
        .style('stroke-width', '1.5');

      // Hyperbolic grid lines (concentric circles in the Poincaré model)
      for (let r = 0.2; r < 1; r += 0.2) {
        svg.append('circle')
          .attr('cx', cx).attr('cy', cy)
          .attr('r', diskR * r)
          .attr('fill', 'none')
          .style('stroke', 'var(--color-text-muted, #888)')
          .style('stroke-width', '0.5')
          .style('opacity', '0.3');
      }
      for (let a = 0; a < Math.PI; a += Math.PI / 6) {
        svg.append('line')
          .attr('x1', cx + diskR * Math.cos(a)).attr('y1', cy + diskR * Math.sin(a))
          .attr('x2', cx - diskR * Math.cos(a)).attr('y2', cy - diskR * Math.sin(a))
          .style('stroke', 'var(--color-text-muted, #888)')
          .style('stroke-width', '0.5')
          .style('opacity', '0.3');
      }

      // Metric ellipses at grid points
      if (showGrid) {
        for (let gx = -0.8; gx <= 0.8; gx += 0.3) {
          for (let gy = -0.8; gy <= 0.8; gy += 0.3) {
            if (gx * gx + gy * gy >= 0.85) continue;
            const lambda = poincareConformalFactor(gx, gy);
            const eR = diskR * 0.04 / lambda;
            svg.append('circle')
              .attr('cx', cx + gx * diskR).attr('cy', cy - gy * diskR)
              .attr('r', Math.max(eR, 1.5))
              .attr('fill', 'none')
              .style('stroke', TEAL)
              .style('stroke-width', '1')
              .style('opacity', '0.5');
          }
        }
      }

      // Current point
      const lambda = poincareConformalFactor(px, py);
      const ellipseR = diskR * 0.08 / lambda;

      svg.append('circle')
        .attr('cx', cx + px * diskR)
        .attr('cy', cy - py * diskR)
        .attr('r', Math.max(ellipseR, 3))
        .attr('fill', TEAL)
        .style('fill-opacity', '0.15')
        .style('stroke', TEAL)
        .style('stroke-width', '2');

      svg.append('circle')
        .attr('cx', cx + px * diskR)
        .attr('cy', cy - py * diskR)
        .attr('r', 5)
        .attr('fill', TEAL)
        .style('stroke', '#fff')
        .style('stroke-width', '1.5')
        .style('cursor', 'grab');

      // Drag
      const drag = d3.drag<SVGSVGElement, unknown>()
        .on('drag', (event) => {
          const nx = (event.x - cx) / diskR;
          const ny = -(event.y - cy) / diskR;
          const r2 = nx * nx + ny * ny;
          if (r2 < 0.9) {
            setPx(nx);
            setPy(ny);
          }
        });
      svg.call(drag);
    },
    [manifold, px, py, showGrid, svgWidth],
  );

  // ── Metric formatting ────────────────────────────────────────────
  const fmt = (v: number) => {
    if (Math.abs(v) < 1e-6) return '0';
    return v.toFixed(3);
  };

  const coordLabels = manifold === 'poincare' ? ['x', 'y'] : ['θ', 'φ'];

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border,#e5e7eb)] bg-[var(--color-bg-secondary,#f9fafb)] p-4">
      <div className="mb-3 text-sm font-semibold" style={{ color: 'var(--color-text-primary, #111)' }}>
        Metric Tensor Explorer
      </div>

      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-4`}>
        {/* SVG Panel */}
        <div style={{ width: isStacked ? '100%' : `${svgWidth}px` }}>
          {manifold === 'poincare' ? (
            <svg ref={poincareRef} width={svgWidth} height={HEIGHT} />
          ) : (
            <svg ref={sphereRef} width={svgWidth} height={HEIGHT} />
          )}
        </div>

        {/* Info Panel */}
        <div style={{ width: isStacked ? '100%' : `${infoWidth}px` }} className="flex flex-col justify-center gap-3 text-sm">
          <div>
            <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-text-muted, #666)' }}>
              Metric tensor g<sub>ij</sub>{manifold === 'poincare' ? ` at (${fmt(px)}, ${fmt(py)})` : ` at (θ=${fmt(theta)}, φ=${fmt(phi)})`}
            </div>
            <div className="rounded border p-2 font-mono text-xs" style={{ borderColor: 'var(--color-border, #e5e7eb)', background: 'var(--color-bg-primary, #fff)' }}>
              <div className="flex justify-center gap-1">
                <span style={{ color: 'var(--color-text-muted, #666)' }}>(</span>
                <div className="text-center">
                  <div style={{ color: TEAL }}>{fmt(metric.g[0][0])}</div>
                  <div style={{ color: 'var(--color-text-muted, #999)' }}>{fmt(metric.g[1][0])}</div>
                </div>
                <div className="text-center">
                  <div style={{ color: 'var(--color-text-muted, #999)' }}>{fmt(metric.g[0][1])}</div>
                  <div style={{ color: TEAL }}>{fmt(metric.g[1][1])}</div>
                </div>
                <span style={{ color: 'var(--color-text-muted, #666)' }}>)</span>
              </div>
            </div>
          </div>

          <div>
            <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-text-muted, #666)' }}>
              det(g) = {fmt(metric.det)}
            </div>
          </div>

          <div>
            <div className="mb-1 text-xs font-medium" style={{ color: PURPLE }}>
              Eigenvalues (principal stretches)
            </div>
            <div className="font-mono text-xs">
              λ₁ = {fmt(eigen.eigenvalues[0])}, λ₂ = {fmt(eigen.eigenvalues[1])}
            </div>
          </div>

          {showEigenvectors && (
            <div>
              <div className="mb-1 text-xs font-medium" style={{ color: AMBER }}>
                Eigenvectors (principal directions)
              </div>
              <div className="font-mono text-xs">
                <div style={{ color: PURPLE }}>
                  e₁ = ({fmt(eigen.eigenvectors[0].x)}, {fmt(eigen.eigenvectors[0].y)})
                </div>
                <div style={{ color: AMBER }}>
                  e₂ = ({fmt(eigen.eigenvectors[1].x)}, {fmt(eigen.eigenvectors[1].y)})
                </div>
              </div>
            </div>
          )}

          <div className="text-xs" style={{ color: 'var(--color-text-muted, #666)' }}>
            {manifold === 'sphere' && 'Ellipse flattens near the poles where sin²θ → 0.'}
            {manifold === 'poincare' && 'Ellipse shrinks toward the boundary as the conformal factor diverges.'}
            {manifold === 'ellipsoid' && 'Ellipse shape varies with both latitude and the semi-axis ratio.'}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Manifold:</span>
          <select
            value={manifold}
            onChange={(e) => {
              const m = e.target.value as ManifoldType;
              setManifold(m);
              if (m === 'poincare') { setPx(0.3); setPy(0.2); }
              else { setTheta(1.2); setPhi(0.8); }
            }}
            className="rounded border px-1.5 py-0.5 text-xs"
            style={{ borderColor: 'var(--color-border, #d1d5db)', background: 'var(--color-bg-primary, #fff)', color: 'var(--color-text-primary, #111)' }}
          >
            <option value="sphere">Sphere S²</option>
            <option value="poincare">Poincaré Disk 𝔻²</option>
            <option value="ellipsoid">Ellipsoid</option>
          </select>
        </label>

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Show metric ellipses</span>
        </label>

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showEigenvectors} onChange={(e) => setShowEigenvectors(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Show eigenvectors</span>
        </label>

        {manifold !== 'poincare' && (
          <>
            <label className="flex items-center gap-1.5">
              <span style={{ color: 'var(--color-text-muted, #666)' }}>θ:</span>
              <input
                type="range"
                min={0.15}
                max={Math.PI - 0.15}
                step={0.05}
                value={theta}
                onChange={(e) => setTheta(parseFloat(e.target.value))}
                className="w-20"
              />
              <span className="font-mono w-10">{fmt(theta)}</span>
            </label>
            <label className="flex items-center gap-1.5">
              <span style={{ color: 'var(--color-text-muted, #666)' }}>φ:</span>
              <input
                type="range"
                min={0}
                max={2 * Math.PI}
                step={0.1}
                value={phi}
                onChange={(e) => setPhi(parseFloat(e.target.value))}
                className="w-20"
              />
              <span className="font-mono w-10">{fmt(phi)}</span>
            </label>
          </>
        )}
      </div>
    </div>
  );
}

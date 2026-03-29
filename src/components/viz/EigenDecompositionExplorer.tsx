import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Definiteness classification ───

type Definiteness =
  | 'Positive Definite'
  | 'Positive Semidefinite'
  | 'Negative Definite'
  | 'Negative Semidefinite'
  | 'Indefinite';

function classifyDefiniteness(l1: number, l2: number): Definiteness {
  const eps = 1e-9;
  if (l1 > eps && l2 > eps) return 'Positive Definite';
  if (l1 >= -eps && l2 >= -eps) return 'Positive Semidefinite';
  if (l1 < -eps && l2 < -eps) return 'Negative Definite';
  if (l1 <= eps && l2 <= eps) return 'Negative Semidefinite';
  return 'Indefinite';
}

function definitenessBadgeColor(d: Definiteness): string {
  switch (d) {
    case 'Positive Definite':
      return '#16a34a';
    case 'Negative Definite':
      return '#dc2626';
    case 'Positive Semidefinite':
    case 'Negative Semidefinite':
      return '#ca8a04';
    case 'Indefinite':
      return '#6b7280';
  }
}

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const CONTAINER_PADDING = 16;
const PANEL_GAP = 24;
const PANEL_WIDTH_RATIO = 0.55;
const MAX_PANEL_WIDTH = 500;

// ─── Component ───

export default function EigenDecompositionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [a, setA] = useState(3);
  const [b, setB] = useState(1);
  const [c, setC] = useState(2);

  // ─── Eigendecomposition (closed-form 2x2 symmetric) ───

  const eigen = useMemo(() => {
    const trace = a + c;
    const diff = a - c;
    const disc = Math.sqrt(diff * diff + 4 * b * b);

    const lambda1 = (trace - disc) / 2; // smaller
    const lambda2 = (trace + disc) / 2; // larger

    let theta1: number;
    let theta2: number;

    if (Math.abs(b) < 1e-12) {
      theta1 = a <= c ? 0 : Math.PI / 2;
      theta2 = a <= c ? Math.PI / 2 : 0;
    } else {
      theta1 = Math.atan2(b, lambda1 - c);
      theta2 = Math.atan2(b, lambda2 - c);
    }

    const q1: [number, number] = [Math.cos(theta1), Math.sin(theta1)];
    const q2: [number, number] = [Math.cos(theta2), Math.sin(theta2)];
    const definiteness = classifyDefiniteness(lambda1, lambda2);

    return { lambda1, lambda2, q1, q2, definiteness };
  }, [a, b, c]);

  // ─── SVG dimensions ───

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 360;
    return Math.min(
      containerWidth < SM_BREAKPOINT
        ? containerWidth - CONTAINER_PADDING
        : (containerWidth - PANEL_GAP) * PANEL_WIDTH_RATIO,
      MAX_PANEL_WIDTH,
    );
  }, [containerWidth]);
  const panelHeight = 300;
  const margin = { top: 20, right: 20, bottom: 20, left: 20 };

  // ─── D3 rendering ───

  useEffect(() => {
    if (!svgRef.current || panelWidth === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const w = panelWidth - margin.left - margin.right;
    const h = panelHeight - margin.top - margin.bottom;

    // ─── Arrow markers ───
    const arrowColors = ['#2171b5', '#d94801'];
    const defs = svg.append('defs');
    arrowColors.forEach((color) => {
      defs
        .append('marker')
        .attr('id', `arrow-${color.slice(1)}`)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('refX', 6)
        .attr('refY', 3)
        .attr('orient', 'auto')
        .append('polygon')
        .attr('points', '0 0, 6 3, 0 6')
        .attr('fill', color);
    });

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Scale domain: fit eigenvalue magnitudes + unit circle
    const maxR = Math.max(Math.abs(eigen.lambda1), Math.abs(eigen.lambda2), 1.2) * 1.3;
    const xScale = d3.scaleLinear().domain([-maxR, maxR]).range([0, w]);
    const yScale = d3.scaleLinear().domain([-maxR, maxR]).range([h, 0]);

    // Axes
    g.append('line')
      .attr('x1', xScale(-maxR)).attr('y1', yScale(0))
      .attr('x2', xScale(maxR)).attr('y2', yScale(0))
      .style('stroke', 'var(--color-muted)')
      .attr('stroke-width', 0.5);
    g.append('line')
      .attr('x1', xScale(0)).attr('y1', yScale(-maxR))
      .attr('x2', xScale(0)).attr('y2', yScale(maxR))
      .style('stroke', 'var(--color-muted)')
      .attr('stroke-width', 0.5);

    // Unit circle
    const nPts = 200;
    const circleData: [number, number][] = [];
    for (let i = 0; i <= nPts; i++) {
      const theta = (2 * Math.PI * i) / nPts;
      circleData.push([Math.cos(theta), Math.sin(theta)]);
    }

    const lineFn = d3.line<[number, number]>()
      .x((d) => xScale(d[0]))
      .y((d) => yScale(d[1]));

    g.append('path')
      .datum(circleData)
      .attr('d', lineFn)
      .attr('fill', 'none')
      .style('stroke', 'var(--color-muted)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.6);

    // Image ellipse: A * [cos θ, sin θ], computed via spectral decomposition
    // A * v = λ₁(q₁·v)q₁ + λ₂(q₂·v)q₂ — no need for raw matrix entries
    const { lambda1, lambda2, q1, q2 } = eigen;
    const ellipseData: [number, number][] = [];
    for (let i = 0; i <= nPts; i++) {
      const theta = (2 * Math.PI * i) / nPts;
      const cx = Math.cos(theta);
      const sy = Math.sin(theta);
      const dot1 = q1[0] * cx + q1[1] * sy;
      const dot2 = q2[0] * cx + q2[1] * sy;
      ellipseData.push([
        lambda1 * dot1 * q1[0] + lambda2 * dot2 * q2[0],
        lambda1 * dot1 * q1[1] + lambda2 * dot2 * q2[1],
      ]);
    }

    g.append('path')
      .datum(ellipseData)
      .attr('d', lineFn)
      .attr('fill', 'none')
      .style('stroke', 'var(--color-text)')
      .attr('stroke-width', 1.8)
      .attr('opacity', 0.85);

    // ─── Eigenvector arrows ───

    const eigens = [
      { q: eigen.q1, lam: eigen.lambda1, label: 'λ₁' },
      { q: eigen.q2, lam: eigen.lambda2, label: 'λ₂' },
    ];

    eigens.forEach(({ q, lam, label }, idx) => {
      const absLam = Math.abs(lam);
      if (absLam < 1e-9) return;

      const sign = lam < 0 ? -1 : 1;
      const tipX = sign * q[0] * absLam;
      const tipY = sign * q[1] * absLam;

      const color = arrowColors[idx];
      const markerId = `arrow-${color.slice(1)}`;
      const dashed = lam < 0;

      const dx = xScale(tipX) - xScale(0);
      const dy = yScale(tipY) - yScale(0);
      const len = Math.sqrt(dx * dx + dy * dy);

      // Arrow shaft with marker-end
      g.append('line')
        .attr('x1', xScale(0))
        .attr('y1', yScale(0))
        .attr('x2', xScale(tipX))
        .attr('y2', yScale(tipY))
        .attr('stroke', color)
        .attr('stroke-width', 2.5)
        .attr('stroke-dasharray', dashed ? '6,3' : 'none')
        .attr('marker-end', `url(#${markerId})`);

      // Label
      const labelOffset = 14;
      g.append('text')
        .attr('x', len > 0 ? xScale(tipX) + (dx / len) * labelOffset : xScale(tipX))
        .attr('y', len > 0 ? yScale(tipY) + (dy / len) * labelOffset : yScale(tipY))
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('fill', color)
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .text(`${label} = ${lam.toFixed(2)}`);
    });
  }, [eigen, panelWidth]);

  // ─── Slider handlers ───

  const handleA = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setA(parseFloat(e.target.value));
  }, []);
  const handleB = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setB(parseFloat(e.target.value));
  }, []);
  const handleC = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setC(parseFloat(e.target.value));
  }, []);

  // ─── Render ───

  const fmt = (v: number) => v.toFixed(3);

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Two-panel layout */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
        {/* Left panel: SVG */}
        <svg role="img" aria-label="Eigen decomposition explorer visualization"
          ref={svgRef}
          width={panelWidth}
          height={panelHeight}
          className="shrink-0 rounded-lg border border-[var(--color-border)]"
        />

        {/* Right panel: readout */}
        <div
          className="flex-1 space-y-3 rounded-lg border border-[var(--color-border)] p-4"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          {/* Matrix display */}
          <div>
            <div className="mb-1 text-xs font-semibold opacity-60">Matrix A</div>
            <div className="font-mono text-sm leading-relaxed">
              <div className="flex gap-1">
                <span className="text-lg leading-none" style={{ fontFamily: 'serif' }}>[</span>
                <span>{fmt(a)}</span>
                <span className="opacity-40">,</span>
                <span>{fmt(b)}</span>
                <span className="text-lg leading-none" style={{ fontFamily: 'serif' }}>]</span>
              </div>
              <div className="flex gap-1">
                <span className="text-lg leading-none" style={{ fontFamily: 'serif' }}>[</span>
                <span>{fmt(b)}</span>
                <span className="opacity-40">,</span>
                <span>{fmt(c)}</span>
                <span className="text-lg leading-none" style={{ fontFamily: 'serif' }}>]</span>
              </div>
            </div>
          </div>

          {/* Eigenvalues */}
          <div>
            <div className="mb-1 text-xs font-semibold opacity-60">Eigenvalues</div>
            <div className="text-sm">
              <span style={{ color: '#2171b5', fontWeight: 600 }}>λ₁ = {fmt(eigen.lambda1)}</span>
              <span className="mx-2 opacity-30">|</span>
              <span style={{ color: '#d94801', fontWeight: 600 }}>λ₂ = {fmt(eigen.lambda2)}</span>
            </div>
          </div>

          {/* Eigenvectors */}
          <div>
            <div className="mb-1 text-xs font-semibold opacity-60">Eigenvectors</div>
            <div className="font-mono text-sm">
              <div>
                <span style={{ color: '#2171b5' }}>q₁ = [{fmt(eigen.q1[0])}, {fmt(eigen.q1[1])}]</span>
              </div>
              <div>
                <span style={{ color: '#d94801' }}>q₂ = [{fmt(eigen.q2[0])}, {fmt(eigen.q2[1])}]</span>
              </div>
            </div>
          </div>

          {/* Definiteness badge */}
          <div>
            <span
              className="inline-block rounded-md px-2.5 py-1 text-xs font-semibold text-white"
              style={{ backgroundColor: definitenessBadgeColor(eigen.definiteness) }}
            >
              {eigen.definiteness}
            </span>
          </div>

          {/* Factorization label */}
          <div className="text-xs opacity-50" style={{ fontFamily: 'serif', fontStyle: 'italic' }}>
            A = QΛQᵀ
          </div>
        </div>
      </div>

      {/* Sliders */}
      <div className="flex flex-col gap-3 sm:flex-row sm:gap-6">
        <div className="flex flex-1 items-center gap-2">
          <label
            className="min-w-[110px] whitespace-nowrap text-xs font-medium"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            a (top-left): {a.toFixed(1)}
          </label>
          <input
            type="range"
            min={-3}
            max={5}
            step={0.1}
            value={a}
            onChange={handleA}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
        <div className="flex flex-1 items-center gap-2">
          <label
            className="min-w-[110px] whitespace-nowrap text-xs font-medium"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            b (off-diagonal): {b.toFixed(1)}
          </label>
          <input
            type="range"
            min={-3}
            max={3}
            step={0.1}
            value={b}
            onChange={handleB}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
        <div className="flex flex-1 items-center gap-2">
          <label
            className="min-w-[110px] whitespace-nowrap text-xs font-medium"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            c (bottom-right): {c.toFixed(1)}
          </label>
          <input
            type="range"
            min={-3}
            max={5}
            step={0.1}
            value={c}
            onChange={handleC}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
      </div>
    </div>
  );
}

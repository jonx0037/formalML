import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const MD_BREAKPOINT = 1024;
const PANEL_SIZE = 180;
const PANEL_GAP = 8;
const MARGIN = { top: 16, right: 16, bottom: 24, left: 16 };
const N_CIRCLE_PTS = 200;

// ─── Colors ───

const COLOR_V1 = '#2171b5';
const COLOR_V2 = '#d94801';
const COLOR_U1 = '#6a3d9a';
const COLOR_U2 = '#b15928';
const COLOR_ELLIPSE = 'var(--color-text)';
const COLOR_CIRCLE = 'var(--color-muted)';

// ─── SVD math for 2×2 matrices ───

interface SVD2x2 {
  sigma1: number;
  sigma2: number;
  u1: [number, number];
  u2: [number, number];
  v1: [number, number];
  v2: [number, number];
  rank: number;
}

/** Enforce consistent sign: first nonzero entry positive. */
function fixSign(v: [number, number]): [number, number] {
  if (Math.abs(v[0]) > 1e-10) {
    return v[0] > 0 ? v : [-v[0], -v[1]];
  }
  return v[1] >= 0 ? v : [-v[0], -v[1]];
}

/** Closed-form SVD of a 2×2 matrix [[a,b],[c,d]]. */
function svd2x2(a: number, b: number, c: number, d: number): SVD2x2 {
  const eps = 1e-10;

  // A^T A = [[a²+c², ab+cd], [ab+cd, b²+d²]]
  const p = a * a + c * c;
  const q = b * b + d * d;
  const r = a * b + c * d;

  // Eigenvalues of A^T A (symmetric 2×2)
  const sum = p + q;
  const diff = p - q;
  const disc = Math.sqrt(diff * diff + 4 * r * r);

  const lam1 = Math.max(0, (sum + disc) / 2); // larger
  const lam2 = Math.max(0, (sum - disc) / 2); // smaller

  const sigma1 = Math.sqrt(lam1);
  const sigma2 = Math.sqrt(lam2);

  // Eigenvectors of A^T A → right singular vectors V
  let v1: [number, number];
  let v2: [number, number];

  if (Math.abs(r) < eps && Math.abs(diff) < eps) {
    // A^T A is a scalar multiple of I → any orthonormal basis works
    v1 = [1, 0];
    v2 = [0, 1];
  } else if (Math.abs(r) < eps) {
    // Already diagonal
    v1 = p >= q ? [1, 0] : [0, 1];
    v2 = p >= q ? [0, 1] : [1, 0];
  } else {
    // General case: eigenvector for lam1
    const ex = lam1 - q;
    const ey = r;
    const len = Math.sqrt(ex * ex + ey * ey);
    v1 = [ex / len, ey / len];
    v2 = [-v1[1], v1[0]]; // perpendicular
  }

  v1 = fixSign(v1);
  v2 = fixSign(v2);

  // Left singular vectors: u_i = A v_i / sigma_i
  let u1: [number, number];
  let u2: [number, number];

  if (sigma1 > eps) {
    const ux = a * v1[0] + b * v1[1];
    const uy = c * v1[0] + d * v1[1];
    u1 = fixSign([ux / sigma1, uy / sigma1]);
  } else {
    u1 = [1, 0];
  }

  if (sigma2 > eps) {
    const ux = a * v2[0] + b * v2[1];
    const uy = c * v2[0] + d * v2[1];
    u2 = fixSign([ux / sigma2, uy / sigma2]);
  } else {
    // Choose u2 perpendicular to u1
    u2 = [-u1[1], u1[0]];
  }

  const rank = (sigma1 > eps ? 1 : 0) + (sigma2 > eps ? 1 : 0);

  return { sigma1, sigma2, u1, u2, v1, v2, rank };
}

// ─── Geometry helpers ───

/** Points on the unit circle. */
function unitCircle(): [number, number][] {
  return Array.from({ length: N_CIRCLE_PTS + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / N_CIRCLE_PTS;
    return [Math.cos(t), Math.sin(t)] as [number, number];
  });
}

// ─── Component ───

export default function SVDGeometryExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  // Matrix entries
  const [ma, setMa] = useState(2);
  const [mb, setMb] = useState(1);
  const [mc, setMc] = useState(0.5);
  const [md, setMd] = useState(1.5);
  const [symmetric, setSymmetric] = useState(false);

  // Effective matrix (symmetric toggle forces c = b)
  const effC = symmetric ? mb : mc;

  // ─── SVD computation ───

  const svd = useMemo(() => svd2x2(ma, mb, effC, md), [ma, mb, effC, md]);

  // ─── Eigendecomposition for symmetric overlay ───
  const eigen = useMemo(() => {
    if (!symmetric) return null;
    const trace = ma + md;
    const diff = ma - md;
    const disc = Math.sqrt(diff * diff + 4 * mb * mb);
    const lambda1 = (trace + disc) / 2;
    const lambda2 = (trace - disc) / 2;
    return { lambda1, lambda2 };
  }, [symmetric, ma, mb, md]);

  // ─── Panel sizing ───

  const panelW = useMemo(() => {
    if (!containerWidth) return PANEL_SIZE;
    if (containerWidth >= MD_BREAKPOINT) {
      return Math.min(Math.floor((containerWidth - 3 * PANEL_GAP) / 4), 220);
    }
    if (containerWidth >= SM_BREAKPOINT) {
      return Math.min(Math.floor((containerWidth - PANEL_GAP) / 2), 260);
    }
    return Math.min(containerWidth - 16, 320);
  }, [containerWidth]);
  const panelH = panelW;

  // ─── SVG refs ───
  const svgRefs = [
    useRef<SVGSVGElement>(null),
    useRef<SVGSVGElement>(null),
    useRef<SVGSVGElement>(null),
    useRef<SVGSVGElement>(null),
  ];

  // ─── Four-panel D3 rendering ───

  useEffect(() => {
    if (panelW === 0) return;

    const circle = unitCircle();
    const { sigma1, sigma2, u1, u2, v1, v2, rank } = svd;

    const w = panelW - MARGIN.left - MARGIN.right;
    const h = panelH - MARGIN.top - MARGIN.bottom;

    // Panel 1: Input space — unit circle + v₁, v₂
    // Panel 2: After Vᵀ — rotated to axis-aligned
    // Panel 3: After ΣVᵀ — stretched
    // Panel 4: After UΣVᵀ = A — final output

    // Compute transformed point sets
    // Vᵀ has rows = v1, v2. So Vᵀ * x = [v1·x, v2·x]
    const afterVtCorrect = circle.map(([x, y]) => [
      v1[0] * x + v1[1] * y,
      v2[0] * x + v2[1] * y,
    ] as [number, number]);
    const afterSigmaVt = afterVtCorrect.map(([x, y]) => [
      sigma1 * x,
      sigma2 * y,
    ] as [number, number]);
    const afterA = circle.map(([x, y]) => [
      ma * x + mb * y,
      effC * x + md * y,
    ] as [number, number]);

    const panels = [
      { pts: circle, label: 'Input space', vectors: [
        { v: v1, color: COLOR_V1, label: 'v₁' },
        { v: v2, color: COLOR_V2, label: 'v₂' },
      ]},
      { pts: afterVtCorrect, label: 'After Vᵀ', vectors: [
        { v: [1, 0] as [number, number], color: COLOR_V1, label: 'Vᵀv₁' },
        { v: [0, 1] as [number, number], color: COLOR_V2, label: 'Vᵀv₂' },
      ]},
      { pts: afterSigmaVt, label: 'After ΣVᵀ', vectors: [
        { v: [sigma1, 0] as [number, number], color: COLOR_V1, label: `σ₁=${sigma1.toFixed(2)}` },
        { v: [0, sigma2] as [number, number], color: COLOR_V2, label: `σ₂=${sigma2.toFixed(2)}` },
      ]},
      { pts: afterA, label: 'UΣVᵀ = A', vectors: [
        { v: [u1[0] * sigma1, u1[1] * sigma1] as [number, number], color: COLOR_U1, label: 'σ₁u₁' },
        { v: [u2[0] * sigma2, u2[1] * sigma2] as [number, number], color: COLOR_U2, label: 'σ₂u₂' },
      ]},
    ];

    // Compute auto-scale: find the max extent across all panels
    let maxExtent = 1.5;
    for (const panel of panels) {
      for (const [px, py] of panel.pts) {
        maxExtent = Math.max(maxExtent, Math.abs(px), Math.abs(py));
      }
      for (const vec of panel.vectors) {
        maxExtent = Math.max(maxExtent, Math.abs(vec.v[0]), Math.abs(vec.v[1]));
      }
    }
    maxExtent *= 1.15;

    panels.forEach((panel, idx) => {
      const el = svgRefs[idx].current;
      if (!el) return;

      const svg = d3.select(el);
      svg.selectAll('*').remove();

      // Arrow markers
      const defs = svg.append('defs');
      const colors = [...new Set(panel.vectors.map((v) => v.color))];
      colors.forEach((color) => {
        defs
          .append('marker')
          .attr('id', `svd-arrow-${idx}-${color.replace(/[^a-zA-Z0-9]/g, '')}`)
          .attr('markerWidth', 6)
          .attr('markerHeight', 6)
          .attr('refX', 6)
          .attr('refY', 3)
          .attr('orient', 'auto')
          .append('polygon')
          .attr('points', '0 0, 6 3, 0 6')
          .attr('fill', color);
      });

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([-maxExtent, maxExtent]).range([0, w]);
      const yScale = d3.scaleLinear().domain([-maxExtent, maxExtent]).range([h, 0]);

      // Light grid axes
      g.append('line')
        .attr('x1', xScale(-maxExtent)).attr('y1', yScale(0))
        .attr('x2', xScale(maxExtent)).attr('y2', yScale(0))
        .style('stroke', 'var(--color-muted)')
        .attr('stroke-width', 0.5)
        .attr('opacity', 0.5);
      g.append('line')
        .attr('x1', xScale(0)).attr('y1', yScale(-maxExtent))
        .attr('x2', xScale(0)).attr('y2', yScale(maxExtent))
        .style('stroke', 'var(--color-muted)')
        .attr('stroke-width', 0.5)
        .attr('opacity', 0.5);

      // Reference unit circle (dashed) on panels 2-4
      if (idx > 0) {
        const refCircle = unitCircle();
        const lineFn = d3.line<[number, number]>()
          .x((d) => xScale(d[0]))
          .y((d) => yScale(d[1]));
        g.append('path')
          .datum(refCircle)
          .attr('d', lineFn)
          .attr('fill', 'none')
          .style('stroke', COLOR_CIRCLE)
          .attr('stroke-width', 0.5)
          .attr('stroke-dasharray', '3,3')
          .attr('opacity', 0.3);
      }

      // Main curve
      const lineFn = d3.line<[number, number]>()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]));
      g.append('path')
        .datum(panel.pts)
        .attr('d', lineFn)
        .attr('fill', 'none')
        .style('stroke', idx === 0 ? COLOR_CIRCLE : COLOR_ELLIPSE)
        .attr('stroke-width', idx === 0 ? 1.2 : 1.8)
        .attr('opacity', idx === 0 ? 0.7 : 0.85);

      // Vectors
      panel.vectors.forEach(({ v, color, label }) => {
        const mag = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
        if (mag < 1e-9) return;

        const markerId = `svd-arrow-${idx}-${color.replace(/[^a-zA-Z0-9]/g, '')}`;

        g.append('line')
          .attr('x1', xScale(0))
          .attr('y1', yScale(0))
          .attr('x2', xScale(v[0]))
          .attr('y2', yScale(v[1]))
          .attr('stroke', color)
          .attr('stroke-width', 2)
          .attr('marker-end', `url(#${markerId})`);

        // Label offset away from origin
        const dx = xScale(v[0]) - xScale(0);
        const dy = yScale(v[1]) - yScale(0);
        const len = Math.sqrt(dx * dx + dy * dy);
        const offsetX = len > 0 ? (dx / len) * 12 : 0;
        const offsetY = len > 0 ? (dy / len) * 12 : -12;

        g.append('text')
          .attr('x', xScale(v[0]) + offsetX)
          .attr('y', yScale(v[1]) + offsetY)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('fill', color)
          .style('font-family', 'var(--font-sans)')
          .attr('font-size', 10)
          .attr('font-weight', 600)
          .text(label);
      });

      // Rank annotation for degenerate cases
      if (idx === 3 && rank < 2) {
        g.append('text')
          .attr('x', w / 2)
          .attr('y', h - 4)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-muted)')
          .style('font-family', 'var(--font-sans)')
          .attr('font-size', 10)
          .attr('font-style', 'italic')
          .text(rank === 0 ? 'rank 0 — zero matrix' : 'rank 1 — collapses to line');
      }

      // Panel label
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -4)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-muted)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('font-weight', 500)
        .text(panel.label);
    });
  }, [svd, ma, mb, effC, md, panelW, panelH]);

  // ─── Slider handlers ───

  const handleMa = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setMa(parseFloat(e.target.value)), []);
  const handleMb = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setMb(parseFloat(e.target.value)), []);
  const handleMc = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setMc(parseFloat(e.target.value)), []);
  const handleMd = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setMd(parseFloat(e.target.value)), []);
  const handleSymmetric = useCallback(() => setSymmetric((s) => !s), []);

  // ─── Render ───

  const fmt = (v: number) => v.toFixed(3);
  const kappa = svd.sigma2 > 1e-10 ? svd.sigma1 / svd.sigma2 : Infinity;

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Four-panel progression */}
      <div className="flex flex-wrap justify-center gap-2">
        {svgRefs.map((ref, i) => (
          <svg role="img" aria-label={`SVD geometry — ${['Input space', 'After Vᵀ', 'After ΣVᵀ', 'UΣVᵀ = A'][i]}`}
            key={i}
            ref={ref}
            width={panelW}
            height={panelH}
            className="rounded-lg border border-[var(--color-border)]"
          />
        ))}
      </div>

      {/* Readout panel */}
      <div
        className="rounded-lg border border-[var(--color-border)] p-4"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <div className="flex flex-wrap gap-x-8 gap-y-3">
          {/* Matrix display */}
          <div>
            <div className="mb-1 text-xs font-semibold opacity-60">Matrix A</div>
            <div className="font-mono text-sm leading-relaxed">
              <div className="flex gap-1">
                <span className="text-lg leading-none" style={{ fontFamily: 'serif' }}>[</span>
                <span>{fmt(ma)}</span>
                <span className="opacity-40">,</span>
                <span>{fmt(mb)}</span>
                <span className="text-lg leading-none" style={{ fontFamily: 'serif' }}>]</span>
              </div>
              <div className="flex gap-1">
                <span className="text-lg leading-none" style={{ fontFamily: 'serif' }}>[</span>
                <span>{fmt(effC)}</span>
                <span className="opacity-40">,</span>
                <span>{fmt(md)}</span>
                <span className="text-lg leading-none" style={{ fontFamily: 'serif' }}>]</span>
              </div>
            </div>
          </div>

          {/* Singular values */}
          <div>
            <div className="mb-1 text-xs font-semibold opacity-60">Singular Values</div>
            <div className="text-sm">
              <span style={{ color: COLOR_V1, fontWeight: 600 }}>σ₁ = {fmt(svd.sigma1)}</span>
              <span className="mx-2 opacity-30">|</span>
              <span style={{ color: COLOR_V2, fontWeight: 600 }}>σ₂ = {fmt(svd.sigma2)}</span>
            </div>
          </div>

          {/* Singular vectors */}
          <div>
            <div className="mb-1 text-xs font-semibold opacity-60">Right (V)</div>
            <div className="font-mono text-xs">
              <div style={{ color: COLOR_V1 }}>v₁ = [{fmt(svd.v1[0])}, {fmt(svd.v1[1])}]</div>
              <div style={{ color: COLOR_V2 }}>v₂ = [{fmt(svd.v2[0])}, {fmt(svd.v2[1])}]</div>
            </div>
          </div>
          <div>
            <div className="mb-1 text-xs font-semibold opacity-60">Left (U)</div>
            <div className="font-mono text-xs">
              <div style={{ color: COLOR_U1 }}>u₁ = [{fmt(svd.u1[0])}, {fmt(svd.u1[1])}]</div>
              <div style={{ color: COLOR_U2 }}>u₂ = [{fmt(svd.u2[0])}, {fmt(svd.u2[1])}]</div>
            </div>
          </div>

          {/* Badges */}
          <div className="flex flex-col gap-1">
            <span className="inline-block rounded-md bg-[var(--color-muted-bg)] px-2 py-0.5 text-xs font-medium">
              rank {svd.rank}
            </span>
            <span className="inline-block rounded-md bg-[var(--color-muted-bg)] px-2 py-0.5 text-xs font-medium">
              κ = {kappa === Infinity ? '∞' : kappa.toFixed(1)}
            </span>
            {symmetric && eigen && (
              <span className="inline-block rounded-md px-2 py-0.5 text-xs font-semibold text-white" style={{ backgroundColor: '#16a34a' }}>
                Symmetric: λ = {eigen.lambda1.toFixed(2)}, {eigen.lambda2.toFixed(2)}
              </span>
            )}
          </div>
        </div>

        {/* Factorization label */}
        <div className="mt-2 text-xs opacity-50" style={{ fontFamily: 'serif', fontStyle: 'italic' }}>
          A = UΣVᵀ
        </div>
      </div>

      {/* Sliders */}
      <div className="space-y-2">
        <div className="flex flex-col gap-2 sm:flex-row sm:gap-4">
          <div className="flex flex-1 items-center gap-2">
            <label className="min-w-[80px] whitespace-nowrap text-xs font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
              a: {ma.toFixed(1)}
            </label>
            <input type="range" min={-3} max={3} step={0.1} value={ma} onChange={handleMa} className="w-full accent-[var(--color-accent)]" />
          </div>
          <div className="flex flex-1 items-center gap-2">
            <label className="min-w-[80px] whitespace-nowrap text-xs font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
              b: {mb.toFixed(1)}
            </label>
            <input type="range" min={-3} max={3} step={0.1} value={mb} onChange={handleMb} className="w-full accent-[var(--color-accent)]" />
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:gap-4">
          <div className="flex flex-1 items-center gap-2">
            <label
              className="min-w-[80px] whitespace-nowrap text-xs font-medium"
              style={{ fontFamily: 'var(--font-sans)', opacity: symmetric ? 0.4 : 1 }}
            >
              c: {effC.toFixed(1)}
            </label>
            <input
              type="range"
              min={-3}
              max={3}
              step={0.1}
              value={effC}
              onChange={handleMc}
              disabled={symmetric}
              className="w-full accent-[var(--color-accent)]"
              style={{ opacity: symmetric ? 0.4 : 1 }}
            />
          </div>
          <div className="flex flex-1 items-center gap-2">
            <label className="min-w-[80px] whitespace-nowrap text-xs font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
              d: {md.toFixed(1)}
            </label>
            <input type="range" min={-3} max={3} step={0.1} value={md} onChange={handleMd} className="w-full accent-[var(--color-accent)]" />
          </div>
        </div>

        {/* Symmetric toggle */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleSymmetric}
            className="rounded border border-[var(--color-border)] px-3 py-1 text-xs font-medium transition-colors hover:bg-[var(--color-muted-bg)]"
            style={{
              fontFamily: 'var(--font-sans)',
              backgroundColor: symmetric ? 'var(--color-accent)' : undefined,
              color: symmetric ? 'white' : undefined,
            }}
          >
            Make symmetric (c = b)
          </button>
          {symmetric && (
            <span className="text-xs opacity-60" style={{ fontFamily: 'var(--font-sans)' }}>
              SVD reduces to spectral decomposition
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

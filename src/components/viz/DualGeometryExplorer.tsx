import { useState, useMemo, useRef } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  naturalToExpectation,
  solveGeodesicGaussian,
  fisherMetricGaussian,
  clamp,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 420;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';
const GREEN = '#059669';

const MU_RANGE: [number, number] = [-3, 3];
const SIG_RANGE: [number, number] = [0, 3.5];
const DRAG_PAD_X = 0.2;
const DRAG_PAD_Y_LO = 0.15;
const DRAG_PAD_Y_HI = 0.2;

type CoordGrid = 'none' | 'theta' | 'eta';

const fmt = (x: number) => x.toFixed(2);

// ── Helpers ──────────────────────────────────────────────────────────

/** Convert (μ, σ) to natural parameters */
function toNatural(mu: number, sigma: number): [number, number] {
  const s2 = sigma * sigma;
  return [mu / s2, -1 / (2 * s2)];
}

/** Convert natural params to (μ, σ) */
function fromNatural(eta1: number, eta2: number): [number, number] {
  const { mu, sigma2 } = naturalToExpectation(eta1, eta2);
  return [mu, Math.sqrt(Math.max(1e-8, sigma2))];
}

/** Convert (μ, σ) to expectation parameters (E[X], E[X²]) */
function toExpectation(mu: number, sigma: number): [number, number] {
  return [mu, mu * mu + sigma * sigma];
}

/** Convert expectation params to (μ, σ) */
function fromExpectation(e1: number, e2: number): [number, number] {
  const sigma2 = Math.max(1e-8, e2 - e1 * e1);
  return [e1, Math.sqrt(sigma2)];
}

// ── Component ────────────────────────────────────────────────────────

export default function DualGeometryExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [startMu, setStartMu] = useState(-1.5);
  const [startSig, setStartSig] = useState(0.5);
  const [endMu, setEndMu] = useState(1.5);
  const [endSig, setEndSig] = useState(2.0);
  const [showE, setShowE] = useState(true);
  const [showM, setShowM] = useState(true);
  const [showLC, setShowLC] = useState(false);
  const [coordGrid, setCoordGrid] = useState<CoordGrid>('none');
  const draggingStartRef = useRef(true);

  const svgWidth = containerWidth;

  // ── Geodesic curves ──────────────────────────────────────────────
  const curves = useMemo(() => {
    const nPts = 100;

    // e-geodesic: linear in natural parameters
    const [eta1s, eta2s] = toNatural(startMu, startSig);
    const [eta1e, eta2e] = toNatural(endMu, endSig);
    const eGeo: { x: number; y: number }[] = [];
    for (let i = 0; i <= nPts; i++) {
      const t = i / nPts;
      const e1 = eta1s + t * (eta1e - eta1s);
      const e2 = eta2s + t * (eta2e - eta2s);
      const [mu, sig] = fromNatural(e1, e2);
      if (sig > 0.05 && sig < 5 && Math.abs(mu) < 5) {
        eGeo.push({ x: mu, y: sig });
      }
    }

    // m-geodesic: linear in expectation parameters
    const [ex1s, ex2s] = toExpectation(startMu, startSig);
    const [ex1e, ex2e] = toExpectation(endMu, endSig);
    const mGeo: { x: number; y: number }[] = [];
    for (let i = 0; i <= nPts; i++) {
      const t = i / nPts;
      const e1 = ex1s + t * (ex1e - ex1s);
      const e2 = ex2s + t * (ex2e - ex2s);
      // Ensure e2 > e1² (valid σ²)
      if (e2 > e1 * e1 + 0.001) {
        const [mu, sig] = fromExpectation(e1, e2);
        if (sig > 0.05 && sig < 5 && Math.abs(mu) < 5) {
          mGeo.push({ x: mu, y: sig });
        }
      }
    }

    // Levi-Civita geodesic via ODE solver
    // Find initial velocity toward end point (shooting heuristic)
    const dmu0 = endMu - startMu;
    const dsig0 = endSig - startSig;
    // Normalize to unit speed in Fisher metric
    const metric = fisherMetricGaussian(startSig);
    const speed = Math.sqrt(
      metric.g[0][0] * dmu0 * dmu0 + metric.g[1][1] * dsig0 * dsig0
    );
    const scale = speed > 1e-8 ? 1 / speed : 1;
    const lcPts = solveGeodesicGaussian(
      startMu, startSig,
      dmu0 * scale, dsig0 * scale,
      speed, 200
    );
    const lcGeo = lcPts
      .filter((p) => p.y > 0.05 && p.y < 5 && Math.abs(p.x) < 5)
      .map((p) => ({ x: p.x, y: p.y }));

    return { eGeo, mGeo, lcGeo };
  }, [startMu, startSig, endMu, endSig]);

  // ── Coordinate grid ──────────────────────────────────────────────
  const gridLines = useMemo(() => {
    const lines: { pts: { x: number; y: number }[]; dim: number }[] = [];
    const nPts = 80;

    if (coordGrid === 'theta') {
      // Lines of constant η₁ and constant η₂ in (μ, σ) space
      for (let eta1 = -4; eta1 <= 4; eta1 += 0.8) {
        const pts: { x: number; y: number }[] = [];
        for (let i = 0; i <= nPts; i++) {
          const eta2 = -3 + i * 5.9 / nPts;
          if (eta2 < -0.01) {
            const [mu, sig] = fromNatural(eta1, eta2);
            if (sig > 0.05 && sig < 4 && Math.abs(mu) < 4) pts.push({ x: mu, y: sig });
          }
        }
        if (pts.length > 2) lines.push({ pts, dim: 0 });
      }
      for (let eta2 = -3; eta2 <= -0.05; eta2 += 0.3) {
        const pts: { x: number; y: number }[] = [];
        for (let i = 0; i <= nPts; i++) {
          const eta1 = -4 + i * 8 / nPts;
          const [mu, sig] = fromNatural(eta1, eta2);
          if (sig > 0.05 && sig < 4 && Math.abs(mu) < 4) pts.push({ x: mu, y: sig });
        }
        if (pts.length > 2) lines.push({ pts, dim: 1 });
      }
    } else if (coordGrid === 'eta') {
      // Lines of constant μ and constant (μ²+σ²) in (μ, σ) space
      // Constant μ: vertical lines — trivial grid
      for (let mu = -3; mu <= 3; mu += 0.6) {
        const pts: { x: number; y: number }[] = [];
        for (let i = 0; i <= nPts; i++) {
          const sig = 0.1 + i * 3.5 / nPts;
          pts.push({ x: mu, y: sig });
        }
        lines.push({ pts, dim: 0 });
      }
      // Constant E[X²] = μ² + σ²: semicircles in (μ, σ) space
      for (let e2 = 0.5; e2 <= 12; e2 += 1.0) {
        const pts: { x: number; y: number }[] = [];
        for (let i = 0; i <= nPts; i++) {
          const mu = -Math.sqrt(e2) + i * 2 * Math.sqrt(e2) / nPts;
          const sigma2 = e2 - mu * mu;
          if (sigma2 > 0.01) {
            const sig = Math.sqrt(sigma2);
            if (sig > 0.05 && sig < 4 && Math.abs(mu) < 4) pts.push({ x: mu, y: sig });
          }
        }
        if (pts.length > 2) lines.push({ pts, dim: 1 });
      }
    }

    return lines;
  }, [coordGrid]);

  // ── SVG rendering ────────────────────────────────────────────────
  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (svgWidth <= 0) return;

      const margin = { top: 30, right: 20, bottom: 40, left: 50 };
      const w = svgWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain(MU_RANGE).range([0, w]);
      const yScale = d3.scaleLinear().domain(SIG_RANGE).range([h, 0]);

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', w / 2).attr('y', h + 35)
        .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle').style('font-size', '12px').text('μ');
      g.append('text').attr('x', -35).attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle')
        .style('font-size', '12px').attr('transform', `rotate(-90,-35,${h / 2})`).text('σ');

      // Coordinate grid
      const lineGen = d3.line<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y));

      gridLines.forEach((line) => {
        g.append('path')
          .datum(line.pts)
          .attr('d', lineGen)
          .style('fill', 'none')
          .style('stroke', line.dim === 0 ? TEAL : PURPLE)
          .style('stroke-width', 0.6)
          .style('opacity', 0.3);
      });

      // e-geodesic
      if (showE && curves.eGeo.length > 1) {
        g.append('path')
          .datum(curves.eGeo)
          .attr('d', lineGen)
          .style('fill', 'none')
          .style('stroke', TEAL)
          .style('stroke-width', 2.5)
          .style('stroke-dasharray', '6,3');
      }

      // m-geodesic
      if (showM && curves.mGeo.length > 1) {
        g.append('path')
          .datum(curves.mGeo)
          .attr('d', lineGen)
          .style('fill', 'none')
          .style('stroke', PURPLE)
          .style('stroke-width', 2.5)
          .style('stroke-dasharray', '2,3');
      }

      // Levi-Civita geodesic
      if (showLC && curves.lcGeo.length > 1) {
        g.append('path')
          .datum(curves.lcGeo)
          .attr('d', lineGen)
          .style('fill', 'none')
          .style('stroke', GREEN)
          .style('stroke-width', 2.5);
      }

      // Draggable start point
      g.append('circle')
        .attr('cx', xScale(startMu))
        .attr('cy', yScale(startSig))
        .attr('r', 8)
        .style('fill', AMBER)
        .style('stroke', '#fff')
        .style('stroke-width', 2)
        .style('cursor', 'grab');

      // Draggable end point
      g.append('circle')
        .attr('cx', xScale(endMu))
        .attr('cy', yScale(endSig))
        .attr('r', 8)
        .style('fill', AMBER)
        .style('stroke', '#fff')
        .style('stroke-width', 2)
        .style('cursor', 'grab');

      // Invisible overlay for drag — limited to the plot area
      const overlay = g.append('rect')
        .attr('width', w).attr('height', h)
        .style('fill', 'none').style('pointer-events', 'all').style('cursor', 'grab');

      overlay.call(d3.drag<SVGRectElement, unknown>()
        .on('start', (event) => {
          const mx = xScale.invert(event.x);
          const my = yScale.invert(event.y);
          const ds = (mx - startMu) ** 2 + (my - startSig) ** 2;
          const de = (mx - endMu) ** 2 + (my - endSig) ** 2;
          draggingStartRef.current = ds <= de;
        })
        .on('drag', (event) => {
          const mu = clamp(xScale.invert(event.x), MU_RANGE[0] + DRAG_PAD_X, MU_RANGE[1] - DRAG_PAD_X);
          const sig = clamp(yScale.invert(event.y), SIG_RANGE[0] + DRAG_PAD_Y_LO, SIG_RANGE[1] - DRAG_PAD_Y_HI);
          if (draggingStartRef.current) { setStartMu(mu); setStartSig(sig); }
          else { setEndMu(mu); setEndSig(sig); }
        }));

      // Labels
      g.append('text').attr('x', xScale(startMu) + 10).attr('y', yScale(startSig) - 10)
        .style('fill', AMBER).style('font-size', '11px').text(`(${fmt(startMu)}, ${fmt(startSig)})`);
      g.append('text').attr('x', xScale(endMu) + 10).attr('y', yScale(endSig) - 10)
        .style('fill', AMBER).style('font-size', '11px').text(`(${fmt(endMu)}, ${fmt(endSig)})`);

      // Legend
      let ly = 10;
      if (showE) {
        g.append('line').attr('x1', w - 80).attr('x2', w - 55).attr('y1', ly).attr('y2', ly)
          .style('stroke', TEAL).style('stroke-width', 2.5).style('stroke-dasharray', '6,3');
        g.append('text').attr('x', w - 50).attr('y', ly + 4)
          .style('fill', TEAL).style('font-size', '10px').text('e-geodesic');
        ly += 16;
      }
      if (showM) {
        g.append('line').attr('x1', w - 80).attr('x2', w - 55).attr('y1', ly).attr('y2', ly)
          .style('stroke', PURPLE).style('stroke-width', 2.5).style('stroke-dasharray', '2,3');
        g.append('text').attr('x', w - 50).attr('y', ly + 4)
          .style('fill', PURPLE).style('font-size', '10px').text('m-geodesic');
        ly += 16;
      }
      if (showLC) {
        g.append('line').attr('x1', w - 80).attr('x2', w - 55).attr('y1', ly).attr('y2', ly)
          .style('stroke', GREEN).style('stroke-width', 2.5);
        g.append('text').attr('x', w - 50).attr('y', ly + 4)
          .style('fill', GREEN).style('font-size', '10px').text('Levi-Civita');
      }

      // Title
      svg.append('text').attr('x', svgWidth / 2).attr('y', 16)
        .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
        .style('font-size', '13px').style('font-weight', '600')
        .text('Dual Geometry on the Gaussian Manifold');
    },
    [svgWidth, startMu, startSig, endMu, endSig, showE, showM, showLC, curves, gridLines]
  );

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-3">
      <svg role="img" aria-label="Dual geometry explorer visualization" ref={svgRef} width={svgWidth} height={HEIGHT} />

      <div className="mt-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
          <input type="checkbox" checked={showE} onChange={(e) => setShowE(e.target.checked)} className="accent-[var(--color-accent)]" />
          e-geodesic (α = 1)
        </label>
        <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
          <input type="checkbox" checked={showM} onChange={(e) => setShowM(e.target.checked)} className="accent-[var(--color-accent)]" />
          m-geodesic (α = −1)
        </label>
        <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
          <input type="checkbox" checked={showLC} onChange={(e) => setShowLC(e.target.checked)} className="accent-[var(--color-accent)]" />
          Levi-Civita (α = 0)
        </label>

        <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
          Grid:
          <select
            value={coordGrid}
            onChange={(e) => setCoordGrid(e.target.value as CoordGrid)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-2 py-0.5 text-[var(--color-text-primary)]"
          >
            <option value="none">None</option>
            <option value="theta">θ (natural params)</option>
            <option value="eta">η (expectation params)</option>
          </select>
        </label>
      </div>

      <p className="mt-2 text-xs text-[var(--color-text-tertiary)]">
        Drag the two endpoints to compare geodesics under different connections. The e-geodesic is straight in natural parameters; the m-geodesic is straight in expectation parameters; the Levi-Civita geodesic (α = 0) follows the Fisher-Rao metric.
      </p>
    </div>
  );
}

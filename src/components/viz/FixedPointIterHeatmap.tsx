// =============================================================================
// FixedPointIterHeatmap.tsx
//
// §7.1 Implicit-solver convergence is a POSITION-DEPENDENT property of the
// integrator. This viz visualizes the mean fixed-point iteration count per
// generalized-leapfrog step as a heatmap over the banana support:
//
//   At each grid point θ, sample K = 20 momenta from N(0, G(θ)), run a single
//   GL step from (θ, p) at the user-set ε, average the FP iteration counts.
//
// The heatmap reveals where in parameter space the implicit solver works hard
// (high iteration counts) versus where it converges quickly. The §7.1 takeaway
// is that a single ε chosen on the basis of average behavior may produce
// occasional spikes or divergences at specific positions where the metric is
// unfavorable — the chain's worst-case cost lives in the worst-case region.
//
// Heavy-MC slider per CLAUDE.md. Live label on onChange, recompute on
// onMouseUp / onTouchEnd / onKeyUp. The 25 × 25 grid × K = 20 momenta × 1 GL
// step per cell is ~10⁴ GL steps per recompute, runs in under a second.
//
// Controls: ε ∈ [0.05, 0.40] (display-vs-committed), divergence cap.
//
// Computation: in-browser via shared/riemann-hmc.ts.
// Static fallback: /images/topics/riemann-manifold-hmc/07_fp_iter_heatmap.png
// =============================================================================

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bananaLogDensity,
  bananaMetric,
  generalizedLeapfrogStep,
  mulberry32,
  makeGaussian,
  paletteRMHMC,
} from './shared/riemann-hmc';

const SM_BREAKPOINT = 640;
const DEFAULT_EPS = 0.2;
const DEFAULT_MAX_ITERS = 20;
const X_MIN = -3.0;
const X_MAX = 3.0;
const Y_MIN = -5.0;
const Y_MAX = 3.0;
const GRID_N = 25;
const K_MOMENTA = 20;
const BANANA = { a: 1, b: 1 };

interface HeatmapData {
  values: number[]; // GRID_N × GRID_N row-major, mean FP iter count (or NaN if all diverged)
  divergenceFrac: number[]; // GRID_N × GRID_N: fraction of K momenta that diverged
  vMin: number;
  vMax: number;
  totalDivergent: number;
}

function computeHeatmap(eps: number, maxIters: number): HeatmapData {
  const values: number[] = new Array(GRID_N * GRID_N);
  const divFrac: number[] = new Array(GRID_N * GRID_N);
  let vMin = Infinity;
  let vMax = -Infinity;
  let totalDiv = 0;
  const rng = mulberry32(73);
  const gauss = makeGaussian(rng);
  for (let j = 0; j < GRID_N; j++) {
    const t2 = Y_MIN + ((Y_MAX - Y_MIN) * (j + 0.5)) / GRID_N;
    for (let i = 0; i < GRID_N; i++) {
      const t1 = X_MIN + ((X_MAX - X_MIN) * (i + 0.5)) / GRID_N;
      const G = bananaMetric([t1, t2], BANANA);
      // 2×2 Cholesky
      const l00 = Math.sqrt(G[0][0]);
      const l10 = G[1][0] / l00;
      const l11 = Math.sqrt(Math.max(G[1][1] - l10 * l10, 1e-12));
      let sumIters = 0;
      let nConv = 0;
      let nDiv = 0;
      for (let k = 0; k < K_MOMENTA; k++) {
        const z0 = gauss();
        const z1 = gauss();
        const p: [number, number] = [l00 * z0, l10 * z0 + l11 * z1];
        const r = generalizedLeapfrogStep([t1, t2], p, eps, BANANA, { maxIters });
        if (r.converged) {
          sumIters += r.pIters + r.thetaIters;
          nConv++;
        } else {
          nDiv++;
        }
      }
      const idx = j * GRID_N + i;
      if (nConv > 0) {
        values[idx] = sumIters / nConv;
        if (values[idx] < vMin) vMin = values[idx];
        if (values[idx] > vMax) vMax = values[idx];
      } else {
        values[idx] = NaN;
      }
      divFrac[idx] = nDiv / K_MOMENTA;
      totalDiv += nDiv;
    }
  }
  return { values, divergenceFrac: divFrac, vMin, vMax, totalDivergent: totalDiv };
}

// Banana contour cache (independent of ε)
interface ContourData {
  field: number[];
  nx: number;
  ny: number;
  thresholds: number[];
}
function makeBananaContours(): ContourData {
  const nx = 80;
  const ny = 100;
  const field: number[] = new Array(nx * ny);
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = 0; j < ny; j++) {
    const y = Y_MIN + ((Y_MAX - Y_MIN) * j) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / (nx - 1);
      const v = bananaLogDensity([x, y], BANANA);
      field[j * nx + i] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const nLevels = 6;
  const thresholds: number[] = [];
  for (let i = 0; i < nLevels; i++) thresholds.push(lo + ((hi - lo) * (i + 1)) / (nLevels + 1));
  return { field, nx, ny, thresholds };
}

export default function FixedPointIterHeatmap() {
  const [displayEps, setDisplayEps] = useState(DEFAULT_EPS);
  const [committedEps, setCommittedEps] = useState(DEFAULT_EPS);
  const [maxIters, setMaxIters] = useState(DEFAULT_MAX_ITERS);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 720;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.min(620, containerWidth - 24);
  const panelHeight = isMobile ? 380 : 460;

  const contours = useMemo(() => makeBananaContours(), []);
  const heatmap = useMemo(() => computeHeatmap(committedEps, maxIters), [committedEps, maxIters]);

  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 30, right: 60, bottom: 40, left: 50 };
    const innerW = panelWidth - margin.left - margin.right;
    const innerH = panelHeight - margin.top - margin.bottom;
    if (innerW <= 0 || innerH <= 0) return;
    const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
    const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([Y_MIN, Y_MAX]).range([innerH, 0]);

    // Heatmap rectangles
    const cellW = innerW / GRID_N;
    const cellH = innerH / GRID_N;
    // Use Viridis-like sequential scale; clamp at vMax + a bit so divergent cells stand out
    const colorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([heatmap.vMin, Math.max(heatmap.vMax, heatmap.vMin + 1)]);
    for (let j = 0; j < GRID_N; j++) {
      for (let i = 0; i < GRID_N; i++) {
        const v = heatmap.values[j * GRID_N + i];
        const divF = heatmap.divergenceFrac[j * GRID_N + i];
        const x0 = X_MIN + ((X_MAX - X_MIN) * i) / GRID_N;
        const y0 = Y_MIN + ((Y_MAX - Y_MIN) * j) / GRID_N;
        const fill = divF >= 0.5 ? '#d62728' : Number.isFinite(v) ? colorScale(v) : '#d62728';
        g.append('rect')
          .attr('x', xScale(x0))
          .attr('y', yScale(y0) - cellH)
          .attr('width', cellW)
          .attr('height', cellH)
          .style('fill', fill)
          .style('opacity', divF >= 0.5 ? 0.9 : 0.85);
      }
    }
    // Contours overlay (thin, white-ish)
    const contourGen = d3.contours().size([contours.nx, contours.ny]).thresholds(contours.thresholds);
    const cs = contourGen(contours.field);
    const geoPath = d3.geoPath(
      d3.geoTransform({
        point(this: { stream: { point: (x: number, y: number) => void } }, x: number, y: number): void {
          const px = (x / (contours.nx - 1)) * (X_MAX - X_MIN) + X_MIN;
          const py = (y / (contours.ny - 1)) * (Y_MAX - Y_MIN) + Y_MIN;
          this.stream.point(xScale(px), yScale(py));
        },
      }),
    );
    g.append('g')
      .selectAll('path')
      .data(cs)
      .join('path')
      .attr('d', geoPath as unknown as (d: d3.ContourMultiPolygon) => string)
      .style('fill', 'none')
      .style('stroke', 'rgba(255, 255, 255, 0.6)')
      .style('stroke-width', 0.6);

    // Axes
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('fill', 'var(--color-text, #1A1A1A)');
    g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text, #1A1A1A)');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', -8)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .style('fill', 'var(--color-text, #1A1A1A)')
      .text(`Mean FP iterations per GL step (ε = ${committedEps.toFixed(2)})`);
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 32)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'var(--color-text-secondary, #6B6B6B)')
      .text('θ₁');

    // Color scale legend (vertical strip on the right)
    const legendH = innerH;
    const legendX = innerW + 12;
    const legendW = 14;
    const nLegend = 40;
    for (let k = 0; k < nLegend; k++) {
      const v = heatmap.vMin + ((Math.max(heatmap.vMax, heatmap.vMin + 1) - heatmap.vMin) * (nLegend - 1 - k)) / (nLegend - 1);
      g.append('rect')
        .attr('x', legendX)
        .attr('y', (k / nLegend) * legendH)
        .attr('width', legendW)
        .attr('height', legendH / nLegend + 1)
        .style('fill', colorScale(v))
        .style('opacity', 0.85);
    }
    // Legend axis
    const legendScale = d3.scaleLinear().domain([heatmap.vMin, Math.max(heatmap.vMax, heatmap.vMin + 1)]).range([legendH, 0]);
    g.append('g').attr('transform', `translate(${legendX + legendW}, 0)`).call(d3.axisRight(legendScale).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text, #1A1A1A)');
  }, [contours, heatmap, panelWidth, panelHeight, committedEps]);

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 7: Fixed-point iteration count heatmap over the banana"
      style={{ margin: '1.5rem 0' }}
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem 1.25rem',
          alignItems: 'center',
          fontSize: '0.875rem',
          color: 'var(--color-text, #333)',
          marginBottom: '0.75rem',
        }}
      >
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>step size ε:</span>
          <input
            type="range"
            min={0.05}
            max={0.4}
            step={0.01}
            value={displayEps}
            onChange={(e) => setDisplayEps(Number(e.target.value))}
            onMouseUp={(e) => setCommittedEps(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedEps(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedEps(Number((e.target as HTMLInputElement).value))}
            style={{ width: '160px' }}
            aria-label="Step size epsilon"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>ε = {displayEps.toFixed(2)}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>divergence cap:</span>
          <input
            type="number"
            min={5}
            max={50}
            step={5}
            value={maxIters}
            onChange={(e) => setMaxIters(Math.max(5, Math.min(50, Number(e.target.value) || DEFAULT_MAX_ITERS)))}
            style={{ width: '4em', fontSize: '0.85rem' }}
            aria-label="Fixed-point iteration cap"
          />
        </label>
        <span style={{ color: 'var(--color-text-secondary, #6B6B6B)', fontSize: '0.85rem' }}>
          divergent cells: {heatmap.totalDivergent} / {GRID_N * GRID_N * K_MOMENTA}
        </span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <svg ref={svgRef} width={panelWidth} height={panelHeight} role="img" aria-label="Heatmap of mean fixed-point iteration count" />
      </div>
      <figcaption
        style={{
          marginTop: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-text-secondary, #6B6B6B)',
          textAlign: 'center',
        }}
      >
        Figure 7. At each grid point (θ₁, θ₂), {K_MOMENTA} momenta sampled from N(0, G(θ)),
        one generalized-leapfrog step at the chosen ε, mean FP iteration count plotted as
        color. Cells where ≥50% of momenta hit the divergence cap are colored red. At small ε
        iteration counts are uniform; at large ε they spike in regions of unfavorable metric
        geometry — typically near the banana ridge&apos;s outer arms where the metric varies
        fastest. The chain&apos;s worst-case cost lives in the worst-case region.
      </figcaption>
    </figure>
  );
}

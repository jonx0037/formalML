// =============================================================================
// BananaHeadToHeadDashboard.tsx
//
// §9 Worked example: the banana distribution. Side-by-side scatter plots of
// HMC and RMHMC samples on the banana posterior, with a diagnostics table
// summarizing the chain-quality difference. The third sampler — NUTS via
// PyMC — is shown in the static-fallback PNG (09_banana_head_to_head.png)
// since the browser can't run PyMC; the live viz here covers the two
// hand-rolled samplers.
//
// Layout:
//   Top row: two scatter panels (HMC samples, RMHMC samples) overlaid on
//            the banana log-density contour with an optional analytical 1σ
//            ellipse.
//   Bottom row: small diagnostics table — acceptance rate, mean FP iters
//               (RMHMC only), divergences, sample variance, Geyer ESS.
//
// Controls: n_samples slider [400, 3000] (display-vs-committed),
// b ∈ [0, 1.5] slider (display-vs-committed), 1σ ellipse toggle.
//
// Computation: in-browser via shared/riemann-hmc.ts.
// Static fallback: 09_banana_head_to_head.png (HMC + RMHMC + NUTS scatter)
// plus 09_banana_trajectory_geometry.png (single-trajectory comparison).
// =============================================================================

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bananaLogDensity,
  hmcSample,
  rmhmcSample,
  ess,
  mulberry32,
  paletteRMHMC,
} from './shared/riemann-hmc';

const SM_BREAKPOINT = 640;
const DEFAULT_N = 1200;
const DEFAULT_B = 1.0;
const X_MIN = -3.5;
const X_MAX = 3.5;
const Y_MIN = -6;
const Y_MAX = 4;

interface ChainStats {
  samples: number[][];
  acceptance: number;
  divergences: number;
  meanPIters: number;
  meanThetaIters: number;
  wall: number;
  empMean: [number, number];
  empVar: [number, number];
  essBulk: [number, number];
  essPerSec: number;
}

function summarize(c: { samples: number[][]; acceptanceRate: number; divergences: number; meanPIters: number; meanThetaIters: number; wallSeconds: number }): ChainStats {
  const n = c.samples.length;
  let m1 = 0;
  let m2 = 0;
  for (const s of c.samples) {
    m1 += s[0];
    m2 += s[1];
  }
  m1 /= n;
  m2 /= n;
  let v1 = 0;
  let v2 = 0;
  for (const s of c.samples) {
    v1 += (s[0] - m1) ** 2;
    v2 += (s[1] - m2) ** 2;
  }
  v1 /= n;
  v2 /= n;
  const e1 = ess(c.samples.map((s) => s[0]));
  const e2 = ess(c.samples.map((s) => s[1]));
  const essMin = Math.min(e1, e2);
  return {
    samples: c.samples,
    acceptance: c.acceptanceRate,
    divergences: c.divergences,
    meanPIters: c.meanPIters,
    meanThetaIters: c.meanThetaIters,
    wall: c.wallSeconds,
    empMean: [m1, m2],
    empVar: [v1, v2],
    essBulk: [e1, e2],
    essPerSec: c.wallSeconds > 0 ? essMin / c.wallSeconds : 0,
  };
}

interface ContourData {
  field: number[];
  nx: number;
  ny: number;
  thresholds: number[];
}

function makeContours(b: number): ContourData {
  const nx = 70;
  const ny = 90;
  const field: number[] = new Array(nx * ny);
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = 0; j < ny; j++) {
    const y = Y_MIN + ((Y_MAX - Y_MIN) * j) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / (nx - 1);
      const v = bananaLogDensity([x, y], { a: 1, b });
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

function drawPanel(
  svg: SVGSVGElement,
  width: number,
  height: number,
  title: string,
  stats: ChainStats,
  color: string,
  contours: ContourData,
  show1Sigma: boolean,
  analyticVar: [number, number],
): void {
  const sel = d3.select(svg);
  sel.selectAll('*').remove();
  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  if (innerW <= 0 || innerH <= 0) return;
  const g = sel.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
  const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([Y_MIN, Y_MAX]).range([innerH, 0]);

  // Contours
  const contourGen = d3.contours().size([contours.nx, contours.ny]).thresholds(contours.thresholds);
  const cs = contourGen(contours.field);
  const colorScale = d3
    .scaleLinear<string>()
    .domain([contours.thresholds[0], contours.thresholds[contours.thresholds.length - 1]])
    .range(['rgba(44, 62, 80, 0.03)', 'rgba(44, 62, 80, 0.18)']);
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
    .style('fill', (d) => colorScale(d.value))
    .style('stroke', 'var(--color-text-secondary, #6B6B6B)')
    .style('stroke-width', 0.3)
    .style('opacity', 0.85);

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
    .text(title);

  // Samples
  const inBounds = (s: number[]): boolean => s[0] >= X_MIN && s[0] <= X_MAX && s[1] >= Y_MIN && s[1] <= Y_MAX;
  const clipped = stats.samples.filter(inBounds);
  // Subsample to keep DOM size manageable
  const step = Math.max(1, Math.floor(clipped.length / 1500));
  for (let i = 0; i < clipped.length; i += step) {
    g.append('circle')
      .attr('cx', xScale(clipped[i][0]))
      .attr('cy', yScale(clipped[i][1]))
      .attr('r', 1.5)
      .style('fill', color)
      .style('opacity', 0.45);
  }

  // 1σ ellipse around analytical mean = (0, 0), variances = analyticVar
  if (show1Sigma) {
    const rxPix = (Math.sqrt(analyticVar[0]) / (X_MAX - X_MIN)) * innerW;
    const ryPix = (Math.sqrt(analyticVar[1]) / (Y_MAX - Y_MIN)) * innerH;
    g.append('ellipse')
      .attr('cx', xScale(0))
      .attr('cy', yScale(0))
      .attr('rx', rxPix)
      .attr('ry', ryPix)
      .style('fill', 'none')
      .style('stroke', paletteRMHMC.target)
      .style('stroke-width', 1.4)
      .style('stroke-dasharray', '4 3')
      .style('opacity', 0.85);
  }
}

export default function BananaHeadToHeadDashboard() {
  const [displayN, setDisplayN] = useState(DEFAULT_N);
  const [committedN, setCommittedN] = useState(DEFAULT_N);
  const [displayB, setDisplayB] = useState(DEFAULT_B);
  const [committedB, setCommittedB] = useState(DEFAULT_B);
  const [show1Sigma, setShow1Sigma] = useState(true);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 800;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.floor((containerWidth - 16) / 2);
  const panelHeight = isMobile ? 320 : 360;

  const contours = useMemo(() => makeContours(committedB), [committedB]);

  // Analytical reference: mean = (0, 0), var = (a², 1 + 2 a⁴ b²) — with a=1, var = (1, 1 + 2 b²)
  const analyticVar: [number, number] = useMemo(() => [1, 1 + 2 * committedB * committedB], [committedB]);

  const chains = useMemo(() => {
    const params = { a: 1, b: committedB };
    const hmc = hmcSample([0, 0], committedN, 0.1, 25, params, mulberry32(901));
    const rm = rmhmcSample([0, 0], committedN, 0.15, 25, params, mulberry32(902));
    return { hmc: summarize(hmc), rmhmc: summarize(rm) };
  }, [committedN, committedB]);

  const hmcRef = useRef<SVGSVGElement | null>(null);
  const rmhmcRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (hmcRef.current)
      drawPanel(hmcRef.current, panelWidth, panelHeight, 'Standard HMC', chains.hmc, paletteRMHMC.hmc, contours, show1Sigma, analyticVar);
    if (rmhmcRef.current)
      drawPanel(rmhmcRef.current, panelWidth, panelHeight, 'RMHMC (Fisher)', chains.rmhmc, paletteRMHMC.rmhmc, contours, show1Sigma, analyticVar);
  }, [chains, contours, panelWidth, panelHeight, show1Sigma, analyticVar]);

  const cell = (s: ChainStats, key: keyof ChainStats | 'essMin', label: string, formatter: (s: ChainStats) => string) => (
    <td key={key + label} style={{ padding: '0.25rem 0.6rem', fontVariantNumeric: 'tabular-nums', fontSize: '0.85rem' }}>
      {formatter(s)}
    </td>
  );

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 9: Head-to-head HMC vs RMHMC on the banana"
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
          <span>samples / chain:</span>
          <input
            type="range"
            min={400}
            max={3000}
            step={100}
            value={displayN}
            onChange={(e) => setDisplayN(Number(e.target.value))}
            onMouseUp={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            style={{ width: '140px' }}
            aria-label="Samples per chain"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>n = {displayN}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>curvature b:</span>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={displayB}
            onChange={(e) => setDisplayB(Number(e.target.value))}
            onMouseUp={(e) => setCommittedB(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedB(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedB(Number((e.target as HTMLInputElement).value))}
            style={{ width: '140px' }}
            aria-label="Banana curvature b"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>b = {displayB.toFixed(2)}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
          <input
            type="checkbox"
            checked={show1Sigma}
            onChange={(e) => setShow1Sigma(e.target.checked)}
            aria-label="Show analytical 1σ ellipse"
          />
          <span>show analytical 1σ ellipse</span>
        </label>
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: '0.5rem',
          justifyContent: 'center',
        }}
      >
        <svg ref={hmcRef} width={panelWidth} height={panelHeight} role="img" aria-label="HMC samples scatter on banana" />
        <svg ref={rmhmcRef} width={panelWidth} height={panelHeight} role="img" aria-label="RMHMC samples scatter on banana" />
      </div>
      {/* Diagnostics table */}
      <div
        style={{
          marginTop: '0.75rem',
          display: 'flex',
          justifyContent: 'center',
          overflowX: 'auto',
        }}
      >
        <table
          style={{
            borderCollapse: 'collapse',
            fontSize: '0.85rem',
            color: 'var(--color-text, #1A1A1A)',
          }}
        >
          <thead>
            <tr style={{ borderBottom: '1px solid var(--color-border, #ccc)', textAlign: 'right' }}>
              <th style={{ textAlign: 'left', padding: '0.25rem 0.6rem' }}>sampler</th>
              <th style={{ padding: '0.25rem 0.6rem' }}>acc</th>
              <th style={{ padding: '0.25rem 0.6rem' }}>div</th>
              <th style={{ padding: '0.25rem 0.6rem' }}>FP (p, θ)</th>
              <th style={{ padding: '0.25rem 0.6rem' }}>wall (s)</th>
              <th style={{ padding: '0.25rem 0.6rem' }}>ESS<sub>min</sub></th>
              <th style={{ padding: '0.25rem 0.6rem' }}>ESS/s</th>
              <th style={{ padding: '0.25rem 0.6rem' }}>Var[θ₁]</th>
              <th style={{ padding: '0.25rem 0.6rem' }}>Var[θ₂]</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ textAlign: 'right' }}>
              <td style={{ textAlign: 'left', padding: '0.25rem 0.6rem', color: paletteRMHMC.hmc, fontWeight: 600 }}>HMC</td>
              {cell(chains.hmc, 'acceptance', '', (s) => s.acceptance.toFixed(3))}
              {cell(chains.hmc, 'divergences', '', (s) => String(s.divergences))}
              {cell(chains.hmc, 'meanPIters', '', () => '—')}
              {cell(chains.hmc, 'wall', '', (s) => s.wall.toFixed(2))}
              {cell(chains.hmc, 'essMin', '', (s) => Math.min(...s.essBulk).toFixed(0))}
              {cell(chains.hmc, 'essPerSec', '', (s) => s.essPerSec.toFixed(0))}
              {cell(chains.hmc, 'empVar', '1', (s) => s.empVar[0].toFixed(3))}
              {cell(chains.hmc, 'empVar', '2', (s) => s.empVar[1].toFixed(3))}
            </tr>
            <tr style={{ textAlign: 'right' }}>
              <td style={{ textAlign: 'left', padding: '0.25rem 0.6rem', color: paletteRMHMC.rmhmc, fontWeight: 600 }}>RMHMC</td>
              {cell(chains.rmhmc, 'acceptance', '', (s) => s.acceptance.toFixed(3))}
              {cell(chains.rmhmc, 'divergences', '', (s) => String(s.divergences))}
              {cell(chains.rmhmc, 'meanPIters', '', (s) => `(${s.meanPIters.toFixed(1)}, ${s.meanThetaIters.toFixed(1)})`)}
              {cell(chains.rmhmc, 'wall', '', (s) => s.wall.toFixed(2))}
              {cell(chains.rmhmc, 'essMin', '', (s) => Math.min(...s.essBulk).toFixed(0))}
              {cell(chains.rmhmc, 'essPerSec', '', (s) => s.essPerSec.toFixed(0))}
              {cell(chains.rmhmc, 'empVar', '1', (s) => s.empVar[0].toFixed(3))}
              {cell(chains.rmhmc, 'empVar', '2', (s) => s.empVar[1].toFixed(3))}
            </tr>
            <tr style={{ borderTop: '1px solid var(--color-border, #ccc)', color: 'var(--color-text-secondary, #6B6B6B)', textAlign: 'right' }}>
              <td style={{ textAlign: 'left', padding: '0.25rem 0.6rem' }}>analytical</td>
              <td colSpan={5} />
              <td style={{ padding: '0.25rem 0.6rem' }} />
              <td style={{ padding: '0.25rem 0.6rem' }}>{analyticVar[0].toFixed(3)}</td>
              <td style={{ padding: '0.25rem 0.6rem' }}>{analyticVar[1].toFixed(3)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <figcaption
        style={{
          marginTop: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-text-secondary, #6B6B6B)',
          textAlign: 'center',
        }}
      >
        Figure 9. Side-by-side HMC vs RMHMC samples on the banana posterior at the current
        b. Diagnostics table shows acceptance rate, divergences, mean FP iterations
        (RMHMC only), wall-clock, Geyer-style ESS<sub>min</sub>, ESS/sec, and empirical
        variances against the analytical Var[θ₁] = a² = 1, Var[θ₂] = 1 + 2 a⁴ b². The
        third sampler (NUTS) appears in the static-fallback PNG since it requires PyMC.
      </figcaption>
    </figure>
  );
}

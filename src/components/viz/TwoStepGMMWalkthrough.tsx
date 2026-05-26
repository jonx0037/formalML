import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  RUNNING_EXAMPLE_A,
  RUNNING_EXAMPLE_THETA0,
  affineCriterionSurface,
  identityMat,
  mulberry32,
  sampleSensorObservations,
  twoStepAffineGMM,
  type Mat,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 7.1 — Two-step GMM walkthrough (central viz #2).
//
// Three side-by-side panels mirroring Algorithm 7.1:
//   Panel A — step-1 criterion surface with W^(1) = I, marker on θ̂^(1).
//   Panel B — Ω̂_n heatmap (4×4 estimated moment covariance from step-1
//             residuals). Cell shading is value-relative.
//   Panel C — step-2 criterion surface with W = Ω̂_n^{-1}, marker on θ̂^(2).
//             θ̂^(1) is shown as a faint ghost for comparison.
//
// Reader controls n and σ_3 (sensor-3 heteroskedasticity, makes Ω̂ deviate
// from identity). Commit-on-release for n; σ_3 is light enough to refresh
// live.
// -----------------------------------------------------------------------------

const N_OPTIONS = [50, 100, 200, 500, 1000];
const GRID_RES = 40;
const SURFACE_H = 280;
const HEATMAP_H = 240;
const MARGIN = { top: 18, right: 14, bottom: 38, left: 50 };

interface WalkthroughState {
  twoStep: ReturnType<typeof twoStepAffineGMM>;
  Xbar: Vec;
  surf1: ReturnType<typeof affineCriterionSurface>;
  surf2: ReturnType<typeof affineCriterionSurface>;
}

function computeState(n: number, sigma3: number, seed: number): WalkthroughState {
  const A = RUNNING_EXAMPLE_A;
  const theta0 = RUNNING_EXAMPLE_THETA0;
  const sigma = [1, 1, sigma3, 1];
  const rng = mulberry32(seed);
  const X = sampleSensorObservations(rng, theta0, A, n, sigma);
  const Xbar: Vec = [0, 0, 0, 0];
  for (let i = 0; i < n; i++) for (let j = 0; j < 4; j++) Xbar[j] += X[i][j];
  for (let j = 0; j < 4; j++) Xbar[j] /= n;
  const twoStep = twoStepAffineGMM(X, A, identityMat(4));
  const surf1 = affineCriterionSurface(A, Xbar, identityMat(4), [0.4, 1.6], [0.4, 1.6], GRID_RES, n);
  const surf2 = affineCriterionSurface(A, Xbar, twoStep.OmegaHatInv, [0.4, 1.6], [0.4, 1.6], GRID_RES, n);
  return { twoStep, Xbar, surf1, surf2 };
}

function renderSurface(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  surface: ReturnType<typeof affineCriterionSurface>,
  thetaMain: Vec,
  thetaGhost: Vec | null,
  title: string,
) {
  svg.selectAll('*').remove();
  const innerW = w - MARGIN.left - MARGIN.right;
  const innerH = SURFACE_H - MARGIN.top - MARGIN.bottom;
  if (innerW <= 0) return;
  const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

  const xScale = d3.scaleLinear().domain([0.4, 1.6]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([0.4, 1.6]).range([innerH, 0]);

  const flat: number[] = [];
  for (let i = 0; i < GRID_RES; i++) for (let j = 0; j < GRID_RES; j++) flat.push(surface.J[i][j]);
  const Jmax = d3.max(flat) ?? 1;
  const Jmin = d3.min(flat) ?? 0;
  const span = Math.max(Jmax - Jmin, 1e-9);
  const levels: number[] = [];
  for (let k = 1; k <= 7; k++) levels.push(Jmin + Math.pow(k / 7, 2.0) * span);

  const contours = d3.contours().size([GRID_RES, GRID_RES]).thresholds(levels)(flat);
  const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([Jmax, Jmin]);
  const cellW = innerW / (GRID_RES - 1);
  const cellH = innerH / (GRID_RES - 1);

  g.append('g')
    .attr('transform', `translate(0, ${innerH}) scale(${cellW}, ${-cellH})`)
    .selectAll('path')
    .data(contours)
    .enter()
    .append('path')
    .attr('d', d3.geoPath() as never)
    .style('fill', (d) => colorScale(d.value))
    .style('stroke', 'var(--color-border)')
    .style('stroke-width', 0.4 / Math.min(cellW, cellH))
    .style('opacity', 0.85);

  g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
  g.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');

  // θ₀ cross.
  const x0 = xScale(1);
  const y0 = yScale(1);
  g.append('line').attr('x1', x0 - 6).attr('x2', x0 + 6).attr('y1', y0).attr('y2', y0).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);
  g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', y0 - 6).attr('y2', y0 + 6).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);

  if (thetaGhost) {
    g.append('circle').attr('cx', xScale(thetaGhost[0])).attr('cy', yScale(thetaGhost[1])).attr('r', 4).style('fill', 'var(--color-text-secondary)').attr('opacity', 0.6);
  }
  g.append('circle').attr('cx', xScale(thetaMain[0])).attr('cy', yScale(thetaMain[1])).attr('r', 4.5).style('fill', 'var(--color-accent)').style('stroke', 'var(--color-bg)').style('stroke-width', 1.2);

  g.append('text').attr('x', innerW / 2).attr('y', -4).attr('text-anchor', 'middle').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(title);
}

export default function TwoStepGMMWalkthrough() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(2); // n = 200
  const [sigma3, setSigma3] = useState(2.5);
  const [seed, setSeed] = useState(42);

  const n = N_OPTIONS[nIdx];
  const state = useMemo(() => computeState(n, sigma3, seed), [n, sigma3, seed]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const surf1Ref = useD3<SVGSVGElement>((svg) => renderSurface(svg, w, state.surf1, state.twoStep.theta1, null, `Step 1: W = I    θ̂^(1) = (${state.twoStep.theta1[0].toFixed(2)}, ${state.twoStep.theta1[1].toFixed(2)})`), [state, w]);
  const surf2Ref = useD3<SVGSVGElement>((svg) => renderSurface(svg, w, state.surf2, state.twoStep.theta2, state.twoStep.theta1, `Step 2: W = Ω̂⁻¹    θ̂^(2) = (${state.twoStep.theta2[0].toFixed(2)}, ${state.twoStep.theta2[1].toFixed(2)})`), [state, w]);

  const heatRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEATMAP_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const L = 4;
      const cellSize = Math.min(innerW / L, innerH / L);
      const values = state.twoStep.OmegaHat.flat();
      const vMax = Math.max(...values.map(Math.abs));
      const colorScale = d3.scaleSequential(d3.interpolateRdYlBu).domain([vMax, -vMax]);

      for (let i = 0; i < L; i++) {
        for (let j = 0; j < L; j++) {
          const v = state.twoStep.OmegaHat[i][j];
          g.append('rect')
            .attr('x', j * cellSize).attr('y', i * cellSize)
            .attr('width', cellSize - 1).attr('height', cellSize - 1)
            .style('fill', colorScale(v));
          g.append('text')
            .attr('x', j * cellSize + cellSize / 2)
            .attr('y', i * cellSize + cellSize / 2 + 4)
            .attr('text-anchor', 'middle')
            .style('font-family', 'var(--font-mono)')
            .style('font-size', '11px')
            .style('fill', Math.abs(v) > vMax * 0.5 ? '#fff' : 'var(--color-text)')
            .text(v.toFixed(2));
        }
      }

      // Axis labels.
      for (let i = 0; i < L; i++) {
        g.append('text').attr('x', -8).attr('y', i * cellSize + cellSize / 2 + 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text(`g_${i + 1}`);
        g.append('text').attr('x', i * cellSize + cellSize / 2).attr('y', -6).attr('text-anchor', 'middle').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text(`g_${i + 1}`);
      }
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 28).attr('text-anchor', 'middle').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(`Ω̂_n estimated from step-1 residuals (J_n(θ̂^(2), Ω̂⁻¹) = ${state.twoStep.Jhat.toFixed(2)})`);
    },
    [state, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '160px' }}>
          <span>n = <strong>{n}</strong></span>
          <input type="range" min={0} max={N_OPTIONS.length - 1} step={1} value={nIdx} aria-label="Sample size" onChange={(e) => setNIdx(Number(e.target.value))} />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>σ_3 = <strong>{sigma3.toFixed(2)}</strong></span>
          <input type="range" min={0.5} max={4.0} step={0.1} value={sigma3} aria-label="Sensor 3 noise level" onChange={(e) => setSigma3(Number(e.target.value))} />
        </label>
        <button type="button" onClick={() => setSeed((s) => s + 1)} style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)', fontSize: '12px', border: '1px solid var(--color-border)', background: 'var(--color-bg)', color: 'var(--color-text)', borderRadius: '4px', cursor: 'pointer', alignSelf: 'flex-end' }}>new seed</button>
      </div>
      <svg ref={surf1Ref} width={w} height={SURFACE_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="Step 1 criterion surface (W = I)" />
      <svg ref={heatRef} width={w} height={HEATMAP_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Estimated Ω̂_n heatmap from step-1 residuals" />
      <svg ref={surf2Ref} width={w} height={SURFACE_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Step 2 criterion surface (W = Ω̂⁻¹)" />
    </div>
  );
}

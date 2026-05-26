import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gaussianFrom,
  mulberry32,
  olsHetSandwich,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 11.2 — Sandwich saves coverage under heteroskedasticity.
//
// Quasi-MLE on a misspecified Gaussian-homoskedastic model. Y = β X + ε
// with ε | X ~ N(0, σ(X)²), σ(X) = |X| · γ_scale. OLS is QMLE; its naive
// CI assumes homoskedasticity. The sandwich variance (Eicker-Huber-White)
// is consistent under arbitrary heteroskedasticity.
//
// Two panels:
//   Panel A — Bar comparison of empirical 95% CI coverage rates (naive vs
//     sandwich) at the user's heteroskedasticity scale, B = 500 replicates.
//   Panel B — Coverage % vs heteroskedasticity scale: as γ grows, the naive
//     CI under-covers more severely while the sandwich stays near nominal.
// -----------------------------------------------------------------------------

const N = 200;
const B = 500;
const TRUTH = 1.0;
const GAMMA_GRID = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const PANEL_H = 200;
const LINE_H = 280;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 60 };

const COLOR_NAIVE = '#d09a3e';
const COLOR_SAND = 'var(--color-accent)';

interface CoverageRun { naive: number; sandwich: number; }

function runCoverage(n: number, gammaScale: number): CoverageRun {
  let cN = 0; let cS = 0;
  for (let b = 0; b < B; b++) {
    const rng = mulberry32(110000 + b * 11 + Math.floor(gammaScale * 1000));
    const gauss = gaussianFrom(rng);
    const X: Vec = []; const Y: Vec = [];
    for (let i = 0; i < n; i++) {
      const x = (rng() - 0.5) * 4; // U(-2, 2)
      X.push(x);
      const sigma = 0.3 + gammaScale * Math.abs(x);
      Y.push(x * TRUTH + sigma * gauss());
    }
    const r = olsHetSandwich(Y, X);
    if (Math.abs(r.beta - TRUTH) < r.ciNaiveHW) cN += 1;
    if (Math.abs(r.beta - TRUTH) < r.ciSandwichHW) cS += 1;
  }
  return { naive: cN / B, sandwich: cS / B };
}

export default function SandwichSavesCoverage() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [gamma, setGamma] = useState(1.5);
  const [gammaDisplay, setGammaDisplay] = useState(1.5);

  const current = useMemo(() => runCoverage(N, gamma), [gamma]);
  const grid = useMemo(() => GAMMA_GRID.map((g) => ({ g, ...runCoverage(N, g) })), []);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const barRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - 16;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},6)`);

      const labels = ['naive CI', 'sandwich CI'];
      const values = [current.naive, current.sandwich];
      const colors = [COLOR_NAIVE, COLOR_SAND];
      const xScale = d3.scaleBand<string>().domain(labels).range([0, innerW * 0.6]).padding(0.3);
      const yScale = d3.scaleLinear().domain([0, 1.05]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale)).selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)').style('font-size', '11px');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.0%'))).selectAll('text').style('fill', 'var(--color-text)');

      // Nominal 95% reference.
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(0.95)).attr('y2', yScale(0.95)).attr('stroke-dasharray', '4 3').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1);
      g.append('text').attr('x', innerW - 4).attr('y', yScale(0.95) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('nominal 95%');

      labels.forEach((label, i) => {
        const x = (xScale(label) ?? 0);
        g.append('rect').attr('x', x).attr('y', yScale(values[i])).attr('width', xScale.bandwidth()).attr('height', innerH - yScale(values[i])).style('fill', colors[i]).style('opacity', 0.85);
        g.append('text').attr('x', x + xScale.bandwidth() / 2).attr('y', yScale(values[i]) - 6).attr('text-anchor', 'middle').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(`${(values[i] * 100).toFixed(1)}%`);
      });
    },
    [current, w],
  );

  const lineRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = LINE_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0, GAMMA_GRID[GAMMA_GRID.length - 1]]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0.4, 1.05]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.1~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6).tickFormat(d3.format('.0%'))).selectAll('text').style('fill', 'var(--color-text)');

      // 95% nominal line.
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(0.95)).attr('y2', yScale(0.95)).attr('stroke-dasharray', '4 3').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1);
      g.append('text').attr('x', innerW - 4).attr('y', yScale(0.95) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('nominal 95%');

      const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1])).curve(d3.curveCatmullRom);
      const naivePts: [number, number][] = grid.map((p) => [p.g, p.naive]);
      const sandPts: [number, number][] = grid.map((p) => [p.g, p.sandwich]);
      g.append('path').attr('d', lineGen(naivePts)).attr('fill', 'none').style('stroke', COLOR_NAIVE).attr('stroke-width', 2);
      g.append('path').attr('d', lineGen(sandPts)).attr('fill', 'none').style('stroke', COLOR_SAND).attr('stroke-width', 2);
      naivePts.forEach((p) => g.append('circle').attr('cx', xScale(p[0])).attr('cy', yScale(p[1])).attr('r', 3.5).style('fill', COLOR_NAIVE));
      sandPts.forEach((p) => g.append('circle').attr('cx', xScale(p[0])).attr('cy', yScale(p[1])).attr('r', 3.5).style('fill', COLOR_SAND));

      // Current γ marker.
      g.append('line').attr('x1', xScale(gamma)).attr('x2', xScale(gamma)).attr('y1', 0).attr('y2', innerH).attr('stroke-dasharray', '2 3').style('stroke', 'var(--color-text)').style('stroke-width', 1);

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 4)`);
      legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 6).attr('y2', 6).style('stroke', COLOR_NAIVE).style('stroke-width', 2);
      legend.append('text').attr('x', 24).attr('y', 10).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_NAIVE).text('naive (homoskedastic) CI');
      legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 26).attr('y2', 26).style('stroke', COLOR_SAND).style('stroke-width', 2);
      legend.append('text').attr('x', 24).attr('y', 30).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_SAND).text('sandwich (HC0) CI');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('heteroskedasticity scale γ');
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('empirical coverage');
    },
    [grid, gamma, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '240px' }}>
          <span>γ = <strong>{gammaDisplay.toFixed(2)}</strong> (σ(X) = 0.3 + γ · |X|)</span>
          <input type="range" min={0} max={3} step={0.1} value={gammaDisplay} aria-label="Heteroskedasticity scale" onChange={(e) => setGammaDisplay(Number(e.target.value))} onMouseUp={(e) => setGamma(Number((e.target as HTMLInputElement).value))} onTouchEnd={(e) => setGamma(Number((e.target as HTMLInputElement).value))} onKeyUp={(e) => setGamma(Number((e.target as HTMLInputElement).value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', alignSelf: 'flex-end' }}>
          gap: naive under-covers by <strong style={{ color: 'var(--color-text)' }}>{((0.95 - current.naive) * 100).toFixed(1)}pp</strong>;
          sandwich within <strong style={{ color: 'var(--color-text)' }}>{(Math.abs(0.95 - current.sandwich) * 100).toFixed(1)}pp</strong>
        </div>
      </div>
      <svg ref={barRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="Coverage bars at current γ" />
      <svg ref={lineRef} width={w} height={LINE_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Coverage rate vs heteroskedasticity" />
    </div>
  );
}

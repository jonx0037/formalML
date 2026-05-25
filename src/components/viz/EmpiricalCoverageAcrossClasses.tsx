import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gaussianMixtureHW,
  mulberry32,
  gaussianFrom,
  subExponentialHW,
  zHalf,
} from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 5.2 — Empirical false-rejection rate across distribution classes.
//
// Three panels (sub-Gaussian, Poisson(1), Bernoulli(0.05)). In each panel,
// the fixed-n band's empirical false-rejection rate (red) is compared
// against the mixture CS's empirical rate (blue) under the aggressive-
// analyst stopping rule.
//
// The mixture stays bounded by δ uniformly across all three classes; the
// fixed-n band fails in every regime.
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const PANEL_H = 220;
const MARGIN = { top: 26, right: 12, bottom: 36, left: 46 };
const B_REPLICATES = 1500;
const N_MAX = 5000;
const N_GRID = [50, 100, 200, 500, 1000, 2000, 5000];

const COLOR_FIXED = '#c44545';
const COLOR_MIX = 'var(--color-accent)';
const COLOR_NULL = 'var(--color-text-secondary)';

type DGP = 'sub-Gaussian' | 'Poisson' | 'Bernoulli(0.05)';

interface ClassSim {
  fixedRate: { n: number; rate: number }[];
  mixRate: { n: number; rate: number }[];
  label: string;
}

function samplePoisson(rng: () => number, lambda: number): number {
  // Knuth's algorithm
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rng();
  } while (p > L);
  return k - 1;
}

function simulateClass(dgp: DGP, delta: number): ClassSim {
  const rng = mulberry32(dgp === 'sub-Gaussian' ? 20260901 : dgp === 'Poisson' ? 20260902 : 20260903);
  const gauss = gaussianFrom(rng);
  const z = zHalf(delta);

  const firstExitFixed = new Int32Array(B_REPLICATES).fill(-1);
  const firstExitMix = new Int32Array(B_REPLICATES).fill(-1);

  const mu = dgp === 'sub-Gaussian' ? 0 : dgp === 'Poisson' ? 1 : 0.05;

  for (let b = 0; b < B_REPLICATES; b++) {
    let s = 0;
    for (let n = 1; n <= N_MAX; n++) {
      let x: number;
      if (dgp === 'sub-Gaussian') x = gauss();
      else if (dgp === 'Poisson') x = samplePoisson(rng, 1);
      else x = rng() < 0.05 ? 1 : 0;
      s += x - mu; // center
      const mean = s / n;
      // Fixed-n: ±z/√n with σ from class
      let sigmaFixed: number;
      if (dgp === 'sub-Gaussian') sigmaFixed = 1;
      else if (dgp === 'Poisson') sigmaFixed = 1; // sqrt(λ) = 1 at λ=1
      else sigmaFixed = 0.5; // Hoeffding worst-case Bernoulli
      const hwFixed = sigmaFixed * z / Math.sqrt(n);
      // Mixture: class-specific
      let hwMix: number;
      if (dgp === 'sub-Gaussian') hwMix = gaussianMixtureHW(n, 1, delta, 1);
      else if (dgp === 'Poisson') hwMix = subExponentialHW(n, 1, 1, delta);
      else hwMix = gaussianMixtureHW(n, 1, delta, 0.5);
      if (firstExitFixed[b] === -1 && Math.abs(mean) >= hwFixed) firstExitFixed[b] = n;
      if (firstExitMix[b] === -1 && Math.abs(mean) >= hwMix) firstExitMix[b] = n;
    }
  }

  const fixedRate: { n: number; rate: number }[] = [];
  const mixRate: { n: number; rate: number }[] = [];
  for (const nmax of N_GRID) {
    let cf = 0;
    let cm = 0;
    for (let b = 0; b < B_REPLICATES; b++) {
      if (firstExitFixed[b] > 0 && firstExitFixed[b] <= nmax) cf++;
      if (firstExitMix[b] > 0 && firstExitMix[b] <= nmax) cm++;
    }
    fixedRate.push({ n: nmax, rate: cf / B_REPLICATES });
    mixRate.push({ n: nmax, rate: cm / B_REPLICATES });
  }

  return { fixedRate, mixRate, label: dgp };
}

function renderPanel(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  panelW: number,
  sim: ClassSim,
  delta: number,
): void {
  svg.selectAll('*').remove();
  const innerW = panelW - MARGIN.left - MARGIN.right;
  const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
  if (innerW <= 0) return;
  const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

  const xScale = d3.scaleLog().domain([50, N_MAX]).range([0, innerW]);
  const yMax = Math.max(0.5, ...sim.fixedRate.map((d) => d.rate)) * 1.05;
  const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

  g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(3, '~s')).selectAll('text').style('fill', 'var(--color-text)');
  g.append('g').call(d3.axisLeft(yScale).tickFormat(d3.format('.0%'))).selectAll('text').style('fill', 'var(--color-text)');

  g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(delta)).attr('y2', yScale(delta)).style('stroke', COLOR_NULL).attr('stroke-width', 1).attr('stroke-dasharray', '4 3');

  const line = d3.line<{ n: number; rate: number }>().x((d) => xScale(d.n)).y((d) => yScale(d.rate)).curve(d3.curveStepAfter);
  g.append('path').datum(sim.fixedRate).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_FIXED).attr('stroke-width', 1.8);
  g.append('path').datum(sim.mixRate).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_MIX).attr('stroke-width', 1.8);
  // markers
  g.append('g').selectAll('circle').data(sim.fixedRate).join('circle').attr('cx', (d) => xScale(d.n)).attr('cy', (d) => yScale(d.rate)).attr('r', 2.5).style('fill', COLOR_FIXED);
  g.append('g').selectAll('circle').data(sim.mixRate).join('circle').attr('cx', (d) => xScale(d.n)).attr('cy', (d) => yScale(d.rate)).attr('r', 2.5).style('fill', COLOR_MIX);

  g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text(sim.label);
}

export default function EmpiricalCoverageAcrossClasses() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [delta, setDelta] = useState(0.05);

  const sims = useMemo(() => {
    return (['sub-Gaussian', 'Poisson', 'Bernoulli(0.05)'] as DGP[]).map((d) => simulateClass(d, delta));
  }, [delta]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = Math.max(280, Math.min(containerWidth || 720, 820));
  const panelW = isMobile ? w : Math.floor(w / 3) - 8;

  const r0 = useD3<SVGSVGElement>((svg) => renderPanel(svg, panelW, sims[0], delta), [sims, delta, panelW]);
  const r1 = useD3<SVGSVGElement>((svg) => renderPanel(svg, panelW, sims[1], delta), [sims, delta, panelW]);
  const r2 = useD3<SVGSVGElement>((svg) => renderPanel(svg, panelW, sims[2], delta), [sims, delta, panelW]);

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>δ = <strong>{delta.toFixed(3)}</strong></span>
          <input type="range" min={0.01} max={0.20} step={0.005} value={delta} aria-label="Confidence level" onChange={(e) => setDelta(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
          <span style={{ color: COLOR_FIXED }}>■ fixed-n band</span> &nbsp;
          <span style={{ color: COLOR_MIX }}>■ anytime-valid mixture</span>
          <br /> Aggressive stopping rule; B = {B_REPLICATES} replicates per class.
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '8px' }}>
        <svg ref={r0} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={r1} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={r2} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}

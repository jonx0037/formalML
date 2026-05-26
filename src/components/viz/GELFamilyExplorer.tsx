import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  RUNNING_EXAMPLE_A,
  RUNNING_EXAMPLE_THETA0,
  cueAffineGridEstimate,
  elAffineGridEstimate,
  iteratedAffineGMM,
  mulberry32,
  sampleSensorObservations,
  twoStepAffineGMM,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 10.1 — Cressie-Read carrier family and the bias hierarchy.
//
// Two panels:
//   Panel A — Cressie-Read carrier ρ_γ(v) plotted across v ∈ [-0.5, 0.5] for
//     γ ∈ {−1, 0, 1} (ET / EL / CUE). The user slides γ continuously and the
//     curve morphs; the Newey-Smith (2004) GEL family interpolates these.
//   Panel B — Bar chart of mean |bias| of four estimators on the running
//     example at small n (so two-step bias is large enough to see):
//     two-step, iterated, CUE, EL. Reader controls n; B=40 replicates.
//
// EL is grid-search-based and is the slow estimator (~5M flops per replicate).
// Total at B=40, resolution=15: ~30M flops per refresh ≈ 0.4s in browser.
// Commit-on-release for n.
// -----------------------------------------------------------------------------

const N_OPTIONS = [40, 75, 150, 300];
const B = 40;
const EL_GRID_RES = 15;
const PANEL_H = 240;
const BAR_H = 250;
const MARGIN = { top: 24, right: 18, bottom: 42, left: 56 };

const COLOR_TWO = '#9aa3ad';
const COLOR_ITER = '#7ca5c8';
const COLOR_CUE = '#d09a3e';
const COLOR_EL = 'var(--color-accent)';

// Cressie-Read carrier (Newey-Smith 2004 Table 1, normalized to
// ρ'(0) = ρ''(0) = -1). Three special points:
//   γ = -1 → ET (exponential tilting):  ρ(v) = -exp(v) + v + 1 (centered)
//   γ =  0 → EL (empirical likelihood): ρ(v) = log(1 - v)
//   γ =  1 → CUE:                       ρ(v) = -v²/2 - v
// Interpolated form via the Cressie-Read divergence: ρ_γ(v) = -((1 + γv)^{(γ+1)/γ} - 1)/(γ+1)
// (γ ≠ 0; the EL form is the limit at γ → 0).
function rhoCressieRead(v: number, gamma: number): number {
  if (Math.abs(gamma) < 1e-8) return Math.log(Math.max(1 - v, 1e-12));
  const base = 1 + gamma * v;
  if (base <= 0) return -Infinity;
  return -(Math.pow(base, (gamma + 1) / gamma) - 1) / (gamma + 1);
}

interface BiasBars {
  twoStep: number;
  iterated: number;
  cue: number;
  el: number;
}

function computeBias(n: number): BiasBars {
  const theta0 = RUNNING_EXAMPLE_THETA0;
  const A = RUNNING_EXAMPLE_A;
  const sigma = [1, 1, 1.5, 1];
  let bTwo = 0; let bIter = 0; let bCue = 0; let bEL = 0;
  for (let b = 0; b < B; b++) {
    const rng = mulberry32(100000 + b * 23);
    const X = sampleSensorObservations(rng, theta0, A, n, sigma);
    const twoStep = twoStepAffineGMM(X, A);
    const iter = iteratedAffineGMM(X, A, undefined, 1e-6, 20);
    const cue = cueAffineGridEstimate(X, A, [0.5, 1.5, 0.5, 1.5], EL_GRID_RES);
    const el = elAffineGridEstimate(X, A, [0.5, 1.5, 0.5, 1.5], EL_GRID_RES);
    bTwo += Math.hypot(twoStep.theta2[0] - theta0[0], twoStep.theta2[1] - theta0[1]);
    bIter += Math.hypot(iter.fixedPoint[0] - theta0[0], iter.fixedPoint[1] - theta0[1]);
    bCue += Math.hypot(cue[0] - theta0[0], cue[1] - theta0[1]);
    bEL += Math.hypot(el[0] - theta0[0], el[1] - theta0[1]);
  }
  return { twoStep: bTwo / B, iterated: bIter / B, cue: bCue / B, el: bEL / B };
}

export default function GELFamilyExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [gamma, setGamma] = useState(0); // EL
  const [nIdx, setNIdx] = useState(0); // n = 40 (smallest, bias visible)
  const [nIdxDisplay, setNIdxDisplay] = useState(0);

  const n = N_OPTIONS[nIdx];
  const bias = useMemo(() => computeBias(n), [n]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const carrierRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([-0.4, 0.4]).range([0, innerW]);
      // y range: evaluate ρ_γ at γ = -1, 0, 1 across the v grid to pick a reasonable y window.
      let yMin = Infinity; let yMax = -Infinity;
      const ref: [number, number, number][] = [];
      for (let i = 0; i <= 200; i++) {
        const v = -0.4 + (i / 200) * 0.8;
        const r1 = rhoCressieRead(v, gamma);
        const rEL = rhoCressieRead(v, 0);
        const rCUE = rhoCressieRead(v, 1);
        if (isFinite(r1)) { yMin = Math.min(yMin, r1); yMax = Math.max(yMax, r1); }
        if (isFinite(rEL)) { yMin = Math.min(yMin, rEL); yMax = Math.max(yMax, rEL); }
        if (isFinite(rCUE)) { yMin = Math.min(yMin, rCUE); yMax = Math.max(yMax, rCUE); }
        ref.push([v, rEL, rCUE]);
      }
      const yScale = d3.scaleLinear().domain([Math.max(yMin, -1), Math.min(yMax, 0.5)]).nice().range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.2~f'))).selectAll('text').style('fill', 'var(--color-text)');

      const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1])).curve(d3.curveBasis);

      // EL reference (gray dashed).
      const elPts: [number, number][] = ref.map((p) => [p[0], p[1]]);
      g.append('path').attr('d', lineGen(elPts.filter((p) => isFinite(p[1])))).attr('fill', 'none').style('stroke', COLOR_EL).attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);
      const cuePts: [number, number][] = ref.map((p) => [p[0], p[2]]);
      g.append('path').attr('d', lineGen(cuePts.filter((p) => isFinite(p[1])))).attr('fill', 'none').style('stroke', COLOR_CUE).attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);

      // User's γ curve (solid, prominent).
      const userPts: [number, number][] = [];
      for (let i = 0; i <= 200; i++) {
        const v = -0.4 + (i / 200) * 0.8;
        const r = rhoCressieRead(v, gamma);
        if (isFinite(r)) userPts.push([v, r]);
      }
      g.append('path').attr('d', lineGen(userPts)).attr('fill', 'none').style('stroke', 'var(--color-text)').attr('stroke-width', 2.5);

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('v = λ ⋅ g(X_i, θ)');
      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`ρ_γ(v) at γ = ${gamma.toFixed(2)}`);

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 4)`);
      legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 6).attr('y2', 6).style('stroke', 'var(--color-text)').style('stroke-width', 2.5);
      legend.append('text').attr('x', 24).attr('y', 10).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(`user γ = ${gamma.toFixed(2)}`);
      legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 26).attr('y2', 26).style('stroke', COLOR_EL).style('stroke-width', 1.2).attr('stroke-dasharray', '4 3');
      legend.append('text').attr('x', 24).attr('y', 30).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_EL).text('EL (γ=0)');
      legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 46).attr('y2', 46).style('stroke', COLOR_CUE).style('stroke-width', 1.2).attr('stroke-dasharray', '4 3');
      legend.append('text').attr('x', 24).attr('y', 50).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_CUE).text('CUE (γ=1)');
    },
    [gamma, w],
  );

  const barRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = BAR_H - 16;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},6)`);

      const labels = ['two-step', 'iterated', 'CUE', 'EL'];
      const values = [bias.twoStep, bias.iterated, bias.cue, bias.el];
      const colors = [COLOR_TWO, COLOR_ITER, COLOR_CUE, COLOR_EL];
      const xScale = d3.scaleBand<string>().domain(labels).range([0, innerW]).padding(0.25);
      const yScale = d3.scaleLinear().domain([0, Math.max(...values) * 1.2]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale)).selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)').style('font-size', '11px');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');

      labels.forEach((label, i) => {
        const x = (xScale(label) ?? 0);
        g.append('rect').attr('x', x).attr('y', yScale(values[i])).attr('width', xScale.bandwidth()).attr('height', innerH - yScale(values[i])).style('fill', colors[i]).style('opacity', 0.85);
        g.append('text').attr('x', x + xScale.bandwidth() / 2).attr('y', yScale(values[i]) - 6).attr('text-anchor', 'middle').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(values[i].toFixed(3));
      });

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`mean ‖θ̂ − θ₀‖ across ${B} replicates at n = ${n}`);
    },
    [bias, w, n],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '240px' }}>
          <span>Cressie-Read γ = <strong>{gamma.toFixed(2)}</strong> <span style={{ color: 'var(--color-text-secondary)' }}>(−1 = ET, 0 = EL, 1 = CUE)</span></span>
          <input type="range" min={-1.5} max={1.5} step={0.05} value={gamma} aria-label="Cressie-Read divergence parameter" onChange={(e) => setGamma(Number(e.target.value))} />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>n = <strong>{N_OPTIONS[nIdxDisplay]}</strong> <span style={{ color: 'var(--color-text-secondary)' }}>(bias visible at small n)</span></span>
          <input type="range" min={0} max={N_OPTIONS.length - 1} step={1} value={nIdxDisplay} aria-label="Sample size" onChange={(e) => setNIdxDisplay(Number(e.target.value))} onMouseUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} onTouchEnd={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} onKeyUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} />
        </label>
      </div>
      <svg ref={carrierRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="Cressie-Read carrier ρ_γ(v)" />
      <svg ref={barRef} width={w} height={BAR_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Bias hierarchy bar chart" />
    </div>
  );
}

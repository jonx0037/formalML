import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  andersonRubinSet,
  firstStageF,
  mulberry32,
  simulateLinearIV,
  twoStageLeastSquares,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 9.3 — Weak-instrument diagnostics.
//
// Two panels driven by the first-stage strength scale τ:
//   Panel A — 2SLS sampling distribution at fixed n, B replicates. As τ → 0
//     the distribution becomes biased toward OLS and develops heavy tails;
//     the standard-error-based 95% CI under-covers. A live F-statistic bar
//     compares the mean empirical F to the Staiger-Stock threshold of 10.
//   Panel B — Anderson-Rubin confidence set as a horizontal bar. The set
//     widens dramatically as τ → 0, accurately reflecting weak-identification
//     uncertainty (versus the over-narrow 2SLS CI).
// -----------------------------------------------------------------------------

const N = 300;
const L = 4;
const RHO = 0.5;
const THETA0 = 1.0;
const B = 200;
const SY_THRESHOLD = 10;
const PANEL_H = 280;
const AR_H = 130;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 60 };

const COLOR_2SLS = '#d09a3e';
const COLOR_AR = 'var(--color-accent)';
const COLOR_THRESH = '#9aa3ad';

interface DiagState {
  thetaTsls: Vec; // length B
  meanF: number;
  arSets: { thetaGrid: Vec; accepted: boolean[]; lower: number | null; upper: number | null };
  ciTsls: { lower: number; upper: number };
}

function runDiagnostics(tau: number, seed: number): DiagState {
  const thetaTsls: Vec = [];
  let sumF = 0;
  for (let b = 0; b < B; b++) {
    const rng = mulberry32(seed + b * 11);
    const ivd = simulateLinearIV(rng, THETA0, N, L, RHO, 0, tau);
    const tsls = twoStageLeastSquares(ivd.Y, ivd.X, ivd.Z);
    thetaTsls.push(tsls.theta);
    sumF += firstStageF(ivd.X, ivd.Z);
  }
  const meanF = sumF / B;

  // AR set at one representative sample (the first replicate).
  const rngRep = mulberry32(seed);
  const ivdRep = simulateLinearIV(rngRep, THETA0, N, L, RHO, 0, tau);
  const grid: Vec = [];
  for (let g = 0; g <= 250; g++) grid.push(-1 + g * 0.02);
  const arResult = andersonRubinSet(ivdRep.Y, ivdRep.X, ivdRep.Z, grid);

  // 2SLS Wald CI (one representative).
  const tslsRep = twoStageLeastSquares(ivdRep.Y, ivdRep.X, ivdRep.Z);
  // Approximate SE from empirical std across replicates.
  let m = 0;
  for (const v of thetaTsls) m += v;
  m /= thetaTsls.length;
  let sq = 0;
  for (const v of thetaTsls) sq += (v - m) * (v - m);
  const se = Math.sqrt(sq / (thetaTsls.length - 1));
  const ciTsls = { lower: tslsRep.theta - 1.96 * se, upper: tslsRep.theta + 1.96 * se };

  return { thetaTsls, meanF, arSets: arResult, ciTsls };
}

export default function WeakInstrumentDiagnostics() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [tau, setTau] = useState(0.3);
  const [tauDisplay, setTauDisplay] = useState(0.3);
  const [seed, setSeed] = useState(50000);

  const state = useMemo(() => runDiagnostics(tau, seed), [tau, seed]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const histRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const ext = d3.extent(state.thetaTsls) as [number, number];
      const span = Math.max(ext[1] - ext[0], 0.2);
      const center = (ext[0] + ext[1]) / 2;
      const xMin = Math.min(center - span * 0.7, THETA0 - 0.3);
      const xMax = Math.max(center + span * 0.7, THETA0 + 0.3);
      const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);

      const binner = d3.bin().domain([xMin, xMax]).thresholds(30);
      const bins = binner(state.thetaTsls);
      const yMax = d3.max(bins, (b) => b.length) ?? 0;
      const yScale = d3.scaleLinear().domain([0, yMax * 1.15]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.2~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      bins.forEach((b) => {
        if (b.x0 === undefined || b.x1 === undefined) return;
        g.append('rect').attr('x', xScale(b.x0)).attr('y', yScale(b.length)).attr('width', Math.max(0, xScale(b.x1) - xScale(b.x0) - 1)).attr('height', innerH - yScale(b.length)).style('fill', COLOR_2SLS).style('opacity', 0.7);
      });

      // θ₀ line.
      g.append('line').attr('x1', xScale(THETA0)).attr('x2', xScale(THETA0)).attr('y1', 0).attr('y2', innerH).attr('stroke-dasharray', '4 3').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1.2);
      g.append('text').attr('x', xScale(THETA0) + 4).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('θ₀ = 1');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`2SLS sampling distribution at τ = ${tau.toFixed(2)} (B = ${B}, n = ${N})`);
    },
    [state, w, tau],
  );

  const arRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = AR_H - 12;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left}, 4)`);

      const xMin = Math.min(...state.arSets.thetaGrid);
      const xMax = Math.max(...state.arSets.thetaGrid);
      const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);

      g.append('g').attr('transform', `translate(0, ${innerH - 30})`).call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.2~f'))).selectAll('text').style('fill', 'var(--color-text)');

      // Background bar (all grid).
      g.append('rect').attr('x', 0).attr('y', 10).attr('width', innerW).attr('height', 14).style('fill', '#444').style('opacity', 0.3);
      // AR set bar (highlighted).
      const ar = state.arSets;
      const acceptedPoints = ar.thetaGrid.filter((_, i) => ar.accepted[i]);
      if (acceptedPoints.length > 0) {
        const minT = Math.min(...acceptedPoints);
        const maxT = Math.max(...acceptedPoints);
        g.append('rect').attr('x', xScale(minT)).attr('y', 10).attr('width', xScale(maxT) - xScale(minT)).attr('height', 14).style('fill', COLOR_AR).style('opacity', 0.7);
      }
      // 2SLS CI bar (overlay).
      g.append('rect').attr('x', xScale(state.ciTsls.lower)).attr('y', 30).attr('width', xScale(state.ciTsls.upper) - xScale(state.ciTsls.lower)).attr('height', 14).style('fill', COLOR_2SLS).style('opacity', 0.65);

      // Truth line.
      g.append('line').attr('x1', xScale(THETA0)).attr('x2', xScale(THETA0)).attr('y1', 4).attr('y2', innerH - 28).attr('stroke-dasharray', '3 2').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1.2);

      // Labels.
      g.append('text').attr('x', 0).attr('y', 22).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_AR).text('AR set');
      g.append('text').attr('x', 0).attr('y', 42).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_2SLS).text('2SLS 95% CI');
      g.append('text').attr('x', innerW / 2).attr('y', innerH).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ-grid for AR test vs Wald-style 2SLS CI');
    },
    [state, w],
  );

  const fBarColor = state.meanF >= SY_THRESHOLD ? COLOR_AR : COLOR_2SLS;

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '240px' }}>
          <span>τ = <strong>{tauDisplay.toFixed(2)}</strong> (first-stage strength)</span>
          <input type="range" min={0.05} max={1.0} step={0.025} value={tauDisplay} aria-label="First-stage strength scale" onChange={(e) => setTauDisplay(Number(e.target.value))} onMouseUp={(e) => setTau(Number((e.target as HTMLInputElement).value))} onTouchEnd={(e) => setTau(Number((e.target as HTMLInputElement).value))} onKeyUp={(e) => setTau(Number((e.target as HTMLInputElement).value))} />
        </label>
        <button type="button" onClick={() => setSeed((s) => s + 1)} style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)', fontSize: '12px', border: '1px solid var(--color-border)', background: 'var(--color-bg)', color: 'var(--color-text)', borderRadius: '4px', cursor: 'pointer', alignSelf: 'flex-end' }}>new seed</button>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', alignSelf: 'flex-end' }}>
          mean first-stage F = <strong style={{ color: fBarColor }}>{state.meanF.toFixed(1)}</strong>
          {' '}vs SY threshold = {SY_THRESHOLD}
          {state.meanF < SY_THRESHOLD ? ' — weak instruments' : ' — strong instruments'}
        </div>
      </div>
      <svg ref={histRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="2SLS sampling distribution" />
      <svg ref={arRef} width={w} height={AR_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Anderson-Rubin set vs 2SLS Wald CI" />
    </div>
  );
}

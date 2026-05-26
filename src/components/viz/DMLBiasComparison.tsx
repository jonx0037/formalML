import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  dmlPartialLinear,
  mulberry32,
  simulatePartialLinear,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 12.1 — DML vs naive plug-in vs OLS for the partial-linear model.
//
// Y = θ_0 X + g_0(W) + ε,   X = m_0(W) + v.
// g_0(w) = sin(πw), m_0(w) = w² — smooth nonlinear nuisances.
//
// Three estimators:
//   • OLS: regress Y on X. Biased: doesn't adjust for W.
//   • Naive plug-in: fit g, m on whole sample, residualize, OLS slope.
//     Biased at small/moderate n because the same data is used for nuisance
//     estimation and for the structural estimator (Chernozhukov et al. 2018).
//   • DML: K-fold cross-fitting. Fit g, m on one fold, evaluate on another.
//     Bias from nuisance estimation is decorrelated; the structural estimator
//     achieves the semiparametric efficiency bound at √n rate even with
//     ML-estimated nuisances at slower than √n rate.
//
// Nuisances are degree-4 polynomial regressions in W (the brief uses random
// forests; we use polynomial in the browser to keep MC cheap). The
// qualitative bias-hierarchy story (DML > naive > OLS in bias-control) is
// the same.
// -----------------------------------------------------------------------------

const N_OPTIONS = [100, 200, 400, 800];
const B = 200;
const K_FOLDS = 5;
const POLY_DEG = 4;
const TRUTH = 1.0;
const PANEL_H = 280;
const HIST_H = 260;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 60 };

const COLOR_OLS = '#9aa3ad';
const COLOR_NAIVE = '#d09a3e';
const COLOR_DML = 'var(--color-accent)';

interface Replicates { ols: Vec; naive: Vec; dml: Vec; }

function runReplicates(n: number): Replicates {
  const ols: Vec = []; const naive: Vec = []; const dml: Vec = [];
  for (let b = 0; b < B; b++) {
    const rng = mulberry32(130000 + b * 13);
    const data = simulatePartialLinear(rng, TRUTH, n);
    const r = dmlPartialLinear(data.Y, data.X, data.W, TRUTH, K_FOLDS, POLY_DEG);
    ols.push(r.thetaOLS);
    naive.push(r.thetaNaive);
    dml.push(r.thetaDML);
  }
  return { ols, naive, dml };
}

function sampleMean(v: Vec): number {
  let s = 0;
  for (const x of v) s += x;
  return s / v.length;
}

function sampleStd(v: Vec): number {
  const m = sampleMean(v);
  let s = 0;
  for (const x of v) s += (x - m) * (x - m);
  return Math.sqrt(s / Math.max(v.length - 1, 1));
}

export default function DMLBiasComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(1); // n = 200
  const [nIdxDisplay, setNIdxDisplay] = useState(1);

  const n = N_OPTIONS[nIdx];
  const data = useMemo(() => runReplicates(n), [n]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const barRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const labels = ['OLS', 'naive plug-in', 'DML'];
      const biases = [Math.abs(sampleMean(data.ols) - TRUTH), Math.abs(sampleMean(data.naive) - TRUTH), Math.abs(sampleMean(data.dml) - TRUTH)];
      const stds = [sampleStd(data.ols), sampleStd(data.naive), sampleStd(data.dml)];
      const colors = [COLOR_OLS, COLOR_NAIVE, COLOR_DML];
      const yMax = Math.max(...biases, ...stds) * 1.2;
      const xScale = d3.scaleBand<string>().domain(labels).range([0, innerW]).padding(0.3);
      const subScale = d3.scaleBand<string>().domain(['|bias|', 'SD']).range([0, xScale.bandwidth()]).padding(0.1);
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale)).selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)').style('font-size', '11px');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');

      labels.forEach((label, i) => {
        const gx = xScale(label) ?? 0;
        // bias bar
        const bx = gx + (subScale('|bias|') ?? 0);
        g.append('rect').attr('x', bx).attr('y', yScale(biases[i])).attr('width', subScale.bandwidth()).attr('height', innerH - yScale(biases[i])).style('fill', colors[i]).style('opacity', 0.85);
        g.append('text').attr('x', bx + subScale.bandwidth() / 2).attr('y', yScale(biases[i]) - 6).attr('text-anchor', 'middle').style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', 'var(--color-text)').text(biases[i].toFixed(3));
        // SD bar (faded)
        const sx = gx + (subScale('SD') ?? 0);
        g.append('rect').attr('x', sx).attr('y', yScale(stds[i])).attr('width', subScale.bandwidth()).attr('height', innerH - yScale(stds[i])).style('fill', colors[i]).style('opacity', 0.4);
        g.append('text').attr('x', sx + subScale.bandwidth() / 2).attr('y', yScale(stds[i]) - 6).attr('text-anchor', 'middle').style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', 'var(--color-text)').text(stds[i].toFixed(3));
      });

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 130}, 4)`);
      legend.append('rect').attr('width', 14).attr('height', 14).style('fill', 'var(--color-text)').style('opacity', 0.85);
      legend.append('text').attr('x', 20).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('|bias|');
      legend.append('rect').attr('x', 0).attr('y', 22).attr('width', 14).attr('height', 14).style('fill', 'var(--color-text)').style('opacity', 0.4);
      legend.append('text').attr('x', 20).attr('y', 34).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('SD');
    },
    [data, w],
  );

  const histRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HIST_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const combined = data.ols.concat(data.naive).concat(data.dml);
      const ext = d3.extent(combined) as [number, number];
      const pad = (ext[1] - ext[0]) * 0.1;
      const xMin = Math.min(ext[0] - pad, TRUTH - 0.3);
      const xMax = Math.max(ext[1] + pad, TRUTH + 0.3);
      const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
      const binner = d3.bin().domain([xMin, xMax]).thresholds(25);
      const binsO = binner(data.ols); const binsN = binner(data.naive); const binsD = binner(data.dml);
      const yMax = Math.max(d3.max(binsO, (b) => b.length) ?? 0, d3.max(binsN, (b) => b.length) ?? 0, d3.max(binsD, (b) => b.length) ?? 0);
      const yScale = d3.scaleLinear().domain([0, yMax * 1.15]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      const renderBins = (bins: d3.Bin<number, number>[], color: string, opacity: number) => {
        for (const b of bins) {
          if (b.x0 === undefined || b.x1 === undefined) continue;
          g.append('rect').attr('x', xScale(b.x0)).attr('y', yScale(b.length)).attr('width', Math.max(0, xScale(b.x1) - xScale(b.x0) - 1)).attr('height', innerH - yScale(b.length)).style('fill', color).style('opacity', opacity);
        }
      };
      renderBins(binsO, COLOR_OLS, 0.55);
      renderBins(binsN, COLOR_NAIVE, 0.55);
      renderBins(binsD, COLOR_DML, 0.5);

      // Truth.
      g.append('line').attr('x1', xScale(TRUTH)).attr('x2', xScale(TRUTH)).attr('y1', 0).attr('y2', innerH).attr('stroke-dasharray', '4 3').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1.2);
      g.append('text').attr('x', xScale(TRUTH) + 4).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text(`θ₀ = ${TRUTH}`);

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 240}, 4)`);
      [['OLS', COLOR_OLS], ['naive plug-in', COLOR_NAIVE], ['DML (K=5)', COLOR_DML]].forEach(([label, color], i) => {
        legend.append('rect').attr('x', i * 80).attr('width', 12).attr('height', 12).style('fill', color as string).style('opacity', 0.55);
        legend.append('text').attr('x', i * 80 + 18).attr('y', 10).style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', 'var(--color-text)').text(label as string);
      });

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`θ̂ across ${B} replicates at n = ${n}, K=${K_FOLDS}, polynomial deg ${POLY_DEG} nuisances`);
    },
    [data, w, n],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>n = <strong>{N_OPTIONS[nIdxDisplay]}</strong></span>
          <input type="range" min={0} max={N_OPTIONS.length - 1} step={1} value={nIdxDisplay} aria-label="Sample size" onChange={(e) => setNIdxDisplay(Number(e.target.value))} onMouseUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} onTouchEnd={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} onKeyUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', alignSelf: 'flex-end' }}>
          OLS bias = <strong style={{ color: COLOR_OLS }}>{Math.abs(sampleMean(data.ols) - TRUTH).toFixed(3)}</strong> ·
          naive = <strong style={{ color: COLOR_NAIVE }}>{Math.abs(sampleMean(data.naive) - TRUTH).toFixed(3)}</strong> ·
          DML = <strong style={{ color: COLOR_DML }}>{Math.abs(sampleMean(data.dml) - TRUTH).toFixed(3)}</strong>
        </div>
      </div>
      <svg ref={barRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="DML, naive plug-in, OLS bias and SD bars" />
      <svg ref={histRef} width={w} height={HIST_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="DML, naive plug-in, OLS sampling distributions" />
    </div>
  );
}

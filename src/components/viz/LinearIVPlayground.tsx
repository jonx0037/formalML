import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  efficientGMM_IV,
  mulberry32,
  olsScalar,
  simulateLinearIV,
  twoStageLeastSquares,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 9.1 — Linear IV playground.
//
// Three sampling distributions overlaid on the same axis:
//   • OLS (biased under endogeneity)
//   • 2SLS (consistent under exogenous instruments; inefficient under
//     heteroskedasticity)
//   • Efficient GMM (consistent + heteroskedasticity-robust efficient)
//
// Reader controls endogeneity ρ ∈ [0, 0.9], heteroskedasticity γ ∈ [0, 4],
// and sample size n ∈ {100, 200, 500, 1000}. First-stage strength is fixed
// at τ = 1.0 (strong instruments) to isolate the endogeneity/het effects.
//
// MC cost: B=300 replicates × 3 estimators × n ≈ 0.6M flops. Commit-on-
// release for n.
// -----------------------------------------------------------------------------

const N_OPTIONS = [100, 200, 500, 1000];
const B_REPLICATES = 300;
const L = 4;
const TAU = 1.0;
const THETA0 = 1.0;
const PANEL_H = 320;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 56 };

const COLOR_OLS = '#9aa3ad';
const COLOR_2SLS = '#d09a3e';
const COLOR_EFF = 'var(--color-accent)';

interface Replicates {
  ols: Vec;
  tsls: Vec;
  eff: Vec;
}

function computeReplicates(n: number, rho: number, gamma: number): Replicates {
  const ols: Vec = [];
  const tsls: Vec = [];
  const eff: Vec = [];
  for (let b = 0; b < B_REPLICATES; b++) {
    const rng = mulberry32(20000 + b * 19);
    const ivd = simulateLinearIV(rng, THETA0, n, L, rho, gamma, TAU);
    ols.push(olsScalar(ivd.Y, ivd.X).theta);
    tsls.push(twoStageLeastSquares(ivd.Y, ivd.X, ivd.Z).theta);
    eff.push(efficientGMM_IV(ivd.Y, ivd.X, ivd.Z).theta);
  }
  return { ols, tsls, eff };
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

export default function LinearIVPlayground() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(1); // n = 200
  const [nIdxDisplay, setNIdxDisplay] = useState(1);
  const [rho, setRho] = useState(0.5);
  const [gamma, setGamma] = useState(1.5);

  const n = N_OPTIONS[nIdx];

  const data = useMemo(() => computeReplicates(n, rho, gamma), [n, rho, gamma]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const combined = data.ols.concat(data.tsls).concat(data.eff);
      const ext = d3.extent(combined) as [number, number];
      const pad = (ext[1] - ext[0]) * 0.1;
      const xMin = Math.min(ext[0] - pad, THETA0 - 0.5);
      const xMax = Math.max(ext[1] + pad, THETA0 + 0.5);
      const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);

      const binner = d3.bin().domain([xMin, xMax]).thresholds(28);
      const binsO = binner(data.ols);
      const binsT = binner(data.tsls);
      const binsE = binner(data.eff);
      const yMax = Math.max(
        d3.max(binsO, (b) => b.length) ?? 0,
        d3.max(binsT, (b) => b.length) ?? 0,
        d3.max(binsE, (b) => b.length) ?? 0,
      );
      const yScale = d3.scaleLinear().domain([0, yMax * 1.1]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      const renderBins = (bins: d3.Bin<number, number>[], color: string, opacity: number) => {
        for (const b of bins) {
          if (b.x0 === undefined || b.x1 === undefined) continue;
          g.append('rect').attr('x', xScale(b.x0)).attr('y', yScale(b.length)).attr('width', Math.max(0, xScale(b.x1) - xScale(b.x0) - 1)).attr('height', innerH - yScale(b.length)).style('fill', color).style('opacity', opacity);
        }
      };
      renderBins(binsO, COLOR_OLS, 0.55);
      renderBins(binsT, COLOR_2SLS, 0.55);
      renderBins(binsE, COLOR_EFF, 0.45);

      // θ₀ vertical line.
      g.append('line').attr('x1', xScale(THETA0)).attr('x2', xScale(THETA0)).attr('y1', 0).attr('y2', innerH).attr('stroke-dasharray', '4 3').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1.2);
      g.append('text').attr('x', xScale(THETA0) + 4).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('θ₀ = 1');

      // Mean markers.
      const meanMarker = (m: number, color: string, label: string, dy: number) => {
        g.append('line').attr('x1', xScale(m)).attr('x2', xScale(m)).attr('y1', innerH - 10 - dy).attr('y2', innerH).style('stroke', color).style('stroke-width', 2);
        g.append('text').attr('x', xScale(m) + 4).attr('y', innerH - 14 - dy).style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', color).text(label);
      };
      meanMarker(sampleMean(data.ols), COLOR_OLS, 'OLS', 0);
      meanMarker(sampleMean(data.tsls), COLOR_2SLS, '2SLS', 14);
      meanMarker(sampleMean(data.eff), COLOR_EFF, 'eff', 28);

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 220}, 4)`);
      [['OLS', COLOR_OLS, 0.55], ['2SLS', COLOR_2SLS, 0.55], ['Efficient GMM', COLOR_EFF, 0.55]].forEach(([label, color, op], i) => {
        legend.append('rect').attr('x', i * 70).attr('width', 12).attr('height', 12).style('fill', color as string).style('opacity', op as number);
        legend.append('text').attr('x', i * 70 + 18).attr('y', 10).style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', 'var(--color-text)').text(label as string);
      });

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`θ̂ across ${B_REPLICATES} replicates (n = ${n}, τ = ${TAU})`);
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
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>ρ = <strong>{rho.toFixed(2)}</strong> (endogeneity)</span>
          <input type="range" min={0} max={0.9} step={0.05} value={rho} aria-label="Endogeneity correlation" onChange={(e) => setRho(Number(e.target.value))} />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>γ = <strong>{gamma.toFixed(2)}</strong> (heteroskedasticity)</span>
          <input type="range" min={0} max={4} step={0.1} value={gamma} aria-label="Heteroskedasticity scale" onChange={(e) => setGamma(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', alignSelf: 'flex-end' }}>
          SD: OLS={sampleStd(data.ols).toFixed(3)} · 2SLS={sampleStd(data.tsls).toFixed(3)} · eff={sampleStd(data.eff).toFixed(3)}
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="OLS, 2SLS, efficient GMM sampling distributions" />
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  RUNNING_EXAMPLE_A,
  RUNNING_EXAMPLE_THETA0,
  confidenceEllipse2D,
  efficientWeight,
  gmmEstimateAffine,
  identityMat,
  mulberry32,
  sampleSensorObservations,
  sandwichVariance,
  type Mat,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 4.1 — Consistency of GMM estimators.
//
// Two panels driven by Monte Carlo over B replicates:
//   Panel A — log-log RMSE vs sample size for identity weighting and efficient
//     (oracle-Ω⁻¹) weighting. Both lines should follow the parametric n^{-1/2}
//     slope; the efficient line sits below the identity line.
//   Panel B — scatter cloud of θ̂ estimates at a fixed n=200, with the
//     analytical 95% confidence ellipse from the sandwich variance overlaid.
//
// Compute cost: 6 n-values × B replicates × (n samples + 2 GLS solves). At
// B=300, B · sum(n) ≈ 4.8M scalar samples per refresh. Commit-on-release.
// -----------------------------------------------------------------------------

const N_GRID = [25, 50, 100, 200, 400, 800];
const N_SCATTER = 200;
const B_REPLICATES = 300;
const PANEL_H = 280;
const SCATTER_H = 280;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 56 };

const COLOR_IDENT = '#9aa3ad';
const COLOR_EFF = 'var(--color-accent)';

interface Run {
  rmseIdent: Vec; // length N_GRID.length
  rmseEff: Vec;
  scatterIdent: Vec[]; // [θ̂_1, θ̂_2] pairs
  scatterEff: Vec[];
  ellipseIdent: { rx: number; ry: number; angle: number };
  ellipseEff: { rx: number; ry: number; angle: number };
}

function buildOmega(sigma3: number): Mat {
  const sigma = [1, 1, sigma3, 1];
  const Omega: Mat = identityMat(4);
  for (let i = 0; i < 4; i++) Omega[i][i] = sigma[i] * sigma[i];
  return Omega;
}

function computeRun(sigma3: number): Run {
  const A = RUNNING_EXAMPLE_A;
  const theta0 = RUNNING_EXAMPLE_THETA0;
  const sigma = [1, 1, sigma3, 1];
  const Omega = buildOmega(sigma3);
  const Wident = identityMat(4);
  const Weff = efficientWeight(Omega);

  const rmseIdent: Vec = [];
  const rmseEff: Vec = [];
  let scatterIdent: Vec[] = [];
  let scatterEff: Vec[] = [];

  for (const n of N_GRID) {
    let sumSqIdent = 0;
    let sumSqEff = 0;
    const cloudI: Vec[] = [];
    const cloudE: Vec[] = [];
    for (let b = 0; b < B_REPLICATES; b++) {
      const rng = mulberry32(7000 + b * 37 + n);
      const X = sampleSensorObservations(rng, theta0, A, n, sigma);
      const Xbar = [0, 0, 0, 0];
      for (let i = 0; i < n; i++) for (let j = 0; j < 4; j++) Xbar[j] += X[i][j];
      for (let j = 0; j < 4; j++) Xbar[j] /= n;
      const tI = gmmEstimateAffine(A, Xbar, Wident);
      const tE = gmmEstimateAffine(A, Xbar, Weff);
      sumSqIdent += (tI[0] - theta0[0]) ** 2 + (tI[1] - theta0[1]) ** 2;
      sumSqEff += (tE[0] - theta0[0]) ** 2 + (tE[1] - theta0[1]) ** 2;
      if (n === N_SCATTER) {
        cloudI.push(tI);
        cloudE.push(tE);
      }
    }
    rmseIdent.push(Math.sqrt(sumSqIdent / B_REPLICATES));
    rmseEff.push(Math.sqrt(sumSqEff / B_REPLICATES));
    if (n === N_SCATTER) {
      scatterIdent = cloudI;
      scatterEff = cloudE;
    }
  }

  // Theoretical 95% confidence ellipses at n=N_SCATTER.
  const G: Mat = A.map((row) => row.map((v) => -v));
  const V_I = sandwichVariance(G, Wident, Omega);
  const V_E = sandwichVariance(G, Weff, Omega);
  // Scale variances by 1/n to get the variance of θ̂_W (rather than √n(θ̂-θ₀)).
  const V_I_scaled: Mat = V_I.map((r) => r.map((v) => v / N_SCATTER));
  const V_E_scaled: Mat = V_E.map((r) => r.map((v) => v / N_SCATTER));
  const ellipseIdent = confidenceEllipse2D(V_I_scaled, 0.05);
  const ellipseEff = confidenceEllipse2D(V_E_scaled, 0.05);

  return { rmseIdent, rmseEff, scatterIdent, scatterEff, ellipseIdent, ellipseEff };
}

export default function ConsistencyCloudExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sigma3, setSigma3] = useState(2.0);
  const [sigma3Display, setSigma3Display] = useState(2.0);

  const run = useMemo(() => computeRun(sigma3), [sigma3]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const rmseRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([N_GRID[0] * 0.8, N_GRID[N_GRID.length - 1] * 1.2]).range([0, innerW]);
      const allRmse = run.rmseIdent.concat(run.rmseEff);
      const yScale = d3.scaleLog().domain([d3.min(allRmse)! * 0.5, d3.max(allRmse)! * 1.5]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(6, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '~e')).selectAll('text').style('fill', 'var(--color-text)');

      // n^{-1/2} reference line anchored at largest n's RMSE for identity.
      const ref: [number, number][] = N_GRID.map((n) => [n, run.rmseIdent[N_GRID.length - 1] * Math.sqrt(N_GRID[N_GRID.length - 1] / n)]);
      const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1]));
      g.append('path').attr('d', lineGen(ref))
        .attr('fill', 'none')
        .attr('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '4 3')
        .attr('stroke-width', 1);
      g.append('text').attr('x', xScale(N_GRID[1])).attr('y', yScale(ref[1][1]) - 6).style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('n^{−1/2} reference');

      // Lines.
      const identPts: [number, number][] = N_GRID.map((n, i) => [n, run.rmseIdent[i]]);
      const effPts: [number, number][] = N_GRID.map((n, i) => [n, run.rmseEff[i]]);
      g.append('path').attr('d', lineGen(identPts)).attr('fill', 'none').attr('stroke', COLOR_IDENT).attr('stroke-width', 2);
      g.append('path').attr('d', lineGen(effPts)).attr('fill', 'none').attr('stroke', COLOR_EFF).attr('stroke-width', 2);

      // Markers.
      identPts.forEach((p) => g.append('circle').attr('cx', xScale(p[0])).attr('cy', yScale(p[1])).attr('r', 3.5).style('fill', COLOR_IDENT));
      effPts.forEach((p) => g.append('circle').attr('cx', xScale(p[0])).attr('cy', yScale(p[1])).attr('r', 3.5).style('fill', COLOR_EFF));

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 4)`);
      legend.append('rect').attr('width', 14).attr('height', 4).attr('y', 4).style('fill', COLOR_IDENT);
      legend.append('text').attr('x', 20).attr('y', 11).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('W = I (identity)');
      legend.append('rect').attr('x', 0).attr('y', 22).attr('width', 14).attr('height', 4).style('fill', COLOR_EFF);
      legend.append('text').attr('x', 20).attr('y', 29).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('W = Ω⁻¹ (efficient)');

      // Axis labels.
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('sample size n (log)');
      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('RMSE‖θ̂ − θ₀‖ (log)');
    },
    [run, w],
  );

  const scatterRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = SCATTER_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const allX = run.scatterIdent.concat(run.scatterEff).map((p) => p[0]);
      const allY = run.scatterIdent.concat(run.scatterEff).map((p) => p[1]);
      const xExt = d3.extent(allX) as [number, number];
      const yExt = d3.extent(allY) as [number, number];
      const padX = (xExt[1] - xExt[0]) * 0.1;
      const padY = (yExt[1] - yExt[0]) * 0.1;
      const xScale = d3.scaleLinear().domain([xExt[0] - padX, xExt[1] + padX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([yExt[0] - padY, yExt[1] + padY]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');

      // Plot identity-weighted points.
      run.scatterIdent.forEach((p) => g.append('circle').attr('cx', xScale(p[0])).attr('cy', yScale(p[1])).attr('r', 2.2).style('fill', COLOR_IDENT).attr('opacity', 0.55));
      run.scatterEff.forEach((p) => g.append('circle').attr('cx', xScale(p[0])).attr('cy', yScale(p[1])).attr('r', 2.2).style('fill', COLOR_EFF).attr('opacity', 0.55));

      // 95% confidence ellipses.
      const x0 = xScale(RUNNING_EXAMPLE_THETA0[0]);
      const y0 = yScale(RUNNING_EXAMPLE_THETA0[1]);
      const drawEllipse = (ell: { rx: number; ry: number; angle: number }, color: string) => {
        const rx = Math.abs(xScale(RUNNING_EXAMPLE_THETA0[0] + ell.rx) - xScale(RUNNING_EXAMPLE_THETA0[0]));
        const ry = Math.abs(yScale(RUNNING_EXAMPLE_THETA0[1] + ell.ry) - yScale(RUNNING_EXAMPLE_THETA0[1]));
        const angleDeg = (-ell.angle * 180) / Math.PI; // negate because y-axis is inverted
        g.append('ellipse')
          .attr('cx', x0).attr('cy', y0)
          .attr('rx', rx).attr('ry', ry)
          .attr('transform', `rotate(${angleDeg}, ${x0}, ${y0})`)
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 2);
      };
      drawEllipse(run.ellipseIdent, COLOR_IDENT);
      drawEllipse(run.ellipseEff, COLOR_EFF);

      // θ₀ cross.
      g.append('line').attr('x1', x0 - 7).attr('x2', x0 + 7).attr('y1', y0).attr('y2', y0).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);
      g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', y0 - 7).attr('y2', y0 + 7).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`θ̂_1 at n = ${N_SCATTER}, B = ${B_REPLICATES}`);
      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ̂_2');
    },
    [run, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '220px' }}>
          <span>
            σ_3 = <strong>{sigma3Display.toFixed(2)}</strong>
            <span style={{ color: 'var(--color-text-secondary)', marginLeft: 6 }}>
              (sensor-3 noise; larger ⇒ wider efficient-vs-identity gap)
            </span>
          </span>
          <input
            type="range"
            min={0.5}
            max={4.0}
            step={0.1}
            value={sigma3Display}
            aria-label="Sensor 3 noise level"
            onChange={(e) => setSigma3Display(Number(e.target.value))}
            onMouseUp={(e) => setSigma3(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setSigma3(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setSigma3(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', alignSelf: 'flex-end' }}>
          RMSE ratio at n = {N_GRID[N_GRID.length - 1]}: identity / efficient = <strong style={{ color: 'var(--color-text)' }}>{(run.rmseIdent[N_GRID.length - 1] / run.rmseEff[N_GRID.length - 1]).toFixed(2)}×</strong>
        </div>
      </div>
      <svg ref={rmseRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="RMSE vs sample size, log-log" />
      <svg ref={scatterRef} width={w} height={SCATTER_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label={`Sampling cloud at n = ${N_SCATTER}`} />
    </div>
  );
}

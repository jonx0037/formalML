import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  generateDgp1,
  lassoIsta,
  operatorNorm,
  predictMse,
} from './shared/high-dim-regression';

// =============================================================================
// OracleInequalityRiskBound — interactive companion to §5.5 Figure 5.1.
// Empirical lasso prediction risk ‖X_test (β̂_lasso − β*)‖² / n_test vs sample
// size n on smaller-scale DGP-1 (p = 200, s = 10, σ = 0.5 fixed; n varies),
// at the theory-guided λ = 2σ√(2 log(p)/n).
//
// Three curves on log-log axes:
//   - Empirical risk (single MC rep per n; shown as dots + connecting line)
//   - BRT bound: 72 σ² s log(2p) / (n κ²)  with κ² = 1/3 (DGP-1 ρ = 0.5 limit)
//   - Calibrated bound: c · σ² s log(p) / n  with c fit to the largest-n point
//
// Numerical anchor: empirical risk slope -1 on log-log; BRT bound parallel
// (also slope -1) but ~10-100x higher (constant 72 is loose, the Δ between
// proof-clean and practically-tight constants).
//
// Static fallback: public/images/topics/high-dimensional-regression/fig_05_01_oracle_inequality_rate.png
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 16, bottom: 50, left: 60 };

const P = 200;
const S = 10;
const SIGMA = 0.5;
const RHO = 0.5;
const N_VALUES = [50, 80, 120, 200, 320, 500, 800];
const ISTA_ITERS = 300;
const KAPPA2_RE = 1 / 3; // RE constant for AR(1) Toeplitz with ρ = 0.5 (limit p → ∞)
const BRT_CONSTANT = 72;

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const SLATE = '#6B6B6B';

interface CurvePoint {
  n: number;
  empirical: number;
  brtBound: number;
  calibratedBound: number;
}

// -----------------------------------------------------------------------------
// Compute risk + bounds at each n.
// -----------------------------------------------------------------------------

function computeOracleCurve(): CurvePoint[] {
  const test = generateDgp1({ n: 200, p: P, s: S, sigma: SIGMA, rho: RHO, seed: 999 });
  const empiricals: { n: number; empirical: number }[] = [];
  for (const n of N_VALUES) {
    const train = generateDgp1({ n, p: P, s: S, sigma: SIGMA, rho: RHO, seed: 100 + n });
    const L = operatorNorm(train.X);
    const lambdaTheory = 2 * SIGMA * Math.sqrt((2 * Math.log(P)) / n);
    const beta = lassoIsta(train.X, train.y, lambdaTheory, L, ISTA_ITERS);
    // Prediction risk on test set, against true β*.
    let sse = 0;
    const nTest = test.X.length;
    for (let i = 0; i < nTest; i++) {
      let pred = 0;
      let truePred = 0;
      for (let j = 0; j < P; j++) {
        pred += test.X[i][j] * beta[j];
        truePred += test.X[i][j] * test.betaStar[j];
      }
      const r = pred - truePred;
      sse += r * r;
    }
    empiricals.push({ n, empirical: sse / nTest });
  }
  // Calibrate the constant c such that the bound at the largest n matches empirical:
  //   c · σ² s log(p) / n_max = empirical(n_max)
  const nMax = N_VALUES[N_VALUES.length - 1];
  const empMax = empiricals[empiricals.length - 1].empirical;
  const calibrationConstant = (empMax * nMax) / (SIGMA * SIGMA * S * Math.log(P));
  // Build the full curve.
  return empiricals.map(({ n, empirical }) => ({
    n,
    empirical,
    brtBound: (BRT_CONSTANT * SIGMA * SIGMA * S * Math.log(2 * P)) / (n * KAPPA2_RE),
    calibratedBound: (calibrationConstant * SIGMA * SIGMA * S * Math.log(P)) / n,
  }));
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function OracleInequalityRiskBound() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const curve = useMemo(() => computeOracleCurve(), []);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([N_VALUES[0] * 0.85, N_VALUES[N_VALUES.length - 1] * 1.15]).range([0, innerW]);
      const yMin = Math.min(...curve.map((c) => c.empirical)) * 0.5;
      const yMax = Math.max(...curve.map((c) => c.brtBound)) * 1.3;
      const yScale = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]).clamp(true);

      const xAxis = d3.axisBottom(xScale).ticks(isMobile ? 4 : 6, '~g').tickSize(-innerH);
      const yAxis = d3.axisLeft(yScale).ticks(isMobile ? 4 : 6, '~g').tickSize(-innerW);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis).call((sel) => {
        sel.selectAll('line').style('stroke', 'var(--color-border)');
        sel.selectAll('path').style('stroke', 'var(--color-text-secondary)');
        sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)');
      });
      g.append('g').call(yAxis).call((sel) => {
        sel.selectAll('line').style('stroke', 'var(--color-border)');
        sel.selectAll('path').style('stroke', 'var(--color-text-secondary)');
        sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)');
      });

      g.append('text').attr('text-anchor', 'middle').attr('x', innerW / 2).attr('y', innerH + 38).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '13px').text('sample size n (log scale)');
      g.append('text').attr('text-anchor', 'middle').attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '13px').text('prediction risk ‖X(β̂ − β*)‖²/n (log scale)');

      const draw = (
        accessor: (c: CurvePoint) => number,
        color: string,
        dash: string | null,
        showDots: boolean,
        label: string,
        legendIdx: number,
      ) => {
        const lineGen = d3.line<CurvePoint>().x((c) => xScale(c.n)).y((c) => yScale(Math.max(yMin, accessor(c))));
        const path = g.append('path').datum(curve).attr('d', lineGen).style('fill', 'none').style('stroke', color).style('stroke-width', 2);
        if (dash) path.style('stroke-dasharray', dash);
        if (showDots) {
          g.selectAll(`.dot-${legendIdx}`).data(curve).enter().append('circle')
            .attr('class', `dot-${legendIdx}`)
            .attr('cx', (c) => xScale(c.n))
            .attr('cy', (c) => yScale(Math.max(yMin, accessor(c))))
            .attr('r', 4)
            .style('fill', color)
            .style('stroke', 'var(--color-bg)')
            .style('stroke-width', 1);
        }
        const legend = g.append('g').attr('transform', `translate(${innerW - 200},${4 + legendIdx * 18})`);
        const leg = legend.append('rect').attr('width', 14).attr('height', 4).attr('y', 6).style('fill', color);
        if (dash) leg.style('stroke', color).style('stroke-dasharray', dash);
        legend.append('text').attr('x', 20).attr('y', 11).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').text(label);
      };
      draw((c) => c.empirical, TEAL, null, true, 'empirical risk', 0);
      draw((c) => c.brtBound, AMBER, '5 3', false, 'BRT bound (c = 72)', 1);
      draw((c) => c.calibratedBound, PURPLE, '2 2', false, 'calibrated bound', 2);
    },
    [curve, w, isMobile],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={renderRef} width={w} height={HEIGHT} role="img" aria-label="Lasso oracle inequality rate visualization on log-log axes" />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        Lasso prediction risk on smaller-scale DGP-1 (p = {P}, s = {S}, σ = {SIGMA}, AR(1) ρ = {RHO}, λ = 2σ√(2 log(p)/n) per Theorem 1) as n varies from {N_VALUES[0]} to {N_VALUES[N_VALUES.length - 1]}. Empirical (teal dots) sits one to two orders of magnitude below the BRT bound (amber, c = {BRT_CONSTANT}); the calibrated bound (purple, constant fit to the n = {N_VALUES[N_VALUES.length - 1]} point) sits right on top of the empirical curve. All three lines have the same -1 slope on log-log axes — the substantive confirmation of the σ²s log(p)/n rate. The constant 72 is mathematically clean (each proof step contributes a factor of 2 or 3) but practically loose by 10-100×. Computed live in-browser via single-rep ISTA at each n (~500 ms total).
      </p>
    </div>
  );
}

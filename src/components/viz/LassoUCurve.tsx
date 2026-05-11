import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  generateDgp1,
  lassoIsta,
  operatorNorm,
  xMul,
} from './shared/high-dim-regression';

// =============================================================================
// LassoUCurve — interactive companion to §4.3 Figure 4.1.
// Empirical bias², variance, and total prediction MSE on a held-out test set
// as a function of λ on a smaller-scale DGP-1 (p = 200 instead of 500 for
// in-browser tractability), computed by Monte Carlo over B = 20 replicate
// draws of DGP-1.
//
// Bias-variance decomposition at each test point x*_i:
//   Pred^(b)(x*_i) = x*_i^T β̂^(b)(λ)
//   Bias(x*_i, λ)  = E_b[Pred^(b)(x*_i)] - x*_i^T β*
//   Var(x*_i, λ)   = Var_b(Pred^(b)(x*_i))
//   MSE(x*_i, λ)   = Bias(x*_i, λ)^2 + Var(x*_i, λ)
//
// Aggregated by averaging over test points: IBE(λ), IVar(λ), IMSE(λ).
//
// Static fallback: public/images/topics/high-dimensional-regression/fig_04_01_u_curve.png
// Numerical anchor: U-curve minimum near λ ≈ 0.05-0.07 (CV-selected range).
// Compute budget: ~3-5 seconds (B × |λ-grid| × ISTA-with-warm-starts).
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 16, bottom: 50, left: 60 };

const N = 200;
const P = 200; // smaller than DGP-1's 500 to keep client-side compute under 5s
const S = 10;
const SIGMA = 0.5;
const RHO = 0.5;
const B = 20;
const ISTA_ITERS_FIRST = 200; // first lambda gets cold start
const ISTA_ITERS_WARM = 50; // subsequent get warm-started from larger lambda
const N_LAMBDA = 25;
const LAMBDA_MIN = 0.001;
const LAMBDA_MAX_CAP = 1.0;

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const RED = '#B91C1C';
const SLATE = '#6B6B6B';

interface CurvePoint {
  lambda: number;
  bias2: number;
  variance: number;
  mse: number;
}

function logspace(lo: number, hi: number, n: number): number[] {
  const lLo = Math.log(lo);
  const lHi = Math.log(hi);
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(Math.exp(lLo + ((lHi - lLo) * i) / (n - 1)));
  return out;
}

// -----------------------------------------------------------------------------
// Compute bias-variance decomposition.
// -----------------------------------------------------------------------------

function computeUCurve(): CurvePoint[] {
  const opts = { n: N, p: P, s: S, sigma: SIGMA, rho: RHO, seed: 0 };
  const test = generateDgp1({ ...opts, seed: 999 });
  // True predictions on test set: x*_i^T β*
  const truePreds = xMul(test.X, test.betaStar);

  // Fit B replicates at all lambdas (largest first for warm starts).
  const lambdas = logspace(LAMBDA_MIN, LAMBDA_MAX_CAP, N_LAMBDA).reverse(); // descending
  // predsMatrix[b][lambdaIdx] = Float64Array of length n_test (predictions on test set)
  const predsMatrix: Float64Array[][] = [];

  for (let b = 0; b < B; b++) {
    const train = generateDgp1({ ...opts, seed: 100 + b });
    const L = operatorNorm(train.X);
    const replicate: Float64Array[] = [];
    let prevBeta: Float64Array | undefined = undefined;
    for (let li = 0; li < lambdas.length; li++) {
      const iters = li === 0 ? ISTA_ITERS_FIRST : ISTA_ITERS_WARM;
      const beta = lassoIsta(train.X, train.y, lambdas[li], L, iters, prevBeta);
      // Predict on test set.
      const preds = xMul(test.X, beta);
      replicate.push(preds);
      prevBeta = beta;
    }
    predsMatrix.push(replicate);
  }

  // Aggregate bias-variance at each lambda (re-reverse so output is ascending λ).
  const out: CurvePoint[] = [];
  for (let li = lambdas.length - 1; li >= 0; li--) {
    const nTest = test.X.length;
    let totalBias2 = 0;
    let totalVar = 0;
    for (let i = 0; i < nTest; i++) {
      // Average prediction across replicates at test point i.
      let avg = 0;
      for (let b = 0; b < B; b++) avg += predsMatrix[b][li][i];
      avg /= B;
      // Bias² at test point i.
      const bias = avg - truePreds[i];
      totalBias2 += bias * bias;
      // Variance at test point i.
      let v = 0;
      for (let b = 0; b < B; b++) {
        const d = predsMatrix[b][li][i] - avg;
        v += d * d;
      }
      v /= B;
      totalVar += v;
    }
    out.push({
      lambda: lambdas[li],
      bias2: totalBias2 / nTest,
      variance: totalVar / nTest,
      mse: (totalBias2 + totalVar) / nTest,
    });
  }
  return out;
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function LassoUCurve() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const curve = useMemo(() => computeUCurve(), []);

  // Find argmin of MSE for the marker.
  const argminMse = useMemo(() => {
    let best = 0;
    for (let i = 1; i < curve.length; i++) {
      if (curve[i].mse < curve[best].mse) best = i;
    }
    return best;
  }, [curve]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([LAMBDA_MIN, LAMBDA_MAX_CAP]).range([0, innerW]);
      const yMin = Math.max(1e-4, Math.min(...curve.map((c) => Math.min(c.bias2, c.variance, c.mse))) * 0.5);
      const yMax = Math.max(...curve.map((c) => c.mse)) * 1.3;
      const yScale = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]).clamp(true);

      // Axes.
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

      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('x', innerW / 2)
        .attr('y', innerH + 38)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '13px')
        .text('λ (log scale)');
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '13px')
        .text('decomposition (log scale)');

      // Three curves: bias², variance, MSE.
      const draw = (
        accessor: (c: CurvePoint) => number,
        color: string,
        dash: string | null,
        label: string,
        legendIdx: number,
      ) => {
        const lineGen = d3
          .line<CurvePoint>()
          .x((c) => xScale(c.lambda))
          .y((c) => yScale(Math.max(yMin, accessor(c))));
        const path = g.append('path').datum(curve).attr('d', lineGen).style('fill', 'none').style('stroke', color).style('stroke-width', 2);
        if (dash) path.style('stroke-dasharray', dash);
        // Legend entry.
        const legend = g.append('g').attr('transform', `translate(${innerW - 110},${4 + legendIdx * 18})`);
        const leg = legend.append('rect').attr('width', 14).attr('height', 4).attr('y', 6).style('fill', color);
        if (dash) leg.style('stroke', color).style('stroke-dasharray', dash);
        legend.append('text').attr('x', 20).attr('y', 11).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').text(label);
      };
      draw((c) => c.bias2, TEAL, '5 3', 'bias²', 0);
      draw((c) => c.variance, PURPLE, '2 2', 'variance', 1);
      draw((c) => c.mse, AMBER, null, 'MSE = bias² + var', 2);

      // Marker at argmin MSE.
      const minPoint = curve[argminMse];
      g.append('line')
        .attr('x1', xScale(minPoint.lambda))
        .attr('x2', xScale(minPoint.lambda))
        .attr('y1', 0)
        .attr('y2', innerH)
        .style('stroke', RED)
        .style('stroke-dasharray', '4 3')
        .style('stroke-width', 1.5)
        .style('opacity', 0.7);
      g.append('text')
        .attr('x', xScale(minPoint.lambda) + 4)
        .attr('y', 12)
        .style('fill', RED)
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .text(`λ_min = ${minPoint.lambda.toFixed(3)}`);
    },
    [curve, argminMse, w, isMobile],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={renderRef} width={w} height={HEIGHT} role="img" aria-label="Lasso bias-variance U-curve over lambda" />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        Empirical bias-variance decomposition on a smaller-scale DGP-1 (n = {N}, p = {P}, s = {S}, σ = {SIGMA}, AR(1) ρ = {RHO}, B = {B} replicates). MSE = bias² + variance is the canonical U; bias² (teal, dashed) grows with λ as constant shrinkage hits active coords harder; variance (purple, dotted) decays with λ as the active-set size shrinks. The red marker sits at the empirical λ minimizer of MSE — close to the LassoCV-selected operating point covered in §7. Computed live in-browser via warm-started ISTA across the 25-point log-spaced λ-grid (~3-5 s precompute).
      </p>
    </div>
  );
}

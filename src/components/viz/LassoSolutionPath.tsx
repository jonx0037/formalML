import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  generateDgp1,
  lambdaMax,
  lassoIsta,
  operatorNorm,
} from './shared/high-dim-regression';

// =============================================================================
// LassoSolutionPath — interactive companion to §4.4 Figure 4.2.
// Coefficient paths β̂_j(λ) vs log λ for all p coordinates on a smaller-scale
// DGP-1 (n = 200, p = 200, s = 10 — matches the LassoUCurve viz). The 10 true
// active coordinates are plotted in black; 190 inactive coordinates in light
// gray. Vertical marker at λ_CV ≈ 0.056.
//
// Computed via warm-started ISTA across a 30-point log-spaced λ-grid descending
// from λ_max = ‖Xᵀy/n‖_∞ down to 10⁻³ · λ_max. Warm starts mean each ISTA call
// after the first converges in ~30 iterations rather than 200, so the full path
// computes in ~200 ms.
//
// Static fallback: public/images/topics/high-dimensional-regression/fig_04_02_solution_path.png
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 16, bottom: 50, left: 60 };

const N = 200;
const P = 200; // matches LassoUCurve scale; see brief notes on p < 500 for in-browser
const S = 10;
const SIGMA = 0.5;
const RHO = 0.5;
const SEED = 42;
const N_LAMBDA = 30;
const LAMBDA_RATIO = 1e-3; // λ_min / λ_max
const ISTA_ITERS_FIRST = 200;
const ISTA_ITERS_WARM = 50;

// CV-selected λ (hardcoded from the LassoUCurve at this same DGP-1 scale).
const LAMBDA_CV = 0.056;

const ACTIVE_COLOR = '#1A1A1A';
const INACTIVE_COLOR = '#9CA3AF';
const RED = '#B91C1C';

interface PathData {
  /** Length-N_LAMBDA descending log-spaced λ values. */
  lambdas: number[];
  /** Length-P, each entry is a length-N_LAMBDA array of β̂_j values across the path. */
  paths: number[][];
}

// -----------------------------------------------------------------------------
// Compute the lasso path via warm-started ISTA.
// -----------------------------------------------------------------------------

function computePath(): PathData {
  const opts = { n: N, p: P, s: S, sigma: SIGMA, rho: RHO, seed: SEED };
  const sample = generateDgp1(opts);
  const L = operatorNorm(sample.X);
  const lMax = lambdaMax(sample.X, sample.y);
  // Descending λ-grid from λ_max down to LAMBDA_RATIO · λ_max.
  const logLo = Math.log(lMax * LAMBDA_RATIO);
  const logHi = Math.log(lMax);
  const lambdas: number[] = [];
  for (let i = 0; i < N_LAMBDA; i++) {
    lambdas.push(Math.exp(logHi - ((logHi - logLo) * i) / (N_LAMBDA - 1)));
  }
  const paths: number[][] = Array.from({ length: P }, () => new Array<number>(N_LAMBDA).fill(0));
  let prevBeta: Float64Array | undefined = undefined;
  for (let li = 0; li < lambdas.length; li++) {
    const iters = li === 0 ? ISTA_ITERS_FIRST : ISTA_ITERS_WARM;
    const beta = lassoIsta(sample.X, sample.y, lambdas[li], L, iters, prevBeta);
    for (let j = 0; j < P; j++) paths[j][li] = beta[j];
    prevBeta = beta;
  }
  return { lambdas, paths };
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function LassoSolutionPath() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const data = useMemo(() => computePath(), []);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([data.lambdas[data.lambdas.length - 1], data.lambdas[0]]).range([0, innerW]);
      // Symmetric y-domain from coefficient extremes.
      let yMaxAbs = 0;
      for (const path of data.paths) for (const v of path) yMaxAbs = Math.max(yMaxAbs, Math.abs(v));
      const yScale = d3.scaleLinear().domain([-yMaxAbs * 1.1, yMaxAbs * 1.1]).range([innerH, 0]);

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

      g.append('text').attr('text-anchor', 'middle').attr('x', innerW / 2).attr('y', innerH + 38).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '13px').text('λ (log scale)');
      g.append('text').attr('text-anchor', 'middle').attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '13px').text('β̂_j(λ)');

      // Zero line.
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(0)).attr('y2', yScale(0)).style('stroke', 'var(--color-border)').style('stroke-width', 0.8);

      // Inactive paths first (gray, transparent), then active on top (black, solid).
      const lineGen = d3
        .line<number>()
        .x((_, idx) => xScale(data.lambdas[idx]))
        .y((v) => yScale(v));

      // Inactive paths — only draw those with non-trivial magnitude to keep SVG light.
      for (let j = S; j < P; j++) {
        const pathData = data.paths[j];
        let maxAbs = 0;
        for (const v of pathData) maxAbs = Math.max(maxAbs, Math.abs(v));
        if (maxAbs < 0.005) continue; // skip flat-zero paths
        g.append('path').datum(pathData).attr('d', lineGen).style('fill', 'none').style('stroke', INACTIVE_COLOR).style('stroke-width', 0.8).style('opacity', 0.5);
      }

      // Active paths.
      for (let j = 0; j < S; j++) {
        g.append('path').datum(data.paths[j]).attr('d', lineGen).style('fill', 'none').style('stroke', ACTIVE_COLOR).style('stroke-width', 1.5);
      }

      // λ_CV vertical marker.
      g.append('line').attr('x1', xScale(LAMBDA_CV)).attr('x2', xScale(LAMBDA_CV)).attr('y1', 0).attr('y2', innerH).style('stroke', RED).style('stroke-dasharray', '4 3').style('stroke-width', 1.5).style('opacity', 0.7);
      g.append('text').attr('x', xScale(LAMBDA_CV) + 4).attr('y', 12).style('fill', RED).style('font-family', 'var(--font-mono)').style('font-size', '11px').text(`λ_CV ≈ ${LAMBDA_CV}`);

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 130},${4})`);
      const items: [string, string][] = [
        [`active (j < ${S})`, ACTIVE_COLOR],
        [`inactive (${P - S} of ${P})`, INACTIVE_COLOR],
      ];
      items.forEach(([label, color], i) => {
        const lg = legend.append('g').attr('transform', `translate(0,${i * 18})`);
        lg.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 8).attr('y2', 8).style('stroke', color).style('stroke-width', 2);
        lg.append('text').attr('x', 20).attr('y', 11).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').text(label);
      });
    },
    [data, w, isMobile],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={renderRef} width={w} height={HEIGHT} role="img" aria-label="Lasso solution path versus log lambda" />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        Lasso solution path on a smaller-scale DGP-1 (n = {N}, p = {P}, s = {S}, σ = {SIGMA}, AR(1) ρ = {RHO}). The 10 true active coordinates (j &lt; {S}) plot in black; the {P - S} inactive coordinates in gray (only those with |β̂_j| &gt; 0.005 anywhere on the path are drawn — most stay flat-zero and are omitted to keep the SVG light). Vertical marker at λ_CV ≈ {LAMBDA_CV} (the LassoUCurve minimizer above). The active features enter the path first as λ decreases from λ_max ≈ 1; at λ_CV the active set is a tight superset of the true S. Computed live via warm-started ISTA across {N_LAMBDA} log-spaced λ values (~200 ms precompute).
      </p>
    </div>
  );
}

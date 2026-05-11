import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  choleskyFactor,
  matVec,
  solveLowerTriangular,
  solveUpperTriangularT,
} from './shared/gaussian-processes';

// =============================================================================
// OlsOverfitsAtHighDim — interactive companion to §1.2 Figure 1.1.
// Shows OLS train MSE → 0 and test MSE → ∞ as the active feature count p_used
// approaches the sample size n on DGP-1.
//
// DGP-1 (fixed, matches notebook Cell 6):
//   n = 200, p = 200 (active design pool; we sweep p_used ≤ p)
//   X rows iid N(0, Σ) with Σ_jk = 0.5^|j-k| (AR(1) Toeplitz)
//   β*_j = 1 for j ∈ {0..9}, 0 otherwise
//   y = Xβ* + ε,  ε ~ N(0, σ²I) with σ = 0.5
//   Test set: same DGP, n_test = 200, independent draw, fresh seed.
//
// At each p_used, OLS solves (X[:,:p_used]ᵀX[:,:p_used] + εI) β = X[:,:p_used]ᵀy
// via Cholesky. The ε = 1e-8 jitter prevents the Cholesky pivot from going
// negative when p_used → n; the substantive curve is unaffected (the jitter is
// orders of magnitude below any meaningful eigenvalue at p_used < n).
//
// Static fallback: public/images/topics/high-dimensional-regression/fig_01_01_ols_overfit.png
// Numerical anchor: train MSE → ~0 at p_used = 199; test MSE → ~10² at p_used = 199.
// =============================================================================

const HEIGHT = 420;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 16, bottom: 50, left: 60 };

const N = 200;
const P = 200;
const SIGMA = 0.5;
const RHO = 0.5;
const S = 10;
const JITTER = 1e-8;
const SEED_TRAIN = 42;
const SEED_TEST = 142;

// p_used grid: dense near s = 10 and near n = 200 to show both inflection points clearly.
const P_USED_GRID: number[] = (() => {
  const points = new Set<number>([2, 4, 6, 8, 10, 12, 14, 17, 20, 25, 30, 40, 50, 70, 100, 130, 160, 180, 190, 195, 197, 199]);
  return Array.from(points).sort((a, b) => a - b);
})();

const TEAL = '#0F6E56';
const AMBER = '#D97706';
const RED = '#B91C1C';
const SLATE = '#6B6B6B';

// -----------------------------------------------------------------------------
// Deterministic RNG: mulberry32 + Box-Muller (used across the topic; see
// proximalUtils for the same pattern).
// -----------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRng(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

// -----------------------------------------------------------------------------
// AR(1) sampler: x_j = ρ x_{j-1} + sqrt(1-ρ²) z_j, with z_j iid N(0,1).
// Marginal cov is Σ_jk = ρ^|j-k|, so this matches DGP-1 exactly.
// -----------------------------------------------------------------------------

function ar1Row(p: number, rho: number, gauss: () => number): Float64Array {
  const x = new Float64Array(p);
  const sqrt1mr2 = Math.sqrt(1 - rho * rho);
  x[0] = gauss();
  for (let j = 1; j < p; j++) x[j] = rho * x[j - 1] + sqrt1mr2 * gauss();
  return x;
}

function generateData(seed: number): { X: Float64Array[]; y: Float64Array } {
  const rng = mulberry32(seed);
  const gauss = gaussianRng(rng);
  const X: Float64Array[] = [];
  const y = new Float64Array(N);
  // Truth: β*_j = 1 for j < S, 0 otherwise. y_i = sum_{j<S} X_{ij} + σ z_i.
  for (let i = 0; i < N; i++) {
    const row = ar1Row(P, RHO, gauss);
    X.push(row);
    let signal = 0;
    for (let j = 0; j < S; j++) signal += row[j];
    y[i] = signal + SIGMA * gauss();
  }
  return { X, y };
}

// -----------------------------------------------------------------------------
// Ridge-regularized OLS via Cholesky.
//   β̂ = (XᵀX + ε I)⁻¹ Xᵀy with ε = JITTER (numerical stability only).
// X[i] is row i of length p_used; XtX is p_used × p_used.
// -----------------------------------------------------------------------------

function ridgeOls(X: Float64Array[], y: Float64Array, pUsed: number): Float64Array {
  // XtX (p_used × p_used) using only first p_used columns.
  const XtX: number[][] = [];
  for (let j = 0; j < pUsed; j++) {
    const row = new Array<number>(pUsed).fill(0);
    for (let k = 0; k <= j; k++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += X[i][j] * X[i][k];
      row[k] = s;
    }
    XtX.push(row);
  }
  // Symmetrize and add jitter.
  for (let j = 0; j < pUsed; j++) {
    for (let k = 0; k < j; k++) XtX[k][j] = XtX[j][k];
    XtX[j][j] += JITTER;
  }
  // Xty (length p_used).
  const Xty = new Array<number>(pUsed).fill(0);
  for (let j = 0; j < pUsed; j++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += X[i][j] * y[i];
    Xty[j] = s;
  }
  // Solve (XtX) β = Xty via Cholesky.
  const L = choleskyFactor(XtX);
  const yt = solveLowerTriangular(L, Xty);
  const betaArr = solveUpperTriangularT(L, yt);
  const beta = new Float64Array(pUsed);
  for (let i = 0; i < pUsed; i++) beta[i] = betaArr[i];
  return beta;
}

function predictMse(X: Float64Array[], y: Float64Array, beta: Float64Array): number {
  const pUsed = beta.length;
  let sse = 0;
  for (let i = 0; i < N; i++) {
    let pred = 0;
    for (let j = 0; j < pUsed; j++) pred += X[i][j] * beta[j];
    const r = y[i] - pred;
    sse += r * r;
  }
  return sse / N;
}

interface CurvePoint {
  pUsed: number;
  trainMse: number;
  testMse: number;
}

function computeCurve(
  Xtr: Float64Array[],
  ytr: Float64Array,
  Xte: Float64Array[],
  yte: Float64Array,
): CurvePoint[] {
  return P_USED_GRID.map((pUsed) => {
    const beta = ridgeOls(Xtr, ytr, pUsed);
    return {
      pUsed,
      trainMse: predictMse(Xtr, ytr, beta),
      testMse: predictMse(Xte, yte, beta),
    };
  });
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function OlsOverfitsAtHighDim() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const curve = useMemo(() => {
    const train = generateData(SEED_TRAIN);
    const test = generateData(SEED_TEST);
    return computeCurve(train.X, train.y, test.X, test.y);
  }, []);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0, P]).range([0, innerW]);
      const yMin = Math.max(1e-4, Math.min(...curve.map((p) => p.trainMse)) / 5);
      const yMax = Math.max(...curve.map((p) => p.testMse)) * 1.5;
      const yScale = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]).clamp(true);

      // Axes.
      const xAxis = d3
        .axisBottom(xScale)
        .ticks(isMobile ? 5 : 8)
        .tickSize(-innerH);
      const yAxis = d3
        .axisLeft(yScale)
        .ticks(isMobile ? 4 : 6, '~g')
        .tickSize(-innerW);

      g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${innerH})`)
        .call(xAxis)
        .call((sel) => {
          sel.selectAll('line').style('stroke', 'var(--color-border)');
          sel.selectAll('path').style('stroke', 'var(--color-text-secondary)');
          sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)');
        });

      g.append('g')
        .attr('class', 'y-axis')
        .call(yAxis)
        .call((sel) => {
          sel.selectAll('line').style('stroke', 'var(--color-border)');
          sel.selectAll('path').style('stroke', 'var(--color-text-secondary)');
          sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)');
        });

      // Axis labels.
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('x', innerW / 2)
        .attr('y', innerH + 38)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '13px')
        .text('Number of features used in OLS fit, p_used');

      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '13px')
        .text('MSE (log scale)');

      // Vertical reference markers: p_used = s (true sparsity), p_used = n (rank-deficiency).
      const drawVLine = (xVal: number, label: string, color: string) => {
        g.append('line')
          .attr('x1', xScale(xVal))
          .attr('x2', xScale(xVal))
          .attr('y1', 0)
          .attr('y2', innerH)
          .style('stroke', color)
          .style('stroke-dasharray', '4 3')
          .style('stroke-width', 1.5)
          .style('opacity', 0.6);
        g.append('text')
          .attr('x', xScale(xVal) + 4)
          .attr('y', 12)
          .style('fill', color)
          .style('font-family', 'var(--font-mono)')
          .style('font-size', '11px')
          .text(label);
      };
      drawVLine(S, `s = ${S}`, SLATE);
      drawVLine(N, `n = ${N}`, RED);

      // Line generator.
      const lineGen = (accessor: (d: CurvePoint) => number) =>
        d3
          .line<CurvePoint>()
          .x((d) => xScale(d.pUsed))
          .y((d) => yScale(Math.max(yMin, accessor(d))));

      // Train curve (teal, solid).
      g.append('path')
        .datum(curve)
        .attr('d', lineGen((d) => d.trainMse))
        .style('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', 2);

      // Test curve (amber, solid).
      g.append('path')
        .datum(curve)
        .attr('d', lineGen((d) => d.testMse))
        .style('fill', 'none')
        .style('stroke', AMBER)
        .style('stroke-width', 2);

      // Data-point markers + tooltip.
      const tooltip = svg
        .append('g')
        .attr('class', 'tooltip')
        .style('opacity', 0)
        .style('pointer-events', 'none');
      tooltip
        .append('rect')
        .attr('width', 200)
        .attr('height', 56)
        .attr('rx', 4)
        .style('fill', 'var(--color-surface)')
        .style('stroke', 'var(--color-border)');
      const tooltipText = tooltip
        .append('text')
        .attr('x', 8)
        .attr('y', 18)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px');

      const drawMarkers = (accessor: (d: CurvePoint) => number, color: string, kind: string) => {
        g.selectAll(`.dot-${kind}`)
          .data(curve)
          .enter()
          .append('circle')
          .attr('class', `dot-${kind}`)
          .attr('cx', (d) => xScale(d.pUsed))
          .attr('cy', (d) => yScale(Math.max(yMin, accessor(d))))
          .attr('r', 3.5)
          .style('fill', color)
          .style('stroke', 'var(--color-bg)')
          .style('stroke-width', 1)
          .style('cursor', 'pointer')
          .on('mouseenter', function (_event, d) {
            tooltip.style('opacity', 1);
            tooltipText.selectAll('tspan').remove();
            tooltipText.append('tspan').attr('x', 8).attr('dy', 0).text(`p_used = ${d.pUsed}`);
            tooltipText.append('tspan').attr('x', 8).attr('dy', 14).text(`train MSE = ${d.trainMse.toExponential(2)}`);
            tooltipText.append('tspan').attr('x', 8).attr('dy', 14).text(`test  MSE = ${d.testMse.toExponential(2)}`);
            const xPx = xScale(d.pUsed) + MARGIN.left;
            const yPx = yScale(Math.max(yMin, accessor(d))) + MARGIN.top;
            tooltip.attr('transform', `translate(${Math.min(xPx + 8, w - 208)},${Math.max(yPx - 64, 4)})`);
          })
          .on('mouseleave', () => tooltip.style('opacity', 0));
      };
      drawMarkers((d) => d.trainMse, TEAL, 'train');
      drawMarkers((d) => d.testMse, AMBER, 'test');

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 130},${4})`);
      const legendItems: [string, string][] = [
        ['train MSE', TEAL],
        ['test MSE', AMBER],
      ];
      legendItems.forEach(([label, color], i) => {
        const lg = legend.append('g').attr('transform', `translate(0,${i * 18})`);
        lg.append('rect').attr('width', 14).attr('height', 4).attr('y', 6).style('fill', color);
        lg.append('text')
          .attr('x', 20)
          .attr('y', 11)
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '12px')
          .text(label);
      });
    },
    [curve, w, isMobile],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={renderRef} width={w} height={HEIGHT} role="img" aria-label="OLS train and test MSE versus active feature count" />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        OLS on DGP-1 (n = {N}, σ = {SIGMA}, AR(1) ρ = {RHO}, s = {S}). Train MSE drops to zero as p_used → n; test MSE bottoms near p_used = s and explodes near the rank-deficiency boundary p_used = n. Hover any point for exact values. Computed live in-browser with Cholesky-based ridge OLS (jitter ε = {JITTER.toExponential(0)} for numerical stability at large p_used).
      </p>
    </div>
  );
}

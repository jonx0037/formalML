import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  generateDgp1,
  operatorNorm,
  xMul,
  xtMul,
  lassoIsta,
} from './shared/high-dim-regression';
import {
  choleskyFactor,
  solveLowerTriangular,
  solveUpperTriangularT,
} from './shared/gaussian-processes';

// =============================================================================
// RidgeVsLassoCoefBars — interactive companion to §1.3+§1.4 Figure 1.2/1.3.
// 6-panel coefficient bar chart on the same DGP-1 sample:
//   Top row:    ridge at α ∈ {0.01, 1, 100}    (all dense)
//   Bottom row: lasso at λ ∈ {0.001, 0.0559, 1} (sparse at moderate λ)
//
// DGP-1 (matches notebook Cell 10):
//   n = 200, p = 500, s = 10, σ = 0.5, AR(1) ρ = 0.5
//   β*_j = 1 for j ∈ {0..9}, 0 otherwise
//
// The middle lasso panel uses the CV-selected λ ≈ 0.0559 from the notebook —
// hardcoded here to keep the viz client-side and fast (LassoCV across 500
// features × 10 folds is too heavy for in-browser computation).
//
// Solvers:
//   ridge: closed-form (XᵀX + nαI)⁻¹ Xᵀy via Cholesky
//   lasso: 200-iteration ISTA with step η = 1/L where L = ‖XᵀX/n‖_op
//          (estimated by 30-iteration power method)
//
// Static fallback: public/images/topics/high-dimensional-regression/fig_01_02_03_ridge_vs_lasso_coefs.png
// Numerical anchor: lasso at λ_CV produces ~12 nonzero coefficients on the §1
// sample (notebook Cell 10 prints 13).
// =============================================================================

const HEIGHT_PER_PANEL = 130;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 24, right: 8, bottom: 30, left: 38 };

const N = 200;
const P = 500;
const SIGMA = 0.5;
const RHO = 0.5;
const S = 10;
const SEED = 42;
const ISTA_ITERS = 200;
// Thresholds for "nonzero" count display. Ridge always has all p coefficients
// mathematically nonzero (closed-form (XᵀX + nαI)⁻¹ Xᵀy hits zero only on a
// measure-zero set), so we count anything above 1e-12 (numerical-zero only).
// Lasso reaches exact zeros via soft-thresholding once ISTA converges; small
// residuals during the convergence tail are filtered with a 0.01 threshold
// (1% of the true β* = 1 — substantively sensible cutoff matching how the
// notebook reports active-set sizes via numpy's `np.sum(np.abs(beta) > tol)`).
const RIDGE_NONZERO_THRESH = 1e-12;
const LASSO_NONZERO_THRESH = 0.01;

const RIDGE_ALPHAS = [0.01, 1.0, 100.0];
const LASSO_LAMBDAS = [0.001, 0.0559, 1.0];
const LASSO_LABELS = ['λ = 0.001', 'λ ≈ 0.056 (CV-selected)', 'λ = 1.0'];

const TEAL = '#0F6E56';
const SLATE_BAR = '#9CA3AF';
const ACTIVE_BAR = '#1A1A1A';

// -----------------------------------------------------------------------------
// XᵀX (lower triangle only, then mirrored). Shared across the 3 ridge fits so
// the O(np²) Gram construction is paid once instead of 3×.
// -----------------------------------------------------------------------------
function buildXtX(X: Float64Array[]): number[][] {
  const A: number[][] = [];
  for (let j = 0; j < P; j++) A.push(new Array<number>(P).fill(0));
  for (let j = 0; j < P; j++) {
    for (let k = 0; k <= j; k++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += X[i][j] * X[i][k];
      A[j][k] = s;
      A[k][j] = s;
    }
  }
  return A;
}

// -----------------------------------------------------------------------------
// Ridge with precomputed XᵀX: for each α, clone A, shift diagonal by +n·α,
// Cholesky-solve. Cholesky is the dominant cost per call (~p³/6); reusing XtX
// saves the O(np²) Gram rebuild that previously fired once per α.
// -----------------------------------------------------------------------------
function ridgeFitFromGram(
  XtX: number[][],
  Xty: Float64Array,
  alpha: number,
): Float64Array {
  const A: number[][] = XtX.map((row) => row.slice());
  for (let j = 0; j < P; j++) A[j][j] += N * alpha;
  const L = choleskyFactor(A);
  const yt = solveLowerTriangular(L, Array.from(Xty));
  const betaArr = solveUpperTriangularT(L, yt);
  const beta = new Float64Array(P);
  for (let j = 0; j < P; j++) beta[j] = betaArr[j];
  return beta;
}

interface PanelData {
  title: string;
  beta: Float64Array;
  nonzeroCount: number;
  family: 'ridge' | 'lasso';
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function RidgeVsLassoCoefBars() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const panels = useMemo<PanelData[]>(() => {
    const { X, y } = generateDgp1({ n: N, p: P, s: S, sigma: SIGMA, rho: RHO, seed: SEED });
    const L = operatorNorm(X);
    // Pre-compute XᵀX + Xᵀy once; ridge fits at different α only need a
    // diagonal shift on a clone of the Gram matrix.
    const XtX = buildXtX(X);
    const Xty = xtMul(X, y);
    const ridgePanels: PanelData[] = RIDGE_ALPHAS.map((alpha) => {
      const beta = ridgeFitFromGram(XtX, Xty, alpha);
      let nonzero = 0;
      for (let j = 0; j < P; j++) if (Math.abs(beta[j]) > RIDGE_NONZERO_THRESH) nonzero++;
      return { title: `Ridge (α = ${alpha})`, beta, nonzeroCount: nonzero, family: 'ridge' };
    });
    const lassoPanels: PanelData[] = LASSO_LAMBDAS.map((lambda, i) => {
      const beta = lassoIsta(X, y, lambda, L, ISTA_ITERS);
      let nonzero = 0;
      for (let j = 0; j < P; j++) if (Math.abs(beta[j]) > LASSO_NONZERO_THRESH) nonzero++;
      return { title: `Lasso (${LASSO_LABELS[i]})`, beta, nonzeroCount: nonzero, family: 'lasso' };
    });
    return [...ridgePanels, ...lassoPanels];
  }, []);

  // Symmetric y range across all 6 panels for direct visual comparison.
  const yMax = useMemo(() => {
    let m = 0;
    for (const p of panels) for (let j = 0; j < P; j++) m = Math.max(m, Math.abs(p.beta[j]));
    return m * 1.05;
  }, [panels]);

  const totalH = HEIGHT_PER_PANEL * 2 + 24; // 2 rows + spacing

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const cols = isMobile ? 1 : 3;
      const rows = isMobile ? 6 : 2;
      const panelW = w / cols;
      const panelH = isMobile ? HEIGHT_PER_PANEL : HEIGHT_PER_PANEL;
      const totalHeight = rows * panelH + (isMobile ? 8 * (rows - 1) : 24);
      svg.attr('height', totalHeight);

      panels.forEach((panel, idx) => {
        const col = isMobile ? 0 : idx % cols;
        const row = isMobile ? idx : Math.floor(idx / cols);
        const xOff = col * panelW;
        const yOff = row * panelH + (isMobile ? row * 8 : row * 24);
        const innerW = panelW - MARGIN.left - MARGIN.right;
        const innerH = panelH - MARGIN.top - MARGIN.bottom;
        const g = svg
          .append('g')
          .attr('transform', `translate(${xOff + MARGIN.left},${yOff + MARGIN.top})`);

        const xScale = d3.scaleLinear().domain([0, P]).range([0, innerW]);
        const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerH, 0]);

        // Zero-axis line.
        g.append('line')
          .attr('x1', 0)
          .attr('x2', innerW)
          .attr('y1', yScale(0))
          .attr('y2', yScale(0))
          .style('stroke', 'var(--color-border)')
          .style('stroke-width', 0.5);

        // Bars: 500 thin vertical lines (faster than 500 <rect> elements).
        // Active coordinates (j < S) drawn LAST so they sit on top.
        const barW = Math.max(0.6, innerW / P);
        // Only draw bars above a small visual threshold so 500 ridge bars don't
        // smear into solid color. Ridge: 0.005 (just above visual noise floor).
        // Lasso: same 0.01 used for the count.
        const drawThresh = panel.family === 'ridge' ? 0.005 : LASSO_NONZERO_THRESH;
        for (let j = S; j < P; j++) {
          const v = panel.beta[j];
          if (Math.abs(v) < drawThresh) continue;
          g.append('line')
            .attr('x1', xScale(j))
            .attr('x2', xScale(j))
            .attr('y1', yScale(0))
            .attr('y2', yScale(v))
            .style('stroke', SLATE_BAR)
            .style('stroke-width', barW)
            .style('opacity', 0.55);
        }
        for (let j = 0; j < S; j++) {
          const v = panel.beta[j];
          g.append('line')
            .attr('x1', xScale(j))
            .attr('x2', xScale(j))
            .attr('y1', yScale(0))
            .attr('y2', yScale(v))
            .style('stroke', ACTIVE_BAR)
            .style('stroke-width', Math.max(1.2, barW * 1.5));
        }

        // Title with nonzero count.
        g.append('text')
          .attr('x', innerW / 2)
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '12px')
          .style('font-weight', 600)
          .text(`${panel.title} · ${panel.nonzeroCount} nonzero`);

        // y-axis (only on leftmost panel of each row).
        if (col === 0) {
          const yAxis = d3.axisLeft(yScale).ticks(3, '~g').tickSize(0);
          g.append('g')
            .call(yAxis)
            .call((sel) => {
              sel.selectAll('path').remove();
              sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)').style('font-size', '10px');
            });
        }
        // x-axis (only on bottom row).
        if (row === rows - 1) {
          const xAxis = d3.axisBottom(xScale).ticks(isMobile ? 3 : 5, '~g').tickSize(0);
          g.append('g')
            .attr('transform', `translate(0,${innerH})`)
            .call(xAxis)
            .call((sel) => {
              sel.selectAll('path').remove();
              sel.selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-mono)').style('font-size', '10px');
            });
          g.append('text')
            .attr('x', innerW / 2)
            .attr('y', innerH + 22)
            .attr('text-anchor', 'middle')
            .style('fill', 'var(--color-text-secondary)')
            .style('font-family', 'var(--font-sans)')
            .style('font-size', '11px')
            .text('coordinate j');
        }
      });
    },
    [panels, yMax, w, isMobile],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={renderRef} width={w} height={totalH} role="img" aria-label="Ridge versus lasso coefficient bar charts at three penalty levels" />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        Ridge (top row, three α levels) vs lasso (bottom row, three λ levels) on DGP-1 (n = {N}, p = {P}, s = {S}, σ = {SIGMA}). True active coordinates (j &lt; {S}) in black; 490 inactive coordinates in gray. Ridge is dense at every α; lasso at the CV-selected λ ≈ 0.056 produces a sparse fit concentrated at the true active set. Computed live in-browser via Cholesky-based ridge with pre-computed XᵀX (shared across α levels) and {ISTA_ITERS}-iteration ISTA for lasso. Compute ~1 second; viz is hidden until scrolled into view.
      </p>
    </div>
  );
}

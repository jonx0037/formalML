import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  generateDgp1,
  operatorNorm,
  xMul,
  xtMul,
} from './shared/high-dim-regression';
import { softThreshold } from './shared/proximalUtils';

// =============================================================================
// FistaConvergenceTrace — interactive companion to §3.4 Figure 3.1.
//
// Log-log convergence trace F(β^k) − F* vs iteration k for three lasso
// solvers on a smaller-scale DGP-1 (n = 150, p = 200, s = 10, λ = 0.05, the
// regime where solver-rate differences are visually maximal):
//   • ISTA:  proximal gradient, O(1/k) — slope ≈ −1 on log-log
//   • FISTA: Nesterov-accelerated, O(1/k²) — slope ≈ −2 on log-log
//   • CD:    coordinate descent, asymptotically linear after active-set
//            stabilization — eventually faster than either above
//
// F* is the lasso objective at a high-precision FISTA reference (REFERENCE_ITERS
// iterations). The clamp at 1e-12 protects log(0) when a solver dips below F*
// (rare; only the reference itself should hit F*).
//
// Objective convention: F(β) = (1/2n)‖Xβ - y‖² + λ‖β‖₁ — matches the rest of
// the topic (high-dim-regression.ts `lassoIsta`).
//
// Static fallback: public/images/topics/high-dimensional-regression/fig_03_01_solver_convergence.png
// Compute budget: ~150-250 ms (one-shot, no slider; cached in useMemo).
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 16, bottom: 50, left: 60 };

const N = 150;
const P = 200;
const S = 10;
const SIGMA = 0.5;
const RHO = 0.5;
const LAMBDA = 0.05;
const ITERS = 400; // outer iters for ISTA/FISTA; sweeps for CD
const REFERENCE_ITERS = 5000;
const SEED = 42;

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const SLATE = '#6B6B6B';

interface Trace {
  label: string;
  color: string;
  dash: string | null;
  values: number[]; // F(β^k) at each k, length = ITERS + 1 (includes β^0)
}

// -----------------------------------------------------------------------------
// Objective F(β) = (1/(2n))‖Xβ - y‖² + λ‖β‖₁
// -----------------------------------------------------------------------------

function objective(
  X: Float64Array[],
  y: Float64Array,
  beta: Float64Array,
  lambda: number,
): number {
  const n = X.length;
  const Xb = xMul(X, beta);
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = Xb[i] - y[i];
    sse += r * r;
  }
  let l1 = 0;
  for (let j = 0; j < beta.length; j++) l1 += Math.abs(beta[j]);
  return sse / (2 * n) + lambda * l1;
}

// -----------------------------------------------------------------------------
// ISTA with trace.
//
// β^{k+1} = S(β^k + (η/n) Xᵀ(y - Xβ^k), η λ),  η = 1/L
// -----------------------------------------------------------------------------

function lassoIstaTrace(
  X: Float64Array[],
  y: Float64Array,
  lambda: number,
  L: number,
  iters: number,
): number[] {
  const n = X.length;
  const p = X[0].length;
  const eta = 1 / L;
  let beta = new Float64Array(p);
  const trace: number[] = [objective(X, y, beta, lambda)];
  for (let it = 0; it < iters; it++) {
    const Xb = xMul(X, beta);
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) r[i] = y[i] - Xb[i];
    const grad = xtMul(X, r);
    const z = new Array<number>(p);
    for (let j = 0; j < p; j++) z[j] = beta[j] + (eta / n) * grad[j];
    beta = new Float64Array(softThreshold(z, eta * lambda));
    trace.push(objective(X, y, beta, lambda));
  }
  return trace;
}

// -----------------------------------------------------------------------------
// FISTA with trace.
//
// z^k = β^k + ((t_k - 1)/t_{k+1}) (β^k - β^{k-1})
// β^{k+1} = S(z^k + (η/n) Xᵀ(y - Xz^k), η λ)
// t_{k+1} = (1 + sqrt(1 + 4 t_k²)) / 2
// -----------------------------------------------------------------------------

function lassoFistaTrace(
  X: Float64Array[],
  y: Float64Array,
  lambda: number,
  L: number,
  iters: number,
): { trace: number[]; final: Float64Array } {
  const n = X.length;
  const p = X[0].length;
  const eta = 1 / L;
  let beta = new Float64Array(p);
  let betaPrev = new Float64Array(p);
  let tk = 1;
  const trace: number[] = [objective(X, y, beta, lambda)];
  for (let it = 0; it < iters; it++) {
    const tNext = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2;
    const mom = (tk - 1) / tNext;
    const z = new Float64Array(p);
    for (let j = 0; j < p; j++) z[j] = beta[j] + mom * (beta[j] - betaPrev[j]);
    const Xz = xMul(X, z);
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) r[i] = y[i] - Xz[i];
    const grad = xtMul(X, r);
    const zStep = new Array<number>(p);
    for (let j = 0; j < p; j++) zStep[j] = z[j] + (eta / n) * grad[j];
    betaPrev = beta;
    beta = new Float64Array(softThreshold(zStep, eta * lambda));
    tk = tNext;
    trace.push(objective(X, y, beta, lambda));
  }
  return { trace, final: beta };
}

// -----------------------------------------------------------------------------
// Coordinate descent with trace.
//
// One "iteration" of CD = one full sweep through j = 0..p-1, each updating
// β_j by the closed-form scalar lasso solution conditional on β_{-j}. Total
// work per sweep is O(np), matching ISTA/FISTA per outer step. Maintains Xβ
// incrementally for O(n) per coordinate (O(np) per sweep), not O(np²).
//
//   c_j = X_j^T (y - X β^{(-j)}) / n = X_j^T (y - Xβ + X_j β_j) / n
//   D_j = ‖X_j‖² / n
//   β_j^new = S(c_j, λ) / D_j
// -----------------------------------------------------------------------------

function lassoCdTrace(
  X: Float64Array[],
  y: Float64Array,
  lambda: number,
  sweeps: number,
): number[] {
  const n = X.length;
  const p = X[0].length;
  const beta = new Float64Array(p);

  // Precompute column norms ‖X_j‖² / n.
  const colNorm2 = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j] * X[i][j];
    colNorm2[j] = s / n;
  }

  // Maintain Xβ incrementally.
  const Xb = new Float64Array(n);

  const trace: number[] = [objective(X, y, beta, lambda)];
  for (let sw = 0; sw < sweeps; sw++) {
    for (let j = 0; j < p; j++) {
      // c_j = X_j^T (y - Xβ + X_j β_j) / n
      let cj = 0;
      const bj = beta[j];
      for (let i = 0; i < n; i++) cj += X[i][j] * (y[i] - Xb[i] + X[i][j] * bj);
      cj /= n;
      const Dj = colNorm2[j];
      const num = Math.sign(cj) * Math.max(Math.abs(cj) - lambda, 0);
      const newBj = Dj > 1e-12 ? num / Dj : 0;
      const delta = newBj - bj;
      if (delta !== 0) {
        for (let i = 0; i < n; i++) Xb[i] += delta * X[i][j];
        beta[j] = newBj;
      }
    }
    trace.push(objective(X, y, beta, lambda));
  }
  return trace;
}

// -----------------------------------------------------------------------------
// Compute the three traces + F* reference.
// -----------------------------------------------------------------------------

function computeTraces(): { traces: Trace[]; fStar: number } {
  const { X, y } = generateDgp1({ n: N, p: P, s: S, sigma: SIGMA, rho: RHO, seed: SEED });
  const L = operatorNorm(X);

  // Reference F* via high-precision FISTA.
  const { trace: refTrace } = lassoFistaTrace(X, y, LAMBDA, L, REFERENCE_ITERS);
  const fStar = refTrace[refTrace.length - 1];

  const ista = lassoIstaTrace(X, y, LAMBDA, L, ITERS);
  const { trace: fista } = lassoFistaTrace(X, y, LAMBDA, L, ITERS);
  const cd = lassoCdTrace(X, y, LAMBDA, ITERS);

  return {
    traces: [
      { label: 'ISTA — O(1/k)', color: TEAL, dash: '5 3', values: ista },
      { label: 'FISTA — O(1/k²)', color: PURPLE, dash: null, values: fista },
      { label: 'Coordinate descent', color: AMBER, dash: '2 2', values: cd },
    ],
    fStar,
  };
}

// -----------------------------------------------------------------------------
// Component.
// -----------------------------------------------------------------------------

export default function FistaConvergenceTrace() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const { traces, fStar } = useMemo(() => computeTraces(), []);

  // Convert each trace value to F − F*, clamped at 1e-12 for log display.
  const gaps = useMemo(
    () =>
      traces.map((t) => ({
        ...t,
        values: t.values.map((v) => Math.max(1e-12, v - fStar)),
      })),
    [traces, fStar],
  );

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      // x-axis: log iteration count (skip k=0 since log(0) = -∞).
      const xScale = d3.scaleLog().domain([1, ITERS]).range([0, innerW]);
      // y-axis: log F − F*, with floor at 10^-10 (above the 10^-12 clamp).
      const allVals = gaps.flatMap((t) => t.values.slice(1));
      const yMin = Math.max(1e-10, d3.min(allVals) ?? 1e-10);
      const yMax = (d3.max(allVals) ?? 1) * 1.5;
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

      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('x', innerW / 2)
        .attr('y', innerH + 38)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '13px')
        .text('iteration k (log scale)');
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '13px')
        .text('F(βᵏ) − F* (log scale)');

      // Draw curves: skip k=0.
      gaps.forEach((tr, ti) => {
        const data: [number, number][] = tr.values.slice(1).map((v, k) => [k + 1, v]);
        const lineGen = d3
          .line<[number, number]>()
          .x((d) => xScale(d[0]))
          .y((d) => yScale(d[1]));
        const path = g
          .append('path')
          .datum(data)
          .attr('d', lineGen)
          .style('fill', 'none')
          .style('stroke', tr.color)
          .style('stroke-width', 2);
        if (tr.dash) path.style('stroke-dasharray', tr.dash);

        // Legend.
        const legend = g.append('g').attr('transform', `translate(${innerW - 168},${4 + ti * 18})`);
        const leg = legend.append('rect').attr('width', 14).attr('height', 4).attr('y', 6).style('fill', tr.color);
        if (tr.dash) leg.style('stroke', tr.color).style('stroke-dasharray', tr.dash);
        legend
          .append('text')
          .attr('x', 20)
          .attr('y', 11)
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '12px')
          .text(tr.label);
      });

      // Slope reference lines (faint, with annotation): k^-1 and k^-2 from k=2.
      const slopeLine = (exp: number, color: string, label: string, yIdx: number) => {
        // Anchor at k=3 for visibility; y = A k^-exp.
        const kAnchor = 3;
        const yAnchorIdx = Math.min(kAnchor, gaps[1].values.length - 1);
        const A = gaps[1].values[yAnchorIdx] * Math.pow(kAnchor, exp);
        const data: [number, number][] = [];
        for (let k = 2; k <= ITERS; k *= 1.5) {
          data.push([k, Math.max(yMin, A * Math.pow(k, -exp))]);
        }
        data.push([ITERS, Math.max(yMin, A * Math.pow(ITERS, -exp))]);
        const lineGen = d3
          .line<[number, number]>()
          .x((d) => xScale(d[0]))
          .y((d) => yScale(d[1]));
        g.append('path')
          .datum(data)
          .attr('d', lineGen)
          .style('fill', 'none')
          .style('stroke', color)
          .style('stroke-width', 1)
          .style('stroke-dasharray', '1 3')
          .style('opacity', 0.45);
        // Annotate end-point.
        g.append('text')
          .attr('x', innerW - 4)
          .attr('y', yScale(Math.max(yMin, A * Math.pow(ITERS, -exp))) + yIdx)
          .attr('text-anchor', 'end')
          .style('fill', color)
          .style('font-family', 'var(--font-mono)')
          .style('font-size', '10px')
          .style('opacity', 0.7)
          .text(label);
      };
      slopeLine(1, SLATE, 'k⁻¹', -4);
      slopeLine(2, SLATE, 'k⁻²', 10);
    },
    [gaps, w, isMobile],
  );

  // Headline iteration counts to reach a tolerance.
  const itersTo = (vals: number[], tol: number): number | null => {
    for (let k = 1; k < vals.length; k++) if (vals[k] < tol) return k;
    return null;
  };
  const TOL = 1e-3;
  const istaK = itersTo(gaps[0].values, TOL);
  const fistaK = itersTo(gaps[1].values, TOL);
  const cdK = itersTo(gaps[2].values, TOL);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg
        ref={renderRef}
        width={w}
        height={HEIGHT}
        role="img"
        aria-label="Log-log convergence trace of ISTA, FISTA, and coordinate descent for the lasso"
      />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        Log-log convergence trace on a smaller-scale DGP-1 (n = {N}, p = {P}, s = {S}, σ = {SIGMA}, AR(1) ρ = {RHO}) at λ = {LAMBDA}. F* is computed by a {REFERENCE_ITERS.toLocaleString()}-iteration FISTA reference. Reading off the slopes: ISTA tracks k⁻¹ (Theorem 3.2), FISTA tracks k⁻² (Theorem 3.3), and coordinate descent matches FISTA early then asymptotically beats both once the active set stabilizes — the lasso restricted to a fixed active set is a strictly-convex quadratic, where CD converges linearly. Iterations to reach F − F* &lt; 10⁻³: ISTA = {istaK ?? '> ' + ITERS}, FISTA = {fistaK ?? '> ' + ITERS}, CD = {cdK ?? '> ' + ITERS}.
      </p>
    </div>
  );
}

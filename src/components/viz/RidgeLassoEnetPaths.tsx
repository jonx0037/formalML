import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  generateDgp1,
  operatorNorm,
  xMul,
  xtMul,
  lambdaMax,
} from './shared/high-dim-regression';
import { softThreshold } from './shared/proximalUtils';
import {
  choleskyFactor,
  solveLowerTriangular,
  solveUpperTriangularT,
} from './shared/gaussian-processes';

// =============================================================================
// RidgeLassoEnetPaths — interactive companion to §8.4 Figure 8.1.
//
// Tabbed three-panel comparison on a smaller-scale DGP-1 (n = 150, p = 200,
// s = 10) of regularization paths β̂_j(t) for:
//   • Ridge:       closed-form β̂(α) = (XᵀX + n α I)⁻¹ Xᵀy via Cholesky,
//                  α swept on log-scale.
//   • Lasso:       warm-started ISTA, λ swept downward from λ_max.
//   • Elastic net: warm-started ISTA on the (γ-fixed) modified prox
//                  prox_{η}(z; λ, γ) = S(z, η λ) / (1 + η γ).
//
// Display: 10 true active coords colored, 190 inactive in light gray. The
// active coords are the first to leave zero as the lasso / enet penalty
// shrinks (sparse path geometry); ridge has every coord nonzero everywhere.
//
// fig_08_02 (adaptive lasso vs vanilla lasso) stays as a static <Figure> — it
// belongs structurally to §8.3 content (adaptive lasso oracle property), not
// the §8.4 paths comparison this component replaces.
//
// Compute budget: ~600 ms (Cholesky for ridge ≈ 250 ms at p=200; ISTA paths
// for lasso/enet ≈ 200 ms each with warm starts). One-time, cached in useMemo
// across tab switches.
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 16, bottom: 50, left: 56 };

const N = 150;
const P = 200;
const S = 10;
const SIGMA = 0.5;
const RHO = 0.5;
const SEED = 42;
const N_PATH = 30;
const ISTA_ITERS_FIRST = 200;
const ISTA_ITERS_WARM = 50;
const ENET_GAMMA = 0.5; // L2 weight in the elastic-net mix

const ACTIVE_COLORS = d3.schemeCategory10;
const SLATE = '#9CA3AF';
const SLATE_DARK = '#6B6B6B';

type Method = 'ridge' | 'lasso' | 'enet';

interface PathPoint {
  t: number; // x-axis value: α for ridge, λ for lasso/enet
  beta: Float64Array; // length P
}

interface PathBundle {
  method: Method;
  points: PathPoint[];
  xLabel: string;
}

// -----------------------------------------------------------------------------
// Ridge path: full-Cholesky on AᵀA + n α I at each α (p × p, expensive but
// p=200 keeps it under 300 ms total across N_PATH alphas).
// -----------------------------------------------------------------------------

function buildXtX(X: Float64Array[]): number[][] {
  const n = X.length;
  const p = X[0].length;
  const A: number[][] = [];
  for (let j = 0; j < p; j++) A.push(new Array<number>(p).fill(0));
  for (let j = 0; j < p; j++) {
    for (let k = 0; k <= j; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i][j] * X[i][k];
      A[j][k] = s;
      A[k][j] = s;
    }
  }
  return A;
}

function ridgePath(X: Float64Array[], y: Float64Array, alphas: number[]): PathPoint[] {
  const n = X.length;
  const p = X[0].length;
  const XtX = buildXtX(X);
  const Xty = Array.from(xtMul(X, y));
  const points: PathPoint[] = [];
  for (const alpha of alphas) {
    const A: number[][] = XtX.map((row) => row.slice());
    for (let j = 0; j < p; j++) A[j][j] += n * alpha;
    const L = choleskyFactor(A);
    const yt = solveLowerTriangular(L, Xty);
    const betaArr = solveUpperTriangularT(L, yt);
    const beta = new Float64Array(p);
    for (let j = 0; j < p; j++) beta[j] = betaArr[j];
    points.push({ t: alpha, beta });
  }
  return points;
}

// -----------------------------------------------------------------------------
// Lasso / elastic-net path: warm-started ISTA from large λ down to small λ.
// The shared `lassoIsta` doesn't take an enet γ, so this is an inline solver
// that handles both: at γ = 0 it's the lasso, at γ > 0 it's the elastic net.
//
// Update: β ← S(β + (η/n) Xᵀ(y - Xβ), η λ) / (1 + η γ)
// -----------------------------------------------------------------------------

function lassoEnetStep(
  X: Float64Array[],
  y: Float64Array,
  betaInit: Float64Array,
  lambda: number,
  gamma: number,
  L: number,
  iters: number,
): Float64Array {
  const n = X.length;
  const p = X[0].length;
  const eta = 1 / L;
  let beta = new Float64Array(betaInit);
  const divisor = 1 + eta * gamma;
  for (let it = 0; it < iters; it++) {
    const Xb = xMul(X, beta);
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) r[i] = y[i] - Xb[i];
    const grad = xtMul(X, r);
    const z = new Array<number>(p);
    for (let j = 0; j < p; j++) z[j] = beta[j] + (eta / n) * grad[j];
    const thr = softThreshold(z, eta * lambda);
    for (let j = 0; j < p; j++) thr[j] /= divisor;
    beta = new Float64Array(thr);
  }
  return beta;
}

function lassoOrEnetPath(
  X: Float64Array[],
  y: Float64Array,
  lambdas: number[],
  gamma: number,
  L: number,
): PathPoint[] {
  // Build descending λ-grid for warm starts.
  const desc = [...lambdas].sort((a, b) => b - a);
  const p = X[0].length;
  let prev = new Float64Array(p);
  const byLambda: Map<number, Float64Array> = new Map();
  for (let i = 0; i < desc.length; i++) {
    const iters = i === 0 ? ISTA_ITERS_FIRST : ISTA_ITERS_WARM;
    prev = lassoEnetStep(X, y, prev, desc[i], gamma, L, iters);
    byLambda.set(desc[i], new Float64Array(prev));
  }
  // Return in input order.
  return lambdas.map((t) => ({ t, beta: byLambda.get(t)! }));
}

// -----------------------------------------------------------------------------
// Compute all three paths once and cache.
// -----------------------------------------------------------------------------

function logspace(lo: number, hi: number, n: number): number[] {
  const a = Math.log(lo);
  const b = Math.log(hi);
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(Math.exp(a + ((b - a) * i) / (n - 1)));
  return out;
}

function computeBundles(): Record<Method, PathBundle> {
  const { X, y } = generateDgp1({ n: N, p: P, s: S, sigma: SIGMA, rho: RHO, seed: SEED });
  const L = operatorNorm(X);
  const lmax = lambdaMax(X, y);

  const ridgeAlphas = logspace(1e-3, 1e2, N_PATH);
  const lassoLambdas = logspace(lmax / 200, lmax * 1.05, N_PATH);
  const enetLambdas = logspace(lmax / 200, lmax * 1.05, N_PATH);

  return {
    ridge: { method: 'ridge', points: ridgePath(X, y, ridgeAlphas), xLabel: 'α (log scale)' },
    lasso: { method: 'lasso', points: lassoOrEnetPath(X, y, lassoLambdas, 0, L), xLabel: 'λ (log scale)' },
    enet: {
      method: 'enet',
      points: lassoOrEnetPath(X, y, enetLambdas, ENET_GAMMA, L),
      xLabel: `λ (log scale) — elastic net at γ = ${ENET_GAMMA}`,
    },
  };
}

// -----------------------------------------------------------------------------
// Component.
// -----------------------------------------------------------------------------

export default function RidgeLassoEnetPaths() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const bundles = useMemo(() => computeBundles(), []);
  const [method, setMethod] = useState<Method>('lasso');
  const bundle = bundles[method];

  // Global y-range across all three methods to keep visual scale comparable.
  const yMax = useMemo(() => {
    let m = 0;
    for (const b of Object.values(bundles)) {
      for (const pt of b.points) {
        for (let j = 0; j < P; j++) m = Math.max(m, Math.abs(pt.beta[j]));
      }
    }
    return m * 1.05;
  }, [bundles]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const tExtent = d3.extent(bundle.points, (p) => p.t) as [number, number];
      const xScale = d3.scaleLog().domain(tExtent).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerH, 0]);

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
        .text(bundle.xLabel);
      g.append('text')
        .attr('text-anchor', 'middle')
        .attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '13px')
        .text('β̂_j(t)');

      // Per-coord line generator (clean closure over j).
      const lineForCoord = (j: number) =>
        d3
          .line<PathPoint>()
          .x((pt) => xScale(pt.t))
          .y((pt) => yScale(pt.beta[j]));
      // Inactive coordinates first (gray, low opacity).
      for (let j = S; j < P; j++) {
        g.append('path')
          .datum(bundle.points)
          .attr('d', lineForCoord(j))
          .style('fill', 'none')
          .style('stroke', SLATE)
          .style('stroke-width', 0.8)
          .style('opacity', 0.35);
      }
      // Active coordinates on top (colored).
      for (let j = 0; j < S; j++) {
        const color = ACTIVE_COLORS[j % ACTIVE_COLORS.length];
        g.append('path')
          .datum(bundle.points)
          .attr('d', lineForCoord(j))
          .style('fill', 'none')
          .style('stroke', color)
          .style('stroke-width', 1.6);
      }

      // True-value reference line β* = 1 (active coords).
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(1))
        .attr('y2', yScale(1))
        .style('stroke', SLATE_DARK)
        .style('stroke-dasharray', '2 3')
        .style('stroke-width', 0.8)
        .style('opacity', 0.5);
      g.append('text')
        .attr('x', 6)
        .attr('y', yScale(1) - 4)
        .style('fill', SLATE_DARK)
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '10px')
        .style('opacity', 0.65)
        .text('β* = 1');

      // Zero line.
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-width', 0.5);
    },
    [bundle, yMax, w, isMobile],
  );

  // Count nonzero active + nonzero noise at the median penalty (for caption).
  const activeNonzero = useMemo(() => {
    const mid = Math.floor(bundle.points.length / 2);
    const beta = bundle.points[mid].beta;
    let actNZ = 0;
    let noiseNZ = 0;
    const tol = bundle.method === 'ridge' ? 0.05 : 0.01;
    for (let j = 0; j < P; j++) {
      if (Math.abs(beta[j]) > tol) {
        if (j < S) actNZ++;
        else noiseNZ++;
      }
    }
    return { actNZ, noiseNZ, midT: bundle.points[mid].t };
  }, [bundle]);

  const tabBtn = (m: Method, label: string) => (
    <button
      key={m}
      type="button"
      onClick={() => setMethod(m)}
      style={{
        padding: '6px 14px',
        marginRight: 6,
        marginBottom: 4,
        cursor: 'pointer',
        background: m === method ? 'var(--color-text)' : 'transparent',
        color: m === method ? 'var(--color-bg)' : 'var(--color-text)',
        border: '1px solid var(--color-border)',
        borderRadius: 4,
        fontFamily: 'var(--font-sans)',
        fontSize: 13,
        fontWeight: m === method ? 600 : 400,
      }}
      aria-pressed={m === method}
    >
      {label}
    </button>
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 12 }}>
        {tabBtn('ridge', 'Ridge')}
        {tabBtn('lasso', 'Lasso')}
        {tabBtn('enet', `Elastic net (γ = ${ENET_GAMMA})`)}
      </div>
      <svg
        ref={renderRef}
        width={w}
        height={HEIGHT}
        role="img"
        aria-label={`${method} coefficient path on DGP-1`}
      />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        Regularization paths β̂_j(t) on DGP-1 (n = {N}, p = {P}, s = {S}, σ = {SIGMA}, AR(1) ρ = {RHO}). True active coordinates (j &lt; {S}) drawn in distinct colors; {P - S} inactive coordinates in light gray. Dashed reference at β* = 1. At the median penalty t ≈ {activeNonzero.midT.toExponential(2)}, the {method} fit has {activeNonzero.actNZ}/{S} active and {activeNonzero.noiseNZ}/{P - S} noise coordinates nonzero. Ridge keeps every coefficient nonzero at every α; lasso and elastic net both reach exact zeros at moderate penalty — the active coords are the first to leave zero as the penalty shrinks. Switch tabs to compare path geometries directly.
      </p>
    </div>
  );
}

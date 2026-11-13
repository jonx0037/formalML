import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { GROUP_SIZES } from './shared/mixed-effects';
import { NOTEBOOK_SIX_CLASSROOM_ROWS } from '../../data/mixed-effects-data';

// =============================================================================
// MLvsREMLLikelihoodSurface — embedded after §4.4 of the mixed-effects topic.
// Two side-by-side contour heatmaps of the profile log-likelihoods over
// (log τ, log σ) — ML on the left, REML on the right. Each surface is
// evaluated on a 60×60 grid in TS; markers show the truth and each
// surface's optimum (grid argmax, polished by a Nelder–Mead search on the
// same closed-form objective). The pedagogical claim from §4.4 —
// "REML's optimum sits at slightly larger variance components than ML's"
// — becomes the visible offset between the two crosses.
//
// Data: the notebook's own 60 (x, y, classroom) rows, pinned in
// src/data/mixed-effects-data.ts, so the optima reproduce the article's
// §4.4 numbers — ML (τ̂², σ̂²) ≈ (2.14, 45.2), REML ≈ (3.30, 46.2). (The
// mulberry32 generator in shared/mixed-effects.ts draws a different sample.)
//
// Block-diagonal V structure makes this synchronous in browser. With
// V = τ²ZZᵀ + σ²I and Z a J=6 block indicator, V splits into six
// compound-symmetric blocks V_j = τ²1_n1_nᵀ + σ²I_n. Each block has
// closed-form eigenstructure:
//   eigenvalues: σ²+n_jτ² (mult 1, eigenvector 1/√n_j), σ² (mult n_j−1)
//   log det V_j = log(σ²+n_jτ²) + (n_j−1)·log σ²
//   V_j⁻¹y_j = (1/σ²)y_j − τ²/(σ²(σ²+n_jτ²)) · sum(y_j) · 1_n
// This makes per-evaluation cost O(J), so the 60×60 grid evaluates in well
// under 100 ms — no precompute pipeline needed.
//
// ML profile log-likelihood (eq. 4.3):
//   ℓ_ML(τ², σ²) = −½ [N log 2π + log|V| + (y − Xβ̂)ᵀV⁻¹(y − Xβ̂)]
// REML log-likelihood (eq. 4.4) adds the −½ log|XᵀV⁻¹X| correction.
// Additive constants are dropped; they don't move either optimum.
// =============================================================================

const PANEL_HEIGHT = 460;
const N_GRID = 60;
// (log τ, log σ) ranges. Cover the optima — ML τ̂ = √2.14 (log ≈ 0.38),
// REML τ̂ = √3.30 (log ≈ 0.60), σ̂ ≈ √46 ≈ 6.8 (log ≈ 1.92) — plus the truth
// at (log 5, log 8) ≈ (1.61, 2.08).
const LOG_TAU_DOMAIN: [number, number] = [-1.5, 3.0];
const LOG_SIGMA_DOMAIN: [number, number] = [0.5, 3.5];

const COLORS = {
  truth: '#000000',
  ml: '#dc2626',         // red — ML optimum
  reml: '#0891b2',       // cyan — REML optimum
  axis: '#374151',
};

interface SurfaceData {
  values: Float64Array; // length N_GRID*N_GRID, row-major: [logTauIdx][logSigmaIdx]
  min: number;
  max: number;
  argmaxIdx: number;
  optLogTau: number;
  optLogSigma: number;
}

// Per-classroom sufficient statistics: everything the closed-form
// log-likelihood needs, so each evaluation is O(J).
interface BlockSums {
  J: number;
  N: number;
  n: number[];
  sumX: Float64Array;
  sumY: Float64Array;
  sumXX: Float64Array;
  sumXY: Float64Array;
  sumYY: Float64Array;
}

function blockSums(rows: readonly { classroom: number; x: number; y: number }[]): BlockSums {
  const J = GROUP_SIZES.length;
  const n = new Array<number>(J).fill(0);
  const sumX = new Float64Array(J);
  const sumY = new Float64Array(J);
  const sumXX = new Float64Array(J);
  const sumXY = new Float64Array(J);
  const sumYY = new Float64Array(J);
  for (const { classroom: j, x, y } of rows) {
    n[j] += 1;
    sumX[j] += x;
    sumY[j] += y;
    sumXX[j] += x * x;
    sumXY[j] += x * y;
    sumYY[j] += y * y;
  }
  return { J, N: rows.length, n, sumX, sumY, sumXX, sumXY, sumYY };
}

// ML profile (reml = false) or REML (reml = true) log-likelihood at
// (log τ, log σ), up to an additive constant.
function logLik(s: BlockSums, logTau: number, logSigma: number, reml: boolean): number {
  const tauSq = Math.exp(2 * logTau);
  const sigmaSq = Math.exp(2 * logSigma);
  const inv1 = 1 / sigmaSq;
  const tauSqInv1 = tauSq * inv1;

  // log|V| via block eigenstructure; the (n_j − 1)·log σ² terms sum to (N − J)·log σ².
  let logDetV = (s.N - s.J) * Math.log(sigmaSq);
  // V_j⁻¹ = (1/σ²)I − c_j 1 1ᵀ with c_j = τ²/(σ²(σ² + n_j τ²)).
  // Accumulate XᵀV⁻¹X (2×2), XᵀV⁻¹y (2×1), yᵀV⁻¹y; X = [1, x].
  let a11 = 0;
  let a12 = 0;
  let a22 = 0;
  let b1 = 0;
  let b2 = 0;
  let yVy = 0;
  for (let j = 0; j < s.J; j++) {
    const nj = s.n[j];
    const spike = sigmaSq + nj * tauSq; // the large eigenvalue of V_j
    logDetV += Math.log(spike);
    const c = tauSqInv1 / spike;
    const sx = s.sumX[j];
    const sy = s.sumY[j];
    a11 += nj / spike;                       // 1ᵀV_j⁻¹1
    a12 += sx / spike;                       // 1ᵀV_j⁻¹x
    a22 += s.sumXX[j] * inv1 - c * sx * sx;  // xᵀV_j⁻¹x
    b1 += sy / spike;                        // 1ᵀV_j⁻¹y
    b2 += s.sumXY[j] * inv1 - c * sx * sy;   // xᵀV_j⁻¹y
    yVy += s.sumYY[j] * inv1 - c * sy * sy;  // yᵀV_j⁻¹y
  }

  // GLS β̂ = (XᵀV⁻¹X)⁻¹ XᵀV⁻¹y, then (y − Xβ̂)ᵀV⁻¹(y − Xβ̂) = yᵀV⁻¹y − β̂ᵀXᵀV⁻¹y.
  const det = a11 * a22 - a12 * a12;
  const beta1 = (a22 * b1 - a12 * b2) / det;
  const beta2 = (a11 * b2 - a12 * b1) / det;
  const rss = yVy - (beta1 * b1 + beta2 * b2);

  let ll = -0.5 * (logDetV + rss);
  if (reml) ll -= 0.5 * Math.log(det);
  return ll;
}

// Minimal 2-D Nelder–Mead (maximization) used to polish the grid argmax so
// the reported optimum isn't quantized to the grid spacing.
function nelderMeadMax(
  f: (p: [number, number]) => number,
  start: [number, number],
  step = 0.05,
  maxIter = 500,
  tol = 1e-12,
): [number, number] {
  let simplex: [number, number][] = [
    start,
    [start[0] + step, start[1]],
    [start[0], start[1] + step],
  ];
  let vals = simplex.map(f);
  for (let it = 0; it < maxIter; it++) {
    const order = [0, 1, 2].sort((i, k) => vals[k] - vals[i]); // best first
    simplex = order.map((i) => simplex[i]);
    vals = order.map((i) => vals[i]);
    if (Math.abs(vals[0] - vals[2]) < tol) break;
    const cx = (simplex[0][0] + simplex[1][0]) / 2;
    const cy = (simplex[0][1] + simplex[1][1]) / 2;
    const worst = simplex[2];
    const at = (t: number): [number, number] => [cx + t * (worst[0] - cx), cy + t * (worst[1] - cy)];
    const r = at(-1);
    const fr = f(r);
    if (fr > vals[0]) {
      const e = at(-2);
      const fe = f(e);
      if (fe > fr) {
        simplex[2] = e;
        vals[2] = fe;
      } else {
        simplex[2] = r;
        vals[2] = fr;
      }
    } else if (fr > vals[1]) {
      simplex[2] = r;
      vals[2] = fr;
    } else {
      const ct = fr > vals[2] ? at(-0.5) : at(0.5);
      const fc = f(ct);
      if (fc > Math.max(fr, vals[2])) {
        simplex[2] = ct;
        vals[2] = fc;
      } else {
        for (const i of [1, 2]) {
          simplex[i] = [
            (simplex[0][0] + simplex[i][0]) / 2,
            (simplex[0][1] + simplex[i][1]) / 2,
          ];
          vals[i] = f(simplex[i]);
        }
      }
    }
  }
  const best = vals.indexOf(Math.max(...vals));
  return simplex[best];
}

function evaluateSurface(s: BlockSums, reml: boolean): SurfaceData {
  const values = new Float64Array(N_GRID * N_GRID);
  let bestIdx = 0;
  let bestVal = -Infinity;

  for (let iTau = 0; iTau < N_GRID; iTau++) {
    const logTau =
      LOG_TAU_DOMAIN[0] + (iTau / (N_GRID - 1)) * (LOG_TAU_DOMAIN[1] - LOG_TAU_DOMAIN[0]);
    for (let iSigma = 0; iSigma < N_GRID; iSigma++) {
      const logSigma =
        LOG_SIGMA_DOMAIN[0] +
        (iSigma / (N_GRID - 1)) * (LOG_SIGMA_DOMAIN[1] - LOG_SIGMA_DOMAIN[0]);
      const ll = logLik(s, logTau, logSigma, reml);
      const idx = iTau * N_GRID + iSigma;
      values[idx] = ll;
      if (ll > bestVal) {
        bestVal = ll;
        bestIdx = idx;
      }
    }
  }

  // Compute min on a window near the max (tighter color range than the global min,
  // which can be very negative far from the mode).
  const max = bestVal;
  let min = max;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min && values[i] > max - 30) min = values[i];
  }

  // Polish the grid argmax with Nelder–Mead on the same closed-form objective.
  const gridLogTau =
    LOG_TAU_DOMAIN[0] +
    (Math.floor(bestIdx / N_GRID) / (N_GRID - 1)) * (LOG_TAU_DOMAIN[1] - LOG_TAU_DOMAIN[0]);
  const gridLogSigma =
    LOG_SIGMA_DOMAIN[0] +
    ((bestIdx % N_GRID) / (N_GRID - 1)) * (LOG_SIGMA_DOMAIN[1] - LOG_SIGMA_DOMAIN[0]);
  const [optLogTau, optLogSigma] = nelderMeadMax(
    ([lt, ls]) => logLik(s, lt, ls, reml),
    [gridLogTau, gridLogSigma],
  );

  return {
    values,
    min,
    max,
    argmaxIdx: bestIdx,
    optLogTau,
    optLogSigma,
  };
}

export default function MLvsREMLLikelihoodSurface() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [showTruth, setShowTruth] = useState<boolean>(true);
  const [showOpts, setShowOpts] = useState<boolean>(true);

  const { mlSurface, remlSurface, info } = useMemo(() => {
    const sums = blockSums(NOTEBOOK_SIX_CLASSROOM_ROWS);
    const ml = evaluateSurface(sums, false);
    const reml = evaluateSurface(sums, true);
    return {
      mlSurface: ml,
      remlSurface: reml,
      info: {
        N: sums.N,
        J: sums.J,
        // Truth from §4 prose: τ_t = 5, σ_t = 8.
        truthLogTau: Math.log(5),
        truthLogSigma: Math.log(8),
      },
    };
  }, []);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 36, right: 12, bottom: 44, left: 52 };
      const gap = 24;
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      const panelW = (innerW - gap) / 2;
      const sizeY = Math.min(innerH, panelW * 1.05);

      // Cell dimensions (square heatmap inside each panel).
      const cellW = panelW / N_GRID;
      const cellH = sizeY / N_GRID;

      const drawPanel = (
        offsetX: number,
        surface: SurfaceData,
        title: string,
        optColor: string,
      ) => {
        const root = svg
          .append('g')
          .attr('transform', `translate(${margin.left + offsetX},${margin.top})`);

        const xScale = d3.scaleLinear().domain(LOG_TAU_DOMAIN).range([0, panelW]);
        const yScale = d3.scaleLinear().domain(LOG_SIGMA_DOMAIN).range([sizeY, 0]);

        const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([surface.min, surface.max]);

        // Heatmap cells.
        for (let iTau = 0; iTau < N_GRID; iTau++) {
          for (let iSigma = 0; iSigma < N_GRID; iSigma++) {
            const v = surface.values[iTau * N_GRID + iSigma];
            const clamped = v < surface.min ? surface.min : v;
            root
              .append('rect')
              .attr('x', iTau * cellW)
              .attr('y', sizeY - (iSigma + 1) * cellH)
              .attr('width', cellW + 0.5)
              .attr('height', cellH + 0.5)
              .attr('fill', colorScale(clamped));
          }
        }

        // Axes.
        root
          .append('g')
          .attr('transform', `translate(0,${sizeY})`)
          .call(d3.axisBottom(xScale).ticks(5).tickSizeOuter(0));
        root.append('g').call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0));
        root
          .append('text')
          .attr('transform', `translate(${panelW / 2},${sizeY + 36})`)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .text('log τ');
        root
          .append('text')
          .attr('transform', `translate(-40,${sizeY / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .text('log σ');

        // Title.
        root
          .append('text')
          .attr('x', panelW / 2)
          .attr('y', -12)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .attr('font-weight', 600)
          .text(title);

        // Truth marker.
        if (showTruth) {
          const tx = xScale(info.truthLogTau);
          const ty = yScale(info.truthLogSigma);
          const r = 6;
          root
            .append('line')
            .attr('x1', tx - r)
            .attr('x2', tx + r)
            .attr('y1', ty - r)
            .attr('y2', ty + r)
            .attr('stroke', COLORS.truth)
            .attr('stroke-width', 2);
          root
            .append('line')
            .attr('x1', tx - r)
            .attr('x2', tx + r)
            .attr('y1', ty + r)
            .attr('y2', ty - r)
            .attr('stroke', COLORS.truth)
            .attr('stroke-width', 2);
        }

        // Optimum marker.
        if (showOpts) {
          const ox = xScale(surface.optLogTau);
          const oy = yScale(surface.optLogSigma);
          const r = 7;
          root
            .append('line')
            .attr('x1', ox - r)
            .attr('x2', ox + r)
            .attr('y1', oy)
            .attr('y2', oy)
            .attr('stroke', optColor)
            .attr('stroke-width', 2.4);
          root
            .append('line')
            .attr('x1', ox)
            .attr('x2', ox)
            .attr('y1', oy - r)
            .attr('y2', oy + r)
            .attr('stroke', optColor)
            .attr('stroke-width', 2.4);
          root
            .append('circle')
            .attr('cx', ox)
            .attr('cy', oy)
            .attr('r', 2.5)
            .attr('fill', optColor);
        }
      };

      drawPanel(0, mlSurface, 'ML profile log-likelihood ℓ_ML', COLORS.ml);
      drawPanel(panelW + gap, remlSurface, 'REML log-likelihood ℓ_REML', COLORS.reml);
    },
    [mlSurface, remlSurface, info, showTruth, showOpts, width],
  );

  const tauSqML = Math.exp(2 * mlSurface.optLogTau);
  const sigmaSqML = Math.exp(2 * mlSurface.optLogSigma);
  const tauSqREML = Math.exp(2 * remlSurface.optLogTau);
  const sigmaSqREML = Math.exp(2 * remlSurface.optLogSigma);

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showTruth}
            onChange={(e) => setShowTruth(e.target.checked)}
          />
          <span>truth × (τ=5, σ=8)</span>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showOpts}
            onChange={(e) => setShowOpts(e.target.checked)}
          />
          <span>optima +</span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-xs text-[var(--color-text-muted)]">
        <span>
          <strong style={{ color: COLORS.ml }}>ML optimum:</strong>{' '}
          <span className="tabular-nums">
            τ̂² = {tauSqML.toFixed(2)}, σ̂² = {sigmaSqML.toFixed(2)}
          </span>
        </span>
        <span>
          <strong style={{ color: COLORS.reml }}>REML optimum:</strong>{' '}
          <span className="tabular-nums">
            τ̂² = {tauSqREML.toFixed(2)}, σ̂² = {sigmaSqREML.toFixed(2)}
          </span>
        </span>
        <span>
          <strong>truth:</strong>{' '}
          <span className="tabular-nums">τ² = 25, σ² = 64</span>
        </span>
      </div>
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Both surfaces are evaluated on the notebook's six-classroom data
        (N = 60) over a 60×60 grid in (log τ, log σ), using the
        block-diagonal V eigenstructure for the random-intercept model; each
        cell evaluates ℓ_ML or ℓ_REML in closed form, and each optimum is the
        grid argmax polished by a Nelder–Mead search. The two optima sit at
        slightly different locations: REML's lies at larger variance
        components than ML's because the −½ log|XᵀV⁻¹X| correction in
        eq. (4.4) penalizes parameter combinations that produce large
        fixed-effects information, undoing ML's tendency to push σ² downward.
        On <em>this</em> p=2 realization, the gap is small but visible —
        on small-N high-p data it dominates and ML becomes seriously biased.
        The truth (black ×) sits above both optima: this realization
        under-shoots τ² badly and σ² modestly — sampling noise on J=6
        classrooms.
      </div>
    </div>
  );
}

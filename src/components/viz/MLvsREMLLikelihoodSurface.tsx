import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { generateSixClassrooms, GROUP_SIZES } from './shared/mixed-effects';

// =============================================================================
// MLvsREMLLikelihoodSurface — embedded after §4.4 of the mixed-effects topic.
// Two side-by-side contour heatmaps of the profile log-likelihoods over
// (log τ, log σ) — ML on the left, REML on the right. Each surface is
// evaluated on a 60×60 grid in TS; markers show the truth, the ML grid
// argmax, and the REML grid argmax. The pedagogical claim from §4.4 —
// "REML's optimum sits at slightly larger variance components than ML's"
// — becomes the visible offset between the two crosses.
//
// Block-diagonal V structure makes this synchronous in browser. With
// V = τ²ZZᵀ + σ²I and Z a J=6 block indicator, V splits into six
// compound-symmetric blocks V_j = τ²1_n1_nᵀ + σ²I_n. Each block has
// closed-form eigenstructure:
//   eigenvalues: σ²+n_jτ² (mult 1, eigenvector 1/√n_j), σ² (mult n_j−1)
//   log det V_j = log(σ²+n_jτ²) + (n_j−1)·log σ²
//   V_j⁻¹y_j = (1/σ²)y_j − τ²/(σ²(σ²+n_jτ²)) · sum(y_j) · 1_n
// This makes per-grid-point cost O(J), so the 60×60 grid evaluates in well
// under 100 ms — no precompute pipeline needed.
//
// ML profile log-likelihood (eq. 4.3):
//   ℓ_ML(τ², σ²) = −½ [N log 2π + log|V| + (y − Xβ̂)ᵀV⁻¹(y − Xβ̂)]
// REML log-likelihood (eq. 4.4) adds the −½ log|XᵀV⁻¹X| correction.
// =============================================================================

const PANEL_HEIGHT = 460;
const N_GRID = 60;
// (log τ, log σ) ranges. Cover the topic's REML optimum at √3.30 ≈ 1.82
// (log ≈ 0.6) for τ and √56 ≈ 7.5 (log ≈ 2.0) for σ, plus the truth at
// (log 5, log 8) ≈ (1.61, 2.08).
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

function evaluateSurface(
  classrooms: ReturnType<typeof generateSixClassrooms>['classrooms'],
  N: number,
  J: number,
  X1: Float64Array,
  X2: Float64Array,
  y: Float64Array,
  classroomIdx: Int32Array,
  reml: boolean,
): SurfaceData {
  const values = new Float64Array(N_GRID * N_GRID);
  let bestIdx = 0;
  let bestVal = -Infinity;

  // Pre-compute per-block sums needed at every grid point.
  const groupSizes = GROUP_SIZES;
  const blockSumY = new Float64Array(J);
  const blockSumX1 = new Float64Array(J);  // = n_j (since X1 is intercept = 1s)
  const blockSumX2 = new Float64Array(J);
  const blockSumY2 = new Float64Array(J);
  const blockSumX2X2 = new Float64Array(J);
  const blockSumX1X2 = new Float64Array(J);
  const blockSumX2Y = new Float64Array(J);
  const blockSumY1 = new Float64Array(J);
  for (let i = 0; i < N; i++) {
    const j = classroomIdx[i];
    blockSumY[j] += y[i];
    blockSumX1[j] += X1[i];
    blockSumX2[j] += X2[i];
    blockSumY2[j] += y[i] * y[i];
    blockSumX2X2[j] += X2[i] * X2[i];
    blockSumX1X2[j] += X1[i] * X2[i];
    blockSumX2Y[j] += X2[i] * y[i];
    blockSumY1[j] += y[i];
  }

  for (let iTau = 0; iTau < N_GRID; iTau++) {
    const logTau =
      LOG_TAU_DOMAIN[0] + (iTau / (N_GRID - 1)) * (LOG_TAU_DOMAIN[1] - LOG_TAU_DOMAIN[0]);
    const tauSq = Math.exp(2 * logTau);
    for (let iSigma = 0; iSigma < N_GRID; iSigma++) {
      const logSigma =
        LOG_SIGMA_DOMAIN[0] +
        (iSigma / (N_GRID - 1)) * (LOG_SIGMA_DOMAIN[1] - LOG_SIGMA_DOMAIN[0]);
      const sigmaSq = Math.exp(2 * logSigma);
      // Hoist constants out of the per-block loop. logSigmaSq and inv1 don't
      // depend on j; the (n_j - 1)·logSigmaSq term sums to (N - J)·logSigmaSq.
      const logSigmaSq = Math.log(sigmaSq);
      const inv1 = 1 / sigmaSq;
      const tauSqInv1 = tauSq * inv1;

      // log|V| via block eigenstructure.
      let logDetV = (N - J) * logSigmaSq;
      // Per-block coefficients for V⁻¹: V_j⁻¹ = (1/σ²)I − c_j 1 1ᵀ
      // where c_j = τ²/(σ²(σ² + n_j τ²)).
      // Need X^T V⁻¹ X (2×2) and X^T V⁻¹ y (2×1).
      let xtVx_11 = 0;
      let xtVx_12 = 0;
      let xtVx_22 = 0;
      let xtVy_1 = 0;
      let xtVy_2 = 0;
      let yVy = 0;

      for (let j = 0; j < J; j++) {
        const nj = groupSizes[j];
        const lambdaSpike = sigmaSq + nj * tauSq; // big-eigenvalue
        logDetV += Math.log(lambdaSpike);

        // Per-block sums precomputed.
        const sumY = blockSumY[j];
        const sumX1 = blockSumX1[j]; // = n_j
        const sumX2 = blockSumX2[j];

        // c_j = τ²/(σ² · λ_spike)
        const c = tauSqInv1 / lambdaSpike;

        // X1ᵀ V_j⁻¹ X1 = n_j/σ² − c·n_j² = n_j/λ_spike
        xtVx_11 += nj / lambdaSpike;
        // X1ᵀ V_j⁻¹ X2 = sumX2/σ² − c·n_j·sumX2 = sumX2/λ_spike
        xtVx_12 += sumX2 / lambdaSpike;
        // X2ᵀ V_j⁻¹ X2 = sumX2² /σ² − c·sumX2² + (X2ᵀX2 − sumX2²/n_j)/σ²
        // Actually carefully: V_j⁻¹ x2_j = (1/σ²)x2_j − c·sumX2·1
        //   so x2_jᵀ V_j⁻¹ x2_j = (1/σ²)Σx² − c·(sumX2)²
        const sumX2X2 = blockSumX2X2[j];
        xtVx_22 += sumX2X2 * inv1 - c * sumX2 * sumX2;

        // X1ᵀ V_j⁻¹ y = sumY/σ² − c·n_j·sumY = sumY/λ_spike
        xtVy_1 += sumY / lambdaSpike;
        // X2ᵀ V_j⁻¹ y = (1/σ²)Σx2y − c·sumX2·sumY
        xtVy_2 += blockSumX2Y[j] * inv1 - c * sumX2 * sumY;

        // yᵀ V_j⁻¹ y = (1/σ²)Σy² − c·sumY²
        yVy += blockSumY2[j] * inv1 - c * sumY * sumY;
      }

      // Solve (XᵀV⁻¹X) β̂ = XᵀV⁻¹y for the 2×2 system.
      const det = xtVx_11 * xtVx_22 - xtVx_12 * xtVx_12;
      const beta1 = (xtVx_22 * xtVy_1 - xtVx_12 * xtVy_2) / det;
      const beta2 = (xtVx_11 * xtVy_2 - xtVx_12 * xtVy_1) / det;

      // (y − Xβ̂)ᵀ V⁻¹ (y − Xβ̂) = yᵀV⁻¹y − 2 βᵀ XᵀV⁻¹y + βᵀ XᵀV⁻¹X β
      const beta_xtVy = beta1 * xtVy_1 + beta2 * xtVy_2;
      const beta_xtVx_beta =
        beta1 * (xtVx_11 * beta1 + xtVx_12 * beta2) +
        beta2 * (xtVx_12 * beta1 + xtVx_22 * beta2);
      const rss = yVy - 2 * beta_xtVy + beta_xtVx_beta;

      // ML: ℓ_ML = −½ [N log 2π + log|V| + rss]
      // REML: ℓ_REML = −½ [(N − p) log 2π + log|V| + log|XᵀV⁻¹X| + rss]
      let ll = -0.5 * (logDetV + rss);
      if (reml) {
        const logDetA = Math.log(det);
        ll -= 0.5 * logDetA;
      }

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
  let max = bestVal;
  let min = max;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min && values[i] > max - 30) min = values[i];
  }

  const optTauIdx = Math.floor(bestIdx / N_GRID);
  const optSigmaIdx = bestIdx % N_GRID;
  const optLogTau =
    LOG_TAU_DOMAIN[0] + (optTauIdx / (N_GRID - 1)) * (LOG_TAU_DOMAIN[1] - LOG_TAU_DOMAIN[0]);
  const optLogSigma =
    LOG_SIGMA_DOMAIN[0] +
    (optSigmaIdx / (N_GRID - 1)) * (LOG_SIGMA_DOMAIN[1] - LOG_SIGMA_DOMAIN[0]);

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
    const data = generateSixClassrooms();
    const N = data.N;
    const J = GROUP_SIZES.length;
    const X1 = new Float64Array(N).fill(1);
    const X2 = new Float64Array(N);
    const y = new Float64Array(N);
    const classroomIdx = new Int32Array(N);
    let idx = 0;
    for (let j = 0; j < J; j++) {
      const c = data.classrooms[j];
      for (let i = 0; i < c.nj; i++) {
        X2[idx] = c.x[i];
        y[idx] = c.y[i];
        classroomIdx[idx] = j;
        idx++;
      }
    }
    const ml = evaluateSurface(data.classrooms, N, J, X1, X2, y, classroomIdx, false);
    const reml = evaluateSurface(data.classrooms, N, J, X1, X2, y, classroomIdx, true);
    return {
      mlSurface: ml,
      remlSurface: reml,
      info: {
        N,
        J,
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
          <strong style={{ color: COLORS.ml }}>ML grid argmax:</strong>{' '}
          <span className="tabular-nums">
            τ̂² = {tauSqML.toFixed(2)}, σ̂² = {sigmaSqML.toFixed(2)}
          </span>
        </span>
        <span>
          <strong style={{ color: COLORS.reml }}>REML grid argmax:</strong>{' '}
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
        Both surfaces are evaluated on a 60×60 grid in (log τ, log σ) using
        the block-diagonal V eigenstructure for the random-intercept model;
        each cell evaluates ℓ_ML or ℓ_REML in closed form. The two optima
        sit at slightly different locations: REML's lies at larger variance
        components than ML's because the −½ log|XᵀV⁻¹X| correction in
        eq. (4.4) penalizes parameter combinations that produce large
        fixed-effects information, undoing ML's tendency to push σ² downward.
        On <em>this</em> p=2 realization, the gap is small but visible —
        on small-N high-p data it dominates and ML becomes seriously biased.
        The truth (black ×) sits between the two optima, slightly above
        both — sampling noise on J=6 classrooms.
      </div>
    </div>
  );
}

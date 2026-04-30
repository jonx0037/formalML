import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// PoolingSpectrumExplorer — embedded after §1.2 of the mixed-effects topic.
// One panel: scatter of N=60 student-level (prep-hours, exam-score) pairs
// across J=6 classrooms with sizes (4, 6, 8, 10, 12, 20), color-coded.
// Six classroom lines share a slope and use intercepts that morph between
// no-pooling (per-classroom) and complete-pooling (overall mean) under the
// random-intercept BLUP shrinkage rule
//   λⱼ = τ²·nⱼ / (τ²·nⱼ + σ²),
// with σ² fixed at the within-group residual variance and τ² controlled by a
// slider. At τ² = 0 the six lines collapse onto the population-mean line
// (complete pooling); at τ² → ∞ each line returns to its OLS fit (no pooling).
// The default τ²/σ² ≈ 0.07 reproduces the REML shrinkage factors (~0.22 for
// n=4, ~0.59 for n=20) the topic reports at line 249. The size-dependent
// drift — small classrooms collapsing hard, large classrooms barely moving —
// is the §1.2 pedagogical payoff before §3 names the BLUP formula.
// =============================================================================

const PANEL_HEIGHT = 380;
const X_DOMAIN: [number, number] = [0, 5];
const GROUP_SIZES = [4, 6, 8, 10, 12, 20] as const;
const ALPHA_TRUE = 50;
const BETA_TRUE = 5;
const TAU_TRUE = 5;
const SIGMA_TRUE = 8;
const SEED = 20260429;

// Six color-blind-friendly hues for the six classrooms.
const PALETTE = [
  '#1f77b4', // blue
  '#ff7f0e', // orange
  '#2ca02c', // green
  '#d62728', // red
  '#9467bd', // purple
  '#8c564b', // brown
];

// Mulberry32 — small, fast, deterministic 32-bit PRNG. Good enough for synthetic
// classroom data; exact byte-equality with NumPy isn't required because the
// qualitative shape (group sizes, score range, slope) carries the pedagogy.
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

// Box–Muller standard-normal sampler from a uniform PRNG.
function gaussianSampler(rng: () => number): () => number {
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = rng();
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

interface ClassroomData {
  j: number;        // classroom index 0..5
  nj: number;       // group size
  x: Float64Array;  // prep-hours
  y: Float64Array;  // exam scores
  xMean: number;    // x̄_j
  yMean: number;    // ȳ_j
}

interface SyntheticDataset {
  classrooms: ClassroomData[];
  N: number;        // total observations
  betaWithin: number; // shared within-group slope (no-pooling estimate)
  alphaPool: number;  // population-mean intercept under shared slope
  sigma2Hat: number;  // pooled within-group residual variance estimate
  alphasNoPool: Float64Array; // per-classroom OLS intercepts
}

function generateSixClassrooms(): SyntheticDataset {
  const rng = mulberry32(SEED);
  const sample = gaussianSampler(rng);
  const J = GROUP_SIZES.length;

  // Sample classroom random intercepts a_j ~ N(0, τ²).
  const aTrue = new Float64Array(J);
  for (let j = 0; j < J; j++) aTrue[j] = TAU_TRUE * sample();

  const classrooms: ClassroomData[] = [];
  let N = 0;
  for (let j = 0; j < J; j++) {
    const nj = GROUP_SIZES[j];
    const x = new Float64Array(nj);
    const y = new Float64Array(nj);
    for (let i = 0; i < nj; i++) {
      x[i] = X_DOMAIN[0] + rng() * (X_DOMAIN[1] - X_DOMAIN[0]);
      y[i] = ALPHA_TRUE + BETA_TRUE * x[i] + aTrue[j] + SIGMA_TRUE * sample();
    }
    let xs = 0;
    let ys = 0;
    for (let i = 0; i < nj; i++) {
      xs += x[i];
      ys += y[i];
    }
    classrooms.push({
      j,
      nj,
      x,
      y,
      xMean: xs / nj,
      yMean: ys / nj,
    });
    N += nj;
  }

  // Within-group OLS: β̂ = Σⱼᵢ (x − x̄ⱼ)(y − ȳⱼ) / Σⱼᵢ (x − x̄ⱼ)².
  let sxy = 0;
  let sxx = 0;
  for (const c of classrooms) {
    for (let i = 0; i < c.nj; i++) {
      const dx = c.x[i] - c.xMean;
      const dy = c.y[i] - c.yMean;
      sxy += dx * dy;
      sxx += dx * dx;
    }
  }
  const betaWithin = sxy / sxx;

  // No-pooling intercepts: α̂ⱼ = ȳⱼ − β̂·x̄ⱼ.
  const alphasNoPool = new Float64Array(J);
  for (let j = 0; j < J; j++) {
    const c = classrooms[j];
    alphasNoPool[j] = c.yMean - betaWithin * c.xMean;
  }

  // Population-mean intercept under the shared slope.
  let xBar = 0;
  let yBar = 0;
  for (const c of classrooms) {
    for (let i = 0; i < c.nj; i++) {
      xBar += c.x[i];
      yBar += c.y[i];
    }
  }
  xBar /= N;
  yBar /= N;
  const alphaPool = yBar - betaWithin * xBar;

  // Pooled within-group residual variance σ̂² = Σ (yᵢⱼ − α̂ⱼ − β̂·xᵢⱼ)² / (N − J − 1).
  let rss = 0;
  for (let j = 0; j < J; j++) {
    const c = classrooms[j];
    for (let i = 0; i < c.nj; i++) {
      const r = c.y[i] - alphasNoPool[j] - betaWithin * c.x[i];
      rss += r * r;
    }
  }
  const sigma2Hat = rss / (N - J - 1);

  return { classrooms, N, betaWithin, alphaPool, sigma2Hat, alphasNoPool };
}

export default function PoolingSpectrumExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  // τ²/σ² slider — log-space for fine resolution near the REML estimate.
  // u ∈ [0, 1] maps to τ²/σ² ∈ [0, 5] via u² so most of the screen real estate
  // covers the pedagogically interesting [0, 1] range.
  const [u, setU] = useState<number>(Math.sqrt(0.07 / 5));
  const [showPopulationLine, setShowPopulationLine] = useState<boolean>(true);

  const data = useMemo(() => generateSixClassrooms(), []);

  const tauSqOverSigmaSq = useMemo<number>(() => 5 * u * u, [u]);

  // BLUP shrinkage factors λⱼ = τ²·nⱼ / (τ²·nⱼ + σ²) = ρ·nⱼ / (ρ·nⱼ + 1).
  const lambdas = useMemo<Float64Array>(() => {
    const J = data.classrooms.length;
    const out = new Float64Array(J);
    const rho = tauSqOverSigmaSq;
    for (let j = 0; j < J; j++) {
      const n = data.classrooms[j].nj;
      out[j] = (rho * n) / (rho * n + 1);
    }
    return out;
  }, [data, tauSqOverSigmaSq]);

  // Partial-pooled intercept: α̃ⱼ = λⱼ · α̂ⱼ^(no-pool) + (1 − λⱼ) · α̂^(pool).
  const alphasPartial = useMemo<Float64Array>(() => {
    const J = data.classrooms.length;
    const out = new Float64Array(J);
    for (let j = 0; j < J; j++) {
      out[j] =
        lambdas[j] * data.alphasNoPool[j] + (1 - lambdas[j]) * data.alphaPool;
    }
    return out;
  }, [data, lambdas]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 20, right: 20, bottom: 48, left: 52 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;

      // Compute y-domain from data + line endpoints to avoid clipping.
      let yMin = Infinity;
      let yMax = -Infinity;
      for (const c of data.classrooms) {
        for (let i = 0; i < c.nj; i++) {
          if (c.y[i] < yMin) yMin = c.y[i];
          if (c.y[i] > yMax) yMax = c.y[i];
        }
      }
      const yPad = (yMax - yMin) * 0.06;
      const yDomain: [number, number] = [yMin - yPad, yMax + yPad];

      const xScale = d3.scaleLinear().domain(X_DOMAIN).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(yDomain).range([innerH, 0]);

      const root = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      root
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
      root.append('g').call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));

      root
        .append('text')
        .attr('transform', `translate(${innerW / 2},${innerH + 36})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('prep-hours index');

      root
        .append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('exam score');

      // Population-mean line (complete-pooling reference).
      if (showPopulationLine) {
        const x0 = X_DOMAIN[0];
        const x1 = X_DOMAIN[1];
        root
          .append('line')
          .attr('x1', xScale(x0))
          .attr('x2', xScale(x1))
          .attr('y1', yScale(data.alphaPool + data.betaWithin * x0))
          .attr('y2', yScale(data.alphaPool + data.betaWithin * x1))
          .attr('stroke', '#374151')
          .attr('stroke-dasharray', '3 4')
          .attr('stroke-width', 1.2)
          .attr('opacity', 0.85);
      }

      // Six per-classroom partial-pooled lines.
      for (let j = 0; j < data.classrooms.length; j++) {
        const c = data.classrooms[j];
        const a = alphasPartial[j];
        const b = data.betaWithin;
        const x0 = X_DOMAIN[0];
        const x1 = X_DOMAIN[1];
        root
          .append('line')
          .attr('x1', xScale(x0))
          .attr('x2', xScale(x1))
          .attr('y1', yScale(a + b * x0))
          .attr('y2', yScale(a + b * x1))
          .attr('stroke', PALETTE[j])
          .attr('stroke-width', 2.2)
          .attr('opacity', 0.92);
      }

      // Scatter points, colored by classroom.
      for (let j = 0; j < data.classrooms.length; j++) {
        const c = data.classrooms[j];
        const grp = root.append('g');
        for (let i = 0; i < c.nj; i++) {
          grp
            .append('circle')
            .attr('cx', xScale(c.x[i]))
            .attr('cy', yScale(c.y[i]))
            .attr('r', 3.2)
            .attr('fill', PALETTE[j])
            .attr('opacity', 0.7);
        }
      }

      // Legend with classroom sizes and current shrinkage factors.
      const legend = root.append('g').attr('transform', `translate(${innerW - 200},10)`);
      legend
        .append('text')
        .attr('x', 0)
        .attr('y', 0)
        .attr('font-size', 10)
        .attr('font-weight', 600)
        .text('Classroom (nⱼ, λⱼ)');
      data.classrooms.forEach((c, j) => {
        const row = legend.append('g').attr('transform', `translate(0,${10 + (j + 1) * 13})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 18)
          .attr('y1', 4)
          .attr('y2', 4)
          .attr('stroke', PALETTE[j])
          .attr('stroke-width', 2.2);
        row
          .append('text')
          .attr('x', 24)
          .attr('y', 7)
          .attr('font-size', 10)
          .text(`n = ${c.nj}, λ = ${lambdas[j].toFixed(2)}`);
      });
    },
    [data, alphasPartial, lambdas, showPopulationLine, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          pooling strength:
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={u}
            onChange={(e) => setU(Number(e.target.value))}
            className="w-40"
            aria-label="pooling strength slider"
          />
          <span className="tabular-nums w-24 text-right text-xs text-[var(--color-text-muted)]">
            τ²/σ² = {tauSqOverSigmaSq.toFixed(3)}
          </span>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showPopulationLine}
            onChange={(e) => setShowPopulationLine(e.target.checked)}
          />
          <span>Show population-mean line</span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Six classrooms, sizes (4, 6, 8, 10, 12, 20). Slide pooling strength
        toward 0 — the classroom-level lines collapse onto the dashed
        population-mean line (complete pooling). Slide it toward 1 — each line
        returns to the OLS fit through its own data (no pooling). Anywhere in
        between, the smallest classroom (n = 4) shrinks toward the dashed line
        much faster than the largest (n = 20) because λⱼ depends on group
        size. The default position reproduces the topic's REML shrinkage
        factors (~0.22 for n = 4, rising to ~0.59 for n = 20).
      </div>
    </div>
  );
}

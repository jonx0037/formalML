import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// MeanFieldStructuralError — embedded after Theorem 5 in §5.1 of the
// variational-inference topic. Visualizes the structural error of mean-field
// VI on the cleanest case: a 2D zero-mean Gaussian target with correlation ρ.
//
// The target is N(0, Σ) with Σ = [[1, ρ], [ρ, 1]]. Three families overlap on
// one panel:
//   - True target: a tilted ellipse with axes (1+ρ, 1−ρ) — the diagonal
//     is filled with light blue scatter samples from N=1500 Cholesky draws.
//   - Mean-field projection: an axis-aligned ellipse with both axes equal
//     to √(1−ρ²) (Theorem 5's reverse-KL projection: marginal variance
//     shrinks from Σⱼⱼ=1 to 1/(Σ⁻¹)ⱼⱼ = 1−ρ²).
//   - Full-rank projection: matches the target exactly (§5.2 preview —
//     dashed green ellipse on top of the true contour).
//
// Slider for ρ ∈ [0, 0.95]; default 0.85 (topic's reference value at line
// 739). Toggle to show/hide full-rank overlay. As ρ increases the orange
// mean-field ellipse shrinks isotropically while the true ellipse tilts —
// the gap between them is the §5.1 structural error of mean-field.
//
// Closed-form throughout: Cholesky-based sampling from the target plus
// 95% confidence-ellipse parameters via eigendecomposition of Σ. No
// precompute needed despite Tier-2 plan's expectation.
// =============================================================================

const PANEL_HEIGHT = 460;
const N_SAMPLES = 1500;
const VIEW_DOMAIN: [number, number] = [-3, 3];
// 95% confidence ellipse for 2D Gaussian: r = √(χ²₂,0.95) ≈ √5.991.
const CHI2_95 = 2.447746830680816;

const COLORS = {
  truth: '#2563eb',     // blue   — true target (samples + ellipse)
  meanfield: '#ea580c', // orange — mean-field reverse-KL projection
  fullrank: '#15803d',  // green  — full-rank Gaussian VI (matches truth)
  axis: '#374151',
  guide: '#9ca3af',
};

// Box–Muller standard normal generator from a deterministic LCG.
function makeNormalSampler(seed: number) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = next();
    const u2 = next();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

export default function MeanFieldStructuralError() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [rho, setRho] = useState<number>(0.85);
  const [showFullRank, setShowFullRank] = useState<boolean>(true);

  // Independent N(0,1) base samples — fixed once so the scatter is stable
  // when ρ slides. Cholesky-shaped per ρ in `samples`.
  const baseEpsilons = useMemo<{ e1: Float64Array; e2: Float64Array }>(() => {
    const sample = makeNormalSampler(20260430);
    const e1 = new Float64Array(N_SAMPLES);
    const e2 = new Float64Array(N_SAMPLES);
    for (let i = 0; i < N_SAMPLES; i++) {
      e1[i] = sample();
      e2[i] = sample();
    }
    return { e1, e2 };
  }, []);

  // Cholesky of [[1, ρ], [ρ, 1]] is [[1, 0], [ρ, √(1−ρ²)]].
  // Sample θ = L · ε.
  const samples = useMemo<{ x: Float64Array; y: Float64Array }>(() => {
    const x = new Float64Array(N_SAMPLES);
    const y = new Float64Array(N_SAMPLES);
    const r = Math.sqrt(Math.max(1 - rho * rho, 0));
    for (let i = 0; i < N_SAMPLES; i++) {
      const e1 = baseEpsilons.e1[i];
      const e2 = baseEpsilons.e2[i];
      x[i] = e1;
      y[i] = rho * e1 + r * e2;
    }
    return { x, y };
  }, [baseEpsilons, rho]);

  // Mean-field marginal variance: 1 − ρ² (Theorem 5).
  const meanFieldVar = useMemo<number>(() => 1 - rho * rho, [rho]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 20, right: 20, bottom: 44, left: 52 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      // Square panel — equal aspect for the 2D scatter, no skew.
      const size = Math.min(innerW, innerH);
      const pixelsPerUnit = size / (VIEW_DOMAIN[1] - VIEW_DOMAIN[0]);

      const root = svg
        .append('g')
        .attr(
          'transform',
          `translate(${margin.left + Math.max(0, (innerW - size) / 2)},${margin.top})`,
        );

      const xScale = d3.scaleLinear().domain(VIEW_DOMAIN).range([0, size]);
      const yScale = d3.scaleLinear().domain(VIEW_DOMAIN).range([size, 0]);

      root
        .append('g')
        .attr('transform', `translate(0,${size})`)
        .call(d3.axisBottom(xScale).ticks(7).tickSizeOuter(0));
      root.append('g').call(d3.axisLeft(yScale).ticks(7).tickSizeOuter(0));
      root
        .append('text')
        .attr('transform', `translate(${size / 2},${size + 36})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('θ₁');
      root
        .append('text')
        .attr('transform', `translate(-40,${size / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('θ₂');

      // Reference cross at origin.
      root
        .append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(0))
        .attr('y1', 0)
        .attr('y2', size)
        .attr('stroke', COLORS.guide)
        .attr('stroke-dasharray', '2 4')
        .attr('stroke-width', 0.6);
      root
        .append('line')
        .attr('x1', 0)
        .attr('x2', size)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .attr('stroke', COLORS.guide)
        .attr('stroke-dasharray', '2 4')
        .attr('stroke-width', 0.6);

      // True target samples (N=1500). Skip points outside view.
      const sampleGroup = root.append('g');
      for (let i = 0; i < N_SAMPLES; i++) {
        const xv = samples.x[i];
        const yv = samples.y[i];
        if (xv < VIEW_DOMAIN[0] || xv > VIEW_DOMAIN[1]) continue;
        if (yv < VIEW_DOMAIN[0] || yv > VIEW_DOMAIN[1]) continue;
        sampleGroup
          .append('circle')
          .attr('cx', xScale(xv))
          .attr('cy', yScale(yv))
          .attr('r', 1.4)
          .attr('fill', COLORS.truth)
          .attr('opacity', 0.32);
      }

      // ── Confidence ellipses ────────────────────────────────────────────
      // True target N(0, Σ) with Σ = [[1, ρ], [ρ, 1]]:
      // eigenvalues are (1+ρ, 1−ρ); eigenvectors at ±45°.
      const lambdaPlus = 1 + rho;
      const lambdaMinus = 1 - rho;
      const ellipseTrueRx = CHI2_95 * Math.sqrt(lambdaPlus) * pixelsPerUnit;
      const ellipseTrueRy = CHI2_95 * Math.sqrt(lambdaMinus) * pixelsPerUnit;

      root
        .append('ellipse')
        .attr('cx', xScale(0))
        .attr('cy', yScale(0))
        .attr('rx', ellipseTrueRx)
        .attr('ry', ellipseTrueRy)
        .attr('fill', 'none')
        .attr('stroke', COLORS.truth)
        .attr('stroke-width', 2)
        .attr('transform', `rotate(-45 ${xScale(0)} ${yScale(0)})`);

      // Mean-field projection: axis-aligned, both axes √(1−ρ²).
      const ellipseMfR = CHI2_95 * Math.sqrt(meanFieldVar) * pixelsPerUnit;
      root
        .append('ellipse')
        .attr('cx', xScale(0))
        .attr('cy', yScale(0))
        .attr('rx', ellipseMfR)
        .attr('ry', ellipseMfR)
        .attr('fill', 'none')
        .attr('stroke', COLORS.meanfield)
        .attr('stroke-width', 2.4);

      // Full-rank projection: matches truth exactly.
      if (showFullRank) {
        root
          .append('ellipse')
          .attr('cx', xScale(0))
          .attr('cy', yScale(0))
          .attr('rx', ellipseTrueRx)
          .attr('ry', ellipseTrueRy)
          .attr('fill', 'none')
          .attr('stroke', COLORS.fullrank)
          .attr('stroke-dasharray', '6 4')
          .attr('stroke-width', 1.6)
          .attr(
            'transform',
            `rotate(-45 ${xScale(0)} ${yScale(0)})`,
          );
      }

      // Legend.
      const legend = root.append('g').attr('transform', `translate(${size - 240},10)`);
      const items = [
        { label: 'Target N(0, Σ) — samples + 95% ellipse', color: COLORS.truth, dash: false },
        { label: 'Mean-field reverse-KL projection', color: COLORS.meanfield, dash: false },
      ];
      if (showFullRank) {
        items.push({ label: 'Full-rank Gaussian VI (matches truth)', color: COLORS.fullrank, dash: true });
      }
      items.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 24)
          .attr('y1', 8)
          .attr('y2', 8)
          .attr('stroke', it.color)
          .attr('stroke-width', it.dash ? 1.6 : 2.2)
          .attr('stroke-dasharray', it.dash ? '6 4' : null);
        row
          .append('text')
          .attr('x', 30)
          .attr('y', 11)
          .attr('font-size', 10)
          .text(it.label);
      });
    },
    [samples, meanFieldVar, rho, showFullRank, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          correlation ρ:
          <input
            type="range"
            min={0}
            max={0.95}
            step={0.01}
            value={rho}
            onChange={(e) => setRho(Number(e.target.value))}
            className="w-40"
            aria-label="correlation slider"
          />
          <span className="tabular-nums w-10 text-right text-xs text-[var(--color-text-muted)]">
            {rho.toFixed(2)}
          </span>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showFullRank}
            onChange={(e) => setShowFullRank(e.target.checked)}
          />
          <span>Show full-rank ellipse</span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-xs text-[var(--color-text-muted)]">
        <span>
          <strong style={{ color: COLORS.truth }}>true Var(θⱼ):</strong>{' '}
          <span className="tabular-nums">1.000</span>
        </span>
        <span>
          <strong style={{ color: COLORS.meanfield }}>mean-field Var(θⱼ):</strong>{' '}
          <span className="tabular-nums">{meanFieldVar.toFixed(3)}</span>{' '}
          (= 1 − ρ²)
        </span>
        <span>
          <strong>shrinkage factor:</strong>{' '}
          <span className="tabular-nums">{meanFieldVar.toFixed(3)}×</span>
        </span>
      </div>
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Target: N(0, Σ) with Σ = [[1, ρ], [ρ, 1]] — the blue dots are 1500
        Cholesky samples and the blue ellipse is the 95% confidence region.
        The orange circle is the reverse-KL mean-field projection from
        Theorem 5: an axis-aligned Gaussian whose marginal variance is
        1/(Σ⁻¹)ⱼⱼ = 1 − ρ², strictly smaller than the true marginal variance
        Σⱼⱼ = 1 whenever ρ ≠ 0. The dashed green ellipse is full-rank
        Gaussian VI (§5.2) — for a Gaussian target it sits exactly on top
        of the truth. Drag ρ to 0.95 and the orange circle shrinks to ~10%
        of the true contour's footprint while the blue ellipse tilts: that
        gap is the structural error a credible interval read off the
        mean-field posterior would inherit.
      </div>
    </div>
  );
}

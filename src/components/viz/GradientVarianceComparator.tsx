import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// GradientVarianceComparator — embedded after §4.4 of the variational-inference
// topic. Demonstrates Theorem 3 (score-function gradient) and Theorem 4
// (reparameterization gradient) on the simplest test case where both apply
// and the variance gap is closed-form: target p = N(0, 1), variational
// q_φ = N(μ, σ²), gradient w.r.t. μ.
//
// True ∇_μ ELBO at this q is exactly −μ. The viz computes K=300 independent
// estimates of ∇_μ ELBO from each estimator at MC sample size S, plots
// them as overlapping histograms, and renders the §4.4 line 668 claim
// "the second has variance one to several orders of magnitude lower than
// the first" as a visible width gap between the orange (score-function)
// and blue (reparameterization) histograms.
//
// One-sample formulas at θ = μ + σε with ε ∼ N(0, 1):
//   SF:      ĝ_SF      = (log p(θ) − log q(θ)) · (θ − μ)/σ²
//                       = (−½θ² + ½ε² + log σ + const) · ε/σ
//   Reparam: ĝ_reparam = ∇_θ(log p − log q) at θ = μ + σε
//                       = (−θ) − (−ε/σ) + (explicit ∂_μ log q with mean 0)
//                       = −θ        (after the score-zero simplification)
//
// Both center on −μ (confirmed: E[ĝ_SF] = E[ĝ_reparam] = −μ); per-sample
// variance is wildly different.
//
// Sliders: μ ∈ [−2, 2], log σ ∈ [−1, 1.5], S (MC sample size) ∈ [1, 200]. As
// S grows, both histograms narrow; as σ moves away from 1, the SF histogram
// inflates faster than the reparam one. Closed-form, no precompute despite
// the Tier 3 plan's "5 sample sizes × 10 seeds" estimate.
// =============================================================================

const PANEL_HEIGHT = 380;
const N_REPLICATIONS = 300;
const N_BINS = 36;
// Plot range for the gradient estimate axis.
const G_VIEW_DOMAIN: [number, number] = [-6, 6];

const COLORS = {
  scoreFn: '#ea580c',     // orange — high-variance score-function estimator
  reparam: '#2563eb',     // blue   — low-variance reparameterization estimator
  truth: '#111827',       // near-black — true ∇_μ ELBO
  axis: '#374151',
};

// Mulberry32 + Box–Muller — deterministic for a given seed.
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

// One score-function estimator value: (1/S) Σ_s (log p − log q) · score
// Score = ∇_μ log q(θ) = (θ − μ)/σ² = ε/σ.
function scoreFunctionEstimate(
  mu: number,
  sigma: number,
  epsilons: Float64Array,
): number {
  let sum = 0;
  const S = epsilons.length;
  const logSigma = Math.log(sigma);
  for (let s = 0; s < S; s++) {
    const eps = epsilons[s];
    const theta = mu + sigma * eps;
    // log p(θ) − log q(θ) up to a μ,σ-free constant. The constant cancels
    // because E_q[const · score] = 0, so it doesn't bias or change variance
    // beyond a baseline that vanishes in expectation; we drop it.
    const logRatio = -0.5 * theta * theta + 0.5 * eps * eps + logSigma;
    const score = eps / sigma;
    sum += logRatio * score;
  }
  return sum / S;
}

// One reparameterization estimator value: (1/S) Σ_s (−θ_s).
// Derived above: after the score-zero simplification, only the path through
// log p remains, and ∇_θ log p = −θ for a standard-Normal target.
function reparamEstimate(
  mu: number,
  sigma: number,
  epsilons: Float64Array,
): number {
  let sum = 0;
  const S = epsilons.length;
  for (let s = 0; s < S; s++) {
    sum += -(mu + sigma * epsilons[s]);
  }
  return sum / S;
}

export default function GradientVarianceComparator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [mu, setMu] = useState<number>(1.5);
  const [logSigma, setLogSigma] = useState<number>(0.5);
  const [S, setS] = useState<number>(8);

  const sigma = useMemo<number>(() => Math.exp(logSigma), [logSigma]);

  // Generate K replications of each estimator at the current (μ, σ, S).
  // Fresh ε samples per replication — that's what makes the estimator
  // a random variable.
  const replications = useMemo<{ sf: Float64Array; rep: Float64Array }>(() => {
    const sf = new Float64Array(N_REPLICATIONS);
    const rep = new Float64Array(N_REPLICATIONS);
    const sampler = makeNormalSampler(20260430);
    const eps = new Float64Array(S);
    for (let k = 0; k < N_REPLICATIONS; k++) {
      for (let s = 0; s < S; s++) eps[s] = sampler();
      sf[k] = scoreFunctionEstimate(mu, sigma, eps);
      rep[k] = reparamEstimate(mu, sigma, eps);
    }
    return { sf, rep };
  }, [mu, sigma, S]);

  const stats = useMemo(() => {
    const summarize = (arr: Float64Array) => {
      let sum = 0;
      for (let i = 0; i < arr.length; i++) sum += arr[i];
      const mean = sum / arr.length;
      let sqsum = 0;
      for (let i = 0; i < arr.length; i++) sqsum += (arr[i] - mean) ** 2;
      return { mean, std: Math.sqrt(sqsum / arr.length) };
    };
    return {
      sf: summarize(replications.sf),
      rep: summarize(replications.rep),
    };
  }, [replications]);

  // Bin replications into histograms over G_VIEW_DOMAIN.
  const histograms = useMemo(() => {
    const bin = (arr: Float64Array) => {
      const counts = new Int32Array(N_BINS);
      const lo = G_VIEW_DOMAIN[0];
      const hi = G_VIEW_DOMAIN[1];
      const span = hi - lo;
      let outside = 0;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v < lo || v > hi) {
          outside++;
          continue;
        }
        const idx = Math.min(N_BINS - 1, Math.floor(((v - lo) / span) * N_BINS));
        counts[idx]++;
      }
      return { counts, outside };
    };
    return { sf: bin(replications.sf), rep: bin(replications.rep) };
  }, [replications]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 24, right: 20, bottom: 48, left: 52 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;

      const xScale = d3.scaleLinear().domain(G_VIEW_DOMAIN).range([0, innerW]);

      const maxCount = Math.max(
        ...histograms.sf.counts,
        ...histograms.rep.counts,
        1,
      );
      const yScale = d3
        .scaleLinear()
        .domain([0, maxCount * 1.05])
        .range([innerH, 0]);

      const root = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      root
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(7).tickSizeOuter(0));
      root.append('g').call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0));
      root
        .append('text')
        .attr('transform', `translate(${innerW / 2},${innerH + 36})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('∇μ ELBO estimate');
      root
        .append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text(`count of ${N_REPLICATIONS} replications`);

      const binW = innerW / N_BINS;

      const drawHistogram = (
        counts: Int32Array,
        color: string,
        opacity: number,
      ) => {
        for (let b = 0; b < N_BINS; b++) {
          if (counts[b] === 0) continue;
          root
            .append('rect')
            .attr('x', b * binW + binW * 0.05)
            .attr('y', yScale(counts[b]))
            .attr('width', binW * 0.9)
            .attr('height', innerH - yScale(counts[b]))
            .attr('fill', color)
            .attr('opacity', opacity);
        }
      };

      drawHistogram(histograms.sf.counts, COLORS.scoreFn, 0.55);
      drawHistogram(histograms.rep.counts, COLORS.reparam, 0.65);

      // Truth line at −μ.
      const trueGrad = -mu;
      root
        .append('line')
        .attr('x1', xScale(trueGrad))
        .attr('x2', xScale(trueGrad))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', COLORS.truth)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5 4');
      root
        .append('text')
        .attr('x', xScale(trueGrad))
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .attr('fill', COLORS.truth)
        .text(`true ∇μ ELBO = −μ = ${trueGrad.toFixed(2)}`);

      // Legend.
      const legend = root.append('g').attr('transform', `translate(${innerW - 220}, 6)`);
      const items = [
        { label: 'Score-function (Theorem 3)', color: COLORS.scoreFn },
        { label: 'Reparameterization (Theorem 4)', color: COLORS.reparam },
      ];
      items.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        row
          .append('rect')
          .attr('x', 0)
          .attr('y', 2)
          .attr('width', 14)
          .attr('height', 10)
          .attr('fill', it.color)
          .attr('opacity', 0.65);
        row
          .append('text')
          .attr('x', 18)
          .attr('y', 11)
          .attr('font-size', 10)
          .text(it.label);
      });
    },
    [histograms, mu, width],
  );

  const stdRatio = stats.rep.std > 1e-9 ? stats.sf.std / stats.rep.std : NaN;

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          μ:
          <input
            type="range"
            min={-2}
            max={2}
            step={0.05}
            value={mu}
            onChange={(e) => setMu(Number(e.target.value))}
            className="w-32"
            aria-label="variational mean slider"
          />
          <span className="tabular-nums w-10 text-right text-xs text-[var(--color-text-muted)]">
            {mu.toFixed(2)}
          </span>
        </label>
        <label className="flex items-center gap-2">
          σ:
          <input
            type="range"
            min={-1}
            max={1.5}
            step={0.05}
            value={logSigma}
            onChange={(e) => setLogSigma(Number(e.target.value))}
            className="w-32"
            aria-label="variational sigma slider"
          />
          <span className="tabular-nums w-10 text-right text-xs text-[var(--color-text-muted)]">
            {sigma.toFixed(2)}
          </span>
        </label>
        <label className="flex items-center gap-2">
          MC samples S:
          <input
            type="range"
            min={1}
            max={200}
            step={1}
            value={S}
            onChange={(e) => setS(Number(e.target.value))}
            className="w-32"
            aria-label="MC sample size slider"
          />
          <span className="tabular-nums w-10 text-right text-xs text-[var(--color-text-muted)]">
            {S}
          </span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-xs text-[var(--color-text-muted)]">
        <span>
          <strong style={{ color: COLORS.scoreFn }}>SF std:</strong>{' '}
          <span className="tabular-nums">{stats.sf.std.toFixed(3)}</span>{' '}
          (mean {stats.sf.mean.toFixed(3)})
        </span>
        <span>
          <strong style={{ color: COLORS.reparam }}>Reparam std:</strong>{' '}
          <span className="tabular-nums">{stats.rep.std.toFixed(3)}</span>{' '}
          (mean {stats.rep.mean.toFixed(3)})
        </span>
        <span>
          <strong>std ratio (SF / Reparam):</strong>{' '}
          <span className="tabular-nums">
            {Number.isFinite(stdRatio) ? stdRatio.toFixed(2) : '∞'}×
          </span>
        </span>
      </div>
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Both estimators are unbiased (Theorems 3 and 4) — at any (μ, σ) both
        histograms center on the dashed black line at −μ, the closed-form
        true gradient. The width difference is the structural cost of
        score-function's universality: each Monte-Carlo sample multiplies
        the integrand value (log p − log q) by the score (θ − μ)/σ², a
        product whose variance scales with how far q sits from p. The
        reparameterization estimator passes the gradient through the model
        directly and lands a much narrower distribution. Drag σ away from 1
        (the true variance of the target) and the orange histogram inflates
        much faster than the blue. Drag S upward and both narrow at the
        same Monte-Carlo √S rate, but the SF curve still trails — that's
        why §4.4 calls reparameterization the default for continuous
        latents and reserves SF for the discrete-variable case.
      </div>
    </div>
  );
}

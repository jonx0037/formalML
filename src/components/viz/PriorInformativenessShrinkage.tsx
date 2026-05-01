import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { GROUP_SIZES, PALETTE_CLASSROOMS } from './shared/mixed-effects';

// =============================================================================
// PriorInformativenessShrinkage — embedded after §5.5's three-takeaways
// paragraph in the mixed-effects topic. Tests the §5.5 claim "Bayesian and
// REML agree numerically when data is informative" by sweeping the τ-prior
// scale across HalfNormal(0.5, 2, 10, 50).
//
// Single-panel forest plot: six classroom rows on the y-axis, random-effects
// value a_j on the x-axis. Per row, the viz shows
//   - true a_j as a black ×
//   - REML BLUP point estimate + 95% Wald CI in orange
//   - Bayesian posterior mean + 95% credible interval in blue
//
// A dropdown selects which fit's blue intervals to display. As the reader
// moves from scale=0.5 → 50, the blue points and intervals migrate from
// "shrunk hard toward zero" (prior dominates) toward "matches REML"
// (data dominates). The orange REML reference is invariant — it doesn't
// depend on the prior. A τ posterior readout below the panel shows the
// posterior mean and 95% interval at the chosen scale alongside REML's
// √τ̂² for direct comparison.
//
// Data source: /sample-data/mixed-effects/prior_informativeness.json,
// produced by notebooks/mixed-effects/precompute_prior_informativeness.py
// (NumPy seed 20260429 → matches §1's notebook data).
// =============================================================================

const PANEL_HEIGHT = 380;
const DATA_URL = '/sample-data/mixed-effects/prior_informativeness.json';

const COLORS = {
  reml: '#ea580c',     // orange — frequentist REML reference
  bayes: '#2563eb',    // blue   — Bayesian posterior under chosen prior
  truth: '#111827',    // near-black — true a_j
  axis: '#374151',
  guide: '#9ca3af',
};

interface FitPayload {
  prior_scale: number;
  n_chains: number;
  n_draws_per_chain: number;
  tau_summary: { mean: number; median: number; lo: number; hi: number };
  sigma_summary: { mean: number; median: number; lo: number; hi: number };
  tau_samples: number[];
  a_mean: number[];
  a_lo: number[];
  a_hi: number[];
  rhat_tau: number;
  ess_bulk_tau: number;
}

interface RemlReference {
  alpha_hat: number;
  beta_hat: number;
  tau_sq_hat: number;
  sigma_sq_hat: number;
  lambdas: number[];
  blup_a: number[];
  blup_a_lo: number[];
  blup_a_hi: number[];
}

interface PriorPayload {
  metadata: {
    pymc_version: string;
    n_classrooms: number;
    group_sizes: number[];
    seed: number;
    true_values: { alpha: number; beta: number; tau: number; sigma: number };
    prior_scales: number[];
  };
  data: {
    classroom: number[];
    x: number[];
    y: number[];
    a_true: number[];
  };
  reml: RemlReference;
  fits: FitPayload[];
}

export default function PriorInformativenessShrinkage() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<PriorPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Default to scale=10 (the topic's default in the prose).
  const [scaleIdx, setScaleIdx] = useState<number>(2);

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: PriorPayload) => {
        if (!cancelled) setPayload(j);
      })
      .catch((e) => {
        if (!cancelled) setLoadError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const fit = payload?.fits[scaleIdx];

  // Domain pads cover both REML and Bayesian intervals plus the truth.
  const aDomain = useMemo<[number, number]>(() => {
    if (!payload) return [-10, 10];
    let lo = Infinity;
    let hi = -Infinity;
    const update = (v: number) => {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    };
    payload.data.a_true.forEach(update);
    payload.reml.blup_a.forEach(update);
    payload.reml.blup_a_lo.forEach(update);
    payload.reml.blup_a_hi.forEach(update);
    payload.fits.forEach((f) => {
      f.a_mean.forEach(update);
      f.a_lo.forEach(update);
      f.a_hi.forEach(update);
    });
    const pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  }, [payload]);

  const ref = useD3(
    (svg) => {
      if (!payload || !fit) return;
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 22, right: 24, bottom: 44, left: 78 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;

      const xScale = d3.scaleLinear().domain(aDomain).range([0, innerW]);
      const J = payload.metadata.n_classrooms;
      const rowH = innerH / J;

      const root = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      // X-axis with zero-line emphasis.
      root
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(8).tickSizeOuter(0));
      root
        .append('text')
        .attr('transform', `translate(${innerW / 2},${innerH + 32})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('classroom random effect aⱼ');

      // Vertical guide at zero (population mean).
      root
        .append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(0))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', COLORS.guide)
        .attr('stroke-dasharray', '3 4')
        .attr('stroke-width', 1)
        .attr('opacity', 0.7);

      // One row per classroom.
      for (let j = 0; j < J; j++) {
        const yMid = (j + 0.5) * rowH;
        const color = PALETTE_CLASSROOMS[j];

        // Y-axis label: classroom + group size.
        root
          .append('text')
          .attr('x', -8)
          .attr('y', yMid + 4)
          .attr('text-anchor', 'end')
          .attr('font-size', 11)
          .attr('fill', color)
          .attr('font-weight', 600)
          .text(`Classroom ${j + 1} (n=${payload.metadata.group_sizes[j]})`);

        // REML BLUP interval — orange, slightly above row midpoint.
        const remlY = yMid - 6;
        root
          .append('line')
          .attr('x1', xScale(payload.reml.blup_a_lo[j]))
          .attr('x2', xScale(payload.reml.blup_a_hi[j]))
          .attr('y1', remlY)
          .attr('y2', remlY)
          .attr('stroke', COLORS.reml)
          .attr('stroke-width', 2.4);
        root
          .append('circle')
          .attr('cx', xScale(payload.reml.blup_a[j]))
          .attr('cy', remlY)
          .attr('r', 4)
          .attr('fill', COLORS.reml);

        // Bayesian posterior interval — blue, slightly below row midpoint.
        const bayesY = yMid + 6;
        root
          .append('line')
          .attr('x1', xScale(fit.a_lo[j]))
          .attr('x2', xScale(fit.a_hi[j]))
          .attr('y1', bayesY)
          .attr('y2', bayesY)
          .attr('stroke', COLORS.bayes)
          .attr('stroke-width', 2.4);
        root
          .append('circle')
          .attr('cx', xScale(fit.a_mean[j]))
          .attr('cy', bayesY)
          .attr('r', 4)
          .attr('fill', COLORS.bayes);

        // True a_j as a × at the row midpoint.
        const tx = xScale(payload.data.a_true[j]);
        const tickLen = 4;
        root
          .append('line')
          .attr('x1', tx - tickLen)
          .attr('x2', tx + tickLen)
          .attr('y1', yMid - tickLen)
          .attr('y2', yMid + tickLen)
          .attr('stroke', COLORS.truth)
          .attr('stroke-width', 1.6);
        root
          .append('line')
          .attr('x1', tx - tickLen)
          .attr('x2', tx + tickLen)
          .attr('y1', yMid + tickLen)
          .attr('y2', yMid - tickLen)
          .attr('stroke', COLORS.truth)
          .attr('stroke-width', 1.6);
      }

      // Legend at top-right.
      const legend = root.append('g').attr('transform', `translate(${innerW - 220},-12)`);
      const legendItems = [
        { label: 'REML BLUP (95% CI)', color: COLORS.reml, marker: 'line' },
        { label: 'Bayesian posterior (95%)', color: COLORS.bayes, marker: 'line' },
        { label: 'true aⱼ', color: COLORS.truth, marker: 'x' },
      ];
      legendItems.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(${i * 130},0)`);
        if (it.marker === 'line') {
          row
            .append('line')
            .attr('x1', 0)
            .attr('x2', 18)
            .attr('y1', 7)
            .attr('y2', 7)
            .attr('stroke', it.color)
            .attr('stroke-width', 2.4);
          row
            .append('circle')
            .attr('cx', 9)
            .attr('cy', 7)
            .attr('r', 3)
            .attr('fill', it.color);
        } else {
          row
            .append('line')
            .attr('x1', 5)
            .attr('x2', 13)
            .attr('y1', 3)
            .attr('y2', 11)
            .attr('stroke', it.color)
            .attr('stroke-width', 1.6);
          row
            .append('line')
            .attr('x1', 5)
            .attr('x2', 13)
            .attr('y1', 11)
            .attr('y2', 3)
            .attr('stroke', it.color)
            .attr('stroke-width', 1.6);
        }
        row
          .append('text')
          .attr('x', 22)
          .attr('y', 10)
          .attr('font-size', 10)
          .text(it.label);
      });
    },
    [payload, fit, aDomain, width, scaleIdx],
  );

  const m = payload?.metadata;
  const remlTau = payload ? Math.sqrt(payload.reml.tau_sq_hat) : null;

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          τ prior scale (HalfNormal):
          <select
            value={scaleIdx}
            onChange={(e) => setScaleIdx(Number(e.target.value))}
            className="rounded border px-1"
            aria-label="prior scale selector"
            disabled={!payload}
          >
            {payload
              ? payload.metadata.prior_scales.map((s, i) => (
                  <option key={s} value={i}>
                    {s}{' '}
                    {s <= 0.5
                      ? '(very tight)'
                      : s <= 2
                        ? '(tight)'
                        : s <= 10
                          ? '(weakly informative — default)'
                          : '(loose)'}
                  </option>
                ))
              : null}
          </select>
        </label>
      </div>
      <div
        ref={containerRef}
        className="w-full"
        style={{ minHeight: PANEL_HEIGHT }}
      >
        {loadError ? (
          <div className="text-sm">
            Failed to load prior-sweep data: {loadError}. Make sure
            <code className="mx-1">
              notebooks/mixed-effects/precompute_prior_informativeness.py
            </code>
            has been run.
          </div>
        ) : payload ? (
          <svg ref={ref} />
        ) : (
          <div className="text-sm text-[var(--color-text-muted)]">
            Loading prior-sweep posteriors…
          </div>
        )}
      </div>
      {fit && remlTau !== null && (
        <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-xs text-[var(--color-text-muted)]">
          <span>
            <strong style={{ color: COLORS.bayes }}>τ posterior:</strong>{' '}
            <span className="tabular-nums">
              mean {fit.tau_summary.mean.toFixed(3)}, 95% [
              {fit.tau_summary.lo.toFixed(3)}, {fit.tau_summary.hi.toFixed(3)}
              ]
            </span>
          </span>
          <span>
            <strong style={{ color: COLORS.reml }}>REML τ̂:</strong>{' '}
            <span className="tabular-nums">{remlTau.toFixed(3)}</span>
          </span>
          <span>
            <strong>R̂(τ):</strong>{' '}
            <span
              className="tabular-nums"
              style={{ color: fit.rhat_tau > 1.01 ? COLORS.reml : 'inherit' }}
            >
              {fit.rhat_tau.toFixed(4)}
            </span>
          </span>
          <span>
            <strong>True τ:</strong>{' '}
            <span className="tabular-nums">{m!.true_values.tau.toFixed(2)}</span>
          </span>
        </div>
      )}
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Each row is one classroom; the orange interval is the REML BLUP 95%
        Wald CI from statsmodels (frequentist plug-in path); the blue
        interval is the Bayesian 95% credible interval from a non-centered
        PyMC fit at the chosen τ prior. The orange reference is fixed — REML
        ignores the Bayesian prior. The blue intervals shift with the
        dropdown: at scale = 0.5 the prior shrinks every classroom toward 0
        regardless of group data; at scale = 10 (the topic's default)
        Bayesian and REML agree numerically; at scale = 50 they remain in
        agreement because the data has dominated the prior. The §5.5 claim
        "Bayesian and REML are not different methods, just different ways of
        handling the variance components" becomes the visible behavior of
        the blue intervals collapsing into the orange ones as the prior
        relaxes.
      </div>
    </div>
  );
}

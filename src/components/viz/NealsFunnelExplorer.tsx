import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// NealsFunnelExplorer — embedded after §5.2's pedagogical-point paragraph in
// the probabilistic-programming topic. Scatter of the centered eight-schools
// fit's joint draws of (θ_j, log τ) for a user-selected school j ∈ {A..H},
// with divergent transitions painted in red. The funnel shape is wide at the
// top (large τ → loose θ_j) and pinches at the bottom (small τ → θ_j locked
// to μ). Divergent draws cluster at the neck where the leapfrog integrator's
// fixed step size mismatches the local curvature.
//
// Data source: /sample-data/probabilistic-programming/neals_funnel.json,
// produced by notebooks/probabilistic-programming/precompute_neals_funnel.py
// (PyMC NUTS, 4 chains × 1000 draws, target_accept=0.8). The JSON includes
// metadata with R-hat, ESS bulk, and divergence rate which the viz surfaces
// as a diagnostics panel — the reader sees the divergent count is non-zero
// AND that R-hat is > 1.01, both signs of trouble §5.3 will fix via the
// non-centered reparameterization.
// =============================================================================

const PANEL_HEIGHT = 420;
const DATA_URL = '/sample-data/probabilistic-programming/neals_funnel.json';

const SCHOOL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] as const;

const COLORS = {
  normal: '#3b82f6',     // blue — non-divergent draws
  divergent: '#dc2626',  // red — divergent draws
  axis: '#374151',
  guide: '#9ca3af',
};

interface FunnelMetadata {
  model: string;
  engine: string;
  pymc_version: string;
  n_chains: number;
  n_draws_per_chain: number;
  n_total: number;
  n_divergent: number;
  divergence_rate: number;
  target_accept: number;
  tune_steps: number;
  seed: number;
  rhat: { mu: number; tau: number };
  ess_bulk: { mu: number; tau: number };
}

interface FunnelPayload {
  metadata: FunnelMetadata;
  schools: string[];
  y_obs: number[];
  sigma_obs: number[];
  draws: {
    mu: number[];
    log_tau: number[];
    theta: number[][];   // [n_total][8]
    divergent: boolean[];
  };
}

export default function NealsFunnelExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<FunnelPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [schoolIdx, setSchoolIdx] = useState<number>(0);
  const [highlightDivergent, setHighlightDivergent] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: FunnelPayload) => {
        if (!cancelled) setPayload(j);
      })
      .catch((e) => {
        if (!cancelled) setLoadError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Pre-extract the chosen school's θ_j and log_tau columns once per change.
  const points = useMemo<{
    x: Float64Array;
    y: Float64Array;
    div: boolean[];
  } | null>(() => {
    if (!payload) return null;
    const n = payload.draws.log_tau.length;
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = payload.draws.theta[i][schoolIdx];
      y[i] = payload.draws.log_tau[i];
    }
    return { x, y, div: payload.draws.divergent };
  }, [payload, schoolIdx]);

  const ref = useD3(
    (svg) => {
      if (!points || !payload) return;
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 20, right: 20, bottom: 48, left: 56 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;

      // Domains from data with a little pad.
      let xLo = Infinity;
      let xHi = -Infinity;
      let yLo = Infinity;
      let yHi = -Infinity;
      for (let i = 0; i < points.x.length; i++) {
        const xv = points.x[i];
        const yv = points.y[i];
        if (xv < xLo) xLo = xv;
        if (xv > xHi) xHi = xv;
        if (yv < yLo) yLo = yv;
        if (yv > yHi) yHi = yv;
      }
      const xPad = (xHi - xLo) * 0.04;
      const yPad = (yHi - yLo) * 0.04;
      const xScale = d3
        .scaleLinear()
        .domain([xLo - xPad, xHi + xPad])
        .range([0, innerW]);
      const yScale = d3
        .scaleLinear()
        .domain([yLo - yPad, yHi + yPad])
        .range([innerH, 0]);

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
        .text(`θ_${SCHOOL_LABELS[schoolIdx]} (school ${SCHOOL_LABELS[schoolIdx]})`);

      root
        .append('text')
        .attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('log τ');

      // Plot non-divergent draws first (so divergent draws sit on top).
      const normalGroup = root.append('g');
      const divergentGroup = root.append('g');

      for (let i = 0; i < points.x.length; i++) {
        const isDiv = points.div[i];
        const target = isDiv ? divergentGroup : normalGroup;
        target
          .append('circle')
          .attr('cx', xScale(points.x[i]))
          .attr('cy', yScale(points.y[i]))
          .attr(
            'r',
            isDiv && highlightDivergent ? 2.6 : 1.7,
          )
          .attr(
            'fill',
            isDiv && highlightDivergent ? COLORS.divergent : COLORS.normal,
          )
          .attr(
            'opacity',
            isDiv && highlightDivergent ? 0.85 : 0.4,
          );
      }

      // Legend.
      const legend = root.append('g').attr('transform', `translate(${innerW - 180},10)`);
      const legendItems = [
        { label: 'Non-divergent draws', color: COLORS.normal, r: 1.7, opacity: 0.55 },
        { label: 'Divergent transitions', color: COLORS.divergent, r: 2.6, opacity: 0.95 },
      ];
      legendItems.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        row
          .append('circle')
          .attr('cx', 6)
          .attr('cy', 6)
          .attr('r', it.r + 0.5)
          .attr('fill', it.color)
          .attr('opacity', it.opacity);
        row
          .append('text')
          .attr('x', 18)
          .attr('y', 9)
          .attr('font-size', 11)
          .text(it.label);
      });
    },
    [points, highlightDivergent, payload, width, schoolIdx],
  );

  // Keep the containerRef'd <div> mounted from the very first render so the
  // useResizeObserver effect (which runs once on mount and depends on
  // ref.current being non-null at that moment) wires up correctly. Swap only
  // the inner SVG vs loading/error placeholder.
  const m = payload?.metadata;

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          school θⱼ:
          <select
            value={schoolIdx}
            onChange={(e) => setSchoolIdx(Number(e.target.value))}
            className="rounded border px-1"
            aria-label="school selector"
            disabled={!payload}
          >
            {SCHOOL_LABELS.map((label, idx) => (
              <option key={label} value={idx}>
                {label}
                {payload
                  ? ` (yⱼ = ${payload.y_obs[idx].toFixed(0)}, σⱼ = ${payload.sigma_obs[idx].toFixed(0)})`
                  : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={highlightDivergent}
            onChange={(e) => setHighlightDivergent(e.target.checked)}
            disabled={!payload}
          />
          <span>Highlight divergences</span>
        </label>
      </div>
      <div
        ref={containerRef}
        className="w-full"
        style={{ minHeight: PANEL_HEIGHT }}
      >
        {loadError ? (
          <div className="text-sm">
            Failed to load funnel trace data: {loadError}. Make sure
            <code className="mx-1">
              notebooks/probabilistic-programming/precompute_neals_funnel.py
            </code>
            has been run.
          </div>
        ) : payload ? (
          <svg ref={ref} />
        ) : (
          <div className="text-sm text-[var(--color-text-muted)]">
            Loading centered eight-schools NUTS trace…
          </div>
        )}
      </div>
      {m && (
        <div className="mt-3 flex flex-wrap items-baseline gap-x-5 gap-y-1 text-xs text-[var(--color-text-muted)]">
          <span>
            <strong>Divergent transitions:</strong>{' '}
            <span className="tabular-nums">
              {m.n_divergent} / {m.n_total} ({(100 * m.divergence_rate).toFixed(2)}%)
            </span>
          </span>
          <span>
            <strong>R̂(τ):</strong>{' '}
            <span
              className="tabular-nums"
              style={{ color: m.rhat.tau > 1.01 ? COLORS.divergent : 'inherit' }}
            >
              {m.rhat.tau.toFixed(4)}
            </span>
          </span>
          <span>
            <strong>ESS bulk(τ):</strong>{' '}
            <span className="tabular-nums">{m.ess_bulk.tau}</span>
          </span>
          <span>
            <strong>Engine:</strong> {m.engine} {m.pymc_version}
          </span>
        </div>
      )}
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Each blue dot is one NUTS draw of (θⱼ, log τ) from the centered fit;
        red dots are draws PyMC flagged as divergent transitions. The funnel
        opens at the top (large τ → loose θⱼ) and pinches at the bottom
        (small τ → θⱼ pulled tight to μ). Divergences cluster at the neck
        where the leapfrog integrator's fixed step size cannot accommodate
        the rapidly-changing local curvature. Switching schools shows that
        the funnel shape is a property of the joint geometry, not specific
        to school A. The combination of non-zero divergences AND R̂(τ) above
        1.01 is exactly what §5.3's non-centered reparameterization fixes.
      </div>
    </div>
  );
}

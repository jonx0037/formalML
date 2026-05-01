import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// CenteredVsNonCenteredMorph — embedded after §5.3 of the probabilistic-
// programming topic. Side-by-side scatter of the two parameterizations'
// joint posteriors:
//
//   Left  (centered):     (θ_j, log τ) with divergent transitions in red.
//                         The funnel shape is unmistakable.
//   Right (non-centered): (z_j, log τ) with no divergences (or very few).
//                         The "z = (θ − μ)/τ" reparameterization unwinds
//                         the funnel into a roughly rectangular blob.
//
// Both panels share the same log τ axis, so the reader sees the funnel
// pinch at the bottom of the centered panel collapse into a uniform-width
// strip on the right. Switching schools via dropdown shows the same
// disappearance for every θ_j / z_j pair.
//
// A diagnostics readout below the panels reports per-fit divergence count,
// R̂(τ), and ESS bulk(τ). The textbook before/after — divergences:
// 107 → 0, R̂(τ): 1.027 → 1.000, ESS bulk: 251 → 2612 — is the §5.3
// quantitative evidence that the reparameterization isn't just visually
// different, it's diagnostically clean.
//
// Data source: /sample-data/probabilistic-programming/neals_funnel.json,
// extended in this branch to also include the non-centered fit. Same
// precompute pipeline as B3 NealsFunnelExplorer; both viz read the same
// JSON file.
// =============================================================================

const PANEL_HEIGHT = 460;
const PANEL_GAP = 24;
const DATA_URL = '/sample-data/probabilistic-programming/neals_funnel.json';
const SCHOOL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] as const;

const COLORS = {
  normal: '#3b82f6',
  divergent: '#dc2626',
  axis: '#374151',
};

interface FitMetadata {
  model: string;
  n_total: number;
  n_divergent: number;
  divergence_rate: number;
  rhat: { mu: number; tau: number };
  ess_bulk: { mu: number; tau: number };
}

interface CenteredDraws {
  mu: number[];
  log_tau: number[];
  theta: number[][];
  divergent: boolean[];
}

interface NonCenteredDraws extends CenteredDraws {
  z: number[][];
}

interface FunnelPayload {
  metadata: FitMetadata & { engine: string; pymc_version: string };
  schools: string[];
  draws: CenteredDraws;
  non_centered: {
    metadata: FitMetadata;
    draws: NonCenteredDraws;
  };
}

export default function CenteredVsNonCenteredMorph() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<FunnelPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [schoolIdx, setSchoolIdx] = useState<number>(0);

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: FunnelPayload) => {
        if (!cancelled) {
          if (!j.non_centered) {
            setLoadError(
              'JSON missing non_centered field — re-run notebooks/probabilistic-programming/precompute_neals_funnel.py to regenerate.',
            );
          } else {
            setPayload(j);
          }
        }
      })
      .catch((e) => {
        if (!cancelled) setLoadError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Pre-extract scatter points for both panels at the chosen school.
  const points = useMemo<{
    centered: { x: Float64Array; y: Float64Array; div: boolean[] };
    nonCentered: { x: Float64Array; y: Float64Array };
  } | null>(() => {
    if (!payload) return null;
    const N = payload.draws.log_tau.length;
    const cx = new Float64Array(N);
    const cy = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      cx[i] = payload.draws.theta[i][schoolIdx];
      cy[i] = payload.draws.log_tau[i];
    }
    const Nn = payload.non_centered.draws.log_tau.length;
    const nx = new Float64Array(Nn);
    const ny = new Float64Array(Nn);
    for (let i = 0; i < Nn; i++) {
      nx[i] = payload.non_centered.draws.z[i][schoolIdx];
      ny[i] = payload.non_centered.draws.log_tau[i];
    }
    return {
      centered: { x: cx, y: cy, div: payload.draws.divergent },
      nonCentered: { x: nx, y: ny },
    };
  }, [payload, schoolIdx]);

  // Shared y-axis range so the panels are visually comparable.
  const yDomain = useMemo<[number, number]>(() => {
    if (!points) return [-3, 3];
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < points.centered.y.length; i++) {
      if (points.centered.y[i] < lo) lo = points.centered.y[i];
      if (points.centered.y[i] > hi) hi = points.centered.y[i];
    }
    for (let i = 0; i < points.nonCentered.y.length; i++) {
      if (points.nonCentered.y[i] < lo) lo = points.nonCentered.y[i];
      if (points.nonCentered.y[i] > hi) hi = points.nonCentered.y[i];
    }
    return [lo - 0.1, hi + 0.1];
  }, [points]);

  const ref = useD3(
    (svg) => {
      if (!points || !payload) return;
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 36, right: 16, bottom: 48, left: 56 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      const panelW = (innerW - PANEL_GAP) / 2;

      const yScale = d3.scaleLinear().domain(yDomain).range([innerH, 0]);

      const drawPanel = (
        offsetX: number,
        title: string,
        xData: Float64Array,
        yData: Float64Array,
        divFlags: boolean[] | null,
        xAxisLabel: string,
        showYAxis: boolean,
      ) => {
        // Compute x-domain per panel.
        let xLo = Infinity;
        let xHi = -Infinity;
        for (let i = 0; i < xData.length; i++) {
          if (xData[i] < xLo) xLo = xData[i];
          if (xData[i] > xHi) xHi = xData[i];
        }
        const xPad = (xHi - xLo) * 0.04;
        const xScale = d3
          .scaleLinear()
          .domain([xLo - xPad, xHi + xPad])
          .range([0, panelW]);

        const root = svg
          .append('g')
          .attr('transform', `translate(${margin.left + offsetX},${margin.top})`);

        root
          .append('g')
          .attr('transform', `translate(0,${innerH})`)
          .call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
        if (showYAxis) {
          root.append('g').call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
          root
            .append('text')
            .attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`)
            .attr('text-anchor', 'middle')
            .attr('font-size', 12)
            .text('log τ');
        }
        root
          .append('text')
          .attr('transform', `translate(${panelW / 2},${innerH + 36})`)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .text(xAxisLabel);

        // Title.
        root
          .append('text')
          .attr('x', panelW / 2)
          .attr('y', -12)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .attr('font-weight', 600)
          .text(title);

        // Plot non-divergent first, divergent on top so they're visible.
        const normalGroup = root.append('g');
        const divergentGroup = root.append('g');
        const N = xData.length;
        for (let i = 0; i < N; i++) {
          const isDiv = divFlags && divFlags[i];
          const target = isDiv ? divergentGroup : normalGroup;
          target
            .append('circle')
            .attr('cx', xScale(xData[i]))
            .attr('cy', yScale(yData[i]))
            .attr('r', isDiv ? 2.2 : 1.4)
            .attr('fill', isDiv ? COLORS.divergent : COLORS.normal)
            .attr('opacity', isDiv ? 0.85 : 0.35);
        }
      };

      drawPanel(
        0,
        'Centered: (θⱼ, log τ)',
        points.centered.x,
        points.centered.y,
        points.centered.div,
        `θ_${SCHOOL_LABELS[schoolIdx]}`,
        true,
      );
      drawPanel(
        panelW + PANEL_GAP,
        'Non-centered: (zⱼ, log τ)',
        points.nonCentered.x,
        points.nonCentered.y,
        null,
        `z_${SCHOOL_LABELS[schoolIdx]}`,
        false,
      );
    },
    [points, payload, yDomain, schoolIdx, width],
  );

  const cMeta = payload?.metadata;
  const ncMeta = payload?.non_centered.metadata;

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          school:
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
              </option>
            ))}
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
            Failed to load eight-schools traces: {loadError}
          </div>
        ) : payload ? (
          <svg ref={ref} />
        ) : (
          <div className="text-sm text-[var(--color-text-muted)]">
            Loading centered + non-centered NUTS traces…
          </div>
        )}
      </div>
      {cMeta && ncMeta && (
        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1 text-xs text-[var(--color-text-muted)]">
          <div>
            <strong>Centered:</strong>
            <ul className="ml-4 list-disc">
              <li>
                Divergences:{' '}
                <span
                  className="tabular-nums"
                  style={{ color: COLORS.divergent }}
                >
                  {cMeta.n_divergent} / {cMeta.n_total}
                </span>{' '}
                ({(100 * cMeta.divergence_rate).toFixed(2)}%)
              </li>
              <li>
                R̂(τ):{' '}
                <span
                  className="tabular-nums"
                  style={{ color: cMeta.rhat.tau > 1.01 ? COLORS.divergent : 'inherit' }}
                >
                  {cMeta.rhat.tau.toFixed(4)}
                </span>
              </li>
              <li>
                ESS bulk(τ):{' '}
                <span className="tabular-nums">{cMeta.ess_bulk.tau}</span>
              </li>
            </ul>
          </div>
          <div>
            <strong>Non-centered:</strong>
            <ul className="ml-4 list-disc">
              <li>
                Divergences:{' '}
                <span className="tabular-nums">
                  {ncMeta.n_divergent} / {ncMeta.n_total}
                </span>{' '}
                ({(100 * ncMeta.divergence_rate).toFixed(2)}%)
              </li>
              <li>
                R̂(τ):{' '}
                <span className="tabular-nums">{ncMeta.rhat.tau.toFixed(4)}</span>
              </li>
              <li>
                ESS bulk(τ):{' '}
                <span className="tabular-nums">{ncMeta.ess_bulk.tau}</span>
              </li>
            </ul>
          </div>
        </div>
      )}
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Same data, same model, two parameterizations. The centered panel
        shows the funnel that §5.2 diagnosed: dense at the top (large τ →
        loose θⱼ), tight at the bottom (small τ → θⱼ pulled to μ),
        divergent transitions in red clustering at the neck. The
        non-centered panel shows the rectangular geometry NUTS actually
        traverses when the user writes θⱼ = μ + τ·zⱼ instead — same
        posterior on θⱼ marginally (the models are probabilistically
        equivalent), but no funnel for the sampler to choke on. The
        diagnostic table below each panel quantifies the difference: 107
        divergences → 0, R̂(τ) 1.0275 → 0.9999, ESS bulk 251 → 2612 — the
        non-centered fit is "clean" by every standard PyMC diagnostic, and
        the §5.3 claim "the rewrite is purely a user-side change to the
        model code; the engine's behavior is identical" becomes the
        observable fact that the same sampler with the same settings
        produces a usable trace from one and a problematic trace from the
        other.
      </div>
    </div>
  );
}

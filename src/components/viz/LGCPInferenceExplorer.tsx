// =============================================================================
// LGCPInferenceExplorer.tsx
//
// §11.3 of sequential-monte-carlo. Interactive viewer over precomputed
// log-Gaussian Cox process SMC inference at three grid resolutions m ∈
// {8, 12, 16} (K = 64 / 144 / 256). K = 256 is too heavy for browser-side
// SMC, so precompute_lgcp.py runs the sampler offline and writes a single
// JSON payload at /sample-data/sequential-monte-carlo/lgcp.json.
//
// Four panels:
//   (A) True intensity field on the m × m grid.
//   (B) SMC posterior mean intensity (RMSE printed in the panel title).
//   (C) ESS through the adaptive path.
//   (D) Posterior-mean trajectory at the high- and low-intensity cells,
//       with truth lines for reference.
//
// Reader-discoverable behaviours:
//   - m = 8: low-resolution recovery, RMSE modest.
//   - m = 16: high-resolution, longer adaptive path, sharper intensity recovery.
//   - The ESS profile stays near τ_adapt N until convergence; the high-cell
//     posterior trajectory climbs from the prior mean toward the truth.
//
// Static fallback: /images/topics/sequential-monte-carlo/11_lgcp_smc.png
// =============================================================================

import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteSMC } from './shared/sequential-monte-carlo';

type Grid = '8' | '12' | '16';

interface GridPayload {
  m: number;
  K: number;
  centers: number[];
  y_obs: number[];
  lambda_true: number[];
  lambda_post: number[];
  rmse_intensity: number;
  beta_trace: number[];
  ess_trace: number[];
  log_z_trace: number[];
  lambda_high_trace: number[];
  lambda_low_trace: number[];
  high_cell_idx: number;
  low_cell_idx: number;
  lambda_high_truth: number;
  lambda_low_truth: number;
  lambda_max: number;
  T_steps: number;
  wall_seconds: number;
  log_z_T: number;
}

interface Payload {
  seed: number;
  sigma_z_sq: number;
  ell: number;
  mu_0: number;
  tau_adapt: number;
  tau_resample: number;
  N: number;
  grids: Record<Grid, GridPayload>;
}

const DATA_URL = '/sample-data/sequential-monte-carlo/lgcp.json';

export default function LGCPInferenceExplorer() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [m, setM] = useState<Grid>('16');

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (!cancelled) setPayload(data as Payload);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 760;

  // Even when payload is loading, keep containerRef mounted so the
  // resize-observer attaches (per CLAUDE.md loading-state JSX rule).
  const grid = payload?.grids[m] ?? null;

  const heatmapSvgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!grid || width <= 0) return;
      const panelGap = 14;
      const panelH = 280;
      const margin = { top: 24, right: 12, bottom: 32, left: 44 };
      const panelW = (width - panelGap) / 2;
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;
      const mDim = grid.m;
      const cellW = innerW / mDim;
      const cellH = innerH / mDim;
      const lambdaMax = Math.max(grid.lambda_max, ...grid.lambda_post);
      // Use a Reds-like color scale for intensity
      const color = d3.scaleSequential(d3.interpolateReds).domain([0, lambdaMax]);

      const drawHeatmap = (ki: number, label: string, lambdas: number[]) => {
        const g = svg.append('g').attr('transform', `translate(${ki * (panelW + panelGap)}, 0)`);
        const inner = g.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
        // cells
        for (let i = 0; i < mDim; i++) {
          for (let j = 0; j < mDim; j++) {
            const idx = i * mDim + j;
            inner
              .append('rect')
              .attr('x', j * cellW)
              .attr('y', (mDim - 1 - i) * cellH)
              .attr('width', cellW)
              .attr('height', cellH)
              .style('fill', color(lambdas[idx]))
              .style('stroke', 'none');
          }
        }
        const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
        const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
        inner
          .append('g')
          .attr('transform', `translate(0, ${innerH})`)
          .call(d3.axisBottom(x).ticks(3))
          .call((s) => s.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
          .call((s) => s.selectAll('line, path').style('stroke', 'var(--color-muted)'));
        inner
          .append('g')
          .call(d3.axisLeft(y).ticks(3))
          .call((s) => s.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
          .call((s) => s.selectAll('line, path').style('stroke', 'var(--color-muted)'));
        g.append('text')
          .attr('x', margin.left)
          .attr('y', 14)
          .style('font-size', '11px')
          .style('font-weight', '600')
          .style('fill', 'var(--color-text)')
          .text(label);
      };

      drawHeatmap(0, '(A) true intensity', grid.lambda_true);
      drawHeatmap(
        1,
        `(B) SMC posterior mean (RMSE = ${grid.rmse_intensity.toFixed(1)})`,
        grid.lambda_post,
      );
    },
    [width, grid],
  );

  const tracesSvgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!grid || width <= 0) return;
      const panelGap = 14;
      const panelH = 200;
      const margin = { top: 24, right: 12, bottom: 32, left: 50 };
      const panelW = (width - panelGap) / 2;
      const innerW = panelW - margin.left - margin.right;
      const innerH = panelH - margin.top - margin.bottom;

      // Panel C: ESS through adaptive path (vs step index)
      const gC = svg.append('g').attr('transform', 'translate(0, 0)');
      const innerC = gC.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xC = d3.scaleLinear().domain([0, grid.T_steps]).range([0, innerW]);
      const yC = d3.scaleLinear().domain([0, payload!.N * 1.05]).range([innerH, 0]);
      innerC
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xC).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerC
        .append('g')
        .call(d3.axisLeft(yC).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerC
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('step t');
      innerC
        .append('text')
        .attr('transform', `translate(${-36}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('ESS');
      // tau_adapt and tau_resample references
      innerC
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yC(payload!.tau_adapt * payload!.N))
        .attr('y2', yC(payload!.tau_adapt * payload!.N))
        .style('stroke', paletteSMC.cloud)
        .style('stroke-width', 0.8)
        .style('stroke-dasharray', '2 2')
        .style('opacity', 0.7);
      innerC
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yC(payload!.tau_resample * payload!.N))
        .attr('y2', yC(payload!.tau_resample * payload!.N))
        .style('stroke', paletteSMC.accent)
        .style('stroke-width', 0.8)
        .style('stroke-dasharray', '4 3')
        .style('opacity', 0.8);
      const lnC = d3
        .line<number>()
        .x((_, i) => xC(i))
        .y((e) => yC(e));
      innerC
        .append('path')
        .datum(grid.ess_trace)
        .attr('d', lnC)
        .style('fill', 'none')
        .style('stroke', paletteSMC.cloud)
        .style('stroke-width', 1.6);
      gC.append('text')
        .attr('x', margin.left)
        .attr('y', 14)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(C) ESS through adaptive path');

      // Panel D: representative-cell posterior trajectory (log-y)
      const gD = svg.append('g').attr('transform', `translate(${panelW + panelGap}, 0)`);
      const innerD = gD.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xD = d3.scaleLinear().domain([0, grid.T_steps]).range([0, innerW]);
      const yAll = [
        ...grid.lambda_high_trace,
        ...grid.lambda_low_trace,
        grid.lambda_high_truth,
        grid.lambda_low_truth,
      ].filter((v) => v > 0);
      const yMin = Math.max(1e-2, Math.min(...yAll));
      const yMax = Math.max(...yAll) * 1.2;
      const yD = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);
      innerD
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xD).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerD
        .append('g')
        .call(d3.axisLeft(yD).ticks(4, '.0f'))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerD
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('step t');
      innerD
        .append('text')
        .attr('transform', `translate(${-36}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('λ̂ (log)');
      const lnD = d3
        .line<number>()
        .x((_, i) => xD(i))
        .y((v) => yD(Math.max(v, yMin)));
      // truth lines
      innerD
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yD(Math.max(grid.lambda_high_truth, yMin)))
        .attr('y2', yD(Math.max(grid.lambda_high_truth, yMin)))
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 0.8)
        .style('stroke-dasharray', '2 3');
      innerD
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yD(Math.max(grid.lambda_low_truth, yMin)))
        .attr('y2', yD(Math.max(grid.lambda_low_truth, yMin)))
        .style('stroke', paletteSMC.cloud)
        .style('stroke-width', 0.8)
        .style('stroke-dasharray', '2 3');
      // high-cell trajectory
      innerD
        .append('path')
        .datum(grid.lambda_high_trace)
        .attr('d', lnD)
        .style('fill', 'none')
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.6);
      // low-cell trajectory
      innerD
        .append('path')
        .datum(grid.lambda_low_trace)
        .attr('d', lnD)
        .style('fill', 'none')
        .style('stroke', paletteSMC.cloud)
        .style('stroke-width', 1.6);
      gD.append('text')
        .attr('x', margin.left)
        .attr('y', 14)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(D) high/low cell posterior trajectory');
    },
    [width, grid, payload],
  );

  return (
    <div ref={containerRef} className="w-full">
      <div
        className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs"
        style={{ color: 'var(--color-text)' }}
      >
        <label className="inline-flex items-center gap-2">
          <span>grid m × m:</span>
          <select
            value={m}
            onChange={(e) => setM(e.target.value as Grid)}
            className="rounded border px-1 py-0.5 text-xs"
            aria-label="LGCP grid resolution"
            disabled={!payload}
          >
            <option value="8">8 × 8 (K = 64)</option>
            <option value="12">12 × 12 (K = 144)</option>
            <option value="16">16 × 16 (K = 256)</option>
          </select>
        </label>
        {payload && grid && (
          <span className="font-mono text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>
            T = {grid.T_steps}, log Ẑ = {grid.log_z_T.toFixed(2)}, RMSE = {grid.rmse_intensity.toFixed(1)}, wall = {grid.wall_seconds.toFixed(1)}s (Python precompute)
          </span>
        )}
      </div>
      {error ? (
        <div className="rounded border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-800">
          Failed to load LGCP precompute: {error}. Confirm
          /sample-data/sequential-monte-carlo/lgcp.json is served.
        </div>
      ) : !payload ? (
        <div className="px-3 py-6 text-center text-xs italic" style={{ color: 'var(--color-text-secondary)' }}>
          Loading LGCP precompute payload…
        </div>
      ) : (
        <>
          <svg
            ref={heatmapSvgRef}
            width={width}
            height={280}
            viewBox={`0 0 ${width} 280`}
            role="img"
            aria-label="True intensity field and SMC posterior mean for the log-Gaussian Cox process."
          />
          <svg
            ref={tracesSvgRef}
            width={width}
            height={200}
            viewBox={`0 0 ${width} 200`}
            role="img"
            aria-label="ESS through the adaptive path and per-cell posterior trajectory."
          />
        </>
      )}
    </div>
  );
}

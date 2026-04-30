import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paretoKBand, PARETO_K_BAND_COLORS } from './shared/bayesian-ml-stacking';

// =============================================================================
// ParetoKExplorer — embedded alongside §4.4.
// Scatter plot of Pareto-k values per (observation, candidate) at n=100.
// Hover over any point links to (x_i, y_i) on the side scatter; filter k ≥ 0.5
// or 0.7; toggle between all candidates and a single one. The diagnostic is a
// model-criticism tool: high-k observations cluster near the discontinuity at
// x = 0.5, where the smooth-kernel candidates struggle.
// =============================================================================

const PANEL_HEIGHT = 360;

interface ParetoKData {
  n: number;
  x_train: number[];
  y_train: number[];
  /**
   * Per-(candidate, observation) Pareto-k diagnostic value, or `null` when the
   * diagnostic is undefined for that candidate. The closed-form GP LOO
   * (Rasmussen–Williams 5.12–5.13) does not produce Pareto-k values, so the GP
   * column is serialized as `null`s.
   */
  pareto_k: Record<string, (number | null)[]>;
}

type FilterMode = 'all' | 'k>=0.5' | 'k>=0.7';

export default function ParetoKExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [data, setData] = useState<ParetoKData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [candidate, setCandidate] = useState<string>('all');
  const [filter, setFilter] = useState<FilterMode>('all');
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/stacking-and-predictive-ensembles/pareto_k_n100.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: ParetoKData) => {
        if (!cancelled) setData(j);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const candidates = useMemo(() => (data ? Object.keys(data.pareto_k) : []), [data]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const height = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', height);
      svg.selectAll('*').remove();

      if (!data) {
        svg
          .append('text')
          .attr('x', w / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .attr('fill', '#666')
          .text(error ? `Awaiting precomputed data — ${error}` : 'Loading Pareto-k data...');
        return;
      }

      const margin = { top: 24, right: 16, bottom: 40, left: 48 };
      const innerW = (w - margin.left - margin.right - 24) / 2;
      const innerH = height - margin.top - margin.bottom;

      const candidatesToShow = candidate === 'all' ? candidates : [candidate];

      // Left panel: Pareto-k scatter (x = observation index, y = k).
      const left = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const allK: { idx: number; k: number; cand: string }[] = [];
      const skippedByCandidate = new Map<string, number>();
      candidatesToShow.forEach((cn) => {
        const series = data.pareto_k[cn] ?? [];
        let skipped = 0;
        series.forEach((k, idx) => {
          if (k == null || !Number.isFinite(k)) {
            skipped++;
            return;
          }
          if (filter === 'all' || (filter === 'k>=0.5' && k >= 0.5) || (filter === 'k>=0.7' && k >= 0.7)) {
            allK.push({ idx, k, cand: cn });
          }
        });
        if (skipped > 0) skippedByCandidate.set(cn, skipped);
      });

      const xK = d3.scaleLinear().domain([0, data.n]).range([0, innerW]);
      const yK = d3.scaleLinear().domain([0, Math.max(1.2, d3.max(allK, (d) => d.k) ?? 1)]).range([innerH, 0]);

      // Threshold lines.
      [0.5, 0.7, 1.0].forEach((t) => {
        left
          .append('line')
          .attr('x1', 0)
          .attr('x2', innerW)
          .attr('y1', yK(t))
          .attr('y2', yK(t))
          .attr('stroke', '#999')
          .attr('stroke-dasharray', '3 3')
          .attr('stroke-width', 0.6);
        left.append('text').attr('x', innerW - 24).attr('y', yK(t) - 2).attr('font-size', 9).attr('fill', '#666').text(`k=${t}`);
      });

      left.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xK).ticks(5));
      left.append('g').call(d3.axisLeft(yK).ticks(5));
      left.append('text').attr('transform', `translate(${innerW / 2},${innerH + 28})`).attr('text-anchor', 'middle').attr('font-size', 11).text('observation index i');
      left.append('text').attr('transform', `translate(-32,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 11).text('Pareto k̂');

      left
        .selectAll('circle.dot')
        .data(allK)
        .join('circle')
        .attr('class', 'dot')
        .attr('cx', (d) => xK(d.idx))
        .attr('cy', (d) => yK(d.k))
        .attr('r', 3)
        .attr('fill', (d) => PARETO_K_BAND_COLORS[paretoKBand(d.k)])
        .attr('opacity', 0.78)
        .style('cursor', 'pointer')
        .on('mouseover', (_, d) => setHoverIdx(d.idx))
        .on('mouseout', () => setHoverIdx(null));

      left.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).text(candidate === 'all' ? 'All candidates' : candidate);

      // Annotate any candidate whose Pareto-k values are not defined (e.g. GP under
      // closed-form LOO). The diagnostic is a PSIS-LOO artifact, so it's not
      // available for every candidate; we say so explicitly rather than silently
      // dropping points.
      if (skippedByCandidate.size > 0) {
        const note = Array.from(skippedByCandidate.entries())
          .map(([c, n]) => `${c}: ${n} (closed-form LOO)`)
          .join(', ');
        left
          .append('text')
          .attr('x', innerW / 2)
          .attr('y', innerH + 22)
          .attr('text-anchor', 'middle')
          .attr('font-size', 9)
          .attr('fill', '#666')
          .text(`Pareto-k undefined: ${note}`);
      }

      // Right panel: data scatter (x_i, y_i), highlighting the hovered observation.
      const right = svg.append('g').attr('transform', `translate(${margin.left + innerW + 24},${margin.top})`);
      const xD = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yD = d3
        .scaleLinear()
        .domain([d3.min(data.y_train) ?? -2, d3.max(data.y_train) ?? 2])
        .range([innerH, 0]);
      right.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xD).ticks(5));
      right.append('g').call(d3.axisLeft(yD).ticks(5));
      right.append('text').attr('transform', `translate(${innerW / 2},${innerH + 28})`).attr('text-anchor', 'middle').attr('font-size', 11).text('x');
      right.append('text').attr('transform', `translate(-32,${innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 11).text('y');

      right
        .selectAll('circle.data')
        .data(data.x_train.map((x, i) => ({ x, y: data.y_train[i], i })))
        .join('circle')
        .attr('class', 'data')
        .attr('cx', (d) => xD(d.x))
        .attr('cy', (d) => yD(d.y))
        .attr('r', (d) => (hoverIdx === d.i ? 6 : 3))
        .attr('fill', (d) => (hoverIdx === d.i ? '#000' : '#7f7f7f'))
        .attr('opacity', (d) => (hoverIdx === d.i ? 1 : 0.55));

      right.append('line').attr('x1', xD(0.5)).attr('x2', xD(0.5)).attr('y1', 0).attr('y2', innerH).attr('stroke', '#888').attr('stroke-dasharray', '3 3').attr('stroke-width', 0.6);
      right.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).text('Training data (linked)');
    },
    [data, error, candidate, filter, hoverIdx, candidates, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap gap-3 text-sm">
        <label>
          candidate:&nbsp;
          <select value={candidate} onChange={(e) => setCandidate(e.target.value)} className="rounded border px-1">
            <option value="all">all</option>
            {candidates.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label>
          filter:&nbsp;
          <select value={filter} onChange={(e) => setFilter(e.target.value as FilterMode)} className="rounded border px-1">
            <option value="all">all</option>
            <option value="k>=0.5">k ≥ 0.5</option>
            <option value="k>=0.7">k ≥ 0.7</option>
          </select>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
    </div>
  );
}

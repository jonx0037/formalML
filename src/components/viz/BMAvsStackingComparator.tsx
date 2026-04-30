import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteStacking } from './shared/bayesian-ml-stacking';

// =============================================================================
// BMAvsStackingComparator — embedded alongside §4.3.
// Side-by-side bar charts of BMA and stacking weights at four §4 candidates,
// with K_eff annotation. Loads precomputed PyMC pipeline outputs from
// src/data/sampleData/stacking-and-predictive-ensembles/pymc_pipeline_n*.json.
// Snap slider over n ∈ {40, 60, 80, 100} cached values; toggle weights vs.
// elpd_loo deltas.
// =============================================================================

const PANEL_HEIGHT = 320;
const N_VALUES = [40, 60, 80, 100] as const;

interface PipelineData {
  n: number;
  candidates: string[];
  elpd_loo: Record<string, number>;
  w_bma: Record<string, number>;
  w_stack: Record<string, number>;
  K_eff_bma: number;
  K_eff_stack: number;
  holdout_log_score: { single_best: number; bma_mixture: number; stacking_mixture: number };
}

type DisplayMode = 'weights' | 'delta_elpd';

export default function BMAvsStackingComparator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [nValue, setNValue] = useState<number>(100);
  const [mode, setMode] = useState<DisplayMode>('weights');
  const [data, setData] = useState<Record<number, PipelineData | null>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const entries = await Promise.all(
          N_VALUES.map(async (n) => {
            const res = await fetch(`/sample-data/stacking-and-predictive-ensembles/pymc_pipeline_n${n}.json`);
            if (!res.ok) throw new Error(`HTTP ${res.status} for n=${n}`);
            const j = (await res.json()) as PipelineData;
            return [n, j] as const;
          }),
        );
        if (!cancelled) {
          const next: Record<number, PipelineData> = {};
          for (const [n, j] of entries) next[n] = j;
          setData(next);
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        if (!cancelled) setError(msg);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const current = data[nValue] ?? null;

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const height = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', height);
      svg.selectAll('*').remove();

      if (!current) {
        svg
          .append('text')
          .attr('x', w / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .attr('fill', '#666')
          .text(error ? `Awaiting precomputed data — ${error}` : 'Loading precomputed PyMC pipeline outputs...');
        return;
      }

      const margin = { top: 24, right: 12, bottom: 40, left: 50 };
      const innerW = w - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const colW = (innerW - 24) / 2;

      const candidates = current.candidates;
      const candidateColors: Record<string, string> = {
        BLR: paletteStacking.blr,
        'BPR-4': paletteStacking.bpr,
        GP: paletteStacking.gp,
        BART: paletteStacking.bart,
      };

      const rowHeight = innerH;
      const xScale = d3.scaleBand<string>().domain(candidates).range([0, colW]).padding(0.18);

      let yScale: d3.ScaleLinear<number, number>;
      let valueA: (k: string) => number;
      let valueB: (k: string) => number;
      let yLabel = '';
      if (mode === 'weights') {
        yScale = d3.scaleLinear().domain([0, 1]).range([rowHeight, 0]);
        valueA = (k) => current.w_bma[k] ?? 0;
        valueB = (k) => current.w_stack[k] ?? 0;
        yLabel = 'weight';
      } else {
        const elpdMax = d3.max(candidates, (k) => current.elpd_loo[k]) ?? 0;
        const deltas = candidates.map((k) => current.elpd_loo[k] - elpdMax);
        const lo = (d3.min(deltas) ?? 0) - 0.5;
        yScale = d3.scaleLinear().domain([lo, 0]).range([rowHeight, 0]);
        valueA = (k) => current.elpd_loo[k] - elpdMax;
        valueB = valueA;
        yLabel = 'Δ elpd_loo';
      }

      const drawCol = (root: d3.Selection<SVGGElement, unknown, null, undefined>, title: string, getValue: (k: string) => number, kEff: number) => {
        root.append('text').attr('x', colW / 2).attr('y', -8).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).text(title);
        root.append('g').attr('transform', `translate(0,${rowHeight})`).call(d3.axisBottom(xScale));
        root.append('g').call(d3.axisLeft(yScale).ticks(5));
        root
          .selectAll('rect.bar')
          .data(candidates)
          .join('rect')
          .attr('class', 'bar')
          .attr('x', (k) => xScale(k) ?? 0)
          .attr('width', xScale.bandwidth())
          .attr('y', (k) => Math.min(yScale(getValue(k)), yScale(0)))
          .attr('height', (k) => Math.abs(yScale(getValue(k)) - yScale(0)))
          .attr('fill', (k) => candidateColors[k] ?? '#888');
        root
          .selectAll('text.bar-value')
          .data(candidates)
          .join('text')
          .attr('class', 'bar-value')
          .attr('x', (k) => (xScale(k) ?? 0) + xScale.bandwidth() / 2)
          .attr('y', (k) => yScale(getValue(k)) - 3)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .text((k) => getValue(k).toFixed(3));
        if (mode === 'weights') {
          root
            .append('text')
            .attr('x', colW - 4)
            .attr('y', 12)
            .attr('text-anchor', 'end')
            .attr('font-size', 10)
            .attr('fill', '#444')
            .text(`K_eff = ${kEff.toFixed(2)}`);
        }
      };

      const colA = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      drawCol(colA as any, 'BMA', valueA, current.K_eff_bma);

      const colBOffset = margin.left + colW + 24;
      const colB = svg.append('g').attr('transform', `translate(${colBOffset},${margin.top})`);
      drawCol(colB as any, 'Stacking', valueB, current.K_eff_stack);

      // Y-axis label.
      svg
        .append('text')
        .attr('transform', `translate(14,${margin.top + rowHeight / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .text(yLabel);

      // Held-out log-score caption.
      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', height - 6)
        .attr('text-anchor', 'middle')
        .attr('font-size', 10)
        .attr('fill', '#444')
        .text(
          `Held-out log-score / obs (200-pt grid):  best single = ${current.holdout_log_score.single_best.toFixed(4)},  BMA = ${current.holdout_log_score.bma_mixture.toFixed(4)},  stacking = ${current.holdout_log_score.stacking_mixture.toFixed(4)}`,
        );
    },
    [current, mode, error, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap gap-3 text-sm">
        <label>
          n:&nbsp;
          <select value={nValue} onChange={(e) => setNValue(Number(e.target.value))} className="rounded border px-1">
            {N_VALUES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          display:&nbsp;
          <select value={mode} onChange={(e) => setMode(e.target.value as DisplayMode)} className="rounded border px-1">
            <option value="weights">weights</option>
            <option value="delta_elpd">Δ elpd_loo</option>
          </select>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
    </div>
  );
}

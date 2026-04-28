import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  dedhEstimator,
  hillEstimator,
  mulberry32,
  pickandsEstimator,
  sampleParent,
  trueXi,
  type ParentPreset,
} from '../../data/extreme-value-theory';

// =============================================================================
// HillPlotExplorer — embedded after Figure 5.4 in §5.5.
//
// Single-panel overlay of the three classical tail-index estimators (Hill,
// Pickands, DEdH) as functions of the upper-order-statistic count k. A
// draggable k cursor (slider) reveals the three estimator values at the
// chosen k; a dashed horizontal line marks ξ_true when the parent has a
// known truth.
//
// Implementation details:
//   - Sample N = 10,000 raw observations from the chosen parent.
//   - Evaluate Hill / Pickands / DEdH on an integer k-grid from k = 5 to
//     k = N/2 (Hill, DEdH) or k = ⌊(N-1)/4⌋ (Pickands valid range).
//   - Pickands fails when 4k ≥ n; the slider greys-out the right portion of
//     the track to communicate this. The other two traces continue past
//     k = N/4.
//
// Brief §11.3 + the §5.5 textual setup:
//   - Hill: ξ > 0 only (Fréchet); MLE for ξ in the GPD restricted to ξ > 0.
//   - Pickands: any ξ ∈ ℝ; higher variance than Hill at typical ξ.
//   - DEdH: any ξ ∈ ℝ; bias-corrected; uses the first two log-spacing moments.
// =============================================================================

const HEIGHT = 360;

const N_SAMPLE = 10_000;
const K_MIN = 5;
const K_MAX_DEFAULT = Math.floor(N_SAMPLE / 2); // for Hill / DEdH
const K_MAX_PICKANDS = Math.floor((N_SAMPLE - 1) / 4); // 4k < n requirement
const K_GRID_STEP = 5;

const COLOR_HILL = '#DC2626';     // red
const COLOR_PICKANDS = '#F59E0B'; // amber
const COLOR_DEDH = '#7C3AED';     // violet
const COLOR_TRUTH = '#6B7280';    // gray (truth line)

const PARENT_LABELS: Record<'normal' | 'pareto2' | 't3', string> = {
  normal: 'Normal (truth ξ = 0)',
  pareto2: 'Pareto(α=2) (truth ξ = 0.5)',
  t3: 'Student-t₃ (truth ξ ≈ 0.333)',
};

const fmt = (x: number, digits = 3) => (Number.isFinite(x) ? x.toFixed(digits) : '—');

export default function HillPlotExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [parent, setParent] = useState<'normal' | 'pareto2' | 't3'>('pareto2');
  const [seed, setSeed] = useState(42);
  const [k, setK] = useState(100);

  // ── Generate sample once per (parent, seed); evaluate two estimator grids ──
  // Sample once per (parent, seed). Two evaluation grids are computed:
  //   - kArr (K_GRID_STEP spacing): used for plotting the trace polylines
  //     so the SVG path doesn't carry ~5000 vertices per estimator.
  //   - cursorKArr (every integer in [K_MIN, K_MAX_DEFAULT]): used for the
  //     cursor readout, so slider drags are O(1) lookup-only.
  // Both grids share the same underlying sorted sample inside each estimator,
  // so the upfront cost is dominated by three Float64Array.slice().sort() calls.
  // Hill / DEdH need positive observations (Hill takes log directly). For
  // Normal-parent samples the bulk is positive at large k (the upper-order
  // statistics are positive once we're past the 50th percentile), so this
  // works in practice. For data with mass on the negative axis (t₃), Hill
  // returns NaN where log(X_(n-i+1)) is undefined.
  const traces = useMemo(() => {
    const rng = mulberry32(seed);
    const data = sampleParent(parent, N_SAMPLE, rng);

    const plotKVec: number[] = [];
    for (let kVal = K_MIN; kVal <= K_MAX_DEFAULT; kVal += K_GRID_STEP) {
      plotKVec.push(kVal);
    }
    const kArr = new Int32Array(plotKVec);
    const hill = hillEstimator(data, kArr);
    const pickands = pickandsEstimator(data, kArr);
    const dedh = dedhEstimator(data, kArr);

    const cursorKArr = new Int32Array(K_MAX_DEFAULT - K_MIN + 1);
    for (let i = 0; i < cursorKArr.length; i++) cursorKArr[i] = K_MIN + i;
    const cursorHill = hillEstimator(data, cursorKArr);
    const cursorPickands = pickandsEstimator(data, cursorKArr);
    const cursorDedh = dedhEstimator(data, cursorKArr);

    return { kArr, hill, pickands, dedh, cursorHill, cursorPickands, cursorDedh };
  }, [parent, seed]);

  // ── Estimator values at the cursor k (O(1) lookup into integer-k grids) ──
  const cursorIdx = Math.max(0, Math.min(K_MAX_DEFAULT - K_MIN, Math.round(k) - K_MIN));
  const cursorValues = {
    hill: traces.cursorHill[cursorIdx],
    pickands: traces.cursorPickands[cursorIdx],
    dedh: traces.cursorDedh[cursorIdx],
  };

  const truthXi = trueXi(parent);

  // ── y-axis range ─────────────────────────────────────────────────
  const yRange = useMemo<[number, number]>(() => {
    let lo = -0.5, hi = 1.0;
    for (const trace of [traces.hill, traces.pickands, traces.dedh]) {
      for (let i = 0; i < trace.length; i++) {
        const v = trace[i];
        if (Number.isFinite(v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
    if (Number.isFinite(truthXi)) {
      lo = Math.min(lo, truthXi - 0.2);
      hi = Math.max(hi, truthXi + 0.2);
    }
    return [lo, hi];
  }, [traces, truthXi]);

  // ── D3 chart ─────────────────────────────────────────────────────
  const chartRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      const margin = { top: 32, right: 28, bottom: 40, left: 52 };
      const w = containerWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([K_MIN, K_MAX_DEFAULT]).range([0, w]).clamp(true);
      const yScale = d3.scaleLinear().domain(yRange).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(8, '~s'))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');

      g.append('text').attr('x', w / 2).attr('y', h + 32)
        .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle')
        .style('font-size', '11px').text('k (number of upper order statistics, log scale)');
      g.append('text').attr('x', -38).attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle')
        .style('font-size', '11px')
        .attr('transform', `rotate(-90,-38,${h / 2})`).text('ξ̂(k)');

      // Greyed-out region where 4k ≥ n (Pickands undefined)
      g.append('rect')
        .attr('x', xScale(K_MAX_PICKANDS + 1))
        .attr('y', 0)
        .attr('width', Math.max(0, w - xScale(K_MAX_PICKANDS + 1)))
        .attr('height', h)
        .style('fill', 'var(--color-muted-border)')
        .style('opacity', 0.18);
      g.append('text')
        .attr('x', xScale(K_MAX_PICKANDS) - 4)
        .attr('y', 12)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px').style('font-style', 'italic')
        .style('text-anchor', 'end')
        .text('4k ≥ n →');

      // Truth line (when finite — i.e. for known parent presets)
      if (Number.isFinite(truthXi)) {
        g.append('line')
          .attr('x1', 0).attr('x2', w)
          .attr('y1', yScale(truthXi)).attr('y2', yScale(truthXi))
          .style('stroke', COLOR_TRUTH).style('stroke-dasharray', '4 3')
          .style('stroke-width', 1).style('opacity', 0.7);
        g.append('text')
          .attr('x', w - 6).attr('y', yScale(truthXi) - 4)
          .style('fill', COLOR_TRUTH).style('text-anchor', 'end')
          .style('font-size', '10px').style('font-style', 'italic')
          .text(`ξ_true = ${fmt(truthXi, 3)}`);
      }

      // Trace plots
      const drawTrace = (
        values: Float64Array,
        color: string,
        strokeWidth: number,
      ) => {
        const line = d3.line<number>()
          .defined((i) => Number.isFinite(values[i]))
          .x((i) => xScale(traces.kArr[i]))
          .y((i) => yScale(values[i]));
        g.append('path')
          .datum(d3.range(traces.kArr.length))
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', color)
          .style('stroke-width', strokeWidth);
      };
      drawTrace(traces.hill, COLOR_HILL, 2.0);
      drawTrace(traces.pickands, COLOR_PICKANDS, 1.6);
      drawTrace(traces.dedh, COLOR_DEDH, 2.0);

      // Cursor (vertical line at current k)
      const cursorX = xScale(k);
      g.append('line')
        .attr('x1', cursorX).attr('x2', cursorX)
        .attr('y1', 0).attr('y2', h)
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', 1.2)
        .style('stroke-dasharray', '5 3')
        .style('opacity', 0.6);

      // Cursor dots
      const drawDot = (val: number, color: string) => {
        if (!Number.isFinite(val)) return;
        g.append('circle')
          .attr('cx', cursorX).attr('cy', yScale(val))
          .attr('r', 4.5).style('fill', color).style('stroke', 'var(--color-bg)').style('stroke-width', 1.5);
      };
      drawDot(cursorValues.hill, COLOR_HILL);
      drawDot(cursorValues.pickands, COLOR_PICKANDS);
      drawDot(cursorValues.dedh, COLOR_DEDH);

      // Title
      svg.append('text')
        .attr('x', containerWidth / 2).attr('y', 16)
        .style('fill', 'var(--color-text)').style('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600')
        .text(`Tail-index estimators on ${PARENT_LABELS[parent]} (N = ${N_SAMPLE.toLocaleString()})`);
      // (in-chart legend omitted — the colored readout boxes below the chart
      // already convey the estimator → color mapping)
    },
    [containerWidth, traces, yRange, truthXi, k, parent, cursorValues],
  );

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3">
      <svg
        role="img"
        aria-label={`Tail-index estimator traces (Hill in red, Pickands in amber, DEdH in violet) versus k on a log-x axis. Truth line at xi = ${fmt(truthXi, 2)}. Greyed-out right region marks where 4k >= n and Pickands becomes undefined.`}
        ref={chartRef}
        width={containerWidth || 800}
        height={HEIGHT}
      />

      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <div className="rounded border p-2" style={{ borderColor: COLOR_HILL + '55' }}>
          <div className="font-medium" style={{ color: COLOR_HILL }}>Hill ξ̂(k = {k})</div>
          <div className="font-mono mt-1">{fmt(cursorValues.hill, 3)}</div>
        </div>
        <div className="rounded border p-2" style={{ borderColor: COLOR_PICKANDS + '55' }}>
          <div className="font-medium" style={{ color: COLOR_PICKANDS }}>Pickands ξ̂(k = {k})</div>
          <div className="font-mono mt-1">{fmt(cursorValues.pickands, 3)}</div>
          {4 * k >= N_SAMPLE && (
            <div className="text-[10px] italic text-[var(--color-text-secondary)]">undefined for 4k ≥ n</div>
          )}
        </div>
        <div className="rounded border p-2" style={{ borderColor: COLOR_DEDH + '55' }}>
          <div className="font-medium" style={{ color: COLOR_DEDH }}>DEdH ξ̂(k = {k})</div>
          <div className="font-mono mt-1">{fmt(cursorValues.dedh, 3)}</div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span>Parent</span>
          <select
            value={parent}
            onChange={(e) => setParent(e.target.value as 'normal' | 'pareto2' | 't3')}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-[var(--color-text)]"
          >
            {(Object.keys(PARENT_LABELS) as ('normal' | 'pareto2' | 't3')[]).map((p) => (
              <option key={p} value={p}>{PARENT_LABELS[p]}</option>
            ))}
          </select>
        </label>

        <label className="flex flex-1 items-center gap-2 text-[var(--color-text-secondary)] min-w-[260px]">
          <span className="font-mono w-4">k</span>
          <input
            type="range" min={K_MIN} max={K_MAX_DEFAULT} step={1}
            value={k} onChange={(e) => setK(parseInt(e.target.value, 10))}
            className="flex-1 accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{k}</span>
        </label>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
        >
          Resample
        </button>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Drag the k cursor to read off the three estimator values at any k. Hill is the smoothest in the middle range — its asymptotic variance ξ²/k is the smallest of the three for typical ξ. Pickands is noisier (its asymptotic variance is roughly 5–10× Hill's at ξ ≈ 0.5) but works for any ξ ∈ ℝ. DEdH tracks Hill in the Fréchet plateau and corrects for the bias Hill exhibits at ξ ≤ 0. The dashed grey region on the right marks where 4k ≥ N — Pickands needs four nested upper-order-statistic levels and is undefined past there. The horizontal dashed line is ξ_true; in production tail-index analysis there is no ξ_true to draw, so the standard practice is to choose k from a stable plateau in the middle range and report the corresponding ξ̂.
      </p>
    </div>
  );
}

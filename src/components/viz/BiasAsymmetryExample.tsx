import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// BiasAsymmetryExample — §7 Wang–Blei consistency on hierarchical-vs-pooled
// =============================================================================
// Loads precomputed JSON at /sample-data/variational-bayes-for-model-selection/
// bias_flip.json — output of `precompute_bias_flip.py` (PyMC ADVI + SMC on a
// hierarchical Bayesian logistic regression vs pooled comparison).
//
// Two-panel layout:
//   Left: per-data-point log-evidence vs n_per_group, four traces
//         (ELBO/SMC × M_hier/M_pool). M_hier always above M_pool — ranking
//         preserved at every n.
//   Right: bar chart at each n showing |ELBO − SMC| for both models alongside
//          the model gap, log-scale y-axis. Annotation: model_gap / max_bias_gap.
//
// JSON fix: the bias_flip.json file ships with literal `NaN` for pareto_k
// (PyMC version-incompatibility per brief §9 Q2′). We replace `NaN` with
// `null` before JSON.parse since pareto_k is non-blocking and unused here.
// =============================================================================

const PANEL_HEIGHT = 360;
const MARGIN = { top: 22, right: 18, bottom: 56, left: 56 };

const HIER_COLOR = '#c0504d';
const POOL_COLOR = '#1f4e79';

interface ModelStats {
  vbms_elbo: number;
  smc_log_evidence: number;
  pareto_k: number | null;
  elbo_history_thinned?: number[];
}

interface ResultRow {
  n_per_group: number;
  total_n: number;
  M_hier: ModelStats;
  M_pool: ModelStats;
}

interface BiasFlipPayload {
  metadata: { n_grid: number[]; true_beta: number; alpha_group_sd: number };
  results: ResultRow[];
}

const FETCH_URL = '/sample-data/variational-bayes-for-model-selection/bias_flip.json';

export default function BiasAsymmetryExample() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<BiasFlipPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(FETCH_URL)
      .then((r) => r.text())
      .then((text) => {
        // bias_flip.json contains literal `NaN` for pareto_k; replace with
        // null so JSON.parse succeeds. Pareto-k̂ is non-blocking per brief Q2′.
        const cleaned = text.replace(/\bNaN\b/g, 'null');
        const parsed: BiasFlipPayload = JSON.parse(cleaned);
        if (!cancelled) setPayload(parsed);
      })
      .catch((err) => {
        if (!cancelled) setError(`Failed to load bias_flip.json: ${(err as Error).message}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const isMobile = containerWidth > 0 && containerWidth < 640;
  const panelW = isMobile ? containerWidth : Math.max(0, Math.floor(containerWidth / 2) - 8);

  const stats = useMemo(() => {
    if (!payload) return null;
    // Track the per-row ratio gap/max(bias_h, bias_p) and report the maximum
    // observed across rows along with the row that achieves it. The previous
    // implementation reported max(gap) / max(bias) — those maxes can come from
    // different rows, overstating the ratio.
    let bestRatio = 0;
    let bestRow: ResultRow | null = null;
    for (const row of payload.results) {
      const biasH = Math.abs(row.M_hier.vbms_elbo - row.M_hier.smc_log_evidence);
      const biasP = Math.abs(row.M_pool.vbms_elbo - row.M_pool.smc_log_evidence);
      const gap = Math.abs(row.M_hier.smc_log_evidence - row.M_pool.smc_log_evidence);
      const ratio = gap / Math.max(Math.max(biasH, biasP), 1e-9);
      if (!bestRow || ratio > bestRatio) {
        bestRatio = ratio;
        bestRow = row;
      }
    }
    return {
      ratioMaxN: bestRatio,
      maxRow: bestRow,
    };
  }, [payload]);

  const leftRef = useRef<SVGSVGElement>(null);
  const rightRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const leftSvg = leftRef.current;
    if (!leftSvg || !payload || panelW <= 0) return;
    const W = panelW;
    const H = PANEL_HEIGHT;
    const w = W - MARGIN.left - MARGIN.right;
    const h = H - MARGIN.top - MARGIN.bottom;
    const sel = d3.select(leftSvg);
    sel.selectAll('*').remove();
    const g = sel.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const data = payload.results;
    const xValues = payload.metadata.n_grid;
    const xScale = d3
      .scaleLog()
      .domain([Math.min(...xValues) * 0.9, Math.max(...xValues) * 1.1])
      .range([0, w]);
    const ys = data.flatMap((row) => [
      row.M_hier.vbms_elbo / row.total_n,
      row.M_hier.smc_log_evidence / row.total_n,
      row.M_pool.vbms_elbo / row.total_n,
      row.M_pool.smc_log_evidence / row.total_n,
    ]);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const yPad = (yMax - yMin) * 0.1;
    const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).tickValues(xValues).tickFormat((v) => String(v)))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '11px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '11px');
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 36)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '12px')
      .text('Observations per group');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2)
      .attr('y', -42)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '12px')
      .text('Per-data-point log-evidence (nats)');

    type Trace = { name: string; color: string; dash: boolean; values: { x: number; y: number }[] };
    const traces: Trace[] = [
      {
        name: 'M_hier ELBO',
        color: HIER_COLOR,
        dash: false,
        values: data.map((row) => ({ x: row.n_per_group, y: row.M_hier.vbms_elbo / row.total_n })),
      },
      {
        name: 'M_hier SMC',
        color: HIER_COLOR,
        dash: true,
        values: data.map((row) => ({ x: row.n_per_group, y: row.M_hier.smc_log_evidence / row.total_n })),
      },
      {
        name: 'M_pool ELBO',
        color: POOL_COLOR,
        dash: false,
        values: data.map((row) => ({ x: row.n_per_group, y: row.M_pool.vbms_elbo / row.total_n })),
      },
      {
        name: 'M_pool SMC',
        color: POOL_COLOR,
        dash: true,
        values: data.map((row) => ({ x: row.n_per_group, y: row.M_pool.smc_log_evidence / row.total_n })),
      },
    ];
    const line = d3
      .line<{ x: number; y: number }>()
      .x((p) => xScale(p.x))
      .y((p) => yScale(p.y));
    traces.forEach((tr) => {
      g.append('path')
        .datum(tr.values)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', tr.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', tr.dash ? '5,3' : '');
      g.selectAll(`circle.${tr.name.replace(/\s/g, '_')}`)
        .data(tr.values)
        .enter()
        .append('circle')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', 3)
        .attr('fill', tr.color);
    });
    const legend = g.append('g').attr('transform', `translate(${w - 130},6)`);
    traces.forEach((tr, i) => {
      const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
      row
        .append('line')
        .attr('x1', 0)
        .attr('x2', 22)
        .attr('y1', 0)
        .attr('y2', 0)
        .style('stroke', tr.color)
        .style('stroke-width', 2)
        .style('stroke-dasharray', tr.dash ? '5,3' : '');
      row
        .append('text')
        .attr('x', 28)
        .attr('y', 4)
        .style('fill', 'var(--color-text)')
        .style('font-size', '11px')
        .text(tr.name);
    });
  }, [payload, panelW]);

  useEffect(() => {
    const rightSvg = rightRef.current;
    if (!rightSvg || !payload || !stats || panelW <= 0) return;
    const W = panelW;
    const H = PANEL_HEIGHT;
    const w = W - MARGIN.left - MARGIN.right;
    const h = H - MARGIN.top - MARGIN.bottom;
    const sel = d3.select(rightSvg);
    sel.selectAll('*').remove();
    const g = sel.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const data = payload.results;
    type Bar = { n: number; biasH: number; biasP: number; gap: number };
    const bars: Bar[] = data.map((row) => ({
      n: row.n_per_group,
      biasH: Math.abs(row.M_hier.vbms_elbo - row.M_hier.smc_log_evidence),
      biasP: Math.abs(row.M_pool.vbms_elbo - row.M_pool.smc_log_evidence),
      gap: Math.abs(row.M_hier.smc_log_evidence - row.M_pool.smc_log_evidence),
    }));
    const xScale = d3
      .scaleBand<number>()
      .domain(bars.map((b) => b.n))
      .range([0, w])
      .padding(0.18);
    const groupScale = d3
      .scaleBand<string>()
      .domain(['biasH', 'biasP', 'gap'])
      .range([0, xScale.bandwidth()])
      .padding(0.08);
    const allValues = bars.flatMap((b) => [Math.max(b.biasH, 0.001), Math.max(b.biasP, 0.001), b.gap]);
    const yScale = d3
      .scaleLog()
      .domain([Math.max(0.01, Math.min(...allValues) * 0.5), Math.max(...allValues) * 1.2])
      .range([h, 0]);
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).tickFormat((v) => String(v)))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(6, '~g'))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '11px');
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 36)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '12px')
      .text('Observations per group');
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2)
      .attr('y', -42)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '12px')
      .text('Magnitude (nats, log scale)');

    bars.forEach((b) => {
      const x0 = xScale(b.n) ?? 0;
      const drawBar = (key: 'biasH' | 'biasP' | 'gap', value: number, color: string, opacity: number) => {
        const bw = groupScale.bandwidth();
        const bx = x0 + (groupScale(key) ?? 0);
        const safeVal = Math.max(value, 0.01);
        const yTop = yScale(safeVal);
        g.append('rect')
          .attr('x', bx)
          .attr('y', yTop)
          .attr('width', bw)
          .attr('height', Math.max(1, h - yTop))
          .attr('fill', color)
          .attr('fill-opacity', opacity);
      };
      drawBar('biasH', b.biasH, HIER_COLOR, 0.85);
      drawBar('biasP', b.biasP, POOL_COLOR, 0.85);
      drawBar('gap', b.gap, '#534AB7', 0.85);
    });

    const legend = g.append('g').attr('transform', `translate(${w - 160},6)`);
    [
      { color: HIER_COLOR, label: '|ELBO − SMC| hier' },
      { color: POOL_COLOR, label: '|ELBO − SMC| pool' },
      { color: '#534AB7', label: 'model gap |hier − pool|' },
    ].forEach((entry, i) => {
      const row = legend.append('g').attr('transform', `translate(0,${i * 14})`);
      row.append('rect').attr('width', 10).attr('height', 10).attr('fill', entry.color).attr('fill-opacity', 0.85);
      row.append('text').attr('x', 14).attr('y', 9).style('fill', 'var(--color-text)').style('font-size', '10px').text(entry.label);
    });

    g.append('text')
      .attr('x', w - 6)
      .attr('y', h - 6)
      .attr('text-anchor', 'end')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '11px')
      .style('font-style', 'italic')
      .text(`max ratio gap/bias = ${stats.ratioMaxN.toFixed(0)}×`);
  }, [payload, stats, panelW]);

  return (
    <div ref={containerRef} className="my-6">
      <div
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          borderRadius: 8,
          padding: 16,
          fontFamily: 'var(--font-sans)',
        }}
      >
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text)' }}>
            Wang–Blei consistency on hierarchical vs pooled Bayesian logistic regression
          </div>
          <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
            Both VBMS (ELBO) and SMC consistently rank M<sub>hier</sub> &gt; M<sub>pool</sub> at every n. Bias asymmetric (mean-field hits hierarchical posterior correlations); model gap dominates.
          </div>
        </div>
        {error && (
          <div style={{ color: '#b91c1c', padding: 8, fontSize: 12 }}>
            {error}
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
          <svg ref={leftRef} width={panelW} height={PANEL_HEIGHT} role="img" aria-label="Per-data-point log-evidence vs n for both models showing M_hier dominates." />
          <svg ref={rightRef} width={panelW} height={PANEL_HEIGHT} role="img" aria-label="Bar chart of bias and model gap on log scale across n values." />
        </div>
        {!payload && !error && (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--color-text-secondary)', fontSize: 12 }}>
            Loading bias_flip.json…
          </div>
        )}
      </div>
    </div>
  );
}

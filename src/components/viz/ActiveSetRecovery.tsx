import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// ActiveSetRecovery — §9 four-prior comparison on synthetic high-dim regression.
//
// Loads /sample-data/sparse-bayesian-priors/synthetic_high_dim.json (produced
// by precompute_synthetic_high_dim.py at production scale p=200, n=100, k=10)
// and renders a coefficient-wise posterior interval plot for the user-selected
// prior. Threshold slider switches the inclusion-probability annotation among
// the precomputed thresholds {0.01, 0.05, 0.10, 0.20, 0.50}.
//
// Per CLAUDE.md fetch-based-viz rule: container <div ref={containerRef}> is
// mounted from first render so useResizeObserver attaches; only the inner
// content swaps between loading / SVG.
// =============================================================================

const PANEL_HEIGHT = 420;
const MARGIN = { top: 24, right: 32, bottom: 50, left: 56 };

interface PriorRecord {
  name: string;
  color: string;
  beta_mean: number[];
  beta_q05: number[];
  beta_q95: number[];
  inclusion: Record<string, number[]>;
  rmse: number;
  test_ll_mean: number;
  test_ll_se: number;
  divergences: number;
  rhat_beta_max: number;
  ess_beta_min: number;
}

interface Payload {
  metadata: {
    p: number;
    n_train: number;
    n_test: number;
    k_true: number;
    active_indices: number[];
    beta_true: number[];
    inclusion_thresholds: number[];
  };
  priors: PriorRecord[];
}

const THRESHOLD_KEYS = ['thr_0_01', 'thr_0_05', 'thr_0_10', 'thr_0_20', 'thr_0_50'] as const;
const THRESHOLD_LABELS = ['0.01', '0.05', '0.10', '0.20', '0.50'];

export default function ActiveSetRecovery() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [priorIdx, setPriorIdx] = useState(1); // default to Reg. horseshoe
  const [thrIdx, setThrIdx] = useState(2); // default to 0.10

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/sparse-bayesian-priors/synthetic_high_dim.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<Payload>;
      })
      .then((j) => {
        if (!cancelled) setPayload(j);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedPrior = payload?.priors[priorIdx];

  const inclusionAtActive = useMemo(() => {
    if (!payload || !selectedPrior) return null;
    const key = THRESHOLD_KEYS[thrIdx];
    const incl = selectedPrior.inclusion[key];
    if (!incl) return null;
    return payload.metadata.active_indices.map((i) => ({
      idx: i,
      prob: incl[i],
    }));
  }, [payload, selectedPrior, thrIdx]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 720;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);

      if (!payload || !selectedPrior) {
        svg
          .append('text')
          .attr('x', w / 2)
          .attr('y', h / 2)
          .attr('text-anchor', 'middle')
          .style('font-size', '12px')
          .text(error ?? 'Loading synthetic_high_dim.json…');
        return;
      }

      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = h - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const p = payload.metadata.p;
      const xScale = d3.scaleLinear().domain([-1, p]).range([0, innerW]);
      const yMax = Math.max(
        d3.max(selectedPrior.beta_q95) ?? 1,
        Math.abs(d3.min(selectedPrior.beta_q05) ?? -1),
        ...payload.metadata.beta_true.map((v) => Math.abs(v)),
      );
      const yScale = d3
        .scaleLinear()
        .domain([-yMax * 1.1, yMax * 1.1])
        .range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(8));
      g.append('g').call(d3.axisLeft(yScale).ticks(6));

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text('coefficient index j');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '12px')
        .text('β_j (posterior 90% interval)');

      // Zero line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .attr('stroke', '#aaa')
        .attr('stroke-width', 1);

      // Posterior intervals
      for (let j = 0; j < p; j++) {
        const cx = xScale(j);
        g.append('line')
          .attr('x1', cx)
          .attr('x2', cx)
          .attr('y1', yScale(selectedPrior.beta_q05[j]))
          .attr('y2', yScale(selectedPrior.beta_q95[j]))
          .attr('stroke', selectedPrior.color)
          .attr('stroke-width', 0.9)
          .attr('opacity', 0.55);
        g.append('circle')
          .attr('cx', cx)
          .attr('cy', yScale(selectedPrior.beta_mean[j]))
          .attr('r', 1.4)
          .attr('fill', selectedPrior.color);
      }

      // True active markers
      payload.metadata.active_indices.forEach((j) => {
        g.append('path')
          .attr(
            'd',
            d3.symbol().type(d3.symbolCross).size(60)(),
          )
          .attr(
            'transform',
            `translate(${xScale(j)},${yScale(payload.metadata.beta_true[j])})`,
          )
          .attr('fill', '#222')
          .attr('stroke', '#fff')
          .attr('stroke-width', 0.5);
      });

      // Title with diagnostics
      g.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(
          `${selectedPrior.name} | divs = ${selectedPrior.divergences} | RMSE = ${selectedPrior.rmse.toFixed(3)} | test ll = ${selectedPrior.test_ll_mean.toFixed(3)}`,
        );
    },
    [payload, selectedPrior, error, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 760 }}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 18,
          marginBottom: 8,
          alignItems: 'center',
          fontSize: 13,
        }}
      >
        <label>
          Prior:{' '}
          <select
            value={priorIdx}
            onChange={(e) => setPriorIdx(Number(e.target.value))}
            disabled={!payload}
          >
            {payload?.priors.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Inclusion threshold |β| &gt; {THRESHOLD_LABELS[thrIdx]}{' '}
          <input
            type="range"
            min={0}
            max={THRESHOLD_LABELS.length - 1}
            step={1}
            value={thrIdx}
            onChange={(e) => setThrIdx(Number(e.target.value))}
            style={{ width: 160 }}
            disabled={!payload}
          />
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="Active-set recovery on synthetic regression" />
      {inclusionAtActive && (
        <div style={{ fontSize: 12, marginTop: 6, color: '#555' }}>
          Inclusion at true-active indices: {inclusionAtActive.map((a) => `j=${a.idx}: ${a.prob.toFixed(2)}`).join(', ')}
        </div>
      )}
    </div>
  );
}

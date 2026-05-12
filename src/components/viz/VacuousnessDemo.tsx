import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// VacuousnessDemo — §12.  Two panels:
//   Left  — threshold-class empirical gap vs Cor 3 bound (tight) overlaid with
//           the binary-MNIST MLP's empirical gap and classical Rademacher bound
//           (vacuous), all on the same log-y axis.
//   Right — distribution of MLP predicted P(y=1|x) on train, colored by true y.
//
// Consumes precomputed JSON from precompute_vacuousness.py.
// =============================================================================

const HEIGHT = 340;
const SM_BREAKPOINT = 1000;
const PAYLOAD_URL = '/sample-data/generalization-bounds/vacuousness.json';

type Payload = {
  config: { n_train: number; n_test: number; hidden_width: number; seed: number; delta: number };
  results: {
    train_accuracy: number;
    test_accuracy: number;
    empirical_gap: number;
    W1_op_norm: number;
    w2_l2_norm: number;
    R_X: number;
    classical_rademacher_bound: number;
    corollary3_bound: number;
    ratio_bound_over_gap: number;
  };
  train_probs: number[];
  test_probs: number[];
  y_train: number[];
  y_test: number[];
  comparison: {
    threshold_class: {
      n_grid: number[];
      emp_gap_mean: number[];
      bound: number[];
    };
  };
};

export default function VacuousnessDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(PAYLOAD_URL)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() as Promise<Payload>; })
      .then((p) => { if (alive) setPayload(p); })
      .catch((e) => { if (alive) setError(String(e)); });
    return () => { alive = false; };
  }, []);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor(containerWidth / 2);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || panelWidth <= 0) return;
      const margin = { top: 32, right: 12, bottom: 40, left: 56 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const tc = payload.comparison.threshold_class;
      const mlpGap = payload.results.empirical_gap;
      const mlpBound = payload.results.corollary3_bound;
      const xMin = Math.min(...tc.n_grid);
      const xMax = Math.max(...tc.n_grid, payload.config.n_train);
      const yMin = Math.min(mlpGap, ...tc.emp_gap_mean, ...tc.bound) * 0.7;
      const yMax = Math.max(mlpBound, ...tc.bound) * 1.3;

      const x = d3.scaleLog().domain([xMin, xMax]).range([0, w]);
      const y = d3.scaleLog().domain([Math.max(yMin, 1e-4), yMax]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('sample size n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('gap / bound');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Classical theory: tight on tabular, vacuous on MLPs');

      // Threshold class lines
      const empPts = tc.n_grid.map((n, i) => ({ n, v: tc.emp_gap_mean[i] }));
      const bndPts = tc.n_grid.map((n, i) => ({ n, v: tc.bound[i] }));
      const line = d3.line<{ n: number; v: number }>().x((p) => x(p.n)).y((p) => y(Math.max(1e-4, p.v)));
      g.append('path').datum(empPts).attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', line);
      g.append('path').datum(bndPts).attr('fill', 'none').style('stroke', '#1e40af').style('stroke-width', 1.5).style('stroke-dasharray', '4 3').attr('d', line);
      empPts.forEach((p) => g.append('circle').attr('cx', x(p.n)).attr('cy', y(Math.max(1e-4, p.v))).attr('r', 3).style('fill', '#3b82f6'));

      // Trivial bound = 1 reference
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(1)).attr('y2', y(1))
        .style('stroke', '#94a3b8').style('stroke-dasharray', '2 2');
      g.append('text').attr('x', 4).attr('y', y(1) - 4).style('font-size', '10px').style('fill', '#475569').text('trivial bound = 1');

      // MLP empirical gap (single point)
      g.append('circle').attr('cx', x(payload.config.n_train)).attr('cy', y(Math.max(1e-4, mlpGap))).attr('r', 6)
        .style('fill', '#16a34a').style('stroke', 'white').style('stroke-width', 1.5);
      g.append('text').attr('x', x(payload.config.n_train) - 8).attr('y', y(Math.max(1e-4, mlpGap)) - 8).attr('text-anchor', 'end')
        .style('font-size', '10px').style('fill', '#15803d').text(`MLP gap ${mlpGap.toFixed(4)}`);
      // MLP classical bound
      g.append('circle').attr('cx', x(payload.config.n_train)).attr('cy', y(mlpBound)).attr('r', 6)
        .style('fill', '#dc2626').style('stroke', 'white').style('stroke-width', 1.5);
      g.append('text').attr('x', x(payload.config.n_train) - 8).attr('y', y(mlpBound) - 8).attr('text-anchor', 'end')
        .style('font-size', '10px').style('fill', '#7f1d1d').text(`MLP bound ${mlpBound.toFixed(1)} (vacuous)`);

      // Legend
      const lg = g.append('g').attr('transform', `translate(${w - 220},${20})`);
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0).style('stroke', '#3b82f6').style('stroke-width', 2);
      lg.append('text').attr('x', 22).attr('y', 4).style('font-size', '11px').style('fill', 'var(--color-text)').text('threshold class: gap');
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 16).attr('y2', 16).style('stroke', '#1e40af').style('stroke-dasharray', '4 3');
      lg.append('text').attr('x', 22).attr('y', 20).style('font-size', '11px').style('fill', 'var(--color-text)').text('threshold class: Cor 3');
    },
    [payload, panelWidth],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || panelWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 40, left: 50 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const probsY0 = payload.train_probs.filter((_, i) => payload.y_train[i] === 0);
      const probsY1 = payload.train_probs.filter((_, i) => payload.y_train[i] === 1);
      const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const bins0 = d3.bin().domain([0, 1]).thresholds(30)(probsY0);
      const bins1 = d3.bin().domain([0, 1]).thresholds(30)(probsY1);
      const maxBin = Math.max(d3.max(bins0, (b) => b.length) ?? 1, d3.max(bins1, (b) => b.length) ?? 1);
      const y = d3.scaleLinear().domain([0, maxBin]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('predicted P(y = 1 | x)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -36).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('count');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text(`MLP predictions  (train acc ${payload.results.train_accuracy.toFixed(3)}, test acc ${payload.results.test_accuracy.toFixed(3)})`);

      g.selectAll('rect.y0').data(bins0).enter().append('rect')
        .attr('x', (d) => x(d.x0 ?? 0) + 1)
        .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 2))
        .attr('y', (d) => y(d.length))
        .attr('height', (d) => h - y(d.length))
        .style('fill', '#3b82f6').attr('fill-opacity', 0.55);
      g.selectAll('rect.y1').data(bins1).enter().append('rect')
        .attr('x', (d) => x(d.x0 ?? 0) + 1)
        .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 2))
        .attr('y', (d) => y(d.length))
        .attr('height', (d) => h - y(d.length))
        .style('fill', '#dc2626').attr('fill-opacity', 0.55);

      const lg = g.append('g').attr('transform', `translate(${w - 110},${10})`);
      lg.append('rect').attr('x', 0).attr('y', -8).attr('width', 12).attr('height', 12).style('fill', '#3b82f6').attr('fill-opacity', 0.55);
      lg.append('text').attr('x', 16).attr('y', 2).style('font-size', '11px').style('fill', 'var(--color-text)').text('y = 0 (digit 0)');
      lg.append('rect').attr('x', 0).attr('y', 12).attr('width', 12).attr('height', 12).style('fill', '#dc2626').attr('fill-opacity', 0.55);
      lg.append('text').attr('x', 16).attr('y', 22).style('font-size', '11px').style('fill', 'var(--color-text)').text('y = 1 (digit 1)');
    },
    [payload, panelWidth],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      {payload ? (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: isStacked ? '1fr' : '1fr 1fr', gap: 8 }}>
            <svg ref={leftRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Threshold-class gap vs MLP gap and bound." />
            <svg ref={rightRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="MLP prediction histogram by true class." />
          </div>
          <figcaption style={{ marginTop: 12, fontSize: 12, color: 'var(--color-text-secondary)' }}>
            Classical Rademacher bound ≈ {payload.results.corollary3_bound.toFixed(1)}; empirical generalization gap ≈ {payload.results.empirical_gap.toFixed(4)}. Bound is ~{Math.round(payload.results.ratio_bound_over_gap)}× looser than the actual gap.
          </figcaption>
        </>
      ) : (
        <div style={{
          height: HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--color-text-secondary)', fontSize: 13,
        }}>
          {error ? `failed to load: ${error}` : 'loading vacuousness payload…'}
        </div>
      )}
    </figure>
  );
}

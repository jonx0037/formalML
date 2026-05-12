import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// DznRReproduction — §11.5
// Fetch-based: loads /sample-data/pac-bayes-bounds/dziugaite_roy.json (precomputed
// in notebooks/pac-bayes-bounds/precompute_dziugaite_roy.py).  Renders three
// figures:
//   A — bound trajectory over Adam steps at the best prior
//   B — final-bound decomposition into empirical risk, KL slack contributions,
//       and log(K/δ) overhead
//   C — baseline comparison (SGD test 0/1, Rademacher vacuous bound, PAC-Bayes)
//
// Wrapper-stays-mounted loading pattern (CLAUDE.md "Loading-state JSX rule"):
// the containerRef'd div is rendered on every pass; only inner content swaps.
// =============================================================================

type DznRPayload = {
  bound_trajectory: { steps: number[]; values: number[] };
  final_decomposition: {
    empirical_risk: number;
    mean_shift_kl_nats: number;
    variance_kl_nats: number;
    log_K_delta_nats: number;
    numerator_nats: number;
    slack: number;
    total: number;
  };
  baseline_comparison: {
    sgd_test_01: number;
    rademacher_bound: number;
    pac_bayes_bound: number;
  };
  best_prior_sigma2: number;
  prior_grid: number[];
  config: {
    n_train: number;
    n_test: number;
    hidden: number;
    adam_steps: number;
    K_final_mc: number;
    delta: number;
  };
};

const HEIGHT = 320;
const SM_BREAKPOINT = 640;

export default function DznRReproduction() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<DznRPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/sample-data/pac-bayes-bounds/dziugaite_roy.json')
      .then((res) => {
        if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
        return res.json() as Promise<DznRPayload>;
      })
      .then(setPayload)
      .catch((e: Error) => setError(e.message));
  }, []);

  const isMobile = (containerWidth || 800) < SM_BREAKPOINT;
  const wideWidth = containerWidth || 800;
  const narrowWidth = isMobile ? wideWidth : Math.max(280, wideWidth / 2 - 8);

  return (
    <div
      ref={containerRef}
      style={{
        margin: '1.5rem 0',
        padding: '1rem',
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: '0.5rem',
        minHeight: 360,
      }}
    >
      {error && (
        <div style={{ padding: '1rem', color: 'var(--color-text-secondary)', fontSize: '0.85rem' }}>
          Could not load Dziugaite–Roy precomputed bound. Run
          <code style={{ marginLeft: 4, fontFamily: 'var(--font-mono)' }}>
            cd notebooks/pac-bayes-bounds && .venv/bin/python precompute_dziugaite_roy.py
          </code>
          to generate the JSON. (error: {error})
        </div>
      )}
      {!error && !payload && (
        <div style={{ padding: '2rem', color: 'var(--color-text-secondary)', fontSize: '0.85rem', textAlign: 'center' }}>
          Loading Dziugaite–Roy reproduction…
        </div>
      )}
      {payload && (
        <>
          <div style={{ marginBottom: '0.5rem', fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
            MNIST-binary 0-vs-1: n_train = {payload.config.n_train}, hidden = {payload.config.hidden}, δ = {payload.config.delta}, prior grid K = {payload.prior_grid.length} <br />
            best σ²_P = {payload.best_prior_sigma2}, certificate <strong>{payload.final_decomposition.total.toFixed(4)}</strong> &nbsp;|&nbsp; Rademacher = 1 (vacuous), SGD test 0/1 ≈ {payload.baseline_comparison.sgd_test_01.toFixed(4)}
          </div>
          <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
            <TrajectoryPanel payload={payload} width={narrowWidth} />
            <DecompositionPanel payload={payload} width={narrowWidth} />
          </div>
          <div style={{ marginTop: '0.5rem' }}>
            <BaselinePanel payload={payload} width={wideWidth} />
          </div>
        </>
      )}
    </div>
  );
}

function TrajectoryPanel({ payload, width }: { payload: DznRPayload; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 20, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const steps = payload.bound_trajectory.steps;
      const values = payload.bound_trajectory.values;
      const x = d3.scaleLinear().domain([0, steps[steps.length - 1]]).range([0, w]);
      const yMax = Math.max(...values) * 1.05;
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('Adam step');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('PAC-Bayes bound');

      // Vacuous baseline at y = 1
      if (yMax > 1) {
        g.append('line').attr('x1', 0).attr('x2', w)
          .attr('y1', y(1)).attr('y2', y(1))
          .style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '3,3');
        g.append('text').attr('x', w - 4).attr('y', y(1) - 4).attr('text-anchor', 'end')
          .style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('vacuous boundary');
      }

      const lineGen = d3.line<number>()
        .x((_, i) => x(steps[i]))
        .y((v) => y(v));
      g.append('path').datum(values)
        .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.2)
        .attr('d', lineGen);

      g.append('circle')
        .attr('cx', x(steps[steps.length - 1]))
        .attr('cy', y(values[values.length - 1]))
        .attr('r', 4)
        .style('fill', 'var(--color-accent)');

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Figure A — bound trajectory at best prior');
    },
    [payload, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

function DecompositionPanel({ payload, width }: { payload: DznRPayload; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 20, bottom: 60, left: 60 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Plot the three ADDITIVE nats components inside the Catoni-slack numerator
      // (mean-shift KL + variance KL + log(K/δ) = numerator), and the total
      // numerator as a separate bar.  This is mathematically honest: the three
      // KL-nats pieces sum to `numerator_nats`, then slack = sqrt(numerator / 2n).
      const decomp = payload.final_decomposition;
      const segs = [
        { name: 'mean-shift KL', value: decomp.mean_shift_kl_nats, color: '#534AB7' },
        { name: 'variance KL', value: decomp.variance_kl_nats, color: '#0F6E56' },
        { name: 'log(K/δ)', value: decomp.log_K_delta_nats, color: 'var(--color-text-secondary)' },
        { name: 'sum (numerator)', value: decomp.numerator_nats, color: '#1A1A1A' },
      ];

      const x = d3.scaleBand().domain(segs.map((s) => s.name)).range([0, w]).padding(0.2);
      const yMax = decomp.numerator_nats * 1.12;
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x))
        .selectAll('text').attr('transform', 'rotate(-25)').style('text-anchor', 'end');
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -48).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('KL nats (additive)');

      segs.forEach((s) => {
        g.append('rect')
          .attr('x', x(s.name)!)
          .attr('y', y(s.value))
          .attr('width', x.bandwidth())
          .attr('height', h - y(s.value))
          .style('fill', s.color);
        g.append('text')
          .attr('x', (x(s.name) ?? 0) + x.bandwidth() / 2)
          .attr('y', y(s.value) - 4)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px').style('fill', 'var(--color-text-secondary)')
          .text(s.value.toFixed(2));
      });

      // Annotation: slack = sqrt(numerator/(2n)), final bound = E_Q[R̂] + slack
      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Figure B — KL-nats decomposition inside the slack');
      g.append('text').attr('x', 4).attr('y', 14)
        .style('font-size', '9.5px').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-mono)')
        .text(`slack = √(${decomp.numerator_nats.toFixed(1)} / (2n)) = ${decomp.slack.toFixed(4)}`);
      g.append('text').attr('x', 4).attr('y', 26)
        .style('font-size', '9.5px').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-mono)')
        .text(`cert = E_Q[R̂] (${decomp.empirical_risk.toFixed(4)}) + slack = ${decomp.total.toFixed(4)}`);
    },
    [payload, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

function BaselinePanel({ payload, width }: { payload: DznRPayload; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 20, bottom: 44, left: 80 };
      const w = width - margin.left - margin.right;
      const h = 240 - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const cmp = payload.baseline_comparison;
      const rows = [
        { name: 'Rademacher bound', value: cmp.rademacher_bound, color: 'var(--color-text-secondary)' },
        { name: 'PAC-Bayes certificate', value: cmp.pac_bayes_bound, color: 'var(--color-accent)' },
        { name: 'SGD test 0/1 error', value: cmp.sgd_test_01, color: '#534AB7' },
      ];

      const y = d3.scaleBand().domain(rows.map((r) => r.name)).range([0, h]).padding(0.3);
      const x = d3.scaleLinear().domain([0, Math.max(1.05, cmp.rademacher_bound * 1.05)]).range([0, w]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(8));
      g.append('g').call(d3.axisLeft(y));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('upper bound on / observed E[R]');

      rows.forEach((r) => {
        g.append('rect')
          .attr('y', y(r.name)!)
          .attr('x', 0)
          .attr('width', x(r.value))
          .attr('height', y.bandwidth())
          .style('fill', r.color);
        g.append('text')
          .attr('x', x(r.value) + 6)
          .attr('y', (y(r.name) ?? 0) + y.bandwidth() / 2 + 4)
          .style('font-size', '11px').style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)')
          .text(r.value.toFixed(4));
      });

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Figure C — Rademacher (vacuous) vs PAC-Bayes (non-vacuous) vs SGD test error');
    },
    [payload, width],
  );
  return <svg ref={ref} width={width} height={240} style={{ overflow: 'visible' }} />;
}

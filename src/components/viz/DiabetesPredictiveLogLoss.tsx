import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// DiabetesPredictiveLogLoss — §10 five-method diabetes benchmark.
//
// Loads /sample-data/sparse-bayesian-priors/diabetes_predictive.json (produced
// by precompute_diabetes.py) and renders a horizontal bar chart of test
// predictive log-likelihood for the five methods (ridge baseline + four sparse
// priors) with bootstrap 95% CI error bars.
//
// Per CLAUDE.md fetch-based-viz rule: container <div ref={containerRef}> stays
// mounted; only inner content swaps between loading / SVG.
// =============================================================================

const PANEL_HEIGHT = 320;
const MARGIN = { top: 22, right: 32, bottom: 50, left: 130 };

interface MethodRecord {
  name: string;
  color: string;
  test_ll_mean: number;
  test_ll_se: number;
  ci_low: number;
  ci_high: number;
  divergences: number;
}

interface Payload {
  metadata: {
    n: number;
    p: number;
    n_train: number;
    n_test: number;
  };
  methods: MethodRecord[];
}

export default function DiabetesPredictiveLogLoss() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/sparse-bayesian-priors/diabetes_predictive.json')
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

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 720;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);

      if (!payload) {
        svg
          .append('text')
          .attr('x', w / 2)
          .attr('y', h / 2)
          .attr('text-anchor', 'middle')
          .style('font-size', '12px')
          .text(error ?? 'Loading diabetes_predictive.json…');
        return;
      }

      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = h - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const methods = payload.methods;
      const lls = methods.map((m) => m.test_ll_mean);
      const cilows = methods.map((m) => m.ci_low);
      const cihighs = methods.map((m) => m.ci_high);

      const xMin = Math.min(...cilows) - 0.005;
      const xMax = Math.max(...cihighs) + 0.005;

      const yScale = d3
        .scaleBand()
        .domain(methods.map((m) => m.name))
        .range([0, innerH])
        .padding(0.18);
      const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
      // All test log-likelihoods are negative on the standardized diabetes
      // response, so we draw bars from the chart's left edge (xScale(xMin))
      // outward to the per-method mean. Hoisted out of the loop since xMin
      // is loop-invariant.
      const barX = xScale(xMin);

      g.append('g').call(d3.axisLeft(yScale));
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6).tickFormat((d) => (d as number).toFixed(3)));

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 38)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text('test predictive log-likelihood per observation (higher is better)');

      // Bars
      methods.forEach((m) => {
        const yBand = yScale(m.name)!;
        const barH = yScale.bandwidth();
        const barW = xScale(m.test_ll_mean) - barX;
        g.append('rect')
          .attr('x', barX)
          .attr('y', yBand)
          .attr('width', Math.max(0, barW))
          .attr('height', barH)
          .attr('fill', m.color)
          .attr('opacity', 0.78)
          .attr('stroke', '#222')
          .attr('stroke-width', 0.4);
        // Error bar (95% bootstrap CI)
        g.append('line')
          .attr('x1', xScale(m.ci_low))
          .attr('x2', xScale(m.ci_high))
          .attr('y1', yBand + barH / 2)
          .attr('y2', yBand + barH / 2)
          .attr('stroke', '#222')
          .attr('stroke-width', 1.2);
        g.append('line')
          .attr('x1', xScale(m.ci_low))
          .attr('x2', xScale(m.ci_low))
          .attr('y1', yBand + barH / 2 - 4)
          .attr('y2', yBand + barH / 2 + 4)
          .attr('stroke', '#222')
          .attr('stroke-width', 1.2);
        g.append('line')
          .attr('x1', xScale(m.ci_high))
          .attr('x2', xScale(m.ci_high))
          .attr('y1', yBand + barH / 2 - 4)
          .attr('y2', yBand + barH / 2 + 4)
          .attr('stroke', '#222')
          .attr('stroke-width', 1.2);
        // Label
        g.append('text')
          .attr('x', xScale(m.test_ll_mean) + 6)
          .attr('y', yBand + barH / 2 + 4)
          .style('font-size', '10.5px')
          .style('fill', '#333')
          .text(`${m.test_ll_mean.toFixed(4)} (divs=${m.divergences})`);
      });

      // Mark the best method
      const best = methods.reduce((a, b) => (a.test_ll_mean > b.test_ll_mean ? a : b));
      g.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(`Best on diabetes (n_train=${payload.metadata.n_train}, n_test=${payload.metadata.n_test}): ${best.name}`);

      // Reference vertical line at ridge baseline
      const ridge = methods.find((m) => m.name === 'Ridge');
      if (ridge) {
        g.append('line')
          .attr('x1', xScale(ridge.test_ll_mean))
          .attr('x2', xScale(ridge.test_ll_mean))
          .attr('y1', 0)
          .attr('y2', innerH)
          .attr('stroke', '#7f7f7f')
          .attr('stroke-dasharray', '4 4')
          .attr('stroke-width', 0.8);
      }
    },
    [payload, error, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 760 }}>
      <svg ref={svgRef} role="img" aria-label="Diabetes test-predictive log-likelihood per method" />
    </div>
  );
}

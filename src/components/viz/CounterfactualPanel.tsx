import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  gaussianRng,
  sigmoid,
} from './shared/causal-inference-methods';

// =============================================================================
// CounterfactualPanel — §2 fundamental problem of causal inference.
//
// Two panels:
//  (A) joint (Y(0), Y(1)) scatter colored by realized D, with the diagonal
//      Y(1) = Y(0) + τ overlaid;
//  (B) first 30 units shown as line segments from Y(0) to Y(1), with a filled
//      dot at the observed Y_i = Y_i(D_i) and an open dot at the counterfactual.
//
// One slider: heterogeneity α ∈ [0, 1] (0 = constant τ for all units, 1 =
// per-unit τ_i = τ + α·ε where ε ~ N(0, 1)).
//
// Static fallback: public/images/topics/causal-inference-methods/02_potential_outcomes.png
// =============================================================================

const HEIGHT_PER_PANEL = 320;
const N_UNITS = 300;
const N_SEGMENTS = 30;
const SM_BREAKPOINT = 640;

type Counterfactual = {
  X: Float64Array;
  Y0: Float64Array;
  Y1: Float64Array;
  D: Int8Array;
};

function generate(alpha: number, seed: number): Counterfactual {
  const rng = mulberry32(seed);
  const gauss = gaussianRng(rng);
  const X = new Float64Array(N_UNITS);
  const Y0 = new Float64Array(N_UNITS);
  const Y1 = new Float64Array(N_UNITS);
  const D = new Int8Array(N_UNITS);
  for (let i = 0; i < N_UNITS; i++) {
    const x = gauss();
    X[i] = x;
    const tauI = 1 + alpha * gauss();
    Y0[i] = 0.6 * x + gauss();
    Y1[i] = Y0[i] + tauI;
    D[i] = rng() < sigmoid(1.2 * x) ? 1 : 0;
  }
  return { X, Y0, Y1, D };
}

export default function CounterfactualPanel() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [alpha, setAlpha] = useState(0.4);
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => generate(alpha, 20260512), [alpha]);

  const scatterRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 16, right: 16, bottom: 36, left: 48 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT_PER_PANEL - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT_PER_PANEL}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const extent = [Math.min(d3.min(data.Y0)!, d3.min(data.Y1)!),
                      Math.max(d3.max(data.Y0)!, d3.max(data.Y1)!)];
      const xScale = d3.scaleLinear().domain(extent).range([0, W]).nice();
      const yScale = d3.scaleLinear().domain(extent).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      // y = x line, then y = x + 1 (true ATE).
      g.append('line')
        .attr('x1', xScale(extent[0])).attr('x2', xScale(extent[1]))
        .attr('y1', yScale(extent[0])).attr('y2', yScale(extent[1]))
        .style('stroke', 'var(--color-border)').style('stroke-dasharray', '4 4');
      g.append('line')
        .attr('x1', xScale(extent[0])).attr('x2', xScale(extent[1] - 1))
        .attr('y1', yScale(extent[0] + 1)).attr('y2', yScale(extent[1]))
        .style('stroke', 'var(--color-accent)').style('stroke-width', 1.5);
      // Points.
      g.selectAll('circle.unit')
        .data(d3.range(N_UNITS))
        .join('circle')
        .attr('class', 'unit')
        .attr('cx', (i) => xScale(data.Y0[i]))
        .attr('cy', (i) => yScale(data.Y1[i]))
        .attr('r', 2.5)
        .style('fill', (i) => data.D[i] === 1 ? '#c0504d' : '#1f4e79')
        .style('opacity', 0.65);
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('Y(0) — control potential outcome');
      g.append('text').attr('transform', `translate(-36, ${H / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('Y(1) — treated potential outcome');
      // Legend.
      const lg = g.append('g').attr('transform', `translate(${W - 110}, 8)`);
      lg.append('circle').attr('cx', 6).attr('cy', 6).attr('r', 4).style('fill', '#1f4e79');
      lg.append('text').attr('x', 16).attr('y', 10).style('fill', 'var(--color-text)').style('font-size', '10px').text('D = 0');
      lg.append('circle').attr('cx', 6).attr('cy', 24).attr('r', 4).style('fill', '#c0504d');
      lg.append('text').attr('x', 16).attr('y', 28).style('fill', 'var(--color-text)').style('font-size', '10px').text('D = 1');
    },
    [containerWidth, data],
  );

  const segmentsRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 16, right: 16, bottom: 36, left: 48 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT_PER_PANEL - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT_PER_PANEL}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const yMin = Math.min(d3.min(data.Y0.subarray(0, N_SEGMENTS))!, d3.min(data.Y1.subarray(0, N_SEGMENTS))!);
      const yMax = Math.max(d3.max(data.Y0.subarray(0, N_SEGMENTS))!, d3.max(data.Y1.subarray(0, N_SEGMENTS))!);
      const xScale = d3.scaleLinear().domain([0, N_SEGMENTS - 1]).range([0, W]);
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      for (let i = 0; i < N_SEGMENTS; i++) {
        g.append('line')
          .attr('x1', xScale(i)).attr('x2', xScale(i))
          .attr('y1', yScale(data.Y0[i])).attr('y2', yScale(data.Y1[i]))
          .style('stroke', 'var(--color-border)').style('stroke-width', 1);
        const obs = data.D[i] === 1 ? data.Y1[i] : data.Y0[i];
        const counter = data.D[i] === 1 ? data.Y0[i] : data.Y1[i];
        // Observed = filled.
        g.append('circle').attr('cx', xScale(i)).attr('cy', yScale(obs)).attr('r', 4)
          .style('fill', data.D[i] === 1 ? '#c0504d' : '#1f4e79');
        // Counterfactual = open.
        g.append('circle').attr('cx', xScale(i)).attr('cy', yScale(counter)).attr('r', 4)
          .style('fill', 'none').style('stroke', data.D[i] === 1 ? '#c0504d' : '#1f4e79').style('stroke-width', 1.5);
      }
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('Unit index (i = 1, …, 30)');
      g.append('text').attr('transform', `translate(-36, ${H / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('Potential outcome');
    },
    [containerWidth, data],
  );

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          Heterogeneity α =
          <input
            type="range" min={0} max={1} step={0.05} value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            aria-label="Heterogeneity strength"
            style={{ width: 180 }}
          />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{alpha.toFixed(2)}</span>
        </label>
        <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
          α = 0: τ<sub>i</sub> = τ for every unit. α &gt; 0: τ<sub>i</sub> = τ + α·ε<sub>i</sub>.
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '1rem' }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '0.25rem' }}>
            (A) Counterfactual world: both Y(0) and Y(1) for every unit.
          </div>
          <svg ref={scatterRef} style={{ width: '100%', height: HEIGHT_PER_PANEL }} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '0.25rem' }}>
            (B) The fundamental problem: filled dot = observed, open dot = counterfactual.
          </div>
          <svg ref={segmentsRef} style={{ width: '100%', height: HEIGHT_PER_PANEL }} />
        </div>
      </div>
    </div>
  );
}

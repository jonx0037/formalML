import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { linspace } from './shared/uncertainty-quantification';

// =============================================================================
// ProperScoringRulesComparator — §4
//
// Slider for true class probability q ∈ (0, 1). Plots the expected Brier score
// and expected log-loss as functions of the forecast p ∈ (0, 1); marks the
// closed-form minima at p = q with values q(1-q) and H(q). Verifies the strict
// properness statement of Theorem 2 numerically.
// Static fallback: public/images/topics/uncertainty-quantification/fig_04_strict_properness.png
// =============================================================================

const HEIGHT = 430;

export default function ProperScoringRulesComparator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [q, setQ] = useState(0.30);

  const data = useMemo(() => {
    const grid = linspace(0.005, 0.995, 250);
    const expBrier = grid.map((p) => q * (p - 1) * (p - 1) + (1 - q) * p * p);
    const expLogLoss = grid.map((p) => -q * Math.log(p) - (1 - q) * Math.log(1 - p));
    const Hq = -q * Math.log(q) - (1 - q) * Math.log(1 - q);
    const qOneMinusQ = q * (1 - q);
    return { grid, expBrier, expLogLoss, Hq, qOneMinusQ };
  }, [q]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 24, bottom: 48, left: 56 };
      const panelW = (w - margin.left - margin.right - 32) / 2;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, panelW]);

      // ---- Panel (a): Brier ----
      const gA = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const yMaxA = Math.max(...data.expBrier, 1) * 1.05;
      const yScaleA = d3.scaleLinear().domain([0, yMaxA]).range([H, 0]).nice();
      const lineA = d3.line<number>()
        .x((_, i) => xScale(data.grid[i])).y((d) => yScaleA(d));
      gA.append('path').datum(data.expBrier).attr('d', lineA).attr('fill', 'none')
        .style('stroke', '#8b5cf6').attr('stroke-width', 2);
      gA.append('line').attr('x1', xScale(q)).attr('x2', xScale(q))
        .attr('y1', 0).attr('y2', H).style('stroke', '#94a3b8')
        .attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);
      gA.append('circle').attr('cx', xScale(q)).attr('cy', yScaleA(data.qOneMinusQ))
        .attr('r', 6).style('fill', '#dc2626');
      gA.append('text').attr('x', xScale(q) + 8).attr('y', yScaleA(data.qOneMinusQ) - 8)
        .style('font-size', '11px').style('fill', '#dc2626')
        .text(`min q(1−q) = ${data.qOneMinusQ.toFixed(3)}`);
      gA.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      gA.append('g').call(d3.axisLeft(yScaleA).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text)');
      gA.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      gA.append('text').attr('x', panelW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('(a) E[Brier(p, Y)] over p');
      gA.append('text').attr('x', panelW / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)').text('forecast p');

      // ---- Panel (b): Log-loss ----
      const gB = svg.append('g')
        .attr('transform', `translate(${margin.left + panelW + 32},${margin.top})`);
      const finiteLL = data.expLogLoss.filter((v) => Number.isFinite(v) && v < 8);
      const yMaxB = Math.max(...finiteLL) * 1.05;
      const yScaleB = d3.scaleLinear().domain([0, yMaxB]).range([H, 0]).nice();
      const lineB = d3.line<number>()
        .defined((d) => Number.isFinite(d) && d < yMaxB)
        .x((_, i) => xScale(data.grid[i])).y((d) => yScaleB(d));
      gB.append('path').datum(data.expLogLoss).attr('d', lineB).attr('fill', 'none')
        .style('stroke', '#8b5cf6').attr('stroke-width', 2);
      gB.append('line').attr('x1', xScale(q)).attr('x2', xScale(q))
        .attr('y1', 0).attr('y2', H).style('stroke', '#94a3b8')
        .attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);
      gB.append('circle').attr('cx', xScale(q)).attr('cy', yScaleB(data.Hq))
        .attr('r', 6).style('fill', '#dc2626');
      gB.append('text').attr('x', xScale(q) + 8).attr('y', yScaleB(data.Hq) - 8)
        .style('font-size', '11px').style('fill', '#dc2626')
        .text(`min H(q) = ${data.Hq.toFixed(3)}`);
      gB.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      gB.append('g').call(d3.axisLeft(yScaleB).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text)');
      gB.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      gB.append('text').attr('x', panelW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('(b) E[LogLoss(p, Y)] over p');
      gB.append('text').attr('x', panelW / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)').text('forecast p');
    },
    [containerWidth, data, q],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Expected Brier and log-loss as functions of forecast p, with the closed-form minimum at p=q marked." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 240 }}>
          <span>true positive rate q: <strong>{q.toFixed(2)}</strong></span>
          <input type="range" min={0.01} max={0.99} step={0.01} value={q}
            onChange={(e) => setQ(parseFloat(e.target.value))}
            aria-label="True Bernoulli probability q" />
        </label>
      </div>
      <p style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        Drag q to verify Theorem 2 numerically: both expected losses are uniquely minimized at p = q,
        with closed-form minimum values q(1−q) (Brier) and H(q) (log-loss).
      </p>
    </div>
  );
}

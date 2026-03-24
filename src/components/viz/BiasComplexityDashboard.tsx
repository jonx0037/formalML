import { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 56 };

const COLORS = {
  approximation: '#2563EB',
  estimation: '#DC2626',
  total: '#0F6E56',
  optimal: '#D97706',
} as const;

/** Estimation error from the VC bound: sqrt((8d log(2en/d) + 8 log(4/δ)) / n) */
function estimationError(d: number, n: number, delta: number): number {
  if (d <= 0 || n <= 0) return 0;
  const term = (8 * d * Math.log(2 * Math.E * n / d) + 8 * Math.log(4 / delta)) / n;
  return term > 0 ? Math.sqrt(term) : 0;
}

export default function BiasComplexityDashboard() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [n, setN] = useState(500);
  const [decayRate, setDecayRate] = useState(0.08);
  const [delta, setDelta] = useState(0.05);

  const chartW = Math.max(containerWidth - 32, 200);
  const chartH = Math.min(320, Math.max(220, chartW * 0.5));

  const curves = useMemo(() => {
    const dRange = Array.from({ length: 100 }, (_, i) => i + 1);
    const approx = dRange.map(d => 0.5 * Math.exp(-decayRate * d));
    const estim = dRange.map(d => estimationError(d, n, delta));
    const total = dRange.map((_, i) => approx[i] + estim[i]);
    const optIdx = total.reduce((best, val, i) => val < total[best] ? i : best, 0);
    return { dRange, approx, estim, total, optD: dRange[optIdx], optVal: total[optIdx] };
  }, [n, decayRate, delta]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || chartW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { dRange, approx, estim, total, optD, optVal } = curves;

    const xScale = d3.scaleLinear().domain([1, 100]).range([MARGIN.left, chartW - MARGIN.right]);
    const yMax = Math.min(d3.max(total.slice(0, 50)) || 1.5, 2);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([chartH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g').attr('transform', `translate(0,${chartH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(8).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    // Axis labels
    sel.append('text').attr('x', chartW / 2).attr('y', chartH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('Model complexity d (VC dimension)');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(chartH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('Error');

    // Helper to draw a line
    const drawLine = (data: number[], color: string, dashed = false) => {
      const lineGen = d3.line<number>().x((_, i) => xScale(dRange[i])).y(d => yScale(Math.min(d, yMax)));
      const path = sel.append('path').datum(data).attr('d', lineGen)
        .style('fill', 'none').style('stroke', color).style('stroke-width', 2);
      if (dashed) path.style('stroke-dasharray', '6 3');
    };

    drawLine(approx, COLORS.approximation);
    drawLine(estim, COLORS.estimation);
    drawLine(total, COLORS.total);

    // Optimal d* line
    sel.append('line')
      .attr('x1', xScale(optD)).attr('x2', xScale(optD))
      .attr('y1', yScale(0)).attr('y2', yScale(yMax))
      .style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '4 3').style('opacity', 0.6);

    // Optimal d* marker
    sel.append('circle')
      .attr('cx', xScale(optD)).attr('cy', yScale(optVal))
      .attr('r', 5).style('fill', COLORS.optimal).style('stroke', '#fff').style('stroke-width', 1.5);

    sel.append('text')
      .attr('x', xScale(optD) + 8).attr('y', yScale(yMax) + 16)
      .style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', COLORS.optimal).style('font-weight', '600')
      .text(`d* = ${optD}`);

    // Legend
    const legendY = MARGIN.top + 4;
    const legendX = MARGIN.left + 12;
    const items = [
      { label: 'Approximation', color: COLORS.approximation },
      { label: 'Estimation (VC)', color: COLORS.estimation },
      { label: 'Total', color: COLORS.total },
    ];
    items.forEach(({ label, color }, i) => {
      sel.append('line').attr('x1', legendX).attr('x2', legendX + 16).attr('y1', legendY + i * 14).attr('y2', legendY + i * 14)
        .style('stroke', color).style('stroke-width', 2);
      sel.append('text').attr('x', legendX + 20).attr('y', legendY + i * 14 + 3.5)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text(label);
    });

    // Title
    sel.append('text').attr('x', chartW - MARGIN.right).attr('y', MARGIN.top - 8)
      .attr('text-anchor', 'end').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('Bias–Complexity Tradeoff');
  }, [curves, chartW, chartH]);

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, marginBottom: '12px', color: 'var(--color-text)' }}>
        Structural Risk Minimization Dashboard
      </div>

      <svg ref={svgRef} width={chartW} height={chartH} style={{
        border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
      }} />

      {/* Sliders */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginTop: '12px' }}>
        <div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            n (samples): <strong style={{ color: 'var(--color-text)' }}>{n.toLocaleString()}</strong>
          </div>
          <input type="range" min={50} max={5000} step={50} value={n}
            onChange={e => setN(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
        </div>
        <div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            Approx. decay b: <strong style={{ color: 'var(--color-text)' }}>{decayRate.toFixed(2)}</strong>
          </div>
          <input type="range" min={0.02} max={0.2} step={0.005} value={decayRate}
            onChange={e => setDecayRate(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
        </div>
        <div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            δ (failure prob): <strong style={{ color: 'var(--color-text)' }}>{delta.toFixed(2)}</strong>
          </div>
          <input type="range" min={0.01} max={0.5} step={0.01} value={delta}
            onChange={e => setDelta(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
        </div>
      </div>

      {/* Interpretation */}
      <div style={{
        marginTop: '12px', padding: '10px', borderRadius: '6px',
        background: 'var(--color-definition-bg)', border: '1px solid var(--color-definition-border)',
        fontSize: '12px', fontFamily: 'var(--font-sans)', color: 'var(--color-text)',
      }}>
        Optimal model complexity <strong>d* = {curves.optD}</strong> — balances approximation error
        (decreasing with complexity) against estimation error from the VC bound (increasing with complexity).
        {n < 200 && <span style={{ color: '#D97706' }}> With few samples, the optimal complexity is low — simpler models generalize better.</span>}
      </div>
    </div>
  );
}

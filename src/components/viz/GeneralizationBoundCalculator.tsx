import { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 56 };

export default function GeneralizationBoundCalculator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [n, setN] = useState(1000);
  const [logH, setLogH] = useState(2); // log10(|H|)
  const [logDelta, setLogDelta] = useState(-1.3); // log10(delta)
  const [M, setM] = useState(1.0);

  const H = Math.pow(10, logH);
  const delta = Math.pow(10, logDelta);

  // Generalization bound: ε = M * sqrt(log(2|H|/δ) / (2n))
  const epsilon = M * Math.sqrt(Math.log(2 * H / delta) / (2 * n));

  const isDesktop = (containerWidth || 0) > 640;
  const chartW = isDesktop ? Math.floor(containerWidth * 0.55) : containerWidth;
  const chartH = Math.min(280, Math.max(200, chartW * 0.6));

  // ─── Sample complexity curve ───

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || chartW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const nPts = 150;
    const nRange = Array.from({ length: nPts }, (_, i) => 50 + (9950) * i / (nPts - 1));
    const epsVals = nRange.map(nVal => M * Math.sqrt(Math.log(2 * H / delta) / (2 * nVal)));

    const xScale = d3.scaleLog().domain([50, 10000]).range([MARGIN.left, chartW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([0, Math.min(d3.max(epsVals) || 1, 2)]).range([chartH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g').attr('transform', `translate(0,${chartH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(4, ',.0f').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    // Axis labels
    sel.append('text').attr('x', chartW / 2).attr('y', chartH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('n (sample size)');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(chartH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('ε (bound)');

    // Target lines
    [0.05, 0.01].forEach(target => {
      if (target < yScale.domain()[1]) {
        sel.append('line')
          .attr('x1', MARGIN.left).attr('x2', chartW - MARGIN.right)
          .attr('y1', yScale(target)).attr('y2', yScale(target))
          .style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '4 3').style('opacity', 0.4);
        sel.append('text').attr('x', chartW - MARGIN.right - 2).attr('y', yScale(target) - 3)
          .attr('text-anchor', 'end').style('font-size', '8px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
          .text(`ε = ${target}`);
      }
    });

    // Curve
    const lineGen = d3.line<number>()
      .x((_, i) => xScale(nRange[i]))
      .y(d => yScale(d));
    sel.append('path').datum(epsVals).attr('d', lineGen)
      .style('fill', 'none').style('stroke', '#0F6E56').style('stroke-width', 2.5);

    // Current point
    if (n >= 50 && n <= 10000 && epsilon <= yScale.domain()[1]) {
      sel.append('circle')
        .attr('cx', xScale(n)).attr('cy', yScale(epsilon))
        .attr('r', 5).style('fill', '#D97706').style('stroke', '#fff').style('stroke-width', 1.5);
    }

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('Sample Complexity Curve');
  }, [n, H, delta, M, epsilon, chartW, chartH]);

  // ─── Interpretation ───

  const confidence = ((1 - delta) * 100).toFixed(1);
  const epsStr = epsilon < 0.001 ? epsilon.toExponential(2) : epsilon.toFixed(3);
  const hStr = H >= 1000 ? H.toExponential(0) : H.toFixed(0);

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
        Generalization Bound Calculator
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: isDesktop ? 'auto 1fr' : '1fr', gap: '16px' }}>
        {/* Sliders */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', minWidth: isDesktop ? 200 : undefined }}>
          {/* n slider */}
          <div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
              n (samples): <strong style={{ color: 'var(--color-text)' }}>{n.toLocaleString()}</strong>
            </div>
            <input type="range" min={50} max={10000} step={50} value={n}
              onChange={e => setN(parseInt(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
          </div>

          {/* |H| slider (log scale) */}
          <div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
              |H| (hypotheses): <strong style={{ color: 'var(--color-text)' }}>{hStr}</strong>
            </div>
            <input type="range" min={0} max={5} step={0.1} value={logH}
              onChange={e => setLogH(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
          </div>

          {/* δ slider (log scale) */}
          <div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
              δ (failure prob): <strong style={{ color: 'var(--color-text)' }}>{delta.toFixed(4)}</strong>
            </div>
            <input type="range" min={-3} max={Math.log10(0.5).toFixed(2)} step={0.05} value={logDelta}
              onChange={e => setLogDelta(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
          </div>

          {/* M slider */}
          <div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
              M (loss bound): <strong style={{ color: 'var(--color-text)' }}>{M.toFixed(1)}</strong>
            </div>
            <input type="range" min={0.1} max={1.0} step={0.1} value={M}
              onChange={e => setM(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
          </div>

          {/* Big epsilon display */}
          <div style={{
            padding: '12px', borderRadius: '6px', background: 'var(--color-muted-bg)',
            border: '1px solid var(--color-border)', textAlign: 'center',
          }}>
            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 4 }}>
              Generalization bound ε
            </div>
            <div style={{ fontSize: '24px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#0F6E56' }}>
              ±{epsStr}
            </div>
          </div>
        </div>

        {/* Chart */}
        <svg ref={svgRef} width={chartW} height={chartH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
      </div>

      {/* Interpretation */}
      <div style={{
        marginTop: '12px', padding: '10px', borderRadius: '6px',
        background: 'var(--color-definition-bg)', border: '1px solid var(--color-definition-border)',
        fontSize: '12px', fontFamily: 'var(--font-sans)', color: 'var(--color-text)',
      }}>
        With <strong>{n.toLocaleString()}</strong> samples and <strong>{hStr}</strong> hypotheses,
        the empirical risk is within <strong>±{epsStr}</strong> of the true risk
        with <strong>{confidence}%</strong> confidence.
        {epsilon > 0.5 && <span style={{ color: '#D97706' }}> (bound is loose — increase n or decrease |H|)</span>}
      </div>
    </div>
  );
}

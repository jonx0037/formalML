import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { classicalBiasVariance, mulberry32 } from './shared/double-descent';

// =============================================================================
// ClassicalBiasVariance — §2.1 (Figure 2)
//
// Polynomial regression in Legendre basis on noisy sin(πx), Monte Carlo
// bias-variance decomposition. Log y-scale, degree d ∈ [0, dMax].
// Static fallback: public/images/topics/double-descent/02_classical_u_curve.png
// =============================================================================

const HEIGHT = 440;
const TEST_GRID = (() => {
  const g = new Float64Array(50);
  for (let i = 0; i < 50; i++) g[i] = -1 + (2 * i) / 49;
  return g;
})();

export default function ClassicalBiasVariance() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nDisplay, setNDisplay] = useState(100);
  const [nCommitted, setNCommitted] = useState(100);
  const [sigmaDisplay, setSigmaDisplay] = useState(0.2);
  const [sigmaCommitted, setSigmaCommitted] = useState(0.2);
  const [showDecomp, setShowDecomp] = useState(true);
  const dMax = 15;

  const bv = useMemo(() => classicalBiasVariance({
    n: nCommitted,
    sigma: sigmaCommitted,
    dMax,
    B: 80,
    testGrid: TEST_GRID,
    rng: mulberry32(2024),
  }), [nCommitted, sigmaCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 32, bottom: 52, left: 64 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xs = Array.from({ length: dMax + 1 }, (_, i) => i);
      const allVals = [...bv.bias2, ...bv.variance, ...bv.total].filter((v) => v > 0);
      const yMin = Math.max(1e-5, d3.min(allVals) ?? 1e-4);
      const yMax = d3.max(allVals) ?? 1;

      const xScale = d3.scaleLinear().domain([0, dMax]).range([0, W]);
      const yScale = d3.scaleLog().domain([yMin * 0.7, yMax * 1.4]).range([H, 0]);

      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xScale).ticks(8))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6, '.0e'))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 38).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').text('polynomial degree d');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -48)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px')
        .text('error (log scale)');

      const seriesList: { name: string; values: number[]; color: string; dash: string }[] = [
        { name: 'total', values: Array.from(bv.total), color: 'var(--color-accent)', dash: '' },
      ];
      if (showDecomp) {
        seriesList.push(
          { name: 'bias²', values: Array.from(bv.bias2), color: '#D97706', dash: '5 3' },
          { name: 'variance', values: Array.from(bv.variance), color: '#534AB7', dash: '2 4' },
        );
      }

      const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0]))
        .y((d) => yScale(Math.max(yMin * 0.7, d[1])));

      for (const s of seriesList) {
        const data: [number, number][] = xs.map((d, i) => [d, s.values[i]]);
        g.append('path').datum(data).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', s.color).attr('stroke-width', 2)
          .attr('stroke-dasharray', s.dash);
      }

      // Legend
      seriesList.forEach((s, i) => {
        const ly = 12 + i * 18;
        g.append('line').attr('x1', W - 110).attr('x2', W - 90).attr('y1', ly).attr('y2', ly)
          .style('stroke', s.color).attr('stroke-width', 2).attr('stroke-dasharray', s.dash);
        g.append('text').attr('x', W - 84).attr('y', ly + 4)
          .style('fill', 'var(--color-text)').style('font-size', '12px').text(s.name);
      });

      // Mark argmin total
      let argmin = 0;
      for (let i = 1; i < bv.total.length; i++) if (bv.total[i] < bv.total[argmin]) argmin = i;
      g.append('circle').attr('cx', xScale(argmin))
        .attr('cy', yScale(Math.max(yMin * 0.7, bv.total[argmin]))).attr('r', 5)
        .style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 1.5);
      g.append('text').attr('x', xScale(argmin)).attr('y', yScale(Math.max(yMin * 0.7, bv.total[argmin])) - 10)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-accent)').style('font-size', '11px')
        .text(`U min at d = ${argmin}`);
    },
    [containerWidth, bv, showDecomp],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Classical bias-variance U curve in Legendre polynomial basis." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
          <span>n (training points): <strong>{nDisplay}</strong></span>
          <input type="range" min={20} max={500} step={10} value={nDisplay}
            onChange={(e) => setNDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setNCommitted(nDisplay)} onTouchEnd={() => setNCommitted(nDisplay)}
            onKeyUp={() => setNCommitted(nDisplay)} aria-label="Training set size n" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
          <span>σ (noise): <strong>{sigmaDisplay.toFixed(2)}</strong></span>
          <input type="range" min={0.05} max={1.0} step={0.05} value={sigmaDisplay}
            onChange={(e) => setSigmaDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setSigmaCommitted(sigmaDisplay)}
            onTouchEnd={() => setSigmaCommitted(sigmaDisplay)}
            onKeyUp={() => setSigmaCommitted(sigmaDisplay)} aria-label="Noise sigma" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showDecomp} onChange={(e) => setShowDecomp(e.target.checked)} />
          <span>show bias²/variance decomposition</span>
        </label>
      </div>
    </div>
  );
}

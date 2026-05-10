import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { boundaryBiasConstant, kGaussian, paletteKR } from './shared/kernel-regression';

// =============================================================================
// OddEvenDegreeBoundary — analytical boundary bias-constant table for §5.3.
// At c < ∞, the parity-zero pattern of §4.3 collapses; b_p^(c)(K) is nonzero
// for every p, and the boundary leading rate is uniformly h^{p+1}.
//
// Slider: c ∈ [0, 3] (continuous boundary parameter, where c = x/h is distance
// from the support edge in bandwidth units; c = ∞ recovers the interior limit).
// Bar chart of |b_p^(c)(K)| for p ∈ {0, ..., 4}; values continuously updated.
// =============================================================================

const HEIGHT = 380;
const SM_BREAKPOINT = 640;

export default function OddEvenDegreeBoundary() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [cDisplay, setCDisplay] = useState(0); // strict boundary
  const [cCommitted, setCCommitted] = useState(0);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const data = useMemo(() => {
    const cVal = cCommitted >= 3 ? Infinity : cCommitted;
    return [0, 1, 2, 3, 4].map((p) => {
      const b = boundaryBiasConstant(kGaussian, p, cVal);
      return { p, b, absB: Math.abs(b), rate: `h^${p + 1}` };
    });
  }, [cCommitted]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 30, right: 16, bottom: 50, left: 56 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleBand().domain(data.map((d) => `p=${d.p}`)).range([0, innerW]).padding(0.2);
      const yMax = Math.max(...data.map((d) => d.absB)) * 1.15 || 1;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale))
        .selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '13px');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 36).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)')
        .style('font-size', '12px').text('polynomial degree');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -42)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '12px').text('|b_p^(c)(K)|');

      g.selectAll('rect').data(data).enter().append('rect')
        .attr('x', (d) => xScale(`p=${d.p}`) ?? 0)
        .attr('y', (d) => yScale(d.absB))
        .attr('width', xScale.bandwidth())
        .attr('height', (d) => innerH - yScale(d.absB))
        .style('fill', (d) => (d.b < 0 ? paletteKR.truth : paletteKR.posterior))
        .style('opacity', 0.85);

      g.selectAll('text.label').data(data).enter().append('text').attr('class', 'label')
        .attr('x', (d) => (xScale(`p=${d.p}`) ?? 0) + xScale.bandwidth() / 2)
        .attr('y', (d) => yScale(d.absB) - 4)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)')
        .style('font-size', '11px').text((d) => `${d.b.toFixed(3)}  (${d.rate})`);
    },
    [w, data],
  );

  const cValDisplay = cDisplay >= 3 ? '∞' : cDisplay.toFixed(2);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <label style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)' }}>
          boundary parameter c = <strong>{cValDisplay}</strong>
        </label>
        <input type="range" min={0} max={3} step={0.05} value={cDisplay}
          onChange={(e) => setCDisplay(parseFloat(e.target.value))}
          onMouseUp={(e) => setCCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => setCCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => setCCommitted(parseFloat((e.target as HTMLInputElement).value))}
          aria-label="boundary parameter c" style={{ flex: 1, minWidth: 200 }} />
        <span style={{ fontFamily: 'var(--font-sans)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          (c = 0 strict boundary  ·  c = 3 ≈ interior)
        </span>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} />
    </div>
  );
}

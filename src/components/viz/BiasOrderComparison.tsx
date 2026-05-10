import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { biasConstant, kGaussian, paletteKR, varianceConstant } from './shared/kernel-regression';

// =============================================================================
// BiasOrderComparison — analytical interior bias-rate ladder for §4.3.
// Two side-by-side panels:
//   (1) bar chart of |b_p(K)| for p ∈ {0, ..., 5} (with parity-zero columns
//       drawn as dotted "0" markers since they're exactly zero).
//   (2) bar chart of R*_p(K), variance penalty growing with p.
// Both annotated with the leading interior-bias rate h^{p+1} (parity-bonus
// h^{p+2} for even p).
//
// All values pulled from biasConstant / varianceConstant (numerical
// integration on the Gaussian kernel) — these are the analytical anchors that
// drive the verification suite (Tests 4 and 5).
// =============================================================================

const HEIGHT = 360;
const SM_BREAKPOINT = 640;

export default function BiasOrderComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    return [0, 1, 2, 3, 4, 5].map((p) => {
      const b = biasConstant(kGaussian, p, Infinity);
      const R = varianceConstant(kGaussian, p, Infinity);
      const isEven = p % 2 === 0;
      const rate = isEven ? `h^${p + 2}` : `h^${p + 1}`;
      return { p, b, absB: Math.abs(b), R, rate, isEven };
    });
  }, []);

  const panelW = containerWidth > 0 ? (isMobile ? containerWidth : containerWidth / 2 - 8) : 0;

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: isMobile ? 12 : 16 }}>
        <BarPanel title="bias constant |b_p(K)|" subtitle="Gaussian, interior" data={data} field="absB" width={panelW} color={paletteKR.posterior} parityZero />
        <BarPanel title="variance constant R*_p(K)" subtitle="Gaussian, interior" data={data} field="R" width={panelW} color={paletteKR.alt} parityZero={false} />
      </div>
      <p style={{ fontFamily: 'var(--font-sans)', fontSize: 13, color: 'var(--color-text-secondary)', marginTop: 12 }}>
        Even-p bias constants are <strong>exactly zero</strong> at interior with symmetric K (parity).
        The leading-order bias picks up the next-order term, giving the parity-bonus rate
        h^(p+2) for even p versus h^(p+1) for odd p. Variance constants pair (R*_0 = R*_1,
        R*_2 = R*_3, R*_4 = R*_5) — the §3 odd-with-even pairing rendered as a constant.
      </p>
    </div>
  );
}

function BarPanel({
  title,
  subtitle,
  data,
  field,
  width,
  color,
  parityZero,
}: {
  title: string;
  subtitle: string;
  data: Array<{ p: number; absB: number; R: number; rate: string; isEven: boolean }>;
  field: 'absB' | 'R';
  width: number;
  color: string;
  parityZero: boolean;
}) {
  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 38, right: 12, bottom: 40, left: 44 };
      const innerW = width - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleBand().domain(data.map((d) => `p=${d.p}`)).range([0, innerW]).padding(0.18);
      const yMax = Math.max(...data.map((d) => d[field])) * 1.15 || 1;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale))
        .selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '12px');
      g.append('g').call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');

      // Title
      svg.append('text').attr('x', margin.left).attr('y', 14).style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '13px').style('font-weight', '600').text(title);
      svg.append('text').attr('x', margin.left).attr('y', 28).style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)').style('font-size', '11px').text(subtitle);

      // Bars
      g.selectAll('rect.bar').data(data).enter().append('rect').attr('class', 'bar')
        .attr('x', (d) => xScale(`p=${d.p}`) ?? 0)
        .attr('y', (d) => yScale(d[field]))
        .attr('width', xScale.bandwidth())
        .attr('height', (d) => innerH - yScale(d[field]))
        .style('fill', (d) => (parityZero && d.isEven ? 'var(--color-border)' : color))
        .style('opacity', (d) => (parityZero && d.isEven ? 0.4 : 0.85));

      // Value labels
      g.selectAll('text.val').data(data).enter().append('text').attr('class', 'val')
        .attr('x', (d) => (xScale(`p=${d.p}`) ?? 0) + xScale.bandwidth() / 2)
        .attr('y', (d) => yScale(d[field]) - 4)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '10.5px')
        .text((d) => {
          if (parityZero && d.isEven) return '0 (parity)';
          if (field === 'absB') {
            const b = data.find((x) => x.p === d.p);
            return b ? b['absB'].toFixed(3) : '';
          }
          return d[field].toFixed(3);
        });

      // Rate annotation row at the bottom inside the chart.
      g.selectAll('text.rate').data(data).enter().append('text').attr('class', 'rate')
        .attr('x', (d) => (xScale(`p=${d.p}`) ?? 0) + xScale.bandwidth() / 2)
        .attr('y', innerH - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '10px')
        .text((d) => (field === 'absB' ? d.rate : ''));
    },
    [width, data, field, color, parityZero],
  );

  return (
    <div style={{ width: width > 0 ? width : '100%' }}>
      <svg ref={renderRef} width={width} height={HEIGHT} />
    </div>
  );
}

import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// DoubleDescentTeaser — small static teaser for the planned `double-descent`
// topic. Sketches the classical-U-then-second-descent curve as a function of
// model capacity, with the interpolation threshold marked at W ≈ n. Pure
// schematic — no MC, no sliders.
//
// (Companion to §12.4 forward-pointer; static fallback exists in the §12.4
// prose itself as plain text "Double Descent (coming soon)".)
// =============================================================================

const HEIGHT = 260;

export default function DoubleDescentTeaser() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 16, right: 24, bottom: 36, left: 48 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      // Schematic double-descent curve: classical U from 0 to threshold, peak
      // at threshold, second descent past it.
      const x = d3.scaleLinear().domain([0, 100]).range([0, W]);
      const y = d3.scaleLinear().domain([0, 1]).range([H, 0]);
      const points: [number, number][] = [];
      for (let i = 0; i <= 100; i++) {
        let v: number;
        if (i < 30) {
          // Bias regime: high error, decreasing
          v = 0.85 - 0.6 * (i / 30) + 0.05 * Math.sin(i / 4);
        } else if (i < 50) {
          // Approaching interpolation threshold: U-shape rising
          const t = (i - 30) / 20;
          v = 0.25 + 0.65 * t * t;
        } else if (i <= 55) {
          // Peak at interpolation threshold (W ≈ n)
          v = 0.9 + 0.05 * Math.cos((i - 50) * 0.6);
        } else {
          // Modern regime: second descent
          const t = (i - 55) / 45;
          v = 0.85 - 0.55 * Math.pow(t, 0.7);
        }
        points.push([i, Math.max(0.1, Math.min(0.95, v))]);
      }
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5).tickFormat(() => ''));
      g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(() => ''));
      g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('model capacity W →');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -32).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('test risk');
      const line = d3.line<[number, number]>().x((p) => x(p[0])).y((p) => y(p[1])).curve(d3.curveBasis);
      g.append('path').datum(points).attr('d', line).attr('fill', 'none').attr('stroke', '#7c3aed').attr('stroke-width', 2.4);
      // Mark interpolation threshold
      g.append('line').attr('x1', x(52)).attr('x2', x(52)).attr('y1', 0).attr('y2', H).attr('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3').attr('opacity', 0.6);
      g.append('text').attr('x', x(52)).attr('y', -4).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('W ≈ n');
      // Regime labels
      g.append('text').attr('x', x(15)).attr('y', H - 8).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text('classical U-curve');
      g.append('text').attr('x', x(78)).attr('y', H - 8).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text('modern: second descent');
    },
    [containerWidth],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        The double-descent picture: classical U-curve (bias → variance) up to the interpolation threshold W ≈ n, catastrophic peak at the threshold, then a second descent in the modern overparameterised regime (W ≫ n). Sketch only — full treatment in <strong>Double Descent</strong> <em>(coming soon)</em>.
      </div>
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  growthHalfLine,
  growthHalfPlane,
  growthInterval,
  paletteVC,
  sauerShelahBinomialSum,
} from './shared/vc-dimension';

// =============================================================================
// GrowthFunctionPlotter — §3.5
//
// Log-scale plot of Pi(n) for half-lines, intervals, half-planes, axis-rectangles
// (Sauer–Shelah ceiling at d=4), plus the 2^n trivial bound. Slider for max n.
// Static fallback: public/images/topics/vc-dimension/03_growth_function_gallery.png
// =============================================================================

const HEIGHT = 420;
const N_MIN = 1;

export default function GrowthFunctionPlotter() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nMax, setNMax] = useState(16);

  const { ns, halfLine, interval, halfPlane, rect, trivial } = useMemo(() => {
    const xs = Array.from({ length: nMax - N_MIN + 1 }, (_, i) => N_MIN + i);
    return {
      ns: xs,
      halfLine: xs.map((n) => growthHalfLine(n)),
      interval: xs.map((n) => growthInterval(n)),
      halfPlane: xs.map((n) => growthHalfPlane(n)),
      // Rectangle empirical/ceiling: use Sauer–Shelah ceiling at d_VC = 4 capped at 2^n.
      rect: xs.map((n) => Math.min(Math.pow(2, n), sauerShelahBinomialSum(n, 4))),
      trivial: xs.map((n) => Math.pow(2, n)),
    };
  }, [nMax]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 36, bottom: 48, left: 60 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([N_MIN, nMax]).range([0, W]);
      const yMaxVal = Math.max(...trivial);
      const y = d3.scaleLog().domain([1, yMaxVal * 1.2]).range([H, 0]);

      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(Math.min(nMax, 12)));
      g.append('g').call(d3.axisLeft(y).ticks(6, '~s'));
      g.append('text').attr('x', W / 2).attr('y', H + 34).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('sample size n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -42).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('Π(n) (log scale)');

      const line = (vals: number[]) =>
        d3
          .line<number>()
          .x((_d, i) => x(ns[i]))
          .y((d) => y(Math.max(1, d)));

      const series = [
        { label: '2ⁿ trivial', vals: trivial, color: paletteVC.muted, dash: '3 4', width: 1.6 },
        { label: 'half-line (Pi = n+1)', vals: halfLine, color: paletteVC.primary, width: 2.0 },
        { label: 'interval', vals: interval, color: paletteVC.highlight, width: 2.0 },
        { label: 'half-plane (n² − n + 2)', vals: halfPlane, color: paletteVC.alt, width: 2.4 },
        { label: 'rectangle (≤ Sauer–Shelah d=4)', vals: rect, color: paletteVC.emp, width: 2.0, dash: '5 3' },
      ];
      series.forEach((s) => {
        g.append('path').datum(s.vals).attr('d', line(s.vals)).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', s.width).attr('stroke-dasharray', s.dash || '');
      });

      // Phase-transition arrows at n = d_VC + 1
      const arrows: Array<[string, number, string]> = [
        ['HL d_VC+1=2', 2, paletteVC.primary],
        ['IV d_VC+1=3', 3, paletteVC.highlight],
        ['HP d_VC+1=4', 4, paletteVC.alt],
        ['rect d_VC+1=5', 5, paletteVC.emp],
      ];
      arrows.forEach(([_, atN, color], i) => {
        if (atN > nMax) return;
        g.append('line').attr('x1', x(atN)).attr('x2', x(atN)).attr('y1', H).attr('y2', H - 8).attr('stroke', color).attr('stroke-width', 1.5);
        g.append('text').attr('x', x(atN)).attr('y', H - 12).attr('text-anchor', 'middle').style('fill', color).style('font-size', '10px').text(`d_VC+1=${atN}`);
      });

      const legend = g.append('g').attr('transform', `translate(${Math.max(8, W - 200)}, 6)`);
      series.forEach((s, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', i * 16 + 6).attr('y2', i * 16 + 6).attr('stroke', s.color).attr('stroke-width', s.width).attr('stroke-dasharray', s.dash || '');
        legend.append('text').attr('x', 22).attr('y', i * 16 + 10).style('fill', 'var(--color-text)').style('font-size', '10px').text(s.label);
      });
    },
    [ns, halfLine, interval, halfPlane, rect, trivial, nMax, containerWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img" aria-label="Growth functions for four canonical classes against 2^n on log scale" />
      <div style={{ marginTop: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
        <label htmlFor="growth-nmax">Max n: {nMax}</label>
        <input
          id="growth-nmax"
          type="range"
          min={4}
          max={16}
          value={nMax}
          onChange={(e) => setNMax(Number(e.target.value))}
          aria-label="Maximum sample size"
          style={{ flex: 1 }}
        />
      </div>
    </div>
  );
}

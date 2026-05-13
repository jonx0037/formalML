import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  paletteVC,
  sauerShelahBinomialSum,
  sauerShelahClosedForm,
} from './shared/vc-dimension';

// =============================================================================
// SauerShelahDemo — §5.5
//
// Two panels:
//   (a) bounds vs n on log scale for selected d, with 2^n trivial line
//   (b) ratio (en/d)^d / C(n,<=d) converging to Stirling factor e^d/sqrt(2 pi d)
// Static fallback: public/images/topics/vc-dimension/05_sauer_shelah_bounds.png
// =============================================================================

const HEIGHT = 360;
const N_MAX = 30;

const D_OPTIONS = [1, 2, 3, 4, 5, 10] as const;

function stirlingFactor(d: number): number {
  return Math.exp(d) / Math.sqrt(2 * Math.PI * d);
}

export default function SauerShelahDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [d, setD] = useState(3);

  const ns = useMemo(() => Array.from({ length: N_MAX }, (_, i) => i + 1), []);
  const data = useMemo(() => {
    return ns.map((n) => ({
      n,
      binom: sauerShelahBinomialSum(n, d),
      closed: n >= d ? sauerShelahClosedForm(n, d) : Math.pow(2, n),
      trivial: Math.pow(2, n),
      ratio: n >= d ? sauerShelahClosedForm(n, d) / Math.max(sauerShelahBinomialSum(n, d), 1) : NaN,
    }));
  }, [ns, d]);

  const isMobile = containerWidth > 0 && containerWidth < 720;
  const half = useMemo(() => Math.max(280, ((containerWidth || 720) - 24) / (isMobile ? 1 : 2)), [containerWidth, isMobile]);

  const refA = useD3<SVGSVGElement>(
    (svg) => {
      const margin = { top: 22, right: 30, bottom: 40, left: 56 };
      const W = half - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${half} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([1, N_MAX]).range([0, W]);
      const yMax = Math.max(...data.map((d) => d.closed), Math.pow(2, N_MAX));
      const y = d3.scaleLog().domain([1, yMax]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(6, '~s'));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('sample size n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -42).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('bound value (log)');

      const line = (key: 'binom' | 'closed' | 'trivial') =>
        d3.line<typeof data[number]>().x((d) => x(d.n)).y((d) => y(Math.max(1, d[key])));

      g.append('path').datum(data).attr('d', line('trivial')).attr('fill', 'none').attr('stroke', paletteVC.muted).attr('stroke-width', 1.6).attr('stroke-dasharray', '3 4');
      g.append('path').datum(data).attr('d', line('binom')).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2.4);
      g.append('path').datum(data).attr('d', line('closed')).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2.0).attr('stroke-dasharray', '5 3');

      const legend = g.append('g').attr('transform', `translate(${Math.max(8, W - 200)}, 6)`);
      const items = [
        { label: '2ⁿ trivial', color: paletteVC.muted, dash: '3 4' },
        { label: `C(n, ≤${d}) binomial sum`, color: paletteVC.primary, dash: '' },
        { label: `(en/${d})^${d} closed form`, color: paletteVC.emp, dash: '5 3' },
      ];
      items.forEach((it, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', i * 16 + 6).attr('y2', i * 16 + 6).attr('stroke', it.color).attr('stroke-width', 2).attr('stroke-dasharray', it.dash);
        legend.append('text').attr('x', 22).attr('y', i * 16 + 10).style('fill', 'var(--color-text)').style('font-size', '10px').text(it.label);
      });
    },
    [data, d, half, containerWidth],
  );

  const refB = useD3<SVGSVGElement>(
    (svg) => {
      const margin = { top: 22, right: 30, bottom: 40, left: 56 };
      const W = half - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${half} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xs = data.filter((r) => isFinite(r.ratio));
      const x = d3.scaleLinear().domain([d, N_MAX]).range([0, W]);
      const stirling = stirlingFactor(d);
      const yMax = Math.max(...xs.map((r) => r.ratio), stirling * 1.2);
      const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('sample size n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -42).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('ratio (en/d)^d / C(n,≤d)');

      const lineRatio = d3.line<typeof data[number]>().x((d) => x(d.n)).y((d) => y(d.ratio));
      g.append('path').datum(xs).attr('d', lineRatio).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2.4);
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', y(stirling)).attr('y2', y(stirling)).attr('stroke', paletteVC.emp).attr('stroke-dasharray', '4 4').attr('stroke-width', 1.6);
      g.append('text').attr('x', W - 6).attr('y', y(stirling) - 4).attr('text-anchor', 'end').style('fill', paletteVC.emp).style('font-size', '10px').text(`Stirling: e^${d}/√(2π·${d}) ≈ ${stirling.toFixed(2)}`);
    },
    [data, d, half, containerWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        <svg ref={refA} width={half} height={HEIGHT} role="img" aria-label="Sauer–Shelah bounds versus 2 to the n" />
        <svg ref={refB} width={half} height={HEIGHT} role="img" aria-label="Ratio of closed form to binomial sum converging to the Stirling factor" />
      </div>
      <div style={{ marginTop: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
        <label htmlFor="ss-d">VC dimension d: {d}</label>
        <select id="ss-d" value={d} onChange={(e) => setD(Number(e.target.value))} aria-label="Choose VC dimension d">
          {D_OPTIONS.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

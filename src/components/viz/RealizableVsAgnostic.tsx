import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  agnosticSampleComplexity,
  paletteVC,
  realizableSampleComplexity,
} from './shared/vc-dimension';

// =============================================================================
// RealizableVsAgnostic — §7.5
//
// Sample-complexity curves: n_realizable(eps) vs n_agnostic(eps), at fixed
// d_VC, delta. Sliders for d, delta. Toggle between linear and log-log axes.
// Static fallback: public/images/topics/vc-dimension/07_realizable_vs_agnostic.png
// =============================================================================

const HEIGHT = 400;

export default function RealizableVsAgnostic() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [d, setD] = useState(5);
  const [delta, setDelta] = useState(0.05);
  const [logLog, setLogLog] = useState(true);

  const data = useMemo(() => {
    const epsValues: number[] = [];
    for (let exp = -3; exp <= -1.0; exp += 0.05) epsValues.push(Math.pow(10, exp));
    return epsValues.map((eps) => ({
      eps,
      realizable: realizableSampleComplexity(eps, delta, d),
      agnostic: agnosticSampleComplexity(eps, delta, d),
    }));
  }, [d, delta]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 22, right: 36, bottom: 48, left: 64 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xValues = data.map((r) => r.eps);
      const yValues = data.flatMap((r) => [r.realizable, r.agnostic]);

      const x = logLog ? d3.scaleLog().domain([Math.min(...xValues), Math.max(...xValues)]).range([0, W]) : d3.scaleLinear().domain([Math.min(...xValues), Math.max(...xValues)]).range([0, W]);
      const yDomain: [number, number] = logLog ? [Math.min(...yValues), Math.max(...yValues)] : [0, Math.max(...yValues)];
      const y = logLog ? d3.scaleLog().domain(yDomain).range([H, 0]) : d3.scaleLinear().domain(yDomain).range([H, 0]);

      g.append('g').attr('transform', `translate(0,${H})`).call((logLog ? d3.axisBottom(x).ticks(5, '~s') : d3.axisBottom(x).ticks(6)) as any);
      g.append('g').call((logLog ? d3.axisLeft(y).ticks(6, '~s') : d3.axisLeft(y).ticks(6, '~s')) as any);
      g.append('text').attr('x', W / 2).attr('y', H + 34).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text(logLog ? 'ε (log scale)' : 'ε (linear scale)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -50).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('sample complexity n*');

      const lineR = d3.line<typeof data[number]>().x((d) => x(d.eps)).y((d) => y(Math.max(1, d.realizable)));
      const lineA = d3.line<typeof data[number]>().x((d) => x(d.eps)).y((d) => y(Math.max(1, d.agnostic)));

      g.append('path').datum(data).attr('d', lineR).attr('fill', 'none').attr('stroke', paletteVC.primary).attr('stroke-width', 2.4);
      g.append('path').datum(data).attr('d', lineA).attr('fill', 'none').attr('stroke', paletteVC.emp).attr('stroke-width', 2.4);

      const legend = g.append('g').attr('transform', `translate(${Math.max(8, W - 220)}, 6)`);
      const items = [
        { label: 'realizable (1/ε rate)', color: paletteVC.primary },
        { label: 'agnostic (1/ε² rate)', color: paletteVC.emp },
      ];
      items.forEach((it, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', i * 16 + 6).attr('y2', i * 16 + 6).attr('stroke', it.color).attr('stroke-width', 2.4);
        legend.append('text').attr('x', 22).attr('y', i * 16 + 10).style('fill', 'var(--color-text)').style('font-size', '11px').text(it.label);
      });

      // Annotation at eps=0.01
      const annoEps = 0.01;
      const annoR = realizableSampleComplexity(annoEps, delta, d);
      const annoA = agnosticSampleComplexity(annoEps, delta, d);
      if (isFinite(annoR) && isFinite(annoA) && annoEps >= Math.min(...xValues) && annoEps <= Math.max(...xValues)) {
        g.append('line').attr('x1', x(annoEps)).attr('x2', x(annoEps)).attr('y1', 0).attr('y2', H).attr('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3').attr('opacity', 0.5);
        g.append('text').attr('x', x(annoEps) + 4).attr('y', 14).style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text(`at ε=0.01: ratio ≈ ${(annoA / annoR).toFixed(0)}×`);
      }
    },
    [data, d, delta, logLog, containerWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img" aria-label="Realizable versus agnostic sample-complexity curves" />
      <div style={{ marginTop: '0.6rem', display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '0.4rem 0.8rem', fontSize: '0.85rem', alignItems: 'center' }}>
        <label htmlFor="rva-d">VC dimension d: {d}</label>
        <input id="rva-d" type="range" min={1} max={20} value={d} onChange={(e) => setD(Number(e.target.value))} aria-label="VC dimension" />
        <label htmlFor="rva-delta">δ: {delta.toFixed(3)}</label>
        <input id="rva-delta" type="range" min={0.01} max={0.5} step={0.005} value={delta} onChange={(e) => setDelta(Number(e.target.value))} aria-label="Confidence parameter delta" />
        <label htmlFor="rva-loglog">Axis scale:</label>
        <span><input id="rva-loglog" type="checkbox" checked={logLog} onChange={(e) => setLogLog(e.target.checked)} /> log-log</span>
      </div>
    </div>
  );
}

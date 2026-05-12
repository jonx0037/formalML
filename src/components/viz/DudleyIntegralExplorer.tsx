import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dudleyIntegral, linearCoveringLogN } from './shared/generalization-bounds';

// =============================================================================
// DudleyIntegralExplorer — §8.  Closed-form viz: √(log N(ε)) integrand vs. ε
// with dyadic chain points marked, and the running integral underneath.
// All client-side, no MC.
// =============================================================================

const HEIGHT = 360;

export default function DudleyIntegralExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [d, setD] = useState(10);
  const [B, setB] = useState(1.0);
  const [n, setN] = useState(200);
  const [K, setK] = useState(8);

  const data = useMemo(() => {
    const Dn = 1.0;
    const epsDense = d3.range(Dn / 240, Dn + 1e-9, Dn / 240);
    const integrand = epsDense.map((e) => Math.sqrt(Math.max(0, linearCoveringLogN(e, d, B))));
    // Running integral via trapezoidal rule
    const running = new Float64Array(epsDense.length);
    let acc = 0;
    for (let i = 1; i < epsDense.length; i++) {
      const dx = epsDense[i] - epsDense[i - 1];
      acc += 0.5 * (integrand[i] + integrand[i - 1]) * dx;
      running[i] = acc;
    }
    const scales = d3.range(K).map((k) => Dn * Math.pow(2, -k));
    const total = dudleyIntegral(n, d, B, Dn, 200);
    return { epsDense, integrand, running, scales, total };
  }, [d, B, n, K]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 80, left: 60 };
      const w = containerWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([0.005, 1.0]).range([0, w]).clamp(true);
      const yMax = (d3.max(data.integrand) ?? 1) * 1.1;
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);
      const yR = d3.scaleLinear().domain([0, Math.max(0.001, data.running[data.running.length - 1])]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6, '~g'));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('g').attr('transform', `translate(${w},0)`).call(d3.axisRight(yR).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('ε (log scale)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', '#3b82f6').text('√(log N(ε))');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', w + 36).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', '#10b981').text('running integral');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text(`Dudley integrand for linear class d=${d}, B=${B.toFixed(1)}, Dudley ≈ ${data.total.toFixed(3)} (12/√n × area)`);

      // Integrand curve (filled)
      const line1 = d3.line<number>().x((_, i) => x(data.epsDense[i])).y((v) => y(v));
      const area = d3.area<number>().x((_, i) => x(data.epsDense[i])).y0(h).y1((v) => y(v));
      g.append('path').datum(Array.from(data.integrand))
        .attr('fill', '#3b82f6').attr('fill-opacity', 0.15).attr('d', area);
      g.append('path').datum(Array.from(data.integrand))
        .attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', line1);

      // Running integral
      const line2 = d3.line<number>().x((_, i) => x(data.epsDense[i])).y((v) => yR(v));
      g.append('path').datum(Array.from(data.running))
        .attr('fill', 'none').style('stroke', '#10b981').style('stroke-width', 2).attr('d', line2);

      // Dyadic chain markers
      data.scales.forEach((s, k) => {
        const xs = x(Math.max(0.005, s));
        g.append('line').attr('x1', xs).attr('x2', xs).attr('y1', 0).attr('y2', h)
          .style('stroke', '#dc2626').style('stroke-dasharray', '3 3').style('stroke-width', 1);
        if (k % 2 === 0) {
          g.append('text').attr('x', xs).attr('y', h + 18).attr('text-anchor', 'middle')
            .style('font-size', '9px').style('fill', '#7f1d1d').text(`ε_${k} = 2⁻${k}`);
        }
      });
    },
    [data, containerWidth, d, B],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <svg ref={svgRef} width={containerWidth || 720} height={HEIGHT} role="img" aria-label="Dudley integrand and running integral with dyadic chain markers." />
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          dimension d:
          <select value={d} onChange={(e) => setD(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8, fontSize: 12, padding: '2px 6px', borderRadius: 6, border: '1px solid var(--color-border)', background: 'var(--color-surface)', color: 'var(--color-text)' }}>
            {[1, 2, 5, 10, 25, 50, 100].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          norm B:
          <input type="range" min={0.1} max={5.0} step={0.1} value={B}
            onChange={(e) => setB(parseFloat(e.target.value))}
            style={{ marginLeft: 8, verticalAlign: 'middle' }} aria-label="norm B" />
          <span style={{ marginLeft: 6 }}>{B.toFixed(1)}</span>
        </label>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          n:
          <input type="range" min={50} max={5000} step={50} value={n}
            onChange={(e) => setN(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8, verticalAlign: 'middle' }} aria-label="sample size n" />
          <span style={{ marginLeft: 6 }}>{n}</span>
        </label>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          chain depth K:
          <input type="range" min={4} max={12} step={1} value={K}
            onChange={(e) => setK(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8, verticalAlign: 'middle' }} aria-label="chain depth K" />
          <span style={{ marginLeft: 6 }}>{K}</span>
        </label>
      </figcaption>
    </figure>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// -----------------------------------------------------------------------------
// Viz 12.1 — Boundary-evaluation runtime hierarchy.
//
// Log-log: closed-form Gaussian mixture O(1), implicit + bisection
// O(log(1/ε)), MC verification O(B·n). Static cost curves with notebook-
// derived anchor points.
// -----------------------------------------------------------------------------

const MARGIN = { top: 26, right: 14, bottom: 38, left: 60 };
const PANEL_H = 340;

const COLOR_CF = '#0f6e56';
const COLOR_IMPLICIT = '#7c3aed';
const COLOR_MC = '#c44545';

// Anchor points from notebook cell 31 / brief §12.2:
// closed-form ~10 μs at n=10^5, implicit ~5 ms, MC at B=1000 ~2 s.
function rt_closedform(n: number): number {
  return 1e-5; // ~10 μs flat
}
function rt_implicit(n: number): number {
  return 5e-3; // ~5 ms via bisection (log(1/ε) factor — flat-ish)
}
function rt_mc(n: number, B: number): number {
  return (B * n) * 2e-8; // empirical scaling
}

export default function BoundaryEvaluationRuntimeBenchmark() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [B, setB] = useState(1000);

  const ng = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i < 60; i++) out.push(10 ** (2 + 5 * (i / 59)));
    return out;
  }, []);

  const data = useMemo(() => ({
    cf: ng.map((n) => ({ n, t: rt_closedform(n) })),
    implicit: ng.map((n) => ({ n, t: rt_implicit(n) })),
    mc: ng.map((n) => ({ n, t: rt_mc(n, B) })),
  }), [ng, B]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([1e2, 1e7]).range([0, innerW]);
      const yScale = d3.scaleLog().domain([1e-6, 1e3]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text)');

      const line = d3.line<{ n: number; t: number }>().x((d) => xScale(d.n)).y((d) => yScale(d.t));
      const series = [
        { values: data.cf, color: COLOR_CF, label: 'closed-form O(1) — ~10 μs' },
        { values: data.implicit, color: COLOR_IMPLICIT, label: 'implicit + bisection O(log 1/ε) — ~5 ms' },
        { values: data.mc, color: COLOR_MC, label: `MC verification O(B·n), B=${B}` },
      ];
      series.forEach((s, i) => {
        g.append('path').datum(s.values).attr('d', line as never).style('fill', 'none').style('stroke', s.color).attr('stroke-width', 2);
        g.append('text').attr('x', innerW - 4).attr('y', 14 + i * 16).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', s.color).text(s.label);
      });

      // 1-second reference
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(1)).attr('y2', yScale(1)).style('stroke', 'var(--color-text-secondary)').attr('stroke-width', 0.8).attr('stroke-dasharray', '3 3');
      g.append('text').attr('x', 4).attr('y', yScale(1) - 4).style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('1 second');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('n (log scale)');
      g.append('text').attr('x', -innerH / 2).attr('y', -48).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('runtime per CS update (seconds, log scale)');
    },
    [data, w, B],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <span>MC replicates B = <strong>{B.toLocaleString()}</strong></span>
          <input type="range" min={100} max={10000} step={100} value={B} aria-label="MC replicates" onChange={(e) => setB(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
          Anchors from notebook cell 31: closed-form (~10 μs flat), implicit (~5 ms flat), MC at B = 1000, n = 10⁵ takes ~2 s. Use closed-form in production.
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

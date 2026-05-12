import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { finiteClassBound, finiteClassSampleComplexity } from './shared/generalization-bounds';

// =============================================================================
// UnionBoundCalculator — §3.  Closed-form panels for Theorem 1 and Corollary 1.
//
//   Left  panel — Theorem 1 bound vs. sample size n, one curve per |H|.
//   Right panel — Sample complexity n* vs. target ε, one curve per |H|.
//
// Closed-form arithmetic only — no MC, no precompute.
// =============================================================================

const HEIGHT = 280;
const SM_BREAKPOINT = 900;
const H_CHOICES = [10, 100, 1000, 10000];
const H_COLORS = ['#1E40AF', '#3B82F6', '#60A5FA', '#93C5FD'];

export default function UnionBoundCalculator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [delta, setDelta] = useState(0.05);
  const [M, setM] = useState(1.0);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor(containerWidth / 2);

  // Pre-sample the n grid and the epsilon grid
  const data = useMemo(() => {
    const nGrid = d3.range(20, 5001, 20);
    const epsGrid = d3.range(0.02, 0.31, 0.005);
    const boundCurves = H_CHOICES.map((N) =>
      nGrid.map((n) => ({ n, bound: finiteClassBound(N, n, delta, M) })),
    );
    const complexityCurves = H_CHOICES.map((N) =>
      epsGrid.map((eps) => ({ eps, nStar: finiteClassSampleComplexity(N, eps, delta, M) })),
    );
    return { nGrid, epsGrid, boundCurves, complexityCurves };
  }, [delta, M]);

  // Left panel: bound vs n
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 30, right: 12, bottom: 40, left: 50 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([20, 5000]).range([0, w]);
      const y = d3.scaleLinear().domain([0, Math.max(0.5, M * 1.05)]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text')
        .attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text('sample size n');
      g.append('text')
        .attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -38).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text('bound');
      g.append('text')
        .attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Theorem 1 bound vs. n');

      const line = d3.line<{ n: number; bound: number }>()
        .x((d) => x(d.n)).y((d) => y(d.bound));

      data.boundCurves.forEach((curve, i) => {
        g.append('path')
          .datum(curve)
          .attr('fill', 'none')
          .style('stroke', H_COLORS[i])
          .style('stroke-width', 2)
          .attr('d', line);
        // Label at the right end
        const last = curve[curve.length - 1];
        g.append('text')
          .attr('x', x(last.n) - 4).attr('y', y(last.bound) - 4)
          .attr('text-anchor', 'end')
          .style('font-size', '10px').style('fill', H_COLORS[i])
          .text(`|H|=${H_CHOICES[i].toLocaleString()}`);
      });
    },
    [data, panelWidth, M],
  );

  // Right panel: sample complexity vs epsilon
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 30, right: 12, bottom: 40, left: 56 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const maxN = d3.max(data.complexityCurves.flat(), (d) => d.nStar) ?? 1;
      const x = d3.scaleLinear().domain([0, 0.31]).range([0, w]);
      const y = d3.scaleLog().domain([10, Math.max(maxN, 100)]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~s'));
      g.append('text')
        .attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text('target ε');
      g.append('text')
        .attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text('sample complexity n*');
      g.append('text')
        .attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Corollary 1 sample complexity vs. ε');

      const line = d3.line<{ eps: number; nStar: number }>()
        .x((d) => x(d.eps)).y((d) => y(Math.max(10, d.nStar)));

      data.complexityCurves.forEach((curve, i) => {
        g.append('path')
          .datum(curve)
          .attr('fill', 'none')
          .style('stroke', H_COLORS[i])
          .style('stroke-width', 2)
          .attr('d', line);
      });
    },
    [data, panelWidth],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isStacked ? '1fr' : '1fr 1fr',
          gap: 8,
        }}
      >
        <svg ref={leftRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Theorem 1 bound vs sample size n for various class sizes." />
        <svg ref={rightRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Corollary 1 sample complexity vs target epsilon." />
      </div>
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          confidence δ:
          <input
            type="range" min={0.005} max={0.5} step={0.005} value={delta}
            onChange={(e) => setDelta(parseFloat(e.target.value))}
            style={{ marginLeft: 8, verticalAlign: 'middle' }}
            aria-label="confidence delta"
          />
          <span style={{ marginLeft: 6 }}>{delta.toFixed(3)}</span>
        </label>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          loss range M:
          <input
            type="range" min={0.1} max={2.0} step={0.1} value={M}
            onChange={(e) => setM(parseFloat(e.target.value))}
            style={{ marginLeft: 8, verticalAlign: 'middle' }}
            aria-label="loss range M"
          />
          <span style={{ marginLeft: 6 }}>{M.toFixed(1)}</span>
        </label>
      </figcaption>
    </figure>
  );
}

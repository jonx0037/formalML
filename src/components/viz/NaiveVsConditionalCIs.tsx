import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// Locked notebook outputs from §4.5 (cell 19). Coverage × category × procedure.
// Median naive width: 0.401–0.404. Median conditional width (finite): 0.483 (signal), 1.300 (null).
// Infinite-CI rate: 3.0% (signal), 12.4% (null).

const COVERAGE = [
  { category: 'Signal', procedure: 'Naive Wald', coverage: 0.925 },
  { category: 'Signal', procedure: 'Conditional', coverage: 0.955 },
  { category: 'Null', procedure: 'Naive Wald', coverage: 0.789 },
  { category: 'Null', procedure: 'Conditional', coverage: 0.967 },
];

const WIDTH_STATS = [
  { category: 'Signal · naive', median: 0.404, p10: 0.31, p90: 0.55, infFrac: 0 },
  { category: 'Signal · conditional', median: 0.483, p10: 0.34, p90: 0.92, infFrac: 0.030 },
  { category: 'Null · naive', median: 0.401, p10: 0.30, p90: 0.55, infFrac: 0 },
  { category: 'Null · conditional', median: 1.300, p10: 0.55, p90: 3.6, infFrac: 0.124 },
];

const PROC_COLOR: Record<string, string> = {
  'Naive Wald': '#8a8a8a',
  Conditional: 'var(--color-accent)',
};

export default function NaiveVsConditionalCIs() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [showRule, setShowRule] = useState(true);

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const isMobile = W < 640;
  const PANEL_W = isMobile ? W : (W - 16) / 2;
  const PANEL_H = Math.round(PANEL_W * 0.72);
  const M = { t: 22, r: 14, b: 50, l: 50 };

  const coverageRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - M.l - M.r;
      const ih = PANEL_H - M.t - M.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

      const categories = ['Signal', 'Null'];
      const procedures = ['Naive Wald', 'Conditional'];
      const x0 = d3.scaleBand<string>().domain(categories).range([0, iw]).padding(0.3);
      const x1 = d3.scaleBand<string>().domain(procedures).range([0, x0.bandwidth()]).padding(0.12);
      const y = d3.scaleLinear().domain([0.7, 1]).range([ih, 0]);

      for (const cat of categories) {
        const groupG = g.append('g').attr('transform', `translate(${x0(cat)},0)`);
        for (const proc of procedures) {
          const row = COVERAGE.find((r) => r.category === cat && r.procedure === proc)!;
          groupG
            .append('rect')
            .attr('x', x1(proc) ?? 0)
            .attr('y', y(row.coverage))
            .attr('width', x1.bandwidth())
            .attr('height', ih - y(row.coverage))
            .attr('fill', PROC_COLOR[proc])
            .attr('opacity', 0.85);
          groupG
            .append('text')
            .attr('x', (x1(proc) ?? 0) + x1.bandwidth() / 2)
            .attr('y', y(row.coverage) - 4)
            .attr('text-anchor', 'middle')
            .style('fill', 'var(--color-text)')
            .style('font-family', 'var(--font-sans)')
            .style('font-size', '0.74rem')
            .text(row.coverage.toFixed(3));
        }
      }

      // 0.95 nominal line
      if (showRule) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', iw)
          .attr('y1', y(0.95))
          .attr('y2', y(0.95))
          .style('stroke', 'var(--color-text)')
          .attr('stroke-dasharray', '4 3')
          .attr('opacity', 0.6);
        g.append('text')
          .attr('x', iw - 2)
          .attr('y', y(0.95) - 4)
          .attr('text-anchor', 'end')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '0.7rem')
          .text('nominal 0.95');
      }

      g.append('g')
        .attr('transform', `translate(0,${ih})`)
        .call(d3.axisBottom(x0))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('empirical coverage');
    },
    [PANEL_W, PANEL_H, showRule],
  );

  const widthsRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = PANEL_W - M.l - M.r;
      const ih = PANEL_H - M.t - M.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

      const y = d3.scaleBand<string>().domain(WIDTH_STATS.map((s) => s.category)).range([0, ih]).padding(0.25);
      const x = d3.scaleLog().domain([0.2, 6]).range([0, iw]).clamp(true);

      g.append('g')
        .attr('transform', `translate(0,${ih})`)
        .call(d3.axisBottom(x).tickValues([0.3, 0.5, 1, 2, 4]).tickFormat(d3.format('.1f')))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y)).selectAll('text').style('fill', 'var(--color-text)').style('font-size', '0.72rem');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');

      for (const stat of WIDTH_STATS) {
        const yMid = (y(stat.category) ?? 0) + y.bandwidth() / 2;
        const fill = stat.category.includes('conditional') ? PROC_COLOR.Conditional : PROC_COLOR['Naive Wald'];
        // p10–p90 range
        g.append('line')
          .attr('x1', x(stat.p10))
          .attr('x2', x(stat.p90))
          .attr('y1', yMid)
          .attr('y2', yMid)
          .style('stroke', fill)
          .attr('stroke-width', 2.5)
          .attr('opacity', 0.85);
        // median point
        g.append('circle').attr('cx', x(stat.median)).attr('cy', yMid).attr('r', 5).attr('fill', fill);
        // infinite-CI tag
        if (stat.infFrac > 0) {
          g.append('text')
            .attr('x', x(5.5))
            .attr('y', yMid + 4)
            .attr('text-anchor', 'end')
            .style('fill', 'var(--color-text-secondary)')
            .style('font-size', '0.7rem')
            .text(`${(stat.infFrac * 100).toFixed(1)}% to ±∞`);
        }
      }
      g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.7rem').text('CI width (log scale)');
      g.append('text').attr('x', 8).attr('y', 10).style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('median + 10–90% range');
    },
    [PANEL_W, PANEL_H],
  );

  const summary = useMemo(() => {
    const sigImpr = ((0.955 - 0.925) * 100).toFixed(1);
    const nullImpr = ((0.967 - 0.789) * 100).toFixed(1);
    return { sigImpr, nullImpr };
  }, []);

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-wrap items-center gap-4 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={showRule} onChange={(e) => setShowRule(e.target.checked)} />
          <span>show nominal 0.95</span>
        </label>
        <span>
          Conditional procedure gains{' '}
          <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>
            +{summary.sigImpr}
          </span>{' '}
          pp on signals,{' '}
          <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>
            +{summary.nullImpr}
          </span>{' '}
          pp on nulls.
        </span>
      </div>
      <div className={isMobile ? 'mt-2' : 'mt-2 flex gap-4'}>
        <svg ref={coverageRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Empirical coverage by category and procedure" />
        <svg ref={widthsRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Confidence-interval width distribution" />
      </div>
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        B = 200 Monte Carlo replicates on the §1 low-dim toy at fixed λ; locked outputs from §4.5. The conditional procedure
        delivers the nominal 0.95 on both signal and null categories; null-feature coverage of the naive procedure sits at
        0.789. The width-distribution panel shows the price: median conditional CI is 0.483 on signals (vs naive 0.404) and
        1.300 on nulls (vs naive 0.401), with 12.4% of null-feature conditional CIs extending to ±∞.
      </p>
    </div>
  );
}

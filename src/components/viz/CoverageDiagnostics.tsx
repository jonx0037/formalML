import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  crossFitDmlPlr,
  linearRobinsonNuisanceFitter,
  mulberry32,
  paletteSemi,
  polyDeg3RobinsonNuisanceFitter,
  ROBINSON_THETA_0,
  sampleRobinson,
} from './shared/semiparametric-inference';

// =============================================================================
// CoverageDiagnostics — §10
// Empirical coverage of Wald CIs at three target levels (90%, 95%, 99%) as a
// function of n. Two nuisance choices (linear vs polynomial-degree-3).
// Right: CI half-width as a function of n.
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 640;
const N_GRID = [200, 500, 1000, 2000];
const ALPHA_LEVELS = [
  { name: '90%', z: 1.645, target: 0.90, color: paletteSemi.accentB },
  { name: '95%', z: 1.96, target: 0.95, color: paletteSemi.oneStep },
  { name: '99%', z: 2.576, target: 0.99, color: paletteSemi.dml },
];

export default function CoverageDiagnostics() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [usePoly3, setUsePoly3] = useState(true);
  const [displayB, setDisplayB] = useState(80);
  const [committedB, setCommittedB] = useState(80);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const fitter = usePoly3 ? polyDeg3RobinsonNuisanceFitter : linearRobinsonNuisanceFitter;
    const result: { n: number; cov90: number; cov95: number; cov99: number; halfWidth95: number }[] = [];
    for (const n of N_GRID) {
      let c90 = 0, c95 = 0, c99 = 0;
      let halfWidthSum = 0;
      for (let b = 0; b < committedB; b++) {
        const rng = mulberry32(20260515 + b * 7 + n);
        const sample = sampleRobinson(n, ROBINSON_THETA_0, rng);
        const r = crossFitDmlPlr(sample.X, sample.D, sample.Y, 5, fitter, rng);
        if (Math.abs(r.theta - ROBINSON_THETA_0) <= ALPHA_LEVELS[0].z * r.se) c90 += 1;
        if (Math.abs(r.theta - ROBINSON_THETA_0) <= ALPHA_LEVELS[1].z * r.se) c95 += 1;
        if (Math.abs(r.theta - ROBINSON_THETA_0) <= ALPHA_LEVELS[2].z * r.se) c99 += 1;
        halfWidthSum += ALPHA_LEVELS[1].z * r.se;
      }
      result.push({
        n,
        cov90: c90 / committedB,
        cov95: c95 / committedB,
        cov99: c99 / committedB,
        halfWidth95: halfWidthSum / committedB,
      });
    }
    return result;
  }, [usePoly3, committedB]);

  const refL = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 24, right: 14, bottom: 36, left: 56 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLog().domain([N_GRID[0] * 0.9, N_GRID[N_GRID.length - 1] * 1.1]).range([0, W]);
      const yScale = d3.scaleLinear().domain([0.7, 1.02]).range([H, 0]);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5, '.0f'));
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '.0%'));
      // Nominal target lines.
      ALPHA_LEVELS.forEach((L) => {
        g.append('line').attr('x1', 0).attr('x2', W)
          .attr('y1', yScale(L.target)).attr('y2', yScale(L.target))
          .style('stroke', L.color).style('stroke-width', 0.8)
          .style('stroke-dasharray', '3,3').style('opacity', 0.6);
      });
      // Coverage curves.
      const curves = [
        { key: 'cov90' as const, color: ALPHA_LEVELS[0].color, label: '90%' },
        { key: 'cov95' as const, color: ALPHA_LEVELS[1].color, label: '95%' },
        { key: 'cov99' as const, color: ALPHA_LEVELS[2].color, label: '99%' },
      ];
      curves.forEach((c) => {
        const ys = data.map((d) => d[c.key]);
        g.append('path').datum(ys).attr('d',
          d3.line<number>().x((_d, i) => xScale(N_GRID[i])).y((d) => yScale(d)) as unknown as string)
          .style('fill', 'none').style('stroke', c.color).style('stroke-width', 2);
        ys.forEach((y, i) => {
          g.append('circle').attr('cx', xScale(N_GRID[i])).attr('cy', yScale(y))
            .attr('r', 3).style('fill', c.color);
        });
        g.append('text').attr('x', xScale(N_GRID[N_GRID.length - 1])).attr('y', yScale(ys[ys.length - 1]) - 6)
          .style('fill', c.color).style('font-family', 'var(--font-mono)').style('font-size', 11)
          .attr('text-anchor', 'end').text(c.label);
      });
      g.append('text').attr('x', 0).attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text('Empirical coverage vs n (target ↔ horizontal dashed lines)');
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('n (log)');
    },
    [data, containerWidth, isMobile],
  );

  const refR = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 24, right: 14, bottom: 36, left: 56 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLog().domain([N_GRID[0] * 0.9, N_GRID[N_GRID.length - 1] * 1.1]).range([0, W]);
      const ys = data.map((d) => d.halfWidth95);
      const yScale = d3.scaleLog().domain([Math.min(...ys) * 0.8, Math.max(...ys) * 1.2]).range([H, 0]);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5, '.0f'));
      g.append('g').call(d3.axisLeft(yScale).ticks(4, '.3f'));
      g.append('path').datum(ys).attr('d',
        d3.line<number>().x((_d, i) => xScale(N_GRID[i])).y((d) => yScale(d)) as unknown as string)
        .style('fill', 'none').style('stroke', paletteSemi.oneStep).style('stroke-width', 2);
      ys.forEach((y, i) => {
        g.append('circle').attr('cx', xScale(N_GRID[i])).attr('cy', yScale(y))
          .attr('r', 3).style('fill', paletteSemi.oneStep);
      });
      g.append('text').attr('x', 0).attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text('95% Wald CI half-width vs n (log-log)');
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('n (log)');
    },
    [data, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          <input type="checkbox" checked={usePoly3} onChange={(e) => setUsePoly3(e.target.checked)} />
          Polynomial-degree-3 nuisance (uncheck for linear-control nuisance)
        </label>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          MC replications B per n: <strong>{displayB}</strong>
          <input
            type="range"
            min={40}
            max={150}
            step={10}
            value={displayB}
            onChange={(e) => setDisplayB(+e.target.value)}
            onMouseUp={() => setCommittedB(displayB)}
            onTouchEnd={() => setCommittedB(displayB)}
            onKeyUp={() => setCommittedB(displayB)}
            aria-label="Monte Carlo replications"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 8 }}>
        <svg ref={refL} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT, display: 'block' }} />
        <svg ref={refR} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT, display: 'block' }} />
      </div>
    </div>
  );
}

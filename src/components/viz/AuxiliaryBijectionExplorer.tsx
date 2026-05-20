// =============================================================================
// AuxiliaryBijectionExplorer.tsx
//
// §5 of reversible-jump-mcmc. Concrete 2D-to-2D auxiliary-variable bijection
// T(β, u) = (β - α u, β + α u), the simplest "centered split". |det J_T| = 2α
// is the area-distortion factor. Slider controls α; both grid and test
// rectangle update live.
//
// Static fallback: /images/topics/reversible-jump-mcmc/fig05_auxiliary_bijection.png
// =============================================================================
import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

interface GridPoint {
  beta: number;
  u: number;
}

export default function AuxiliaryBijectionExplorer() {
  const [alpha, setAlpha] = useState(1.0);

  const sourceGrid = useMemo(() => {
    const pts: GridPoint[] = [];
    for (let bi = 0; bi < 13; bi++) {
      for (let ui = 0; ui < 7; ui++) {
        pts.push({ beta: -3 + bi * 0.5, u: 0.5 + ui * 0.4 });
      }
    }
    return pts;
  }, []);

  const apply = (pt: GridPoint): { x: number; y: number } => ({
    x: pt.beta - alpha * pt.u,
    y: pt.beta + alpha * pt.u,
  });

  // Source test rectangle, unit area
  const testRect = useMemo(
    () => [
      { beta: -0.5, u: 1.5 },
      { beta: 0.5, u: 1.5 },
      { beta: 0.5, u: 2.5 },
      { beta: -0.5, u: 2.5 },
    ],
    [],
  );

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 2;
  const panelH = 320;

  const detJ = Math.abs(2 * alpha);

  const srcRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 24, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const x = d3.scaleLinear().domain([-4, 4]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 4]).range([h, 0]);

      g.selectAll('circle.gp')
        .data(sourceGrid)
        .join('circle')
        .attr('class', 'gp')
        .attr('cx', (p) => x(p.beta))
        .attr('cy', (p) => y(p.u))
        .attr('r', 3)
        .style('fill', 'var(--color-border)');

      g.append('polygon')
        .attr('points', testRect.map((p) => `${x(p.beta)},${y(p.u)}`).join(' '))
        .style('fill', 'var(--color-accent)')
        .style('opacity', 0.35)
        .style('stroke', 'var(--color-accent)')
        .style('stroke-width', 1.5);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('β');
      g.append('text').attr('x', -h / 2).attr('y', -28).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('u');
      g.append('text').attr('x', w / 2).attr('y', -8).attr('text-anchor', 'middle').style('font-size', '13px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('Source (β, u) — test rectangle area 1');
    },
    [sourceGrid, testRect, panelW, panelH],
  );

  const tgtRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 24, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const x = d3.scaleLinear().domain([-7, 7]).range([0, w]);
      const y = d3.scaleLinear().domain([-7, 7]).range([h, 0]);

      g.selectAll('circle.gp')
        .data(sourceGrid)
        .join('circle')
        .attr('class', 'gp')
        .attr('cx', (p) => {
          const t = apply(p);
          return x(t.x);
        })
        .attr('cy', (p) => {
          const t = apply(p);
          return y(t.y);
        })
        .attr('r', 3)
        .style('fill', 'var(--color-border)');

      const imageRect = testRect.map(apply);
      g.append('polygon')
        .attr('points', imageRect.map((p) => `${x(p.x)},${y(p.y)}`).join(' '))
        .style('fill', 'var(--color-accent)')
        .style('opacity', 0.35)
        .style('stroke', 'var(--color-accent)')
        .style('stroke-width', 1.5);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('β₁ = β − α u');
      g.append('text').attr('x', -h / 2).attr('y', -28).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('β₂ = β + α u');
      g.append('text').attr('x', w / 2).attr('y', -8).attr('text-anchor', 'middle').style('font-size', '13px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text(`Target T(β, u) — area = |det J_T| = ${detJ.toFixed(2)}`);
    },
    [sourceGrid, testRect, alpha, panelW, panelH],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          marginBottom: 12,
          fontFamily: 'var(--font-sans)',
          fontSize: 13,
          color: 'var(--color-text)',
        }}
      >
        <label htmlFor="alpha" style={{ minWidth: 110 }}>
          α = <strong>{alpha.toFixed(2)}</strong>
        </label>
        <input
          id="alpha"
          type="range"
          min={0.25}
          max={3.0}
          step={0.05}
          value={alpha}
          onChange={(e) => setAlpha(parseFloat(e.target.value))}
          style={{ flex: 1 }}
          aria-label="Bijection scale alpha"
        />
        <span
          style={{
            minWidth: 110,
            fontFamily: 'var(--font-mono)',
            color: 'var(--color-text-secondary)',
          }}
        >
          |det J_T| = {detJ.toFixed(2)}
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: 24,
          alignItems: 'flex-start',
        }}
      >
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={srcRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={tgtRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}

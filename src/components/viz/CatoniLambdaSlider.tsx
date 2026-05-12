import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  empiricalRiskOnGrid,
  gibbsOnGrid,
  mulberry32,
  sampleNormalThresholdProblem,
  thresholdGrid,
} from './shared/pac-bayes-bounds';

// =============================================================================
// CatoniLambdaSlider — §6.5a
// Gibbs distribution Q*_λ ∝ P · exp(−λ · R̂_S) on the threshold grid at five
// representative temperatures, plus a slider showing the current Q*_λ.  At
// λ = 0 we recover P (uniform); at λ → ∞ we get point-mass on the ERM cell.
// =============================================================================

const N = 200;
const ETA = 0.05;
const MU_STAR = 0.3;
const SEED = 20260511;
const HEIGHT = 340;
const REFERENCE_LAMBDAS = [0, 10, 50, 200, 5000];

export default function CatoniLambdaSlider() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [lambda, setLambda] = useState(50);

  const data = useMemo(() => {
    const rng = mulberry32(SEED >>> 0);
    const { X, Y } = sampleNormalThresholdProblem(N, ETA, MU_STAR, rng);
    const grid = thresholdGrid();
    const risks = empiricalRiskOnGrid(X, Y, grid);
    const P = new Float64Array(grid.length).fill(1 / grid.length);
    const references = REFERENCE_LAMBDAS.map((lam) => ({ lam, Q: gibbsOnGrid(P, risks, lam) }));
    const current = gibbsOnGrid(P, risks, lambda);
    return { grid, risks, P, references, current };
  }, [lambda]);

  return (
    <div
      ref={containerRef}
      style={{
        margin: '1.5rem 0',
        padding: '1rem',
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: '0.5rem',
      }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', alignItems: 'center', marginBottom: '0.5rem' }}>
        <label htmlFor="cls-lam" style={{ display: 'flex', flexDirection: 'column', fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>
          <span>temperature λ = {lambda} (log slider)</span>
          <input
            id="cls-lam"
            type="range"
            value={Math.log10(Math.max(1, lambda))}
            min={0}
            max={4}
            step={0.05}
            onChange={(e) => setLambda(Math.round(Math.pow(10, parseFloat(e.target.value))))}
            style={{ width: '13rem' }}
          />
        </label>
        <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          Q*_λ(h) ∝ P(h) · exp(−λ R̂_S(h)) &nbsp;|&nbsp; sharpens from uniform (λ=0) toward point-mass on ERM
        </div>
      </div>
      <Panel data={data} currentLambda={lambda} width={containerWidth || 800} />
    </div>
  );
}

type DataT = {
  grid: Float64Array;
  risks: Float64Array;
  P: Float64Array;
  references: { lam: number; Q: Float64Array }[];
  current: Float64Array;
};

function Panel({ data, currentLambda, width }: { data: DataT; currentLambda: number; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 110, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([-2, 2]).range([0, w]);
      const yMax = Math.max(
        d3.max(data.current) ?? 0.1,
        ...data.references.map((r) => d3.max(r.Q) ?? 0.1),
      );
      const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3f')));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('threshold t');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('Q*_λ(h_t)');

      const refColors = d3.schemeBlues[7].slice(2);
      const lineGen = d3.line<number>().x((_, i) => x(data.grid[i])).y((v) => y(v));

      data.references.forEach((r, idx) => {
        g.append('path').datum(Array.from(r.Q))
          .style('fill', 'none').style('stroke', refColors[idx])
          .style('stroke-width', 1.3).style('stroke-dasharray', '3,2').style('opacity', 0.75)
          .attr('d', lineGen);
      });

      g.append('path').datum(Array.from(data.current))
        .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.6)
        .attr('d', lineGen);

      // Legend
      const legend = g.append('g').attr('transform', `translate(${w + 12},14)`);
      data.references.forEach((r, idx) => {
        legend.append('line').attr('x1', 0).attr('x2', 14)
          .attr('y1', idx * 14).attr('y2', idx * 14)
          .style('stroke', refColors[idx]).style('stroke-width', 1.6).style('stroke-dasharray', '3,2');
        legend.append('text').attr('x', 18).attr('y', idx * 14 + 3)
          .style('font-size', '10px').style('fill', 'var(--color-text-secondary)')
          .text(`λ = ${r.lam}`);
      });
      const cur = data.references.length * 14;
      legend.append('line').attr('x1', 0).attr('x2', 14)
        .attr('y1', cur).attr('y2', cur)
        .style('stroke', 'var(--color-accent)').style('stroke-width', 2.6);
      legend.append('text').attr('x', 18).attr('y', cur + 3)
        .style('font-size', '10px').style('fill', 'var(--color-accent)')
        .text(`λ = ${currentLambda}`);

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Gibbs sharpening with temperature');
    },
    [data, currentLambda, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

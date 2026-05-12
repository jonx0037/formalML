import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  empiricalRiskOnGrid,
  ermOnGrid,
  mulberry32,
  sampleNormalThresholdProblem,
  thresholdGrid,
  trueRiskOnGrid,
} from './shared/pac-bayes-bounds';

// =============================================================================
// UniformConvergenceVsPACBayes — §1.5
// Two-panel running-example demo: data scatter with μ* and ERM threshold marked,
// then empirical + true risk across the |T|=101 threshold grid.  All sync TS;
// reshuffle by changing seed.
// =============================================================================

const N = 200;
const ETA = 0.05;
const MU_STAR = 0.3;
const HEIGHT = 340;
const SM_BREAKPOINT = 640;

export default function UniformConvergenceVsPACBayes() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [seed, setSeed] = useState(20260511);

  const data = useMemo(() => {
    const rng = mulberry32(seed >>> 0);
    const { X, Y } = sampleNormalThresholdProblem(N, ETA, MU_STAR, rng);
    const grid = thresholdGrid();
    const empRisks = empiricalRiskOnGrid(X, Y, grid);
    const trueRisks = trueRiskOnGrid(grid, ETA, MU_STAR);
    const erm = ermOnGrid(grid, empRisks);
    return { X, Y, grid, empRisks, trueRisks, erm };
  }, [seed]);

  const isMobile = (containerWidth || 800) < SM_BREAKPOINT;
  const panelWidth = isMobile ? (containerWidth || 800) : Math.max(280, (containerWidth || 800) / 2 - 8);

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
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
        <label htmlFor="ucp-seed" style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
          seed
        </label>
        <input
          id="ucp-seed"
          type="number"
          value={seed}
          onChange={(e) => setSeed(parseInt(e.target.value || '0', 10))}
          style={{
            width: '8rem',
            padding: '0.25rem 0.5rem',
            background: 'var(--color-bg)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text)',
            borderRadius: '0.25rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.85rem',
          }}
        />
        <span style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
          n = {N}, η = {ETA}, μ* = {MU_STAR}, ERM τ̂ = {data.erm.tau.toFixed(2)}, R̂(τ̂) = {data.erm.risk.toFixed(3)}
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
        <ScatterPanel data={data} width={panelWidth} />
        <RiskCurvePanel data={data} width={panelWidth} />
      </div>
    </div>
  );
}

type DataT = {
  X: Float64Array;
  Y: Int8Array;
  grid: Float64Array;
  empRisks: Float64Array;
  trueRisks: Float64Array;
  erm: { tau: number; risk: number };
};

function ScatterPanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (width <= 0) return;
    const margin = { top: 26, right: 20, bottom: 36, left: 44 };
    const w = width - margin.left - margin.right;
    const h = HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([-3.5, 3.5]).range([0, w]);
    const yPos = h * 0.32;
    const yNeg = h * 0.68;

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(7));
    g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('X');
    g.append('text').attr('x', -10).attr('y', yPos + 4).attr('text-anchor', 'end')
      .style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('y = +1');
    g.append('text').attr('x', -10).attr('y', yNeg + 4).attr('text-anchor', 'end')
      .style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('y = −1');

    // μ* line (truth)
    g.append('line')
      .attr('x1', x(MU_STAR)).attr('x2', x(MU_STAR))
      .attr('y1', 0).attr('y2', h)
      .style('stroke', 'var(--color-accent)').style('stroke-width', 2)
      .style('stroke-dasharray', '4,3');
    g.append('text').attr('x', x(MU_STAR) + 4).attr('y', 12)
      .style('font-size', '10px').style('fill', 'var(--color-accent)').text(`μ* = ${MU_STAR}`);

    // ERM line
    g.append('line')
      .attr('x1', x(data.erm.tau)).attr('x2', x(data.erm.tau))
      .attr('y1', 0).attr('y2', h)
      .style('stroke', '#D97706').style('stroke-width', 2);
    g.append('text').attr('x', x(data.erm.tau) + 4).attr('y', 24)
      .style('font-size', '10px').style('fill', '#D97706').text(`τ̂ = ${data.erm.tau.toFixed(2)}`);

    g.append('g').selectAll('circle').data(Array.from(data.X)).join('circle')
      .attr('cx', (xi) => x(xi))
      .attr('cy', (_, i) => (data.Y[i] === 1 ? yPos : yNeg))
      .attr('r', 3.2)
      .style('fill', (_, i) => (data.Y[i] === 1 ? 'var(--color-accent)' : '#534AB7'))
      .style('opacity', 0.55);

    g.append('text').attr('x', 0).attr('y', -8)
      .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
      .text('Panel A — sample (n = 200), truth at μ*, ERM at τ̂');
  }, [data, width]);
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

function RiskCurvePanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (width <= 0) return;
    const margin = { top: 26, right: 20, bottom: 36, left: 44 };
    const w = width - margin.left - margin.right;
    const h = HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([-2, 2]).range([0, w]);
    const yMax = Math.max(
      d3.max(data.empRisks) ?? 0.6,
      d3.max(data.trueRisks) ?? 0.6,
    );
    const y = d3.scaleLinear().domain([0, Math.min(0.6, yMax + 0.05)]).range([h, 0]);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
    g.append('g').call(d3.axisLeft(y).ticks(5));
    g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('threshold t');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -34).attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('risk');

    const lineGen = d3.line<number>()
      .x((_, i) => x(data.grid[i]))
      .y((v) => y(v));

    g.append('path').datum(Array.from(data.trueRisks))
      .style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 2)
      .attr('d', lineGen);
    g.append('path').datum(Array.from(data.empRisks))
      .style('fill', 'none').style('stroke', '#D97706').style('stroke-width', 2)
      .attr('d', lineGen);

    g.append('line')
      .attr('x1', x(MU_STAR)).attr('x2', x(MU_STAR))
      .attr('y1', 0).attr('y2', h)
      .style('stroke', 'var(--color-accent)').style('stroke-dasharray', '4,3');
    g.append('line')
      .attr('x1', x(data.erm.tau)).attr('x2', x(data.erm.tau))
      .attr('y1', 0).attr('y2', h)
      .style('stroke', '#D97706').style('stroke-dasharray', '2,2');

    g.append('text').attr('x', 0).attr('y', -8)
      .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
      .text('Panel B — risks across the threshold grid');

    // Legend (top-right of plot area)
    const legend = g.append('g').attr('transform', `translate(${w - 96},10)`);
    legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0)
      .style('stroke', 'var(--color-text)').style('stroke-width', 2);
    legend.append('text').attr('x', 22).attr('y', 3)
      .style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('R(h_t)');
    legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 14).attr('y2', 14)
      .style('stroke', '#D97706').style('stroke-width', 2);
    legend.append('text').attr('x', 22).attr('y', 17)
      .style('font-size', '10px').style('fill', 'var(--color-text-secondary)').text('R̂_S(h_t)');
  }, [data, width]);
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

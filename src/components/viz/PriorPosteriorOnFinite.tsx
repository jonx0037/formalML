import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  empiricalRiskOnGrid,
  ermOnGrid,
  expectedUnderQ,
  gaussianOnGridPosterior,
  klDiscrete,
  mulberry32,
  sampleNormalThresholdProblem,
  thresholdGrid,
} from './shared/pac-bayes-bounds';

// =============================================================================
// PriorPosteriorOnFinite — §2.5
// Two-panel: P uniform vs Q_narrow / Q_broad on the threshold grid; empirical
// risk curve with horizontal markers at E_Q[Rhat] for each posterior.
// Sliders: σ_Q_narrow, σ_Q_broad.  All sync TS.
// =============================================================================

const N = 200;
const ETA = 0.05;
const MU_STAR = 0.3;
const SEED = 20260511;
const HEIGHT = 340;
const SM_BREAKPOINT = 640;

export default function PriorPosteriorOnFinite() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sigmaNarrow, setSigmaNarrow] = useState(0.10);
  const [sigmaBroad, setSigmaBroad] = useState(0.50);

  const data = useMemo(() => {
    const rng = mulberry32(SEED >>> 0);
    const { X, Y } = sampleNormalThresholdProblem(N, ETA, MU_STAR, rng);
    const grid = thresholdGrid();
    const empRisks = empiricalRiskOnGrid(X, Y, grid);
    const erm = ermOnGrid(grid, empRisks);
    const P = new Float64Array(grid.length).fill(1 / grid.length);
    const Qn = gaussianOnGridPosterior(grid, erm.tau, sigmaNarrow, P);
    const Qb = gaussianOnGridPosterior(grid, erm.tau, sigmaBroad, P);
    return {
      grid,
      empRisks,
      P,
      Qn,
      Qb,
      erm,
      klNarrow: klDiscrete(Qn, P),
      klBroad: klDiscrete(Qb, P),
      ehatNarrow: expectedUnderQ(Qn, empRisks),
      ehatBroad: expectedUnderQ(Qb, empRisks),
    };
  }, [sigmaNarrow, sigmaBroad]);

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
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', alignItems: 'center', marginBottom: '0.5rem' }}>
        <SliderField
          label={`σ_Q narrow = ${sigmaNarrow.toFixed(2)}`}
          value={sigmaNarrow}
          min={0.03}
          max={0.40}
          step={0.01}
          onChange={setSigmaNarrow}
          id="ppof-sn"
        />
        <SliderField
          label={`σ_Q broad = ${sigmaBroad.toFixed(2)}`}
          value={sigmaBroad}
          min={0.15}
          max={1.5}
          step={0.05}
          onChange={setSigmaBroad}
          id="ppof-sb"
        />
        <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          KL(Q_n ∥ P) = {data.klNarrow.toFixed(2)} &nbsp;|&nbsp; E_Q_n[R̂] = {data.ehatNarrow.toFixed(3)}<br />
          KL(Q_b ∥ P) = {data.klBroad.toFixed(2)} &nbsp;|&nbsp; E_Q_b[R̂] = {data.ehatBroad.toFixed(3)}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
        <DistributionPanel data={data} width={panelWidth} />
        <RiskWithExpectationPanel data={data} width={panelWidth} />
      </div>
    </div>
  );
}

function SliderField({
  label, value, min, max, step, onChange, id,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  id: string;
}) {
  return (
    <label htmlFor={id} style={{ display: 'flex', flexDirection: 'column', fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>
      <span>{label}</span>
      <input
        id={id}
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: '11rem' }}
      />
    </label>
  );
}

type DataT = {
  grid: Float64Array;
  empRisks: Float64Array;
  P: Float64Array;
  Qn: Float64Array;
  Qb: Float64Array;
  erm: { tau: number; risk: number };
  klNarrow: number;
  klBroad: number;
  ehatNarrow: number;
  ehatBroad: number;
};

function DistributionPanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 20, bottom: 36, left: 50 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([-2, 2]).range([0, w]);
      const yMax = Math.max(d3.max(data.Qn) ?? 0.1, d3.max(data.Qb) ?? 0.05, 1 / data.grid.length + 0.005);
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3f')));
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('threshold t');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -40).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('probability mass');

      const barW = w / data.grid.length;
      g.append('g').selectAll('rect').data(Array.from(data.P)).join('rect')
        .attr('x', (_, i) => x(data.grid[i]) - barW / 2)
        .attr('y', (v) => y(v))
        .attr('width', barW * 0.9)
        .attr('height', (v) => h - y(v))
        .style('fill', 'var(--color-border)');

      const lineGen = d3.line<number>().x((_, i) => x(data.grid[i])).y((v) => y(v));
      g.append('path').datum(Array.from(data.Qn))
        .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.4)
        .attr('d', lineGen);
      g.append('path').datum(Array.from(data.Qb))
        .style('fill', 'none').style('stroke', '#534AB7').style('stroke-width', 2.4)
        .attr('d', lineGen);

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Panel A — prior P and two posteriors');

      const legend = g.append('g').attr('transform', `translate(${w - 96},14)`);
      legend.append('rect').attr('x', 0).attr('y', -6).attr('width', 12).attr('height', 8).style('fill', 'var(--color-border)');
      legend.append('text').attr('x', 16).attr('y', 2).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('P (uniform)');
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 14).attr('y2', 14)
        .style('stroke', 'var(--color-accent)').style('stroke-width', 2.4);
      legend.append('text').attr('x', 18).attr('y', 17).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('Q_narrow');
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 28).attr('y2', 28)
        .style('stroke', '#534AB7').style('stroke-width', 2.4);
      legend.append('text').attr('x', 18).attr('y', 31).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('Q_broad');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

function RiskWithExpectationPanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 20, bottom: 36, left: 44 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([-2, 2]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 0.6]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('threshold t');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -34).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('empirical risk');

      const lineGen = d3.line<number>().x((_, i) => x(data.grid[i])).y((v) => y(v));
      g.append('path').datum(Array.from(data.empRisks))
        .style('fill', 'none').style('stroke', '#D97706').style('stroke-width', 2)
        .attr('d', lineGen);

      // E_Q[Rhat] horizontal markers
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(data.ehatNarrow)).attr('y2', y(data.ehatNarrow))
        .style('stroke', 'var(--color-accent)').style('stroke-dasharray', '4,3').style('stroke-width', 1.5);
      g.append('text').attr('x', w - 4).attr('y', y(data.ehatNarrow) - 3).attr('text-anchor', 'end')
        .style('font-size', '10px').style('fill', 'var(--color-accent)')
        .text(`E_Q_n[R̂] = ${data.ehatNarrow.toFixed(3)}`);

      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(data.ehatBroad)).attr('y2', y(data.ehatBroad))
        .style('stroke', '#534AB7').style('stroke-dasharray', '4,3').style('stroke-width', 1.5);
      g.append('text').attr('x', w - 4).attr('y', y(data.ehatBroad) - 3).attr('text-anchor', 'end')
        .style('font-size', '10px').style('fill', '#534AB7')
        .text(`E_Q_b[R̂] = ${data.ehatBroad.toFixed(3)}`);

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Panel B — empirical risk + E_Q[R̂] markers');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

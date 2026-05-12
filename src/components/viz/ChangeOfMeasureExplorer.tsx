import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  empiricalRiskOnGrid,
  ermOnGrid,
  gaussianOnGridPosterior,
  gibbsOnGrid,
  klDiscrete,
  logExpEnvelope,
  mulberry32,
  sampleNormalThresholdProblem,
  thresholdGrid,
  variationalFunctional,
} from './shared/pac-bayes-bounds';

// =============================================================================
// ChangeOfMeasureExplorer — §3.2 / §3.4
// Panel A: variational functional g(Q_c) = E_Q[f] − KL(Q ∥ P) as the center c
// sweeps over T for a Gaussian-on-grid family of fixed bandwidth σ_Q.  The
// universal envelope log E_P[e^f] (Donsker–Varadhan) is marked as a horizontal
// line — the family's restricted maximum sits strictly below it.
// Panel B: the unrestricted Gibbs maximizer Q* ∝ P · e^f vs the family's best
// Gaussian-on-grid approximator — visualizes the gap.
// =============================================================================

const N = 200;
const ETA = 0.05;
const MU_STAR = 0.3;
const SEED = 20260511;
const HEIGHT = 340;
const SM_BREAKPOINT = 640;

export default function ChangeOfMeasureExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [lambda, setLambda] = useState(40);
  const [sigmaFamily, setSigmaFamily] = useState(0.15);

  const data = useMemo(() => {
    const rng = mulberry32(SEED >>> 0);
    const { X, Y } = sampleNormalThresholdProblem(N, ETA, MU_STAR, rng);
    const grid = thresholdGrid();
    const empRisks = empiricalRiskOnGrid(X, Y, grid);
    const P = new Float64Array(grid.length).fill(1 / grid.length);
    // f(h_t) = −λ R̂_S(h_t) — the Catoni choice that produces the Gibbs at temperature λ
    const f = new Float64Array(grid.length);
    for (let i = 0; i < grid.length; i++) f[i] = -lambda * empRisks[i];

    // Universal envelope (DV)
    const envelope = logExpEnvelope(P, f);

    // Unrestricted Gibbs maximizer Q* ∝ P · e^f
    const Qstar = gibbsOnGrid(P, empRisks, lambda);

    // Sweep family centers across T, build Gaussian-on-grid posterior at each,
    // record g(Q_c) and find argmax c*.
    const gOfC = new Float64Array(grid.length);
    let bestG = -Infinity;
    let bestIdx = 0;
    for (let i = 0; i < grid.length; i++) {
      const Qc = gaussianOnGridPosterior(grid, grid[i], sigmaFamily, P);
      const gv = variationalFunctional(Qc, P, f);
      gOfC[i] = gv;
      if (gv > bestG) {
        bestG = gv;
        bestIdx = i;
      }
    }
    const familyBest = gaussianOnGridPosterior(grid, grid[bestIdx], sigmaFamily, P);

    const erm = ermOnGrid(grid, empRisks);

    return {
      grid,
      P,
      empRisks,
      Qstar,
      familyBest,
      gOfC,
      envelope,
      familyBestG: bestG,
      familyBestC: grid[bestIdx],
      familyBestKL: klDiscrete(familyBest, P),
      starKL: klDiscrete(Qstar, P),
      ermTau: erm.tau,
    };
  }, [lambda, sigmaFamily]);

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
        <Slider id="come-lam" label={`λ (f-coef) = ${lambda}`} value={lambda} min={1} max={200} step={1} onChange={setLambda} />
        <Slider id="come-sig" label={`σ_Q family = ${sigmaFamily.toFixed(2)}`} value={sigmaFamily} min={0.05} max={0.5} step={0.01} onChange={setSigmaFamily} />
        <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          log E_P[e^f] = {data.envelope.toFixed(3)} (envelope) &nbsp;|&nbsp; family best g = {data.familyBestG.toFixed(3)} at c = {data.familyBestC.toFixed(2)}<br />
          gap = {(data.envelope - data.familyBestG).toFixed(3)} nats &nbsp;|&nbsp; KL(Q* ∥ P) = {data.starKL.toFixed(2)}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
        <FunctionalSweepPanel data={data} width={panelWidth} />
        <GibbsVsFamilyPanel data={data} width={panelWidth} />
      </div>
    </div>
  );
}

function Slider({
  id, label, value, min, max, step, onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
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
  P: Float64Array;
  empRisks: Float64Array;
  Qstar: Float64Array;
  familyBest: Float64Array;
  gOfC: Float64Array;
  envelope: number;
  familyBestG: number;
  familyBestC: number;
  familyBestKL: number;
  starKL: number;
  ermTau: number;
};

function FunctionalSweepPanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 20, bottom: 36, left: 50 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([-2, 2]).range([0, w]);
      const yLo = Math.min(d3.min(data.gOfC) ?? 0, data.envelope) - 0.5;
      const yHi = data.envelope + 0.5;
      const y = d3.scaleLinear().domain([yLo, yHi]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('family center c');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -40).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('g(Q_c)');

      // Envelope (universal upper bound)
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(data.envelope)).attr('y2', y(data.envelope))
        .style('stroke', 'var(--color-text)').style('stroke-dasharray', '5,3').style('stroke-width', 1.6);
      g.append('text').attr('x', w - 4).attr('y', y(data.envelope) - 4).attr('text-anchor', 'end')
        .style('font-size', '10px').style('fill', 'var(--color-text)')
        .text(`envelope log E_P[e^f] = ${data.envelope.toFixed(3)}`);

      const lineGen = d3.line<number>().x((_, i) => x(data.grid[i])).y((v) => y(v));
      g.append('path').datum(Array.from(data.gOfC))
        .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.2)
        .attr('d', lineGen);

      // Mark the family's argmax
      g.append('circle')
        .attr('cx', x(data.familyBestC)).attr('cy', y(data.familyBestG)).attr('r', 4)
        .style('fill', 'var(--color-accent)');

      // Gap shading
      g.append('line').attr('x1', x(data.familyBestC)).attr('x2', x(data.familyBestC))
        .attr('y1', y(data.envelope)).attr('y2', y(data.familyBestG))
        .style('stroke', '#D97706').style('stroke-dasharray', '2,2').style('stroke-width', 1.4);

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('§3.2 — family g(Q_c) vs c (envelope above)');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

function GibbsVsFamilyPanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 20, bottom: 36, left: 50 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([-2, 2]).range([0, w]);
      const yMax = Math.max(d3.max(data.Qstar) ?? 0.1, d3.max(data.familyBest) ?? 0.1);
      const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3f')));
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('threshold t');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -40).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('probability mass');

      const lineGen = d3.line<number>().x((_, i) => x(data.grid[i])).y((v) => y(v));
      g.append('path').datum(Array.from(data.Qstar))
        .style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 2.4)
        .attr('d', lineGen);
      g.append('path').datum(Array.from(data.familyBest))
        .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.4)
        .attr('d', lineGen);

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('§3.4 — unrestricted Gibbs Q* vs family best');

      const legend = g.append('g').attr('transform', `translate(${w - 100},14)`);
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 0).attr('y2', 0)
        .style('stroke', 'var(--color-text)').style('stroke-width', 2.4);
      legend.append('text').attr('x', 18).attr('y', 3).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('Q* (Gibbs)');
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 14).attr('y2', 14)
        .style('stroke', 'var(--color-accent)').style('stroke-width', 2.4);
      legend.append('text').attr('x', 18).attr('y', 17).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('best Gaussian-family');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

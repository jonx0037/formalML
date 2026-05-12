import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  catoniHoeffdingSlack,
  tolstikhinSeldinSlack,
} from './shared/pac-bayes-bounds';

// =============================================================================
// BennettVsHoeffding — §7.5
// Hoeffding (Catoni-optimized) vs empirical-Bernstein (Tolstikhin–Seldin) slack
// as p̂ ∈ [0, 0.5] sweeps, at fixed (n, KL, δ, K).  At the running-example's
// p̂ ≈ 0.064 with n=200 / K=10 grid, the ratio is 0.74 — i.e., Hoeffding is
// actually TIGHTER here (the §7.2 large-n claim flips at small n with this K).
// =============================================================================

const HEIGHT = 360;
const P_GRID = 200;

export default function BennettVsHoeffding() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(200);
  const [kl, setKl] = useState(2.2);
  const [k, setK] = useState(10);
  const [delta] = useState(0.05);

  const data = useMemo(() => {
    const pGrid = new Float64Array(P_GRID);
    const hoeffding = new Float64Array(P_GRID);
    const bernstein = new Float64Array(P_GRID);
    for (let i = 0; i < P_GRID; i++) {
      const p = (i / (P_GRID - 1)) * 0.5;
      pGrid[i] = p;
      hoeffding[i] = catoniHoeffdingSlack(kl, n, delta);
      bernstein[i] = tolstikhinSeldinSlack(p, kl, n, delta, k);
    }
    // Reference point at p̂ = 0.064 (Q_narrow regime)
    const pRef = 0.064;
    const hRef = catoniHoeffdingSlack(kl, n, delta);
    const bRef = tolstikhinSeldinSlack(pRef, kl, n, delta, k);
    return { pGrid, hoeffding, bernstein, pRef, hRef, bRef, ratio: bRef / hRef };
  }, [n, kl, k, delta]);

  const hoeffdingTighter = data.ratio > 1;

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
        <Slider id="bvh-n" label={`n = ${n}`} value={n} min={50} max={5000} step={50} onChange={setN} />
        <Slider id="bvh-kl" label={`KL = ${kl.toFixed(2)}`} value={kl} min={0.1} max={6} step={0.1} onChange={setKl} />
        <Slider id="bvh-k" label={`K (union grid) = ${k}`} value={k} min={1} max={30} step={1} onChange={setK} />
        <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          δ = {delta}, at p̂ = 0.064 (Q_narrow regime):<br />
          Hoeffding slack = {data.hRef.toFixed(4)}, Bernstein slack = {data.bRef.toFixed(4)}<br />
          ratio Bernstein/Hoeffding = <strong>{data.ratio.toFixed(2)}</strong> &nbsp;
          ({hoeffdingTighter ? 'Hoeffding tighter — small-n regime' : 'Bernstein tighter — large-n / low-variance regime'})
        </div>
      </div>
      <Panel data={data} width={containerWidth || 800} />
    </div>
  );
}

function Slider({
  id, label, value, min, max, step, onChange,
}: { id: string; label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
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
        style={{ width: '10rem' }}
      />
    </label>
  );
}

type DataT = {
  pGrid: Float64Array;
  hoeffding: Float64Array;
  bernstein: Float64Array;
  pRef: number;
  hRef: number;
  bRef: number;
  ratio: number;
};

function Panel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 130, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([0, 0.5]).range([0, w]);
      const yMax = Math.max(d3.max(data.hoeffding) ?? 0.3, d3.max(data.bernstein) ?? 0.3) * 1.05;
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      // Improvement region (Bernstein under Hoeffding where it wins)
      const minLine = new Float64Array(data.pGrid.length);
      for (let i = 0; i < data.pGrid.length; i++) minLine[i] = Math.min(data.hoeffding[i], data.bernstein[i]);
      const area = d3.area<number>()
        .x((_, i) => x(data.pGrid[i]))
        .y0((_, i) => y(Math.max(data.hoeffding[i], data.bernstein[i])))
        .y1((_, i) => y(Math.min(data.hoeffding[i], data.bernstein[i])));
      g.append('path').datum(Array.from(data.hoeffding))
        .style('fill', 'var(--color-accent)').style('fill-opacity', 0.10)
        .attr('d', area);

      const lineGen = (arr: Float64Array) => d3.line<number>()
        .x((_, i) => x(data.pGrid[i]))
        .y((v) => y(v))(Array.from(arr));

      g.append('path').attr('d', lineGen(data.hoeffding))
        .style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 2.2);
      g.append('path').attr('d', lineGen(data.bernstein))
        .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.2);

      // Reference point at p̂ = 0.064
      g.append('line').attr('x1', x(data.pRef)).attr('x2', x(data.pRef))
        .attr('y1', 0).attr('y2', h)
        .style('stroke', '#D97706').style('stroke-dasharray', '3,3');
      g.append('circle').attr('cx', x(data.pRef)).attr('cy', y(data.hRef)).attr('r', 4)
        .style('fill', 'var(--color-text)');
      g.append('circle').attr('cx', x(data.pRef)).attr('cy', y(data.bRef)).attr('r', 4)
        .style('fill', 'var(--color-accent)');

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('p̂ = E_Q[R̂_S]');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('certificate slack');

      const legend = g.append('g').attr('transform', `translate(${w + 12},14)`);
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 0).attr('y2', 0)
        .style('stroke', 'var(--color-text)').style('stroke-width', 2.2);
      legend.append('text').attr('x', 18).attr('y', 3).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('Hoeffding (Catoni-opt)');
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 16).attr('y2', 16)
        .style('stroke', 'var(--color-accent)').style('stroke-width', 2.2);
      legend.append('text').attr('x', 18).attr('y', 19).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('Bernstein (T-S, K-union)');
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 32).attr('y2', 32)
        .style('stroke', '#D97706').style('stroke-dasharray', '3,3');
      legend.append('text').attr('x', 18).attr('y', 35).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('Q_narrow p̂ = 0.064');

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Variance-adaptive vs worst-case slack — regime depends on (n, K)');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

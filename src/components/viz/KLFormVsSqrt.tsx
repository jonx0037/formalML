import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mcAllesterCertificate, seegerCertificate } from './shared/pac-bayes-bounds';

// =============================================================================
// KLFormVsSqrt — §5.5
// McAllester vs Seeger certificates as p̂ ∈ [0, 0.5] sweeps, at fixed (n, KL,
// δ).  Shaded "Seeger improvement" region between the two curves.
// =============================================================================

const HEIGHT = 360;
const P_GRID = 200;

export default function KLFormVsSqrt() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(500);
  const [kl, setKl] = useState(3.0);
  const [delta] = useState(0.05);

  const data = useMemo(() => {
    const pGrid = new Float64Array(P_GRID);
    const mc = new Float64Array(P_GRID);
    const sg = new Float64Array(P_GRID);
    for (let i = 0; i < P_GRID; i++) {
      const p = (i / (P_GRID - 1)) * 0.5;
      pGrid[i] = p;
      mc[i] = mcAllesterCertificate(p, kl, n, delta);
      sg[i] = seegerCertificate(p, kl, n, delta);
    }
    // Tightening factor at p̂ = 0: Seeger is `seegerTighterBy`× tighter than McAllester
    // (i.e., McAllester certificate is this multiple of Seeger's).  Larger ⇒ Seeger wins
    // by more; we compute as mc/sg so the value is the multiplier directly.
    const seegerTighterBy = sg[0] > 0 ? mc[0] / sg[0] : Number.POSITIVE_INFINITY;
    return { pGrid, mc, sg, seegerTighterBy };
  }, [n, kl, delta]);

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
        <Slider id="klf-n" label={`n = ${n}`} value={n} min={50} max={5000} step={50} onChange={setN} />
        <Slider id="klf-kl" label={`KL = ${kl.toFixed(2)}`} value={kl} min={0.1} max={10} step={0.1} onChange={setKl} />
        <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          δ = {delta}, Seeger vs McAllester at p̂ = 0:&nbsp;
          <strong>{Number.isFinite(data.seegerTighterBy) ? `${data.seegerTighterBy.toFixed(2)}×` : '∞'} tighter</strong>
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
        style={{ width: '11rem' }}
      />
    </label>
  );
}

type DataT = {
  pGrid: Float64Array;
  mc: Float64Array;
  sg: Float64Array;
  seegerTighterBy: number;
};

function Panel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 110, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([0, 0.5]).range([0, w]);
      const yMax = Math.max(d3.max(data.mc) ?? 1, d3.max(data.sg) ?? 1) * 1.02;
      const y = d3.scaleLinear().domain([0, Math.min(1, yMax)]).range([h, 0]);

      // Shaded improvement region (between Seeger and McAllester curves)
      const area = d3.area<number>()
        .x((_, i) => x(data.pGrid[i]))
        .y0((v) => y(v))
        .y1((_, i) => y(data.sg[i]));
      g.append('path').datum(Array.from(data.mc))
        .style('fill', 'var(--color-accent)').style('fill-opacity', 0.12)
        .attr('d', area);

      const lineGen = (arr: Float64Array) => d3.line<number>()
        .x((_, i) => x(data.pGrid[i]))
        .y((v) => y(v))(Array.from(arr));

      g.append('path').attr('d', lineGen(data.mc))
        .style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 2.2);
      g.append('path').attr('d', lineGen(data.sg))
        .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.2);

      // Diagonal y = p̂ floor
      const floorPath = d3.line<number>()
        .x((p) => x(p))
        .y((p) => y(p))([0, 0.5]);
      g.append('path').attr('d', floorPath)
        .style('fill', 'none').style('stroke', 'var(--color-text-secondary)')
        .style('stroke-dasharray', '3,3').style('stroke-width', 1);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('p̂ = E_Q[R̂_S]');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('upper bound on E_Q[R]');

      // Legend
      const legend = g.append('g').attr('transform', `translate(${w + 12},20)`);
      legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 0).attr('y2', 0)
        .style('stroke', 'var(--color-text)').style('stroke-width', 2.2);
      legend.append('text').attr('x', 20).attr('y', 3).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('McAllester');
      legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 16).attr('y2', 16)
        .style('stroke', 'var(--color-accent)').style('stroke-width', 2.2);
      legend.append('text').attr('x', 20).attr('y', 19).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('Seeger');
      legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 32).attr('y2', 32)
        .style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '3,3');
      legend.append('text').attr('x', 20).attr('y', 35).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('y = p̂ floor');
      legend.append('rect').attr('x', 0).attr('y', 44).attr('width', 16).attr('height', 8)
        .style('fill', 'var(--color-accent)').style('fill-opacity', 0.12);
      legend.append('text').attr('x', 20).attr('y', 52).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('improvement');

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Seeger dominates McAllester near the boundary (small p̂)');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

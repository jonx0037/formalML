import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mcAllesterCertificate } from './shared/pac-bayes-bounds';

// =============================================================================
// McAllesterBoundSurface — §4.5
// Heatmap of the McAllester certificate B(Q) = E_Q[R̂] + √((KL + log(2√n/δ))/(2n))
// over (KL, n) at fixed (E_Q[R̂], δ).  Iso-bound contours mark the {0.05, 0.1,
// 0.2, 0.4, 1.0} levels.  The 1.0 isoline traces the vacuousness boundary.
// =============================================================================

const NX = 80;
const NY = 70;
const HEIGHT = 380;
const ISO_LEVELS = [0.05, 0.10, 0.20, 0.40, 1.0];

export default function McAllesterBoundSurface() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [empRisk, setEmpRisk] = useState(0.05);
  const [delta, setDelta] = useState(0.05);

  const data = useMemo(() => {
    const klMin = 0;
    const klMax = 12;
    const nMin = Math.log10(50);
    const nMax = Math.log10(20000);
    const values = new Float64Array(NX * NY);
    for (let j = 0; j < NY; j++) {
      const logN = nMin + (j / (NY - 1)) * (nMax - nMin);
      const n = Math.pow(10, logN);
      for (let i = 0; i < NX; i++) {
        const kl = klMin + (i / (NX - 1)) * (klMax - klMin);
        const b = mcAllesterCertificate(empRisk, kl, n, delta);
        values[j * NX + i] = Math.min(1.2, b);
      }
    }
    return { values, klMin, klMax, nMin, nMax };
  }, [empRisk, delta]);

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
        <Slider id="mab-er" label={`E_Q[R̂] = ${empRisk.toFixed(2)}`} value={empRisk} min={0} max={0.5} step={0.01} onChange={setEmpRisk} />
        <Slider id="mab-d" label={`δ = ${delta.toFixed(3)}`} value={delta} min={0.001} max={0.2} step={0.001} onChange={setDelta} />
        <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          B(Q) = E_Q[R̂] + √((KL + log(2√n/δ)) / (2n)) &nbsp;|&nbsp; iso = {ISO_LEVELS.join(', ')}
        </div>
      </div>
      <SurfacePanel data={data} width={containerWidth || 800} />
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
  values: Float64Array;
  klMin: number;
  klMax: number;
  nMin: number;
  nMax: number;
};

function SurfacePanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 90, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([data.klMin, data.klMax]).range([0, w]);
      const yScale = d3.scaleLinear().domain([data.nMin, data.nMax]).range([h, 0]);

      const cellW = w / NX;
      const cellH = h / NY;
      const color = d3.scaleSequential(d3.interpolateViridis).domain([1.0, 0.0]);

      // Heatmap
      const cells = g.append('g');
      for (let j = 0; j < NY; j++) {
        for (let i = 0; i < NX; i++) {
          cells.append('rect')
            .attr('x', i * cellW)
            .attr('y', (NY - 1 - j) * cellH)
            .attr('width', cellW + 0.5)
            .attr('height', cellH + 0.5)
            .style('fill', color(data.values[j * NX + i]));
        }
      }

      // Iso-contours via d3.contours (umbrella export)
      const contourGen = d3.contours().size([NX, NY]).thresholds(ISO_LEVELS);
      const contoursData = contourGen(Array.from(data.values));
      const path = d3.geoPath().projection(d3.geoIdentity().reflectY(true)
        .scale(Math.min(w / NX, h / NY))
        .translate([0, h]));
      g.append('g').selectAll('path').data(contoursData).join('path')
        .attr('d', path as never)
        .style('fill', 'none')
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', 1.2);

      // Iso labels
      contoursData.forEach((c: d3.ContourMultiPolygon) => {
        g.append('text')
          .attr('x', w - 6)
          .attr('y', yScale(data.nMin + (1 - (c.value === 1 ? 0.95 : c.value * 2)) * (data.nMax - data.nMin)) - 2)
          .attr('text-anchor', 'end')
          .style('font-size', '9px')
          .style('fill', 'var(--color-text-secondary)')
          .text(`B=${c.value.toFixed(2)}`);
      });

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(
        d3.axisLeft(yScale).ticks(5).tickFormat((v) => {
          const n = Math.pow(10, +v);
          return n >= 1000 ? `${(n / 1000).toFixed(0)}k` : `${n.toFixed(0)}`;
        }),
      );
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('KL(Q ∥ P) (nats)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('sample size n (log)');

      // Colorbar (right margin)
      const cbW = 14;
      const cbX = w + 16;
      const cbN = 50;
      const cbSeg = h / cbN;
      const cb = g.append('g');
      for (let k = 0; k < cbN; k++) {
        const v = (k / (cbN - 1)) * 1.0;
        cb.append('rect').attr('x', cbX).attr('y', h - (k + 1) * cbSeg)
          .attr('width', cbW).attr('height', cbSeg + 0.5)
          .style('fill', color(v));
      }
      const cbScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);
      g.append('g').attr('transform', `translate(${cbX + cbW},0)`)
        .call(d3.axisRight(cbScale).ticks(5).tickFormat(d3.format('.1f')));
      g.append('text').attr('x', cbX + cbW + 32).attr('y', h / 2)
        .attr('transform', `rotate(-90 ${cbX + cbW + 32} ${h / 2})`)
        .attr('text-anchor', 'middle').style('font-size', '10px').style('fill', 'var(--color-text-secondary)')
        .text('B(Q)');

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('McAllester certificate surface vs (KL, n) — vacuousness at B = 1');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

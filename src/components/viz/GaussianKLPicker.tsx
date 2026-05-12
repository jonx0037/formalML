import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { gaussianKLIsotropic } from './shared/pac-bayes-bounds';

// =============================================================================
// GaussianKLPicker — §8.5
// Two-panel: KL(Q ∥ P) vs σ_Q (isotropic σ_{Q,i} ≡ σ_Q) at fixed μ, for several
// values of ‖μ‖/σ_P; decomposition of the same KL into mean-shift and variance
// terms at a high-d representative setting (d = 10⁴, ‖μ‖ = 50, σ_P = 1).
// Log-log axes throughout.
// =============================================================================

const HEIGHT = 360;
const SM_BREAKPOINT = 640;
const SIG_GRID = 200;
const MU_RATIOS = [0, 10, 30, 100];

export default function GaussianKLPicker() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [d, setD] = useState(10_000);
  const [sigmaP, setSigmaP] = useState(1.0);

  const data = useMemo(() => {
    const sigLo = Math.log10(0.01);
    const sigHi = Math.log10(10);
    const sigmas = new Float64Array(SIG_GRID);
    for (let i = 0; i < SIG_GRID; i++) {
      sigmas[i] = Math.pow(10, sigLo + (i / (SIG_GRID - 1)) * (sigHi - sigLo));
    }
    const familyCurves = MU_RATIOS.map((ratio) => {
      const muNormSq = (ratio * sigmaP) ** 2;
      const kl = new Float64Array(SIG_GRID);
      for (let i = 0; i < SIG_GRID; i++) {
        kl[i] = gaussianKLIsotropic(muNormSq, sigmas[i] ** 2, sigmaP ** 2, d);
      }
      return { ratio, muNormSq, kl };
    });

    // Decomposition at representative ‖μ‖ = 50, σ_P fixed by user
    const muRefNorm = 50;
    const meanShift = muRefNorm ** 2 / (2 * sigmaP ** 2);
    const variance = new Float64Array(SIG_GRID);
    for (let i = 0; i < SIG_GRID; i++) {
      const rho = (sigmas[i] / sigmaP) ** 2;
      variance[i] = (d / 2) * (rho - 1 - Math.log(rho));
    }
    return { sigmas, familyCurves, meanShift, variance };
  }, [d, sigmaP]);

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
        <Slider id="gkp-d" label={`d = ${d.toLocaleString('en-US')}`} value={Math.log10(d)} min={2} max={6} step={0.05} onChange={(v) => setD(Math.round(Math.pow(10, v)))} />
        <Slider id="gkp-sp" label={`σ_P = ${sigmaP.toFixed(2)}`} value={sigmaP} min={0.1} max={3} step={0.05} onChange={setSigmaP} />
        <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          KL = ‖μ‖²/(2σ²_P) + (d/2)·[ρ − 1 − log ρ] where ρ = σ²_Q/σ²_P
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
        <FamilyPanel data={data} width={panelWidth} />
        <DecompositionPanel data={data} width={panelWidth} />
      </div>
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
  sigmas: Float64Array;
  familyCurves: { ratio: number; muNormSq: number; kl: Float64Array }[];
  meanShift: number;
  variance: Float64Array;
};

function FamilyPanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 90, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([data.sigmas[0], data.sigmas[data.sigmas.length - 1]]).range([0, w]);
      const allMax = Math.max(...data.familyCurves.flatMap((c) => Array.from(c.kl)));
      const allMin = Math.max(1e-2, Math.min(...data.familyCurves.flatMap((c) => Array.from(c.kl).filter((v) => v > 0))));
      const y = d3.scaleLog().domain([allMin, allMax * 1.5]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(6, '~s'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('σ_Q (log)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('KL(Q ∥ P) (log)');

      const palette = ['#0F6E56', '#534AB7', '#D97706', '#1A1A1A'];
      data.familyCurves.forEach((curve, idx) => {
        const lineGen = d3.line<number>()
          .x((_, i) => x(data.sigmas[i]))
          .y((v) => y(Math.max(1e-3, v)));
        g.append('path').datum(Array.from(curve.kl))
          .style('fill', 'none').style('stroke', palette[idx]).style('stroke-width', 2)
          .attr('d', lineGen);
      });

      const legend = g.append('g').attr('transform', `translate(${w + 6},14)`);
      data.familyCurves.forEach((curve, idx) => {
        legend.append('line').attr('x1', 0).attr('x2', 14)
          .attr('y1', idx * 14).attr('y2', idx * 14)
          .style('stroke', palette[idx]).style('stroke-width', 2);
        legend.append('text').attr('x', 18).attr('y', idx * 14 + 3)
          .style('font-size', '10px').style('fill', 'var(--color-text-secondary)')
          .text(`‖μ‖/σ_P = ${curve.ratio}`);
      });

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Panel A — KL vs σ_Q for several ‖μ‖/σ_P');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

function DecompositionPanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 110, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([data.sigmas[0], data.sigmas[data.sigmas.length - 1]]).range([0, w]);
      const total = new Float64Array(data.sigmas.length);
      for (let i = 0; i < data.sigmas.length; i++) total[i] = data.meanShift + data.variance[i];
      const yMax = Math.max(
        data.meanShift,
        Math.max(...Array.from(data.variance)),
        Math.max(...Array.from(total)),
      );
      const yMin = Math.max(1e-2, Math.min(data.meanShift, Math.min(...Array.from(data.variance).filter((v) => v > 0))));
      const y = d3.scaleLog().domain([yMin, yMax * 1.4]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(6, '~s'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('σ_Q (log)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('contribution to KL (log)');

      // Mean-shift (constant in σ_Q)
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(data.meanShift)).attr('y2', y(data.meanShift))
        .style('stroke', '#D97706').style('stroke-width', 2);

      // Variance term
      const lineGen = (arr: Float64Array) => d3.line<number>()
        .x((_, i) => x(data.sigmas[i]))
        .y((v) => y(Math.max(yMin, v)))(Array.from(arr));
      g.append('path').attr('d', lineGen(data.variance))
        .style('fill', 'none').style('stroke', '#534AB7').style('stroke-width', 2);

      // Total
      g.append('path').attr('d', lineGen(total))
        .style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 2.4);

      const legend = g.append('g').attr('transform', `translate(${w + 12},14)`);
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 0).attr('y2', 0)
        .style('stroke', '#D97706').style('stroke-width', 2);
      legend.append('text').attr('x', 18).attr('y', 3).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('mean-shift ‖μ‖²/(2σ²_P)');
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 16).attr('y2', 16)
        .style('stroke', '#534AB7').style('stroke-width', 2);
      legend.append('text').attr('x', 18).attr('y', 19).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('variance (d/2)·[ρ−1−log ρ]');
      legend.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 32).attr('y2', 32)
        .style('stroke', 'var(--color-text)').style('stroke-width', 2.4);
      legend.append('text').attr('x', 18).attr('y', 35).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text('total KL');

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Panel B — decomposition at ‖μ‖ = 50, current d / σ_P');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

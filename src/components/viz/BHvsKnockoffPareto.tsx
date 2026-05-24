import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32 } from './shared/normalizing-flows';
import { gaussianFrom } from './shared/generalization-bounds';

// Locked notebook outputs from cell 54 (§12.5 GWAS-like comparison, B = 20 replicates).
type Proc = { name: string; color: string; meanFdp: number; sdFdp: number; meanPower: number; nRej: number };
const PROCS: Proc[] = [
  { name: 'Bonferroni', color: '#737373', meanFdp: 0.9263, sdFdp: 0.0044, meanPower: 1.0, nRej: 136.2 },
  { name: 'BH-debiased', color: '#1d4ed8', meanFdp: 0.0091, sdFdp: 0.028, meanPower: 1.0, nRej: 10.1 },
  { name: 'model-X knockoff+', color: '#92400e', meanFdp: 0.0973, sdFdp: 0.1067, meanPower: 1.0, nRej: 11.25 },
];

const B = 20;
const NOMINAL_Q = 0.1;

function generateCloud(seed: number): { proc: string; fdp: number; power: number; color: string }[] {
  const rng = mulberry32(seed);
  const gauss = gaussianFrom(rng);
  const points: { proc: string; fdp: number; power: number; color: string }[] = [];
  for (const p of PROCS) {
    for (let i = 0; i < B; i++) {
      const fdp = Math.max(0, Math.min(1, p.meanFdp + p.sdFdp * gauss()));
      // power has no recorded sd; on this strong-signal toy all three saturate at 1.0
      const power = Math.max(0, Math.min(1, p.meanPower - Math.abs(gauss()) * 0.005));
      points.push({ proc: p.name, fdp, power, color: p.color });
    }
  }
  return points;
}

export default function BHvsKnockoffPareto() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [seed, setSeed] = useState(7);

  const points = useMemo(() => generateCloud(seed), [seed]);

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const H = Math.round(W * 0.62);
  const Mg = { t: 22, r: 22, b: 44, l: 56 };

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = W - Mg.l - Mg.r;
      const ih = H - Mg.t - Mg.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${Mg.l},${Mg.t})`);
      const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
      const y = d3.scaleLinear().domain([0.94, 1.005]).range([ih, 0]);

      // q = 0.10 reference
      g.append('line').attr('x1', x(NOMINAL_Q)).attr('x2', x(NOMINAL_Q)).attr('y1', 0).attr('y2', ih).style('stroke', 'var(--color-text)').attr('stroke-dasharray', '4 3').attr('opacity', 0.45);
      g.append('text').attr('x', x(NOMINAL_Q) + 4).attr('y', 12).style('fill', 'var(--color-text-secondary)').style('font-size', '0.68rem').text('nominal q = 0.10');

      // per-replicate cloud
      g.selectAll('circle.repl')
        .data(points)
        .enter()
        .append('circle')
        .attr('cx', (d) => x(d.fdp))
        .attr('cy', (d) => y(d.power))
        .attr('r', 2)
        .attr('fill', (d) => d.color)
        .attr('opacity', 0.45);

      // mean markers
      for (const p of PROCS) {
        g.append('circle').attr('cx', x(p.meanFdp)).attr('cy', y(p.meanPower)).attr('r', 7).attr('fill', p.color).attr('stroke', 'white').attr('stroke-width', 1.5);
      }

      // axes
      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', iw / 2).attr('y', ih + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('empirical FDP');
      g.append('text').attr('transform', `translate(-44,${ih / 2}) rotate(-90)`).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.72rem').text('empirical power');
    },
    [W, H, points],
  );

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-wrap items-end gap-3 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
        <label className="flex flex-col">
          <span>cloud seed = <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>{seed}</span></span>
          <input type="range" min={0} max={200} step={1} value={seed} onChange={(e) => setSeed(parseInt(e.target.value, 10))} aria-label="Seed for the simulated replicate cloud" className="w-56" />
        </label>
        <div className="flex flex-wrap gap-3 text-xs">
          {PROCS.map((p) => (
            <span key={p.name}>
              <span style={{ color: p.color }}>●</span> {p.name}: E[FDP] = <span className="tabular-nums">{p.meanFdp.toFixed(3)}</span> · E[#rej] = <span className="tabular-nums">{p.nRej.toFixed(2)}</span>
            </span>
          ))}
        </div>
      </div>
      <svg ref={svgRef} width={W} height={H} className="mt-2" role="img" aria-label="FDP vs power Pareto scatter for three procedures" />
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        Three procedures on a GWAS-like toy (n = 500, p = 200, AR(1) ρ = 0.4, β★ = 2.0, 10 signals, B = 20). Per-replicate
        points are simulated from the locked notebook means and standard deviations (cell 54). All three procedures hit
        power 1.000 on this strong-signal regime; the headline is the FDP catastrophe of Bonferroni at FDP ≈ 0.93 — 136 of
        146 "rejections" are false discoveries from LD-correlated nulls. BH-debiased (FDP ≈ 0.009) and model-X knockoff+
        (FDP ≈ 0.097, near the nominal q = 0.10) both make ≈10 honest discoveries per replicate.
      </p>
    </div>
  );
}

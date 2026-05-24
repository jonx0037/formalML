import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { normalCdf } from './shared/pac-bayes-bounds';
import { mulberry32 } from './shared/normalizing-flows';
import { gaussianFrom } from './shared/generalization-bounds';

const M = 1000;
const MU_ALT = 3;
const Q = 0.1;
const STOREY_LAMBDA = 0.5;
const SEED = 42;

type ProcedureResult = {
  pvals: Float64Array;
  isNull: Uint8Array;
  rBh: number;
  rStorey: number;
  rBonf: number;
  fdpBh: number;
  fdpStorey: number;
  fdpBonf: number;
  piHat: number;
};

function generate(pi0: number): ProcedureResult {
  const rng = mulberry32(SEED);
  const gauss = gaussianFrom(rng);
  const m0 = Math.round(pi0 * M);
  const pvals = new Float64Array(M);
  const isNull = new Uint8Array(M);
  for (let i = 0; i < M; i++) {
    if (i < m0) {
      pvals[i] = rng();
      isNull[i] = 1;
    } else {
      const z = MU_ALT + gauss();
      pvals[i] = 2 * (1 - normalCdf(Math.abs(z)));
      isNull[i] = 0;
    }
  }
  // sort by p-value, keeping null label
  const idx = Array.from({ length: M }, (_, i) => i);
  idx.sort((a, b) => pvals[a] - pvals[b]);
  const sortedP = new Float64Array(M);
  const sortedNull = new Uint8Array(M);
  for (let i = 0; i < M; i++) {
    sortedP[i] = pvals[idx[i]];
    sortedNull[i] = isNull[idx[i]];
  }

  // BH
  let rBh = 0;
  for (let k = M; k >= 1; k--) {
    if (sortedP[k - 1] <= (k * Q) / M) {
      rBh = k;
      break;
    }
  }
  // Storey
  let above = 0;
  for (let i = 0; i < M; i++) if (sortedP[i] > STOREY_LAMBDA) above++;
  const piHat = (1 + above) / (M * (1 - STOREY_LAMBDA));
  const qStorey = Q / Math.min(1, piHat);
  let rStorey = 0;
  for (let k = M; k >= 1; k--) {
    if (sortedP[k - 1] <= (k * qStorey) / M) {
      rStorey = k;
      break;
    }
  }
  // Bonferroni
  const bonfT = Q / M;
  let rBonf = 0;
  for (let i = 0; i < M; i++) if (sortedP[i] <= bonfT) rBonf++;

  let vBh = 0;
  for (let i = 0; i < rBh; i++) if (sortedNull[i]) vBh++;
  let vStorey = 0;
  for (let i = 0; i < rStorey; i++) if (sortedNull[i]) vStorey++;
  let vBonf = 0;
  for (let i = 0; i < rBonf; i++) if (sortedNull[i]) vBonf++;
  const fdpBh = rBh > 0 ? vBh / rBh : 0;
  const fdpStorey = rStorey > 0 ? vStorey / rStorey : 0;
  const fdpBonf = rBonf > 0 ? vBonf / rBonf : 0;

  return { pvals: sortedP, isNull: sortedNull, rBh, rStorey, rBonf, fdpBh, fdpStorey, fdpBonf, piHat };
}

export default function BHFDRReview() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [pi0Display, setPi0Display] = useState(0.85);
  const [pi0Committed, setPi0Committed] = useState(0.85);

  const result = useMemo(() => generate(pi0Committed), [pi0Committed]);

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const H = Math.round(W * 0.5);
  const Mg = { t: 20, r: 18, b: 38, l: 56 };

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const iw = W - Mg.l - Mg.r;
      const ih = H - Mg.t - Mg.b;
      if (iw <= 0 || ih <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${Mg.l},${Mg.t})`);

      const RANK_MAX = Math.max(result.rBh, result.rStorey, result.rBonf, 50) * 2.4;
      const x = d3.scaleLinear().domain([1, RANK_MAX]).range([0, iw]);
      const y = d3.scaleLog().domain([1e-5, 1]).clamp(true).range([ih, 0]);

      // p-values
      const ranks = d3.range(0, Math.min(RANK_MAX, M));
      g.append('g')
        .selectAll('circle')
        .data(ranks)
        .enter()
        .append('circle')
        .attr('cx', (k) => x(k + 1))
        .attr('cy', (k) => y(Math.max(result.pvals[k], 1e-6)))
        .attr('r', 1.6)
        .attr('fill', (k) => (result.isNull[k] ? 'var(--color-text-secondary)' : 'var(--color-accent)'))
        .attr('opacity', 0.7);

      // BH line kq/m
      const bhLine = d3
        .line<number>()
        .x((k) => x(k))
        .y((k) => y(Math.max((k * Q) / M, 1e-6)));
      g.append('path').datum(d3.range(1, RANK_MAX + 1) as unknown as number[]).attr('d', bhLine).attr('fill', 'none').style('stroke', '#2e8b57').attr('stroke-width', 1.6);

      // Storey line
      const qStorey = Q / Math.min(1, result.piHat);
      const storeyLine = d3
        .line<number>()
        .x((k) => x(k))
        .y((k) => y(Math.max((k * qStorey) / M, 1e-6)));
      g.append('path').datum(d3.range(1, RANK_MAX + 1) as unknown as number[]).attr('d', storeyLine).attr('fill', 'none').style('stroke', '#1d4ed8').attr('stroke-width', 1.6);

      // Bonferroni horizontal
      g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(Q / M)).attr('y2', y(Q / M)).style('stroke', '#dc2626').attr('stroke-width', 1.4).attr('stroke-dasharray', '3 3');

      // rejection markers
      const procs = [
        { name: 'Bonferroni', r: result.rBonf, fdp: result.fdpBonf, color: '#dc2626' },
        { name: 'BH', r: result.rBh, fdp: result.fdpBh, color: '#2e8b57' },
        { name: 'Storey', r: result.rStorey, fdp: result.fdpStorey, color: '#1d4ed8' },
      ];
      for (const p of procs) {
        if (p.r > 0) {
          g.append('line').attr('x1', x(p.r)).attr('x2', x(p.r)).attr('y1', y(1e-5)).attr('y2', y(1)).style('stroke', p.color).attr('stroke-dasharray', '2 4').attr('opacity', 0.6);
        }
      }

      g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', iw / 2).attr('y', ih + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.7rem').text('rank k');
      g.append('text').attr('transform', `translate(-44,${ih / 2}) rotate(-90)`).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '0.7rem').text('p-value (log)');
    },
    [W, H, result],
  );

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>
            π₀ = <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>{pi0Display.toFixed(2)}</span>{' '}
            <span style={{ color: 'var(--color-text-secondary)' }}>(true null fraction)</span>
          </span>
          <input
            type="range"
            min={0.5}
            max={0.99}
            step={0.01}
            value={pi0Display}
            onChange={(e) => setPi0Display(parseFloat(e.target.value))}
            onMouseUp={() => setPi0Committed(pi0Display)}
            onTouchEnd={() => setPi0Committed(pi0Display)}
            onKeyUp={() => setPi0Committed(pi0Display)}
            aria-label="Proportion of true nulls"
            className="w-56"
          />
        </label>
        <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
          Storey π̂₀ ={' '}
          <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>
            {result.piHat.toFixed(3)}
          </span>
        </div>
      </div>
      <svg ref={svgRef} width={W} height={H} className="mt-2" role="img" aria-label="Sorted p-values with BH and Storey thresholds" />
      <div className="mt-1 grid grid-cols-3 gap-2 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        <span><span style={{ color: '#dc2626' }}>■</span> Bonferroni: {result.rBonf} rej · FDP {result.fdpBonf.toFixed(3)}</span>
        <span><span style={{ color: '#2e8b57' }}>■</span> BH: {result.rBh} rej · FDP {result.fdpBh.toFixed(3)}</span>
        <span><span style={{ color: '#1d4ed8' }}>■</span> Storey: {result.rStorey} rej · FDP {result.fdpStorey.toFixed(3)}</span>
      </div>
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        m = 1000 Gaussian-z tests, μ_alt = 3, q = 0.10. BH (green) and Storey (blue) thresholds slope as kq/m; Bonferroni
        (red dashed) is the horizontal q/m. Storey's π̂₀ correction inflates the threshold by ≈ 1/π̂₀; at high π₀ the inflation
        is small, at low π₀ the gain in power is substantial.
      </p>
    </div>
  );
}

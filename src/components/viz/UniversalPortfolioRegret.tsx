import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, binaryKL } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 9.2 — Universal-portfolio regret.
//
// For each μ_1 in a small set, plot
//   regret_n = log E_n(μ_1*) − log E_n^univ
// where E_n(μ_1*) is the Kelly LR (the oracle) and E_n^univ is the mixture
// over a μ-grid. Theorem 9.5 says regret = O(log n).
// -----------------------------------------------------------------------------

const MU0 = 0.5;
const MARGIN = { top: 26, right: 14, bottom: 38, left: 56 };
const PANEL_H = 320;
const N_MAX = 5000;
const MU_GRID_SIZE = 21;

function clip(x: number, m: number): number {
  return Math.max(-m, Math.min(m, x));
}

interface Sim {
  ng: number[];
  regretByMu: { mu1: number; regret: number[] }[];
}

function simulate(): Sim {
  const mu1Values = [0.55, 0.65, 0.80];
  const rng = mulberry32(20261501);
  // Generate one stream per μ_1, then compute both Kelly (oracle) and universal.
  const ng = Array.from({ length: N_MAX }, (_, i) => i + 1);
  const regretByMu: { mu1: number; regret: number[] }[] = [];

  // μ-grid for universal mixture
  const muGrid: number[] = [];
  for (let i = 0; i < MU_GRID_SIZE; i++) muGrid.push(0.01 + 0.98 * (i / (MU_GRID_SIZE - 1)));
  const lambdaGrid = muGrid.map((m) => clip((m - MU0) / (MU0 * (1 - MU0)), 1.0));

  for (const mu1 of mu1Values) {
    const lambdaKelly = clip((mu1 - MU0) / (MU0 * (1 - MU0)), 1.0);
    let logKelly = 0;
    const logUniv = new Float64Array(MU_GRID_SIZE);
    const regret = new Float64Array(N_MAX);
    for (let n = 1; n <= N_MAX; n++) {
      const x = rng() < mu1 ? 1 : 0;
      logKelly += Math.log1p(lambdaKelly * (x - MU0));
      for (let i = 0; i < MU_GRID_SIZE; i++) {
        logUniv[i] += Math.log1p(lambdaGrid[i] * (x - MU0));
      }
      // Universal: log(mean over grid of exp(log E_i)) = logsumexp − log(grid)
      let m = -Infinity;
      for (let i = 0; i < MU_GRID_SIZE; i++) if (logUniv[i] > m) m = logUniv[i];
      let s = 0;
      for (let i = 0; i < MU_GRID_SIZE; i++) s += Math.exp(logUniv[i] - m);
      const logUnivCombined = m + Math.log(s / MU_GRID_SIZE);
      regret[n - 1] = logKelly - logUnivCombined;
    }
    regretByMu.push({ mu1, regret: Array.from(regret) });
  }

  return { ng, regretByMu };
}

const COLORS = ['#356fa3', '#0f6e56', '#c44545'];

export default function UniversalPortfolioRegret() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [delta, setDelta] = useState(0.05); // unused on plot but exposed

  const sim = useMemo(() => simulate(), []);
  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([1, N_MAX]).range([0, innerW]);
      const allY = sim.regretByMu.flatMap((s) => s.regret);
      const yMax = Math.max(...allY);
      const yScale = d3.scaleLinear().domain([0, yMax * 1.05]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      // Reference log n curve normalized to the median series
      const medianMaxRegret = sim.regretByMu[1].regret[sim.regretByMu[1].regret.length - 1];
      const refScale = medianMaxRegret / Math.log(N_MAX);
      const logRef = sim.ng.map((n) => refScale * Math.log(n));
      g.append('path').datum(logRef).attr('d', d3.line<number>().x((_, i) => xScale(i + 1)).y((v) => yScale(v)) as never).style('fill', 'none').style('stroke', 'var(--color-text-secondary)').attr('stroke-width', 1).attr('stroke-dasharray', '4 3');

      sim.regretByMu.forEach((s, i) => {
        const line = d3.line<number>().x((_, idx) => xScale(idx + 1)).y((v) => yScale(v));
        g.append('path').datum(s.regret).attr('d', line as never).style('fill', 'none').style('stroke', COLORS[i]).attr('stroke-width', 2);
        g.append('text').attr('x', innerW - 4).attr('y', 12 + i * 14).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLORS[i]).text(`μ₁ = ${s.mu1}, KL = ${binaryKL(s.mu1, MU0).toFixed(3)}`);
      });
      g.append('text').attr('x', innerW - 4).attr('y', 12 + sim.regretByMu.length * 14).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('~ log n reference');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('n (log scale)');
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('regret = log E_oracle − log E_univ');
    },
    [sim, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', maxWidth: '600px' }}>
          Universal mixture over 21-point μ-grid in [0.01, 0.99]. Regret vs the oracle Kelly LR grows like log n (Theorem 9.5) — the cost of not knowing the true alternative.
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

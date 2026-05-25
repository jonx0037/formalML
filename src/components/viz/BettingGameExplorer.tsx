import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, binaryKL } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 6.1 — The betting game.
//
// Two-panel: top shows log K_n trajectories under H_0 and H_1; bottom shows
// empirical sup-crossing probability vs n.
//
// Sliders: μ_1 (alternative mean), bet strategy (fixed λ / Kelly-optimal /
// empirical Kelly).
// -----------------------------------------------------------------------------

const MU0 = 0.5;
const N_MAX = 1500;
const B_REPLICATES = 800;
const N_PATHS_SHOWN = 30;
const TOP_H = 280;
const BOT_H = 220;
const MARGIN = { top: 26, right: 14, bottom: 36, left: 56 };

const COLOR_H0 = '#356fa3';
const COLOR_H1 = '#c44545';
const COLOR_THRESH = 'var(--color-accent)';

type Strategy = 'fixed' | 'kelly' | 'empirical';

function exactKelly(mu1: number): number {
  // exact Kelly for Bernoulli(μ_1) vs Bernoulli(μ_0):
  // λ* = (μ_1 − μ_0)/[μ_0 (1 − μ_0)]. Equivalent to Kelly bet fraction.
  return (mu1 - MU0) / (MU0 * (1 - MU0));
}

function clipLambda(lambda: number): number {
  // |λ| < 1/max(μ_0, 1 − μ_0) = 1/0.5 = 2. Use 0.5 safety factor.
  return Math.max(-1, Math.min(1, lambda));
}

interface Sim {
  pathsH0: Float64Array[];
  pathsH1: Float64Array[];
  crossingH0: Float64Array;
  crossingH1: Float64Array;
  empGrowthH1: number;
  theoryGrowth: number;
  lambdaUsed: number;
}

function simulate(mu1: number, strategy: Strategy, delta: number): Sim {
  const lambdaKelly = clipLambda(exactKelly(mu1));
  let lambdaFixed: number;
  if (strategy === 'fixed') lambdaFixed = 1.0; // over-aggressive demonstration
  else lambdaFixed = lambdaKelly;
  const logThr = Math.log(1 / delta);

  const rng0 = mulberry32(20261001);
  const rng1 = mulberry32(20261002);

  const pathsH0: Float64Array[] = [];
  const pathsH1: Float64Array[] = [];
  const crossingH0 = new Float64Array(N_MAX);
  const crossingH1 = new Float64Array(N_MAX);
  const crossedH0 = new Uint8Array(B_REPLICATES);
  const crossedH1 = new Uint8Array(B_REPLICATES);

  let sumLogKAtNmax = 0;
  let countH1 = 0;

  for (let b = 0; b < B_REPLICATES; b++) {
    const keep = b < N_PATHS_SHOWN;
    const p0 = keep ? new Float64Array(N_MAX) : null;
    const p1 = keep ? new Float64Array(N_MAX) : null;
    let logK0 = 0;
    let logK1 = 0;
    // For empirical Kelly: running mean and variance under H_1 (need only one stream)
    let runningSum1 = 0;
    let runningSumSq1 = 0;
    for (let n = 1; n <= N_MAX; n++) {
      const x0 = rng0() < MU0 ? 1 : 0;
      const x1 = rng1() < mu1 ? 1 : 0;
      let lambda1: number;
      if (strategy === 'empirical' && n > 1) {
        const meanPrev = runningSum1 / (n - 1);
        const varPrev = Math.max(runningSumSq1 / (n - 1) - meanPrev * meanPrev, 1e-4);
        lambda1 = clipLambda((meanPrev - MU0) / varPrev);
      } else {
        lambda1 = lambdaFixed;
      }
      // Apply same λ (fixed or kelly) to H_0 trajectory; empirical Kelly uses H_1 stream's
      // own running stats — we use a deterministic λ_fixed under H_0 to keep H_0 the null.
      const lambda0 = strategy === 'empirical' ? clipLambda(exactKelly(mu1)) : lambdaFixed;
      logK0 += Math.log1p(lambda0 * (x0 - MU0));
      logK1 += Math.log1p(lambda1 * (x1 - MU0));
      runningSum1 += x1;
      runningSumSq1 += x1 * x1;
      if (p0) p0[n - 1] = logK0;
      if (p1) p1[n - 1] = logK1;
      if (!crossedH0[b] && logK0 >= logThr) crossedH0[b] = 1;
      if (!crossedH1[b] && logK1 >= logThr) crossedH1[b] = 1;
      crossingH0[n - 1] += crossedH0[b];
      crossingH1[n - 1] += crossedH1[b];
    }
    sumLogKAtNmax += logK1;
    countH1++;
    if (p0) pathsH0.push(p0);
    if (p1) pathsH1.push(p1);
  }
  for (let n = 0; n < N_MAX; n++) {
    crossingH0[n] /= B_REPLICATES;
    crossingH1[n] /= B_REPLICATES;
  }
  const empGrowthH1 = sumLogKAtNmax / countH1 / N_MAX;
  const theoryGrowth = binaryKL(mu1, MU0);
  return { pathsH0, pathsH1, crossingH0, crossingH1, empGrowthH1, theoryGrowth, lambdaUsed: lambdaFixed };
}

export default function BettingGameExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayMu, setDisplayMu] = useState(0.6);
  const [committedMu, setCommittedMu] = useState(0.6);
  const [strategy, setStrategy] = useState<Strategy>('kelly');
  const [delta, setDelta] = useState(0.05);

  const sim = useMemo(() => simulate(committedMu, strategy, delta), [committedMu, strategy, delta]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));
  const logThr = Math.log(1 / delta);

  const topRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = TOP_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xScale = d3.scaleLog().domain([1, N_MAX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-3, Math.max(logThr + 4, 10)]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(4, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(logThr)).attr('y2', yScale(logThr)).style('stroke', COLOR_THRESH).attr('stroke-width', 1.4).attr('stroke-dasharray', '6 3');
      g.append('text').attr('x', innerW - 4).attr('y', yScale(logThr) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_THRESH).text(`log(1/δ) = ${logThr.toFixed(2)}`);

      const line = d3.line<number>().x((_, i) => xScale(i + 1)).y((v) => yScale(v));
      for (const p of sim.pathsH0) {
        g.append('path').datum(Array.from(p)).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_H0).attr('stroke-width', 0.6).attr('opacity', 0.5);
      }
      for (const p of sim.pathsH1) {
        g.append('path').datum(Array.from(p)).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_H1).attr('stroke-width', 0.6).attr('opacity', 0.5);
      }

      g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text(`log K_n — 30 H_0: Bern(${MU0}), 30 H_1: Bern(${committedMu})`);
    },
    [sim, w, delta],
  );

  const botRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = BOT_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xScale = d3.scaleLog().domain([1, N_MAX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(4, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).tickFormat(d3.format('.0%'))).selectAll('text').style('fill', 'var(--color-text)');

      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(delta)).attr('y2', yScale(delta)).style('stroke', COLOR_THRESH).attr('stroke-width', 1).attr('stroke-dasharray', '4 3');

      const line = d3.line<number>().x((_, i) => xScale(i + 1)).y((v) => yScale(v));
      g.append('path').datum(Array.from(sim.crossingH0)).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_H0).attr('stroke-width', 1.8);
      g.append('path').datum(Array.from(sim.crossingH1)).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_H1).attr('stroke-width', 1.8);

      g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text(`Empirical P(sup K_m ≥ 1/δ) — H_0 stays ≤ δ, H_1 → 1`);
    },
    [sim, w, delta],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', padding: '10px 0', fontSize: '13px' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>Alt mean μ₁ = <strong>{displayMu.toFixed(2)}</strong> (μ₀ = 0.5)</span>
          <input
            type="range"
            min={0.52}
            max={0.95}
            step={0.01}
            value={displayMu}
            aria-label="Alternative mean"
            onChange={(e) => setDisplayMu(Number(e.target.value))}
            onMouseUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '220px' }}>
          <span>Bet strategy</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            {(['fixed', 'kelly', 'empirical'] as Strategy[]).map((s) => (
              <label key={s} style={{ fontSize: '12px', cursor: 'pointer' }}>
                <input type="radio" checked={strategy === s} onChange={() => setStrategy(s)} /> {s === 'fixed' ? 'λ = 1 (over-bet)' : s === 'kelly' ? 'Kelly λ★' : 'empirical Kelly'}
              </label>
            ))}
          </div>
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '120px' }}>
          <span>δ = <strong>{delta.toFixed(3)}</strong></span>
          <input type="range" min={0.01} max={0.10} step={0.01} value={delta} aria-label="Type-I error" onChange={(e) => setDelta(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', minWidth: '280px' }}>
          Theoretical Kelly growth = kl(μ₁ ‖ μ₀) = <strong>{sim.theoryGrowth.toFixed(4)}</strong>.
          {' '}Empirical Ĝ ≈ <strong style={{ color: sim.empGrowthH1 < 0 ? COLOR_H1 : COLOR_H0 }}>{sim.empGrowthH1.toFixed(4)}</strong>{' '}per step.
          {sim.empGrowthH1 < 0 && ' (wealth shrinks — over-bet)'}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <svg ref={topRef} width={w} height={TOP_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={botRef} width={w} height={BOT_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}

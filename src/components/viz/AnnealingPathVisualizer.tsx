// =============================================================================
// AnnealingPathVisualizer.tsx
//
// §7 of sequential-monte-carlo. Closed-form exploration of the four
// hand-tuned annealing schedules — linear, power-law (k=2), power-law
// (k=0.5), log-uniform — across path length T and a "posterior
// concentration" parameter C that models how var_{π_t}[log p(y|θ)] grows
// with the data size.
//
// The chi-squared between adjacent targets along a geometric annealing
// path satisfies (brief §7.2 Proposition 3):
//   χ²(π_{t+1} ∥ π_t) ≈ (β_{t+1} - β_t)² · Var_{π_t}[log p(y|θ)].
// Modeling Var_{π_t}[log p(y|θ)] = C · g(β_t) with g(β) = max(β, ε)^{-1} (so
// the variance is largest near the prior end of the path, which is the SMC
// pathology adaptive schedules fix), the predicted ESS at step t+1 is
//   ESS_{t+1} ≈ N / (1 + χ²_t).
//
// Pure TS, no MC sampling. Slider changes update both panels instantly;
// no commit-on-release needed.
//
// Panels:
//   (A) β_t vs t/T for each schedule, with chi-squared bars per step.
//   (B) Predicted ESS per step, with the τ N resampling threshold marked.
//
// Static fallback: /images/topics/sequential-monte-carlo/07_smc_sampler.png
// =============================================================================

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteSMC } from './shared/sequential-monte-carlo';

type ScheduleKey = 'linear' | 'pow-fast' | 'pow-slow' | 'log-uniform';
const SCHEDULES: ScheduleKey[] = ['linear', 'pow-fast', 'pow-slow', 'log-uniform'];
const SCHEDULE_LABELS: Record<ScheduleKey, string> = {
  linear: 'linear (β = t/T)',
  'pow-fast': 'power-2 (front-loaded)',
  'pow-slow': 'power-½ (back-loaded)',
  'log-uniform': 'log-uniform',
};
const SCHEDULE_COLORS: Record<ScheduleKey, string> = {
  linear: paletteSMC.muted,
  'pow-fast': paletteSMC.accent,
  'pow-slow': paletteSMC.cloud,
  'log-uniform': paletteSMC.target,
};

const N_REF = 500;

function schedule(kind: ScheduleKey, T: number): number[] {
  const out = new Array<number>(T + 1);
  for (let t = 0; t <= T; t++) {
    const u = t / T;
    if (kind === 'linear') out[t] = u;
    else if (kind === 'pow-fast') out[t] = u * u;
    else if (kind === 'pow-slow') out[t] = Math.sqrt(u);
    else {
      // log-uniform between epsilon and 1
      const eps = 1e-3;
      out[t] = t === 0 ? 0 : eps * Math.pow(1 / eps, u);
    }
  }
  // ensure boundary anchors
  out[0] = 0;
  out[T] = 1;
  return out;
}

/** Modeled Var_{π_t}[log p(y|θ)] = C / max(β, β_floor).
 * High variance near the prior (β small) is the SMC pathology that
 * adaptive schedules manage by taking small Δβ steps there. */
function modeledVariance(beta: number, C: number): number {
  const floor = 1e-3;
  return C / Math.max(beta, floor);
}

/** Predicted ESS at step t+1 given current weights ~ uniform and the
 * incremental chi-squared at the geometric-path step. */
function predictedEss(betaT: number, betaNext: number, C: number, N: number): number {
  const dBeta = betaNext - betaT;
  // Use the variance at the midpoint of the step for slightly tighter prediction.
  const v = modeledVariance(0.5 * (betaT + betaNext), C);
  const chiSq = dBeta * dBeta * v;
  return N / (1 + chiSq);
}

interface Result {
  schedules: Record<ScheduleKey, number[]>; // [T+1]
  dBeta: Record<ScheduleKey, number[]>; // [T]
  chiSq: Record<ScheduleKey, number[]>; // [T]
  ess: Record<ScheduleKey, number[]>; // [T+1]
}

function compute(T: number, C: number): Result {
  const out = { schedules: {}, dBeta: {}, chiSq: {}, ess: {} } as Result;
  for (const k of SCHEDULES) {
    const beta = schedule(k, T);
    const d = new Array<number>(T);
    const chi = new Array<number>(T);
    const ess = new Array<number>(T + 1).fill(N_REF);
    for (let t = 0; t < T; t++) {
      d[t] = beta[t + 1] - beta[t];
      const v = modeledVariance(0.5 * (beta[t] + beta[t + 1]), C);
      chi[t] = d[t] * d[t] * v;
      ess[t + 1] = predictedEss(beta[t], beta[t + 1], C, N_REF);
    }
    out.schedules[k] = beta;
    out.dBeta[k] = d;
    out.chiSq[k] = chi;
    out.ess[k] = ess;
  }
  return out;
}

export default function AnnealingPathVisualizer() {
  const [T, setT] = useState(20);
  const [C, setC] = useState(50);

  const result = useMemo(() => compute(T, C), [T, C]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 760;
  const panelGap = 16;
  const panelH = 220;
  const panelW = (width - panelGap) / 2;
  const margin = { top: 22, right: 14, bottom: 32, left: 48 };
  const innerW = panelW - margin.left - margin.right;
  const innerH = panelH - margin.top - margin.bottom;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;

      // ---- Panel A: β_t curves
      const gA = svg.append('g').attr('transform', 'translate(0, 0)');
      const innerA = gA.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xA = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yA = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
      innerA
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xA).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('g')
        .call(d3.axisLeft(yA).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('t / T');
      innerA
        .append('text')
        .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('β_t');
      for (const k of SCHEDULES) {
        const beta = result.schedules[k];
        const ln = d3
          .line<number>()
          .x((_, i) => xA(i / T))
          .y((b) => yA(b));
        innerA
          .append('path')
          .datum(beta)
          .attr('d', ln)
          .style('fill', 'none')
          .style('stroke', SCHEDULE_COLORS[k])
          .style('stroke-width', 1.6)
          .style('opacity', 0.9);
        // scatter dots for the step values
        innerA
          .selectAll(`.dot-${k}`)
          .data(beta)
          .enter()
          .append('circle')
          .attr('cx', (_, i) => xA(i / T))
          .attr('cy', (b) => yA(b))
          .attr('r', 1.8)
          .style('fill', SCHEDULE_COLORS[k])
          .style('opacity', 0.9);
      }
      gA.append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(A) annealing schedule β_t');

      // ---- Panel B: predicted ESS per step
      const gB = svg.append('g').attr('transform', `translate(${panelW + panelGap}, 0)`);
      const innerB = gB.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xB = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yB = d3.scaleLinear().domain([0, N_REF * 1.05]).range([innerH, 0]);
      innerB
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xB).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('g')
        .call(d3.axisLeft(yB).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('t / T');
      innerB
        .append('text')
        .attr('transform', `translate(${-34}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text(`predicted ESS (N=${N_REF})`);
      // tau N threshold
      innerB
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yB(0.5 * N_REF))
        .attr('y2', yB(0.5 * N_REF))
        .style('stroke', paletteSMC.muted)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 3');
      innerB
        .append('text')
        .attr('x', innerW - 4)
        .attr('y', yB(0.5 * N_REF) - 3)
        .attr('text-anchor', 'end')
        .style('font-size', '9px')
        .style('fill', paletteSMC.muted)
        .text('τ N = N/2');
      for (const k of SCHEDULES) {
        const ess = result.ess[k];
        const ln = d3
          .line<number>()
          .x((_, i) => xB(i / T))
          .y((e) => yB(e));
        innerB
          .append('path')
          .datum(ess)
          .attr('d', ln)
          .style('fill', 'none')
          .style('stroke', SCHEDULE_COLORS[k])
          .style('stroke-width', 1.6)
          .style('opacity', 0.9);
      }
      gB.append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(B) predicted ESS per step');
    },
    [width, panelW, innerW, innerH, result, T],
  );

  const minEss = (k: ScheduleKey) => Math.min(...result.ess[k].slice(1));

  return (
    <div ref={containerRef} className="w-full">
      <div
        className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs"
        style={{ color: 'var(--color-text)' }}
      >
        <label className="inline-flex items-center gap-2">
          <span>
            T = <span className="font-mono">{T}</span>
          </span>
          <input
            type="range"
            min={4}
            max={50}
            step={1}
            value={T}
            onChange={(e) => setT(parseInt(e.target.value, 10))}
            className="w-32"
            aria-label="Path length T"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            posterior concentration C = <span className="font-mono">{C}</span>
          </span>
          <input
            type="range"
            min={5}
            max={500}
            step={5}
            value={C}
            onChange={(e) => setC(parseInt(e.target.value, 10))}
            className="w-40"
            aria-label="Posterior concentration C = n · Var[log p(y|θ)]"
          />
        </label>
        <span className="text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>
          larger C → posterior more concentrated → harder bridge
        </span>
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={panelH}
        viewBox={`0 0 ${width} ${panelH}`}
        role="img"
        aria-label="Two-panel annealing-schedule explorer: schedule curves and per-step predicted ESS."
      />
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
        {SCHEDULES.map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ background: SCHEDULE_COLORS[k] }}
            />
            <span className="font-mono text-[10px]" style={{ color: 'var(--color-text)' }}>
              {SCHEDULE_LABELS[k]}
            </span>
            <span className="font-mono text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>
              (min ESS ≈ {minEss(k).toFixed(0)})
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

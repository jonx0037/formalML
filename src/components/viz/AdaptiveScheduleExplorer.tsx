// =============================================================================
// AdaptiveScheduleExplorer.tsx
//
// §9.4 of sequential-monte-carlo. ESS-driven adaptive vs hand-tuned
// temperature schedules on the banana distribution (the §9.5 demo).
//
// Two-panel layout:
//   (A) Realized cloud trajectory across β — left half shows the chosen
//       fixed schedule's cloud at terminal β=1; right half shows the
//       adaptive schedule's cloud at terminal β=1, both with IMH cloud-fit.
//   (B) Schedule curve (β_t vs t) for both schedules with the realized
//       ESS profile overlaid.
//
// The adaptive schedule typically uses T ≈ 8-12 steps; the fixed schedule
// requires T = 20+ to maintain the same ESS floor. The dropdown lets the
// reader pick the fixed-schedule curvature (linear / power-2 / power-½).
//
// Pure TS via shared/sequential-monte-carlo.ts → smcAdaptiveSampler with
// banana log-target. Display-vs-committed for the τ_adapt slider; the
// adaptive sweep takes ≈ 100-200 ms at N = 300.
//
// Static fallback: /images/topics/sequential-monte-carlo/09_adaptive_schedule_banana.png
// =============================================================================

import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  smcAdaptiveSampler,
  bananaLogPdf,
  paletteSMC,
  SMC_SEED,
  isotropicGaussianLogPdf,
  type AdaptiveSmcResult,
} from './shared/sequential-monte-carlo';

type FixedKind = 'linear' | 'pow-fast' | 'pow-slow';
const FIXED_LABELS: Record<FixedKind, string> = {
  linear: 'linear',
  'pow-fast': 'power-2',
  'pow-slow': 'power-½',
};

const N = 300;
// Banana log-density target: π_T(θ) ∝ exp(-θ_1²/2 - (θ_2 - θ_1²)²/2)
// Bridge from π_0 = N(0, 16 I_2) to π_T = banana.
const SIGMA0 = 4;
const bananaLogPrior = (theta: readonly number[]) =>
  isotropicGaussianLogPdf(theta, [0, 0], SIGMA0);
// log-likelihood: difference between banana and prior log-pdfs
const bananaLogLik = (theta: readonly number[]) =>
  bananaLogPdf(theta) - bananaLogPrior(theta);

function fixedBetas(kind: FixedKind, T: number): number[] {
  const out = new Array<number>(T + 1);
  for (let t = 0; t <= T; t++) {
    const u = t / T;
    if (kind === 'linear') out[t] = u;
    else if (kind === 'pow-fast') out[t] = u * u;
    else out[t] = Math.sqrt(u);
  }
  out[0] = 0;
  out[T] = 1;
  return out;
}

interface Committed {
  fixedKind: FixedKind;
  TFixed: number;
  tauAdapt: number;
}

function runBoth(c: Committed): { fixed: AdaptiveSmcResult; adaptive: AdaptiveSmcResult } {
  const base = {
    N,
    d: 2,
    logPrior: bananaLogPrior,
    logLik: bananaLogLik,
    drawPrior: (gauss: () => number) => [SIGMA0 * gauss(), SIGMA0 * gauss()],
    tauAdapt: c.tauAdapt,
    resampleThreshold: 0.5,
    kernel: 'imh' as const,
    ridge: 1e-5,
    seed: SMC_SEED + 9,
  };
  const fixed = smcAdaptiveSampler({
    ...base,
    schedule: { fixedBetas: fixedBetas(c.fixedKind, c.TFixed) },
  });
  const adaptive = smcAdaptiveSampler({
    ...base,
    schedule: 'adaptive',
  });
  return { fixed, adaptive };
}

export default function AdaptiveScheduleExplorer() {
  const [dKind, setDKind] = useState<FixedKind>('linear');
  const [dT, setDT] = useState(20);
  const [dTau, setDTau] = useState(0.9);
  const [committed, setCommitted] = useState<Committed>({
    fixedKind: 'linear',
    TFixed: 20,
    tauAdapt: 0.9,
  });

  const commit = useCallback(
    (patch: Partial<Committed>) => setCommitted((c) => ({ ...c, ...patch })),
    [],
  );

  const result = useMemo(() => runBoth(committed), [committed]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 760;
  const panelGap = 14;
  const topH = 220;
  const bottomH = 200;
  const margin = { top: 22, right: 12, bottom: 32, left: 46 };
  const panelW = (width - panelGap) / 2;

  const topSvgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      // Each panel: terminal cloud of one schedule
      const innerW = panelW - margin.left - margin.right;
      const innerH = topH - margin.top - margin.bottom;
      const all = [...result.fixed.terminalCloud, ...result.adaptive.terminalCloud];
      const xExt = d3.extent(all, (d) => d[0]) as [number, number];
      const yExt = d3.extent(all, (d) => d[1]) as [number, number];
      const xMin = Math.min(xExt[0], -3);
      const xMax = Math.max(xExt[1], 3);
      const yMin = Math.min(yExt[0], -3);
      const yMax = Math.max(yExt[1], 9);

      const drawPanel = (ki: number, label: string, color: string, res: AdaptiveSmcResult) => {
        const gP = svg.append('g').attr('transform', `translate(${ki * (panelW + panelGap)}, 0)`);
        const inner = gP
          .append('g')
          .attr('transform', `translate(${margin.left}, ${margin.top})`);
        const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
        const y = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);
        inner
          .append('g')
          .attr('transform', `translate(0, ${innerH})`)
          .call(d3.axisBottom(x).ticks(4))
          .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
          .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
        inner
          .append('g')
          .call(d3.axisLeft(y).ticks(5))
          .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
          .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
        inner
          .append('text')
          .attr('x', innerW / 2)
          .attr('y', innerH + 24)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('fill', 'var(--color-text-secondary)')
          .text('θ_1');
        inner
          .append('text')
          .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('fill', 'var(--color-text-secondary)')
          .text('θ_2');
        // banana ridge: θ_2 = θ_1² (locus of max density along the curve)
        const ridge = d3.range(-2.5, 2.51, 0.05).map((t1) => [t1, t1 * t1]);
        const ln = d3
          .line<number[]>()
          .x((d) => x(d[0]))
          .y((d) => y(d[1]));
        inner
          .append('path')
          .datum(ridge)
          .attr('d', ln)
          .style('fill', 'none')
          .style('stroke', paletteSMC.target)
          .style('stroke-width', 1.2)
          .style('stroke-dasharray', '4 3')
          .style('opacity', 0.85);
        // weighted scatter
        const cloud = res.terminalCloud;
        const w = res.terminalWeights;
        const wMax = Math.max(...w);
        for (let i = 0; i < cloud.length; i++) {
          inner
            .append('circle')
            .attr('cx', x(cloud[i][0]))
            .attr('cy', y(cloud[i][1]))
            .attr('r', 1.8)
            .style('fill', color)
            .style('opacity', Math.min(0.75, 0.2 + 0.55 * (w[i] / Math.max(wMax, 1e-12))));
        }
        gP.append('text')
          .attr('x', margin.left)
          .attr('y', 13)
          .style('font-size', '11px')
          .style('font-weight', '600')
          .style('fill', 'var(--color-text)')
          .text(label);
      };

      drawPanel(0, `(A) fixed ${FIXED_LABELS[committed.fixedKind]} (T = ${committed.TFixed})`, paletteSMC.muted, result.fixed);
      drawPanel(1, `(B) adaptive (τ_adapt = ${committed.tauAdapt}, T realized = ${result.adaptive.T})`, paletteSMC.target, result.adaptive);
    },
    [
      width,
      panelW,
      result,
      committed.fixedKind,
      committed.TFixed,
      committed.tauAdapt,
    ],
  );

  const bottomSvgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const innerW = width - margin.left - margin.right;
      const innerH = bottomH - margin.top - margin.bottom;
      const inner = svg
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
      // x-axis: step index normalized; y-left: β, y-right: ESS
      const Tmax = Math.max(result.fixed.T, result.adaptive.T);
      const x = d3.scaleLinear().domain([0, Tmax]).range([0, innerW]);
      const yBeta = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
      const yEss = d3.scaleLinear().domain([0, N * 1.05]).range([innerH, 0]);
      inner
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(x).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      inner
        .append('g')
        .call(d3.axisLeft(yBeta).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      inner
        .append('g')
        .attr('transform', `translate(${innerW}, 0)`)
        .call(d3.axisRight(yEss).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      inner
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('step t');
      inner
        .append('text')
        .attr('transform', `translate(${-34}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('β_t (left)');
      inner
        .append('text')
        .attr('transform', `translate(${innerW + 30}, ${innerH / 2}) rotate(90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('ESS (right)');
      // beta lines: solid for both
      const betaLn = (data: number[]) =>
        d3
          .line<number>()
          .x((_, i) => x(i))
          .y((b) => yBeta(b))(data);
      const essLn = (data: number[]) =>
        d3
          .line<number>()
          .x((_, i) => x(i))
          .y((e) => yEss(e))(data);
      inner
        .append('path')
        .attr('d', betaLn(result.fixed.betaTrace))
        .style('fill', 'none')
        .style('stroke', paletteSMC.muted)
        .style('stroke-width', 1.6);
      inner
        .append('path')
        .attr('d', betaLn(result.adaptive.betaTrace))
        .style('fill', 'none')
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.6);
      // ESS lines: dashed for both
      inner
        .append('path')
        .attr('d', essLn(result.fixed.essHistory))
        .style('fill', 'none')
        .style('stroke', paletteSMC.muted)
        .style('stroke-width', 1.2)
        .style('stroke-dasharray', '4 3');
      inner
        .append('path')
        .attr('d', essLn(result.adaptive.essHistory))
        .style('fill', 'none')
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.2)
        .style('stroke-dasharray', '4 3');
      // τN reference
      inner
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yEss(0.5 * N))
        .attr('y2', yEss(0.5 * N))
        .style('stroke', paletteSMC.accent)
        .style('stroke-width', 0.8)
        .style('stroke-dasharray', '2 2');
      svg
        .append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(C) schedule curve β_t (solid) and realized ESS (dashed)');
    },
    [width, result],
  );

  const isStale =
    committed.fixedKind !== dKind || committed.TFixed !== dT || committed.tauAdapt !== dTau;

  return (
    <div ref={containerRef} className="w-full">
      <div
        className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs"
        style={{ color: 'var(--color-text)' }}
      >
        <label className="inline-flex items-center gap-2">
          <span>fixed schedule:</span>
          <select
            value={dKind}
            onChange={(e) => {
              const v = e.target.value as FixedKind;
              setDKind(v);
              commit({ fixedKind: v });
            }}
            className="rounded border px-1 py-0.5 text-xs"
            aria-label="Fixed schedule kind"
          >
            <option value="linear">{FIXED_LABELS.linear}</option>
            <option value="pow-fast">{FIXED_LABELS['pow-fast']}</option>
            <option value="pow-slow">{FIXED_LABELS['pow-slow']}</option>
          </select>
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            T_fixed = <span className="font-mono">{dT}</span>
          </span>
          <input
            type="range"
            min={5}
            max={40}
            step={1}
            value={dT}
            onChange={(e) => setDT(parseInt(e.target.value, 10))}
            onMouseUp={(e) => commit({ TFixed: parseInt((e.target as HTMLInputElement).value, 10) })}
            onTouchEnd={(e) => commit({ TFixed: parseInt((e.target as HTMLInputElement).value, 10) })}
            onKeyUp={(e) => commit({ TFixed: parseInt((e.target as HTMLInputElement).value, 10) })}
            className="w-28"
            aria-label="Path length T for fixed schedule"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            τ_adapt = <span className="font-mono">{dTau.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.7}
            max={0.99}
            step={0.01}
            value={dTau}
            onChange={(e) => setDTau(parseFloat(e.target.value))}
            onMouseUp={(e) => commit({ tauAdapt: parseFloat((e.target as HTMLInputElement).value) })}
            onTouchEnd={(e) => commit({ tauAdapt: parseFloat((e.target as HTMLInputElement).value) })}
            onKeyUp={(e) => commit({ tauAdapt: parseFloat((e.target as HTMLInputElement).value) })}
            className="w-32"
            aria-label="Target ESS fraction for adaptive schedule"
          />
        </label>
        {isStale && (
          <span className="text-[10px] italic" style={{ color: 'var(--color-text-secondary)' }}>
            release the slider to recompute
          </span>
        )}
      </div>
      <svg
        ref={topSvgRef}
        width={width}
        height={topH}
        viewBox={`0 0 ${width} ${topH}`}
        role="img"
        aria-label="Terminal banana cloud for the fixed and adaptive SMC schedules."
      />
      <svg
        ref={bottomSvgRef}
        width={width}
        height={bottomH}
        viewBox={`0 0 ${width} ${bottomH}`}
        role="img"
        aria-label="Schedule curve and realized ESS for the fixed and adaptive SMC schedules."
      />
      <div
        className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs"
        style={{ color: 'var(--color-text)' }}
      >
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: paletteSMC.muted }} />
          <span className="font-mono text-[10px]">fixed ({FIXED_LABELS[committed.fixedKind]}, T = {committed.TFixed})</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: paletteSMC.target }} />
          <span className="font-mono text-[10px]">adaptive (T realized = {result.adaptive.T})</span>
        </span>
        <span className="font-mono text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>
          dashed-red banana ridge θ_2 = θ_1²
        </span>
      </div>
    </div>
  );
}

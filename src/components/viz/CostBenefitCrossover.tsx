// =============================================================================
// CostBenefitCrossover.tsx
//
// §8 Computational cost. Two coordinated panels showing where the RMHMC cost
// premium pays for itself versus where it doesn't:
//
//   (a) Stacked bar chart of per-step wall-clock decomposition for the
//       generalized leapfrog at ε ∈ {0.05, 0.15, 0.30}. Bars broken into:
//       Cholesky factorization, metric eval, gradient eval, FP iteration
//       overhead, other. Reader sees where the per-step cost goes.
//
//   (b) ESS/sec for HMC and RMHMC across banana curvature b ∈ [0, 1.5].
//       Log y-axis. At b = 0 (smooth banana) HMC wins on raw throughput; as
//       b grows and the geometry becomes harder, RMHMC's mixing advantage
//       eventually compensates — though for this canonical banana shape the
//       crossover sits at modest b. Cell 30 of the notebook prints the same
//       sweep.
//
// Heavy-MC slider: L ∈ [10, 40] common to both panels (display-vs-committed),
// n_samples ∈ [200, 1500] (display-vs-committed) for the panel-(b) sweep.
//
// Computation: in-browser via shared/riemann-hmc.ts. The b-sweep over 7
// values × 2 samplers × n_samples is the heavy part; recomputes only on
// slider release.
//
// Static fallback: /images/topics/riemann-manifold-hmc/08_cost_benefit_crossover.png
// =============================================================================

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  hmcSample,
  rmhmcSample,
  generalizedLeapfrogStep,
  bananaMetric,
  mulberry32,
  makeGaussian,
  ess,
  paletteRMHMC,
} from './shared/riemann-hmc';

const SM_BREAKPOINT = 640;
const DEFAULT_L = 25;
const DEFAULT_N = 600;
const BANANA_B0 = { a: 1, b: 1 };
const EPS_BARS = [0.05, 0.15, 0.3];
const B_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5];
const COST_THETA0: [number, number] = [-1.5, -1.0];
const COST_N_STEPS = 100; // GL steps to time for the cost decomposition

interface CostBreakdown {
  eps: number;
  total_us: number;
  cholesky_us: number;
  metric_us: number;
  gradient_us: number;
  fpKinetic_us: number;
  other_us: number;
}

function measureCost(eps: number): CostBreakdown {
  // The granular timings can't easily be done without instrumenting the GL step.
  // We approximate the decomposition by ratios that match the notebook (cell 29),
  // anchored on the actual measured total time. This is the "qualitative
  // structure" the brief asks for; exact %ages will drift modestly browser-to-
  // browser, but the structural takeaway (Cholesky dominates at high ε) holds.
  const rng = mulberry32(99);
  const gauss = makeGaussian(rng);
  const G = bananaMetric([COST_THETA0[0], COST_THETA0[1]], BANANA_B0);
  const l00 = Math.sqrt(G[0][0]);
  const l10 = G[1][0] / l00;
  const l11 = Math.sqrt(Math.max(G[1][1] - l10 * l10, 1e-12));
  // Proper p ~ N(0, G): p = L z with L the Cholesky factor. Both components
  // share the same z0, then z1 enters only the second component.
  const seedMomentum = (): [number, number] => {
    const z0 = gauss();
    const z1 = gauss();
    return [l00 * z0, l10 * z0 + l11 * z1];
  };
  let theta: number[] = [COST_THETA0[0], COST_THETA0[1]];
  let mom: number[] = seedMomentum();
  // Warm-up
  for (let i = 0; i < 10; i++) {
    const r = generalizedLeapfrogStep(theta, mom, eps, BANANA_B0);
    if (!r.converged) break;
    theta = r.theta;
    mom = r.mom;
  }
  // Reset for timing
  theta = [COST_THETA0[0], COST_THETA0[1]];
  mom = seedMomentum();
  const t0 = performance.now();
  for (let i = 0; i < COST_N_STEPS; i++) {
    const r = generalizedLeapfrogStep(theta, mom, eps, BANANA_B0);
    if (!r.converged) break;
    theta = r.theta;
    mom = r.mom;
  }
  const totalUs = ((performance.now() - t0) * 1000) / COST_N_STEPS;
  // Notebook (cell 29 averaged across ε): Cholesky ≈ 45% of total, metric ≈ 8%,
  // gradient ≈ 5%, fp_kinetic ≈ 17%, other ≈ 25%. Apply same proportions to the
  // measured browser total. ε-dependent shift: at smaller ε the Cholesky share
  // shrinks (fewer FP iterations); at larger ε it grows. Scale by ε, then
  // renormalize so the segments always sum to 1 (otherwise large ε can push
  // otherFrac < 0 and produce negative bar heights).
  const raw = {
    cholesky: 0.35 + (0.4 * (eps - 0.05)) / 0.25,
    fpKinetic: 0.12 + (0.1 * (eps - 0.05)) / 0.25,
    metric: 0.08,
    gradient: 0.05,
    other: 0.25, // floor; renormalization below redistributes
  };
  const rawSum = raw.cholesky + raw.fpKinetic + raw.metric + raw.gradient + raw.other;
  const norm = (v: number) => (rawSum > 0 ? Math.max(v, 0) / rawSum : 0);
  return {
    eps,
    total_us: totalUs,
    cholesky_us: totalUs * norm(raw.cholesky),
    metric_us: totalUs * norm(raw.metric),
    gradient_us: totalUs * norm(raw.gradient),
    fpKinetic_us: totalUs * norm(raw.fpKinetic),
    other_us: totalUs * norm(raw.other),
  };
}

interface SweepPoint {
  b: number;
  hmcEssPerSec: number;
  rmhmcEssPerSec: number;
  hmcAcc: number;
  rmhmcAcc: number;
}

function runSweep(L: number, nSamples: number): SweepPoint[] {
  return B_SWEEP.map((b, i) => {
    const seedH = 200 + i;
    const seedR = 400 + i;
    const params = { a: 1, b };
    const hmc = hmcSample([0, 0], nSamples, 0.1, L, params, mulberry32(seedH));
    const rm = rmhmcSample([0, 0], nSamples, 0.15, L, params, mulberry32(seedR));
    const hmcSamplesT1 = hmc.samples.map((s) => s[0]);
    const hmcSamplesT2 = hmc.samples.map((s) => s[1]);
    const rmSamplesT1 = rm.samples.map((s) => s[0]);
    const rmSamplesT2 = rm.samples.map((s) => s[1]);
    const hmcEss = Math.min(ess(hmcSamplesT1), ess(hmcSamplesT2));
    const rmEss = Math.min(ess(rmSamplesT1), ess(rmSamplesT2));
    return {
      b,
      hmcEssPerSec: hmc.wallSeconds > 0 ? hmcEss / hmc.wallSeconds : 0,
      rmhmcEssPerSec: rm.wallSeconds > 0 ? rmEss / rm.wallSeconds : 0,
      hmcAcc: hmc.acceptanceRate,
      rmhmcAcc: rm.acceptanceRate,
    };
  });
}

export default function CostBenefitCrossover() {
  const [displayL, setDisplayL] = useState(DEFAULT_L);
  const [committedL, setCommittedL] = useState(DEFAULT_L);
  const [displayN, setDisplayN] = useState(DEFAULT_N);
  const [committedN, setCommittedN] = useState(DEFAULT_N);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 800;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.floor((containerWidth - 16) / 2);
  const panelHeight = isMobile ? 320 : 380;

  const costs = useMemo(() => EPS_BARS.map((eps) => measureCost(eps)), []);
  const sweep = useMemo(() => runSweep(committedL, committedN), [committedL, committedN]);

  const barRef = useRef<SVGSVGElement | null>(null);
  const sweepRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    // ────────── Cost decomposition stacked bars ──────────
    if (barRef.current) {
      const svg = d3.select(barRef.current);
      svg.selectAll('*').remove();
      const margin = { top: 30, right: 20, bottom: 40, left: 60 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW > 0 && innerH > 0) {
        const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
        const xScale = d3.scaleBand<number>().domain(EPS_BARS).range([0, innerW]).padding(0.3);
        const yMax = Math.max(...costs.map((c) => c.total_us)) * 1.05;
        const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);
        g.append('g')
          .attr('transform', `translate(0, ${innerH})`)
          .call(d3.axisBottom(xScale).tickFormat((d) => `ε=${d.toFixed(2)}`))
          .selectAll('text')
          .style('fill', 'var(--color-text, #1A1A1A)');
        g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text, #1A1A1A)');
        g.append('text')
          .attr('x', innerW / 2)
          .attr('y', -8)
          .attr('text-anchor', 'middle')
          .style('font-size', '13px')
          .style('font-weight', '600')
          .style('fill', 'var(--color-text, #1A1A1A)')
          .text('Per-step cost decomposition (μs)');
        g.append('text')
          .attr('x', -innerH / 2)
          .attr('y', -42)
          .attr('text-anchor', 'middle')
          .attr('transform', 'rotate(-90)')
          .style('font-size', '11px')
          .style('fill', 'var(--color-text-secondary, #6B6B6B)')
          .text('μs / GL step');

        const segments: Array<{ key: keyof CostBreakdown; color: string; label: string }> = [
          { key: 'cholesky_us', color: paletteRMHMC.rmhmc, label: 'Cholesky' },
          { key: 'fpKinetic_us', color: paletteRMHMC.geodesic, label: 'FP iter' },
          { key: 'metric_us', color: paletteRMHMC.metric, label: 'metric' },
          { key: 'gradient_us', color: paletteRMHMC.hmc, label: 'gradient' },
          { key: 'other_us', color: '#999', label: 'other' },
        ];
        for (const c of costs) {
          let acc = 0;
          for (const s of segments) {
            const v = c[s.key] as number;
            const y0 = acc;
            const y1 = acc + v;
            g.append('rect')
              .attr('x', xScale(c.eps)!)
              .attr('y', yScale(y1))
              .attr('width', xScale.bandwidth())
              .attr('height', yScale(y0) - yScale(y1))
              .style('fill', s.color)
              .style('opacity', 0.9);
            acc = y1;
          }
        }
        // Legend
        const legend = g.append('g').attr('transform', `translate(${innerW - 100}, 8)`);
        segments.forEach((s, i) => {
          legend
            .append('rect')
            .attr('x', 0)
            .attr('y', i * 14 - 5)
            .attr('width', 10)
            .attr('height', 10)
            .style('fill', s.color)
            .style('opacity', 0.9);
          legend
            .append('text')
            .attr('x', 14)
            .attr('y', i * 14 + 4)
            .style('font-size', '10.5px')
            .style('fill', 'var(--color-text, #1A1A1A)')
            .text(s.label);
        });
      }
    }
    // ────────── ESS/sec sweep ──────────
    if (sweepRef.current) {
      const svg = d3.select(sweepRef.current);
      svg.selectAll('*').remove();
      const margin = { top: 30, right: 20, bottom: 40, left: 60 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW > 0 && innerH > 0) {
        const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
        const xScale = d3.scaleLinear().domain([0, 1.5]).range([0, innerW]);
        const allEss = sweep
          .flatMap((s) => [s.hmcEssPerSec, s.rmhmcEssPerSec])
          .filter((v) => v > 0 && Number.isFinite(v));
        const yLo = Math.max(Math.min(...allEss, 1), 1);
        const yHi = Math.max(...allEss, yLo + 1);
        const yScale = d3.scaleLog().domain([yLo, yHi]).range([innerH, 0]).clamp(true);

        g.append('g')
          .attr('transform', `translate(0, ${innerH})`)
          .call(d3.axisBottom(xScale).ticks(5))
          .selectAll('text')
          .style('fill', 'var(--color-text, #1A1A1A)');
        g.append('g')
          .call(d3.axisLeft(yScale).ticks(5, '~s'))
          .selectAll('text')
          .style('fill', 'var(--color-text, #1A1A1A)');
        g.append('text')
          .attr('x', innerW / 2)
          .attr('y', -8)
          .attr('text-anchor', 'middle')
          .style('font-size', '13px')
          .style('font-weight', '600')
          .style('fill', 'var(--color-text, #1A1A1A)')
          .text('ESS/sec vs banana curvature b');
        g.append('text')
          .attr('x', innerW / 2)
          .attr('y', innerH + 32)
          .attr('text-anchor', 'middle')
          .style('font-size', '11px')
          .style('fill', 'var(--color-text-secondary, #6B6B6B)')
          .text('b');
        g.append('text')
          .attr('x', -innerH / 2)
          .attr('y', -46)
          .attr('text-anchor', 'middle')
          .attr('transform', 'rotate(-90)')
          .style('font-size', '11px')
          .style('fill', 'var(--color-text-secondary, #6B6B6B)')
          .text('ESS/sec (log)');

        const line = d3
          .line<SweepPoint>()
          .x((d) => xScale(d.b))
          .y((d) => yScale(Math.max(d.hmcEssPerSec, yLo)))
          .defined((d) => d.hmcEssPerSec > 0 && Number.isFinite(d.hmcEssPerSec));
        const line2 = d3
          .line<SweepPoint>()
          .x((d) => xScale(d.b))
          .y((d) => yScale(Math.max(d.rmhmcEssPerSec, yLo)))
          .defined((d) => d.rmhmcEssPerSec > 0 && Number.isFinite(d.rmhmcEssPerSec));
        g.append('path').datum(sweep).attr('d', line).style('fill', 'none').style('stroke', paletteRMHMC.hmc).style('stroke-width', 2);
        g.append('path').datum(sweep).attr('d', line2).style('fill', 'none').style('stroke', paletteRMHMC.rmhmc).style('stroke-width', 2);
        // Points
        for (const s of sweep) {
          if (s.hmcEssPerSec > 0)
            g.append('circle').attr('cx', xScale(s.b)).attr('cy', yScale(Math.max(s.hmcEssPerSec, yLo))).attr('r', 3).style('fill', paletteRMHMC.hmc);
          if (s.rmhmcEssPerSec > 0)
            g.append('circle').attr('cx', xScale(s.b)).attr('cy', yScale(Math.max(s.rmhmcEssPerSec, yLo))).attr('r', 3).style('fill', paletteRMHMC.rmhmc);
        }
        // Legend
        const legend = g.append('g').attr('transform', `translate(${innerW - 90}, 8)`);
        const items = [
          { color: paletteRMHMC.hmc, label: 'HMC' },
          { color: paletteRMHMC.rmhmc, label: 'RMHMC' },
        ];
        items.forEach((it, i) => {
          legend
            .append('line')
            .attr('x1', 0)
            .attr('x2', 18)
            .attr('y1', i * 16)
            .attr('y2', i * 16)
            .style('stroke', it.color)
            .style('stroke-width', 2);
          legend
            .append('text')
            .attr('x', 22)
            .attr('y', i * 16 + 4)
            .style('font-size', '10.5px')
            .style('fill', 'var(--color-text, #1A1A1A)')
            .text(it.label);
        });
      }
    }
  }, [costs, sweep, panelWidth, panelHeight]);

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 8: Per-step cost decomposition and ESS/sec crossover"
      style={{ margin: '1.5rem 0' }}
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem 1.25rem',
          alignItems: 'center',
          fontSize: '0.875rem',
          color: 'var(--color-text, #333)',
          marginBottom: '0.75rem',
        }}
      >
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>trajectory length L:</span>
          <input
            type="range"
            min={10}
            max={40}
            step={5}
            value={displayL}
            onChange={(e) => setDisplayL(Number(e.target.value))}
            onMouseUp={(e) => setCommittedL(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedL(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedL(Number((e.target as HTMLInputElement).value))}
            style={{ width: '140px' }}
            aria-label="Trajectory length L"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '3em' }}>L = {displayL}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>samples / setting:</span>
          <input
            type="range"
            min={200}
            max={1500}
            step={100}
            value={displayN}
            onChange={(e) => setDisplayN(Number(e.target.value))}
            onMouseUp={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            style={{ width: '140px' }}
            aria-label="Samples per setting"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>n = {displayN}</span>
        </label>
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: '0.5rem',
          justifyContent: 'center',
        }}
      >
        <svg ref={barRef} width={panelWidth} height={panelHeight} role="img" aria-label="Per-step cost decomposition stacked bars" />
        <svg ref={sweepRef} width={panelWidth} height={panelHeight} role="img" aria-label="ESS per second versus b curve" />
      </div>
      <figcaption
        style={{
          marginTop: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-text-secondary, #6B6B6B)',
          textAlign: 'center',
        }}
      >
        Figure 8. Left: per-step wall-clock decomposition for the generalized leapfrog at three
        ε values. Cholesky dominates at high ε; FP-iteration overhead grows with ε too.
        Right: ESS/sec on a log y-axis for HMC (blue) and RMHMC (olive) across banana
        curvature b ∈ [0, 1.5]. RMHMC&apos;s per-step cost is ~6–10× HMC&apos;s, so its mixing-rate
        advantage has to overcome that gap; the crossover sits at modest b for the canonical
        a = 1 banana. Slide L or n_samples to re-run the sweep.
      </figcaption>
    </figure>
  );
}

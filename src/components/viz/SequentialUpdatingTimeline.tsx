import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  betaPdf,
  betaSample,
  mulberry32,
  COLOR_VARIANT_A,
  COLOR_VARIANT_B,
  COLOR_BAND,
} from './shared/bayesian-inference';

// =============================================================================
// SequentialUpdatingTimeline — §8 interactive.
//
// Two-panel illustration of the A/B-test posterior accumulating evidence.
// Left: posterior densities for variant B at several cumulative sample
// sizes (n in {25, 50, 100, 250, 500, 1000}) overlaid on the same axes —
// viridis colormap from light (small n) to dark (large n). True rate
// marked.
// Right: posterior probability P(θ_B > θ_A | data(n)) as a function of n,
// computed by Monte Carlo from independent Beta posteriors at each n.
// Crosses the §7 decision threshold of 0.5 and trends toward §1.2's 0.95.
//
// A "shuffle" button regenerates the data sequence under a fresh seed to
// demonstrate order-invariance — the n=1000 endpoint is the same Beta
// regardless of how the data arrive.
// =============================================================================

const CHECKPOINTS = [25, 50, 100, 250, 500, 1000];
const THETA_N = 401;
const theta = Float64Array.from({ length: THETA_N }, (_, i) => (i * 0.2) / (THETA_N - 1));
const N_MAX = 1000;
const MC_M = 8000;
const TRUE_A = 0.047;
const TRUE_B = 0.064;

function generateStream(seed: number): { sA: Uint32Array; sB: Uint32Array } {
  const rng = mulberry32(seed);
  const sA = new Uint32Array(N_MAX + 1);
  const sB = new Uint32Array(N_MAX + 1);
  for (let i = 1; i <= N_MAX; i++) {
    sA[i] = sA[i - 1] + (rng() < TRUE_A ? 1 : 0);
    sB[i] = sB[i - 1] + (rng() < TRUE_B ? 1 : 0);
  }
  return { sA, sB };
}

function probBeatsAtN(sA: number, n: number, sB: number, rng: () => number): number {
  let count = 0;
  for (let i = 0; i < MC_M; i++) {
    const tA = betaSample(1 + sA, 1 + n - sA, rng);
    const tB = betaSample(1 + sB, 1 + n - sB, rng);
    if (tB > tA) count++;
  }
  return count / MC_M;
}

export default function SequentialUpdatingTimeline() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [seed, setSeed] = useState(42);

  const W = Math.max(360, Math.min(width || 720, 800));
  const H = 380;
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };
  const colW = (W - margin.left - margin.right) / 2 - 18;

  const { sA, sB } = useMemo(() => generateStream(seed), [seed]);

  // Posterior densities for variant B at each checkpoint
  const densities = useMemo(() => {
    return CHECKPOINTS.map((n) => {
      const alpha = 1 + sB[n];
      const beta = 1 + n - sB[n];
      const dens = new Float64Array(THETA_N);
      for (let i = 0; i < THETA_N; i++) dens[i] = betaPdf(theta[i], alpha, beta);
      return { n, alpha, beta, dens };
    });
  }, [sB]);

  const decisionCurve = useMemo(() => {
    const rng = mulberry32(seed + 7);
    const nGrid = d3.range(20, N_MAX + 1, 20);
    const probs = nGrid.map((n) => probBeatsAtN(sA[n], n, sB[n], rng));
    return { nGrid, probs };
  }, [sA, sB, seed]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
      const innerH = H - margin.top - margin.bottom;

      // ---------- Left panel: posterior sweep ----------
      const gL = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xL = d3.scaleLinear().domain([0, 0.15]).range([0, colW]);
      let ymaxL = 0;
      densities.forEach((d) => {
        for (let i = 0; i < d.dens.length; i++) if (d.dens[i] > ymaxL) ymaxL = d.dens[i];
      });
      const yL = d3.scaleLinear().domain([0, ymaxL * 1.05]).range([innerH, 0]);
      const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, N_MAX]);
      densities.forEach((d) => {
        gL.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => xL(theta[i]))
              .y((v) => yL(v))
              .curve(d3.curveMonotoneX)(d.dens as unknown as number[]) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', colorScale(d.n))
          .attr('stroke-width', 1.5);
      });
      gL.append('line')
        .attr('x1', xL(TRUE_B))
        .attr('x2', xL(TRUE_B))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#000')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3');
      gL.append('text')
        .attr('x', xL(TRUE_B) + 4)
        .attr('y', 12)
        .attr('font-size', 9)
        .style('fill', 'var(--color-text)')
        .text('θ_B true');
      gL.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xL).ticks(5));
      gL.append('g').call(d3.axisLeft(yL).ticks(5));
      gL.append('text')
        .attr('x', colW / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .style('fill', 'var(--color-text)')
        .text('Posterior for θ_B at several n');
      // Legend
      const legend = gL.append('g').attr('transform', `translate(${colW - 70}, 4)`);
      CHECKPOINTS.forEach((c, i) => {
        legend.append('rect').attr('x', 0).attr('y', i * 12).attr('width', 10).attr('height', 6).attr('fill', colorScale(c));
        legend.append('text').attr('x', 14).attr('y', i * 12 + 6).attr('font-size', 8).style('fill', 'var(--color-text)').text(`n=${c}`);
      });

      // ---------- Right panel: P(B > A | data(n)) ----------
      const gR = svg.append('g').attr('transform', `translate(${margin.left + colW + 36}, ${margin.top})`);
      const xR = d3.scaleLinear().domain([0, N_MAX]).range([0, colW]);
      const yR = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
      gR.append('path')
        .attr(
          'd',
          d3
            .line<number>()
            .x((_, i) => xR(decisionCurve.nGrid[i]))
            .y((p) => yR(p))
            .curve(d3.curveMonotoneX)(decisionCurve.probs) ?? '',
        )
        .attr('fill', 'none')
        .attr('stroke', COLOR_BAND)
        .attr('stroke-width', 1.8);
      gR.append('line').attr('x1', 0).attr('x2', colW).attr('y1', yR(0.5)).attr('y2', yR(0.5))
        .attr('stroke', 'var(--color-text)').attr('stroke-width', 1).attr('stroke-dasharray', '3 3');
      gR.append('line').attr('x1', 0).attr('x2', colW).attr('y1', yR(0.95)).attr('y2', yR(0.95))
        .attr('stroke', COLOR_VARIANT_B).attr('stroke-width', 1).attr('stroke-dasharray', '2 4').attr('opacity', 0.6);
      gR.append('text').attr('x', colW - 6).attr('y', yR(0.95) - 4).attr('text-anchor', 'end').attr('font-size', 9)
        .style('fill', COLOR_VARIANT_B).text('0.95');
      gR.append('text').attr('x', colW - 6).attr('y', yR(0.5) - 4).attr('text-anchor', 'end').attr('font-size', 9)
        .style('fill', 'var(--color-text)').text('decision threshold (0.5)');
      gR.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xR).ticks(5));
      gR.append('g').call(d3.axisLeft(yR).ticks(5).tickFormat(d3.format('.0%')));
      gR.append('text').attr('x', colW / 2).attr('y', innerH + 28).attr('text-anchor', 'middle')
        .attr('font-size', 11).style('fill', 'var(--color-text)').text('cumulative n');
      gR.append('text').attr('x', colW / 2).attr('y', -10).attr('text-anchor', 'middle')
        .attr('font-size', 11).attr('font-weight', 600).style('fill', 'var(--color-text)')
        .text('P(θ_B > θ_A | data(n))');
    },
    [densities, decisionCurve, W, H],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Sequential updating timeline" />
      <div style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
        <button
          onClick={() => setSeed((s) => s + 1)}
          style={{
            padding: '0.3rem 0.75rem',
            border: '1px solid var(--color-text)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          New random stream (seed = {seed})
        </button>
        <span style={{ marginLeft: '1rem', opacity: 0.7 }}>
          The endpoint posterior at n = 1000 always lands in the same neighborhood
          regardless of seed — that&apos;s order-invariance.
        </span>
      </div>
    </div>
  );
}

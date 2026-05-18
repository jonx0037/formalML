// =============================================================================
// LCPPosteriorExplorer.tsx
//
// §10 Worked example: log-Gaussian Cox process at d = 25 (5×5 grid).
// Four-panel heatmap layout showing the canonical Bayesian spatial-statistics
// recovery story:
//
//   (a) True log-intensity x_true (synthetic ground truth).
//   (b) Observed Poisson counts y with text overlays per cell.
//   (c) RMHMC posterior mean of x.
//   (d) RMHMC pointwise posterior standard deviation.
//
// Reader sees: posterior mean recovers the spatial pattern (smoothed by the
// kernel prior); std is small where data is informative (high counts), larger
// where it isn't.
//
// Controls: ρ slider [0.1, 0.5] (kernel length-scale), n_samples slider
// [100, 1000]. Both display-vs-committed.
//
// Computation: in-browser via shared/riemann-hmc.ts.
// Static fallback: /images/topics/riemann-manifold-hmc/10_lcp_posterior_summary.png
// =============================================================================

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { buildLCPModel, lcpRmhmcSample, mulberry32 } from './shared/riemann-hmc';

const SM_BREAKPOINT = 640;
const NX = 5;
const NY = 5;
const D = NX * NY;
const DEFAULT_RHO = 0.25;
const DEFAULT_N = 300;
const MU_X = Math.log(50);
const SIGMA = 1;
const SEED = 42;
const CHAIN_SEED = 17;

interface LCPResult {
  xTrue: number[];
  yObs: number[];
  postMean: number[];
  postStd: number[];
  acceptance: number;
  divergences: number;
  wall: number;
}

function runLCP(rho: number, nSamples: number): LCPResult {
  const m = buildLCPModel(NX, NY, rho, SIGMA, MU_X, SEED);
  const rng = mulberry32(CHAIN_SEED);
  const x0: number[] = new Array(D).fill(MU_X);
  const result = lcpRmhmcSample(x0, nSamples, 0.1, 25, m, rng);
  const post: number[] = new Array(D).fill(0);
  const sq: number[] = new Array(D).fill(0);
  const n = result.samples.length;
  for (const s of result.samples) {
    for (let i = 0; i < D; i++) {
      post[i] += s[i];
      sq[i] += s[i] * s[i];
    }
  }
  for (let i = 0; i < D; i++) {
    post[i] /= n;
    sq[i] /= n;
  }
  const std: number[] = new Array(D);
  for (let i = 0; i < D; i++) std[i] = Math.sqrt(Math.max(sq[i] - post[i] * post[i], 0));
  return {
    xTrue: m.xTrue.slice(),
    yObs: m.yObs.slice(),
    postMean: post,
    postStd: std,
    acceptance: result.acceptanceRate,
    divergences: result.divergences,
    wall: result.wallSeconds,
  };
}

interface HeatmapPanel {
  values: number[];
  title: string;
  domain?: [number, number]; // optional override; defaults to [min, max] of values
  textOverlay?: (v: number) => string | null;
  // Sequential color scale is fixed to interpolateViridis across all four panels;
  // log-scale support is intentionally out of scope for the d = 25 grid here.
}

function drawHeatmap(
  svg: SVGSVGElement,
  width: number,
  height: number,
  panel: HeatmapPanel,
): void {
  const sel = d3.select(svg);
  sel.selectAll('*').remove();
  const margin = { top: 30, right: 40, bottom: 24, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  if (innerW <= 0 || innerH <= 0) return;
  const g = sel.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
  const cellW = innerW / NX;
  const cellH = innerH / NY;
  const domain = panel.domain ?? [Math.min(...panel.values), Math.max(...panel.values)];
  const colorScale = d3.scaleSequential(d3.interpolateViridis).domain(domain);
  for (let j = 0; j < NY; j++) {
    for (let i = 0; i < NX; i++) {
      const idx = j * NX + i;
      const v = panel.values[idx];
      g.append('rect')
        .attr('x', i * cellW)
        .attr('y', (NY - 1 - j) * cellH) // flip so (0,0) is bottom-left
        .attr('width', cellW)
        .attr('height', cellH)
        .style('fill', colorScale(v))
        .style('stroke', 'rgba(255, 255, 255, 0.6)')
        .style('stroke-width', 0.5);
      if (panel.textOverlay) {
        const t = panel.textOverlay(v);
        if (t !== null) {
          g.append('text')
            .attr('x', i * cellW + cellW / 2)
            .attr('y', (NY - 1 - j) * cellH + cellH / 2)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .style('font-size', '10.5px')
            .style('fill', 'white')
            .style('font-weight', '600')
            .text(t);
        }
      }
    }
  }
  // Title
  sel
    .append('text')
    .attr('x', width / 2)
    .attr('y', 18)
    .attr('text-anchor', 'middle')
    .style('font-size', '12.5px')
    .style('font-weight', '600')
    .style('fill', 'var(--color-text, #1A1A1A)')
    .text(panel.title);
  // Mini-colorbar
  const cbW = 12;
  const cbH = innerH;
  const cbX = innerW + 6;
  const nBands = 20;
  for (let k = 0; k < nBands; k++) {
    const v = domain[0] + ((domain[1] - domain[0]) * (nBands - 1 - k)) / (nBands - 1);
    g.append('rect')
      .attr('x', cbX)
      .attr('y', (k / nBands) * cbH)
      .attr('width', cbW)
      .attr('height', cbH / nBands + 1)
      .style('fill', colorScale(v));
  }
  const cbScale = d3.scaleLinear().domain(domain).range([cbH, 0]);
  g.append('g').attr('transform', `translate(${cbX + cbW}, 0)`).call(d3.axisRight(cbScale).ticks(4))
    .selectAll('text').style('fill', 'var(--color-text, #1A1A1A)').style('font-size', '9.5px');
}

export default function LCPPosteriorExplorer() {
  const [displayRho, setDisplayRho] = useState(DEFAULT_RHO);
  const [committedRho, setCommittedRho] = useState(DEFAULT_RHO);
  const [displayN, setDisplayN] = useState(DEFAULT_N);
  const [committedN, setCommittedN] = useState(DEFAULT_N);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 800;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.floor((containerWidth - 16) / 2);
  const panelHeight = isMobile ? 180 : 220;

  const result = useMemo(() => runLCP(committedRho, committedN), [committedRho, committedN]);

  const refA = useRef<SVGSVGElement | null>(null);
  const refB = useRef<SVGSVGElement | null>(null);
  const refC = useRef<SVGSVGElement | null>(null);
  const refD = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    // Use a shared log-intensity domain for (a) and (c) so they're comparable.
    const intMin = Math.min(...result.xTrue, ...result.postMean);
    const intMax = Math.max(...result.xTrue, ...result.postMean);
    const intDomain: [number, number] = [intMin, intMax];
    if (refA.current)
      drawHeatmap(refA.current, panelWidth, panelHeight, {
        values: result.xTrue,
        title: 'True log-intensity x_true',
        domain: intDomain,
      });
    if (refB.current)
      drawHeatmap(refB.current, panelWidth, panelHeight, {
        values: result.yObs,
        title: 'Observed counts y',
        textOverlay: (v) => String(v),
      });
    if (refC.current)
      drawHeatmap(refC.current, panelWidth, panelHeight, {
        values: result.postMean,
        title: 'RMHMC posterior mean',
        domain: intDomain,
      });
    if (refD.current)
      drawHeatmap(refD.current, panelWidth, panelHeight, {
        values: result.postStd,
        title: 'RMHMC posterior std',
      });
  }, [result, panelWidth, panelHeight]);

  // Quality metric
  let dist2 = 0;
  for (let i = 0; i < D; i++) dist2 += (result.postMean[i] - result.xTrue[i]) ** 2;
  const recoveryDist = Math.sqrt(dist2);
  const meanStd = result.postStd.reduce((a, b) => a + b, 0) / D;

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 10: LCP posterior summary on a 5x5 grid"
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
          <span>kernel length-scale ρ:</span>
          <input
            type="range"
            min={0.1}
            max={0.5}
            step={0.02}
            value={displayRho}
            onChange={(e) => setDisplayRho(Number(e.target.value))}
            onMouseUp={(e) => setCommittedRho(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedRho(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedRho(Number((e.target as HTMLInputElement).value))}
            style={{ width: '140px' }}
            aria-label="Kernel length-scale rho"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>ρ = {displayRho.toFixed(2)}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>RMHMC samples:</span>
          <input
            type="range"
            min={100}
            max={1000}
            step={50}
            value={displayN}
            onChange={(e) => setDisplayN(Number(e.target.value))}
            onMouseUp={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedN(Number((e.target as HTMLInputElement).value))}
            style={{ width: '140px' }}
            aria-label="Number of RMHMC samples"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>n = {displayN}</span>
        </label>
        <span style={{ color: 'var(--color-text-secondary, #6B6B6B)', fontSize: '0.85rem', fontVariantNumeric: 'tabular-nums' }}>
          acc={result.acceptance.toFixed(3)}, div={result.divergences}, wall={result.wall.toFixed(2)}s,
          ‖mean − x_true‖={recoveryDist.toFixed(2)}, mean(std)={meanStd.toFixed(3)}
        </span>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
          gap: '0.5rem',
          justifyItems: 'center',
        }}
      >
        <svg ref={refA} width={panelWidth} height={panelHeight} role="img" aria-label="True log-intensity heatmap" />
        <svg ref={refB} width={panelWidth} height={panelHeight} role="img" aria-label="Observed counts heatmap" />
        <svg ref={refC} width={panelWidth} height={panelHeight} role="img" aria-label="RMHMC posterior mean heatmap" />
        <svg ref={refD} width={panelWidth} height={panelHeight} role="img" aria-label="RMHMC posterior standard deviation heatmap" />
      </div>
      <figcaption
        style={{
          marginTop: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-text-secondary, #6B6B6B)',
          textAlign: 'center',
        }}
      >
        Figure 10. Log-Gaussian Cox process on a 5×5 grid (d = 25). Top row: synthetic
        ground truth x<sub>true</sub> and the Poisson counts y it generated. Bottom row:
        RMHMC posterior mean (same color scale as x<sub>true</sub>, so cells should match
        visually) and pointwise posterior standard deviation. Shrinking ρ tightens the
        kernel prior and lets x track local data more aggressively; growing it smooths
        across cells.
      </figcaption>
    </figure>
  );
}

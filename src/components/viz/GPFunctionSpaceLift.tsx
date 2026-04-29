import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gaussianSampler,
  kernelSE,
  mulberry32,
  paletteSamples,
  sampleGPPrior,
} from './shared/gaussian-processes';
import {
  phiPoly,
  phiRBF,
  RBF_CENTERS,
  linspace,
} from '../../data/gaussian-processes-data';

// =============================================================================
// GPFunctionSpaceLift — interactive companion to §1's three-panel function-
// space-lift figure. Reproduces notebook cell 4 (figures/01_function_space_
// motivation.png) and adds three sliders that expose the v2 enhancements
// listed in brief §1:
//
//   Panel (a) — polynomial basis, D ∈ [1, 8]
//   Panel (b) — RBF basis, K ∈ [2, 12] equally-spaced centers across [-3, 3]
//   Panel (c) — GP prior with SE kernel, ℓ ∈ [0.2, 2.0]
//
// Five prior samples per panel; per-panel RNG seed is fixed so the underlying
// random latent is stable across slider movements (the user sees "the same
// random draws at different parameters", not "different draws").
//
// Static fallback: public/images/topics/gaussian-processes/01_function_space_motivation.png
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 640;
const N_GRID = 200;
const N_SAMPLES = 5;
const Y_LIM: [number, number] = [-4, 4];

export default function GPFunctionSpaceLift() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  // Slider state — one per panel; sliders co-located in the controls bar.
  const [polyD, setPolyD] = useState(4);
  const [rbfK, setRbfK] = useState(6);
  const [gpLengthscale, setGpLengthscale] = useState(0.7);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const xGrid = useMemo(() => linspace(-3, 3, N_GRID), []);

  // Panel (a) — polynomial samples. Random weights are drawn once per render
  // with a fixed seed so changing D extends the basis without changing the
  // first D − 1 coefficients (smooth visual continuity).
  const polySamples = useMemo(() => {
    const rng = mulberry32(42);
    const gauss = gaussianSampler(rng);
    // Draw N_SAMPLES * MAX_D coefficients up front; truncate to D
    const MAX_D = 8;
    const W: number[][] = [];
    for (let s = 0; s < N_SAMPLES; s++) {
      const wRow = new Array<number>(MAX_D);
      for (let k = 0; k < MAX_D; k++) wRow[k] = gauss();
      W.push(wRow);
    }
    const phi = phiPoly(xGrid, polyD);
    return W.map((w) => xGrid.map((_, i) => {
      let s = 0;
      for (let k = 0; k < polyD; k++) s += w[k] * phi[i][k];
      return s;
    }));
  }, [polyD, xGrid]);

  // Panel (b) — RBF samples. Centers are linspace(-3, 3, K).
  const rbfSamples = useMemo(() => {
    const rng = mulberry32(43);
    const gauss = gaussianSampler(rng);
    const MAX_K = 12;
    const W: number[][] = [];
    for (let s = 0; s < N_SAMPLES; s++) {
      const wRow = new Array<number>(MAX_K);
      for (let k = 0; k < MAX_K; k++) wRow[k] = gauss();
      W.push(wRow);
    }
    const centers = rbfK === RBF_CENTERS.length
      ? RBF_CENTERS.slice()
      : linspace(-3, 3, rbfK);
    const omega = 0.7;
    const phi = phiRBF(xGrid, centers, omega);
    return W.map((w) => xGrid.map((_, i) => {
      let s = 0;
      for (let k = 0; k < rbfK; k++) s += w[k] * phi[i][k];
      return s;
    }));
  }, [rbfK, xGrid]);

  // Panel (c) — GP samples via Cholesky. Re-seed once; same Z latent for all
  // ℓ, so the user sees "the same draws at different lengthscales".
  const gpSamples = useMemo(() => {
    const rng = mulberry32(44);
    const K = kernelSE(xGrid, xGrid, 1.0, gpLengthscale);
    return sampleGPPrior(K, N_SAMPLES, rng);
  }, [gpLengthscale, xGrid]);

  // ---------------------------------------------------------------------------
  // Render functions for each panel — three independent SVGs sharing scales.
  // ---------------------------------------------------------------------------

  const panelW = useMemo(() => {
    if (w <= 0) return 0;
    return isMobile ? w : Math.floor((w - 24) / 3);
  }, [w, isMobile]);

  const renderPanel = (samples: number[][], title: string) =>
    (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const margin = { top: 28, right: 12, bottom: 32, left: 38 };
      const innerW = panelW - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-3, 3]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(Y_LIM).range([innerH, 0]);

      // Zero line
      g.append('line')
        .attr('x1', 0).attr('x2', innerW)
        .attr('y1', yScale(0)).attr('y2', yScale(0))
        .style('stroke', 'var(--color-border)')
        .style('stroke-dasharray', '2 3')
        .style('stroke-width', 0.8);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      // Axis labels
      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 26)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '10px')
        .text('x');
      g.append('text')
        .attr('x', -28).attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '10px')
        .attr('transform', `rotate(-90,-28,${innerH / 2})`)
        .text('f(x)');

      // Sample paths
      const line = d3.line<number>()
        .x((_, i) => xScale(xGrid[i]))
        .y((d) => yScale(d));
      for (let s = 0; s < samples.length; s++) {
        g.append('path')
          .datum(samples[s])
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', paletteSamples[s % paletteSamples.length])
          .style('stroke-width', 1.4)
          .style('opacity', 0.9);
      }

      // Title
      svg.append('text')
        .attr('x', panelW / 2).attr('y', 16)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text(title);
    };

  const polyRef = useD3<SVGSVGElement>(
    renderPanel(polySamples, `(a) polynomial basis, D = ${polyD}`),
    [panelW, polySamples, polyD],
  );
  const rbfRef = useD3<SVGSVGElement>(
    renderPanel(rbfSamples, `(b) RBF basis, K = ${rbfK}`),
    [panelW, rbfSamples, rbfK],
  );
  const gpRef = useD3<SVGSVGElement>(
    renderPanel(gpSamples, `(c) GP prior, SE kernel, ℓ = ${gpLengthscale.toFixed(2)}`),
    [panelW, gpSamples, gpLengthscale],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
    >
      <div
        className="flex flex-wrap items-center gap-4 mb-4 text-sm"
        style={{ flexDirection: isMobile ? 'column' : 'row', alignItems: isMobile ? 'stretch' : 'center' }}
      >
        <label className="flex items-center gap-2 flex-1 min-w-[180px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">D (poly): {polyD}</span>
          <input
            type="range" min={1} max={8} step={1} value={polyD}
            onChange={(e) => setPolyD(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2 flex-1 min-w-[180px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">K (RBF): {rbfK}</span>
          <input
            type="range" min={2} max={12} step={1} value={rbfK}
            onChange={(e) => setRbfK(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2 flex-1 min-w-[180px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">ℓ (GP): {gpLengthscale.toFixed(2)}</span>
          <input
            type="range" min={0.2} max={2.0} step={0.05} value={gpLengthscale}
            onChange={(e) => setGpLengthscale(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
      </div>

      <div
        className="flex gap-2"
        style={{ flexDirection: isMobile ? 'column' : 'row' }}
      >
        <svg ref={polyRef} width={panelW} height={HEIGHT} />
        <svg ref={rbfRef} width={panelW} height={HEIGHT} />
        <svg ref={gpRef} width={panelW} height={HEIGHT} />
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Reading left to right is the structural lift §1 develops: the same plot
        conventions, increasing functional richness. The polynomial family is
        D-dimensional; the RBF family is K-dimensional; the GP family is
        infinite-dimensional. Drag the sliders to feel each axis.
      </p>
    </div>
  );
}

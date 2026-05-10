import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  kGaussian,
  localLinear,
  mTrueUni,
  mulberry32,
  nadarayaWatson,
  paletteKR,
  sampleToyUni,
} from './shared/kernel-regression';

// =============================================================================
// BoundaryBiasDiagnostic — interactive companion to §8.2's local-linear
// boundary-fix figure. Two stacked panels:
//
//   Top:    NW vs LL fits on the §1 toy at the chosen h. Boundary regions
//           [0, h] and [1-h, 1] are shaded; the reader sees NW pull dramatically
//           away from m near the boundary while LL stays close.
//
//   Bottom: empirical bias as a function of x from B replicates, NW vs LL.
//           The NW curve diverges sharply for x < h; LL is uniformly small.
//
// Numerical anchor: at h = 0.05, B = 500, notebook §8.2 reports
//   |bias_NW(0.005)| = 0.247, |bias_LL(0.005)| = 0.011  (NW/LL ratio = 22.6x)
//   |bias_NW(0.245)| = |bias_LL(0.245)| = 0.050         (interior matches)
//
// Static fallback: public/images/topics/kernel-regression/17_local_linear_boundary.png
//                  public/images/topics/kernel-regression/18_ll_vs_nw_boundary_bias.png
// =============================================================================

const TOP_HEIGHT = 320;
const BOTTOM_HEIGHT = 240;
const SM_BREAKPOINT = 640;
const N = 200;
const SIGMA = 0.2;

export default function BoundaryBiasDiagnostic() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  // Display state (cheap top-panel single-fit) vs committed state (heavy
  // B-replicate MC sweep for the bottom panel). The top panel reads `hDisplay`
  // so dragging stays responsive; the bottom panel reads `hCommitted` and `B`
  // and only updates on slider release. Pattern mirrors
  // FiniteSampleBiasExplorer / SGLDBatchSizeExplorer in this repo.
  const [logHDisplay, setLogHDisplay] = useState(Math.log10(0.05));
  const [logHCommitted, setLogHCommitted] = useState(Math.log10(0.05));
  const [BDisplay, setBDisplay] = useState(150);
  const [BCommitted, setBCommitted] = useState(150);
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;
  const hDisplay = Math.pow(10, logHDisplay);
  const hCommitted = Math.pow(10, logHCommitted);

  // Single canonical sample for the top panel.
  const { X, Y } = useMemo(() => {
    const rng = mulberry32(42);
    return sampleToyUni(N, SIGMA, rng);
  }, []);

  const xGrid = useMemo(() => {
    const arr = new Float64Array(401);
    for (let i = 0; i < arr.length; i++) arr[i] = i / 400;
    return arr;
  }, []);
  const mGrid = useMemo(() => {
    const arr = new Float64Array(xGrid.length);
    for (let i = 0; i < xGrid.length; i++) arr[i] = mTrueUni(xGrid[i]);
    return arr;
  }, [xGrid]);

  // Top panel: cheap single-fit at the display bandwidth — updates live.
  const { mNw, mLl } = useMemo(() => {
    return {
      mNw: nadarayaWatson(X, Y, xGrid, hDisplay, kGaussian),
      mLl: localLinear(X, Y, xGrid, hDisplay, kGaussian),
    };
  }, [X, Y, xGrid, hDisplay]);

  // Bottom panel: heavy MC sweep — only recomputes on slider release.
  const { biasNw, biasLl } = useMemo(() => {
    const G = xGrid.length;
    const sumNw = new Float64Array(G);
    const sumLl = new Float64Array(G);
    const rng = mulberry32(20260601);
    for (let b = 0; b < BCommitted; b++) {
      const { X: Xb, Y: Yb } = sampleToyUni(N, SIGMA, rng);
      const fitNw = nadarayaWatson(Xb, Yb, xGrid, hCommitted, kGaussian);
      const fitLl = localLinear(Xb, Yb, xGrid, hCommitted, kGaussian);
      for (let j = 0; j < G; j++) {
        sumNw[j] += fitNw[j];
        sumLl[j] += fitLl[j];
      }
    }
    const bNw = new Float64Array(G);
    const bLl = new Float64Array(G);
    for (let j = 0; j < G; j++) {
      bNw[j] = sumNw[j] / BCommitted - mGrid[j];
      bLl[j] = sumLl[j] / BCommitted - mGrid[j];
    }
    return { biasNw: bNw, biasLl: bLl };
  }, [xGrid, mGrid, hCommitted, BCommitted]);

  // Top-panel render — NW vs LL fits with boundary shading.
  const topRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!topRef.current || w <= 0) return;
    const svg = d3.select(topRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 16, bottom: 38, left: 50 };
    const innerW = w - margin.left - margin.right;
    const innerH = TOP_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([-1.6, 1.7]).range([innerH, 0]);

    // Boundary shading — top panel uses hDisplay (live).
    g.append('rect')
      .attr('x', xScale(0))
      .attr('y', 0)
      .attr('width', xScale(hDisplay))
      .attr('height', innerH)
      .style('fill', paletteKR.truth)
      .style('opacity', 0.08);
    g.append('rect')
      .attr('x', xScale(1 - hDisplay))
      .attr('y', 0)
      .attr('width', xScale(1) - xScale(1 - hDisplay))
      .attr('height', innerH)
      .style('fill', paletteKR.truth)
      .style('opacity', 0.08);

    g.selectAll('circle.data')
      .data(Array.from(X))
      .enter()
      .append('circle')
      .attr('class', 'data')
      .attr('cx', (d) => xScale(d))
      .attr('cy', (_, i) => yScale(Y[i]))
      .attr('r', 2.4)
      .style('fill', paletteKR.data)
      .style('opacity', 0.35);

    const trueLine = d3
      .line<number>()
      .x((_, i) => xScale(xGrid[i]))
      .y((d) => yScale(d));
    g.append('path')
      .datum(Array.from(mGrid))
      .attr('d', trueLine)
      .style('fill', 'none')
      .style('stroke', paletteKR.truth)
      .style('stroke-width', 1.6);

    const nwLine = d3
      .line<number>()
      .defined((d) => Number.isFinite(d))
      .x((_, i) => xScale(xGrid[i]))
      .y((d) => yScale(d));
    g.append('path')
      .datum(Array.from(mNw))
      .attr('d', nwLine)
      .style('fill', 'none')
      .style('stroke', paletteKR.band)
      .style('stroke-width', 1.6);

    g.append('path')
      .datum(Array.from(mLl))
      .attr('d', nwLine)
      .style('fill', 'none')
      .style('stroke', paletteKR.posterior)
      .style('stroke-width', 1.8);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(6))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .text('x');
    g.append('text')
      .attr('x', -34)
      .attr('y', innerH / 2)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('transform', `rotate(-90,-34,${innerH / 2})`)
      .text('y');

    svg
      .append('text')
      .attr('x', w / 2)
      .attr('y', 18)
      .style('fill', 'var(--color-text)')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`NW vs local-linear,  h = ${hDisplay.toFixed(3)} (boundary regions shaded)`);

    // Legend.
    const legendG = g.append('g').attr('transform', `translate(${innerW - 130}, 8)`);
    const items = [
      { color: paletteKR.truth, label: 'true m(x)' },
      { color: paletteKR.band, label: 'NW' },
      { color: paletteKR.posterior, label: 'local-linear' },
    ];
    items.forEach((it, i) => {
      const yOff = i * 14;
      legendG
        .append('line')
        .attr('x1', 0)
        .attr('y1', yOff)
        .attr('x2', 16)
        .attr('y2', yOff)
        .style('stroke', it.color)
        .style('stroke-width', 1.6);
      legendG
        .append('text')
        .attr('x', 22)
        .attr('y', yOff + 3)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text(it.label);
    });
  }, [w, X, Y, xGrid, mGrid, mNw, mLl, hDisplay]);

  // Bottom-panel render — empirical bias.
  const bottomRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!bottomRef.current || w <= 0) return;
    const svg = d3.select(bottomRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 16, bottom: 38, left: 50 };
    const innerW = w - margin.left - margin.right;
    const innerH = BOTTOM_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
    const allBias = [...Array.from(biasNw), ...Array.from(biasLl)].filter(Number.isFinite);
    const yMax = Math.max(0.4, Math.max(...allBias.map(Math.abs)) * 1.15);
    const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerH, 0]);

    // Zero line.
    g.append('line')
      .attr('x1', 0)
      .attr('y1', yScale(0))
      .attr('x2', innerW)
      .attr('y2', yScale(0))
      .style('stroke', 'var(--color-border)')
      .style('stroke-width', 1);

    // Boundary shading — bottom panel uses hCommitted (matches MC sweep).
    g.append('rect')
      .attr('x', xScale(0))
      .attr('y', 0)
      .attr('width', xScale(hCommitted))
      .attr('height', innerH)
      .style('fill', paletteKR.truth)
      .style('opacity', 0.08);
    g.append('rect')
      .attr('x', xScale(1 - hCommitted))
      .attr('y', 0)
      .attr('width', xScale(1) - xScale(1 - hCommitted))
      .attr('height', innerH)
      .style('fill', paletteKR.truth)
      .style('opacity', 0.08);

    const line = d3
      .line<number>()
      .defined((d) => Number.isFinite(d))
      .x((_, i) => xScale(xGrid[i]))
      .y((d) => yScale(d));

    g.append('path')
      .datum(Array.from(biasNw))
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', paletteKR.band)
      .style('stroke-width', 1.6);
    g.append('path')
      .datum(Array.from(biasLl))
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', paletteKR.posterior)
      .style('stroke-width', 1.6);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .text('x');
    g.append('text')
      .attr('x', -34)
      .attr('y', innerH / 2)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('transform', `rotate(-90,-34,${innerH / 2})`)
      .text('bias(x)');

    svg
      .append('text')
      .attr('x', w / 2)
      .attr('y', 18)
      .style('fill', 'var(--color-text)')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`Empirical bias vs x,  B = ${BCommitted} replicates`);
  }, [w, xGrid, biasNw, biasLl, hCommitted, BCommitted]);

  // Read-out: bias ratio at boundary x = 0.005.
  const boundaryIdx = useMemo(() => {
    let idx = 0;
    let bestGap = Infinity;
    for (let j = 0; j < xGrid.length; j++) {
      const gap = Math.abs(xGrid[j] - 0.005);
      if (gap < bestGap) {
        bestGap = gap;
        idx = j;
      }
    }
    return idx;
  }, [xGrid]);
  const ratio = useMemo(() => {
    const bn = Math.abs(biasNw[boundaryIdx]);
    const bl = Math.abs(biasLl[boundaryIdx]);
    return bl > 1e-12 ? bn / bl : Infinity;
  }, [biasNw, biasLl, boundaryIdx]);

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
    >
      <div
        className="flex flex-wrap items-center gap-4 mb-4 text-sm"
        style={{ flexDirection: isMobile ? 'column' : 'row', alignItems: isMobile ? 'stretch' : 'center' }}
      >
        <label className="flex items-center gap-2 flex-1 min-w-[200px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">
            bandwidth h: {hDisplay.toFixed(3)}
          </span>
          <input
            type="range"
            min={Math.log10(0.01)}
            max={Math.log10(0.2)}
            step={0.02}
            value={logHDisplay}
            onChange={(e) => setLogHDisplay(Number(e.target.value))}
            onMouseUp={(e) => setLogHCommitted(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setLogHCommitted(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setLogHCommitted(Number((e.target as HTMLInputElement).value))}
            className="flex-1 accent-[var(--color-accent)]"
            aria-label="Bandwidth h"
          />
        </label>
        <label className="flex items-center gap-2 flex-1 min-w-[180px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">B: {BDisplay}</span>
          <input
            type="range"
            min={50}
            max={400}
            step={25}
            value={BDisplay}
            onChange={(e) => setBDisplay(Number(e.target.value))}
            onMouseUp={(e) => setBCommitted(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setBCommitted(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setBCommitted(Number((e.target as HTMLInputElement).value))}
            className="flex-1 accent-[var(--color-accent)]"
            aria-label="Number of replicates B"
          />
        </label>
        <span className="ml-auto text-xs text-[var(--color-text-secondary)] font-mono">
          |bias_NW / bias_LL|@x≈0.005 = {Number.isFinite(ratio) ? `${ratio.toFixed(1)}x` : '∞'}
        </span>
      </div>

      <svg ref={topRef} width={w} height={TOP_HEIGHT} />
      <svg ref={bottomRef} width={w} height={BOTTOM_HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        At interior x both estimators have O(h²) bias and visibly track each other. Inside the
        shaded boundary regions the NW curve breaks down to O(h) — the kernel mass falls off the
        support and the estimator pulls toward the side where data exists. Local-linear fixes
        this by reproducing both constants AND linear functions exactly under any kernel weights;
        boundary bias stays O(h²) uniformly.
      </p>
    </div>
  );
}

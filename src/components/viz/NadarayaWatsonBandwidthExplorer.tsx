import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  hStarAmiseUni,
  kGaussian,
  mTrueUni,
  mulberry32,
  nadarayaWatson,
  paletteKR,
  sampleToyUni,
} from './shared/kernel-regression';

// =============================================================================
// NadarayaWatsonBandwidthExplorer — interactive companion to §2.2's three-
// bandwidth NW figure. Reproduces the §1 toy scatter, with a smooth bandwidth
// slider h ∈ [0.005, 0.5] and a sample-size scrubber n ∈ {50, 100, 200, 500}.
// Reader sees: under-smoothed h ≈ 0.005 → near-interpolating spike-train,
// over-smoothed h ≈ 0.5 → near-constant flat line, well-smoothed h ≈ 0.05 →
// clean recovery of the true sinusoid.
//
// Numerical anchor: at h = 0.05, n = 200, seed = 42, the notebook §2.2 reports
// grid-MSE(NW) ≈ 0.008 vs grid-MSE(degree-1 polynomial) ≈ 0.198. The read-out
// at the bottom of this viz prints grid-MSE live as the slider moves, and at
// the canonical settings it lands within ~10% of the notebook value (different
// RNG ⇒ different MC sample, same DGP, same bias-variance trade-off).
//
// Static fallback: public/images/topics/kernel-regression/05_nw_three_bandwidths.png
// =============================================================================

const HEIGHT = 420;
const SM_BREAKPOINT = 640;
const SIGMA = 0.2;
const SEED = 42;
const N_OPTIONS = [50, 100, 200, 500];

export default function NadarayaWatsonBandwidthExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [logH, setLogH] = useState(Math.log10(0.05)); // log-space slider for smooth scrubbing
  const [nIdx, setNIdx] = useState(2); // index into N_OPTIONS, default 200

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;
  const h = Math.pow(10, logH);
  const N = N_OPTIONS[nIdx];

  // Resample whenever N changes — the §1 canonical seed=42 chain stays consistent
  // for a given N, so the user sees a stable scatter at each n.
  const { X, Y } = useMemo(() => {
    const rng = mulberry32(SEED);
    return sampleToyUni(N, SIGMA, rng);
  }, [N]);

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

  const { mhat, gridMse, regime } = useMemo(() => {
    const fit = nadarayaWatson(X, Y, xGrid, h, kGaussian);
    let mse = 0;
    let valid = 0;
    for (let i = 0; i < fit.length; i++) {
      if (Number.isFinite(fit[i])) {
        const r = fit[i] - mGrid[i];
        mse += r * r;
        valid++;
      }
    }
    mse = valid > 0 ? mse / valid : NaN;
    let label: string;
    if (h < 0.015) label = 'under-smoothed';
    else if (h > 0.18) label = 'over-smoothed';
    else label = 'well-smoothed';
    return { mhat: fit, gridMse: mse, regime: label };
  }, [X, Y, xGrid, mGrid, h]);

  // AMISE-optimal h^* for the current n (reference marker on the slider).
  const hStar = useMemo(() => hStarAmiseUni(N, SIGMA), [N]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 28, right: 16, bottom: 38, left: 50 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-1.6, 1.7]).range([innerH, 0]);

      // Data scatter.
      g.selectAll('circle.data')
        .data(Array.from(X))
        .enter()
        .append('circle')
        .attr('class', 'data')
        .attr('cx', (d) => xScale(d))
        .attr('cy', (_, i) => yScale(Y[i]))
        .attr('r', N <= 100 ? 3 : 2.4)
        .style('fill', paletteKR.data)
        .style('opacity', 0.4);

      // True m(x).
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

      // NW estimate.
      const fitLine = d3
        .line<number>()
        .defined((d) => Number.isFinite(d))
        .x((_, i) => xScale(xGrid[i]))
        .y((d) => yScale(d));
      g.append('path')
        .datum(Array.from(mhat))
        .attr('d', fitLine)
        .style('fill', 'none')
        .style('stroke', paletteKR.posterior)
        .style('stroke-width', 1.8);

      // Axes.
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
        .text(`Nadaraya–Watson, n = ${N}, h = ${h.toFixed(3)}  (${regime})`);

      // Legend.
      const legendX = innerW - 130;
      const legendY = 8;
      const legendG = g.append('g').attr('transform', `translate(${legendX}, ${legendY})`);
      const legendItems = [
        { color: paletteKR.posterior, label: 'NW estimate' },
        { color: paletteKR.truth, label: 'true m(x)' },
        { color: paletteKR.data, label: 'data' },
      ];
      legendItems.forEach((it, i) => {
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
    },
    [w, X, Y, xGrid, mGrid, mhat, h, N, regime],
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
        <label className="flex items-center gap-2 flex-1 min-w-[220px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">
            bandwidth h: {h.toFixed(3)}
          </span>
          <input
            type="range"
            min={Math.log10(0.005)}
            max={Math.log10(0.5)}
            step={0.01}
            value={logH}
            onChange={(e) => setLogH(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">n: {N}</span>
          <input
            type="range"
            min={0}
            max={N_OPTIONS.length - 1}
            step={1}
            value={nIdx}
            onChange={(e) => setNIdx(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <button
          type="button"
          onClick={() => setLogH(Math.log10(hStar))}
          className="px-2 py-1 rounded border border-[var(--color-border)] text-xs text-[var(--color-text-secondary)] hover:bg-[var(--color-bg)]"
          title="Snap to AMISE-optimal bandwidth"
        >
          h ← h*({hStar.toFixed(3)})
        </button>
        <span className="ml-auto text-xs text-[var(--color-text-secondary)] font-mono">
          grid-MSE = {Number.isFinite(gridMse) ? gridMse.toFixed(4) : '—'}
        </span>
      </div>

      <svg ref={renderRef} width={w} height={HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Drag h smoothly across log-space (0.005 to 0.5). Watch the bias-variance trade-off in
        caricature: under-smoothing chases noise, over-smoothing flattens the sinusoid. The h*
        button snaps to the AMISE-optimal bandwidth for the current n; at n = 200 that's h* ≈
        0.037, where grid-MSE bottoms out near 0.008.
      </p>
    </div>
  );
}

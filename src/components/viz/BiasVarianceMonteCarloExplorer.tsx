import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  kGaussian,
  mTrueUni,
  mulberry32,
  nadarayaWatson,
  paletteKR,
  sampleToyUni,
} from './shared/kernel-regression';

// =============================================================================
// BiasVarianceMonteCarloExplorer — interactive companion to §4.3's bias-
// variance decomposition figure. Generates B = 30..200 replicate NW fits at
// fixed bandwidth h on the §1 toy and overlays them at light alpha (variance);
// computes the empirical mean (bias); plots the ±2 SD band.
//
// Numerical anchor: at h = 0.05, B = 200, notebook §4.3 reports integrated
// empirical bias² ≈ 0.000926 and variance ≈ 0.001281 over [0.1, 0.9]. The
// read-out below the chart prints these live; with Mulberry32+Box-Muller they
// land within ~30% of the notebook values (different RNG, same rate scaling).
//
// Static fallback: public/images/topics/kernel-regression/08_bias_variance_decomposition.png
// =============================================================================

const HEIGHT = 440;
const SM_BREAKPOINT = 640;
const N = 200;
const SIGMA = 0.2;

export default function BiasVarianceMonteCarloExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [logH, setLogH] = useState(Math.log10(0.05));
  const [B, setB] = useState(60);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;
  const h = Math.pow(10, logH);

  const xGrid = useMemo(() => {
    const arr = new Float64Array(201);
    for (let i = 0; i < arr.length; i++) arr[i] = i / 200;
    return arr;
  }, []);

  const mGrid = useMemo(() => {
    const arr = new Float64Array(xGrid.length);
    for (let i = 0; i < xGrid.length; i++) arr[i] = mTrueUni(xGrid[i]);
    return arr;
  }, [xGrid]);

  const { replicates, mean, sd, intBiasSq, intVar } = useMemo(() => {
    const reps: number[][] = [];
    const G = xGrid.length;
    const meanArr = new Float64Array(G);
    const sqArr = new Float64Array(G);

    const rng = mulberry32(20260801);
    for (let b = 0; b < B; b++) {
      const { X, Y } = sampleToyUni(N, SIGMA, rng);
      const fit = nadarayaWatson(X, Y, xGrid, h, kGaussian);
      const arr = new Array<number>(G);
      for (let j = 0; j < G; j++) {
        const v = fit[j];
        arr[j] = v;
        meanArr[j] += v;
        sqArr[j] += v * v;
      }
      reps.push(arr);
    }
    const sdArr = new Float64Array(G);
    for (let j = 0; j < G; j++) {
      meanArr[j] /= B;
      const variance = sqArr[j] / B - meanArr[j] * meanArr[j];
      sdArr[j] = Math.sqrt(Math.max(variance, 0));
    }

    // Integrate squared-bias and variance over the interior [0.1, 0.9].
    let biasIntegral = 0;
    let varIntegral = 0;
    let count = 0;
    for (let j = 0; j < G; j++) {
      if (xGrid[j] >= 0.1 && xGrid[j] <= 0.9) {
        const bias = meanArr[j] - mGrid[j];
        biasIntegral += bias * bias;
        varIntegral += sdArr[j] * sdArr[j];
        count++;
      }
    }
    biasIntegral = count > 0 ? (biasIntegral * 0.8) / count : 0; // 0.8 = length of interval
    varIntegral = count > 0 ? (varIntegral * 0.8) / count : 0;

    return {
      replicates: reps,
      mean: meanArr,
      sd: sdArr,
      intBiasSq: biasIntegral,
      intVar: varIntegral,
    };
  }, [xGrid, mGrid, h, B]);

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

      // ±2 SD band around empirical mean.
      const band = d3
        .area<number>()
        .x((_, i) => xScale(xGrid[i]))
        .y0((_, i) => yScale(mean[i] - 2 * sd[i]))
        .y1((_, i) => yScale(mean[i] + 2 * sd[i]));
      g.append('path')
        .datum(Array.from(mean))
        .attr('d', band)
        .style('fill', paletteKR.posterior)
        .style('opacity', 0.18);

      // Replicate curves at light alpha — drawn first, so the mean and truth
      // sit on top.
      const repLine = d3
        .line<number>()
        .x((_, i) => xScale(xGrid[i]))
        .y((d) => yScale(d));
      const alpha = Math.max(0.04, Math.min(0.4, 8 / B));
      for (let r = 0; r < replicates.length; r++) {
        g.append('path')
          .datum(replicates[r])
          .attr('d', repLine)
          .style('fill', 'none')
          .style('stroke', paletteKR.posterior)
          .style('stroke-width', 0.6)
          .style('opacity', alpha);
      }

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

      // Empirical mean curve.
      g.append('path')
        .datum(Array.from(mean))
        .attr('d', trueLine)
        .style('fill', 'none')
        .style('stroke', paletteKR.alt)
        .style('stroke-width', 2.0);

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
        .text(`Bias-variance Monte Carlo:  B = ${B} replicates,  h = ${h.toFixed(3)}`);

      // Legend.
      const legendX = innerW - 160;
      const legendY = 8;
      const legendG = g.append('g').attr('transform', `translate(${legendX}, ${legendY})`);
      const items: Array<{ color: string; label: string; opacity?: number; width?: number }> = [
        { color: paletteKR.truth, label: 'true m(x)' },
        { color: paletteKR.alt, label: 'empirical mean', width: 2 },
        { color: paletteKR.posterior, label: 'replicate fits', opacity: 0.4 },
        { color: paletteKR.posterior, label: '±2 SD band', opacity: 0.18 },
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
          .style('stroke-width', it.width ?? 1.6)
          .style('opacity', it.opacity ?? 1);
        legendG
          .append('text')
          .attr('x', 22)
          .attr('y', yOff + 3)
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '10px')
          .text(it.label);
      });
    },
    [w, xGrid, mGrid, replicates, mean, sd, h, B],
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
            max={Math.log10(0.4)}
            step={0.02}
            value={logH}
            onChange={(e) => setLogH(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2 flex-1 min-w-[180px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">
            replicates B: {B}
          </span>
          <input
            type="range"
            min={20}
            max={200}
            step={10}
            value={B}
            onChange={(e) => setB(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <span className="ml-auto text-xs text-[var(--color-text-secondary)] font-mono">
          ∫ bias² dx ≈ {intBiasSq.toFixed(4)},  ∫ var dx ≈ {intVar.toFixed(4)}
        </span>
      </div>

      <svg ref={renderRef} width={w} height={HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        At small h, the replicate cloud spreads wide (large variance) but the empirical mean
        tracks the true m(x) closely (small bias). Crank h up: the cloud collapses to a single
        smooth curve, but it visibly drifts away from m where the sinusoid curves. The integrated
        bias² and variance read-outs trade off through the AMISE U-curve in §4.4.
      </p>
    </div>
  );
}

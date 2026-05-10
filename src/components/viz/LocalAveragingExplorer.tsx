import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mTrueUni,
  mulberry32,
  paletteKR,
  sampleToyUni,
} from './shared/kernel-regression';

// =============================================================================
// LocalAveragingExplorer — interactive companion to §1.2's local-averaging
// figure. Reproduces the §1 toy scatter, with a draggable target x_0 and a
// band-half-width slider h_0; highlights points inside [x_0 - h_0, x_0 + h_0]
// and shows the box-average estimator hat m_box(x_0) as a horizontal segment
// against the true m(x_0) marker.
//
// Numerical anchor: with seed=42, n=200, sigma=0.2, h_0=0.05, the §1.2
// notebook printout shows |hat m_box(x_0) - m(x_0)| ≈ 0.04 to 0.08 at
// x_0 in {0.2, 0.5, 0.8} — the visualization is the dynamic version of that.
//
// Static fallback: public/images/topics/kernel-regression/02_local_averaging_three_snapshots.png
// =============================================================================

const HEIGHT = 380;
const SM_BREAKPOINT = 640;
const N = 200;
const SIGMA = 0.2;
const SEED = 42;

export default function LocalAveragingExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [x0, setX0] = useState(0.5);
  const [h0, setH0] = useState(0.05);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  // Sample once per session — these are the §1 canonical points.
  const { X, Y } = useMemo(() => {
    const rng = mulberry32(SEED);
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

  // Compute box-average and read-out at current (x0, h0).
  const { boxAverage, mTrueAtX0, nInside, sigmaSE } = useMemo(() => {
    let acc = 0;
    let count = 0;
    let sqAcc = 0;
    for (let i = 0; i < N; i++) {
      if (Math.abs(X[i] - x0) <= h0) {
        acc += Y[i];
        sqAcc += Y[i] * Y[i];
        count++;
      }
    }
    const avg = count > 0 ? acc / count : NaN;
    const variance = count > 1 ? sqAcc / count - avg * avg : SIGMA * SIGMA;
    const se = count > 0 ? Math.sqrt(Math.max(variance, 0) / count) : NaN;
    return {
      boxAverage: avg,
      mTrueAtX0: mTrueUni(x0),
      nInside: count,
      sigmaSE: se,
    };
  }, [X, Y, x0, h0]);

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

      // Vertical neighborhood band [x0 - h0, x0 + h0].
      const bandLo = Math.max(0, x0 - h0);
      const bandHi = Math.min(1, x0 + h0);
      g.append('rect')
        .attr('x', xScale(bandLo))
        .attr('y', 0)
        .attr('width', xScale(bandHi) - xScale(bandLo))
        .attr('height', innerH)
        .style('fill', paletteKR.band)
        .style('opacity', 0.12);

      // True m(x) curve (drawn under data).
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

      // Out-of-band data points (light gray).
      const outsideX: number[] = [];
      const outsideY: number[] = [];
      const insideX: number[] = [];
      const insideY: number[] = [];
      for (let i = 0; i < N; i++) {
        if (Math.abs(X[i] - x0) <= h0) {
          insideX.push(X[i]);
          insideY.push(Y[i]);
        } else {
          outsideX.push(X[i]);
          outsideY.push(Y[i]);
        }
      }
      g.selectAll('circle.out')
        .data(outsideX)
        .enter()
        .append('circle')
        .attr('class', 'out')
        .attr('cx', (d) => xScale(d))
        .attr('cy', (_, i) => yScale(outsideY[i]))
        .attr('r', 2.6)
        .style('fill', paletteKR.data)
        .style('opacity', 0.35);

      g.selectAll('circle.in')
        .data(insideX)
        .enter()
        .append('circle')
        .attr('class', 'in')
        .attr('cx', (d) => xScale(d))
        .attr('cy', (_, i) => yScale(insideY[i]))
        .attr('r', 3.6)
        .style('fill', paletteKR.band)
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', 0.8)
        .style('opacity', 0.95);

      // Horizontal local-average line across the band.
      if (Number.isFinite(boxAverage)) {
        g.append('line')
          .attr('x1', xScale(bandLo))
          .attr('y1', yScale(boxAverage))
          .attr('x2', xScale(bandHi))
          .attr('y2', yScale(boxAverage))
          .style('stroke', paletteKR.posterior)
          .style('stroke-width', 2.2);
      }

      // Vertical x_0 marker.
      g.append('line')
        .attr('x1', xScale(x0))
        .attr('y1', 0)
        .attr('x2', xScale(x0))
        .attr('y2', innerH)
        .style('stroke', paletteKR.posterior)
        .style('stroke-dasharray', '3 3')
        .style('stroke-width', 1.2)
        .style('opacity', 0.6);

      // True m(x_0) marker as a hollow ring at (x_0, m(x_0)).
      g.append('circle')
        .attr('cx', xScale(x0))
        .attr('cy', yScale(mTrueAtX0))
        .attr('r', 6)
        .style('fill', 'none')
        .style('stroke', paletteKR.truth)
        .style('stroke-width', 2.0);

      // Box-average marker as a solid disk at (x_0, hat m_box(x_0)).
      if (Number.isFinite(boxAverage)) {
        g.append('circle')
          .attr('cx', xScale(x0))
          .attr('cy', yScale(boxAverage))
          .attr('r', 5)
          .style('fill', paletteKR.posterior)
          .style('stroke', 'var(--color-bg)')
          .style('stroke-width', 1.2);
      }

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

      // Title with read-out.
      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 18)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text(
          `Local box average at x₀ = ${x0.toFixed(2)}, h₀ = ${h0.toFixed(3)}  (n inside = ${nInside})`,
        );

      // Legend.
      const legendX = innerW - 170;
      const legendY = 8;
      const legendG = g.append('g').attr('transform', `translate(${legendX}, ${legendY})`);
      const legendItems: Array<{ color: string; label: string; kind: 'line' | 'ring' | 'dot' }> = [
        { color: paletteKR.truth, label: 'true m(x)', kind: 'line' },
        { color: paletteKR.posterior, label: 'box average', kind: 'line' },
        { color: paletteKR.truth, label: 'true m(x₀)', kind: 'ring' },
        { color: paletteKR.band, label: 'neighborhood', kind: 'dot' },
      ];
      legendItems.forEach((it, i) => {
        const yOff = i * 14;
        if (it.kind === 'line') {
          legendG
            .append('line')
            .attr('x1', 0)
            .attr('y1', yOff)
            .attr('x2', 16)
            .attr('y2', yOff)
            .style('stroke', it.color)
            .style('stroke-width', 1.6);
        } else if (it.kind === 'ring') {
          legendG
            .append('circle')
            .attr('cx', 8)
            .attr('cy', yOff)
            .attr('r', 4)
            .style('fill', 'none')
            .style('stroke', it.color)
            .style('stroke-width', 1.6);
        } else {
          legendG
            .append('circle')
            .attr('cx', 8)
            .attr('cy', yOff)
            .attr('r', 3.6)
            .style('fill', it.color)
            .style('opacity', 0.95);
        }
        legendG
          .append('text')
          .attr('x', 22)
          .attr('y', yOff + 3)
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '10px')
          .text(it.label);
      });
    },
    [w, x0, h0, X, Y, xGrid, mGrid, boxAverage, mTrueAtX0, nInside],
  );

  const gap = Number.isFinite(boxAverage) ? Math.abs(boxAverage - mTrueAtX0) : NaN;

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
            target x&#8320;: {x0.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.02}
            max={0.98}
            step={0.005}
            value={x0}
            onChange={(e) => setX0(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2 flex-1 min-w-[200px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">
            band h&#8320;: {h0.toFixed(3)}
          </span>
          <input
            type="range"
            min={0.005}
            max={0.2}
            step={0.005}
            value={h0}
            onChange={(e) => setH0(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <span className="ml-auto text-xs text-[var(--color-text-secondary)] font-mono">
          |gap| = {Number.isFinite(gap) ? gap.toFixed(3) : '—'}, SE ≈{' '}
          {Number.isFinite(sigmaSE) ? sigmaSE.toFixed(3) : '—'}
        </span>
      </div>

      <svg ref={renderRef} width={w} height={HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Drag x&#8320; along the support; the highlighted points (orange) are the in-band observations
        whose Y values get averaged. The blue segment shows that local box average; the red ring
        marks the true m(x&#8320;). Shrink h&#8320; to see variance dominate (few points → noisy
        average); grow h&#8320; to see bias dominate (many points but they smear m's curvature).
      </p>
    </div>
  );
}

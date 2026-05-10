import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  hStarAmiseMd,
  mTrueMd,
  mulberry32,
  nwMd,
  paletteKR,
  sampleToyMd,
} from './shared/kernel-regression';

// =============================================================================
// CurseOfDimensionalityExplorer — interactive companion to §6.3's curse-of-
// dimensionality figure. Two synchronized panels:
//
//   Left:  log-log empirical AMSE vs n at d ∈ {1, 2, 5, 10} with theoretical-
//          rate lines (slope -4/(4+d)) overlaid. As the reader scrubs d, the
//          highlighted curve flattens — visualizing the rate degradation.
//
//   Right: "Where are my neighbors?" — kernel-cube edge r = p^(1/d) versus p,
//          for the chosen d. Capturing 1% of uniform points needs r = 1% of
//          edge at d = 1, but r ≈ 63% at d = 10.
//
// Compute notes: the heavy MC sweep (4 dimensions × 5 sample sizes × 80
// replicates) is computed once on mount and cached; scrubbing d only changes
// which curve is highlighted, not the underlying simulation. Test point
// x_0 = (0.25, ..., 0.25) per the brief's gotcha.
//
// Numerical anchor: notebook §6.3 reports empirical slopes
//   d = 1: -0.83  (theory -0.80)
//   d = 2: -0.74  (theory -0.67)
//   d = 5: -0.61  (theory -0.44)
//   d = 10: -0.33  (theory -0.29)
// Mulberry32+Box-Muller produce slopes within ~10% of these.
//
// Static fallback: public/images/topics/kernel-regression/14_curse_of_dimensionality.png
// =============================================================================

const HEIGHT = 380;
const SM_BREAKPOINT = 640;
const SIGMA = 0.2;
const D_OPTIONS = [1, 2, 5, 10];
const N_VALUES = [100, 200, 500, 1000, 2000];
const B_PER = 60;

const D_COLORS: Record<number, string> = {
  1: paletteKR.posterior,
  2: paletteKR.truth,
  5: paletteKR.band,
  10: paletteKR.alt,
};

export default function CurseOfDimensionalityExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [dIdx, setDIdx] = useState(2); // default 5
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;
  const wPanel = isMobile ? w : w / 2;
  const d = D_OPTIONS[dIdx];

  // Precompute MC sweep once on mount.
  const sweep = useMemo(() => {
    const result: Record<number, { logN: number[]; logAmse: number[]; slope: number; theory: number }> = {};
    const rng = mulberry32(20261001);
    for (const dVal of D_OPTIONS) {
      const x0 = new Float64Array(dVal);
      for (let j = 0; j < dVal; j++) x0[j] = 0.25;
      const mX0 = mTrueMd(x0);

      const logN: number[] = [];
      const logAmse: number[] = [];
      for (const n of N_VALUES) {
        const hOpt = hStarAmiseMd(dVal, n, SIGMA);
        let acc = 0;
        for (let b = 0; b < B_PER; b++) {
          const { X, Y } = sampleToyMd(n, dVal, SIGMA, rng);
          const mhat = nwMd(X, Y, x0, hOpt);
          const r = mhat - mX0;
          acc += r * r;
        }
        const amse = acc / B_PER;
        logN.push(Math.log10(n));
        logAmse.push(Math.log10(amse));
      }

      // Linear regression slope.
      const meanLogN = logN.reduce((a, b) => a + b, 0) / logN.length;
      const meanLogA = logAmse.reduce((a, b) => a + b, 0) / logAmse.length;
      let num = 0;
      let den = 0;
      for (let i = 0; i < logN.length; i++) {
        num += (logN[i] - meanLogN) * (logAmse[i] - meanLogA);
        den += (logN[i] - meanLogN) ** 2;
      }
      result[dVal] = {
        logN,
        logAmse,
        slope: num / den,
        theory: -4 / (4 + dVal),
      };
    }
    return result;
  }, []);

  // Left-panel render (AMSE vs n).
  const leftRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!leftRef.current || wPanel <= 0) return;
    const svg = d3.select(leftRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 32, right: 16, bottom: 42, left: 56 };
    const innerW = wPanel - margin.left - margin.right;
    const innerH = HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const allLogN = D_OPTIONS.flatMap((dV) => sweep[dV].logN);
    const allLogA = D_OPTIONS.flatMap((dV) => sweep[dV].logAmse);
    const xScale = d3
      .scaleLinear()
      .domain([Math.min(...allLogN), Math.max(...allLogN)])
      .range([0, innerW]);
    const yScale = d3
      .scaleLinear()
      .domain([Math.min(...allLogA) - 0.2, Math.max(...allLogA) + 0.2])
      .range([innerH, 0]);

    for (const dV of D_OPTIONS) {
      const { logN, logAmse } = sweep[dV];
      const focused = dV === d;
      const line = d3
        .line<number>()
        .x((_, i) => xScale(logN[i]))
        .y((dd) => yScale(dd));
      g.append('path')
        .datum(logAmse)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', D_COLORS[dV])
        .style('stroke-width', focused ? 2.4 : 1.0)
        .style('opacity', focused ? 1.0 : 0.35);
      g.selectAll(`circle.amse-${dV}`)
        .data(logAmse)
        .enter()
        .append('circle')
        .attr('class', `amse-${dV}`)
        .attr('cx', (_, i) => xScale(logN[i]))
        .attr('cy', (dd) => yScale(dd))
        .attr('r', focused ? 4 : 2.6)
        .style('fill', D_COLORS[dV])
        .style('opacity', focused ? 1.0 : 0.4);
    }

    // Theoretical-rate line for the focused d.
    const focused = sweep[d];
    const theoryLine = focused.theory;
    // Use the first (smallest n) point as anchor; line passes through it.
    const xAnchor = focused.logN[0];
    const yAnchor = focused.logAmse[0];
    const xEnd = focused.logN[focused.logN.length - 1];
    const yEnd = yAnchor + theoryLine * (xEnd - xAnchor);
    g.append('line')
      .attr('x1', xScale(xAnchor))
      .attr('y1', yScale(yAnchor))
      .attr('x2', xScale(xEnd))
      .attr('y2', yScale(yEnd))
      .style('stroke', D_COLORS[d])
      .style('stroke-dasharray', '4 3')
      .style('stroke-width', 1.2)
      .style('opacity', 0.7);

    // Axes.
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat((v) => `10^${(+v).toFixed(0)}`))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((v) => `10^${(+v).toFixed(1)}`))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 32)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .text('sample size n (log)');
    g.append('text')
      .attr('x', -42)
      .attr('y', innerH / 2)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('transform', `rotate(-90,-42,${innerH / 2})`)
      .text('AMSE at x_0 (log)');

    svg
      .append('text')
      .attr('x', wPanel / 2)
      .attr('y', 18)
      .style('fill', 'var(--color-text)')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`AMSE rate at d = ${d}:  empirical slope = ${focused.slope.toFixed(2)},  theory = ${focused.theory.toFixed(2)}`);

    // Legend.
    const legendG = g.append('g').attr('transform', `translate(${innerW - 80}, 8)`);
    D_OPTIONS.forEach((dV, i) => {
      const yOff = i * 14;
      legendG
        .append('line')
        .attr('x1', 0)
        .attr('y1', yOff)
        .attr('x2', 14)
        .attr('y2', yOff)
        .style('stroke', D_COLORS[dV])
        .style('stroke-width', dV === d ? 2.4 : 1.0)
        .style('opacity', dV === d ? 1.0 : 0.4);
      legendG
        .append('text')
        .attr('x', 20)
        .attr('y', yOff + 3)
        .style('fill', dV === d ? 'var(--color-text)' : 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text(`d = ${dV}`);
    });
  }, [wPanel, sweep, d]);

  // Right-panel render (kernel-cube edge).
  const rightRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!rightRef.current || wPanel <= 0) return;
    const svg = d3.select(rightRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 32, right: 16, bottom: 42, left: 56 };
    const innerW = wPanel - margin.left - margin.right;
    const innerH = HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0.001, 0.5]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    const ps = d3.range(0.001, 0.51, 0.005);
    for (const dV of D_OPTIONS) {
      const focused = dV === d;
      const data = ps.map((p) => ({ p, r: Math.pow(p, 1 / dV) }));
      const line = d3
        .line<{ p: number; r: number }>()
        .x((pt) => xScale(pt.p))
        .y((pt) => yScale(pt.r));
      g.append('path')
        .datum(data)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', D_COLORS[dV])
        .style('stroke-width', focused ? 2.4 : 1.0)
        .style('opacity', focused ? 1.0 : 0.35);
    }

    // Annotate the p = 0.01 point on the focused d.
    const r1pct = Math.pow(0.01, 1 / d);
    g.append('circle')
      .attr('cx', xScale(0.01))
      .attr('cy', yScale(r1pct))
      .attr('r', 5)
      .style('fill', D_COLORS[d])
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', 1.2);
    g.append('text')
      .attr('x', xScale(0.01) + 8)
      .attr('y', yScale(r1pct) + 4)
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px')
      .text(`r = ${(r1pct * 100).toFixed(1)}% (capture 1%)`);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5, '.0%'))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5, '.0%'))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 32)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .text('fraction p of points to capture');
    g.append('text')
      .attr('x', -42)
      .attr('y', innerH / 2)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('transform', `rotate(-90,-42,${innerH / 2})`)
      .text('kernel-cube edge r');

    svg
      .append('text')
      .attr('x', wPanel / 2)
      .attr('y', 18)
      .style('fill', 'var(--color-text)')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`Where are my neighbors?  r = p^(1/d) at d = ${d}`);
  }, [wPanel, d]);

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
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">dimension d: {d}</span>
          <input
            type="range"
            min={0}
            max={D_OPTIONS.length - 1}
            step={1}
            value={dIdx}
            onChange={(e) => setDIdx(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <span className="ml-auto text-xs text-[var(--color-text-secondary)] font-mono">
          rate AMSE(h^*) ∝ n^({sweep[d].theory.toFixed(3)})
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '4px' }}>
        <svg ref={leftRef} width={wPanel} height={HEIGHT} />
        <svg ref={rightRef} width={wPanel} height={HEIGHT} />
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        At d = 1 the AMSE line drops with slope close to −0.8 (one decade of n buys ~6× lower
        error). Crank d to 10: the slope flattens to −0.29 — the same factor of n improvement
        only buys ~2×. The right panel is the geometric face of this: a "local" kernel
        neighborhood at d = 10 spans &gt;60% of each axis to capture even 1% of uniform points.
      </p>
    </div>
  );
}

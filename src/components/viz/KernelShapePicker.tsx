import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  KERNEL_CONSTANTS,
  KERNELS,
  hStarAmiseUni,
  mTrueUni,
  mulberry32,
  nadarayaWatson,
  paletteKR,
  sampleToyUni,
} from './shared/kernel-regression';
import type { KernelName } from './shared/kernel-regression';

// =============================================================================
// KernelShapePicker — interactive companion to §7.1–7.2's canonical-kernel
// theorem figure. Two stacked panels:
//
//   Top:    the kernel zoo overlaid on u ∈ [-3, 3]. The reader toggles which
//           kernels are shown (multi-select) and reads off mu_2(K), R(K),
//           delta(K), C(K) for each.
//
//   Bottom: NW smoothers on the §1 toy at the AMISE-optimal h^*_K = delta(K)·
//           h_bar^* per kernel. With "canonical mode" on, each kernel uses its
//           own AMISE-optimal bandwidth and the smoothers nearly coincide.
//           With "raw h" mode, all kernels share a single user-chosen h and
//           the differences become visible.
//
// Numerical anchor: notebook §7.2 reports h_bar^* = 0.0481 for n = 200, σ =
// 0.2; per-kernel optimal h*_K values: Gaussian 0.037, Epanechnikov 0.083,
// Box 0.065, Triangular 0.091, Quartic 0.098. Max gap from Epanechnikov on
// the interior [0.1, 0.9]: Gaussian 0.021, Box 0.054, Triangular 0.019,
// Quartic 0.010 (canonical-kernel theorem in action).
//
// Static fallback: public/images/topics/kernel-regression/15_canonical_kernel.png
// =============================================================================

const TOP_HEIGHT = 240;
const BOTTOM_HEIGHT = 320;
const SM_BREAKPOINT = 640;
const N = 200;
const SIGMA = 0.2;

const KERNEL_LIST: KernelName[] = ['gaussian', 'epanechnikov', 'box', 'triangular', 'quartic'];
const KERNEL_COLORS: Record<KernelName, string> = {
  gaussian: paletteKR.posterior,
  epanechnikov: paletteKR.truth,
  box: paletteKR.band,
  triangular: paletteKR.alt,
  quartic: paletteKR.accent,
};

const KERNEL_DISPLAY: Record<KernelName, string> = {
  gaussian: 'Gaussian',
  epanechnikov: 'Epanechnikov',
  box: 'Box',
  triangular: 'Triangular',
  quartic: 'Quartic',
};

export default function KernelShapePicker() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [enabled, setEnabled] = useState<Record<KernelName, boolean>>({
    gaussian: true,
    epanechnikov: true,
    box: true,
    triangular: true,
    quartic: true,
  });
  const [canonical, setCanonical] = useState(true);
  const [logH, setLogH] = useState(Math.log10(0.05));

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;
  const h = Math.pow(10, logH);

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

  // h^*_K per kernel (canonical mode) — uses delta(K) · h_bar^*.
  const hStarPerKernel = useMemo(() => {
    const result: Record<KernelName, number> = {} as Record<KernelName, number>;
    for (const k of KERNEL_LIST) {
      result[k] = hStarAmiseUni(N, SIGMA, k);
    }
    return result;
  }, []);

  // Top-panel render — kernel shapes.
  const topRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!topRef.current || w <= 0) return;
    const svg = d3.select(topRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 16, bottom: 36, left: 50 };
    const innerW = w - margin.left - margin.right;
    const innerH = TOP_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([-3, 3]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1.05]).range([innerH, 0]);

    const NPTS = 600;
    const uGrid = d3.range(NPTS).map((i) => -3 + (6 * i) / (NPTS - 1));

    for (const k of KERNEL_LIST) {
      if (!enabled[k]) continue;
      const data = uGrid.map((u) => ({ u, K: KERNELS[k](u) }));
      const line = d3
        .line<{ u: number; K: number }>()
        .x((pt) => xScale(pt.u))
        .y((pt) => yScale(pt.K));
      g.append('path')
        .datum(data)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', KERNEL_COLORS[k])
        .style('stroke-width', 1.6);
    }

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(7))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 28)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .text('u');
    g.append('text')
      .attr('x', -34)
      .attr('y', innerH / 2)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('transform', `rotate(-90,-34,${innerH / 2})`)
      .text('K(u)');

    svg
      .append('text')
      .attr('x', w / 2)
      .attr('y', 18)
      .style('fill', 'var(--color-text)')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text('Kernel shapes');
  }, [w, enabled]);

  // Bottom-panel render — NW smoothers.
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
    const yScale = d3.scaleLinear().domain([-1.6, 1.7]).range([innerH, 0]);

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
      .style('stroke', 'black')
      .style('stroke-width', 1.0)
      .style('opacity', 0.5);

    for (const k of KERNEL_LIST) {
      if (!enabled[k]) continue;
      const hUse = canonical ? hStarPerKernel[k] : h;
      const fit = nadarayaWatson(X, Y, xGrid, hUse, KERNELS[k]);
      const line = d3
        .line<number>()
        .defined((d) => Number.isFinite(d))
        .x((_, i) => xScale(xGrid[i]))
        .y((d) => yScale(d));
      g.append('path')
        .datum(Array.from(fit))
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', KERNEL_COLORS[k])
        .style('stroke-width', 1.6)
        .style('opacity', 0.85);
    }

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
      .text(canonical ? `NW smoothers at h^*_K = δ(K) · h̄^*` : `NW smoothers at common h = ${h.toFixed(3)}`);
  }, [w, X, Y, xGrid, mGrid, enabled, canonical, h, hStarPerKernel]);

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
    >
      <div className="flex flex-wrap items-center gap-3 mb-4 text-xs">
        <span className="text-[var(--color-text-secondary)]">kernels:</span>
        {KERNEL_LIST.map((k) => (
          <label key={k} className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={enabled[k]}
              onChange={(e) => setEnabled({ ...enabled, [k]: e.target.checked })}
              className="accent-[var(--color-accent)]"
            />
            <span style={{ color: KERNEL_COLORS[k], fontWeight: 500 }}>{KERNEL_DISPLAY[k]}</span>
          </label>
        ))}
        <label className="flex items-center gap-1 cursor-pointer ml-3">
          <input
            type="checkbox"
            checked={canonical}
            onChange={(e) => setCanonical(e.target.checked)}
            className="accent-[var(--color-accent)]"
          />
          <span className="text-[var(--color-text)] font-medium">canonical h^*_K</span>
        </label>
        {!canonical && (
          <label className="flex items-center gap-1 ml-2 flex-1 min-w-[180px]">
            <span className="text-[var(--color-text-secondary)] whitespace-nowrap">h: {h.toFixed(3)}</span>
            <input
              type="range"
              min={Math.log10(0.005)}
              max={Math.log10(0.3)}
              step={0.02}
              value={logH}
              onChange={(e) => setLogH(Number(e.target.value))}
              className="flex-1 accent-[var(--color-accent)]"
            />
          </label>
        )}
      </div>

      <svg ref={topRef} width={w} height={TOP_HEIGHT} />

      {/* Constants table. */}
      <div className="my-3 overflow-x-auto">
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="text-[var(--color-text-secondary)] border-b border-[var(--color-border)]">
              <th className="px-2 py-1 text-left">Kernel</th>
              <th className="px-2 py-1 text-right">μ₂(K)</th>
              <th className="px-2 py-1 text-right">R(K)</th>
              <th className="px-2 py-1 text-right">δ(K)</th>
              <th className="px-2 py-1 text-right">C(K)</th>
              <th className="px-2 py-1 text-right">h*_K</th>
            </tr>
          </thead>
          <tbody>
            {KERNEL_LIST.map((k) => {
              const c = KERNEL_CONSTANTS[k];
              const isMin = k === 'epanechnikov';
              return (
                <tr key={k} style={{ color: enabled[k] ? KERNEL_COLORS[k] : 'var(--color-text-secondary)' }}>
                  <td className="px-2 py-1">{KERNEL_DISPLAY[k]}</td>
                  <td className="px-2 py-1 text-right">{c.mu2.toFixed(4)}</td>
                  <td className="px-2 py-1 text-right">{c.R.toFixed(4)}</td>
                  <td className="px-2 py-1 text-right">{c.delta.toFixed(4)}</td>
                  <td className="px-2 py-1 text-right" style={{ fontWeight: isMin ? 600 : 400 }}>
                    {c.C.toFixed(4)}
                    {isMin && <sup>*</sup>}
                  </td>
                  <td className="px-2 py-1 text-right">{hStarPerKernel[k].toFixed(4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-1 text-[10px] text-[var(--color-text-secondary)]">
          *Epanechnikov minimizes C(K) — the Hodges-Lehmann optimality result.
        </p>
      </div>

      <svg ref={bottomRef} width={w} height={BOTTOM_HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Toggle "canonical h^*_K" off and pick a common h to see the smoothers diverge — box
        produces a piecewise-constant zigzag, Gaussian smooths most heavily, Epanechnikov sits
        in between. Toggle it on, and at each kernel's own AMISE-optimal h^*_K, the curves
        nearly overlap. That's the canonical-kernel theorem (Marron-Nolan 1988): kernel choice
        is mostly a calibration on h.
      </p>
    </div>
  );
}

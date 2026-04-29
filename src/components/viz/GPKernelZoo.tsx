import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  kernelMatern12,
  kernelMatern32,
  kernelMatern52,
  kernelPeriodic,
  kernelSE,
  mulberry32,
  paletteSamples,
  sampleGPPrior,
} from './shared/gaussian-processes';
import { linspace } from '../../data/gaussian-processes-data';

// =============================================================================
// GPKernelZoo — interactive companion to §2's prior-samples gallery.
// Reproduces notebook cell 6 (figures/02_prior_samples_gallery.png) and
// promotes the v2 enhancement of §2 (the (ν, ℓ) joint axis) into a single
// interactive panel: kernel selector + lengthscale slider + (when periodic)
// period slider, with a side panel showing the kernel function k(0, h).
//
// Static fallback: public/images/topics/gaussian-processes/02_prior_samples_gallery.png
// =============================================================================

const HEIGHT = 360;
const KERNEL_HEIGHT = 140;
const SM_BREAKPOINT = 640;
const N_GRID = 300;
const N_SAMPLES = 5;
const Y_LIM: [number, number] = [-3.5, 3.5];

type KernelChoice = 'se' | 'matern12' | 'matern32' | 'matern52' | 'periodic';

const KERNEL_LABELS: Record<KernelChoice, string> = {
  se: 'Squared-exponential (ν → ∞)',
  matern12: 'Matérn-1/2 (ν = 1/2)',
  matern32: 'Matérn-3/2 (ν = 3/2)',
  matern52: 'Matérn-5/2 (ν = 5/2)',
  periodic: 'Periodic (MacKay 1998)',
};

function kernelMatrix(
  choice: KernelChoice,
  X1: number[],
  X2: number[],
  ell: number,
  period: number,
): number[][] {
  switch (choice) {
    case 'se': return kernelSE(X1, X2, 1.0, ell);
    case 'matern12': return kernelMatern12(X1, X2, 1.0, ell);
    case 'matern32': return kernelMatern32(X1, X2, 1.0, ell);
    case 'matern52': return kernelMatern52(X1, X2, 1.0, ell);
    case 'periodic': return kernelPeriodic(X1, X2, 1.0, ell, period);
  }
}

export default function GPKernelZoo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [choice, setChoice] = useState<KernelChoice>('se');
  const [ell, setEll] = useState(0.7);
  const [period, setPeriod] = useState(1.5);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const xGrid = useMemo(() => linspace(-3, 3, N_GRID), []);
  const hGrid = useMemo(() => linspace(0, 3, 200), []);

  // Prior samples — fixed seed so the user sees "the same draws at different ℓ".
  const samples = useMemo(() => {
    const K = kernelMatrix(choice, xGrid, xGrid, ell, period);
    const rng = mulberry32(2024);
    return sampleGPPrior(K, N_SAMPLES, rng);
  }, [choice, ell, period, xGrid]);

  // Kernel function k(0, h) on hGrid, normalized so caller can compare across kernels.
  const kernelCurve = useMemo(() => {
    const K = kernelMatrix(choice, [0], hGrid, ell, period);
    return K[0]; // (n_h,) — k(0, h) for h in hGrid
  }, [choice, ell, period, hGrid]);

  // Layout: large sample panel on the left, kernel-curve inset on the right
  // (or stacked on mobile).
  const samplePanelW = useMemo(() => {
    if (w <= 0) return 0;
    return isMobile ? w : Math.floor((w - 16) * 0.62);
  }, [w, isMobile]);
  const kernelPanelW = useMemo(() => {
    if (w <= 0) return 0;
    return isMobile ? w : Math.floor((w - 16) * 0.38);
  }, [w, isMobile]);

  const sampleRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (samplePanelW <= 0) return;
      const margin = { top: 28, right: 16, bottom: 32, left: 44 };
      const innerW = samplePanelW - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-3, 3]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(Y_LIM).range([innerH, 0]);

      g.append('line')
        .attr('x1', 0).attr('x2', innerW)
        .attr('y1', yScale(0)).attr('y2', yScale(0))
        .style('stroke', 'var(--color-border)')
        .style('stroke-dasharray', '2 3')
        .style('stroke-width', 0.8);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 26)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '10px')
        .text('x');
      g.append('text')
        .attr('x', -32).attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '10px')
        .attr('transform', `rotate(-90,-32,${innerH / 2})`)
        .text('f(x)');

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
          .style('opacity', 0.92);
      }

      svg.append('text')
        .attr('x', samplePanelW / 2).attr('y', 16)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text(`${KERNEL_LABELS[choice]} — 5 prior samples`);
    },
    [samplePanelW, samples, choice],
  );

  const kernelRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (kernelPanelW <= 0) return;
      const margin = { top: 28, right: 12, bottom: 32, left: 38 };
      const innerW = kernelPanelW - margin.left - margin.right;
      const innerH = KERNEL_HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 3]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-0.05, 1.1]).range([innerH, 0]);

      g.append('line')
        .attr('x1', 0).attr('x2', innerW)
        .attr('y1', yScale(0)).attr('y2', yScale(0))
        .style('stroke', 'var(--color-border)')
        .style('stroke-dasharray', '2 3')
        .style('stroke-width', 0.8);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(4))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 26)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '10px')
        .text('h = |x - x\'|');
      g.append('text')
        .attr('x', -28).attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '10px')
        .attr('transform', `rotate(-90,-28,${innerH / 2})`)
        .text('k(0, h)');

      const line = d3.line<number>()
        .x((_, i) => xScale(hGrid[i]))
        .y((d) => yScale(d));
      g.append('path')
        .datum(kernelCurve)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', 'var(--color-accent)')
        .style('stroke-width', 1.6);

      svg.append('text')
        .attr('x', kernelPanelW / 2).attr('y', 16)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text('Kernel shape');
    },
    [kernelPanelW, kernelCurve, choice],
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
        <label className="flex items-center gap-2">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">Kernel</span>
          <select
            value={choice}
            onChange={(e) => setChoice(e.target.value as KernelChoice)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] text-[var(--color-text)] px-2 py-1 text-sm"
          >
            <option value="se">Squared-exponential</option>
            <option value="matern12">Matérn-1/2</option>
            <option value="matern32">Matérn-3/2</option>
            <option value="matern52">Matérn-5/2</option>
            <option value="periodic">Periodic</option>
          </select>
        </label>
        <label className="flex items-center gap-2 flex-1 min-w-[180px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">ℓ: {ell.toFixed(2)}</span>
          <input
            type="range" min={0.2} max={2.0} step={0.05} value={ell}
            onChange={(e) => setEll(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        {choice === 'periodic' && (
          <label className="flex items-center gap-2 flex-1 min-w-[180px]">
            <span className="text-[var(--color-text-secondary)] whitespace-nowrap">period: {period.toFixed(2)}</span>
            <input
              type="range" min={0.5} max={3.0} step={0.05} value={period}
              onChange={(e) => setPeriod(Number(e.target.value))}
              className="flex-1 accent-[var(--color-accent)]"
            />
          </label>
        )}
      </div>

      <div
        className="flex gap-2"
        style={{ flexDirection: isMobile ? 'column' : 'row' }}
      >
        <svg ref={sampleRef} width={samplePanelW} height={HEIGHT} />
        <svg ref={kernelRef} width={kernelPanelW} height={KERNEL_HEIGHT} />
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Smaller ν gives rougher samples (Matérn-1/2 is Brownian-motion-like; ν → ∞ gives the
        analytic SE limit). Lengthscale ℓ tunes the spatial correlation distance, independently
        of smoothness. Periodic samples agree exactly across one period; the kernel encodes
        that as inductive bias rather than a fitted property.
      </p>
    </div>
  );
}

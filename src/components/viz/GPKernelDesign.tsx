import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gpPredictDiag,
  kernelByName1D,
  paletteGP,
} from './shared/gaussian-processes';
import {
  fTrue,
  linspace,
  SIGMA_F_TRUE,
  SIGMA_N_TRUE,
  X_TRAIN_SPARSE,
  Y_TRAIN_SPARSE,
} from '../../data/gaussian-processes-data';

// =============================================================================
// GPKernelDesign — interactive companion to §4's three-panel kernel-design
// figure. Reproduces notebook cell 10 (figures/04_kernel_design.png) and
// promotes the v2 enhancement listed in brief §4 (kernel toggle + ℓ slider on
// the §3 sparse data) into a single panel.
//
// User selects which kernels to overlay (Mat-1/2, 3/2, 5/2, SE) and the
// shared lengthscale ℓ; predictive mean and ±2σ band are drawn with
// consistent color coding across the toolbar legend.
//
// Static fallback: public/images/topics/gaussian-processes/04_kernel_design.png
// =============================================================================

const HEIGHT = 380;
const SM_BREAKPOINT = 640;
const N_TEST = 250;
const Y_LIM: [number, number] = [-2.2, 2.2];
const X_LIM: [number, number] = [-3.5, 3.5];

type KernelChoice = 'matern12' | 'matern32' | 'matern52' | 'se';

const KERNEL_DEFS: Array<{ id: KernelChoice; label: string; color: string }> = [
  { id: 'matern12', label: 'Mat-1/2', color: paletteGP.nystrom },
  { id: 'matern32', label: 'Mat-3/2', color: paletteGP.svgp },
  { id: 'matern52', label: 'Mat-5/2', color: paletteGP.posterior },
  { id: 'se',       label: 'SE',      color: paletteGP.truth },
];

export default function GPKernelDesign() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [active, setActive] = useState<Record<KernelChoice, boolean>>({
    matern12: false,
    matern32: true,
    matern52: false,
    se: true,
  });
  const [ell, setEll] = useState(0.6);
  const [showBand, setShowBand] = useState(false);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const xTest = useMemo(() => linspace(X_LIM[0], X_LIM[1], N_TEST), []);
  const Xtrain = useMemo(() => X_TRAIN_SPARSE.slice(), []);
  const Ytrain = useMemo(() => Y_TRAIN_SPARSE.slice(), []);

  // Predict for each active kernel.
  const predictions = useMemo(() => {
    const out: Array<{
      id: KernelChoice; color: string; mean: number[]; sd: number[];
    }> = [];
    for (const def of KERNEL_DEFS) {
      if (!active[def.id]) continue;
      const kfn = kernelByName1D(def.id, { sigmaF: SIGMA_F_TRUE, lengthscale: ell });
      const r = gpPredictDiag(Xtrain, Ytrain, xTest, kfn, SIGMA_N_TRUE);
      out.push({ id: def.id, color: def.color, mean: r.mean, sd: r.sd });
    }
    return out;
  }, [active, ell, xTest, Xtrain, Ytrain]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 28, right: 16, bottom: 38, left: 50 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain(X_LIM).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(Y_LIM).range([innerH, 0]);

      // Truth (dashed, light)
      const truth = d3.line<number>()
        .x((d) => xScale(d))
        .y((d) => yScale(fTrue(d)));
      g.append('path')
        .datum(xTest)
        .attr('d', truth)
        .style('fill', 'none')
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-dasharray', '3 3')
        .style('stroke-width', 0.8)
        .style('opacity', 0.6);

      // Per-kernel band + mean
      for (const p of predictions) {
        if (showBand) {
          const band = d3.area<number>()
            .x((_, i) => xScale(xTest[i]))
            .y0((_, i) => yScale(p.mean[i] - 2 * p.sd[i]))
            .y1((_, i) => yScale(p.mean[i] + 2 * p.sd[i]));
          g.append('path')
            .datum(p.mean)
            .attr('d', band)
            .style('fill', p.color)
            .style('opacity', 0.10);
        }
        const meanLine = d3.line<number>()
          .x((_, i) => xScale(xTest[i]))
          .y((d) => yScale(d));
        g.append('path')
          .datum(p.mean)
          .attr('d', meanLine)
          .style('fill', 'none')
          .style('stroke', p.color)
          .style('stroke-width', 1.6)
          .style('opacity', 0.9);
      }

      // Training points (always on top)
      g.selectAll('circle.train')
        .data(Xtrain)
        .enter()
        .append('circle')
        .attr('class', 'train')
        .attr('cx', (d) => xScale(d))
        .attr('cy', (_, i) => yScale(Ytrain[i]))
        .attr('r', 4.5)
        .style('fill', paletteGP.reference)
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', 1.0);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 30)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .text('x');
      g.append('text')
        .attr('x', -34).attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .attr('transform', `rotate(-90,-34,${innerH / 2})`)
        .text('f(x)');

      svg.append('text')
        .attr('x', w / 2).attr('y', 18)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '13px')
        .style('font-weight', '600')
        .text(`Posterior under different kernels, ℓ = ${ell.toFixed(2)} (n = 6 sparse data)`);
    },
    [w, predictions, Xtrain, Ytrain, xTest, ell, showBand],
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
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">Kernels</span>
          {KERNEL_DEFS.map((def) => (
            <label key={def.id} className="flex items-center gap-1.5 cursor-pointer text-xs">
              <input
                type="checkbox"
                checked={active[def.id]}
                onChange={(e) => setActive((prev) => ({ ...prev, [def.id]: e.target.checked }))}
                className="accent-[var(--color-accent)]"
              />
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ background: def.color, opacity: active[def.id] ? 1 : 0.3 }}
              />
              <span style={{ color: 'var(--color-text)' }}>{def.label}</span>
            </label>
          ))}
        </div>
        <label className="flex items-center gap-2 flex-1 min-w-[180px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">ℓ: {ell.toFixed(2)}</span>
          <input
            type="range" min={0.15} max={2.0} step={0.05} value={ell}
            onChange={(e) => setEll(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showBand}
            onChange={(e) => setShowBand(e.target.checked)}
            className="accent-[var(--color-accent)]"
          />
          <span className="text-[var(--color-text-secondary)]">±2σ bands</span>
        </label>
      </div>

      <svg ref={renderRef} width={w} height={HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Sweeping ℓ from small (e.g. 0.2) to large (e.g. 1.5) traces the bias–variance axis: too
        small overfits, too large underfits. The kernel choice axis is orthogonal — Mat-1/2
        gives jagged interpolations, Mat-5/2 looks similar to SE in this regime. Toggle bands
        on to see how each kernel quantifies its own uncertainty in the gaps.
      </p>
    </div>
  );
}

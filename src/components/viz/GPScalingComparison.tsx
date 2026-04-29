import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gaussianSampler,
  gpPredictDiag,
  gpPredictNystrom,
  gpPredictSVGP,
  kernelByName1D,
  mulberry32,
  paletteGP,
  rffPredict,
} from './shared/gaussian-processes';
import { ELL_TRUE, fTrue, linspace, SIGMA_F_TRUE, SIGMA_N_TRUE } from '../../data/gaussian-processes-data';

// =============================================================================
// GPScalingComparison — interactive companion to §6's scaling discussion.
// Implements the brief §6 v2 enhancement: an m-slider that exposes the
// fundamental approximation-error knob.
//
// No static fallback: notebook cell 13 is markdown only (§6 narrative + refs);
// this component IS the §6 visualization.
//
// At n = 200 training points, exact GP is still feasible (~30 ms Cholesky).
// The user toggles four methods (exact, Nyström, SVGP, RFF) on/off and slides
// m ∈ [10, 80] inducing points (or D for RFF). Wall-clock fit-time is
// reported for each method as a soft scaling preview.
// =============================================================================

const HEIGHT = 420;
const SM_BREAKPOINT = 640;
const N_TRAIN = 200;
const N_TEST = 250;
const X_LIM: [number, number] = [-6, 6];
const Y_LIM: [number, number] = [-2.5, 2.5];

interface MethodResult {
  mean: number[];
  sd: number[] | null; // null for Nyström (variance unreliable per §6)
  fitTimeMs: number;
}

export default function GPScalingComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [m, setM] = useState(30); // inducing points (Nyström, SVGP)
  const [D, setD] = useState(150); // RFF features
  const [showExact, setShowExact] = useState(true);
  const [showNystrom, setShowNystrom] = useState(true);
  const [showSVGP, setShowSVGP] = useState(true);
  const [showRFF, setShowRFF] = useState(true);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  // Synthetic training data — n = 200, deterministic via mulberry32(7).
  const { Xtrain, Ytrain } = useMemo(() => {
    const rng = mulberry32(7);
    const gauss = gaussianSampler(rng);
    const Xtrain = new Array<number>(N_TRAIN);
    const Ytrain = new Array<number>(N_TRAIN);
    for (let i = 0; i < N_TRAIN; i++) {
      Xtrain[i] = X_LIM[0] + (X_LIM[1] - X_LIM[0]) * rng();
      Ytrain[i] = fTrue(Xtrain[i]) + SIGMA_N_TRUE * gauss();
    }
    return { Xtrain, Ytrain };
  }, []);

  // Inducing points — equally spaced in [X_LIM[0]+0.5, X_LIM[1]-0.5]
  const Xinducing = useMemo(() => {
    return linspace(X_LIM[0] + 0.5, X_LIM[1] - 0.5, m);
  }, [m]);

  const xTest = useMemo(() => linspace(X_LIM[0], X_LIM[1], N_TEST), []);

  const kernelFn = useMemo(
    () => kernelByName1D('se', { sigmaF: SIGMA_F_TRUE, lengthscale: ELL_TRUE }),
    [],
  );

  const exact: MethodResult | null = useMemo(() => {
    if (!showExact) return null;
    const t0 = performance.now();
    const r = gpPredictDiag(Xtrain, Ytrain, xTest, kernelFn, SIGMA_N_TRUE);
    const t1 = performance.now();
    return { mean: r.mean, sd: r.sd, fitTimeMs: t1 - t0 };
  }, [showExact, Xtrain, Ytrain, xTest, kernelFn]);

  const nystrom: MethodResult | null = useMemo(() => {
    if (!showNystrom) return null;
    const t0 = performance.now();
    const r = gpPredictNystrom(Xtrain, Ytrain, Xinducing, xTest, kernelFn, SIGMA_N_TRUE);
    const t1 = performance.now();
    return { mean: r.mean, sd: null, fitTimeMs: t1 - t0 };
  }, [showNystrom, Xtrain, Ytrain, Xinducing, xTest, kernelFn]);

  const svgp: MethodResult | null = useMemo(() => {
    if (!showSVGP) return null;
    const t0 = performance.now();
    const r = gpPredictSVGP(Xtrain, Ytrain, Xinducing, xTest, kernelFn, SIGMA_N_TRUE);
    const t1 = performance.now();
    return { mean: r.mean, sd: r.sd, fitTimeMs: t1 - t0 };
  }, [showSVGP, Xtrain, Ytrain, Xinducing, xTest, kernelFn]);

  const rff: MethodResult | null = useMemo(() => {
    if (!showRFF) return null;
    const t0 = performance.now();
    const rng = mulberry32(11);
    const r = rffPredict(
      Xtrain, Ytrain, xTest, SIGMA_F_TRUE, ELL_TRUE, SIGMA_N_TRUE, D, rng,
    );
    const t1 = performance.now();
    return { mean: r.mean, sd: r.sd, fitTimeMs: t1 - t0 };
  }, [showRFF, Xtrain, Ytrain, xTest, D]);

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

      // Truth
      const truth = d3.line<number>()
        .x((d) => xScale(d))
        .y((d) => yScale(fTrue(d)));
      g.append('path')
        .datum(xTest)
        .attr('d', truth)
        .style('fill', 'none')
        .style('stroke', paletteGP.truth)
        .style('stroke-dasharray', '4 3')
        .style('stroke-width', 1.0)
        .style('opacity', 0.7);

      // Training data (subsample for visual clarity)
      const subsample = Xtrain.filter((_, i) => i % 4 === 0);
      const subsampleY = Ytrain.filter((_, i) => i % 4 === 0);
      g.selectAll('circle.train')
        .data(subsample)
        .enter()
        .append('circle')
        .attr('class', 'train')
        .attr('cx', (d) => xScale(d))
        .attr('cy', (_, i) => yScale(subsampleY[i]))
        .attr('r', 1.5)
        .style('fill', paletteGP.data)
        .style('opacity', 0.5);

      // Inducing-point ticks (along bottom)
      if (showNystrom || showSVGP) {
        g.selectAll('line.inducing')
          .data(Xinducing)
          .enter()
          .append('line')
          .attr('class', 'inducing')
          .attr('x1', (d) => xScale(d))
          .attr('x2', (d) => xScale(d))
          .attr('y1', innerH)
          .attr('y2', innerH - 5)
          .style('stroke', paletteGP.svgp)
          .style('stroke-width', 1.4)
          .style('opacity', 0.8);
      }

      // Plot each active method
      const drawMethod = (
        result: MethodResult | null,
        color: string,
        showSdBand: boolean,
      ) => {
        if (!result) return;
        if (showSdBand && result.sd) {
          const band = d3.area<number>()
            .x((_, i) => xScale(xTest[i]))
            .y0((_, i) => yScale(result.mean[i] - 2 * result.sd![i]))
            .y1((_, i) => yScale(result.mean[i] + 2 * result.sd![i]));
          g.append('path')
            .datum(result.mean)
            .attr('d', band)
            .style('fill', color)
            .style('opacity', 0.10);
        }
        const meanLine = d3.line<number>()
          .x((_, i) => xScale(xTest[i]))
          .y((d) => yScale(d));
        g.append('path')
          .datum(result.mean)
          .attr('d', meanLine)
          .style('fill', 'none')
          .style('stroke', color)
          .style('stroke-width', 1.6)
          .style('opacity', 0.92);
      };

      drawMethod(exact, paletteGP.posterior, true);
      drawMethod(nystrom, paletteGP.nystrom, false); // no band: variance unreliable
      drawMethod(svgp, paletteGP.svgp, true);
      drawMethod(rff, '#9467bd', true);

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
        .text(`Sparse-approximation comparison, n = ${N_TRAIN},  m = ${m},  D = ${D}`);
    },
    [w, exact, nystrom, svgp, rff, Xtrain, Ytrain, Xinducing, xTest, m, D, showNystrom, showSVGP],
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
        <label className="flex items-center gap-2 flex-1 min-w-[160px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">m: {m}</span>
          <input
            type="range" min={5} max={80} step={1} value={m}
            onChange={(e) => setM(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2 flex-1 min-w-[160px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">D (RFF): {D}</span>
          <input
            type="range" min={20} max={300} step={10} value={D}
            onChange={(e) => setD(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-3 mb-3 text-xs">
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showExact} onChange={(e) => setShowExact(e.target.checked)} className="accent-[var(--color-accent)]" />
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: paletteGP.posterior }} />
          <span style={{ color: 'var(--color-text)' }}>exact GP {exact && `· ${exact.fitTimeMs.toFixed(1)} ms`}</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showNystrom} onChange={(e) => setShowNystrom(e.target.checked)} className="accent-[var(--color-accent)]" />
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: paletteGP.nystrom }} />
          <span style={{ color: 'var(--color-text)' }}>Nyström (mean only) {nystrom && `· ${nystrom.fitTimeMs.toFixed(1)} ms`}</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showSVGP} onChange={(e) => setShowSVGP(e.target.checked)} className="accent-[var(--color-accent)]" />
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: paletteGP.svgp }} />
          <span style={{ color: 'var(--color-text)' }}>SVGP {svgp && `· ${svgp.fitTimeMs.toFixed(1)} ms`}</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showRFF} onChange={(e) => setShowRFF(e.target.checked)} className="accent-[var(--color-accent)]" />
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: '#9467bd' }} />
          <span style={{ color: 'var(--color-text)' }}>RFF {rff && `· ${rff.fitTimeMs.toFixed(1)} ms`}</span>
        </label>
      </div>

      <svg ref={renderRef} width={w} height={HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Slide m up: Nyström and SVGP converge toward exact GP as m → n. Nyström&apos;s
        variance is omitted because it&apos;s structurally overconfident; SVGP and RFF
        produce calibrated bands. Inducing-point locations are marked along the bottom of
        the plot. Wall-clock fit-times are reported per method — the takeaway is that for
        modest m and D, sparse approximations recover near-exact predictive mean accuracy
        at substantially lower cost than the O(n³) exact GP baseline.
      </p>
    </div>
  );
}

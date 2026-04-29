import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gpPredict,
  kernelByName1D,
  mulberry32,
  paletteGP,
  paletteSamples,
  sampleFromPosterior,
} from './shared/gaussian-processes';
import {
  ELL_TRUE,
  fTrue,
  linspace,
  SIGMA_F_TRUE,
  SIGMA_N_TRUE,
  X_TRAIN_DENSE,
  X_TRAIN_SPARSE,
  Y_TRAIN_DENSE,
  Y_TRAIN_SPARSE,
} from '../../data/gaussian-processes-data';

// =============================================================================
// GPRegressionPosterior — interactive companion to §3's three-panel posterior
// figure. Reproduces notebook cell 8 (figures/03_gp_regression_posterior.png),
// adding the v2 enhancement of an n-slider that sweeps from sparse-to-dense
// training-data and a toggle for the joint-posterior-samples overlay.
//
// Numerical anchor: at n=6 (X_TRAIN_SPARSE, Y_TRAIN_SPARSE), the predictive
// at x=0 should be (μ ≈ -0.137, σ ≈ 0.574) — verified by
// verify-gaussian-processes.ts §3 (iv).
//
// Static fallback: public/images/topics/gaussian-processes/03_gp_regression_posterior.png
// =============================================================================

const HEIGHT = 420;
const SM_BREAKPOINT = 640;
const N_TEST = 300;
const Y_LIM: [number, number] = [-2.2, 2.2];
const X_LIM: [number, number] = [-3.5, 3.5];

export default function GPRegressionPosterior() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nTrain, setNTrain] = useState(6); // 6 ... 30
  const [showSamples, setShowSamples] = useState(false);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const xTest = useMemo(() => linspace(X_LIM[0], X_LIM[1], N_TEST), []);

  // Pull the first nTrain canonical points. The nTrain ∈ [6, 30] regime
  // smoothly interpolates between the §3 sparse and dense panels:
  //   nTrain == 6  → exactly X_TRAIN_SPARSE / Y_TRAIN_SPARSE (panels a, b)
  //   nTrain == 30 → exactly X_TRAIN_DENSE  / Y_TRAIN_DENSE  (panel c)
  // For 6 < n < 30, we use the first n points of the dense set. This keeps the
  // user on canonical (notebook-pinned) data throughout the slider range.
  const { Xtrain, Ytrain } = useMemo(() => {
    if (nTrain === 6) {
      return {
        Xtrain: X_TRAIN_SPARSE.slice(),
        Ytrain: Y_TRAIN_SPARSE.slice(),
      };
    }
    return {
      Xtrain: X_TRAIN_DENSE.slice(0, nTrain),
      Ytrain: Y_TRAIN_DENSE.slice(0, nTrain),
    };
  }, [nTrain]);

  const kernelFn = useMemo(
    () => kernelByName1D('se', { sigmaF: SIGMA_F_TRUE, lengthscale: ELL_TRUE }),
    [],
  );

  const { mean, sd, samples } = useMemo(() => {
    const result = gpPredict(Xtrain, Ytrain, xTest, kernelFn, SIGMA_N_TRUE);
    let posteriorSamples: number[][] = [];
    if (showSamples) {
      const rng = mulberry32(2025);
      posteriorSamples = sampleFromPosterior(result.mean, result.cov, 5, rng);
    }
    return { mean: result.mean, sd: result.sd, samples: posteriorSamples };
  }, [Xtrain, Ytrain, xTest, kernelFn, showSamples]);

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

      // ±2σ band (filled area)
      const band = d3.area<number>()
        .x((_, i) => xScale(xTest[i]))
        .y0((_, i) => yScale(mean[i] - 2 * sd[i]))
        .y1((_, i) => yScale(mean[i] + 2 * sd[i]));
      g.append('path')
        .datum(mean)
        .attr('d', band)
        .style('fill', paletteGP.posterior)
        .style('opacity', 0.18);

      // Truth (dashed)
      const truth = d3.line<number>()
        .x((d) => xScale(d))
        .y((d) => yScale(fTrue(d)));
      g.append('path')
        .datum(xTest)
        .attr('d', truth)
        .style('fill', 'none')
        .style('stroke', paletteGP.truth)
        .style('stroke-dasharray', '4 3')
        .style('stroke-width', 1.2);

      // Posterior samples (when enabled)
      if (showSamples) {
        const sampleLine = d3.line<number>()
          .x((_, i) => xScale(xTest[i]))
          .y((d) => yScale(d));
        for (let i = 0; i < samples.length; i++) {
          g.append('path')
            .datum(samples[i])
            .attr('d', sampleLine)
            .style('fill', 'none')
            .style('stroke', paletteSamples[i % paletteSamples.length])
            .style('stroke-width', 1.0)
            .style('opacity', 0.7);
        }
      }

      // Posterior mean
      const meanLine = d3.line<number>()
        .x((_, i) => xScale(xTest[i]))
        .y((d) => yScale(d));
      g.append('path')
        .datum(mean)
        .attr('d', meanLine)
        .style('fill', 'none')
        .style('stroke', paletteGP.posterior)
        .style('stroke-width', 1.6);

      // Training points
      g.selectAll('circle.train')
        .data(Xtrain)
        .enter()
        .append('circle')
        .attr('class', 'train')
        .attr('cx', (d) => xScale(d))
        .attr('cy', (_, i) => yScale(Ytrain[i]))
        .attr('r', nTrain <= 8 ? 4.5 : 2.8)
        .style('fill', paletteGP.reference)
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', 1.0)
        .style('opacity', 0.9);

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

      // Title with status read-out
      svg.append('text')
        .attr('x', w / 2).attr('y', 18)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '13px')
        .style('font-weight', '600')
        .text(`GP regression posterior, n = ${nTrain},  ℓ = ${ELL_TRUE},  σ_n = ${SIGMA_N_TRUE}`);

      // Legend
      const legendX = innerW - 130;
      const legendY = 8;
      const legendG = g.append('g').attr('transform', `translate(${legendX}, ${legendY})`);
      const legendItems = [
        { color: paletteGP.posterior, label: 'posterior μ_*', dash: false },
        { color: paletteGP.truth, label: 'truth', dash: true },
      ];
      legendItems.forEach((it, i) => {
        const yOff = i * 14;
        legendG.append('line')
          .attr('x1', 0).attr('y1', yOff)
          .attr('x2', 16).attr('y2', yOff)
          .style('stroke', it.color)
          .style('stroke-width', 1.4)
          .style('stroke-dasharray', it.dash ? '4 3' : 'none');
        legendG.append('text')
          .attr('x', 22).attr('y', yOff + 3)
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '10px')
          .text(it.label);
      });
    },
    [w, mean, sd, samples, Xtrain, Ytrain, xTest, nTrain, showSamples],
  );

  // Read-outs at fixed test points (matches notebook §3 (iv) verification anchors)
  const idx0 = useMemo(() => xTest.findIndex((x) => Math.abs(x) < 7 / (N_TEST - 1) / 2), [xTest]);
  const muAt0 = mean[idx0];
  const sdAt0 = sd[idx0];

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
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">n training: {nTrain}</span>
          <input
            type="range" min={6} max={30} step={1} value={nTrain}
            onChange={(e) => setNTrain(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showSamples}
            onChange={(e) => setShowSamples(e.target.checked)}
            className="accent-[var(--color-accent)]"
          />
          <span className="text-[var(--color-text-secondary)]">show 5 posterior samples</span>
        </label>
        <span className="ml-auto text-xs text-[var(--color-text-secondary)] font-mono">
          μ_*(0) = {muAt0.toFixed(3)},  σ_*(0) = {sdAt0.toFixed(3)}
        </span>
      </div>

      <svg ref={renderRef} width={w} height={HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        At n = 6 the band pinches near training points and reverts to the prior in the
        gaps; at n = 30 the band collapses uniformly to the irreducible σ_n-noise level.
        Toggle &ldquo;show posterior samples&rdquo; to see joint draws — coherent functions, not
        pointwise marginal draws.
      </p>
    </div>
  );
}

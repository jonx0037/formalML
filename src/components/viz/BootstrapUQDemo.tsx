import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  baggedPredictGrid,
  bayesPolyPosterior,
  bayesPolyPredict,
  fTrue,
  linspace,
  mulberry32,
  pairsBootstrapFit,
  residualBootstrapFit,
  sampleHetero,
  sigmaTrue,
  wildBootstrapFit,
} from './shared/uncertainty-quantification';

// =============================================================================
// BootstrapUQDemo — §6
//
// Slider for B (commit-on-release per CLAUDE.md slider-perf rule). Dropdown
// for the bootstrap variant. Plots the bagged ±2σ_bag epistemic band against
// the §2.3 analytic epistemic curve (dashed reference).
// Static fallback: public/images/topics/uncertainty-quantification/fig_06_three_variants.png
// =============================================================================

const HEIGHT = 440;
const DEGREE = 7;
const X_GRID = linspace(-3.2, 3.2, 200);

const variants = {
  pairs: { name: 'pairs', color: '#10b981', fit: pairsBootstrapFit },
  residual: { name: 'residual', color: '#6b7280', fit: residualBootstrapFit },
  wild: { name: 'wild', color: '#f59e0b', fit: wildBootstrapFit },
} as const;

type VariantKey = keyof typeof variants;

export default function BootstrapUQDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [bDisplay, setBDisplay] = useState(100);
  const [bCommitted, setBCommitted] = useState(100);
  const [variant, setVariant] = useState<VariantKey>('wild');

  const toy = useMemo(() => {
    const rng = mulberry32(20260514);
    const { X, y } = sampleHetero(200, -3, 3, rng);
    const sigmaTr = X.map(sigmaTrue);
    const post = bayesPolyPosterior(X, y, DEGREE, sigmaTr);
    const pred = bayesPolyPredict(X_GRID, post, X_GRID.map(sigmaTrue));
    return { X, y, epiAnalytic: pred.epistemicVar };
  }, []);

  const bootstrap = useMemo(() => {
    const rng = mulberry32(20260514 + bCommitted * 7 + variant.length);
    const fits = variants[variant].fit(toy.X, toy.y, bCommitted, DEGREE, rng);
    const bagged = baggedPredictGrid(fits, X_GRID, DEGREE);
    return bagged;
  }, [toy, bCommitted, variant]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 24, bottom: 48, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-3.3, 3.3]).range([0, W]);
      const allYs = [
        ...toy.y,
        ...bootstrap.mean.map((m, i) => m - 2 * Math.sqrt(bootstrap.variance[i])),
        ...bootstrap.mean.map((m, i) => m + 2 * Math.sqrt(bootstrap.variance[i])),
      ];
      const yMin = Math.min(...allYs, -3);
      const yMax = Math.max(...allYs, 3);
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([H, 0]).nice();

      // Bagged ±2σ band.
      const color = variants[variant].color;
      const band = d3.area<number>()
        .x((_, i) => xScale(X_GRID[i]))
        .y0((_, i) => yScale(bootstrap.mean[i] - 2 * Math.sqrt(bootstrap.variance[i])))
        .y1((_, i) => yScale(bootstrap.mean[i] + 2 * Math.sqrt(bootstrap.variance[i])));
      g.append('path').datum(X_GRID).attr('d', band)
        .style('fill', color).style('opacity', 0.28);

      const lineGen = d3.line<number>()
        .x((_, i) => xScale(X_GRID[i])).y((d) => yScale(d));
      g.append('path').datum(X_GRID.map(fTrue)).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#94a3b8').attr('stroke-dasharray', '5 3').attr('stroke-width', 1.4);
      g.append('path').datum(bootstrap.mean).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', color).attr('stroke-width', 2);

      // Analytic ±2σ_epi reference (Bayesian §2.3).
      const refBandHi = bootstrap.mean.map((m, i) => m + 2 * Math.sqrt(toy.epiAnalytic[i]));
      const refBandLo = bootstrap.mean.map((m, i) => m - 2 * Math.sqrt(toy.epiAnalytic[i]));
      g.append('path').datum(refBandHi).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#000').attr('stroke-width', 1).attr('stroke-dasharray', '2 3')
        .style('opacity', 0.7);
      g.append('path').datum(refBandLo).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#000').attr('stroke-width', 1).attr('stroke-dasharray', '2 3')
        .style('opacity', 0.7);

      // Training points.
      g.selectAll('.pt').data(toy.X).enter().append('circle').attr('class', 'pt')
        .attr('cx', (d) => xScale(d as number))
        .attr('cy', (_, i) => yScale(toy.y[i]))
        .attr('r', 2).style('fill', '#1f2937').style('opacity', 0.4);

      g.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)').text('x');
      g.append('text').attr('x', W / 2).attr('y', -8).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)')
        .text(`${variants[variant].name} bootstrap, B = ${bCommitted} — solid band: bagged ±2σ; dashed: analytic §2.3 epistemic`);
    },
    [containerWidth, toy, bootstrap, variant, bCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Bagged predictive band for the selected bootstrap variant, overlaid on the analytic §2.3 epistemic reference." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 240 }}>
          <span>B (bootstrap replicates): <strong>{bDisplay}</strong></span>
          <input type="range" min={5} max={500} step={5} value={bDisplay}
            onChange={(e) => setBDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setBCommitted(bDisplay)} onTouchEnd={() => setBCommitted(bDisplay)}
            onKeyUp={() => setBCommitted(bDisplay)} aria-label="Bootstrap replicate count B" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>variant</span>
          <select value={variant} onChange={(e) => setVariant(e.target.value as VariantKey)}
            aria-label="Bootstrap variant">
            <option value="pairs">pairs</option>
            <option value="residual">residual</option>
            <option value="wild">wild</option>
          </select>
        </label>
      </div>
      <p style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        The dashed reference is the analytic Bayesian §2.3 epistemic. Wild bootstrap tracks it best on this
        heteroscedastic toy; residual underestimates because exchangeable-residual assumption fails.
      </p>
    </div>
  );
}

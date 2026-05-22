import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteNB, normCdf, normPdf, bayesRiskGaussianMixture1D } from './shared/naive-bayes';

// =============================================================================
// BayesRiskDecomposition — §3.2 companion. Two equal-variance Gaussians at
// adjustable means and prior weighting. The viewer overlays:
//
//   • the two class-conditional densities p(x | Y = 0) and p(x | Y = 1)
//   • the Bayes-optimal decision boundary (where π·f₁ = (1-π)·f₀)
//   • the Bayes-risk region shaded under min(π·f₁, (1-π)·f₀)
//
// Reader controls: μ separation, common σ, class prior π.
// The reported Bayes risk uses bayesRiskGaussianMixture1D (trapezoidal quadrature);
// for σ₁ = σ₂ = σ, π = 1/2, separation Δ, it equals Φ(−Δ/(2σ)).
// =============================================================================

const X_MIN = -6;
const X_MAX = 6;
const N_POINTS = 401;

export default function BayesRiskDecomposition() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [separation, setSeparation] = useState(2.0);
  const [sigma, setSigma] = useState(1.0);
  const [prior, setPrior] = useState(0.5);

  const mu0 = -separation / 2;
  const mu1 = +separation / 2;

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 320;
  const margin = { top: 16, right: 20, bottom: 38, left: 50 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const data = useMemo(() => {
    const xs: number[] = [];
    const f0: number[] = [];
    const f1: number[] = [];
    const weighted0: number[] = [];
    const weighted1: number[] = [];
    const overlap: number[] = [];
    for (let i = 0; i < N_POINTS; i++) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / (N_POINTS - 1);
      xs.push(x);
      const v0 = normPdf((x - mu0) / sigma) / sigma;
      const v1 = normPdf((x - mu1) / sigma) / sigma;
      f0.push(v0);
      f1.push(v1);
      const w0 = (1 - prior) * v0;
      const w1 = prior * v1;
      weighted0.push(w0);
      weighted1.push(w1);
      overlap.push(Math.min(w0, w1));
    }
    return { xs, f0, f1, weighted0, weighted1, overlap };
  }, [mu0, mu1, sigma, prior]);

  const bayesRisk = useMemo(
    () => bayesRiskGaussianMixture1D(mu1, sigma, mu0, sigma, prior, 4001),
    [mu0, mu1, sigma, prior],
  );

  // Decision boundary: x* where π·f₁(x) = (1-π)·f₀(x); for equal σ this is
  // x* = (μ₀ + μ₁)/2 + σ²/(μ₁ - μ₀) · log((1-π)/π).
  const xStar = (mu0 + mu1) / 2 + (sigma * sigma * Math.log((1 - prior) / prior)) / (mu1 - mu0);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const yMax = Math.max(...data.weighted0, ...data.weighted1) * 1.08;
      const x = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
      const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      // Decision-boundary shaded fill (overlap region).
      const area = d3
        .area<number>()
        .x((_, i) => x(data.xs[i]))
        .y0(innerH)
        .y1((v) => y(v));
      g.append('path')
        .datum(data.overlap)
        .attr('fill-opacity', 0.32)
        .attr('d', area)
        .style('fill', paletteNB.highlight);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(7));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .style('fill', 'currentColor')
        .text('x');
      g.append('text')
        .attr('transform', `translate(-36,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .style('fill', 'currentColor')
        .text('weighted density');

      const line = d3
        .line<number>()
        .x((_, i) => x(data.xs[i]))
        .y((v) => y(v));
      g.append('path')
        .datum(data.weighted0)
        .attr('fill', 'none')
        .attr('stroke-width', 2)
        .attr('d', line)
        .style('stroke', paletteNB.neg);
      g.append('path')
        .datum(data.weighted1)
        .attr('fill', 'none')
        .attr('stroke-width', 2)
        .attr('d', line)
        .style('stroke', paletteNB.pos);

      // Decision-boundary vertical line at x*.
      g.append('line')
        .attr('x1', x(xStar))
        .attr('x2', x(xStar))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3 3')
        .attr('opacity', 0.7)
        .style('stroke', paletteNB.data);
      g.append('text')
        .attr('x', x(xStar))
        .attr('y', 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'currentColor')
        .text(`x* = ${xStar.toFixed(2)}`);

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 168}, 6)`);
      const items = [
        { label: `(1-π)·f(x|Y=0)`, color: paletteNB.neg },
        { label: 'π·f(x|Y=1)', color: paletteNB.pos },
        { label: 'Bayes risk region', color: paletteNB.highlight },
      ];
      items.forEach((item, i) => {
        legend
          .append('rect')
          .attr('x', 0)
          .attr('y', i * 16 - 9)
          .attr('width', 12)
          .attr('height', 4)
          .style('fill', item.color);
        legend
          .append('text')
          .attr('x', 18)
          .attr('y', i * 16)
          .attr('font-size', 11)
          .style('fill', 'currentColor')
          .text(item.label);
      });
    },
    [data, xStar, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Separation Δ = {separation.toFixed(2)}</span>
          <input
            type="range"
            min={0.5}
            max={5}
            step={0.05}
            value={separation}
            onChange={(e) => setSeparation(Number(e.target.value))}
            aria-label="Class-mean separation"
          />
        </label>
        <label className="flex flex-col">
          <span>Common σ = {sigma.toFixed(2)}</span>
          <input
            type="range"
            min={0.3}
            max={2.5}
            step={0.05}
            value={sigma}
            onChange={(e) => setSigma(Number(e.target.value))}
            aria-label="Common standard deviation"
          />
        </label>
        <label className="flex flex-col">
          <span>Prior π = {prior.toFixed(2)}</span>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={prior}
            onChange={(e) => setPrior(Number(e.target.value))}
            aria-label="Class prior"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Bayes risk on 1D Gaussian mixture"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        Bayes risk = <strong>{bayesRisk.toFixed(4)}</strong>. For the symmetric equal-prior case (Δ = {separation.toFixed(2)}, σ = {sigma.toFixed(2)}, π = 0.5)
        this equals Φ(−Δ/(2σ)) = <strong>{normCdf(-separation / (2 * sigma)).toFixed(4)}</strong>; the displayed value is the numerical quadrature for the chosen π.
      </div>
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  betaPdf,
  hierarchicalBetaBinomialEB,
  COLOR_VARIANT_A,
  COLOR_VARIANT_B,
  COLOR_LIKELIHOOD,
} from './shared/bayesian-inference';

// =============================================================================
// HierarchicalShrinkageDemo — §9 interactive.
//
// 8-variant Beta-Binomial hierarchy from notebook cell 29. The reader sees:
// Left: per-group posterior-mean curves as functions of log(kappa) — each
// group starts at its MLE for tiny kappa and converges to the grand mean
// for large kappa. The EB-estimated kappa-hat is annotated.
// Right: per-group posterior densities at the chosen kappa (default kappa-hat
// from empirical Bayes). MLEs as triangles, grand mean as solid vertical line.
//
// The slider drives kappa on a log scale; users can play with no-pooling
// (kappa → 0), complete-pooling (kappa → ∞), and partial-pooling regimes.
// =============================================================================

const FIXTURE = [
  { label: 'A', n: 100, s: 2 },
  { label: 'B', n: 100, s: 9 },
  { label: 'C', n: 200, s: 6 },
  { label: 'D', n: 200, s: 16 },
  { label: 'E', n: 500, s: 15 },
  { label: 'F', n: 500, s: 40 },
  { label: 'G', n: 1000, s: 35 },
  { label: 'H', n: 1000, s: 75 },
];

const THETA_N = 401;
const theta = Float64Array.from({ length: THETA_N }, (_, i) => (i * 0.15) / (THETA_N - 1));
const KAPPA_GRID = d3.range(-1, 4.01, 0.05).map((log10) => Math.pow(10, log10));

export default function HierarchicalShrinkageDemo() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const eb = useMemo(() => hierarchicalBetaBinomialEB(FIXTURE), []);
  const [logKappa, setLogKappa] = useState(Math.log10(eb.kappaHat));

  const kappa = Math.pow(10, logKappa);
  const omega = eb.omegaHat;

  const W = Math.max(360, Math.min(width || 720, 800));
  const H = 380;
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };
  const colW = (W - margin.left - margin.right) / 2 - 18;

  const shrinkageCurves = useMemo(() => {
    return FIXTURE.map((g) => {
      const means = KAPPA_GRID.map((k) => {
        const alpha = k * omega + g.s;
        const beta = k * (1 - omega) + (g.n - g.s);
        return alpha / (alpha + beta);
      });
      return { label: g.label, n: g.n, mle: g.s / g.n, means };
    });
  }, [omega]);

  const perGroupDensities = useMemo(() => {
    return FIXTURE.map((g) => {
      const alpha = kappa * omega + g.s;
      const beta = kappa * (1 - omega) + (g.n - g.s);
      const dens = new Float64Array(THETA_N);
      for (let i = 0; i < THETA_N; i++) dens[i] = betaPdf(theta[i], alpha, beta);
      return { label: g.label, mle: g.s / g.n, density: dens, postMean: alpha / (alpha + beta) };
    });
  }, [kappa, omega]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
      const innerH = H - margin.top - margin.bottom;

      // ---------- Left panel: shrinkage curves ----------
      const gL = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xL = d3.scaleLog().domain([0.1, 10000]).range([0, colW]);
      const yL = d3.scaleLinear().domain([0, 0.12]).range([innerH, 0]);
      const palette = d3.scaleOrdinal<string, string>().domain(FIXTURE.map((g) => g.label))
        .range(d3.schemeCategory10);
      shrinkageCurves.forEach((c) => {
        gL.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => xL(KAPPA_GRID[i]))
              .y((v) => yL(v))(c.means) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', palette(c.label))
          .attr('stroke-width', 1.3)
          .attr('opacity', 0.85);
        // Endpoint labels
        gL.append('text')
          .attr('x', -4)
          .attr('y', yL(c.mle))
          .attr('text-anchor', 'end')
          .attr('font-size', 9)
          .style('fill', palette(c.label))
          .text(c.label);
      });
      // EB kappa-hat marker
      gL.append('line')
        .attr('x1', xL(eb.kappaHat))
        .attr('x2', xL(eb.kappaHat))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#000')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3');
      gL.append('text')
        .attr('x', xL(eb.kappaHat))
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .attr('font-size', 10)
        .style('fill', 'var(--color-text)')
        .text(`κ̂ = ${eb.kappaHat.toFixed(1)}`);
      // Current kappa marker
      gL.append('line')
        .attr('x1', xL(kappa))
        .attr('x2', xL(kappa))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', COLOR_VARIANT_B)
        .attr('stroke-width', 1.5);
      // Grand-mean line
      gL.append('line')
        .attr('x1', 0)
        .attr('x2', colW)
        .attr('y1', yL(omega))
        .attr('y2', yL(omega))
        .attr('stroke', COLOR_LIKELIHOOD)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3 3');
      gL.append('text')
        .attr('x', 4)
        .attr('y', yL(omega) - 4)
        .attr('font-size', 9)
        .style('fill', COLOR_LIKELIHOOD)
        .text(`grand mean ω̂ = ${omega.toFixed(3)}`);
      gL.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xL).ticks(5, '~s'));
      gL.append('g').call(d3.axisLeft(yL).ticks(5));
      gL.append('text')
        .attr('x', colW / 2)
        .attr('y', innerH + 28)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text('κ (log scale)');
      gL.append('text')
        .attr('x', colW / 2)
        .attr('y', -22)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .style('fill', 'var(--color-text)')
        .text('Per-group posterior mean vs κ');

      // ---------- Right panel: per-group densities at chosen kappa ----------
      const gR = svg.append('g').attr('transform', `translate(${margin.left + colW + 36}, ${margin.top})`);
      const xR = d3.scaleLinear().domain([0, 0.12]).range([0, colW]);
      let ymaxR = 0;
      perGroupDensities.forEach((d) => {
        for (let i = 0; i < d.density.length; i++) if (d.density[i] > ymaxR) ymaxR = d.density[i];
      });
      const yR = d3.scaleLinear().domain([0, ymaxR * 1.05]).range([innerH, 0]);
      perGroupDensities.forEach((d) => {
        gR.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => xR(theta[i]))
              .y((v) => yR(v))
              .curve(d3.curveMonotoneX)(d.density as unknown as number[]) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', palette(d.label))
          .attr('stroke-width', 1.2)
          .attr('opacity', 0.85);
        // MLE triangle
        gR.append('path')
          .attr(
            'd',
            d3.symbol().type(d3.symbolTriangle).size(45)() ?? '',
          )
          .attr('transform', `translate(${xR(d.mle)}, ${innerH - 6})`)
          .attr('fill', palette(d.label));
      });
      // Grand-mean line
      gR.append('line')
        .attr('x1', xR(omega))
        .attr('x2', xR(omega))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', COLOR_LIKELIHOOD)
        .attr('stroke-width', 1.4);
      gR.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xR).ticks(5));
      gR.append('g').call(d3.axisLeft(yR).ticks(5));
      gR.append('text')
        .attr('x', colW / 2)
        .attr('y', innerH + 28)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text(`θ_g  (κ = ${kappa.toFixed(1)}, triangles = MLEs)`);
      gR.append('text')
        .attr('x', colW / 2)
        .attr('y', -22)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .style('fill', 'var(--color-text)')
        .text('Per-group posteriors at chosen κ');
    },
    [shrinkageCurves, perGroupDensities, eb, kappa, omega, W, H],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Hierarchical Beta-Binomial shrinkage demo" />
      <label style={{ display: 'block', marginTop: '0.5rem', fontSize: '0.85rem' }}>
        log₁₀(κ): <strong>{logKappa.toFixed(2)}</strong> → κ ≈ {kappa.toFixed(1)}
        {' '}
        <em>(EB estimate κ̂ ≈ {eb.kappaHat.toFixed(1)}; small κ ≈ no pooling, large κ ≈ complete pooling)</em>
        <input
          type="range"
          min={-1}
          max={4}
          step={0.05}
          value={logKappa}
          onChange={(e) => setLogKappa(Number(e.target.value))}
          aria-label="log10 kappa"
          style={{ width: '100%' }}
        />
      </label>
    </div>
  );
}

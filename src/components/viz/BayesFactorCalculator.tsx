import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  logBayesFactorNormalNormal,
  bayesFactorOptimalTauSq,
  normalPdf,
  COLOR_VARIANT_A,
  COLOR_LIKELIHOOD,
  COLOR_BAND,
} from './shared/bayesian-inference';

// =============================================================================
// BayesFactorCalculator — §10 interactive.
//
// Top panel: log10 B_10(τ) vs prior width τ (log scale). Closed form from
// brief §10.3:
//   log B_10(τ) = n² ȳ² τ² / [2(1 + n τ²)] - 0.5 log(1 + n τ²)
// with Jeffreys' substantial / strong / decisive thresholds drawn as
// horizontal reference lines and the interior maximum marked.
//
// Bottom row: three small panels showing the prior π(μ | τ) and likelihood
// p(y | μ) for the chosen ȳ at three τ values — too narrow, optimal, too
// wide — making the Occam mechanic visible.
//
// Pedagogical takeaway: model complexity in the Bayesian frame is paid for
// in prior mass spread on parameter values that don't fit the data.
// =============================================================================

const TAU_LOG_GRID = d3.range(-2, 2.01, 0.02).map((l) => Math.pow(10, l));

export default function BayesFactorCalculator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [yBar, setYBar] = useState(0.3);
  const [n, setN] = useState(30);

  const W = Math.max(360, Math.min(width || 720, 800));
  const H = 520;
  const margin = { top: 30, right: 14, bottom: 36, left: 50 };

  const tauOpt = useMemo(() => Math.sqrt(Math.max(bayesFactorOptimalTauSq(yBar, n), 1e-12)), [yBar, n]);
  const log10Curve = useMemo(
    () => TAU_LOG_GRID.map((tau) => logBayesFactorNormalNormal(yBar, n, tau) / Math.log(10)),
    [yBar, n],
  );

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

      const innerW = W - margin.left - margin.right;
      const topH = 260;
      const botH = 160;

      // ---------- Top: log10 B_10(τ) curve ----------
      const gTop = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const x = d3.scaleLog().domain([0.01, 100]).range([0, innerW]);
      const yExtent = d3.extent(log10Curve) as [number, number];
      const ymax = Math.max(yExtent[1], Math.log10(100));
      const ymin = Math.min(yExtent[0], -Math.log10(100));
      const y = d3.scaleLinear().domain([ymin, ymax]).range([topH, 0]);

      // Jeffreys thresholds (log10 scale)
      const thresholds = [
        { v: Math.log10(3), label: 'substantial', color: '#777' },
        { v: Math.log10(10), label: 'strong', color: '#555' },
        { v: Math.log10(100), label: 'decisive', color: '#333' },
      ];
      thresholds.forEach((t) => {
        gTop.append('line')
          .attr('x1', 0)
          .attr('x2', innerW)
          .attr('y1', y(t.v))
          .attr('y2', y(t.v))
          .attr('stroke', t.color)
          .attr('stroke-width', 0.8)
          .attr('stroke-dasharray', '2 3')
          .attr('opacity', 0.7);
        gTop.append('text')
          .attr('x', innerW - 4)
          .attr('y', y(t.v) - 2)
          .attr('text-anchor', 'end')
          .attr('font-size', 9)
          .style('fill', t.color)
          .text(t.label);
      });

      // Curve
      gTop.append('path')
        .attr(
          'd',
          d3
            .line<number>()
            .x((_, i) => x(TAU_LOG_GRID[i]))
            .y((v) => y(v))(log10Curve) ?? '',
        )
        .attr('fill', 'none')
        .attr('stroke', COLOR_BAND)
        .attr('stroke-width', 2);

      // Zero reference
      gTop.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', y(0))
        .attr('y2', y(0))
        .attr('stroke', 'var(--color-text)')
        .attr('stroke-width', 1);

      // Mark interior maximum
      if (tauOpt > 0) {
        gTop.append('circle')
          .attr('cx', x(tauOpt))
          .attr('cy', y(logBayesFactorNormalNormal(yBar, n, tauOpt) / Math.log(10)))
          .attr('r', 5)
          .attr('fill', COLOR_BAND);
        gTop.append('text')
          .attr('x', x(tauOpt))
          .attr('y', y(logBayesFactorNormalNormal(yBar, n, tauOpt) / Math.log(10)) - 10)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .style('fill', 'var(--color-text)')
          .text(`τ* = ${tauOpt.toFixed(3)}, log₁₀ B = ${(logBayesFactorNormalNormal(yBar, n, tauOpt) / Math.log(10)).toFixed(3)}`);
      }

      gTop.append('g').attr('transform', `translate(0, ${topH})`).call(d3.axisBottom(x).ticks(5, '~s'));
      gTop.append('g').call(d3.axisLeft(y).ticks(6));
      gTop.append('text')
        .attr('x', innerW / 2)
        .attr('y', topH + 28)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text('Prior width τ (log scale)');
      gTop.append('text')
        .attr('x', -topH / 2)
        .attr('y', -36)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text('log₁₀ B₁₀(τ)');

      // ---------- Bottom: three prior/likelihood panels at chosen τ values ----------
      const tauPoints = [tauOpt * 0.3, tauOpt, tauOpt * 4];
      const panels = ['Too narrow', 'Optimal τ*', 'Too wide'];
      const cellW = innerW / 3 - 16;
      tauPoints.forEach((tau, k) => {
        const xCell = margin.left + k * (cellW + 24);
        const yCell = margin.top + topH + 60;
        const gP = svg.append('g').attr('transform', `translate(${xCell - margin.left}, ${yCell})`);
        const muRange: [number, number] = [-Math.max(2 * tau, 2 * Math.abs(yBar) + 1), Math.max(2 * tau, 2 * Math.abs(yBar) + 1)];
        const xm = d3.scaleLinear().domain(muRange).range([0, cellW]);
        const muGrid = d3.range(muRange[0], muRange[1], (muRange[1] - muRange[0]) / 200);
        const priorPdf = muGrid.map((mu) => normalPdf(mu, 0, tau));
        const likPdf = muGrid.map((mu) => normalPdf(yBar, mu, 1 / Math.sqrt(n)));
        const ymaxC = Math.max(...priorPdf, ...likPdf, 1e-6);
        const ym = d3.scaleLinear().domain([0, ymaxC * 1.1]).range([botH, 0]);
        gP.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => xm(muGrid[i]))
              .y((v) => ym(v))(priorPdf) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', COLOR_VARIANT_A)
          .attr('stroke-width', 1.5);
        gP.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => xm(muGrid[i]))
              .y((v) => ym(v))(likPdf) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', COLOR_LIKELIHOOD)
          .attr('stroke-width', 1.5);
        gP.append('line')
          .attr('x1', xm(yBar))
          .attr('x2', xm(yBar))
          .attr('y1', 0)
          .attr('y2', botH)
          .attr('stroke', 'var(--color-text)')
          .attr('stroke-width', 0.8)
          .attr('stroke-dasharray', '2 3');
        gP.append('text')
          .attr('x', cellW / 2)
          .attr('y', -6)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .attr('font-weight', 600)
          .style('fill', 'var(--color-text)')
          .text(`${panels[k]} (τ = ${tau.toFixed(3)})`);
        gP.append('g').attr('transform', `translate(0, ${botH})`).call(d3.axisBottom(xm).ticks(3).tickSizeOuter(0));
      });
    },
    [log10Curve, tauOpt, yBar, n, W, H],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Bayes factor calculator" />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(160px, 1fr))',
          gap: '0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.85rem',
        }}
      >
        <label>
          Effect size ȳ: <strong>{yBar.toFixed(2)}</strong>
          <input
            type="range"
            min={0.05}
            max={1.5}
            step={0.01}
            value={yBar}
            onChange={(e) => setYBar(Number(e.target.value))}
            aria-label="effect size yBar"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          Sample size n: <strong>{n}</strong>
          <input
            type="range"
            min={5}
            max={500}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            aria-label="sample size n"
            style={{ width: '100%' }}
          />
        </label>
      </div>
      <p style={{ opacity: 0.7, marginTop: '0.5rem', fontSize: '0.85rem' }}>
        {`Closed-form optimum: τ*² = ȳ² − 1/n = ${(yBar * yBar - 1 / n).toFixed(4)} (positive when ȳ² > 1/n).`}
        For ȳ&nbsp;=&nbsp;0.30, n&nbsp;=&nbsp;30 the notebook reports log₁₀ B(τ*) ≈ 0.1495 (matches the curve&apos;s peak).
      </p>
    </div>
  );
}

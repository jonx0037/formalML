import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, gaussianFrom } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 10.2 — Product e-value blows up under positive correlation.
//
// Generate m = 5 correlated null p-values via Gaussian copula at pairwise
// correlation ρ ∈ [0, 1], apply Vovk-1/2 calibrator to get e-values, then
// plot the empirical *truncated* mean of E^prod and the empirical mean of
// E^AM vs ρ. The arithmetic mean stays at 1 (Theorem 10.1); the product's
// truncated mean climbs past 1 once ρ > 0, demonstrating that product
// merging is invalid without independence.
//
// Truncation: each replicate's E^prod is winsorized at 10⁶ before averaging
// (otherwise ρ ≈ 1 produces 10¹⁵+ outliers that overwhelm B = 3000 samples'
// MC noise). The truncation only biases DOWN the displayed curve — without
// it, the upward trend is even more dramatic. Labels reflect "trunc. mean"
// per copilot's PR #113 comment.
// -----------------------------------------------------------------------------

const M = 5;
const B_REPLICATES = 3000;
const RHO_GRID = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];

const MARGIN = { top: 26, right: 14, bottom: 38, left: 56 };
const PANEL_H = 320;

const COLOR_AM = 'var(--color-accent)';
const COLOR_PROD = '#c44545';

function vovk12(p: number): number {
  // Vovk-1/2 calibrator: 0.5 p^{-1/2}
  return 0.5 / Math.sqrt(Math.max(p, 1e-12));
}

interface Sim {
  amByRho: { rho: number; mean: number }[];
  prodByRho: { rho: number; mean: number }[];
}

function simulate(): Sim {
  const rng = mulberry32(20261601);
  const gauss = gaussianFrom(rng);

  const amByRho: { rho: number; mean: number }[] = [];
  const prodByRho: { rho: number; mean: number }[] = [];

  for (const rho of RHO_GRID) {
    let sumAM = 0;
    let sumProd = 0;
    for (let b = 0; b < B_REPLICATES; b++) {
      // Gaussian copula at pairwise corr ρ: common Z + sqrt(1-ρ) per-component noise
      const common = gauss();
      const ps: number[] = [];
      const es: number[] = [];
      for (let i = 0; i < M; i++) {
        const Z = Math.sqrt(rho) * common + Math.sqrt(1 - rho) * gauss();
        // Φ(Z) is uniform: p ~ Uniform under H_0
        const p = 0.5 * (1 + erf(Z / Math.SQRT2));
        ps.push(p);
        es.push(vovk12(p));
      }
      const am = es.reduce((s, v) => s + v, 0) / M;
      const prod = es.reduce((s, v) => s * v, 1);
      sumAM += am;
      sumProd += Math.min(prod, 1e6);
    }
    amByRho.push({ rho, mean: sumAM / B_REPLICATES });
    prodByRho.push({ rho, mean: sumProd / B_REPLICATES });
  }

  return { amByRho, prodByRho };
}

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-ax * ax);
  return sign * y;
}

export default function ProductBreaksUnderCorrelation() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const sim = useMemo(simulate, []);
  const [highlightRho, setHighlightRho] = useState(0.5);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yMax = Math.max(...sim.prodByRho.map((p) => p.mean));
      const yScale = d3.scaleLog().domain([Math.min(0.5, ...sim.amByRho.map((p) => p.mean) || [1]), Math.max(yMax * 1.5, 5)]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text)');

      // E[E] = 1 reference
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(1)).attr('y2', yScale(1)).style('stroke', 'var(--color-text-secondary)').attr('stroke-width', 1).attr('stroke-dasharray', '4 3');
      g.append('text').attr('x', innerW - 4).attr('y', yScale(1) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('E[E] = 1 (valid)');

      // AM line
      const lineAM = d3.line<{ rho: number; mean: number }>().x((d) => xScale(d.rho)).y((d) => yScale(Math.max(d.mean, 0.01)));
      g.append('path').datum(sim.amByRho).attr('d', lineAM as never).style('fill', 'none').style('stroke', COLOR_AM).attr('stroke-width', 2);
      g.append('g').selectAll('circle').data(sim.amByRho).join('circle').attr('cx', (d) => xScale(d.rho)).attr('cy', (d) => yScale(Math.max(d.mean, 0.01))).attr('r', 3.5).style('fill', COLOR_AM);

      // Product line
      g.append('path').datum(sim.prodByRho).attr('d', lineAM as never).style('fill', 'none').style('stroke', COLOR_PROD).attr('stroke-width', 2);
      g.append('g').selectAll('circle').data(sim.prodByRho).join('circle').attr('cx', (d) => xScale(d.rho)).attr('cy', (d) => yScale(Math.max(d.mean, 0.01))).attr('r', 3.5).style('fill', COLOR_PROD);

      // Highlight
      g.append('line').attr('x1', xScale(highlightRho)).attr('x2', xScale(highlightRho)).attr('y1', 0).attr('y2', innerH).style('stroke', 'var(--color-text)').attr('stroke-width', 1).attr('opacity', 0.3);

      // Legend
      g.append('text').attr('x', innerW - 4).attr('y', 12).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_AM).text('E_{H_0}[E^AM]');
      g.append('text').attr('x', innerW - 4).attr('y', 26).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_PROD).text('trunc. mean of E^prod (m=5, Vovk-½, winsorized at 10⁶)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('pairwise correlation ρ (Gaussian copula)');
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('E_{H_0}[merger] (log scale)');
    },
    [sim, w, highlightRho],
  );

  const amAtRho = sim.amByRho.find((d) => Math.abs(d.rho - highlightRho) < 0.05)?.mean ?? 1;
  const prodAtRho = sim.prodByRho.find((d) => Math.abs(d.rho - highlightRho) < 0.05)?.mean ?? 1;

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <span>Highlight ρ = <strong>{highlightRho.toFixed(2)}</strong></span>
          <input type="range" min={0} max={1} step={0.05} value={highlightRho} aria-label="Highlight correlation" onChange={(e) => setHighlightRho(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
          At ρ = {highlightRho.toFixed(2)}: E[E^AM] = <strong style={{ color: COLOR_AM }}>{amAtRho.toFixed(3)}</strong>, E[E^prod] = <strong style={{ color: COLOR_PROD }}>{prodAtRho.toFixed(3)}</strong>.
          {' '}AM ≤ 1 always (Theorem 10.1); the product's truncated mean already exceeds 1 for ρ &gt; 0 even after winsorization at 10⁶.
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

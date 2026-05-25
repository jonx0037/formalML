import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { singleLambdaSGHW, gaussianMixtureHW, lilHW } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 4.1 — Single-λ vs Gaussian-mixture envelope (the lower-envelope picture).
//
// Log-log axes (n ∈ [10, 10^6]):
//   - Six single-λ sub-Gaussian boundaries at λ ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}
//   - Gaussian-mixture (4.4) at three ρ² values
//   - LIL rate (dashed brown)
//   - Fixed-n Hoeffding band (dotted gray — anytime-invalid reference)
// -----------------------------------------------------------------------------

const MARGIN = { top: 28, right: 18, bottom: 42, left: 60 };
const PANEL_H = 400;
const LAMBDAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0];
const RHO_SQ_VALUES = [0.1, 1, 10];
const N_MIN = 10;
const N_MAX = 1e6;
const SIGMA = 1;

const COLOR_LAMBDA = '#9aa3ad';
const COLOR_MIX_VARIANTS = ['#0f6e56', '#1d4ed8', '#7c3aed'];
const COLOR_LIL = '#b45309';
const COLOR_FIXED = 'var(--color-text-secondary)';

interface Curve {
  n: number;
  hw: number;
}

function buildGrid(): number[] {
  const out: number[] = [];
  const lo = Math.log10(N_MIN);
  const hi = Math.log10(N_MAX);
  for (let i = 0; i < 100; i++) out.push(10 ** (lo + ((hi - lo) * i) / 99));
  return out;
}

export default function SingleLambdaVsMixtureEnvelope() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [delta, setDelta] = useState(0.05);

  const ng = useMemo(buildGrid, []);
  const data = useMemo(() => {
    const single: Curve[][] = LAMBDAS.map((lam) => ng.map((n) => ({ n, hw: singleLambdaSGHW(n, lam, delta, SIGMA) })));
    const mixes: Curve[][] = RHO_SQ_VALUES.map((r) => ng.map((n) => ({ n, hw: gaussianMixtureHW(n, r, delta, SIGMA) })));
    const lil: Curve[] = ng.map((n) => ({ n, hw: lilHW(n, SIGMA) })).filter((d) => d.hw > 0);
    const fixed: Curve[] = ng.map((n) => ({ n, hw: SIGMA * Math.sqrt((2 * Math.log(2 / delta)) / n) }));
    return { single, mixes, lil, fixed };
  }, [ng, delta]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([N_MIN, N_MAX]).range([0, innerW]);
      const yScale = d3.scaleLog().domain([0.001, 10]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text)');

      const line = d3.line<Curve>().x((d) => xScale(d.n)).y((d) => yScale(Math.max(d.hw, 1e-4)));

      // fixed-n
      g.append('path').datum(data.fixed).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_FIXED).attr('stroke-width', 1.2).attr('stroke-dasharray', '2 3');

      // single-λ lines
      data.single.forEach((curve, i) => {
        g.append('path').datum(curve).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_LAMBDA).attr('stroke-width', 1.0).attr('opacity', 0.8);
        const minPt = curve.reduce((acc, p) => (p.hw < acc.hw ? p : acc), curve[0]);
        g.append('text').attr('x', xScale(minPt.n) + 4).attr('y', yScale(minPt.hw) - 4).style('font-family', 'var(--font-mono)').style('font-size', '10px').style('fill', COLOR_LAMBDA).text(`λ=${LAMBDAS[i]}`);
      });

      // mixture lines
      data.mixes.forEach((curve, i) => {
        g.append('path').datum(curve).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_MIX_VARIANTS[i]).attr('stroke-width', 2.0);
      });

      // LIL
      g.append('path').datum(data.lil).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_LIL).attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');

      // legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 230}, 4)`);
      const entries = [
        { color: COLOR_FIXED, dash: '2 3', label: 'fixed-n Hoeffding (invalid)' },
        { color: COLOR_LAMBDA, dash: '', label: 'single-λ sub-Gaussian' },
        ...RHO_SQ_VALUES.map((r, i) => ({ color: COLOR_MIX_VARIANTS[i], dash: '', label: `Gaussian mix ρ²=${r}` })),
        { color: COLOR_LIL, dash: '5 4', label: 'LIL rate' },
      ];
      entries.forEach((e, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 22).attr('y1', i * 15 + 8).attr('y2', i * 15 + 8).style('stroke', e.color).attr('stroke-width', 1.4).attr('stroke-dasharray', e.dash || 'none');
        legend.append('text').attr('x', 28).attr('y', i * 15 + 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(e.label);
      });

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('n (log scale)');
      g.append('text').attr('x', -innerH / 2).attr('y', -48).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('half-width (log scale)');
    },
    [data, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <span>δ = <strong>{delta.toFixed(3)}</strong></span>
          <input type="range" min={0.01} max={0.20} step={0.005} value={delta} aria-label="Confidence level" onChange={(e) => setDelta(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
          σ = {SIGMA}. The mixture (4.4) tracks the lower envelope of single-λ boundaries; LIL is the asymptotic floor.
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

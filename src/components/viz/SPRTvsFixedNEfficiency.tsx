import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// -----------------------------------------------------------------------------
// Viz 3.2 — SPRT vs fixed-n Neyman–Pearson efficiency.
//
// Single panel: bars showing the fixed-n NP sample size n_NP and the SPRT
// Wald-predicted E[τ] for effect sizes |μ_1 − μ_0|/σ ∈ {0.1, 0.15, 0.2, 0.3,
// 0.5, 0.75, 1.0}. Ratio E[τ]/n_NP is the Wald–Wolfowitz efficiency.
//
// Slider: α = β ∈ {0.01, 0.025, 0.05, 0.10} (symmetric errors keep the bar
// chart legible — one error param drives both).
//
// All values are closed-form — no MC. Matches notebook cell 11.
// -----------------------------------------------------------------------------

const SIGMA = 1;
const EFFECT_SIZES = [0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0];
const ERROR_OPTIONS = [0.01, 0.025, 0.05, 0.10];

const PANEL_H = 320;
const MARGIN = { top: 28, right: 18, bottom: 50, left: 64 };

const COLOR_NP = '#9aa3ad';
const COLOR_SPRT = 'var(--color-accent)';

function normalQuantile(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

interface Row {
  delta: number;
  nNP: number;
  eTauWald: number;
}

function computeRows(alpha: number, beta: number): Row[] {
  const za = normalQuantile(1 - alpha);
  const zb = normalQuantile(1 - beta);
  return EFFECT_SIZES.map((delta) => {
    const KL = (delta * delta) / (2 * SIGMA * SIGMA);
    const logA = Math.log((1 - beta) / alpha);
    const logB = Math.log(beta / (1 - alpha));
    const eTauWald = (alpha * logA + (1 - alpha) * logB) / -KL;
    const nNP = Math.pow(((za + zb) * SIGMA) / delta, 2);
    return { delta, nNP, eTauWald };
  });
}

export default function SPRTvsFixedNEfficiency() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [errIdx, setErrIdx] = useState(2); // 0.05

  const alpha = ERROR_OPTIONS[errIdx];
  const beta = ERROR_OPTIONS[errIdx];

  const rows = useMemo(() => computeRows(alpha, beta), [alpha, beta]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleBand<number>().domain(rows.map((r) => r.delta)).range([0, innerW]).padding(0.25);
      const subScale = d3.scaleBand<string>().domain(['nNP', 'sprt']).range([0, xScale.bandwidth()]).padding(0.1);
      const yMax = Math.max(...rows.map((r) => r.nNP)) * 1.1;
      const yScale = d3.scaleLog().domain([Math.max(1, Math.min(...rows.map((r) => r.eTauWald)) * 0.5), yMax]).range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).tickFormat((d) => `Δ=${(d as number).toFixed(2)}`))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text)');

      rows.forEach((r) => {
        const groupX = xScale(r.delta) ?? 0;
        // NP bar
        g.append('rect')
          .attr('x', groupX + (subScale('nNP') ?? 0))
          .attr('y', yScale(r.nNP))
          .attr('width', subScale.bandwidth())
          .attr('height', innerH - yScale(r.nNP))
          .style('fill', COLOR_NP)
          .style('opacity', 0.85);
        // SPRT bar
        g.append('rect')
          .attr('x', groupX + (subScale('sprt') ?? 0))
          .attr('y', yScale(r.eTauWald))
          .attr('width', subScale.bandwidth())
          .attr('height', innerH - yScale(r.eTauWald))
          .style('fill', COLOR_SPRT);

        // ratio label
        const ratio = r.eTauWald / r.nNP;
        g.append('text')
          .attr('x', groupX + xScale.bandwidth() / 2)
          .attr('y', yScale(Math.max(r.nNP, r.eTauWald)) - 6)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)')
          .style('font-size', '10px')
          .style('fill', 'var(--color-text-secondary)')
          .text(`${(ratio * 100).toFixed(0)}%`);
      });

      // legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 220}, 4)`);
      legend.append('rect').attr('width', 14).attr('height', 14).style('fill', COLOR_NP);
      legend.append('text').attr('x', 20).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('n_NP (fixed-n)');
      legend.append('rect').attr('x', 130).attr('width', 14).attr('height', 14).style('fill', COLOR_SPRT);
      legend.append('text').attr('x', 150).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('E[τ] (SPRT)');

      // axis labels
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 38).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('effect size Δ = |μ_1 − μ_0|/σ');
      g.append('text').attr('x', -innerH / 2).attr('y', -50).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('sample size (log scale)');
    },
    [rows, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '220px' }}>
          <span>
            α = β = <strong>{alpha.toFixed(3)}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={ERROR_OPTIONS.length - 1}
            step={1}
            value={errIdx}
            aria-label="Symmetric error rate"
            onChange={(e) => setErrIdx(Number(e.target.value))}
          />
        </label>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            color: 'var(--color-text-secondary)',
            padding: '6px 10px',
            border: '1px solid var(--color-border)',
            borderRadius: '4px',
          }}
        >
          Mean ratio E[τ]/n_NP = {(rows.reduce((s, r) => s + r.eTauWald / r.nNP, 0) / rows.length * 100).toFixed(1)}%
          {' '}— the Wald–Wolfowitz efficiency gap.
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

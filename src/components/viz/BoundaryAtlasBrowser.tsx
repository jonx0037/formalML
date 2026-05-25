import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gaussianMixtureHW,
  lilHW,
  subBernoulliCSEndpoints,
  subExponentialHW,
} from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 5.1 — Boundary atlas browser.
//
// Dropdown: distribution class. Each class gets a different anytime-valid CS
// vs the same anytime-INVALID fixed-n band reference and the LIL rate.
// -----------------------------------------------------------------------------

type DistClass = 'sub-Gaussian' | 'Poisson' | 'Bernoulli @ 0.05' | 'Bernoulli @ 0.5';
const CLASSES: DistClass[] = ['sub-Gaussian', 'Poisson', 'Bernoulli @ 0.05', 'Bernoulli @ 0.5'];

const MARGIN = { top: 28, right: 18, bottom: 42, left: 60 };
const PANEL_H = 380;
const N_MIN = 100;
const N_MAX = 1e6;
const SIGMA = 1;

const COLOR_FIXED = 'var(--color-text-secondary)';
const COLOR_MIX = 'var(--color-accent)';
const COLOR_LIL = '#b45309';

interface Curve {
  n: number;
  hw: number;
}

function buildGrid(): number[] {
  const out: number[] = [];
  const lo = Math.log10(N_MIN);
  const hi = Math.log10(N_MAX);
  for (let i = 0; i < 90; i++) out.push(10 ** (lo + ((hi - lo) * i) / 89));
  return out;
}

function fixedNHoeffdingHW(n: number, delta: number, sigma: number): number {
  return sigma * Math.sqrt((2 * Math.log(2 / delta)) / n);
}

function computeClassCurves(cls: DistClass, delta: number, ng: number[]): { mix: Curve[]; fixed: Curve[]; lil: Curve[]; label: string } {
  switch (cls) {
    case 'sub-Gaussian':
      return {
        mix: ng.map((n) => ({ n, hw: gaussianMixtureHW(n, 1, delta, SIGMA) })),
        fixed: ng.map((n) => ({ n, hw: fixedNHoeffdingHW(n, delta, SIGMA) })),
        lil: ng.map((n) => ({ n, hw: lilHW(n, SIGMA) })).filter((d) => d.hw > 0),
        label: 'σ = 1; Gaussian mixture (4.4) at ρ² = 1',
      };
    case 'Poisson':
      // λ_Poisson = 1, so ν² = 1, c = 1. Fixed-n band uses the same √(2 ν²/n).
      return {
        mix: ng.map((n) => ({ n, hw: subExponentialHW(n, 1, 1, delta) })),
        fixed: ng.map((n) => ({ n, hw: fixedNHoeffdingHW(n, delta, 1) })),
        lil: ng.map((n) => ({ n, hw: lilHW(n, 1) })).filter((d) => d.hw > 0),
        label: 'Poisson(1): ν² = 1, c = 1. Bennett mixture (5.9)',
      };
    case 'Bernoulli @ 0.05': {
      const muObs = 0.05;
      return {
        mix: ng.map((n) => {
          const [lo, hi] = subBernoulliCSEndpoints(n, muObs, delta);
          return { n, hw: Math.max(muObs - lo, hi - muObs) };
        }),
        // Hoeffding for Bernoulli uses σ² = 1/4 (worst case)
        fixed: ng.map((n) => ({ n, hw: fixedNHoeffdingHW(n, delta, 0.5) })),
        lil: ng.map((n) => ({ n, hw: lilHW(n, Math.sqrt(muObs * (1 - muObs))) })).filter((d) => d.hw > 0),
        label: 'Bernoulli(0.05): Beta-mixture (5.14), σ̂² = 0.0475 for LIL',
      };
    }
    case 'Bernoulli @ 0.5': {
      const muObs = 0.5;
      return {
        mix: ng.map((n) => {
          const [lo, hi] = subBernoulliCSEndpoints(n, muObs, delta);
          return { n, hw: Math.max(muObs - lo, hi - muObs) };
        }),
        fixed: ng.map((n) => ({ n, hw: fixedNHoeffdingHW(n, delta, 0.5) })),
        lil: ng.map((n) => ({ n, hw: lilHW(n, 0.5) })).filter((d) => d.hw > 0),
        label: 'Bernoulli(0.5): Beta-mixture (5.14), σ̂² = 0.25 for LIL',
      };
    }
  }
}

export default function BoundaryAtlasBrowser() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [cls, setCls] = useState<DistClass>('sub-Gaussian');
  const [delta, setDelta] = useState(0.05);

  const ng = useMemo(buildGrid, []);
  const data = useMemo(() => computeClassCurves(cls, delta, ng), [cls, delta, ng]);
  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([N_MIN, N_MAX]).range([0, innerW]);
      const yScale = d3.scaleLog().domain([0.0005, 1.5]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text)');

      const line = d3.line<Curve>().x((d) => xScale(d.n)).y((d) => yScale(Math.max(d.hw, 5e-4)));

      g.append('path').datum(data.fixed).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_FIXED).attr('stroke-width', 1.2).attr('stroke-dasharray', '2 3');
      g.append('path').datum(data.mix).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_MIX).attr('stroke-width', 2.2);
      g.append('path').datum(data.lil).attr('d', line as never).style('fill', 'none').style('stroke', COLOR_LIL).attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');

      // legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 4)`);
      [
        { color: COLOR_FIXED, dash: '2 3', label: 'fixed-n (invalid)' },
        { color: COLOR_MIX, dash: '', label: 'anytime-valid CS' },
        { color: COLOR_LIL, dash: '5 4', label: 'LIL rate' },
      ].forEach((e, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 22).attr('y1', i * 16 + 8).attr('y2', i * 16 + 8).style('stroke', e.color).attr('stroke-width', e.color === COLOR_MIX ? 2.2 : 1.4).attr('stroke-dasharray', e.dash || 'none');
        legend.append('text').attr('x', 28).attr('y', i * 16 + 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(e.label);
      });

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('n (log scale)');
      g.append('text').attr('x', -innerH / 2).attr('y', -48).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('half-width (log scale)');
    },
    [data, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '220px' }}>
          <span>Distribution class</span>
          <select value={cls} onChange={(e) => setCls(e.target.value as DistClass)} style={{ padding: '4px 8px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
            {CLASSES.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>δ = <strong>{delta.toFixed(3)}</strong></span>
          <input type="range" min={0.01} max={0.20} step={0.005} value={delta} aria-label="Confidence level" onChange={(e) => setDelta(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', maxWidth: '420px' }}>
          {data.label}
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

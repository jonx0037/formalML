import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  ivDGP,
  mulberry32,
  waldIV,
} from './shared/causal-inference-methods';

// =============================================================================
// IVStrengthExplorer — §9 weak-IV pathology.
//
// Instrument-strength slider π ∈ [0.05, 1.5]. At each π we run B=200 reps at
// n=500 and plot the distribution of Wald estimates (clipped to [-3, 5] for
// visualization) and the median first-stage F-statistic.
// =============================================================================

const HEIGHT = 360;
const B_REP = 200;
const N = 500;

export default function IVStrengthExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [piDisplay, setPiDisplay] = useState(0.5);
  const [pi, setPi] = useState(0.5);

  const { taus, medianF, iqr } = useMemo(() => {
    const taus = new Float64Array(B_REP);
    const Fs = new Float64Array(B_REP);
    for (let b = 0; b < B_REP; b++) {
      const rng = mulberry32(20260512 + b * 41);
      const { Z, D, Y } = ivDGP(N, pi, 1.0, rng);
      const r = waldIV(Z, D, Y);
      taus[b] = isFinite(r.tau) ? Math.max(-3, Math.min(5, r.tau)) : 0;
      Fs[b] = isFinite(r.firstStageF) ? r.firstStageF : 0;
    }
    const sortedT = taus.slice().sort();
    const sortedF = Fs.slice().sort();
    const medF = sortedF[Math.floor(B_REP / 2)];
    const q1 = sortedT[Math.floor(B_REP / 4)];
    const q3 = sortedT[Math.floor((3 * B_REP) / 4)];
    return { taus, medianF: medF, iqr: q3 - q1 };
  }, [pi]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 24, bottom: 40, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const arr = Array.from(taus);
      const xScale = d3.scaleLinear().domain([-3, 5]).range([0, W]);
      const bins = d3.bin().domain([-3, 5]).thresholds(36)(arr);
      const yScale = d3.scaleLinear().domain([0, d3.max(bins, (d) => d.length)! * 1.1 + 1]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(4));
      g.selectAll('rect.bar').data(bins).join('rect').attr('class', 'bar')
        .attr('x', (d) => xScale(d.x0!) + 1).attr('y', (d) => yScale(d.length))
        .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
        .attr('height', (d) => H - yScale(d.length))
        .style('fill', medianF >= 10 ? '#1f4e79' : '#c0504d').style('opacity', 0.7);
      g.append('line').attr('x1', xScale(1)).attr('x2', xScale(1)).attr('y1', 0).attr('y2', H)
        .style('stroke', '#3a6e3a').style('stroke-width', 2);
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('Wald τ̂ (clipped to [−3, 5])');
    },
    [containerWidth, taus, medianF],
  );

  const commitPi = () => setPi(piDisplay);

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ display: 'flex', gap: '1.25rem', marginBottom: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          Instrument strength π =
          <input type="range" min={0.05} max={1.5} step={0.05} value={piDisplay}
            onChange={(e) => setPiDisplay(parseFloat(e.target.value))}
            onMouseUp={commitPi} onTouchEnd={commitPi} onKeyUp={commitPi}
            aria-label="Instrument strength" style={{ width: 180 }} />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{piDisplay.toFixed(2)}</span>
        </label>
        <span style={{ fontSize: '0.85rem', color: medianF >= 10 ? 'var(--color-text)' : '#c0504d' }}>
          median first-stage F = <span style={{ fontFamily: 'var(--font-mono)' }}>{medianF.toFixed(1)}</span>
          {medianF < 10 ? ' (weak — Wald distribution is heavy-tailed)' : ' (strong)'}
        </span>
        <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
          IQR(τ̂) = <span style={{ fontFamily: 'var(--font-mono)' }}>{iqr.toFixed(3)}</span>
        </span>
      </div>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
    </div>
  );
}

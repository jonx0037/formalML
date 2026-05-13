import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  dmlCrossFit,
  kernelNuisanceFitter,
  lassoNuisanceFitter,
  mean,
  mulberry32,
  oracleNuisanceFitter,
  robinsonDGP,
} from './shared/causal-inference-methods';

// =============================================================================
// DMLComparison — §8 (DML cross-fitting vs naive plug-in).
//
// Slider: number of replicates B ∈ {15, 30, 50}. Selectors: nuisance class
// (oracle / lasso / kernel-proxy), cross-fit toggle.
//
// Each viz call runs B replicates at n=600 with the selected nuisance. The
// resulting τ̂ distribution is plotted vs the τ=1 line. Cross-fitting flag
// controls whether folds are used or the full sample is reused for nuisance
// and score.
// =============================================================================

const HEIGHT = 360;
const N = 600;

type Nuisance = 'oracle' | 'lasso' | 'kernel';

export default function DMLComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nuisance, setNuisance] = useState<Nuisance>('lasso');
  const [crossFit, setCrossFit] = useState(true);
  const [bDisplay, setBDisplay] = useState(20);
  const [B, setB] = useState(20);

  const { taus, meanTau, sdTau } = useMemo(() => {
    const fitter = nuisance === 'oracle' ? oracleNuisanceFitter
                  : nuisance === 'lasso' ? lassoNuisanceFitter
                  : kernelNuisanceFitter;
    const K = crossFit ? 4 : 1;
    const taus = new Float64Array(B);
    for (let b = 0; b < B; b++) {
      const rng = mulberry32(20260512 + b * 31 + (crossFit ? 0 : 999));
      const s = robinsonDGP(N, 10, 1.0, rng);
      // With K=1 (no cross-fit), train and test are the same sample → naive plug-in.
      if (K === 1) {
        const allIdx = Array.from({ length: N }, (_, i) => i);
        const { eHat, mu0Hat, mu1Hat } = fitter(s.X, s.D, s.Y, N, s.p, allIdx, allIdx, rng);
        // Compute AIPW score on this single fit.
        let sum = 0;
        for (let i = 0; i < N; i++) {
          const e = Math.min(Math.max(eHat[i], 0.025), 0.975);
          sum += mu1Hat[i] - mu0Hat[i]
               + (s.D[i] * (s.Y[i] - mu1Hat[i])) / e
               - ((1 - s.D[i]) * (s.Y[i] - mu0Hat[i])) / (1 - e);
        }
        taus[b] = sum / N;
      } else {
        taus[b] = dmlCrossFit(s.X, s.D, s.Y, N, s.p, fitter, K, rng).tau;
      }
    }
    const m = mean(taus);
    let v = 0;
    for (let b = 0; b < B; b++) v += (taus[b] - m) ** 2;
    return { taus, meanTau: m, sdTau: Math.sqrt(v / Math.max(B - 1, 1)) };
  }, [B, nuisance, crossFit]);

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
      const lo = Math.min(0.5, d3.min(arr)!);
      const hi = Math.max(1.5, d3.max(arr)!);
      const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
      const bins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(20)(arr);
      const yScale = d3.scaleLinear().domain([0, d3.max(bins, (d) => d.length)! * 1.1 + 1]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(4));
      g.selectAll('rect.bar').data(bins).join('rect').attr('class', 'bar')
        .attr('x', (d) => xScale(d.x0!) + 1).attr('y', (d) => yScale(d.length))
        .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
        .attr('height', (d) => H - yScale(d.length))
        .style('fill', crossFit ? '#1f4e79' : '#c0504d').style('opacity', 0.7);
      g.append('line').attr('x1', xScale(1)).attr('x2', xScale(1)).attr('y1', 0).attr('y2', H)
        .style('stroke', '#3a6e3a').style('stroke-width', 2);
      g.append('line').attr('x1', xScale(meanTau)).attr('x2', xScale(meanTau)).attr('y1', 0).attr('y2', H)
        .style('stroke', '#7b3c10').style('stroke-width', 1.5).style('stroke-dasharray', '4 4');
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('DML τ̂');
    },
    [containerWidth, taus, meanTau, crossFit],
  );

  const commitB = () => setB(bDisplay);

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ display: 'flex', gap: '1.25rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          Nuisance =
          <select value={nuisance} onChange={(e) => setNuisance(e.target.value as Nuisance)}
            style={{ padding: '0.2rem 0.4rem' }}>
            <option value="oracle">Oracle (correct features)</option>
            <option value="lasso">Lasso CV</option>
            <option value="kernel">Kernel smoother (RF proxy)</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          <input type="checkbox" checked={crossFit} onChange={(e) => setCrossFit(e.target.checked)} />
          Cross-fitting (K = 4)
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          B =
          <input type="range" min={10} max={40} step={5} value={bDisplay}
            onChange={(e) => setBDisplay(parseInt(e.target.value))}
            onMouseUp={commitB} onTouchEnd={commitB} onKeyUp={commitB}
            aria-label="Number of replicates" style={{ width: 120 }} />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{bDisplay}</span>
        </label>
      </div>
      <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '0.4rem', flexWrap: 'wrap' }}>
        <span>mean τ̂ = <span style={{ fontFamily: 'var(--font-mono)' }}>{meanTau.toFixed(3)}</span></span>
        <span>SD = <span style={{ fontFamily: 'var(--font-mono)' }}>{sdTau.toFixed(3)}</span></span>
        <span>{crossFit ? 'Cross-fit DML score' : 'Naive plug-in (in-sample residuals → bias)'}</span>
      </div>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
    </div>
  );
}

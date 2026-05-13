import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  aipwEstimate,
  aipwScore,
  correctOutcomeFeatures,
  correctPropensityFeatures,
  linearFit,
  linearPredict,
  logisticFitPredict,
  mean,
  mulberry32,
  robinsonDGP,
  selectRows,
  selectValues,
  tmleLinear,
} from './shared/causal-inference-methods';

// =============================================================================
// TMLEvsAIPW — §7 side-by-side AIPW and TMLE distributions with the EIF
// augmentation-mean diagnostic.
// Toggle: outcome correctly specified (correct features [sin X_1, X_2^2] vs raw X).
// =============================================================================

const HEIGHT = 360;
const B_REP = 150;
const N = 500;
const SM_BREAKPOINT = 640;

export default function TMLEvsAIPW() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [outCorrect, setOutCorrect] = useState(false);
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { aipwTaus, tmleTaus, aipwAugMean, tmleAugMean } = useMemo(() => {
    const a = new Float64Array(B_REP);
    const t = new Float64Array(B_REP);
    let augA = 0, augT = 0;
    for (let b = 0; b < B_REP; b++) {
      const rng = mulberry32(20260512 + b * 23);
      const s = robinsonDGP(N, 10, 1.0, rng);
      const { X, D, Y, n, p } = s;
      const propF = correctPropensityFeatures(X, n, p);
      const { ePred } = logisticFitPredict(propF, D, n, 3);
      const trIdx: number[] = [], ctIdx: number[] = [];
      for (let i = 0; i < n; i++) { if (D[i] === 1) trIdx.push(i); else ctIdx.push(i); }
      let mu0Hat: Float64Array, mu1Hat: Float64Array;
      if (outCorrect) {
        const F = correctOutcomeFeatures(X, n, p);
        const X1 = selectRows(F, trIdx, 2), Y1 = selectValues(Y, trIdx);
        const X0 = selectRows(F, ctIdx, 2), Y0 = selectValues(Y, ctIdx);
        const b1 = linearFit(X1, Y1, trIdx.length, 2);
        const b0 = linearFit(X0, Y0, ctIdx.length, 2);
        mu1Hat = linearPredict(b1, F, n, 2);
        mu0Hat = linearPredict(b0, F, n, 2);
      } else {
        const X1 = selectRows(X, trIdx, p), Y1 = selectValues(Y, trIdx);
        const X0 = selectRows(X, ctIdx, p), Y0 = selectValues(Y, ctIdx);
        const b1 = linearFit(X1, Y1, trIdx.length, p);
        const b0 = linearFit(X0, Y0, ctIdx.length, p);
        mu1Hat = linearPredict(b1, X, n, p);
        mu0Hat = linearPredict(b0, X, n, p);
      }
      a[b] = aipwEstimate(D, Y, ePred, mu0Hat, mu1Hat).tau;
      const tmleRes = tmleLinear(D, Y, ePred, mu0Hat, mu1Hat);
      t[b] = tmleRes.tau;
      // Augmentation mean = E[ D(Y-mu1)/e - (1-D)(Y-mu0)/(1-e) ].
      let aSum = 0, tSum = 0;
      for (let i = 0; i < n; i++) {
        const e = Math.min(Math.max(ePred[i], 1e-6), 1 - 1e-6);
        aSum += (D[i] * (Y[i] - mu1Hat[i])) / e - ((1 - D[i]) * (Y[i] - mu0Hat[i])) / (1 - e);
        tSum += (D[i] * (Y[i] - tmleRes.mu1Star[i])) / e - ((1 - D[i]) * (Y[i] - tmleRes.mu0Star[i])) / (1 - e);
      }
      augA += Math.abs(aSum / n);
      augT += Math.abs(tSum / n);
    }
    return { aipwTaus: a, tmleTaus: t, aipwAugMean: augA / B_REP, tmleAugMean: augT / B_REP };
  }, [outCorrect]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 20, right: 24, bottom: 40, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const allTaus = [...aipwTaus, ...tmleTaus];
      const lo = Math.min(0.5, d3.min(allTaus)!);
      const hi = Math.max(1.5, d3.max(allTaus)!);
      const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
      const aBins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(24)(Array.from(aipwTaus));
      const tBins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(24)(Array.from(tmleTaus));
      const maxCount = Math.max(d3.max(aBins, (d) => d.length)!, d3.max(tBins, (d) => d.length)!);
      const yScale = d3.scaleLinear().domain([0, maxCount * 1.1 + 1]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(4));
      g.selectAll('rect.aipw').data(aBins).join('rect').attr('class', 'aipw')
        .attr('x', (d) => xScale(d.x0!) + 1).attr('y', (d) => yScale(d.length))
        .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
        .attr('height', (d) => H - yScale(d.length))
        .style('fill', '#1f4e79').style('opacity', 0.5);
      g.selectAll('rect.tmle').data(tBins).join('rect').attr('class', 'tmle')
        .attr('x', (d) => xScale(d.x0!) + 1).attr('y', (d) => yScale(d.length))
        .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
        .attr('height', (d) => H - yScale(d.length))
        .style('fill', '#c0504d').style('opacity', 0.5);
      g.append('line').attr('x1', xScale(1)).attr('x2', xScale(1)).attr('y1', 0).attr('y2', H)
        .style('stroke', '#3a6e3a').style('stroke-width', 2);
      // Legend.
      const lg = g.append('g').attr('transform', `translate(${W - 130}, 0)`);
      lg.append('rect').attr('x', 0).attr('y', 0).attr('width', 14).attr('height', 10).style('fill', '#1f4e79').style('opacity', 0.5);
      lg.append('text').attr('x', 18).attr('y', 10).style('fill', 'var(--color-text)').style('font-size', '10px').text('AIPW');
      lg.append('rect').attr('x', 0).attr('y', 20).attr('width', 14).attr('height', 10).style('fill', '#c0504d').style('opacity', 0.5);
      lg.append('text').attr('x', 18).attr('y', 30).style('fill', 'var(--color-text)').style('font-size', '10px').text('TMLE');
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('Point estimate τ̂');
    },
    [containerWidth, aipwTaus, tmleTaus],
  );

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          <input type="checkbox" checked={outCorrect} onChange={(e) => setOutCorrect(e.target.checked)} />
          Outcome regression correctly specified
        </label>
      </div>
      <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '0.4rem', flexWrap: 'wrap' }}>
        <span>AIPW mean τ̂ = <span style={{ fontFamily: 'var(--font-mono)' }}>{mean(aipwTaus).toFixed(4)}</span></span>
        <span>TMLE mean τ̂ = <span style={{ fontFamily: 'var(--font-mono)' }}>{mean(tmleTaus).toFixed(4)}</span></span>
        <span>|AIPW aug mean| = <span style={{ fontFamily: 'var(--font-mono)' }}>{aipwAugMean.toExponential(2)}</span></span>
        <span>|TMLE aug mean| = <span style={{ fontFamily: 'var(--font-mono)' }}>{tmleAugMean.toExponential(2)}</span></span>
      </div>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
    </div>
  );
}

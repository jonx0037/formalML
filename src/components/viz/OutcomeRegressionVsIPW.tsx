import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  correctOutcomeFeatures,
  correctPropensityFeatures,
  ipwEstimate,
  linearFit,
  linearPredict,
  logisticFitPredict,
  mulberry32,
  orEstimateFromPreds,
  robinsonDGP,
  selectRows,
  selectValues,
} from './shared/causal-inference-methods';

// =============================================================================
// OutcomeRegressionVsIPW — §5 head-to-head OR and IPW under misspec.
// Two independent toggles: outcome-correct? / propensity-correct?
// Side-by-side histograms of OR and IPW distributions across B reps at n=600.
// =============================================================================

const HEIGHT = 360;
const B_REP = 180;
const N = 600;
const SM_BREAKPOINT = 640;

export default function OutcomeRegressionVsIPW() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [outCorrect, setOutCorrect] = useState(true);
  const [propCorrect, setPropCorrect] = useState(true);
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { orTaus, ipwTaus } = useMemo(() => {
    const orTaus = new Float64Array(B_REP);
    const ipwTaus = new Float64Array(B_REP);
    for (let b = 0; b < B_REP; b++) {
      const rng = mulberry32(20260512 + b * 11);
      const s = robinsonDGP(N, 10, 1.0, rng);
      const { X, D, Y, n, p } = s;
      // Propensity.
      let ePred: Float64Array;
      if (propCorrect) {
        const propFeat = correctPropensityFeatures(X, n, p);
        ({ ePred } = logisticFitPredict(propFeat, D, n, 3));
      } else {
        // Misspec: drop X_1 (omit the linear-confounder column).
        const fp = p - 1;
        const Fm = new Float64Array(n * fp);
        for (let i = 0; i < n; i++) for (let j = 1; j < p; j++) Fm[i * fp + (j - 1)] = X[i * p + j];
        ({ ePred } = logisticFitPredict(Fm, D, n, fp, { ridge: 0.1 }));
      }
      ipwTaus[b] = ipwEstimate(D, Y, ePred, { form: 'hajek' }).tau;
      // Outcome.
      const trIdx: number[] = [], ctIdx: number[] = [];
      for (let i = 0; i < n; i++) { if (D[i] === 1) trIdx.push(i); else ctIdx.push(i); }
      let mu0Hat: Float64Array, mu1Hat: Float64Array;
      if (outCorrect) {
        const out = correctOutcomeFeatures(X, n, p);
        const X1 = selectRows(out, trIdx, 2);
        const Y1 = selectValues(Y, trIdx);
        const X0 = selectRows(out, ctIdx, 2);
        const Y0 = selectValues(Y, ctIdx);
        const b1 = linearFit(X1, Y1, trIdx.length, 2);
        const b0 = linearFit(X0, Y0, ctIdx.length, 2);
        mu1Hat = linearPredict(b1, out, n, 2);
        mu0Hat = linearPredict(b0, out, n, 2);
      } else {
        const X1 = selectRows(X, trIdx, p);
        const Y1 = selectValues(Y, trIdx);
        const X0 = selectRows(X, ctIdx, p);
        const Y0 = selectValues(Y, ctIdx);
        const b1 = linearFit(X1, Y1, trIdx.length, p);
        const b0 = linearFit(X0, Y0, ctIdx.length, p);
        mu1Hat = linearPredict(b1, X, n, p);
        mu0Hat = linearPredict(b0, X, n, p);
      }
      orTaus[b] = orEstimateFromPreds(mu0Hat, mu1Hat).tau;
    }
    return { orTaus, ipwTaus };
  }, [outCorrect, propCorrect]);

  const drawPanel = (which: 'or' | 'ipw', color: string, taus: Float64Array, label: string) =>
    useD3<SVGSVGElement>(
      (svg) => {
        const w = containerWidth || 720;
        const panelW = isMobile ? w : w / 2 - 8;
        const margin = { top: 20, right: 14, bottom: 36, left: 48 };
        const W = panelW - margin.left - margin.right;
        const H = HEIGHT - margin.top - margin.bottom;
        svg.selectAll('*').remove();
        if (W <= 0) return;
        svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
        const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
        const tauArr = Array.from(taus);
        const lo = Math.min(0.5, d3.min(tauArr)!);
        const hi = Math.max(1.7, d3.max(tauArr)!);
        const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
        const bins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(28)(tauArr);
        const yScale = d3.scaleLinear().domain([0, d3.max(bins, (d) => d.length)! * 1.1 + 1]).range([H, 0]).nice();
        g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
        g.append('g').call(d3.axisLeft(yScale).ticks(4));
        g.selectAll('rect.bar').data(bins).join('rect').attr('class', 'bar')
          .attr('x', (d) => xScale(d.x0!) + 1).attr('y', (d) => yScale(d.length))
          .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
          .attr('height', (d) => H - yScale(d.length))
          .style('fill', color).style('opacity', 0.7);
        g.append('line').attr('x1', xScale(1)).attr('x2', xScale(1))
          .attr('y1', 0).attr('y2', H).style('stroke', '#3a6e3a').style('stroke-width', 2);
        const m = d3.mean(tauArr)!;
        g.append('line').attr('x1', xScale(m)).attr('x2', xScale(m))
          .attr('y1', 0).attr('y2', H).style('stroke', color).style('stroke-width', 1.5)
          .style('stroke-dasharray', '4 4');
        g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)').style('font-size', '11px')
          .text(`${label} τ̂ (mean = ${m.toFixed(3)})`);
      },
      [containerWidth, taus, color, label, isMobile],
    );

  const orRef = drawPanel('or', '#7b3c10', orTaus, 'OR');
  const ipwRef = drawPanel('ipw', '#1f4e79', ipwTaus, 'IPW');

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          <input type="checkbox" checked={outCorrect} onChange={(e) => setOutCorrect(e.target.checked)} />
          Outcome regression correctly specified
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          <input type="checkbox" checked={propCorrect} onChange={(e) => setPropCorrect(e.target.checked)} />
          Propensity correctly specified
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
        <svg ref={orRef} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT }} />
        <svg ref={ipwRef} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT }} />
      </div>
    </div>
  );
}

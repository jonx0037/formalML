import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  aipwEstimate,
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
// DoublyRobustToggle — §6 SIGNATURE viz: 2×2 grid of AIPW/IPW/OR distributions
// under the four (propensity correctness × outcome correctness) cells.
// AIPW recovers τ in 3 of 4 cells; IPW only in 2; OR only in 2.
// =============================================================================

const HEIGHT = 480;
const B_REP = 120;
const N = 500;
const SM_BREAKPOINT = 720;

type Cell = { propCorrect: boolean; outCorrect: boolean; or: Float64Array; ipw: Float64Array; aipw: Float64Array };

function runCell(propCorrect: boolean, outCorrect: boolean): Cell {
  const or = new Float64Array(B_REP);
  const ipw = new Float64Array(B_REP);
  const aipw = new Float64Array(B_REP);
  for (let b = 0; b < B_REP; b++) {
    const rng = mulberry32(20260512 + b * 17 + (propCorrect ? 1 : 0) * 100003 + (outCorrect ? 1 : 0) * 200003);
    const { X, D, Y, n, p } = robinsonDGP(N, 10, 1.0, rng);
    let ePred: Float64Array;
    if (propCorrect) {
      const F = correctPropensityFeatures(X, n, p);
      ({ ePred } = logisticFitPredict(F, D, n, 3));
    } else {
      const fp = p - 1;
      const Fm = new Float64Array(n * fp);
      for (let i = 0; i < n; i++) for (let j = 1; j < p; j++) Fm[i * fp + (j - 1)] = X[i * p + j];
      ({ ePred } = logisticFitPredict(Fm, D, n, fp, { ridge: 0.1 }));
    }
    const trIdx: number[] = [], ctIdx: number[] = [];
    for (let i = 0; i < n; i++) { if (D[i] === 1) trIdx.push(i); else ctIdx.push(i); }
    let mu0: Float64Array, mu1: Float64Array;
    if (outCorrect) {
      const out = correctOutcomeFeatures(X, n, p);
      const X1 = selectRows(out, trIdx, 2), Y1 = selectValues(Y, trIdx);
      const X0 = selectRows(out, ctIdx, 2), Y0 = selectValues(Y, ctIdx);
      const b1 = linearFit(X1, Y1, trIdx.length, 2);
      const b0 = linearFit(X0, Y0, ctIdx.length, 2);
      mu1 = linearPredict(b1, out, n, 2);
      mu0 = linearPredict(b0, out, n, 2);
    } else {
      const X1 = selectRows(X, trIdx, p), Y1 = selectValues(Y, trIdx);
      const X0 = selectRows(X, ctIdx, p), Y0 = selectValues(Y, ctIdx);
      const b1 = linearFit(X1, Y1, trIdx.length, p);
      const b0 = linearFit(X0, Y0, ctIdx.length, p);
      mu1 = linearPredict(b1, X, n, p);
      mu0 = linearPredict(b0, X, n, p);
    }
    or[b] = orEstimateFromPreds(mu0, mu1).tau;
    ipw[b] = ipwEstimate(D, Y, ePred, { form: 'hajek' }).tau;
    aipw[b] = aipwEstimate(D, Y, ePred, mu0, mu1).tau;
  }
  return { propCorrect, outCorrect, or, ipw, aipw };
}

export default function DoublyRobustToggle() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const cells = useMemo<Cell[]>(() => {
    return [
      runCell(true, true),
      runCell(true, false),
      runCell(false, true),
      runCell(false, false),
    ];
  }, []);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 760;
      const cols = isMobile ? 1 : 2;
      const rows = isMobile ? 4 : 2;
      const cellW = w / cols;
      const cellH = HEIGHT / rows;
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const totalH = cellH * rows;
      svg.attr('viewBox', `0 0 ${w} ${totalH}`);
      cells.forEach((c, idx) => {
        const r = isMobile ? idx : Math.floor(idx / 2);
        const co = isMobile ? 0 : idx % 2;
        const g = svg.append('g').attr('transform', `translate(${co * cellW}, ${r * cellH})`);
        const margin = { top: 20, right: 12, bottom: 30, left: 40 };
        const W = cellW - margin.left - margin.right;
        const H = cellH - margin.top - margin.bottom;
        const inner = g.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
        const allTaus = [...c.or, ...c.ipw, ...c.aipw];
        const lo = Math.min(0.5, d3.min(allTaus)!);
        const hi = Math.max(1.7, d3.max(allTaus)!);
        const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
        const xAxis = d3.axisBottom(xScale).ticks(4);
        inner.append('g').attr('transform', `translate(0, ${H})`).call(xAxis);
        // Kernel-density-ish bins for each estimator.
        const draw = (arr: Float64Array, color: string, offset: number) => {
          const bins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(20)(Array.from(arr));
          const yScale = d3.scaleLinear().domain([0, d3.max(bins, (d) => d.length)! * 1.1 + 1]).range([H / 3, 0]).nice();
          const baseY = (offset * H) / 3;
          inner.selectAll<SVGRectElement, d3.Bin<number, number>>(`rect.${color.replace('#', 'c')}`)
            .data(bins).enter().append('rect')
            .attr('x', (d) => xScale(d.x0!) + 1)
            .attr('y', (d) => baseY + yScale(d.length))
            .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
            .attr('height', (d) => H / 3 - yScale(d.length))
            .style('fill', color).style('opacity', 0.6);
          const m = d3.mean(arr)!;
          inner.append('line')
            .attr('x1', xScale(m)).attr('x2', xScale(m))
            .attr('y1', baseY).attr('y2', baseY + H / 3)
            .style('stroke', color).style('stroke-width', 1.5).style('stroke-dasharray', '3 3');
          inner.append('text').attr('x', W - 4).attr('y', baseY + 10).attr('text-anchor', 'end')
            .style('fill', color).style('font-size', '10px')
            .text(`${color === '#7b3c10' ? 'OR' : color === '#1f4e79' ? 'IPW' : 'AIPW'}: ${m.toFixed(3)}`);
        };
        draw(c.or, '#7b3c10', 0);
        draw(c.ipw, '#1f4e79', 1);
        draw(c.aipw, '#3a6e3a', 2);
        // True τ.
        inner.append('line')
          .attr('x1', xScale(1)).attr('x2', xScale(1))
          .attr('y1', 0).attr('y2', H)
          .style('stroke', '#c0504d').style('stroke-width', 1.5);
        // Title.
        const title = `propensity: ${c.propCorrect ? 'correct' : 'misspec'}, outcome: ${c.outCorrect ? 'correct' : 'misspec'}`;
        g.append('text').attr('x', cellW / 2).attr('y', 14).attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)').style('font-size', '11px').style('font-weight', '600')
          .text(title);
      });
    },
    [containerWidth, cells, isMobile],
  );

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
        Four nuisance-specification cells. AIPW (green) recovers τ in 3 of 4; IPW (blue) and OR (brown) each recover in 2.
      </div>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
    </div>
  );
}

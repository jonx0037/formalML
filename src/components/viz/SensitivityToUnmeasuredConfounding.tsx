import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  aipwEstimate,
  cinelliHazlettRV,
  correctOutcomeFeatures,
  correctPropensityFeatures,
  eValueContinuous,
  linearFit,
  linearPredict,
  logisticFitPredict,
  mulberry32,
  robinsonDGP,
  selectRows,
  selectValues,
  stddev,
} from './shared/causal-inference-methods';

// =============================================================================
// SensitivityToUnmeasuredConfounding — §12.
//
// Three sub-panels:
//  (A) Sensitivity curve: bias of AIPW as a function of hypothesized
//      unmeasured-confounder strength U_strength ∈ [0, 1.5]; tipping point
//      marker where the implied 95% CI lower bound crosses zero.
//  (B) E-value bar chart with VanderWeele–Ding benchmarks (1.5 / 3 / 5).
//  (C) Cinelli–Hazlett contour over (R²_D~Z|X, R²_Y~Z|D,X): the curve where the
//      bias bound SE·√df·√(R²_Y·R²_D/(1 − R²_D)) equals |τ̂| (region above it
//      shaded, since there the bias can reach |τ̂|). It crosses the diagonal at
//      RV_q; the dashed curve is the α-version, crossing it at RV_{q,α}.
//
// All three diagnostics live in one panel per topic-specific note 5.
// =============================================================================

const HEIGHT = 360;
const SM_BREAKPOINT = 920;
const STRENGTHS = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5];
const B_REP = 30;
const N = 600;

export default function SensitivityToUnmeasuredConfounding() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [alphaSig, setAlphaSig] = useState(0.05);
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { biases, ses, ePoint, eCi, rvQ, rvQAlpha, tippingIdx } = useMemo(() => {
    const biases = new Float64Array(STRENGTHS.length);
    const ses = new Float64Array(STRENGTHS.length);
    let baseline: { tauHat: number; se: number; sdY: number } | null = null;
    for (let k = 0; k < STRENGTHS.length; k++) {
      const us = STRENGTHS[k];
      const taus: number[] = [];
      for (let b = 0; b < B_REP; b++) {
        const rng = mulberry32(20260512 + b * 53 + Math.round(us * 1000));
        const s = robinsonDGP(N, 10, 1.0, rng, { unobservedConfounderStrength: us });
        const { X, D, Y, n, p } = s;
        const propF = correctPropensityFeatures(X, n, p);
        const { ePred } = logisticFitPredict(propF, D, n, 3);
        const outF = correctOutcomeFeatures(X, n, p);
        const trIdx: number[] = [], ctIdx: number[] = [];
        for (let i = 0; i < n; i++) { if (D[i] === 1) trIdx.push(i); else ctIdx.push(i); }
        const X1 = selectRows(outF, trIdx, 2), Y1 = selectValues(Y, trIdx);
        const X0 = selectRows(outF, ctIdx, 2), Y0 = selectValues(Y, ctIdx);
        const b1 = linearFit(X1, Y1, trIdx.length, 2);
        const b0 = linearFit(X0, Y0, ctIdx.length, 2);
        const mu1 = linearPredict(b1, outF, n, 2);
        const mu0 = linearPredict(b0, outF, n, 2);
        const res = aipwEstimate(D, Y, ePred, mu0, mu1);
        taus.push(res.tau);
        if (k === 0 && b === 0) {
          baseline = { tauHat: res.tau, se: res.se, sdY: stddev(Y) };
        }
      }
      const m = taus.reduce((a, c) => a + c, 0) / taus.length;
      let v = 0;
      for (const t of taus) v += (t - m) ** 2;
      biases[k] = m - 1;
      // Across-replicate SD of τ̂ (≈ the sampling SE of one estimate), not the
      // MC SE of the mean (which would be SD/√B_REP).
      ses[k] = Math.sqrt(v / Math.max(taus.length - 1, 1));
    }
    if (!baseline) baseline = { tauHat: 1, se: 0.05, sdY: 1.6 };
    // Tipping point: smallest U-strength where the lower edge of the band
    // (mean − 1.96·across-replicate SD, an implied single-sample 95% interval)
    // crosses zero.
    const tip = biases.findIndex((b, i) => (1 + b) - 1.96 * ses[i] <= 0);
    const tippingIdx = tip >= 0 ? tip : -1;
    const ev = eValueContinuous(baseline.tauHat, baseline.se, baseline.sdY);
    const rv = cinelliHazlettRV(baseline.tauHat, baseline.se, N - 5, { q: 1, alpha: alphaSig });
    return {
      biases, ses,
      ePoint: ev.ePoint, eCi: ev.eCi,
      rvQ: rv.rvQ, rvQAlpha: rv.rvQAlpha,
      tippingIdx,
    };
  }, [alphaSig]);

  const curveRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const cellW = isMobile ? w : w / 3 - 8;
      const margin = { top: 20, right: 14, bottom: 36, left: 48 };
      const W = cellW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${cellW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLinear().domain([0, 1.5]).range([0, W]);
      const yMin = Math.min(0, ...Array.from(biases).map((b) => 1 + b - 0.2));
      const yMax = Math.max(2.5, ...Array.from(biases).map((b) => 1 + b + 0.2));
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      // True τ = 1.
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', yScale(1)).attr('y2', yScale(1))
        .style('stroke', '#3a6e3a').style('stroke-width', 1.5);
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', yScale(0)).attr('y2', yScale(0))
        .style('stroke', '#000').style('stroke-width', 0.8).style('stroke-dasharray', '4 4');
      // Mean estimate + implied single-sample 95% band (mean ± 1.96·SD across replicates).
      const bandArea = d3.area<number>()
        .x((_, i) => xScale(STRENGTHS[i]))
        .y0((_, i) => yScale(1 + biases[i] - 1.96 * ses[i]))
        .y1((_, i) => yScale(1 + biases[i] + 1.96 * ses[i]));
      g.append('path').datum(Array.from(biases)).attr('d', bandArea)
        .style('fill', '#1f4e79').style('opacity', 0.15);
      const line = d3.line<number>().x((_, i) => xScale(STRENGTHS[i])).y((b) => yScale(1 + b));
      g.append('path').datum(Array.from(biases)).attr('d', line)
        .style('stroke', '#1f4e79').style('stroke-width', 2).style('fill', 'none');
      for (let i = 0; i < biases.length; i++) {
        g.append('circle').attr('cx', xScale(STRENGTHS[i])).attr('cy', yScale(1 + biases[i])).attr('r', 4)
          .style('fill', '#1f4e79');
      }
      if (tippingIdx >= 0) {
        g.append('line').attr('x1', xScale(STRENGTHS[tippingIdx])).attr('x2', xScale(STRENGTHS[tippingIdx]))
          .attr('y1', 0).attr('y2', H).style('stroke', '#c0504d').style('stroke-width', 1.5).style('stroke-dasharray', '3 3');
      }
      g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('Unmeasured-confounder strength');
      g.append('text').attr('x', 4).attr('y', 12).style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text('(A) AIPW estimate vs U strength (mean ± 1.96·SD across replicates)');
    },
    [containerWidth, biases, ses, tippingIdx, isMobile],
  );

  const eValueRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const cellW = isMobile ? w : w / 3 - 8;
      const margin = { top: 20, right: 14, bottom: 36, left: 60 };
      const W = cellW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${cellW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const benchmarks = [
        { label: 'Modest (1.5)', val: 1.5 },
        { label: 'Moderate (3)', val: 3 },
        { label: 'Strong (5)', val: 5 },
        { label: 'E-value (point)', val: ePoint },
        { label: 'E-value (CI lower)', val: eCi },
      ];
      const maxV = Math.max(6, ePoint * 1.1);
      const xScale = d3.scaleLinear().domain([0, maxV]).range([0, W]);
      const yScale = d3.scaleBand().domain(benchmarks.map((b) => b.label)).range([0, H]).padding(0.2);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale));
      benchmarks.forEach((b) => {
        g.append('rect').attr('x', 0).attr('y', yScale(b.label)!).attr('width', xScale(b.val))
          .attr('height', yScale.bandwidth())
          .style('fill', b.label.includes('E-value') ? '#1f4e79' : '#7f7f7f').style('opacity', 0.7);
        g.append('text').attr('x', xScale(b.val) + 4).attr('y', yScale(b.label)! + yScale.bandwidth() / 2 + 4)
          .style('fill', 'var(--color-text)').style('font-size', '10px').text(b.val.toFixed(2));
      });
      g.append('text').attr('x', 4).attr('y', 12).style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text('(B) E-value vs benchmarks');
    },
    [containerWidth, ePoint, eCi, isMobile],
  );

  const rvRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const cellW = isMobile ? w : w / 3 - 8;
      const margin = { top: 20, right: 14, bottom: 36, left: 48 };
      const W = cellW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${cellW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLinear().domain([0, 1]).range([0, W]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([H, 0]);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(4));
      g.append('g').call(d3.axisLeft(yScale).ticks(4));
      // Contour R²_Y·R²_D/(1 − R²_D) = f², i.e. R²_Y = f²(1 − x)/x, with
      // f² = RV²/(1 − RV) (the identity that defines RV). It passes through (RV, RV).
      const contour = (rv: number, steps = 80): [number, number][] => {
        const f2 = (rv * rv) / (1 - rv);
        const x0 = f2 / (1 + f2); // where the curve meets R²_Y = 1
        return d3.range(steps + 1).map((i) => {
          const x = x0 + ((1 - x0) * i) / steps;
          return [x, Math.min(1, (f2 * (1 - x)) / x)] as [number, number];
        });
      };
      const toPx = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1]));
      const main = contour(rvQ);
      // Shade the region above the contour (bias bound ≥ |τ̂|): curve, then the (1, 1) corner.
      g.append('path').datum([...main, [1, 1] as [number, number]]).attr('d', (d) => `${toPx(d)}Z`)
        .style('fill', '#c0504d').style('opacity', 0.15);
      g.append('path').datum(main).attr('d', toPx)
        .style('fill', 'none').style('stroke', '#c0504d').style('stroke-width', 1.5);
      if (rvQAlpha > 0) {
        g.append('path').datum(contour(rvQAlpha)).attr('d', toPx)
          .style('fill', 'none').style('stroke', '#7b3c10').style('stroke-width', 1).style('stroke-dasharray', '2 4');
      }
      // Diagonal R²_Y = R²_D and the RV_q crossing point.
      g.append('line').attr('x1', xScale(0)).attr('x2', xScale(1)).attr('y1', yScale(0)).attr('y2', yScale(1))
        .style('stroke', 'var(--color-text-secondary)').style('stroke-width', 0.8).style('stroke-dasharray', '3 3');
      g.append('circle').attr('cx', xScale(rvQ)).attr('cy', yScale(rvQ)).attr('r', 3.5).style('fill', '#c0504d');
      // Labels sit in the shaded region, left of the diagonal (which ends at the top-right corner).
      g.append('text').attr('x', xScale(0.3)).attr('y', 40).attr('text-anchor', 'start')
        .style('fill', '#c0504d').style('font-size', '10px').text(`RV_q = ${rvQ.toFixed(3)}`);
      g.append('text').attr('x', xScale(0.3)).attr('y', 56).attr('text-anchor', 'start')
        .style('fill', '#7b3c10').style('font-size', '10px').text(`RV_{q, α} = ${rvQAlpha.toFixed(3)}`);
      g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('R²_{D ~ Z | X}');
      g.append('text').attr('transform', `translate(-32, ${H / 2}) rotate(-90)`).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('R²_{Y ~ Z | D, X}');
      g.append('text').attr('x', 4).attr('y', -6).style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text('(C) Cinelli–Hazlett contour: bias bound = |τ̂|');
    },
    [containerWidth, rvQ, rvQAlpha, isMobile],
  );

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ marginBottom: '0.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          α for RV_{`{q, α}`} =
          <input type="range" min={0.01} max={0.2} step={0.01} value={alphaSig}
            onChange={(e) => setAlphaSig(parseFloat(e.target.value))}
            aria-label="alpha" style={{ width: 140 }} />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{alphaSig.toFixed(2)}</span>
        </label>
        <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
          tipping point at U strength ≈ {tippingIdx >= 0 ? STRENGTHS[tippingIdx].toFixed(2) : '> 1.5'}
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
        <svg ref={curveRef} style={{ width: isMobile ? '100%' : '33%', height: HEIGHT }} />
        <svg ref={eValueRef} style={{ width: isMobile ? '100%' : '33%', height: HEIGHT }} />
        <svg ref={rvRef} style={{ width: isMobile ? '100%' : '33%', height: HEIGHT }} />
      </div>
    </div>
  );
}

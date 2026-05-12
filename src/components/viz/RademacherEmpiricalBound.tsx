import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  polynomialUnitBallRademacher,
  sampleSinTarget,
} from './shared/structural-risk-minimization';

// =============================================================================
// RademacherEmpiricalBound — interactive companion to §5.5.
//
// Two sliders: n (25-500), B (Rademacher MC draws 50-1000). Plots empirical
// Rademacher complexity of the polynomial unit-ball class H_k° as a function
// of k ∈ [1, 15] alongside the VC-implied upper bound √(2(k+1)log(en/(k+1))/n).
// Optional toggle to overlay Vapnik penalty (Definition 3) for direct
// comparison with §4's bound. Commit-on-release for B; n updates live.
//
// Static fallback: public/images/topics/structural-risk-minimization/05_rademacher_srm.png
// =============================================================================

const HEIGHT = 380;
const K_MAX = 15;
const SIGMA = 0.2;
const SM_BREAKPOINT = 640;

export default function RademacherEmpiricalBound() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nDisplay, setNDisplay] = useState(100);
  const [n, setN] = useState(100);
  const [bDisplay, setBDisplay] = useState(300);
  const [B, setB] = useState(300);
  const [showVC, setShowVC] = useState(true);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const commitN = () => setN(nDisplay);
  const commitB = () => setB(bDisplay);

  const { radMean, radSE, vcBound, ks } = useMemo(() => {
    const rng = mulberry32(20260512 + n);
    const { X } = sampleSinTarget(n, SIGMA, rng);
    const radMean = new Float64Array(K_MAX + 1);
    const radSE = new Float64Array(K_MAX + 1);
    const vcBound = new Float64Array(K_MAX + 1);
    for (let k = 0; k <= K_MAX; k++) {
      const est = polynomialUnitBallRademacher(X, k, B, rng);
      radMean[k] = est.mean;
      radSE[k] = est.se;
      const dk = k + 1;
      vcBound[k] = Math.sqrt((2 * dk * Math.log((Math.E * n) / Math.max(dk, 1))) / n);
    }
    return { radMean, radSE, vcBound, ks: Array.from({ length: K_MAX + 1 }, (_, k) => k) };
  }, [n, B]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 16, right: 24, bottom: 36, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0, K_MAX]).range([0, W]);
      const yMax = Math.max(d3.max(showVC ? vcBound : radMean) || 1, 0.1);
      const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(8));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('polynomial degree k');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -40).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('complexity');
      // Empirical Rademacher with error band
      const upper = ks.map((k) => Math.max(radMean[k] + 2 * radSE[k], 0));
      const lower = ks.map((k) => Math.max(radMean[k] - 2 * radSE[k], 0));
      const areaGen = d3.area<number>().x((k) => x(k)).y0((k) => y(lower[k])).y1((k) => y(upper[k]));
      g.append('path').datum(ks).attr('d', areaGen).attr('fill', '#10b981').attr('opacity', 0.2);
      const lineGen = (vals: Float64Array) => d3.line<number>().x((k) => x(k)).y((k) => y(vals[k]));
      g.append('path').datum(ks).attr('d', lineGen(radMean)).attr('fill', 'none').attr('stroke', '#10b981').attr('stroke-width', 2.4);
      g.selectAll(null).data(ks).enter().append('circle').attr('cx', (k) => x(k)).attr('cy', (k) => y(radMean[k])).attr('r', 3).attr('fill', '#10b981');
      if (showVC) {
        g.append('path').datum(ks).attr('d', lineGen(vcBound)).attr('fill', 'none').attr('stroke', '#ef4444').attr('stroke-width', 2).attr('stroke-dasharray', '5 3');
      }
      const legend = g.append('g').attr('transform', `translate(${W - 220}, 6)`);
      legend.append('line').attr('x1', 0).attr('y1', 4).attr('x2', 14).attr('y2', 4).attr('stroke', '#10b981').attr('stroke-width', 2.4);
      legend.append('text').attr('x', 20).attr('y', 7).style('fill', 'var(--color-text)').style('font-size', '11px').text(`empirical R̂(H_k°), ±2 SE (B=${B})`);
      if (showVC) {
        legend.append('line').attr('x1', 0).attr('y1', 20).attr('x2', 14).attr('y2', 20).attr('stroke', '#ef4444').attr('stroke-width', 2).attr('stroke-dasharray', '5 3');
        legend.append('text').attr('x', 20).attr('y', 23).style('fill', 'var(--color-text)').style('font-size', '11px').text('VC upper √(2(k+1)log(en/(k+1))/n)');
      }
    },
    [radMean, radSE, vcBound, ks, showVC, B, containerWidth],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="grid gap-3 mb-3 text-xs" style={{ gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)' }}>
        <label className="flex flex-col gap-1">
          <span><strong>n</strong>: <span className="tabular-nums">{nDisplay}</span></span>
          <input type="range" min={25} max={500} step={5} value={nDisplay} onChange={(e) => setNDisplay(Number(e.target.value))} onMouseUp={commitN} onTouchEnd={commitN} onKeyUp={commitN} aria-label="sample size n" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>B</strong> (Rademacher MC draws): <span className="tabular-nums">{bDisplay}</span></span>
          <input type="range" min={50} max={1000} step={25} value={bDisplay} onChange={(e) => setBDisplay(Number(e.target.value))} onMouseUp={commitB} onTouchEnd={commitB} onKeyUp={commitB} aria-label="Rademacher draws B" />
        </label>
      </div>
      <label className="flex items-center gap-2 text-xs mb-2">
        <input type="checkbox" checked={showVC} onChange={(e) => setShowVC(e.target.checked)} />
        <span>overlay VC upper bound (Massart's lemma + Sauer-Shelah)</span>
      </label>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        Empirical Rademacher sits below the VC upper bound; the gap is the √log(n/d_k) data-dependent savings. The savings shrink at small n because the McDiarmid confidence term in Definition 4 dominates (Rademacher SRM picks $\hat k_R = 1$ at n = 50, 100; cross-over near n ≥ 200).
      </div>
    </div>
  );
}

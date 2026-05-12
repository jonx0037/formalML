import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  polyfitDegree,
  polyvalIncreasing,
  sampleSinTarget,
  targetSin,
  trainingMse,
} from './shared/structural-risk-minimization';

// =============================================================================
// NestedFamilyExplorer — interactive companion to §2.5.
//
// One slider (k ∈ [0, 15]) plus a "re-roll sample" button. Fits a polynomial
// of degree k to a fixed n = 50 sample from the §1 toy and overlays the fit
// on a single static training-data panel with the true sin(πx) function. A
// small inline read-out prints training MSE so the reader can confirm that
// training error is monotone non-increasing in k.
//
// Static fallback: public/images/topics/structural-risk-minimization/02_polynomial_ladder.png
// =============================================================================

const N = 50;
const SIGMA = 0.2;
const HEIGHT = 380;
const K_MAX = 15;

export default function NestedFamilyExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [k, setK] = useState(5);
  const [seed, setSeed] = useState(20260512);

  const { X, Y } = useMemo(() => sampleSinTarget(N, SIGMA, mulberry32(seed)), [seed]);
  const coefs = useMemo(() => polyfitDegree(X, Y, k), [X, Y, k]);
  const mse = useMemo(() => trainingMse(X, Y, k), [X, Y, k]);

  const xGrid = useMemo(() => {
    const arr = new Float64Array(301);
    for (let i = 0; i < arr.length; i++) arr[i] = -1 + (2 * i) / 300;
    return arr;
  }, []);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 16, right: 24, bottom: 36, left: 50 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([-1, 1]).range([0, W]);
      const y = d3.scaleLinear().domain([-2, 2]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('x');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -34).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('y');
      // True function
      const truthLine = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(targetSin(xGrid[i])));
      g.append('path').datum(d3.range(xGrid.length)).attr('d', truthLine).attr('fill', 'none').attr('stroke', '#10b981').attr('stroke-width', 2).attr('stroke-dasharray', '5 3');
      // Fit
      const fitLine = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(polyvalIncreasing(coefs, xGrid[i])));
      g.append('path').datum(d3.range(xGrid.length)).attr('d', fitLine).attr('fill', 'none').attr('stroke', '#ef4444').attr('stroke-width', 2.4);
      // Training data
      g.selectAll(null).data(d3.range(N)).enter().append('circle').attr('cx', (i) => x(X[i])).attr('cy', (i) => y(Y[i])).attr('r', 3).attr('fill', '#3b82f6').attr('opacity', 0.7);
      // Legend
      const legend = g.append('g').attr('transform', `translate(${W - 120}, 6)`);
      legend.append('rect').attr('x', 0).attr('y', 0).attr('width', 12).attr('height', 3).attr('fill', '#10b981');
      legend.append('text').attr('x', 18).attr('y', 4).style('fill', 'var(--color-text)').style('font-size', '11px').text('m(x) = sin(πx)');
      legend.append('rect').attr('x', 0).attr('y', 16).attr('width', 12).attr('height', 3).attr('fill', '#ef4444');
      legend.append('text').attr('x', 18).attr('y', 20).style('fill', 'var(--color-text)').style('font-size', '11px').text(`degree-${k} fit`);
      legend.append('circle').attr('cx', 6).attr('cy', 36).attr('r', 3).attr('fill', '#3b82f6').attr('opacity', 0.7);
      legend.append('text').attr('x', 18).attr('y', 40).style('fill', 'var(--color-text)').style('font-size', '11px').text(`training (n = ${N})`);
    },
    [X, Y, coefs, xGrid, containerWidth],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="flex flex-wrap gap-4 items-end mb-3 text-xs">
        <label className="flex-1 min-w-[240px] flex flex-col gap-1">
          <span><strong>k</strong> (polynomial degree): <span className="tabular-nums">{k}</span></span>
          <input type="range" min={0} max={K_MAX} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label="polynomial degree" />
        </label>
        <button onClick={() => setSeed((s) => s + 1)} className="px-3 py-1 border rounded hover:opacity-80" style={{ borderColor: 'var(--color-border)', color: 'var(--color-text)' }}>
          re-roll sample
        </button>
      </div>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        Training MSE at k = {k}: <strong className="tabular-nums">{mse.toFixed(4)}</strong>. Training error is monotone non-increasing in k — ERM alone tells the wrong story about which k to pick.
      </div>
    </div>
  );
}

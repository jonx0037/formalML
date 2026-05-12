import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  sampleSinTarget,
  srmPickFromArrays,
  trainingMseByDegree,
  vapnikPenaltyArray,
} from './shared/structural-risk-minimization';

// =============================================================================
// VCSRMCurve — interactive companion to §4.5 (and absorbs §3.5).
//
// Three sliders: n (25-500), δ (0.01-0.5), C (Vapnik universal constant 0.1-2.0).
// Plots training MSE, Vapnik penalty (Definition 3), and their sum as functions
// of k ∈ [1, 15] on the polynomial-regression toy. Marks $\hat k_V$ at argmin.
// An optional toggle exposes the three-term decomposition: d_k log(2n/d_k),
// 2 log k, and log(π²/(6δ)) on the same axes.
//
// Static fallback: public/images/topics/structural-risk-minimization/04_vapnik_srm_curve.png
// =============================================================================

const HEIGHT = 380;
const K_MAX = 15;
const SIGMA = 0.2;
const SM_BREAKPOINT = 640;

export default function VCSRMCurve() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nDisplay, setNDisplay] = useState(50);
  const [n, setN] = useState(50);
  const [delta, setDelta] = useState(0.05);
  const [C, setC] = useState(1.0);
  const [showDecomp, setShowDecomp] = useState(false);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const commitN = () => setN(nDisplay);

  const { trainMse, vapnik, total, kHat, ks } = useMemo(() => {
    const rng = mulberry32(20260512 + n);
    const { X, Y } = sampleSinTarget(n, SIGMA, rng);
    const train = trainingMseByDegree(X, Y, K_MAX);
    const pen = vapnikPenaltyArray(K_MAX, n, delta, C);
    const tot = new Float64Array(K_MAX + 1);
    for (let k = 0; k <= K_MAX; k++) tot[k] = train[k] + pen[k];
    return {
      trainMse: train,
      vapnik: pen,
      total: tot,
      kHat: srmPickFromArrays(train, pen),
      ks: Array.from({ length: K_MAX + 1 }, (_, k) => k),
    };
  }, [n, delta, C]);

  const decomp = useMemo(() => {
    if (!showDecomp) return null;
    const cap = new Float64Array(K_MAX + 1);
    const klog = new Float64Array(K_MAX + 1);
    const conf = Math.log((Math.PI * Math.PI) / (6 * delta));
    const confArr = new Float64Array(K_MAX + 1);
    for (let k = 0; k <= K_MAX; k++) {
      const d = k + 1;
      cap[k] = d * Math.log((2 * n) / Math.max(d, 1));
      klog[k] = k <= 1 ? 0 : 2 * Math.log(k);
      confArr[k] = conf;
    }
    return { cap, klog, conf: confArr };
  }, [n, delta, showDecomp]);

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
      const yMax = Math.max(d3.max(total) || 1, 0.05);
      const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(8));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('polynomial degree k');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -40).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('training MSE + penalty');
      const lineGen = (vals: Float64Array) => d3.line<number>().x((k) => x(k)).y((k) => y(vals[k]));
      const series = [
        { vals: trainMse, color: '#3b82f6', name: 'training MSE' },
        { vals: vapnik, color: '#ef4444', name: 'Vapnik penalty', dash: '4 3' },
        { vals: total, color: '#10b981', name: 'total = MSE + pen', width: 2.6 },
      ];
      series.forEach((s) => {
        g.append('path').datum(ks).attr('d', lineGen(s.vals)).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', s.width || 2).attr('stroke-dasharray', s.dash || '');
      });
      g.append('line').attr('x1', x(kHat)).attr('x2', x(kHat)).attr('y1', 0).attr('y2', H).attr('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3').attr('opacity', 0.6);
      g.append('text').attr('x', x(kHat)).attr('y', -4).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text(`k̂_V = ${kHat}`);
      if (decomp) {
        // Overlay normalised decomposition curves (scale to the penalty range)
        const decompSeries = [
          { vals: decomp.cap, color: '#a855f7', name: 'd log(2n/d)' },
          { vals: decomp.klog, color: '#f59e0b', name: '2 log k' },
          { vals: decomp.conf, color: '#06b6d4', name: 'log(π²/(6δ))' },
        ];
        const maxDecomp = Math.max(d3.max(decomp.cap) || 1, d3.max(decomp.klog) || 1, d3.max(decomp.conf) || 1);
        const yScale = yMax / maxDecomp * 0.5;
        decompSeries.forEach((s) => {
          const line = d3.line<number>().x((k) => x(k)).y((k) => y(s.vals[k] * yScale));
          g.append('path').datum(ks).attr('d', line).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 1.5).attr('opacity', 0.6);
        });
      }
      const legend = g.append('g').attr('transform', `translate(${W - 160}, 6)`);
      series.forEach((s, i) => {
        legend.append('line').attr('x1', 0).attr('y1', i * 14 + 4).attr('x2', 14).attr('y2', i * 14 + 4).attr('stroke', s.color).attr('stroke-width', s.width || 2).attr('stroke-dasharray', s.dash || '');
        legend.append('text').attr('x', 20).attr('y', i * 14 + 7).style('fill', 'var(--color-text)').style('font-size', '11px').text(s.name);
      });
    },
    [trainMse, vapnik, total, kHat, ks, decomp, containerWidth],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="grid gap-3 mb-3 text-xs" style={{ gridTemplateColumns: isMobile ? '1fr' : 'repeat(3, 1fr)' }}>
        <label className="flex flex-col gap-1">
          <span><strong>n</strong>: <span className="tabular-nums">{nDisplay}</span></span>
          <input type="range" min={25} max={500} step={5} value={nDisplay} onChange={(e) => setNDisplay(Number(e.target.value))} onMouseUp={commitN} onTouchEnd={commitN} onKeyUp={commitN} aria-label="sample size n" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>δ</strong>: <span className="tabular-nums">{delta.toFixed(2)}</span></span>
          <input type="range" min={0.01} max={0.5} step={0.01} value={delta} onChange={(e) => setDelta(Number(e.target.value))} aria-label="confidence parameter delta" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>C</strong> (Vapnik constant): <span className="tabular-nums">{C.toFixed(2)}</span></span>
          <input type="range" min={0.1} max={2.0} step={0.05} value={C} onChange={(e) => setC(Number(e.target.value))} aria-label="Vapnik universal constant C" />
        </label>
      </div>
      <label className="flex items-center gap-2 text-xs mb-2">
        <input type="checkbox" checked={showDecomp} onChange={(e) => setShowDecomp(e.target.checked)} />
        <span>show §3.5 three-term decomposition (capacity + 2 log k + log(π²/(6δ)))</span>
      </label>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        Vapnik SRM picks <strong>k̂_V = {kHat}</strong> at (n = {n}, δ = {delta.toFixed(2)}, C = {C.toFixed(2)}). On this benign distribution k̂_V is consistently smaller than the oracle k* ≈ 5 — the bound pays for distribution-freeness.
      </div>
    </div>
  );
}

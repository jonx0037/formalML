import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cvPickHistogram,
  kFoldCVCurve,
  mulberry32,
  sampleSinTarget,
} from './shared/structural-risk-minimization';

// =============================================================================
// CrossValidationVsSRM — interactive companion to §10.5.
//
// Three sliders: K (folds 2-20), B (fold-partition rerolls 1-200), n (25-500).
// Plots the mean K-fold CV curve across B rerolls with a ±1σ band, and shows
// the distribution of CV picks as a histogram on the side. Commit-on-release
// for B and n — heavy computation.
//
// Static fallback: public/images/topics/structural-risk-minimization/10_cross_validation_srm.png
// =============================================================================

const HEIGHT = 380;
const K_MAX = 15;
const SIGMA = 0.2;
const SM_BREAKPOINT = 640;

export default function CrossValidationVsSRM() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [K, setK] = useState(5);
  const [bDisplay, setBDisplay] = useState(50);
  const [B, setB] = useState(50);
  const [nDisplay, setNDisplay] = useState(50);
  const [n, setN] = useState(50);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const commitB = () => setB(bDisplay);
  const commitN = () => setN(nDisplay);

  const { meanCurve, stdCurve, picks, mode, ks } = useMemo(() => {
    const rng = mulberry32(20260512 + n);
    const { X, Y } = sampleSinTarget(n, SIGMA, rng);
    const curves: Float64Array[] = [];
    for (let b = 0; b < B; b++) {
      curves.push(kFoldCVCurve(X, Y, K_MAX, K, rng));
    }
    const meanArr = new Float64Array(K_MAX + 1);
    const stdArr = new Float64Array(K_MAX + 1);
    for (let k = 0; k <= K_MAX; k++) {
      let s = 0, sq = 0;
      for (let b = 0; b < B; b++) {
        s += curves[b][k];
        sq += curves[b][k] * curves[b][k];
      }
      meanArr[k] = s / B;
      stdArr[k] = Math.sqrt(Math.max(sq / B - meanArr[k] * meanArr[k], 0));
    }
    const { picks: pickArr, mode } = cvPickHistogram(X, Y, K_MAX, K, B, rng);
    return {
      meanCurve: meanArr,
      stdCurve: stdArr,
      picks: pickArr,
      mode,
      ks: Array.from({ length: K_MAX + 1 }, (_, k) => k),
    };
  }, [n, K, B]);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w * 0.65;
      const margin = { top: 16, right: 16, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0, K_MAX]).range([0, W]);
      const yMax = d3.max(ks.map((k) => meanCurve[k] + stdCurve[k])) || 1;
      const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(8));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('polynomial degree k');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('CV score');
      const upper = ks.map((k) => meanCurve[k] + stdCurve[k]);
      const lower = ks.map((k) => Math.max(meanCurve[k] - stdCurve[k], 0));
      const area = d3.area<number>().x((k) => x(k)).y0((k) => y(lower[k])).y1((k) => y(upper[k]));
      g.append('path').datum(ks).attr('d', area).attr('fill', '#06b6d4').attr('opacity', 0.2);
      const mean = d3.line<number>().x((k) => x(k)).y((k) => y(meanCurve[k]));
      g.append('path').datum(ks).attr('d', mean).attr('fill', 'none').attr('stroke', '#06b6d4').attr('stroke-width', 2.4);
      g.selectAll(null).data(ks).enter().append('circle').attr('cx', (k) => x(k)).attr('cy', (k) => y(meanCurve[k])).attr('r', 2.5).attr('fill', '#06b6d4');
      g.append('line').attr('x1', x(mode)).attr('x2', x(mode)).attr('y1', 0).attr('y2', H).attr('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3').attr('opacity', 0.6);
      g.append('text').attr('x', x(mode)).attr('y', -4).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text(`mode = ${mode}`);
    },
    [meanCurve, stdCurve, mode, ks, containerWidth, isMobile],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w * 0.35 - 16;
      const margin = { top: 16, right: 16, bottom: 36, left: 40 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const counts = new Int32Array(K_MAX + 1);
      for (let i = 0; i < picks.length; i++) counts[picks[i]]++;
      const x = d3.scaleBand().domain(ks.map(String)).range([0, W]).padding(0.15);
      const y = d3.scaleLinear().domain([0, Math.max(d3.max(Array.from(counts)) || 1, 1)]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).tickValues(ks.filter((k) => k % 2 === 0).map(String)));
      g.append('g').call(d3.axisLeft(y).ticks(4));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('CV pick distribution');
      g.selectAll(null).data(ks).enter().append('rect').attr('x', (k) => x(String(k))!).attr('y', (k) => y(counts[k])).attr('width', x.bandwidth()).attr('height', (k) => H - y(counts[k])).attr('fill', '#06b6d4').attr('opacity', 0.7);
    },
    [picks, ks, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="grid gap-3 mb-3 text-xs" style={{ gridTemplateColumns: isMobile ? '1fr' : 'repeat(3, 1fr)' }}>
        <label className="flex flex-col gap-1">
          <span><strong>K</strong> (folds): <span className="tabular-nums">{K}</span></span>
          <input type="range" min={2} max={20} step={1} value={K} onChange={(e) => setK(Number(e.target.value))} aria-label="number of folds K" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>B</strong> (rerolls): <span className="tabular-nums">{bDisplay}</span></span>
          <input type="range" min={1} max={200} step={1} value={bDisplay} onChange={(e) => setBDisplay(Number(e.target.value))} onMouseUp={commitB} onTouchEnd={commitB} onKeyUp={commitB} aria-label="fold partition rerolls B" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>n</strong>: <span className="tabular-nums">{nDisplay}</span></span>
          <input type="range" min={25} max={500} step={5} value={nDisplay} onChange={(e) => setNDisplay(Number(e.target.value))} onMouseUp={commitN} onTouchEnd={commitN} onKeyUp={commitN} aria-label="sample size n" />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '16px' }}>
        <svg ref={leftRef} style={{ width: '100%', height: HEIGHT }} />
        <svg ref={rightRef} style={{ width: '100%', height: HEIGHT }} />
      </div>
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        CV pick mode: <strong>{mode}</strong> across {B} fold rerolls at K={K}, n={n}. The histogram shows fold-partition variance directly — 1–3 degrees of spread is typical at small n.
      </div>
    </div>
  );
}

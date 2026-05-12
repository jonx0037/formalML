import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { biasVarianceMC, mulberry32 } from './shared/structural-risk-minimization';

// =============================================================================
// BiasVarianceCapacityCurve — interactive companion to §1.5.
//
// Three sliders: n (sample size 25-500), σ (noise std 0.05-0.5), B (replicates
// 50-500). Computes bias², variance, MSE on a test grid for each polynomial
// degree k ∈ [0, 15] via Monte Carlo. Left panel: bias²/variance/MSE curves at
// (n, σ, B). Right panel: MSE curves at four sample sizes overlaid to show the
// "more data → larger optimal k" story from §1.3. Commit-on-release for the
// heavy MC sliders.
//
// Static fallback: public/images/topics/structural-risk-minimization/01_bias_variance_ucurve.png
// =============================================================================

const HEIGHT = 380;
const K_MAX = 15;
const SM_BREAKPOINT = 640;

export default function BiasVarianceCapacityCurve() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nDisplay, setNDisplay] = useState(50);
  const [n, setN] = useState(50);
  const [sigmaDisplay, setSigmaDisplay] = useState(0.2);
  const [sigma, setSigma] = useState(0.2);
  const [bDisplay, setBDisplay] = useState(100);
  const [B, setB] = useState(100);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { biasSq, variance, mse, kStar } = useMemo(() => {
    const rng = mulberry32(20260512 + n + Math.round(sigma * 1000));
    return biasVarianceMC(n, sigma, K_MAX, B, rng, 80);
  }, [n, sigma, B]);

  // Overlay panel: MSE curves at four sample sizes
  const overlayCurves = useMemo(() => {
    const sampleSizes = [25, 50, 100, 500];
    return sampleSizes.map((ns) => {
      const rng = mulberry32(20260513 + ns + Math.round(sigma * 1000));
      return { n: ns, mse: biasVarianceMC(ns, sigma, K_MAX, Math.min(B, 100), rng, 80).mse };
    });
  }, [sigma, B]);

  const ks = useMemo(() => Array.from({ length: K_MAX + 1 }, (_, k) => k), []);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 16, right: 24, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0, K_MAX]).range([0, W]);
      const yMax = Math.max(d3.max(mse) || 1, 0.05);
      const y = d3.scaleLog().domain([1e-4, yMax * 1.5]).range([H, 0]).clamp(true);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(8));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('polynomial degree k');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -34).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('error (log scale)');
      const lineGen = (vals: Float64Array) => d3.line<number>().x((k) => x(k)).y((k) => y(Math.max(vals[k], 1e-6)));
      const series = [
        { vals: biasSq, color: '#3b82f6', name: 'bias²' },
        { vals: variance, color: '#ef4444', name: 'variance' },
        { vals: mse, color: '#10b981', name: 'MSE' },
      ];
      series.forEach((s) => {
        g.append('path').datum(ks).attr('d', lineGen(s.vals)).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 2.2);
        g.selectAll(null).data(ks).enter().append('circle').attr('cx', (k) => x(k)).attr('cy', (k) => y(Math.max(s.vals[k], 1e-6))).attr('r', 2.5).attr('fill', s.color);
      });
      // Mark kStar
      g.append('line').attr('x1', x(kStar)).attr('x2', x(kStar)).attr('y1', 0).attr('y2', H).attr('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3').attr('opacity', 0.6);
      g.append('text').attr('x', x(kStar)).attr('y', -4).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text(`k* = ${kStar}`);
      // Legend
      const legend = g.append('g').attr('transform', `translate(${W - 90}, 6)`);
      series.forEach((s, i) => {
        legend.append('rect').attr('x', 0).attr('y', i * 16).attr('width', 12).attr('height', 3).attr('fill', s.color);
        legend.append('text').attr('x', 18).attr('y', i * 16 + 5).style('fill', 'var(--color-text)').style('font-size', '11px').text(s.name);
      });
    },
    [biasSq, variance, mse, kStar, containerWidth, isMobile],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 16, right: 24, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0, K_MAX]).range([0, W]);
      const yMax = Math.max(d3.max(overlayCurves.flatMap((c) => Array.from(c.mse))) || 1, 0.05);
      const y = d3.scaleLog().domain([1e-4, yMax * 1.5]).range([H, 0]).clamp(true);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(8));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('polynomial degree k');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -34).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('MSE (log scale)');
      const colors = ['#fbbf24', '#f97316', '#dc2626', '#7c3aed'];
      overlayCurves.forEach((c, i) => {
        const path = d3.line<number>().x((k) => x(k)).y((k) => y(Math.max(c.mse[k], 1e-6)));
        g.append('path').datum(ks).attr('d', path).attr('fill', 'none').attr('stroke', colors[i]).attr('stroke-width', 2);
      });
      const legend = g.append('g').attr('transform', `translate(${W - 90}, 6)`);
      overlayCurves.forEach((c, i) => {
        legend.append('rect').attr('x', 0).attr('y', i * 16).attr('width', 12).attr('height', 3).attr('fill', colors[i]);
        legend.append('text').attr('x', 18).attr('y', i * 16 + 5).style('fill', 'var(--color-text)').style('font-size', '11px').text(`n = ${c.n}`);
      });
    },
    [overlayCurves, containerWidth, isMobile],
  );

  const sliderClass = 'flex flex-col gap-1 text-xs';
  const commitN = () => setN(nDisplay);
  const commitSigma = () => setSigma(sigmaDisplay);
  const commitB = () => setB(bDisplay);

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="grid gap-3 mb-3" style={{ gridTemplateColumns: isMobile ? '1fr' : 'repeat(3, 1fr)' }}>
        <label className={sliderClass}>
          <span><strong>n</strong> (sample size): <span className="tabular-nums">{nDisplay}</span></span>
          <input type="range" min={25} max={500} step={5} value={nDisplay} onChange={(e) => setNDisplay(Number(e.target.value))} onMouseUp={commitN} onTouchEnd={commitN} onKeyUp={commitN} aria-label="sample size n" />
        </label>
        <label className={sliderClass}>
          <span><strong>σ</strong> (noise std): <span className="tabular-nums">{sigmaDisplay.toFixed(2)}</span></span>
          <input type="range" min={0.05} max={0.5} step={0.01} value={sigmaDisplay} onChange={(e) => setSigmaDisplay(Number(e.target.value))} onMouseUp={commitSigma} onTouchEnd={commitSigma} onKeyUp={commitSigma} aria-label="noise standard deviation" />
        </label>
        <label className={sliderClass}>
          <span><strong>B</strong> (MC replicates): <span className="tabular-nums">{bDisplay}</span></span>
          <input type="range" min={50} max={300} step={10} value={bDisplay} onChange={(e) => setBDisplay(Number(e.target.value))} onMouseUp={commitB} onTouchEnd={commitB} onKeyUp={commitB} aria-label="Monte Carlo replicates B" />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '16px' }}>
        <svg ref={leftRef} style={{ width: '100%', height: HEIGHT }} />
        <svg ref={rightRef} style={{ width: '100%', height: HEIGHT }} />
      </div>
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        At (n = {n}, σ = {sigma.toFixed(2)}, B = {B}), the bias-variance optimum is <strong>k* = {kStar}</strong>. Right panel: shifting n shifts k* rightward, the central claim of §1.3.
      </div>
    </div>
  );
}

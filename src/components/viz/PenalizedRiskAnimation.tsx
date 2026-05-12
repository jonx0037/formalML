import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  logspace,
  mulberry32,
  polyvalIncreasing,
  ridgeFit,
  ridgeSoftSRMPath,
  sampleSinTarget,
  targetSin,
} from './shared/structural-risk-minimization';

// =============================================================================
// PenalizedRiskAnimation — interactive companion to §6.5.
//
// One log-slider for λ ∈ [10⁻⁸, 10⁶]. Two panels: left shows the ridge fit at
// the current λ overlaid on training data; right shows the soft-SRM path
// (training MSE, effective DoF, total) as a function of λ with a moving marker
// at the current value. The picked $\hat\lambda$ on the path is annotated.
//
// Static fallback: public/images/topics/structural-risk-minimization/06_soft_srm_path.png
// =============================================================================

const HEIGHT = 380;
const N = 80;
const SIGMA = 0.2;
const K_MAX = 15;
const DELTA = 0.05;
const SM_BREAKPOINT = 640;

export default function PenalizedRiskAnimation() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [logLambda, setLogLambda] = useState(-2);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const lambda = Math.pow(10, logLambda);

  const { X, Y } = useMemo(() => sampleSinTarget(N, SIGMA, mulberry32(20260512)), []);
  const lambdas = useMemo(() => logspace(-8, 6, 60), []);
  const path = useMemo(() => ridgeSoftSRMPath(X, Y, K_MAX, lambdas, DELTA), [X, Y, lambdas]);
  const currentFit = useMemo(() => ridgeFit(X, Y, K_MAX, lambda), [X, Y, lambda]);

  const xGrid = useMemo(() => {
    const arr = new Float64Array(201);
    for (let i = 0; i < arr.length; i++) arr[i] = -1 + (2 * i) / 200;
    return arr;
  }, []);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 16, right: 16, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([-1, 1]).range([0, W]);
      const y = d3.scaleLinear().domain([-2, 2]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('x');
      const truthLine = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(targetSin(xGrid[i])));
      g.append('path').datum(d3.range(xGrid.length)).attr('d', truthLine).attr('fill', 'none').attr('stroke', '#10b981').attr('stroke-width', 1.6).attr('stroke-dasharray', '4 3');
      const fitLine = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(polyvalIncreasing(currentFit.coefs, xGrid[i])));
      g.append('path').datum(d3.range(xGrid.length)).attr('d', fitLine).attr('fill', 'none').attr('stroke', '#ef4444').attr('stroke-width', 2.4);
      g.selectAll(null).data(d3.range(N)).enter().append('circle').attr('cx', (i) => x(X[i])).attr('cy', (i) => y(Y[i])).attr('r', 2.5).attr('fill', '#3b82f6').attr('opacity', 0.6);
      g.append('text').attr('x', 8).attr('y', 14).style('fill', 'var(--color-text)').style('font-size', '11px').text(`ridge fit, eff DoF = ${currentFit.effectiveDof.toFixed(2)}`);
    },
    [currentFit, xGrid, X, Y, containerWidth, isMobile],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 16, right: 16, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLog().domain([1e-8, 1e6]).range([0, W]);
      const yMax = Math.max(d3.max(path.total) || 1, 0.05);
      const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5, '~e'));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('λ (log scale)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -40).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('value');
      const mse = path.trainingMse;
      const total = path.total;
      const lineMse = d3.line<number>().x((i) => x(lambdas[i])).y((i) => y(mse[i]));
      const lineTotal = d3.line<number>().x((i) => x(lambdas[i])).y((i) => y(total[i]));
      g.append('path').datum(d3.range(lambdas.length)).attr('d', lineMse).attr('fill', 'none').attr('stroke', '#3b82f6').attr('stroke-width', 2);
      g.append('path').datum(d3.range(lambdas.length)).attr('d', lineTotal).attr('fill', 'none').attr('stroke', '#10b981').attr('stroke-width', 2.6);
      g.append('line').attr('x1', x(path.hatLambda)).attr('x2', x(path.hatLambda)).attr('y1', 0).attr('y2', H).attr('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3').attr('opacity', 0.6);
      g.append('text').attr('x', x(path.hatLambda)).attr('y', -4).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text(`λ̂ = ${path.hatLambda.toExponential(1)}`);
      // Moving marker at current λ
      g.append('circle').attr('cx', x(lambda)).attr('cy', y(currentFit.trainingMse)).attr('r', 5).attr('fill', '#ef4444').attr('stroke', 'var(--color-surface)').attr('stroke-width', 2);
      const legend = g.append('g').attr('transform', `translate(${W - 130}, 6)`);
      legend.append('line').attr('x1', 0).attr('y1', 4).attr('x2', 14).attr('y2', 4).attr('stroke', '#3b82f6').attr('stroke-width', 2);
      legend.append('text').attr('x', 18).attr('y', 7).style('fill', 'var(--color-text)').style('font-size', '11px').text('training MSE');
      legend.append('line').attr('x1', 0).attr('y1', 20).attr('x2', 14).attr('y2', 20).attr('stroke', '#10b981').attr('stroke-width', 2.6);
      legend.append('text').attr('x', 18).attr('y', 23).style('fill', 'var(--color-text)').style('font-size', '11px').text('soft-SRM total');
    },
    [path, lambdas, lambda, currentFit, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="mb-3 text-xs">
        <label className="flex flex-col gap-1">
          <span><strong>log₁₀ λ</strong>: <span className="tabular-nums">{logLambda.toFixed(2)}</span> (λ = {lambda.toExponential(2)})</span>
          <input type="range" min={-8} max={6} step={0.1} value={logLambda} onChange={(e) => setLogLambda(Number(e.target.value))} aria-label="log of ridge penalty lambda" />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '16px' }}>
        <svg ref={leftRef} style={{ width: '100%', height: HEIGHT }} />
        <svg ref={rightRef} style={{ width: '100%', height: HEIGHT }} />
      </div>
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        Soft-SRM ridge: λ acts as a continuous capacity dial. Small λ = high capacity (overfit visible on the left); large λ = low capacity (underfit). The soft-SRM total picks λ̂ at the interior minimum, with picked effective DoF = {path.pickedEffectiveDof.toFixed(2)}.
      </div>
    </div>
  );
}

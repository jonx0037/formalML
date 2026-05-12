import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  lassoCD,
  logspace,
  mulberry32,
  nonzeroCount,
  polynomialVandermonde,
  polyvalIncreasing,
  ridgeFit,
  sampleSinTarget,
  targetSin,
} from './shared/structural-risk-minimization';

// =============================================================================
// RegularizationAsSRM — interactive companion to §7.5.
//
// Dual log-sliders for ridge λ_R ∈ [10⁻⁸, 10⁶] and lasso λ_L ∈ [10⁻⁵, 10²].
// Shows ridge and lasso fits side-by-side on the polynomial-regression toy,
// with their respective effective DoF (continuous for ridge, integer-valued
// number of nonzeros for lasso). The reader sees ridge and lasso as soft SRM
// on the ℓ² and ℓ¹ balls respectively.
//
// Static fallback: public/images/topics/structural-risk-minimization/07_regularization_as_srm.png
// =============================================================================

const HEIGHT = 380;
const N = 80;
const SIGMA = 0.2;
const K_MAX = 15;
const SM_BREAKPOINT = 640;

export default function RegularizationAsSRM() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [logLambdaRidge, setLogLambdaRidge] = useState(-2);
  const [logLambdaLasso, setLogLambdaLasso] = useState(-2);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const lambdaR = Math.pow(10, logLambdaRidge);
  const lambdaL = Math.pow(10, logLambdaLasso);

  const { X, Y } = useMemo(() => sampleSinTarget(N, SIGMA, mulberry32(20260512)), []);
  const ridge = useMemo(() => ridgeFit(X, Y, K_MAX, lambdaR), [X, Y, lambdaR]);
  const lassoCoefs = useMemo(() => lassoCD(X, Y, K_MAX, lambdaL, 2000, 1e-6), [X, Y, lambdaL]);
  const lassoNNZ = useMemo(() => nonzeroCount(lassoCoefs), [lassoCoefs]);

  // Lasso training MSE
  const lassoMse = useMemo(() => {
    let s = 0;
    for (let i = 0; i < N; i++) {
      const r = Y[i] - polyvalIncreasing(lassoCoefs, X[i]);
      s += r * r;
    }
    return s / N;
  }, [X, Y, lassoCoefs]);

  const xGrid = useMemo(() => {
    const arr = new Float64Array(201);
    for (let i = 0; i < arr.length; i++) arr[i] = -1 + (2 * i) / 200;
    return arr;
  }, []);

  const buildPanel = (coefs: Float64Array, label: string, dofLabel: string) =>
    function plot(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
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
      const fitLine = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(polyvalIncreasing(coefs, xGrid[i])));
      g.append('path').datum(d3.range(xGrid.length)).attr('d', fitLine).attr('fill', 'none').attr('stroke', '#ef4444').attr('stroke-width', 2.4);
      g.selectAll(null).data(d3.range(N)).enter().append('circle').attr('cx', (i) => x(X[i])).attr('cy', (i) => y(Y[i])).attr('r', 2.5).attr('fill', '#3b82f6').attr('opacity', 0.6);
      g.append('text').attr('x', 8).attr('y', 14).style('fill', 'var(--color-text)').style('font-size', '11px').style('font-weight', '600').text(label);
      g.append('text').attr('x', 8).attr('y', 28).style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text(dofLabel);
    };

  const ridgeRef = useD3<SVGSVGElement>(
    buildPanel(ridge.coefs, 'Ridge (ℓ²)', `eff DoF = ${ridge.effectiveDof.toFixed(2)}, train MSE = ${ridge.trainingMse.toFixed(4)}`),
    [ridge, containerWidth, isMobile],
  );
  const lassoRef = useD3<SVGSVGElement>(
    buildPanel(lassoCoefs, 'Lasso (ℓ¹)', `# nonzeros = ${lassoNNZ}, train MSE = ${lassoMse.toFixed(4)}`),
    [lassoCoefs, lassoNNZ, lassoMse, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="grid gap-3 mb-3 text-xs" style={{ gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)' }}>
        <label className="flex flex-col gap-1">
          <span><strong>Ridge log₁₀ λ</strong>: <span className="tabular-nums">{logLambdaRidge.toFixed(2)}</span> (λ = {lambdaR.toExponential(1)})</span>
          <input type="range" min={-8} max={6} step={0.1} value={logLambdaRidge} onChange={(e) => setLogLambdaRidge(Number(e.target.value))} aria-label="log of ridge lambda" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>Lasso log₁₀ λ</strong>: <span className="tabular-nums">{logLambdaLasso.toFixed(2)}</span> (λ = {lambdaL.toExponential(1)})</span>
          <input type="range" min={-5} max={2} step={0.1} value={logLambdaLasso} onChange={(e) => setLogLambdaLasso(Number(e.target.value))} aria-label="log of lasso lambda" />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '16px' }}>
        <svg ref={ridgeRef} style={{ width: '100%', height: HEIGHT }} />
        <svg ref={lassoRef} style={{ width: '100%', height: HEIGHT }} />
      </div>
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        Ridge and lasso are both soft SRM on different implicit families. Ridge ℓ² ball: rotationally symmetric, smooth effective DoF. Lasso ℓ¹ ball: cross-polytope, integer-valued capacity. The user picks the penalty geometry; SRM picks the level.
      </div>
    </div>
  );
}

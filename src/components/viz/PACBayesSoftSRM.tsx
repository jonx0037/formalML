import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gaussianKLIsotropic,
  logspace,
  mulberry32,
  pacBayesSlack,
  posteriorAveragedMse,
  ridgeFit,
  sampleSinTarget,
  vandermondeTrace,
} from './shared/structural-risk-minimization';

// =============================================================================
// PACBayesSoftSRM — interactive companion to §9.5.
//
// Two sliders: σ_P (prior width 0.1-3.0), τ (posterior width 0.01-1.0). Sweeps
// ridge λ on a log grid; posterior is N(α̂_λ, τ² I), prior N(0, σ_P² I).
// Computes KL via (9.2), posterior-averaged training MSE, and the PAC-Bayes
// total. Two panels: KL + post-MSE vs effective DoF, and the PAC-Bayes total
// curve with picked $\hat\lambda$ marked.
//
// Static fallback: public/images/topics/structural-risk-minimization/09_pac_bayes_srm.png
// =============================================================================

const HEIGHT = 380;
const N = 80;
const SIGMA = 0.2;
const K_MAX = 15;
const DELTA = 0.05;
const SM_BREAKPOINT = 640;

export default function PACBayesSoftSRM() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sigmaP, setSigmaP] = useState(1.0);
  const [tau, setTau] = useState(0.1);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { X, Y } = useMemo(() => sampleSinTarget(N, SIGMA, mulberry32(20260512)), []);
  const lambdas = useMemo(() => logspace(-6, 4, 50), []);
  const vTrace = useMemo(() => vandermondeTrace(X, K_MAX), [X]);

  const path = useMemo(() => {
    const d = K_MAX + 1;
    const sigmaP2 = sigmaP * sigmaP;
    const tau2 = tau * tau;
    const M = lambdas.length;
    const klArr = new Float64Array(M);
    const postMse = new Float64Array(M);
    const total = new Float64Array(M);
    const effDof = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      const fit = ridgeFit(X, Y, K_MAX, lambdas[i]);
      effDof[i] = fit.effectiveDof;
      let muNormSq = 0;
      for (let j = 0; j < d; j++) muNormSq += fit.coefs[j] * fit.coefs[j];
      klArr[i] = gaussianKLIsotropic(muNormSq, tau2, sigmaP2, d);
      postMse[i] = posteriorAveragedMse(fit.trainingMse, tau2, vTrace, N);
      total[i] = postMse[i] + pacBayesSlack(klArr[i], N, DELTA);
    }
    let argmin = 0;
    let best = total[0];
    for (let i = 1; i < M; i++) {
      if (total[i] < best) {
        best = total[i];
        argmin = i;
      }
    }
    return { klArr, postMse, total, effDof, argmin, hatLambda: lambdas[argmin], pickedEffDof: effDof[argmin] };
  }, [X, Y, vTrace, lambdas, sigmaP, tau]);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 16, right: 50, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0, K_MAX + 1]).range([0, W]);
      const yLeftMax = Math.max(d3.max(path.klArr) || 1, 0.1);
      const yLeft = d3.scaleLinear().domain([0, yLeftMax * 1.1]).range([H, 0]);
      const yRightMax = Math.max(d3.max(path.postMse) || 1, 0.05);
      const yRight = d3.scaleLinear().domain([0, yRightMax * 1.1]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(8));
      g.append('g').call(d3.axisLeft(yLeft).ticks(5));
      g.append('g').attr('transform', `translate(${W},0)`).call(d3.axisRight(yRight).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('effective DoF');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', '#a855f7').style('font-size', '11px').text('KL(Q‖P)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', W + 40).attr('text-anchor', 'middle').style('fill', '#10b981').style('font-size', '11px').text('post-MSE');
      const klLine = d3.line<number>().x((i) => x(path.effDof[i])).y((i) => yLeft(path.klArr[i]));
      g.append('path').datum(d3.range(lambdas.length)).attr('d', klLine).attr('fill', 'none').attr('stroke', '#a855f7').attr('stroke-width', 2);
      const mseLine = d3.line<number>().x((i) => x(path.effDof[i])).y((i) => yRight(path.postMse[i]));
      g.append('path').datum(d3.range(lambdas.length)).attr('d', mseLine).attr('fill', 'none').attr('stroke', '#10b981').attr('stroke-width', 2);
    },
    [path, lambdas, containerWidth, isMobile],
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
      const x = d3.scaleLog().domain([1e-6, 1e4]).range([0, W]);
      const yMax = Math.max(d3.max(path.total) || 1, 0.05);
      const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5, '~e'));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('λ');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('PAC-Bayes total');
      const totalLine = d3.line<number>().x((i) => x(lambdas[i])).y((i) => y(path.total[i]));
      g.append('path').datum(d3.range(lambdas.length)).attr('d', totalLine).attr('fill', 'none').attr('stroke', '#0ea5e9').attr('stroke-width', 2.6);
      g.append('line').attr('x1', x(path.hatLambda)).attr('x2', x(path.hatLambda)).attr('y1', 0).attr('y2', H).attr('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3').attr('opacity', 0.6);
      g.append('text').attr('x', x(path.hatLambda)).attr('y', -4).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text(`λ̂ = ${path.hatLambda.toExponential(1)}`);
    },
    [path, lambdas, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="grid gap-3 mb-3 text-xs" style={{ gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)' }}>
        <label className="flex flex-col gap-1">
          <span><strong>σ_P</strong> (prior width): <span className="tabular-nums">{sigmaP.toFixed(2)}</span></span>
          <input type="range" min={0.1} max={3.0} step={0.05} value={sigmaP} onChange={(e) => setSigmaP(Number(e.target.value))} aria-label="prior standard deviation sigma_P" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>τ</strong> (posterior width): <span className="tabular-nums">{tau.toFixed(2)}</span></span>
          <input type="range" min={0.01} max={1.0} step={0.01} value={tau} onChange={(e) => setTau(Number(e.target.value))} aria-label="posterior standard deviation tau" />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '16px' }}>
        <svg ref={leftRef} style={{ width: '100%', height: HEIGHT }} />
        <svg ref={rightRef} style={{ width: '100%', height: HEIGHT }} />
      </div>
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        Picked effective DoF at λ̂: <strong>{path.pickedEffDof.toFixed(2)}</strong>. KL is the continuous capacity measure; the prior σ_P chooses *which* implicit family, the posterior τ chooses *where* on the family to land.
      </div>
    </div>
  );
}

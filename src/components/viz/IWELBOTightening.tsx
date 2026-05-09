import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  iwelboEstimate,
  makeMeanFieldGaussianSampler,
  marginalLikelihoodGaussianRegression,
  meanFieldELBOGaussianRegression,
  meanFieldGaussianLogDensity,
  polynomialDesignMatrix,
  polynomialJointLogProb,
  polynomialPosteriorMoments,
} from './shared/bayesian-ml';

// =============================================================================
// IWELBOTightening — §8 IWELBO monotone tightness (Burda–Grosse–Salakhutdinov 2016)
// =============================================================================
// Top panel: IWELBO_K vs K on log scale, monotonically rising toward
// closed-form log p(y) at d=4 on the polynomial-regression spine.
// Bottom panel: bias = log p(y) − IWELBO_K decaying.
//
// User slider sets the maximum K shown; computation runs incrementally as the
// slider moves. Mean-field q is the §1 reverse-KL projection; importance
// weights have finite variance for the Gaussian-conjugate target so Theorem 6
// applies and asymptotic convergence holds.
// =============================================================================

const TOP_HEIGHT = 220;
const BOT_HEIGHT = 180;
const MARGIN = { top: 22, right: 28, bottom: 36, left: 56 };

const BLUE = '#1f4e79';
const RED = '#c0504d';
const PURPLE = '#534AB7';

const SIGMA2 = 0.0625;
const TAU2 = 4.0;

const X_GRID: number[] = (() => {
  const n = 30;
  const arr = new Array<number>(n);
  for (let i = 0; i < n; i++) arr[i] = -1.0 + (2.0 * i) / (n - 1);
  return arr;
})();

const Y_FIXED: number[] = [
  -0.6800341583638437, -0.6713683737999192, 0.42393608741006306, -0.18415011140592086,
  0.4392941340513951, 0.401582941643184, 0.5767018005195377, 0.7366067978058071,
  0.9685107342232395, 1.2530275339208585, 0.5639749530625211, 0.8229686955412666,
  0.8714689006673965, 0.8850933446551065, 0.37495678320087766, 0.3984638659202808,
  0.01671499435523549, 0.14515680583738416, -0.0663580519877901, 0.8061700843018722,
  0.5567837571476506, 0.35427849216268303, -0.043878391287867025, 0.2962324867317664,
  0.2688206220970354, 0.3847392672095318, 0.7510136535577099, 1.1020278717873264,
  1.550050451751259, 2.0483306722458696,
];

const K_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];

interface IWPoint {
  k: number;
  iwelbo: number;
  bias: number;
}

function computeIWELBOTrace(degree: number, seed: number, S: number): { trace: IWPoint[]; logEvidence: number; elbo: number } {
  const X = polynomialDesignMatrix(X_GRID, degree);
  const { mean, precision } = polynomialPosteriorMoments(X, Y_FIXED, SIGMA2, TAU2);
  const std = precision.map((_, i) => Math.sqrt(1 / precision[i][i]));
  const logEvidence = marginalLikelihoodGaussianRegression(X, Y_FIXED, SIGMA2, TAU2);
  const elbo = meanFieldELBOGaussianRegression(X, Y_FIXED, SIGMA2, TAU2);
  const logJoint = (theta: number[]): number =>
    polynomialJointLogProb(theta, X, Y_FIXED, SIGMA2, TAU2);
  const logQ = (theta: number[]): number => meanFieldGaussianLogDensity(theta, mean, std);
  const trace: IWPoint[] = K_GRID.map((K) => {
    const sampler = makeMeanFieldGaussianSampler(mean, std, seed + K);
    const sActual = K <= 50 ? S : Math.max(80, Math.floor(S / Math.sqrt(K / 50)));
    const iwelbo = iwelboEstimate(sampler, logJoint, logQ, K, sActual);
    return { k: K, iwelbo, bias: logEvidence - iwelbo };
  });
  return { trace, logEvidence, elbo };
}

export default function IWELBOTightening() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [degree, setDegree] = useState(4);

  const { trace, logEvidence, elbo } = useMemo(
    () => computeIWELBOTrace(degree, 7777, 600),
    [degree],
  );

  const topRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const W = containerWidth;
      const H = TOP_HEIGHT;
      const w = W - MARGIN.left - MARGIN.right;
      const h = H - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([1, 1000]).range([0, w]);
      const yMin = Math.min(elbo, ...trace.map((p) => p.iwelbo));
      const yMax = Math.max(logEvidence, ...trace.map((p) => p.iwelbo));
      const yPad = (yMax - yMin) * 0.15 || 0.5;
      const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('IWELBO_K (nats)');

      // Closed-form line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(logEvidence))
        .attr('y2', yScale(logEvidence))
        .style('stroke', BLUE)
        .style('stroke-width', 1.2)
        .style('stroke-dasharray', '5,3');
      g.append('text')
        .attr('x', w - 4)
        .attr('y', yScale(logEvidence) - 4)
        .attr('text-anchor', 'end')
        .style('fill', BLUE)
        .style('font-size', '11px')
        .text(`log p(y) = ${logEvidence.toFixed(2)}`);

      // ELBO line at K=1
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(elbo))
        .attr('y2', yScale(elbo))
        .style('stroke', RED)
        .style('stroke-width', 1)
        .style('stroke-opacity', 0.5)
        .style('stroke-dasharray', '3,3');

      const line = d3.line<IWPoint>().x((p) => xScale(p.k)).y((p) => yScale(p.iwelbo));
      g.append('path').datum(trace).attr('d', line).attr('fill', 'none').attr('stroke', PURPLE).attr('stroke-width', 2.4);
      g.selectAll('circle.iw')
        .data(trace)
        .enter()
        .append('circle')
        .attr('class', 'iw')
        .attr('cx', (p) => xScale(p.k))
        .attr('cy', (p) => yScale(p.iwelbo))
        .attr('r', 3.5)
        .attr('fill', PURPLE);
    },
    [trace, containerWidth, logEvidence, elbo],
  );

  const botRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const W = containerWidth;
      const H = BOT_HEIGHT;
      const w = W - MARGIN.left - MARGIN.right;
      const h = H - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xScale = d3.scaleLog().domain([1, 1000]).range([0, w]);
      const yMin = 0;
      const yMax = Math.max(0.5, ...trace.map((p) => p.bias)) * 1.1;
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('bias = log p(y) − IWELBO_K');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 28)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('K (importance samples, log scale)');
      const line = d3.line<IWPoint>().x((p) => xScale(p.k)).y((p) => yScale(p.bias));
      g.append('path').datum(trace).attr('d', line).attr('fill', 'none').attr('stroke', RED).attr('stroke-width', 2.2);
      g.selectAll('circle.bias')
        .data(trace)
        .enter()
        .append('circle')
        .attr('class', 'bias')
        .attr('cx', (p) => xScale(p.k))
        .attr('cy', (p) => yScale(p.bias))
        .attr('r', 3)
        .attr('fill', RED);
    },
    [trace, containerWidth],
  );

  return (
    <div ref={containerRef} className="my-6">
      <div
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          borderRadius: 8,
          padding: 16,
          fontFamily: 'var(--font-sans)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 8 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text)' }}>
              IWELBO_K monotone tightening on polynomial regression at d = {degree}
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              Bias decay sublinear in K — heavy importance-weight tails cap practical gains; AIS (§9) gets there in fewer samples.
            </div>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--color-text-secondary)' }}>
            d
            <input type="range" min={2} max={9} step={1} value={degree} onChange={(e) => setDegree(parseInt(e.target.value, 10))} style={{ width: 120 }} />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 16, color: 'var(--color-text)' }}>{degree}</span>
          </label>
        </div>
        <svg ref={topRef} width={containerWidth || 720} height={TOP_HEIGHT} role="img" aria-label="IWELBO_K vs K on log scale, rising toward closed-form log p(y)." />
        <svg ref={botRef} width={containerWidth || 720} height={BOT_HEIGHT} role="img" aria-label="Bias log p(y) − IWELBO_K decaying with K." />
      </div>
    </div>
  );
}

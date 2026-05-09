import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bitsBackCodingExpectedLength,
  makeMeanFieldGaussianSampler,
  marginalLikelihoodGaussianRegression,
  meanFieldELBOGaussianRegression,
  meanFieldGaussianLogDensity,
  polynomialDesignMatrix,
  polynomialPosteriorMoments,
} from './shared/bayesian-ml';

// =============================================================================
// BitsBackCodingDiagram — §10 bits-back coding identity (Theorem 8)
// =============================================================================
// Three horizontal bars on the polynomial-regression d=4 mean-field setup:
//   1. Gross sender cost  E_q[-log p(θ) - log p(y|θ)]
//   2. Bayesian-optimal   -log p(y | M)
//   3. Bits-back net      -ELBO(q) = E_q[-log p(θ) - log p(y|θ) + log q(θ)]
//
// The KL gap between (2) and (3) is the §6 reverse-KL projection bias.
// =============================================================================

const PANEL_HEIGHT = 320;
const MARGIN = { top: 36, right: 24, bottom: 56, left: 200 };

const BLUE = '#1f4e79';
const GREEN = '#2ca02c';
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

interface Result {
  grossCost: number;
  bayesianOptimal: number;
  bitsBackNet: number;
  klGap: number;
  qEntropy: number;
}

function computeBitsBack(degree: number, S: number, seed: number): Result {
  const X = polynomialDesignMatrix(X_GRID, degree);
  const { mean, precision } = polynomialPosteriorMoments(X, Y_FIXED, SIGMA2, TAU2);
  const std = precision.map((_, i) => Math.sqrt(1 / precision[i][i]));
  const dim = mean.length;
  const n = X.length;

  const logEvidence = marginalLikelihoodGaussianRegression(X, Y_FIXED, SIGMA2, TAU2);
  const elbo = meanFieldELBOGaussianRegression(X, Y_FIXED, SIGMA2, TAU2);

  const qSampler = makeMeanFieldGaussianSampler(mean, std, seed);
  const logQ = (theta: number[]): number => meanFieldGaussianLogDensity(theta, mean, std);
  const logPrior = (theta: number[]): number => {
    let bb = 0;
    for (let i = 0; i < dim; i++) bb += theta[i] * theta[i];
    return -0.5 * dim * Math.log(2 * Math.PI * TAU2) - 0.5 * bb / TAU2;
  };
  const logLikelihood = (theta: number[]): number => {
    let sse = 0;
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < dim; j++) pred += X[i][j] * theta[j];
      const r = Y_FIXED[i] - pred;
      sse += r * r;
    }
    return -0.5 * n * Math.log(2 * Math.PI * SIGMA2) - 0.5 * sse / SIGMA2;
  };

  const { codeLength, klGap } = bitsBackCodingExpectedLength(
    qSampler,
    logPrior,
    logQ,
    logLikelihood,
    logEvidence,
    S,
  );

  // Gross cost = -ELBO + E_q[log q] = E_q[-log p(θ) - log p(y|θ)]
  const sampler2 = makeMeanFieldGaussianSampler(mean, std, seed + 1);
  let grossSum = 0;
  for (let s = 0; s < S; s++) {
    const theta = sampler2();
    grossSum += -logPrior(theta) - logLikelihood(theta);
  }
  const grossCost = grossSum / S;

  // Closed-form differential entropy of mean-field Gaussian:
  //   H(q) = ½ Σ log(2π e σ²_j)
  let qEntropy = 0;
  for (let i = 0; i < dim; i++) qEntropy += 0.5 * Math.log(2 * Math.PI * Math.E * std[i] * std[i]);

  return {
    grossCost,
    bayesianOptimal: -logEvidence,
    bitsBackNet: -elbo, // = codeLength up to MC noise
    klGap,
    qEntropy,
  };
}

export default function BitsBackCodingDiagram() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [degree, setDegree] = useState(4);

  const result = useMemo(() => computeBitsBack(degree, 8000, 12121), [degree]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const W = containerWidth;
      const H = PANEL_HEIGHT;
      const w = W - MARGIN.left - MARGIN.right;
      const h = H - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const items = [
        { label: 'Gross sender cost', value: result.grossCost, color: BLUE, sub: 'E_q[−log p(θ) − log p(y|θ)]' },
        { label: 'Bayesian-optimal', value: result.bayesianOptimal, color: GREEN, sub: '−log p(y | M)' },
        { label: 'Bits-back net (−ELBO)', value: result.bitsBackNet, color: RED, sub: 'E_q[−log p(θ) − log p(y|θ) + log q]' },
      ];

      const yScale = d3
        .scaleBand<string>()
        .domain(items.map((it) => it.label))
        .range([0, h])
        .padding(0.25);
      const xMin = Math.min(0, ...items.map((it) => it.value)) - 1;
      const xMax = Math.max(...items.map((it) => it.value)) + 2;
      const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, w]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 36)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Description length (nats)');

      items.forEach((it) => {
        const yPos = yScale(it.label) ?? 0;
        const bw = yScale.bandwidth();
        // Label
        g.append('text')
          .attr('x', -8)
          .attr('y', yPos + bw / 2 - 4)
          .attr('text-anchor', 'end')
          .style('fill', 'var(--color-text)')
          .style('font-size', '12px')
          .style('font-weight', 600)
          .text(it.label);
        g.append('text')
          .attr('x', -8)
          .attr('y', yPos + bw / 2 + 12)
          .attr('text-anchor', 'end')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '10px')
          .style('font-family', 'var(--font-mono)')
          .text(it.sub);
        // Bar from 0 to value
        g.append('rect')
          .attr('x', xScale(0))
          .attr('y', yPos)
          .attr('width', Math.max(1, xScale(it.value) - xScale(0)))
          .attr('height', bw)
          .attr('fill', it.color)
          .attr('fill-opacity', 0.78);
        // Value label
        g.append('text')
          .attr('x', xScale(it.value) + 6)
          .attr('y', yPos + bw / 2 + 4)
          .style('fill', 'var(--color-text)')
          .style('font-size', '12px')
          .style('font-weight', 600)
          .text(`${it.value.toFixed(2)} nats`);
      });

      // KL gap annotation: between bayesianOptimal and bitsBackNet
      const optY = (yScale('Bayesian-optimal') ?? 0) + yScale.bandwidth();
      const netY = yScale('Bits-back net (−ELBO)') ?? 0;
      const optEnd = xScale(result.bayesianOptimal);
      const netEnd = xScale(result.bitsBackNet);
      g.append('rect')
        .attr('x', optEnd)
        .attr('y', optY + 4)
        .attr('width', Math.max(1, netEnd - optEnd))
        .attr('height', netY - optY - 8)
        .attr('fill', PURPLE)
        .attr('fill-opacity', 0.18);
      g.append('text')
        .attr('x', (optEnd + netEnd) / 2)
        .attr('y', (optY + netY) / 2 + 4)
        .attr('text-anchor', 'middle')
        .style('fill', PURPLE)
        .style('font-size', '11px')
        .style('font-weight', 600)
        .text(`KL gap = ${result.klGap.toFixed(2)} nats`);
    },
    [result, containerWidth],
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
              Bits-back coding identity at d = {degree}
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              Theorem 8: bits-back net = −ELBO; the gap to the Shannon-optimal length is the §6 KL projection bias.
            </div>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--color-text-secondary)' }}>
            d
            <input type="range" min={2} max={9} step={1} value={degree} onChange={(e) => setDegree(parseInt(e.target.value, 10))} style={{ width: 120 }} />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 16, color: 'var(--color-text)' }}>{degree}</span>
          </label>
        </div>
        <svg ref={svgRef} width={containerWidth || 720} height={PANEL_HEIGHT} role="img" aria-label="Three horizontal bars showing gross sender cost, Bayesian-optimal length, and bits-back net length." />
      </div>
    </div>
  );
}

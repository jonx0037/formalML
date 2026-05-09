import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  marginalLikelihoodGaussianRegression,
  meanFieldELBOGaussianRegression,
  polynomialDesignMatrix,
  polynomialPosteriorMoments,
} from './shared/bayesian-ml';

// =============================================================================
// EvidenceDecompositionBar — §2 evidence decomposition stacked bar
// =============================================================================
// Visualizes Theorem 1: log p(y) = ELBO(q) + KL(q || p(·|y)).
//
// Three nested variational families on the polynomial-regression d=4 target:
//   1. Mean-field Gaussian          (4+1 diagonal variances)
//   2. Block-diagonal              (intercept | rest split into 2 blocks)
//   3. Full-rank Gaussian          (KL = 0 for Gaussian-conjugate posterior)
//
// Total bar height = log p(y) is constant across families. The proportion
// split between ELBO segment (green) and KL gap (red) shifts as the family
// grows.
// =============================================================================

const PANEL_HEIGHT = 380;
const MARGIN = { top: 24, right: 28, bottom: 56, left: 64 };

const ELBO_COLOR = '#2ca02c';
const KL_COLOR = '#c0504d';

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

const SIGMA2 = 0.0625;
const TAU2 = 4.0;

interface FamilyResult {
  name: string;
  parameters: number;
  elbo: number;
  klGap: number;
  logEvidence: number;
}

/**
 * KL gap for matched-mean Gaussians N(μ, Σ_q) || N(μ, Σ_post). Closed form:
 *   KL = ½ [ tr(Λ_post Σ_q) − d + log det Σ_post − log det Σ_q ]
 * For a *block-diagonal* q on a 2-block partition (intercept block + slope/curvature
 * block), Σ_q = blockdiag(Σ₁, Σ₂) where each block uses the corresponding
 * sub-block of Σ_post. This captures all within-block correlation, missing only
 * cross-block correlation.
 */
function computeBlockDiagonalELBO(degree: number): number {
  const X = polynomialDesignMatrix(X_GRID, degree);
  const { precision, precisionLogDet } = polynomialPosteriorMoments(X, Y_FIXED, SIGMA2, TAU2);
  const dim = degree + 1;
  // 2-block partition: first block = {0}, second block = {1, ..., dim-1}
  // Σ_post = Λ_post^{-1}; we need the diagonal blocks of Σ_post, which equal
  // (Λ_post,block)^{-1} only if Λ_post is itself block-diagonal — which is
  // not generally true. Compute Σ_post explicitly via Cholesky-inverse.
  // Σ_post,block = Σ_post[block, block] (sub-matrix), Σ_q = blockdiag(Σ₁, Σ₂).
  const sigmaPost = invertSPD(precision);
  // Block 1 = single coefficient (intercept)
  const block1 = [[sigmaPost[0][0]]];
  // Block 2 = remaining (dim−1) × (dim−1) sub-matrix
  const block2: number[][] = [];
  for (let i = 1; i < dim; i++) {
    const row: number[] = [];
    for (let j = 1; j < dim; j++) row.push(sigmaPost[i][j]);
    block2.push(row);
  }
  // KL = ½ [ tr(Λ_post Σ_q) − d + log det Σ_post − log det Σ_q ]
  const logDetSigmaQ = Math.log(block1[0][0]) + logDet(block2);
  const logDetSigmaPost = -precisionLogDet;
  // tr(Λ_post Σ_q) — only diagonal blocks of Λ contribute
  let trace = 0;
  for (let i = 0; i < dim; i++) {
    if (i === 0) {
      trace += precision[0][0] * block1[0][0];
    } else {
      for (let j = 1; j < dim; j++) trace += precision[i][j] * block2[i - 1][j - 1];
    }
  }
  const kl = 0.5 * (trace - dim + logDetSigmaPost - logDetSigmaQ);
  const logP = marginalLikelihoodGaussianRegression(X, Y_FIXED, SIGMA2, TAU2);
  return logP - kl;
}

function invertSPD(A: number[][]): number[][] {
  const n = A.length;
  const aug: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(2 * n).fill(0);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n + i] = 1;
    aug.push(row);
  }
  for (let i = 0; i < n; i++) {
    let pivot = i;
    let pivotVal = Math.abs(aug[i][i]);
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(aug[k][i]) > pivotVal) {
        pivot = k;
        pivotVal = Math.abs(aug[k][i]);
      }
    }
    if (pivot !== i) [aug[i], aug[pivot]] = [aug[pivot], aug[i]];
    const div = aug[i][i];
    for (let j = 0; j < 2 * n; j++) aug[i][j] /= div;
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = aug[k][i];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) aug[k][j] -= factor * aug[i][j];
    }
  }
  const inv: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n);
    for (let j = 0; j < n; j++) row[j] = aug[i][n + j];
    inv.push(row);
  }
  return inv;
}

function logDet(A: number[][]): number {
  // Compute via LU decomposition on a copy
  const n = A.length;
  const M = A.map((row) => [...row]);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    let pivot = i;
    let pv = Math.abs(M[i][i]);
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > pv) {
        pivot = k;
        pv = Math.abs(M[k][i]);
      }
    }
    if (pivot !== i) [M[i], M[pivot]] = [M[pivot], M[i]];
    if (M[i][i] === 0) return -Infinity;
    acc += Math.log(Math.abs(M[i][i]));
    for (let k = i + 1; k < n; k++) {
      const factor = M[k][i] / M[i][i];
      for (let j = i; j < n; j++) M[k][j] -= factor * M[i][j];
    }
  }
  return acc;
}

function computeFamilies(degree: number): FamilyResult[] {
  const X = polynomialDesignMatrix(X_GRID, degree);
  const logP = marginalLikelihoodGaussianRegression(X, Y_FIXED, SIGMA2, TAU2);
  const elboMF = meanFieldELBOGaussianRegression(X, Y_FIXED, SIGMA2, TAU2);
  const elboBlock = computeBlockDiagonalELBO(degree);
  return [
    {
      name: 'Mean-field',
      parameters: 2 * (degree + 1),
      elbo: elboMF,
      klGap: logP - elboMF,
      logEvidence: logP,
    },
    {
      name: 'Block-diagonal (2 blocks)',
      parameters: 1 + (degree * (degree + 1)) / 2 + 1,
      elbo: elboBlock,
      klGap: logP - elboBlock,
      logEvidence: logP,
    },
    {
      name: 'Full-rank Gaussian',
      parameters: (degree + 1) * (degree + 2) / 2 + (degree + 1),
      elbo: logP,
      klGap: 0,
      logEvidence: logP,
    },
  ];
}

export default function EvidenceDecompositionBar() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [degree, setDegree] = useState(4);

  const families = useMemo(() => computeFamilies(degree), [degree]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const W = containerWidth;
      const H = PANEL_HEIGHT;
      const w = W - MARGIN.left - MARGIN.right;
      const h = H - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3
        .scaleBand<string>()
        .domain(families.map((f) => f.name))
        .range([0, w])
        .padding(0.35);

      // y axis flipped: log p is at top, KL grows downward (i.e., -ELBO is below)
      // We render in a "free energy" frame: bar from 0 to -ELBO, with -log p marked.
      const minNeg = 0;
      const maxNeg = Math.max(...families.map((f) => -f.elbo)) * 1.05;
      const yScale = d3.scaleLinear().domain([minNeg, maxNeg]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '12px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -48)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Description length / variational free energy (nats)');

      // Reference line at -log p (constant across families)
      const negLogP = -families[0].logEvidence;
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(negLogP))
        .attr('y2', yScale(negLogP))
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', 1.5)
        .style('stroke-dasharray', '6,3');
      g.append('text')
        .attr('x', w - 4)
        .attr('y', yScale(negLogP) - 4)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text)')
        .style('font-size', '11px')
        .text(`−log p(y) = ${negLogP.toFixed(2)} nats`);

      // Bars: ELBO segment (green) from 0 to -log p, KL segment (red) above
      families.forEach((f) => {
        const x = xScale(f.name) ?? 0;
        const bw = xScale.bandwidth();
        const elboTop = yScale(negLogP);
        const elboBottom = yScale(0);
        // ELBO segment: from 0 down to -log p (height = -log p)
        g.append('rect')
          .attr('x', x)
          .attr('y', elboTop)
          .attr('width', bw)
          .attr('height', elboBottom - elboTop)
          .attr('fill', ELBO_COLOR)
          .attr('fill-opacity', 0.7);
        // KL segment: from -log p down to -ELBO (height = KL)
        const klTop = yScale(-f.elbo);
        const klBottom = yScale(negLogP);
        if (f.klGap > 1e-6) {
          g.append('rect')
            .attr('x', x)
            .attr('y', klTop)
            .attr('width', bw)
            .attr('height', klBottom - klTop)
            .attr('fill', KL_COLOR)
            .attr('fill-opacity', 0.65);
        }
        // Labels
        g.append('text')
          .attr('x', x + bw / 2)
          .attr('y', yScale(negLogP / 2))
          .attr('text-anchor', 'middle')
          .style('fill', '#0a3a18')
          .style('font-size', '11px')
          .style('font-weight', 600)
          .text(`ELBO ${f.elbo.toFixed(2)}`);
        if (f.klGap > 0.05) {
          g.append('text')
            .attr('x', x + bw / 2)
            .attr('y', yScale(negLogP) - (yScale(negLogP) - klTop) / 2)
            .attr('text-anchor', 'middle')
            .style('fill', '#5b1410')
            .style('font-size', '11px')
            .style('font-weight', 600)
            .text(`KL ${f.klGap.toFixed(2)}`);
        }
      });

      // Legend
      const legend = g.append('g').attr('transform', `translate(${w - 220},6)`);
      [
        { color: ELBO_COLOR, label: 'ELBO segment' },
        { color: KL_COLOR, label: 'KL projection gap' },
      ].forEach((entry, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 18})`);
        row
          .append('rect')
          .attr('width', 14)
          .attr('height', 12)
          .attr('fill', entry.color)
          .attr('fill-opacity', 0.7);
        row
          .append('text')
          .attr('x', 20)
          .attr('y', 10)
          .style('fill', 'var(--color-text)')
          .style('font-size', '11px')
          .text(entry.label);
      });
    },
    [families, containerWidth],
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
              Evidence decomposition: log p(y) = ELBO(q) + KL(q ‖ p(·|y))
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              Bar total height stays at −log p(y); the ELBO/KL split shifts as the family grows.
            </div>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--color-text-secondary)' }}>
            <span>Polynomial degree d</span>
            <input
              type="range"
              min={2}
              max={9}
              step={1}
              value={degree}
              onChange={(e) => setDegree(parseInt(e.target.value, 10))}
              style={{ width: 140 }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 24, color: 'var(--color-text)' }}>{degree}</span>
          </label>
        </div>
        <svg ref={svgRef} width={containerWidth || 720} height={PANEL_HEIGHT} role="img" aria-label="Stacked bars showing log p(y) decomposed into ELBO + KL across three nested variational families." />
      </div>
    </div>
  );
}

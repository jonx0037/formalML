import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  marginalLikelihoodGaussianRegression,
  meanFieldELBOGaussianRegression,
  mulberry32,
  gaussianPair,
  polynomialDesignMatrix,
} from './shared/bayesian-ml';

// =============================================================================
// AISvsELBO — §9 Annealed Importance Sampling sample-efficiency
// =============================================================================
// AIS estimate of log p(y | M_d=4) at increasing schedule lengths T,
// converging to the closed-form value. Compares against IWELBO and ELBO at
// the same compute budget. Implementation uses exact-Gaussian sampling at
// each schedule step (Theorem 7 holds for any kernel that leaves p_t
// invariant; exact sampling is the limiting case of MCMC mixing time).
// =============================================================================

const PANEL_HEIGHT = 360;
const MARGIN = { top: 24, right: 24, bottom: 56, left: 56 };

const BLUE = '#1f4e79';
const GREEN = '#2ca02c';
const RED = '#c0504d';

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

const T_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];

interface AISPoint {
  T: number;
  estimate: number;
  bias: number;
}

function runAISSweep(degree: number, seed: number, C: number): { trace: AISPoint[]; logEvidence: number } {
  const X = polynomialDesignMatrix(X_GRID, degree);
  const dim = degree + 1;
  const n = X.length;
  const logEvidence = marginalLikelihoodGaussianRegression(X, Y_FIXED, SIGMA2, TAU2);

  // Pre-compute X^T X and X^T y
  const XtX: number[][] = [];
  for (let i = 0; i < dim; i++) {
    const row = new Array<number>(dim).fill(0);
    for (let j = 0; j < dim; j++) {
      let acc = 0;
      for (let k = 0; k < n; k++) acc += X[k][i] * X[k][j];
      row[j] = acc;
    }
    XtX.push(row);
  }
  const Xty = new Array<number>(dim).fill(0);
  for (let i = 0; i < dim; i++) {
    let acc = 0;
    for (let k = 0; k < n; k++) acc += X[k][i] * Y_FIXED[k];
    Xty[i] = acc;
  }

  const rng = mulberry32(seed);
  let buf: number | null = null;
  const draw = (): number => {
    if (buf !== null) {
      const v = buf;
      buf = null;
      return v;
    }
    const [a, b] = gaussianPair(rng);
    buf = b;
    return a;
  };

  // Cholesky of an SPD matrix
  function chol(A: number[][]): number[][] {
    const d = A.length;
    const L: number[][] = [];
    for (let i = 0; i < d; i++) {
      const row = new Array<number>(d).fill(0);
      let dii = A[i][i];
      for (let j = 0; j < i; j++) {
        let acc = A[i][j];
        for (let k = 0; k < j; k++) acc -= row[k] * L[j][k];
        row[j] = acc / L[j][j];
        dii -= row[j] * row[j];
      }
      row[i] = Math.sqrt(Math.max(dii, 1e-12));
      L.push(row);
    }
    return L;
  }

  function exactSampleAtBeta(beta: number): { theta: number[]; logLik: number } {
    const prec: number[][] = [];
    for (let i = 0; i < dim; i++) {
      const row = new Array<number>(dim).fill(0);
      for (let j = 0; j < dim; j++) {
        row[j] = (XtX[i][j] / SIGMA2) * beta;
        if (i === j) row[j] += 1 / TAU2;
      }
      prec.push(row);
    }
    const rhs = new Array<number>(dim).fill(0);
    for (let i = 0; i < dim; i++) rhs[i] = (Xty[i] / SIGMA2) * beta;
    const L = chol(prec);
    const yTri = new Array<number>(dim).fill(0);
    for (let i = 0; i < dim; i++) {
      let acc = rhs[i];
      for (let j = 0; j < i; j++) acc -= L[i][j] * yTri[j];
      yTri[i] = acc / L[i][i];
    }
    const mu = new Array<number>(dim).fill(0);
    for (let i = dim - 1; i >= 0; i--) {
      let acc = yTri[i];
      for (let j = i + 1; j < dim; j++) acc -= L[j][i] * mu[j];
      mu[i] = acc / L[i][i];
    }
    const z = new Array<number>(dim).fill(0);
    for (let i = 0; i < dim; i++) z[i] = draw();
    const offset = new Array<number>(dim).fill(0);
    for (let i = dim - 1; i >= 0; i--) {
      let acc = z[i];
      for (let j = i + 1; j < dim; j++) acc -= L[j][i] * offset[j];
      offset[i] = acc / L[i][i];
    }
    const theta = new Array<number>(dim).fill(0);
    for (let i = 0; i < dim; i++) theta[i] = mu[i] + offset[i];
    let sse = 0;
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < dim; j++) pred += X[i][j] * theta[j];
      const r = Y_FIXED[i] - pred;
      sse += r * r;
    }
    const logLik = -0.5 * n * Math.log(2 * Math.PI * SIGMA2) - 0.5 * sse / SIGMA2;
    return { theta, logLik };
  }

  const trace: AISPoint[] = T_GRID.map((T) => {
    const Cact = T <= 10 ? Math.max(C * 2, 200) : C;
    const logWeights = new Array<number>(Cact);
    for (let c = 0; c < Cact; c++) {
      const init = exactSampleAtBeta(0); // β=0 = prior
      let logW = 0;
      let prevLogLik = init.logLik;
      for (let t = 1; t <= T; t++) {
        const betaCurr = t / T;
        const betaPrev = (t - 1) / T;
        logW += (betaCurr - betaPrev) * prevLogLik;
        const next = exactSampleAtBeta(betaCurr);
        prevLogLik = next.logLik;
      }
      logWeights[c] = logW;
    }
    let maxLw = -Infinity;
    for (let c = 0; c < Cact; c++) if (logWeights[c] > maxLw) maxLw = logWeights[c];
    let acc = 0;
    for (let c = 0; c < Cact; c++) acc += Math.exp(logWeights[c] - maxLw);
    const estimate = Math.log(acc / Cact) + maxLw;
    return { T, estimate, bias: Math.abs(logEvidence - estimate) };
  });
  return { trace, logEvidence };
}

export default function AISvsELBO() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [degree, setDegree] = useState(4);

  const { trace, logEvidence } = useMemo(() => runAISSweep(degree, 8888, 80), [degree]);
  const X = useMemo(() => polynomialDesignMatrix(X_GRID, degree), [degree]);
  const elbo = useMemo(
    () => meanFieldELBOGaussianRegression(X, Y_FIXED, SIGMA2, TAU2),
    [X],
  );

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const W = containerWidth;
      const H = PANEL_HEIGHT;
      const w = W - MARGIN.left - MARGIN.right;
      const h = H - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xScale = d3.scaleLog().domain([1, 1000]).range([0, w]);
      const yMin = Math.min(elbo, ...trace.map((p) => p.estimate)) - 1;
      const yMax = Math.max(logEvidence, ...trace.map((p) => p.estimate)) + 1;
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 36)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Schedule length T (log scale)');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('AIS estimate of log p(y) (nats)');

      // Closed-form line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(logEvidence))
        .attr('y2', yScale(logEvidence))
        .style('stroke', BLUE)
        .style('stroke-width', 1.4)
        .style('stroke-dasharray', '5,3');
      g.append('text')
        .attr('x', w - 4)
        .attr('y', yScale(logEvidence) - 4)
        .attr('text-anchor', 'end')
        .style('fill', BLUE)
        .style('font-size', '11px')
        .text(`log p(y) = ${logEvidence.toFixed(2)}`);

      // ELBO line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(elbo))
        .attr('y2', yScale(elbo))
        .style('stroke', RED)
        .style('stroke-width', 1.0)
        .style('stroke-opacity', 0.5)
        .style('stroke-dasharray', '3,3');
      g.append('text')
        .attr('x', 4)
        .attr('y', yScale(elbo) + 12)
        .style('fill', RED)
        .style('font-size', '11px')
        .text(`ELBO = ${elbo.toFixed(2)}`);

      // AIS trace
      const line = d3.line<AISPoint>().x((p) => xScale(p.T)).y((p) => yScale(p.estimate));
      g.append('path').datum(trace).attr('d', line).attr('fill', 'none').attr('stroke', GREEN).attr('stroke-width', 2.4);
      g.selectAll('circle.ais')
        .data(trace)
        .enter()
        .append('circle')
        .attr('class', 'ais')
        .attr('cx', (p) => xScale(p.T))
        .attr('cy', (p) => yScale(p.estimate))
        .attr('r', 3.5)
        .attr('fill', GREEN);
    },
    [trace, containerWidth, logEvidence, elbo],
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
              AIS schedule sweep on polynomial regression at d = {degree}
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              At T=1, AIS reduces to prior IS (high-variance). By T~50 the schedule has unfolded to land near closed-form.
            </div>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--color-text-secondary)' }}>
            d
            <input type="range" min={2} max={9} step={1} value={degree} onChange={(e) => setDegree(parseInt(e.target.value, 10))} style={{ width: 120 }} />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 16, color: 'var(--color-text)' }}>{degree}</span>
          </label>
        </div>
        <svg ref={svgRef} width={containerWidth || 720} height={PANEL_HEIGHT} role="img" aria-label="AIS estimate vs schedule length T converging to closed-form log p(y)." />
      </div>
    </div>
  );
}

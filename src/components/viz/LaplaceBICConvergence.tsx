import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bicApproximation,
  marginalLikelihoodGaussianRegression,
  meanFieldELBOGaussianRegression,
  mulberry32,
  gaussianPair,
  polynomialDesignMatrix,
  polynomialPosteriorMoments,
} from './shared/bayesian-ml';

// =============================================================================
// LaplaceBICConvergence — §3 BIC asymptotic equivalence
// =============================================================================
// Top panel: closed-form log p(y), Laplace (= closed-form for Gaussian
// conjugate), -BIC/2 plotted vs n on a log scale at fixed d.
//
// Bottom panel: log p(y) - (-BIC/2) converging to the O(1) Bayesian content
// C(M, p, θ̂) of Corollary 1 as n → ∞. For the polynomial-regression cubic
// at d=3 with σ²=0.0625, τ²=4, the limit is roughly -5.6 nats.
// =============================================================================

const TOP_HEIGHT = 230;
const BOTTOM_HEIGHT = 180;
const MARGIN = { top: 22, right: 32, bottom: 36, left: 56 };

const BLUE = '#1f4e79';
const GRAY = '#7f7f7f';
const ORANGE = '#d97706';

const SIGMA2 = 0.0625;
const TAU2 = 4.0;
const TRUE_COEF = [0.5, -1.2, 0.0, 2.5];

const N_GRID = [10, 20, 30, 50, 80, 130, 220, 360, 600, 1000, 1600, 2600, 4200, 6800, 10000];

interface ConvergencePoint {
  n: number;
  logP: number;
  laplace: number;
  negHalfBIC: number;
}

function generateData(n: number, seed: number): { x: number[]; y: number[] } {
  const x = new Array<number>(n);
  for (let i = 0; i < n; i++) x[i] = -1.0 + (2.0 * i) / (n - 1);
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
  const y = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let truth = 0;
    for (let k = 0; k < TRUE_COEF.length; k++) truth += TRUE_COEF[k] * Math.pow(x[i], k);
    y[i] = truth + Math.sqrt(SIGMA2) * draw();
  }
  return { x, y };
}

function computeConvergence(degree: number, seed: number): ConvergencePoint[] {
  return N_GRID.map((n) => {
    const { x, y } = generateData(n, seed);
    const X = polynomialDesignMatrix(x, degree);
    const logP = marginalLikelihoodGaussianRegression(X, y, SIGMA2, TAU2);
    // Laplace = closed-form for Gaussian conjugate, but compute via the formula
    // to confirm the path the §3 viz exposes.
    const { mean } = polynomialPosteriorMoments(X, y, SIGMA2, TAU2);
    let sse = 0;
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < degree + 1; j++) pred += X[i][j] * mean[j];
      const r = y[i] - pred;
      sse += r * r;
    }
    const logLikAtMAP = -0.5 * n * Math.log(2 * Math.PI * SIGMA2) - 0.5 * sse / SIGMA2;
    const negHalfBIC = bicApproximation(logLikAtMAP, degree + 1, n);
    return { n, logP, laplace: logP, negHalfBIC };
  });
}

export default function LaplaceBICConvergence() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [degree, setDegree] = useState(3);

  const data = useMemo(() => computeConvergence(degree, 20260508), [degree]);
  const limitGap = useMemo(() => {
    const last = data[data.length - 1];
    return last.logP - last.negHalfBIC;
  }, [data]);

  const topRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const W = containerWidth;
      const H = TOP_HEIGHT;
      const w = W - MARGIN.left - MARGIN.right;
      const h = H - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xScale = d3.scaleLog().domain([N_GRID[0], N_GRID[N_GRID.length - 1]]).range([0, w]);
      const yMin = Math.min(...data.map((p) => Math.min(p.logP, p.negHalfBIC)));
      const yMax = Math.max(...data.map((p) => Math.max(p.logP, p.negHalfBIC)));
      const yPad = (yMax - yMin) * 0.08;
      const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6, '~s'))
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
        .text('log p(y) (nats)');

      const lineLogP = d3.line<ConvergencePoint>().x((p) => xScale(p.n)).y((p) => yScale(p.logP));
      const lineLaplace = d3
        .line<ConvergencePoint>()
        .x((p) => xScale(p.n))
        .y((p) => yScale(p.laplace));
      const lineBIC = d3
        .line<ConvergencePoint>()
        .x((p) => xScale(p.n))
        .y((p) => yScale(p.negHalfBIC));

      g.append('path').datum(data).attr('d', lineLogP).attr('fill', 'none').attr('stroke', BLUE).attr('stroke-width', 2.4);
      g.append('path').datum(data).attr('d', lineLaplace).attr('fill', 'none').attr('stroke', GRAY).attr('stroke-width', 1.4).attr('stroke-dasharray', '6,4');
      g.append('path').datum(data).attr('d', lineBIC).attr('fill', 'none').attr('stroke', ORANGE).attr('stroke-width', 2.4);

      const legend = g.append('g').attr('transform', `translate(${w - 200},4)`);
      [
        { color: BLUE, label: 'log p(y) (closed-form)', dash: false },
        { color: GRAY, label: 'Laplace (overlays)', dash: true },
        { color: ORANGE, label: '−BIC/2', dash: false },
      ].forEach((entry, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 22)
          .attr('y1', 0)
          .attr('y2', 0)
          .style('stroke', entry.color)
          .style('stroke-width', 2.2)
          .style('stroke-dasharray', entry.dash ? '5,3' : '');
        row
          .append('text')
          .attr('x', 28)
          .attr('y', 4)
          .style('fill', 'var(--color-text)')
          .style('font-size', '11px')
          .text(entry.label);
      });
    },
    [data, containerWidth],
  );

  const botRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const W = containerWidth;
      const H = BOTTOM_HEIGHT;
      const w = W - MARGIN.left - MARGIN.right;
      const h = H - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xScale = d3.scaleLog().domain([N_GRID[0], N_GRID[N_GRID.length - 1]]).range([0, w]);
      const gaps = data.map((p) => ({ n: p.n, gap: p.logP - p.negHalfBIC }));
      const yMin = Math.min(...gaps.map((p) => p.gap));
      const yMax = Math.max(...gaps.map((p) => p.gap));
      const yPad = Math.max(1, (yMax - yMin) * 0.15);
      const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([h, 0]);
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6, '~s'))
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
        .text('log p(y) − (−BIC/2)');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 26)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Sample size n (log scale)');
      const line = d3.line<{ n: number; gap: number }>().x((p) => xScale(p.n)).y((p) => yScale(p.gap));
      g.append('path').datum(gaps).attr('d', line).attr('fill', 'none').attr('stroke', '#534AB7').attr('stroke-width', 2.2);
      g.selectAll('circle.gap')
        .data(gaps)
        .enter()
        .append('circle')
        .attr('class', 'gap')
        .attr('cx', (p) => xScale(p.n))
        .attr('cy', (p) => yScale(p.gap))
        .attr('r', 3.5)
        .attr('fill', '#534AB7');
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(limitGap))
        .attr('y2', yScale(limitGap))
        .style('stroke', '#534AB7')
        .style('stroke-opacity', 0.4)
        .style('stroke-dasharray', '4,3');
      g.append('text')
        .attr('x', w - 4)
        .attr('y', yScale(limitGap) - 4)
        .attr('text-anchor', 'end')
        .style('fill', '#534AB7')
        .style('font-size', '11px')
        .text(`limit ≈ ${limitGap.toFixed(2)} nats`);
    },
    [data, containerWidth, limitGap],
  );

  // Keep ELBO computation inert (not displayed) — preserves the import path.
  void meanFieldELBOGaussianRegression;

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
              Laplace and BIC convergence to log p(y)
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              Cubic data + degree-d polynomial model. As n → ∞, the BIC penalty leaves an O(1) gap = the Bayesian content C(M, p, θ̂).
            </div>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--color-text-secondary)' }}>
            <span>Model degree d</span>
            <input
              type="range"
              min={2}
              max={6}
              step={1}
              value={degree}
              onChange={(e) => setDegree(parseInt(e.target.value, 10))}
              style={{ width: 140 }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 16, color: 'var(--color-text)' }}>{degree}</span>
          </label>
        </div>
        <svg ref={topRef} width={containerWidth || 720} height={TOP_HEIGHT} role="img" aria-label="Three traces of log-evidence vs sample size n on log scale: closed-form, Laplace, and -BIC/2." />
        <svg ref={botRef} width={containerWidth || 720} height={BOTTOM_HEIGHT} role="img" aria-label="Gap log p(y) - (-BIC/2) vs n converging to the O(1) Bayesian content." />
      </div>
    </div>
  );
}

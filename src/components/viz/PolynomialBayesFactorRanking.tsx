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
// PolynomialBayesFactorRanking — §5 polynomial Bayes-factor ranking
// =============================================================================
// Three estimators of log p(y | M_d) — closed-form, Laplace (= closed-form for
// Gaussian conjugate), and mean-field ELBO — converted to log Bayes factors
// against the argmax winner. Bars grouped by d=1..9.
//
// Sliders: n, σ², τ². The data is regenerated as n changes; closed-form and
// ELBO update live. The figure mirrors notebook figures/05_polynomial_bayes_factors.png.
// =============================================================================

const PANEL_HEIGHT = 380;
const MARGIN = { top: 28, right: 24, bottom: 56, left: 56 };
const DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9];

const BLUE = '#1f4e79';
const GRAY = '#7f7f7f';
const RED = '#c0504d';

const TRUE_COEF = [0.5, -1.2, 0.0, 2.5];

interface RankingPoint {
  d: number;
  logBFClosed: number;
  logBFLaplace: number;
  logBFELBO: number;
}

function generateData(n: number, sigma2: number, seed: number): { x: number[]; y: number[] } {
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
    y[i] = truth + Math.sqrt(sigma2) * draw();
  }
  return { x, y };
}

function computeRanking(n: number, sigma2: number, tau2: number, seed: number): RankingPoint[] {
  const { x, y } = generateData(n, sigma2, seed);
  const rows = DEGREES.map((d) => {
    const X = polynomialDesignMatrix(x, d);
    const lp = marginalLikelihoodGaussianRegression(X, y, sigma2, tau2);
    const elbo = meanFieldELBOGaussianRegression(X, y, sigma2, tau2);
    return { d, logP: lp, elbo };
  });
  // Argmax by closed-form
  let maxLP = rows[0].logP;
  let maxELBO = rows[0].elbo;
  for (const row of rows) {
    if (row.logP > maxLP) maxLP = row.logP;
    if (row.elbo > maxELBO) maxELBO = row.elbo;
  }
  return rows.map((row) => ({
    d: row.d,
    logBFClosed: row.logP - maxLP,
    logBFLaplace: row.logP - maxLP,
    logBFELBO: row.elbo - maxELBO,
  }));
}

export default function PolynomialBayesFactorRanking() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(30);
  const [sigma2, setSigma2] = useState(0.0625);
  const [tau2, setTau2] = useState(4.0);

  const data = useMemo(
    () => computeRanking(n, sigma2, tau2, 20260508),
    [n, sigma2, tau2],
  );

  const argMax = useMemo(() => {
    const idx = data.findIndex((row) => row.logBFClosed === 0);
    return data[idx >= 0 ? idx : 0].d;
  }, [data]);

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
        .scaleBand<number>()
        .domain(DEGREES)
        .range([0, w])
        .padding(0.18);
      const groupScale = d3
        .scaleBand<string>()
        .domain(['closed', 'laplace', 'elbo'])
        .range([0, xScale.bandwidth()])
        .padding(0.08);

      const allValues = data.flatMap((row) => [row.logBFClosed, row.logBFLaplace, row.logBFELBO]);
      const yMin = Math.min(...allValues);
      const yMax = 0.0;
      const yPad = (yMax - yMin) * 0.06 || 1.0;
      const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + 1]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).tickFormat((v) => String(v)))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '12px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 38)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Polynomial degree d');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('log Bayes factor against argmax (nats)');

      // Zero line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .style('stroke', 'var(--color-text)')
        .style('stroke-opacity', 0.35)
        .style('stroke-width', 1);

      data.forEach((row) => {
        const x0 = xScale(row.d) ?? 0;
        const drawBar = (key: 'closed' | 'laplace' | 'elbo', value: number, fill: string) => {
          const bw = groupScale.bandwidth();
          const bx = x0 + (groupScale(key) ?? 0);
          const top = yScale(Math.max(value, 0));
          const bottom = yScale(Math.min(value, 0));
          g.append('rect')
            .attr('x', bx)
            .attr('y', top)
            .attr('width', bw)
            .attr('height', Math.max(1, bottom - top))
            .attr('fill', fill)
            .attr('fill-opacity', 0.85);
        };
        drawBar('closed', row.logBFClosed, BLUE);
        drawBar('laplace', row.logBFLaplace, GRAY);
        drawBar('elbo', row.logBFELBO, RED);
      });

      // Argmax winner highlight
      const winnerX = (xScale(argMax) ?? 0) + xScale.bandwidth() / 2;
      g.append('text')
        .attr('x', winnerX)
        .attr('y', yScale(0) - 8)
        .attr('text-anchor', 'middle')
        .style('fill', BLUE)
        .style('font-size', '11px')
        .style('font-weight', 600)
        .text('★');

      // Legend
      const legend = g.append('g').attr('transform', `translate(${w - 220},${4})`);
      [
        { color: BLUE, label: 'Closed-form log p' },
        { color: GRAY, label: 'Laplace' },
        { color: RED, label: 'Mean-field ELBO' },
      ].forEach((entry, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        row.append('rect').attr('width', 14).attr('height', 12).attr('fill', entry.color).attr('fill-opacity', 0.85);
        row
          .append('text')
          .attr('x', 20)
          .attr('y', 10)
          .style('fill', 'var(--color-text)')
          .style('font-size', '11px')
          .text(entry.label);
      });
    },
    [data, containerWidth, argMax],
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
              Polynomial Bayes-factor ranking — argmax: <strong>d = {argMax}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              All three estimators agree on rank-order (the §5 Proposition 1 claim) — magnitudes differ.
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 8, fontSize: 12, color: 'var(--color-text-secondary)' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            n
            <input type="range" min={20} max={200} step={5} value={n} onChange={(e) => setN(parseInt(e.target.value, 10))} style={{ width: 120 }} />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 28, color: 'var(--color-text)' }}>{n}</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            σ²
            <input type="range" min={0.005} max={0.5} step={0.005} value={sigma2} onChange={(e) => setSigma2(parseFloat(e.target.value))} style={{ width: 120 }} />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 40, color: 'var(--color-text)' }}>{sigma2.toFixed(3)}</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            τ²
            <input type="range" min={0.25} max={16} step={0.25} value={tau2} onChange={(e) => setTau2(parseFloat(e.target.value))} style={{ width: 120 }} />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 36, color: 'var(--color-text)' }}>{tau2.toFixed(2)}</span>
          </label>
        </div>
        <svg ref={svgRef} width={containerWidth || 720} height={PANEL_HEIGHT} role="img" aria-label="Bar chart of log Bayes factors across polynomial degrees for three estimators." />
      </div>
    </div>
  );
}

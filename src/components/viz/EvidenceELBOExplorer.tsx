import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  marginalLikelihoodGaussianRegression,
  meanFieldELBOGaussianRegression,
  polynomialDesignMatrix,
} from './shared/bayesian-ml';

// =============================================================================
// EvidenceELBOExplorer — §1 Bayesian Occam plot, interactive
// =============================================================================
// Three traces vs polynomial degree d=1..9 on a fixed dataset:
//   - closed-form log p(y | M_d)            (blue)
//   - Laplace approximation                  (gray, overlays exactly here)
//   - mean-field ELBO                        (red)
// User slider: τ² (prior coefficient variance).
//
// The dataset is the §1 spine: y values captured from the notebook
// (NumPy PCG64 seed=20260508) so the in-browser computation matches the
// printed reference figure 01_bayesian_occam.png.
// =============================================================================

const PANEL_HEIGHT = 380;
const MARGIN = { top: 22, right: 32, bottom: 48, left: 56 };
const DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const SIGMA2 = 0.0625;

const BLUE = '#1f4e79';
const GRAY = '#7f7f7f';
const RED = '#c0504d';

const X_GRID: number[] = (() => {
  const n = 30;
  const arr = new Array<number>(n);
  for (let i = 0; i < n; i++) arr[i] = -1.0 + (2.0 * i) / (n - 1);
  return arr;
})();

// y from notebook seed 20260508 (NumPy PCG64); see verify-vbms test fixture.
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

interface DataPoint {
  d: number;
  logP: number;
  laplace: number;
  elbo: number;
  bias: number;
}

function computeTraces(tau2: number): DataPoint[] {
  return DEGREES.map((d) => {
    const X = polynomialDesignMatrix(X_GRID, d);
    const logP = marginalLikelihoodGaussianRegression(X, Y_FIXED, SIGMA2, tau2);
    const elbo = meanFieldELBOGaussianRegression(X, Y_FIXED, SIGMA2, tau2);
    return { d, logP, laplace: logP, elbo, bias: logP - elbo };
  });
}

export default function EvidenceELBOExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [tau2, setTau2] = useState(4.0);

  const data = useMemo(() => computeTraces(tau2), [tau2]);
  const argMaxD = useMemo(() => {
    let bestD = data[0].d;
    let bestLogP = data[0].logP;
    for (const row of data) {
      if (row.logP > bestLogP) {
        bestLogP = row.logP;
        bestD = row.d;
      }
    }
    return bestD;
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

      const xScale = d3.scaleLinear().domain([0.6, 9.4]).range([0, w]);
      const allY = data.flatMap((row) => [row.logP, row.elbo]);
      const yMin = Math.min(...allY);
      const yMax = Math.max(...allY);
      const yPad = (yMax - yMin) * 0.08;
      const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).nice().range([h, 0]);

      // Winner highlight
      g.append('line')
        .attr('x1', xScale(argMaxD))
        .attr('x2', xScale(argMaxD))
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', BLUE)
        .style('stroke-opacity', 0.18)
        .style('stroke-width', 2)
        .style('stroke-dasharray', '4,3');

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).tickValues(DEGREES).tickFormat((v) => String(v)))
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
        .text('Polynomial degree d');
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Log-evidence (nats)');

      // KL gap shaded region between log p and ELBO
      const gapArea = d3
        .area<DataPoint>()
        .x((row) => xScale(row.d))
        .y0((row) => yScale(row.logP))
        .y1((row) => yScale(row.elbo))
        .curve(d3.curveLinear);
      g.append('path')
        .datum(data)
        .attr('d', gapArea)
        .attr('fill', RED)
        .attr('fill-opacity', 0.08);

      const lineLogP = d3.line<DataPoint>().x((row) => xScale(row.d)).y((row) => yScale(row.logP));
      const lineLaplace = d3
        .line<DataPoint>()
        .x((row) => xScale(row.d))
        .y((row) => yScale(row.laplace));
      const lineELBO = d3.line<DataPoint>().x((row) => xScale(row.d)).y((row) => yScale(row.elbo));

      // log p(y) — closed-form, blue line + circles
      g.append('path')
        .datum(data)
        .attr('d', lineLogP)
        .attr('fill', 'none')
        .attr('stroke', BLUE)
        .attr('stroke-width', 2.5);
      g.selectAll('circle.lp')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'lp')
        .attr('cx', (row) => xScale(row.d))
        .attr('cy', (row) => yScale(row.logP))
        .attr('r', 4.5)
        .attr('fill', BLUE);

      // Laplace — gray dashed squares (overlays log p exactly here)
      g.append('path')
        .datum(data)
        .attr('d', lineLaplace)
        .attr('fill', 'none')
        .attr('stroke', GRAY)
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '6,4');

      // ELBO — red triangles
      g.append('path')
        .datum(data)
        .attr('d', lineELBO)
        .attr('fill', 'none')
        .attr('stroke', RED)
        .attr('stroke-width', 2.5);
      g.selectAll('path.elbo-mark')
        .data(data)
        .enter()
        .append('path')
        .attr('class', 'elbo-mark')
        .attr(
          'd',
          d3
            .symbol<DataPoint>()
            .type(d3.symbolTriangle)
            .size(64) as unknown as (row: DataPoint) => string,
        )
        .attr(
          'transform',
          (row) => `translate(${xScale(row.d)},${yScale(row.elbo)})`,
        )
        .attr('fill', RED);

      // Legend
      const legendData = [
        { label: 'log p(y | M_d)', color: BLUE, dash: false },
        { label: 'Laplace (Gaussian-conjugate exact)', color: GRAY, dash: true },
        { label: 'Mean-field ELBO', color: RED, dash: false },
      ];
      const legend = g
        .append('g')
        .attr('transform', `translate(${w - 220},${10})`);
      legendData.forEach((entry, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 18})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 22)
          .attr('y1', 0)
          .attr('y2', 0)
          .style('stroke', entry.color)
          .style('stroke-width', 2.4)
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
    [data, containerWidth, argMaxD],
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
              Bayesian Occam on polynomial regression
            </div>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              n = 30, σ² = {SIGMA2}; argmax<sub>d</sub> log p(y | M<sub>d</sub>) = <strong>d = {argMaxD}</strong>
            </div>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--color-text-secondary)' }}>
            <span>τ² (prior variance)</span>
            <input
              type="range"
              min={0.25}
              max={16}
              step={0.25}
              value={tau2}
              onChange={(e) => setTau2(parseFloat(e.target.value))}
              style={{ width: 160 }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', minWidth: 40, color: 'var(--color-text)' }}>
              {tau2.toFixed(2)}
            </span>
          </label>
        </div>
        <svg ref={svgRef} width={containerWidth || 720} height={PANEL_HEIGHT} role="img" aria-label="Three traces of log-evidence vs polynomial degree showing Bayesian Occam's razor and the mean-field ELBO bias." />
        <div style={{ marginTop: 6, fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Shaded region between blue and red traces is the KL projection bias of §6.
          Increasing τ² weakens the prior, smooths Occam decay; decreasing τ² penalizes high-d models more sharply.
        </div>
      </div>
    </div>
  );
}

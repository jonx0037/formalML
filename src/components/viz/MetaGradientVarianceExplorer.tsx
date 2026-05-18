// =============================================================================
// MetaGradientVarianceExplorer.tsx
//
// §9.4 Empirical demonstration that meta-gradient variance scales as 1/B in
// the task-batch size B. We can't reproduce the notebook's 50K-parameter MAML
// experiment in-browser, so this viz runs the *1D-quadratic analog* of the
// experiment: per-task FOMAML meta-gradients at θ₀ = 0, with task-specific
// (μ_S, μ_Q, λ_S, λ_Q) drawn from a slider-controlled distribution. The 1/B
// scaling identity from Lemma 8.7 holds tightly on this controlled testbed.
//
// Sliders (all live, computation is closed-form O(n_task_draws)):
//   - α (inner step): [0.001, 0.1]
//   - σ_μ (task-mean spread): [0.5, 3]  controls per-task heterogeneity
//   - σ_λ (curvature spread): [0.0, 0.5]
//   - n_task_draws: [50, 500]  log
//
// Static fallback: /images/topics/meta-learning/17_meta_gradient_variance.png
// =============================================================================

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  mulberry32,
  gaussianPair,
  fomamlOuterGradient1D,
  META_PALETTE,
  META_SEED,
} from './shared/meta-learning';

const B_GRID = [1, 2, 5, 10, 25, 50];

export default function MetaGradientVarianceExplorer(): React.JSX.Element {
  const [alpha, setAlpha] = useState(0.01);
  const [sigmaMu, setSigmaMu] = useState(1.5);
  const [sigmaLambda, setSigmaLambda] = useState(0.2);
  const [logNDraws, setLogNDraws] = useState(2.3); // log10 → ~200

  const nDraws = Math.max(50, Math.min(500, Math.round(Math.pow(10, logNDraws))));

  const { empiricalVar, theoreticalVar } = useMemo(() => {
    const rng = mulberry32(META_SEED);
    const perTaskGrads: number[] = new Array(nDraws);
    for (let i = 0; i < nDraws; i++) {
      // Task: random support and query means with per-task curvature
      const muS = sigmaMu * gaussianPair(rng)[0];
      const muQ = sigmaMu * gaussianPair(rng)[0];
      const lamS = Math.max(0.1, 1 + sigmaLambda * gaussianPair(rng)[0]);
      const lamQ = Math.max(0.1, 1 + sigmaLambda * gaussianPair(rng)[0]);
      // FOMAML gradient at θ₀ = 0
      perTaskGrads[i] = fomamlOuterGradient1D(0, muS, muQ, lamS, lamQ, alpha);
    }
    const mean = perTaskGrads.reduce((a, b) => a + b, 0) / nDraws;
    const baseVar = perTaskGrads.reduce((a, b) => a + (b - mean) ** 2, 0) / nDraws;

    const empVar: number[] = new Array(B_GRID.length);
    for (let bi = 0; bi < B_GRID.length; bi++) {
      const B = B_GRID[bi];
      const nBatches = Math.floor(nDraws / B);
      if (nBatches === 0) {
        empVar[bi] = NaN;
        continue;
      }
      let sumSq = 0;
      for (let b = 0; b < nBatches; b++) {
        let mn = 0;
        for (let j = 0; j < B; j++) mn += perTaskGrads[b * B + j];
        mn /= B;
        sumSq += (mn - mean) ** 2;
      }
      empVar[bi] = sumSq / nBatches;
    }
    const theory = B_GRID.map((B) => empVar[0] / B);
    return { empiricalVar: empVar, theoreticalVar: theory };
  }, [alpha, sigmaMu, sigmaLambda, nDraws]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 700;
  const height = 320;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 18, right: 16, bottom: 44, left: 60 };
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([1, 100]).range([0, innerW]);
      const allValues = [...empiricalVar, ...theoreticalVar].filter((v) => isFinite(v) && v > 0);
      const yScale = d3
        .scaleLog()
        .domain([Math.min(...allValues) * 0.7, Math.max(...allValues) * 1.4])
        .range([innerH, 0]);

      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).tickValues([1, 2, 5, 10, 25, 50, 100]).tickFormat(d3.format('d'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '.2e')).selectAll('text').style('fill', 'var(--color-text)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 36).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('task batch size B');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -50).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('variance of meta-gradient estimator');

      const line = d3.line<number>().x((_, i) => xScale(B_GRID[i])).y((d) => yScale(d));

      g.append('path').datum(theoreticalVar).attr('d', line).style('fill', 'none').style('stroke', META_PALETTE[3]).style('stroke-width', 1.4).style('stroke-dasharray', '6 4');
      g.append('path').datum(empiricalVar).attr('d', line).style('fill', 'none').style('stroke', META_PALETTE[0]).style('stroke-width', 2);
      g.selectAll('.emp')
        .data(empiricalVar)
        .enter()
        .append('circle')
        .attr('cx', (_d, i) => xScale(B_GRID[i]))
        .attr('cy', (d) => yScale(d))
        .attr('r', 5)
        .style('fill', META_PALETTE[0]);

      const legendG = g.append('g').attr('transform', `translate(${innerW - 180}, 8)`);
      legendG.append('rect').attr('width', 14).attr('height', 3).attr('y', 5).style('fill', META_PALETTE[0]);
      legendG.append('text').attr('x', 18).attr('y', 8).style('fill', 'var(--color-text)').style('font-size', '11px').text('empirical variance');
      legendG.append('rect').attr('width', 14).attr('height', 3).attr('y', 23).style('fill', META_PALETTE[3]);
      legendG.append('text').attr('x', 18).attr('y', 26).style('fill', 'var(--color-text)').style('font-size', '11px').text('theoretical ∝ 1/B');
    },
    [empiricalVar, theoreticalVar, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 10, fontSize: 11 }}>
        <label>
          α: {alpha.toFixed(3)}
          <input type="range" min={0.001} max={0.1} step={0.001} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} style={{ width: '100%' }} aria-label="alpha" />
        </label>
        <label>
          σ_μ (task-mean spread): {sigmaMu.toFixed(2)}
          <input type="range" min={0.5} max={3} step={0.1} value={sigmaMu} onChange={(e) => setSigmaMu(Number(e.target.value))} style={{ width: '100%' }} aria-label="sigma_mu" />
        </label>
        <label>
          σ_λ (curvature spread): {sigmaLambda.toFixed(2)}
          <input type="range" min={0} max={0.5} step={0.02} value={sigmaLambda} onChange={(e) => setSigmaLambda(Number(e.target.value))} style={{ width: '100%' }} aria-label="sigma_lambda" />
        </label>
        <label>
          n_task_draws: {nDraws}
          <input type="range" min={1.7} max={2.7} step={0.05} value={logNDraws} onChange={(e) => setLogNDraws(Number(e.target.value))} style={{ width: '100%' }} aria-label="n draws" />
        </label>
      </div>
      <svg ref={svgRef} width={width} height={height} role="img" aria-label="meta-gradient variance vs B" />
      <p style={{ fontSize: 10.5, color: 'var(--color-text-secondary)' }}>
        In-browser 1D-quadratic analog of the §9.4 experiment: per-task FOMAML meta-gradients at θ₀=0 with random support and query targets. The 1/B scaling identity is rigid because it follows from independence across task draws.
      </p>
    </div>
  );
}

import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  catoniDecomposition,
  empiricalRiskOnGrid,
  expectedUnderQ,
  gibbsOnGrid,
  klDiscrete,
  mulberry32,
  sampleNormalThresholdProblem,
  thresholdGrid,
} from './shared/pac-bayes-bounds';

// =============================================================================
// GibbsPosteriorVisualizer — §6.5b
// Catoni bound decomposition at Q = Q*_λ as a function of λ.
//   total(λ) = E_{Q*_λ}[R̂_S] + λ/(8n) + (KL(Q*_λ ∥ P) + log(1/δ)) / λ
// Components plotted on log-log axes; empirical λ* (the argmin) is marked.
// =============================================================================

const N = 200;
const ETA = 0.05;
const MU_STAR = 0.3;
const DELTA = 0.05;
const SEED = 20260511;
const HEIGHT = 360;
const LAMBDA_GRID_COUNT = 200;

export default function GibbsPosteriorVisualizer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const data = useMemo(() => {
    const rng = mulberry32(SEED >>> 0);
    const { X, Y } = sampleNormalThresholdProblem(N, ETA, MU_STAR, rng);
    const grid = thresholdGrid();
    const risks = empiricalRiskOnGrid(X, Y, grid);
    const P = new Float64Array(grid.length).fill(1 / grid.length);

    const lambdas = new Float64Array(LAMBDA_GRID_COUNT);
    const lambdaLo = Math.log10(0.5);
    const lambdaHi = Math.log10(1e4);
    for (let i = 0; i < LAMBDA_GRID_COUNT; i++) {
      lambdas[i] = Math.pow(10, lambdaLo + (i / (LAMBDA_GRID_COUNT - 1)) * (lambdaHi - lambdaLo));
    }
    const empRisk = new Float64Array(LAMBDA_GRID_COUNT);
    const lin = new Float64Array(LAMBDA_GRID_COUNT);
    const klOverLambda = new Float64Array(LAMBDA_GRID_COUNT);
    const total = new Float64Array(LAMBDA_GRID_COUNT);
    for (let i = 0; i < LAMBDA_GRID_COUNT; i++) {
      const lam = lambdas[i];
      const Q = gibbsOnGrid(P, risks, lam);
      const er = expectedUnderQ(Q, risks);
      const kl = klDiscrete(Q, P);
      const d = catoniDecomposition(er, kl, N, DELTA, lam);
      empRisk[i] = d.empiricalRisk;
      lin[i] = d.linearizationPenalty;
      klOverLambda[i] = d.klOverLambda;
      total[i] = d.total;
    }
    // Empirical λ*
    let argmin = 0;
    for (let i = 1; i < LAMBDA_GRID_COUNT; i++) if (total[i] < total[argmin]) argmin = i;
    return {
      lambdas,
      empRisk,
      lin,
      klOverLambda,
      total,
      lambdaStar: lambdas[argmin],
      boundAtStar: total[argmin],
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        margin: '1.5rem 0',
        padding: '1rem',
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: '0.5rem',
      }}
    >
      <div style={{ marginBottom: '0.5rem', fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
        Catoni bound decomposition at Q = Q*_λ, n = {N}, δ = {DELTA} &nbsp;|&nbsp;
        empirical λ* = <strong>{data.lambdaStar.toFixed(1)}</strong>, bound at λ* = <strong>{data.boundAtStar.toFixed(3)}</strong>
      </div>
      <Panel data={data} width={containerWidth || 800} />
    </div>
  );
}

type DataT = {
  lambdas: Float64Array;
  empRisk: Float64Array;
  lin: Float64Array;
  klOverLambda: Float64Array;
  total: Float64Array;
  lambdaStar: number;
  boundAtStar: number;
};

function Panel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 130, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([data.lambdas[0], data.lambdas[data.lambdas.length - 1]]).range([0, w]);
      const yMin = Math.max(1e-3, Math.min(
        d3.min(data.empRisk) ?? 1e-3,
        d3.min(data.lin) ?? 1e-3,
        d3.min(data.klOverLambda) ?? 1e-3,
      ));
      const yMax = Math.max(
        d3.max(data.total) ?? 1,
        d3.max(data.klOverLambda) ?? 1,
        d3.max(data.lin) ?? 1,
      );
      const y = d3.scaleLog().domain([yMin, yMax * 1.2]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(
        d3.axisBottom(x).ticks(6, '~s'),
      );
      g.append('g').call(d3.axisLeft(y).ticks(6, '~s'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('temperature λ (log)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('bound component (log)');

      const lineGen = (arr: Float64Array) => d3.line<number>()
        .x((_, i) => x(data.lambdas[i]))
        .y((v) => y(Math.max(yMin, v)))(Array.from(arr));

      const series = [
        { name: 'E_{Q*_λ}[R̂]', color: '#D97706', arr: data.empRisk },
        { name: 'λ/(8n)', color: '#534AB7', arr: data.lin },
        { name: '(KL + log(1/δ))/λ', color: '#0F6E56', arr: data.klOverLambda },
        { name: 'total', color: 'var(--color-text)', arr: data.total },
      ];
      series.forEach((s) => {
        g.append('path').attr('d', lineGen(s.arr))
          .style('fill', 'none').style('stroke', s.color)
          .style('stroke-width', s.name === 'total' ? 2.6 : 1.6);
      });

      // λ* marker
      g.append('line').attr('x1', x(data.lambdaStar)).attr('x2', x(data.lambdaStar))
        .attr('y1', 0).attr('y2', h)
        .style('stroke', 'var(--color-accent)').style('stroke-dasharray', '4,3');
      g.append('circle')
        .attr('cx', x(data.lambdaStar)).attr('cy', y(data.boundAtStar)).attr('r', 4)
        .style('fill', 'var(--color-accent)');

      // Legend
      const legend = g.append('g').attr('transform', `translate(${w + 12},14)`);
      series.forEach((s, idx) => {
        legend.append('line').attr('x1', 0).attr('x2', 14)
          .attr('y1', idx * 14).attr('y2', idx * 14)
          .style('stroke', s.color).style('stroke-width', s.name === 'total' ? 2.6 : 1.6);
        legend.append('text').attr('x', 18).attr('y', idx * 14 + 3)
          .style('font-size', '10px').style('fill', 'var(--color-text-secondary)')
          .text(s.name);
      });

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Catoni bound decomposition vs λ (U-shape minimum at λ*)');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

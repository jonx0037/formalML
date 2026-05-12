import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gibbsOnGrid,
  klDiscrete,
  mulberry32,
} from './shared/pac-bayes-bounds';

// =============================================================================
// BayesianGibbsBridge — §9.5
// Bernoulli toy where the Bayesian–PAC-Bayes correspondence is exact.
// Data: Y_i ~ Bernoulli(θ*=0.3), n=200, θ-grid {0.005, 0.015, ..., 0.995}.
// Per-sample loss = −log p(y_i | θ).
// Panel A: Q*_λ at six λ values + Bayesian posterior P(θ | D) on the grid.
// Panel B: KL(Q*_λ ‖ P(θ | D)) vs λ; minimum at λ = n.
// =============================================================================

const N = 200;
const THETA_STAR = 0.3;
const SEED = 20260511;
const HEIGHT = 340;
const SM_BREAKPOINT = 640;

export default function BayesianGibbsBridge() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const data = useMemo(() => {
    // Sample
    const rng = mulberry32(SEED >>> 0);
    let s = 0;
    for (let i = 0; i < N; i++) if (rng() < THETA_STAR) s++;
    // θ-grid: 100 evenly-spaced cell centers in (0, 1)
    const thetaGrid = new Float64Array(100);
    for (let i = 0; i < 100; i++) thetaGrid[i] = (i + 0.5) / 100;
    // Per-θ empirical NLL: R̂(θ) = −(1/n)·[s·log θ + (n−s)·log(1−θ)]
    const empNLL = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
      const theta = thetaGrid[i];
      empNLL[i] = -((s * Math.log(theta) + (N - s) * Math.log(1 - theta))) / N;
    }
    const P = new Float64Array(100).fill(1 / 100);
    // Bayesian posterior on the grid: P(θ) · likelihood(θ; data), renormalized.
    // log L(θ; D) = s·log θ + (n−s)·log(1−θ) = −n·R̂(θ)
    // Q*_n(θ) ∝ P(θ) · exp(−n·R̂(θ)) — exactly the Bayesian posterior by construction.
    const Qbayes = gibbsOnGrid(P, empNLL, N);

    const referenceLambdas = [0, N / 4, N / 2, N, 2 * N, 4 * N];
    const referenceCurves = referenceLambdas.map((lam) => ({ lam, Q: gibbsOnGrid(P, empNLL, lam) }));

    // KL(Q*_λ ‖ Q_bayes) sweep
    const lambdaCount = 120;
    const lambdaLo = Math.log10(1);
    const lambdaHi = Math.log10(4 * N);
    const lambdas = new Float64Array(lambdaCount);
    const klSeries = new Float64Array(lambdaCount);
    for (let i = 0; i < lambdaCount; i++) {
      lambdas[i] = Math.pow(10, lambdaLo + (i / (lambdaCount - 1)) * (lambdaHi - lambdaLo));
      const Q = gibbsOnGrid(P, empNLL, lambdas[i]);
      klSeries[i] = klDiscrete(Q, Qbayes);
    }

    return {
      thetaGrid,
      empNLL,
      P,
      Qbayes,
      referenceCurves,
      lambdas,
      klSeries,
      observedSuccesses: s,
    };
  }, []);

  const isMobile = (containerWidth || 800) < SM_BREAKPOINT;
  const panelWidth = isMobile ? (containerWidth || 800) : Math.max(280, (containerWidth || 800) / 2 - 8);

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
        Y_i ~ Bernoulli(θ* = {THETA_STAR}), n = {N}, observed {data.observedSuccesses} successes &nbsp;|&nbsp;
        at λ = n, Q*_λ ≡ P(θ | D) (Bayesian posterior) to machine precision
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '0.5rem' }}>
        <CorrespondencePanel data={data} width={panelWidth} />
        <KLToBayesPanel data={data} width={panelWidth} />
      </div>
    </div>
  );
}

type DataT = {
  thetaGrid: Float64Array;
  empNLL: Float64Array;
  P: Float64Array;
  Qbayes: Float64Array;
  referenceCurves: { lam: number; Q: Float64Array }[];
  lambdas: Float64Array;
  klSeries: Float64Array;
  observedSuccesses: number;
};

function CorrespondencePanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 100, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const yMax = Math.max(
        d3.max(data.Qbayes) ?? 0.2,
        ...data.referenceCurves.map((r) => d3.max(r.Q) ?? 0.2),
      );
      const y = d3.scaleLinear().domain([0, yMax * 1.05]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3f')));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('θ');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('Q*_λ(θ) or P(θ | D)');

      // True θ*
      g.append('line').attr('x1', x(THETA_STAR)).attr('x2', x(THETA_STAR))
        .attr('y1', 0).attr('y2', h)
        .style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '3,3');

      const palette = d3.schemeBlues[7].slice(1);
      const lineGen = d3.line<number>().x((_, i) => x(data.thetaGrid[i])).y((v) => y(v));

      // Reference Gibbs curves (faded)
      data.referenceCurves.forEach((r, idx) => {
        g.append('path').datum(Array.from(r.Q))
          .style('fill', 'none').style('stroke', palette[idx]).style('stroke-width', 1.4)
          .style('opacity', 0.75)
          .attr('d', lineGen);
      });

      // Bayesian posterior (solid accent, drawn on top)
      g.append('path').datum(Array.from(data.Qbayes))
        .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2.6).style('stroke-dasharray', '6,2')
        .attr('d', lineGen);

      const legend = g.append('g').attr('transform', `translate(${w + 6},14)`);
      data.referenceCurves.forEach((r, idx) => {
        legend.append('line').attr('x1', 0).attr('x2', 14)
          .attr('y1', idx * 14).attr('y2', idx * 14)
          .style('stroke', palette[idx]).style('stroke-width', 1.4);
        legend.append('text').attr('x', 18).attr('y', idx * 14 + 3)
          .style('font-size', '10px').style('fill', 'var(--color-text-secondary)')
          .text(`λ = ${r.lam === 0 ? '0' : r.lam.toFixed(0)}`);
      });
      const cur = data.referenceCurves.length * 14;
      legend.append('line').attr('x1', 0).attr('x2', 14)
        .attr('y1', cur).attr('y2', cur)
        .style('stroke', 'var(--color-accent)').style('stroke-width', 2.6).style('stroke-dasharray', '6,2');
      legend.append('text').attr('x', 18).attr('y', cur + 3)
        .style('font-size', '10px').style('fill', 'var(--color-accent)')
        .text('Bayes posterior');

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Panel A — Q*_λ across λ, Bayes posterior overlaid');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

function KLToBayesPanel({ data, width }: { data: DataT; width: number }) {
  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 26, right: 20, bottom: 44, left: 56 };
      const w = width - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([data.lambdas[0], data.lambdas[data.lambdas.length - 1]]).range([0, w]);
      const yMin = Math.max(1e-14, Math.min(...Array.from(data.klSeries).filter((v) => v > 0)));
      const yMax = Math.max(...Array.from(data.klSeries));
      const y = d3.scaleLog().domain([yMin, yMax * 2]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(6, '~s'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('temperature λ (log)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -44).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('KL(Q*_λ ‖ P(θ|D)) (log)');

      const lineGen = d3.line<number>()
        .x((_, i) => x(data.lambdas[i]))
        .y((v) => y(Math.max(yMin, v)));
      g.append('path').datum(Array.from(data.klSeries))
        .style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 2.2)
        .attr('d', lineGen);

      // λ = n marker
      g.append('line').attr('x1', x(N)).attr('x2', x(N))
        .attr('y1', 0).attr('y2', h)
        .style('stroke', 'var(--color-accent)').style('stroke-dasharray', '4,3');
      g.append('text').attr('x', x(N) + 4).attr('y', 12)
        .style('font-size', '10px').style('fill', 'var(--color-accent)')
        .text(`λ = n = ${N}`);

      g.append('text').attr('x', 0).attr('y', -8)
        .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--color-text)')
        .text('Panel B — KL to Bayes posterior, minimum at λ = n');
    },
    [data, width],
  );
  return <svg ref={ref} width={width} height={HEIGHT} style={{ overflow: 'visible' }} />;
}

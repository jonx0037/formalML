// =============================================================================
// ResamplingSchemeComparator.tsx
//
// §4.4 of sequential-monte-carlo. Three-panel comparison of the four
// resampling schemes — multinomial, residual, stratified, systematic —
// on a controllable weight cloud. The skewness slider warps a softmax-
// over-iid-scores distribution; offspring counts and Monte-Carlo
// estimator variance update live for all four schemes.
//
// Panels:
//   (A) Sorted normalized weights on log-log, with ESS marker. Shows how
//       skewed the cloud is at the user's chosen β.
//   (B) Per-scheme offspring counts on sorted-by-weight x-axis. Demonstrates
//       Proposition 2 (unbiased offspring) and Theorem 2 (variance ordering).
//   (C) Estimator-variance bars across the four schemes, one bar per scheme
//       over M = 800 replicates of a bounded test function. The systematic
//       bar should be the shortest (Theorem 2: systematic ≤ stratified ≤
//       residual ≤ multinomial).
//
// Interactive control:
//   - β skew slider in [0.5, 5.0], default 2.5. Larger β → more degenerate
//     weight distribution → larger gap between systematic and multinomial.
//
// Pure TS computation via shared/sequential-monte-carlo.ts. N = 200, M = 800.
// Cheap enough that no commit-on-release is needed; the entire compute
// is well under 60 ms per slider tick.
//
// Static fallback: /images/topics/sequential-monte-carlo/04_resampling_variance.png
// =============================================================================

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  mulberry32,
  multinomialResample,
  residualResample,
  stratifiedResample,
  systematicResample,
  normalizeLogW,
  paletteSMC,
  SMC_SEED,
  type ResamplingScheme,
} from './shared/sequential-monte-carlo';

const N = 200;
const M = 800; // MC replicates
const SCHEMES: ResamplingScheme[] = ['multinomial', 'residual', 'stratified', 'systematic'];
const SCHEME_LABELS: Record<ResamplingScheme, string> = {
  multinomial: 'multinomial',
  residual: 'residual',
  stratified: 'stratified',
  systematic: 'systematic',
};
const SCHEME_COLORS: Record<ResamplingScheme, string> = {
  multinomial: paletteSMC.proposal,
  residual: paletteSMC.accent,
  stratified: paletteSMC.cloud,
  systematic: paletteSMC.target,
};

interface ComputeResult {
  wNorm: number[]; // sorted descending
  wOriginal: number[]; // original order
  ess: number;
  // per-scheme: average offspring counts vs particle rank (sorted by weight desc)
  countsByScheme: Record<ResamplingScheme, number[]>;
  // estimator variance per scheme on a bounded test function
  variances: Record<ResamplingScheme, number>;
  // expected count = N * w_sorted
  expectedCount: number[];
  trueMean: number; // analytic mean under the weighted cloud
}

function compute(beta: number): ComputeResult {
  const rng = mulberry32(SMC_SEED + 17);
  // Generate N synthetic iid standard normal scores via Box-Muller pairs.
  const scores = new Array<number>(N);
  for (let i = 0; i < N; i += 2) {
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    scores[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < N) scores[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  const logW = scores.map((z) => beta * z);
  const wOriginal = normalizeLogW(logW).map(Math.exp);
  const ess = 1 / wOriginal.reduce((s, v) => s + v * v, 0);

  // bounded test function on particle index (the same function for all schemes)
  const f = scores.map((z) => Math.tanh(z));
  const trueMean = wOriginal.reduce((s, v, i) => s + v * f[i], 0);

  // sort indices by weight descending (for plotting)
  const sortIdx = wOriginal.map((_, i) => i).sort((a, b) => wOriginal[b] - wOriginal[a]);
  const wSorted = sortIdx.map((i) => wOriginal[i]);
  const expectedCount = wSorted.map((w) => N * w);

  // Monte-Carlo: for each scheme, accumulate offspring counts (sorted by
  // weight) and the estimator variance. We use a separate RNG per scheme
  // so the M replicates are independent.
  const countsByScheme: Record<ResamplingScheme, number[]> = {
    multinomial: new Array(N).fill(0),
    residual: new Array(N).fill(0),
    stratified: new Array(N).fill(0),
    systematic: new Array(N).fill(0),
  };
  const variances: Record<ResamplingScheme, number> = {
    multinomial: 0,
    residual: 0,
    stratified: 0,
    systematic: 0,
  };

  for (const scheme of SCHEMES) {
    const rngS = mulberry32(SMC_SEED + 100 + scheme.length); // distinct seed per scheme
    const estimates = new Array<number>(M);
    const accCounts = new Array<number>(N).fill(0);
    for (let m = 0; m < M; m++) {
      const idx =
        scheme === 'multinomial'
          ? multinomialResample(wOriginal, rngS, N)
          : scheme === 'residual'
            ? residualResample(wOriginal, rngS, N)
            : scheme === 'stratified'
              ? stratifiedResample(wOriginal, rngS, N)
              : systematicResample(wOriginal, rngS, N);
      let s = 0;
      for (let i = 0; i < N; i++) {
        accCounts[idx[i]]++;
        s += f[idx[i]];
      }
      estimates[m] = s / N;
    }
    // average per-particle count, then resort by descending weight
    const avgCount = accCounts.map((c) => c / M);
    countsByScheme[scheme] = sortIdx.map((i) => avgCount[i]);
    // variance of estimator (sample variance, ddof=1)
    const mean = estimates.reduce((s, v) => s + v, 0) / M;
    let v = 0;
    for (let m = 0; m < M; m++) {
      const d = estimates[m] - mean;
      v += d * d;
    }
    variances[scheme] = v / (M - 1);
  }

  return { wNorm: wSorted, wOriginal, ess, countsByScheme, variances, expectedCount, trueMean };
}

export default function ResamplingSchemeComparator() {
  const [beta, setBeta] = useState(2.5);

  const result = useMemo(() => compute(beta), [beta]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 760;
  const panelGap = 16;
  const panelH = 200;
  const panelW = Math.max(160, (width - panelGap * 2) / 3);
  const margin = { top: 18, right: 12, bottom: 32, left: 44 };
  const innerW = panelW - margin.left - margin.right;
  const innerH = panelH - margin.top - margin.bottom;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;

      // ---- Panel A: sorted weights on log-log scale, ESS marker
      const gA = svg.append('g').attr('transform', `translate(0, 0)`);
      const innerA = gA
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xA = d3.scaleLog().domain([1, N]).range([0, innerW]);
      const minW = Math.max(1e-8, d3.min(result.wNorm) || 1e-8);
      const maxW = d3.max(result.wNorm) || 1;
      const yA = d3.scaleLog().domain([minW, maxW]).range([innerH, 0]);
      innerA
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xA).ticks(3, '~s'))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('g')
        .call(d3.axisLeft(yA).ticks(3, '~e'))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 26)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('particle rank (largest weight first)');
      innerA
        .append('text')
        .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('normalized weight');
      // weight line
      const lineA = d3
        .line<number>()
        .x((_, i) => xA(i + 1))
        .y((w) => yA(Math.max(w, 1e-8)));
      innerA
        .append('path')
        .datum(result.wNorm)
        .attr('d', lineA)
        .style('fill', 'none')
        .style('stroke', paletteSMC.cloud)
        .style('stroke-width', 1.6);
      // ESS marker
      const essX = xA(Math.max(1, Math.min(N, result.ess)));
      innerA
        .append('line')
        .attr('x1', essX)
        .attr('x2', essX)
        .attr('y1', 0)
        .attr('y2', innerH)
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 3');
      innerA
        .append('text')
        .attr('x', essX + 4)
        .attr('y', 12)
        .style('fill', paletteSMC.target)
        .style('font-size', '10px')
        .text(`ESS = ${result.ess.toFixed(1)}`);
      gA.append('text')
        .attr('x', margin.left)
        .attr('y', 12)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(A) sorted normalized weights');

      // ---- Panel B: offspring counts per scheme
      const gB = svg.append('g').attr('transform', `translate(${panelW + panelGap}, 0)`);
      const innerB = gB
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xB = d3
        .scaleLinear()
        .domain([0, Math.min(60, N)])
        .range([0, innerW]);
      const maxCount = Math.max(
        ...SCHEMES.flatMap((s) => result.countsByScheme[s].slice(0, 60)),
        ...result.expectedCount.slice(0, 60),
      );
      const yB = d3.scaleLinear().domain([0, maxCount * 1.05]).range([innerH, 0]);
      innerB
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xB).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('g')
        .call(d3.axisLeft(yB).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 26)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('particle rank (top 60 by weight)');
      innerB
        .append('text')
        .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('mean offspring count');
      // expected E[N^(i)] = N w_i  (dashed reference)
      const lineExp = d3
        .line<number>()
        .x((_, i) => xB(i))
        .y((c) => yB(c));
      innerB
        .append('path')
        .datum(result.expectedCount.slice(0, 60))
        .attr('d', lineExp)
        .style('fill', 'none')
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 3')
        .style('opacity', 0.55);
      // each scheme line
      for (const scheme of SCHEMES) {
        const data = result.countsByScheme[scheme].slice(0, 60);
        innerB
          .append('path')
          .datum(data)
          .attr('d', lineExp)
          .style('fill', 'none')
          .style('stroke', SCHEME_COLORS[scheme])
          .style('stroke-width', 1.3)
          .style('opacity', 0.85);
      }
      gB.append('text')
        .attr('x', margin.left)
        .attr('y', 12)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(B) mean offspring counts');

      // ---- Panel C: variance bars
      const gC = svg.append('g').attr('transform', `translate(${(panelW + panelGap) * 2}, 0)`);
      const innerC = gC
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xC = d3
        .scaleBand<ResamplingScheme>()
        .domain(SCHEMES)
        .range([0, innerW])
        .padding(0.2);
      const varVals = SCHEMES.map((s) => result.variances[s]);
      const maxVar = Math.max(...varVals);
      const yC = d3.scaleLinear().domain([0, maxVar * 1.15]).range([innerH, 0]);
      innerC
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xC).tickFormat((s) => SCHEME_LABELS[s].slice(0, 4)))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '9px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerC
        .append('g')
        .call(d3.axisLeft(yC).ticks(4, '.0e'))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerC
        .append('text')
        .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text(`Var of mean-${'f̄'} (M = ${M})`);
      for (const scheme of SCHEMES) {
        innerC
          .append('rect')
          .attr('x', xC(scheme)!)
          .attr('y', yC(result.variances[scheme]))
          .attr('width', xC.bandwidth())
          .attr('height', innerH - yC(result.variances[scheme]))
          .style('fill', SCHEME_COLORS[scheme])
          .style('opacity', 0.8);
      }
      gC.append('text')
        .attr('x', margin.left)
        .attr('y', 12)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(C) estimator variance');
    },
    [
      width,
      panelW,
      innerW,
      innerH,
      result.wNorm,
      result.ess,
      result.expectedCount,
      result.variances,
      result.countsByScheme,
    ],
  );

  // shared legend across panels B + C
  const legend = (
    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
      {SCHEMES.map((s) => (
        <span key={s} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: SCHEME_COLORS[s] }}
          />
          <span className="font-mono text-[10px]" style={{ color: 'var(--color-text)' }}>
            {SCHEME_LABELS[s]}
          </span>
        </span>
      ))}
      <span className="inline-flex items-center gap-1.5">
        <span
          className="inline-block h-0.5 w-4 border-t"
          style={{ borderTop: '1px dashed var(--color-text)' }}
        />
        <span className="font-mono text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>
          E[N⁽ⁱ⁾] = N w
        </span>
      </span>
    </div>
  );

  // variance scoreboard ratios
  const baselineVar = result.variances.multinomial;
  const ratios = Object.fromEntries(
    SCHEMES.map((s) => [s, result.variances[s] / Math.max(baselineVar, 1e-30)] as const),
  ) as Record<ResamplingScheme, number>;

  return (
    <div ref={containerRef} className="w-full">
      <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-4">
        <label className="text-xs">
          <span style={{ color: 'var(--color-text)' }}>
            <span className="font-mono">β</span> (log-weight scale) = <span className="font-mono">{beta.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.1}
            value={beta}
            onChange={(e) => setBeta(parseFloat(e.target.value))}
            className="ml-2 w-48 align-middle"
            aria-label="β skewness of weight distribution"
          />
        </label>
        <span className="text-[11px]" style={{ color: 'var(--color-text-secondary)' }}>
          larger β → more degenerate weights → wider variance gap
        </span>
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={panelH}
        viewBox={`0 0 ${width} ${panelH}`}
        role="img"
        aria-label="Three-panel comparison of multinomial, residual, stratified, and systematic resampling on a controllable weight cloud."
      />
      {legend}
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-4">
        {SCHEMES.map((s) => (
          <div key={s} className="font-mono" style={{ color: 'var(--color-text)' }}>
            <span style={{ color: SCHEME_COLORS[s] }}>■</span>{' '}
            {SCHEME_LABELS[s]}:{' '}
            <span>Var = {result.variances[s].toExponential(2)}</span>{' '}
            <span style={{ color: 'var(--color-text-secondary)' }}>
              ({(ratios[s] * 100).toFixed(0)}% of mult.)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

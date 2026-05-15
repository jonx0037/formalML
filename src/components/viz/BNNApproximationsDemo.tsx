import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bayesPolyPosterior,
  bayesPolyPredict,
  ensemblePredict,
  fTrue,
  interpAleatoricCenters,
  linspace,
  mcDropoutPredict,
  mulberry32,
  sigmaTrue,
  toMlpCoefs,
  type PayloadMember,
} from './shared/uncertainty-quantification';

// =============================================================================
// BNNApproximationsDemo — §7
//
// Three method modes — Laplace (closed form), MC-dropout (over the first
// ensemble member's coefs, with adjustable dropout rate p), deep ensemble
// (full M=50 members). Consumes pre-trained MLP coefficients via fetch to
// /sample-data/uncertainty-quantification/ensemble.json (dual-located per CLAUDE.md).
// Static fallback: public/images/topics/uncertainty-quantification/fig_07_three_way.png
// =============================================================================

const HEIGHT = 460;
const DEGREE = 7;
const X_GRID = linspace(-3.2, 3.2, 200);

type EnsemblePayload = {
  X: number[];
  y: number[];
  members: PayloadMember[];
  aleatoric: { centers: number[]; vals: number[] };
};

export default function BNNApproximationsDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<EnsemblePayload | null>(null);
  const [method, setMethod] = useState<'laplace' | 'mcdropout' | 'ensemble'>('ensemble');
  const [pDropDisplay, setPDropDisplay] = useState(0.3);
  const [pDropCommitted, setPDropCommitted] = useState(0.3);
  const [mDisplay, setMDisplay] = useState(20);
  const [mCommitted, setMCommitted] = useState(20);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/uncertainty-quantification/ensemble.json')
      .then((r) => r.json())
      .then((data) => { if (!cancelled) setPayload(data); })
      .catch((err) => console.error('[BNNApproximationsDemo] fetch failed:', err));
    return () => { cancelled = true; };
  }, []);

  const result = useMemo(() => {
    if (!payload) return null;
    const aleatoricVar = interpAleatoricCenters(
      payload.aleatoric.centers, payload.aleatoric.vals, X_GRID);

    if (method === 'laplace') {
      const post = bayesPolyPosterior(payload.X, payload.y, DEGREE,
        payload.X.map(sigmaTrue));
      const pred = bayesPolyPredict(X_GRID, post, X_GRID.map(sigmaTrue));
      return {
        mean: pred.fMean,
        totalSd: pred.totalVar.map(Math.sqrt),
        epiSd: pred.epistemicVar.map(Math.sqrt),
      };
    }

    if (method === 'mcdropout') {
      const mlp = toMlpCoefs(payload.members[0]);
      const rng = mulberry32(20260514);
      const T = 100;
      const pred = mcDropoutPredict(X_GRID, mlp, pDropCommitted, T, rng);
      const epiSd = pred.variance.map(Math.sqrt);
      const totalSd = pred.variance.map((v, i) => Math.sqrt(v + aleatoricVar[i]));
      return { mean: pred.mean, totalSd, epiSd };
    }

    // Deep ensemble.
    const members = payload.members.slice(0, mCommitted).map((m) => toMlpCoefs(m));
    const pred = ensemblePredict(X_GRID, members);
    const epiSd = pred.variance.map(Math.sqrt);
    const totalSd = pred.variance.map((v, i) => Math.sqrt(v + aleatoricVar[i]));
    return { mean: pred.mean, totalSd, epiSd };
  }, [payload, method, pDropCommitted, mCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 24, bottom: 48, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0 || !payload || !result) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-3.3, 3.3]).range([0, W]);
      const yScale = d3.scaleLinear().domain([-3, 3]).range([H, 0]).nice();

      const colorByMethod = {
        laplace: '#8b5cf6',
        mcdropout: '#0ea5e9',
        ensemble: '#f59e0b',
      } as const;
      const color = colorByMethod[method];

      // Total ±2σ band.
      const bandTotal = d3.area<number>()
        .x((_, i) => xScale(X_GRID[i]))
        .y0((_, i) => yScale(result.mean[i] - 2 * result.totalSd[i]))
        .y1((_, i) => yScale(result.mean[i] + 2 * result.totalSd[i]));
      g.append('path').datum(X_GRID).attr('d', bandTotal)
        .style('fill', color).style('opacity', 0.16);

      // Epistemic-only band (darker).
      const bandEpi = d3.area<number>()
        .x((_, i) => xScale(X_GRID[i]))
        .y0((_, i) => yScale(result.mean[i] - 2 * result.epiSd[i]))
        .y1((_, i) => yScale(result.mean[i] + 2 * result.epiSd[i]));
      g.append('path').datum(X_GRID).attr('d', bandEpi)
        .style('fill', color).style('opacity', 0.38);

      const lineGen = d3.line<number>()
        .x((_, i) => xScale(X_GRID[i])).y((d) => yScale(d));
      g.append('path').datum(X_GRID.map(fTrue)).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#94a3b8').attr('stroke-dasharray', '5 3').attr('stroke-width', 1.4);
      g.append('path').datum(result.mean).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', color).attr('stroke-width', 2);

      g.selectAll('.pt').data(payload.X).enter().append('circle').attr('class', 'pt')
        .attr('cx', (d) => xScale(d as number))
        .attr('cy', (_, i) => yScale(payload.y[i]))
        .attr('r', 1.8).style('fill', '#1f2937').style('opacity', 0.45);

      g.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      const titles = {
        laplace: 'Laplace approximation (closed form, exact for linear-in-θ)',
        mcdropout: `MC-dropout (p = ${pDropCommitted.toFixed(2)}, T = 100)`,
        ensemble: `Deep ensemble (M = ${mCommitted})`,
      } as const;
      g.append('text').attr('x', W / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)').text(titles[method]);
    },
    [containerWidth, payload, result, method, pDropCommitted, mCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      {payload ? (
        <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
          aria-label="Three Bayesian-UQ approximations on the heteroscedastic toy: Laplace, MC-dropout, and deep ensemble." />
      ) : (
        <div style={{ height: HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--color-text-secondary)', fontFamily: 'var(--font-sans)' }}>
          Loading pre-trained ensemble (~340 KB) …
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>method</span>
          <select value={method} onChange={(e) => setMethod(e.target.value as typeof method)}
            aria-label="UQ approximation method">
            <option value="laplace">Laplace</option>
            <option value="mcdropout">MC-dropout</option>
            <option value="ensemble">Deep ensemble</option>
          </select>
        </label>
        {method === 'mcdropout' && (
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
            <span>dropout p: <strong>{pDropDisplay.toFixed(2)}</strong></span>
            <input type="range" min={0.05} max={0.6} step={0.05} value={pDropDisplay}
              onChange={(e) => setPDropDisplay(parseFloat(e.target.value))}
              onMouseUp={() => setPDropCommitted(pDropDisplay)} onTouchEnd={() => setPDropCommitted(pDropDisplay)}
              onKeyUp={() => setPDropCommitted(pDropDisplay)} aria-label="Dropout rate p" />
          </label>
        )}
        {method === 'ensemble' && (
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
            <span>ensemble size M: <strong>{mDisplay}</strong></span>
            <input type="range" min={2} max={50} step={1} value={mDisplay}
              onChange={(e) => setMDisplay(parseInt(e.target.value, 10))}
              onMouseUp={() => setMCommitted(mDisplay)} onTouchEnd={() => setMCommitted(mDisplay)}
              onKeyUp={() => setMCommitted(mDisplay)} aria-label="Ensemble size M" />
          </label>
        )}
      </div>
    </div>
  );
}

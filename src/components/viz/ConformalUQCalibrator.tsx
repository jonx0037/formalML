import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bayesPolyPosterior,
  bayesPolyPredict,
  coverage,
  ensemblePredict,
  fTrue,
  interpAleatoricCenters,
  linspace,
  meanWidth,
  mulberry32,
  sampleHetero,
  sigmaTrue,
  splitConformalConstant,
  splitConformalLocallyWeighted,
  toMlpCoefs,
  type PayloadMember,
} from './shared/uncertainty-quantification';

// =============================================================================
// ConformalUQCalibrator — §8
//
// Pre vs post conformal coverage repair. Toggle base predictor (Laplace, deep
// ensemble); toggle constant vs locally-weighted conformal; α slider. Reports
// pre/post coverage and mean width. Consumes ensemble.json for the ensemble base.
// Static fallback: public/images/topics/uncertainty-quantification/fig_08_pre_post_locally_weighted.png
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

export default function ConformalUQCalibrator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<EnsemblePayload | null>(null);
  const [base, setBase] = useState<'laplace' | 'ensemble'>('ensemble');
  const [locallyWeighted, setLocallyWeighted] = useState(true);
  const [alpha, setAlpha] = useState(0.10);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/uncertainty-quantification/ensemble.json')
      .then((r) => r.json())
      .then((data) => { if (!cancelled) setPayload(data); })
      .catch((err) => console.error('[ConformalUQCalibrator] fetch failed:', err));
    return () => { cancelled = true; };
  }, []);

  const stats = useMemo(() => {
    if (!payload) return null;
    const rng = mulberry32(20260514 + 21);
    // Split payload.X, payload.y into 100/50/2000.
    const n = payload.X.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const Xtr = idx.slice(0, 100).map((i) => payload.X[i]);
    const yTr = idx.slice(0, 100).map((i) => payload.y[i]);
    const Xcal = idx.slice(100, 150).map((i) => payload.X[i]);
    const yCal = idx.slice(100, 150).map((i) => payload.y[i]);
    const evalSet = sampleHetero(2000, -3, 3, rng);
    const aleGrid = interpAleatoricCenters(
      payload.aleatoric.centers, payload.aleatoric.vals, X_GRID);
    const aleEval = interpAleatoricCenters(
      payload.aleatoric.centers, payload.aleatoric.vals, evalSet.X);

    let muFn: (X: number[]) => number[];
    let sigmaFn: (X: number[]) => number[];
    let muGrid: number[];
    let sigmaGrid: number[];

    if (base === 'laplace') {
      const post = bayesPolyPosterior(Xtr, yTr, DEGREE, Xtr.map(sigmaTrue));
      muFn = (XX) => bayesPolyPredict(XX, post, XX.map(sigmaTrue)).fMean;
      sigmaFn = (XX) => bayesPolyPredict(XX, post, XX.map(sigmaTrue))
        .totalVar.map(Math.sqrt);
      const predGrid = bayesPolyPredict(X_GRID, post, X_GRID.map(sigmaTrue));
      muGrid = predGrid.fMean;
      sigmaGrid = predGrid.totalVar.map(Math.sqrt);
    } else {
      const members = payload.members.slice(0, 20).map((m) => toMlpCoefs(m));
      muFn = (XX) => ensemblePredict(XX, members).mean;
      sigmaFn = (XX) => {
        const pred = ensemblePredict(XX, members);
        const ale = interpAleatoricCenters(
          payload.aleatoric.centers, payload.aleatoric.vals, XX);
        return pred.variance.map((v, i) => Math.sqrt(v + ale[i]));
      };
      const predGrid = ensemblePredict(X_GRID, members);
      muGrid = predGrid.mean;
      sigmaGrid = predGrid.variance.map((v, i) => Math.sqrt(v + aleGrid[i]));
    }

    // Pre-conformal (Gaussian band).
    const z = 1.645; // ≈ inverse normal CDF at 0.95.
    const preLo = muGrid.map((m, i) => m - z * sigmaGrid[i]);
    const preHi = muGrid.map((m, i) => m + z * sigmaGrid[i]);
    const muEval = muFn(evalSet.X);
    const sigmaEval = sigmaFn(evalSet.X);
    const preLoEval = muEval.map((m, i) => m - z * sigmaEval[i]);
    const preHiEval = muEval.map((m, i) => m + z * sigmaEval[i]);
    const preCov = coverage(preLoEval, preHiEval, evalSet.y);
    const preW = meanWidth(preLoEval, preHiEval);

    // Conformal repair on (Xcal, yCal) → predict on Xeval and X_GRID.
    const confEval = locallyWeighted
      ? splitConformalLocallyWeighted(Xcal, yCal, evalSet.X, muFn, sigmaFn, alpha)
      : splitConformalConstant(Xcal, yCal, evalSet.X, muFn, alpha);
    const confGrid = locallyWeighted
      ? splitConformalLocallyWeighted(Xcal, yCal, X_GRID, muFn, sigmaFn, alpha)
      : splitConformalConstant(Xcal, yCal, X_GRID, muFn, alpha);
    const postCov = coverage(confEval.lo, confEval.hi, evalSet.y);
    const postW = meanWidth(confEval.lo, confEval.hi);

    return {
      Xtr, yTr,
      muGrid,
      preLo, preHi,
      postLo: confGrid.lo, postHi: confGrid.hi,
      preCov, preW, postCov, postW,
    };
  }, [payload, base, locallyWeighted, alpha]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 24, bottom: 48, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0 || !stats) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-3.3, 3.3]).range([0, W]);
      const yScale = d3.scaleLinear().domain([-3.5, 3.5]).range([H, 0]).nice();

      // Pre band (faint).
      const bandPre = d3.area<number>()
        .x((_, i) => xScale(X_GRID[i]))
        .y0((_, i) => yScale(stats.preLo[i]))
        .y1((_, i) => yScale(stats.preHi[i]));
      g.append('path').datum(X_GRID).attr('d', bandPre)
        .style('fill', '#94a3b8').style('opacity', 0.20);

      // Post band (vibrant).
      const bandPost = d3.area<number>()
        .x((_, i) => xScale(X_GRID[i]))
        .y0((_, i) => yScale(stats.postLo[i]))
        .y1((_, i) => yScale(stats.postHi[i]));
      g.append('path').datum(X_GRID).attr('d', bandPost)
        .style('fill', '#06b6d4').style('opacity', 0.35);

      const lineGen = d3.line<number>()
        .x((_, i) => xScale(X_GRID[i])).y((d) => yScale(d));
      g.append('path').datum(X_GRID.map(fTrue)).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#94a3b8').attr('stroke-dasharray', '5 3').attr('stroke-width', 1.4);
      g.append('path').datum(stats.muGrid).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#06b6d4').attr('stroke-width', 2);

      g.selectAll('.pt').data(stats.Xtr).enter().append('circle').attr('class', 'pt')
        .attr('cx', (d) => xScale(d as number))
        .attr('cy', (_, i) => yScale(stats.yTr[i]))
        .attr('r', 1.8).style('fill', '#1f2937').style('opacity', 0.45);

      g.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      // Stats badge.
      const badge = g.append('g').attr('transform', `translate(${W - 220}, 8)`);
      [
        `pre coverage: ${stats.preCov.toFixed(3)}`,
        `pre width:    ${stats.preW.toFixed(2)}`,
        `post coverage: ${stats.postCov.toFixed(3)}`,
        `post width:   ${stats.postW.toFixed(2)}`,
      ].forEach((t, i) => {
        badge.append('text').attr('x', 0).attr('y', i * 16)
          .style('font-size', '12px').style('fill', 'var(--color-text)').text(t);
      });
    },
    [containerWidth, stats],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      {payload ? (
        <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
          aria-label="Pre-conformal Gaussian band (faint) vs post-conformal repaired band (vibrant) with live coverage and width readouts." />
      ) : (
        <div style={{ height: HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--color-text-secondary)' }}>
          Loading pre-trained ensemble …
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>base predictor</span>
          <select value={base} onChange={(e) => setBase(e.target.value as typeof base)}
            aria-label="Base UQ method">
            <option value="laplace">Laplace</option>
            <option value="ensemble">Deep ensemble</option>
          </select>
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
          <span>α: <strong>{alpha.toFixed(2)}</strong> (target {(1 - alpha).toFixed(2)})</span>
          <input type="range" min={0.02} max={0.30} step={0.01} value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            aria-label="Miscoverage α" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={locallyWeighted}
            onChange={(e) => setLocallyWeighted(e.target.checked)} />
          <span>locally-weighted (adaptive width)</span>
        </label>
      </div>
    </div>
  );
}

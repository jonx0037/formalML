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
  gaussianFrom,
  linspace,
  mulberry32,
  sampleHetero,
  sigmaTrue,
  splitConformalLocallyWeighted,
  type MlpCoefs,
} from './shared/uncertainty-quantification';

// =============================================================================
// DistributionShiftDegradation — §11
//
// Sweep covariate-shift magnitude s ∈ [0, 2] on the §2 toy. Three methods —
// Laplace (closed form), deep ensemble (M = 20), and locally-weighted conformal
// on the ensemble. Plot marginal coverage vs s; toggle methods.
// Static fallback: public/images/topics/uncertainty-quantification/fig_11_coverage_degradation.png
// =============================================================================

const HEIGHT = 440;
const DEGREE = 7;
const ALPHA = 0.10;
const SHIFT_GRID = linspace(0, 2, 11);

type EnsemblePayload = {
  X: number[];
  y: number[];
  members: { coefs: number[][][]; intercepts: number[][] }[];
  aleatoric: { centers: number[]; vals: number[] };
};

function interpAleatoric(payload: EnsemblePayload, X: number[]): number[] {
  const { centers, vals } = payload.aleatoric;
  return X.map((x) => {
    if (x <= centers[0]) return vals[0];
    if (x >= centers[centers.length - 1]) return vals[centers.length - 1];
    let lo = 0;
    let hi = centers.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (centers[mid] <= x) lo = mid;
      else hi = mid;
    }
    const t = (x - centers[lo]) / (centers[hi] - centers[lo]);
    return vals[lo] + t * (vals[hi] - vals[lo]);
  });
}

function shiftedSample(s: number, n: number, rng: () => number) {
  const g = gaussianFrom(rng);
  const X: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = -3 + s + 6 * rng();
    X.push(xi);
    y.push(fTrue(xi) + sigmaTrue(xi) * g());
  }
  return { X, y };
}

export default function DistributionShiftDegradation() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<EnsemblePayload | null>(null);
  const [showLaplace, setShowLaplace] = useState(true);
  const [showEnsemble, setShowEnsemble] = useState(true);
  const [showConformal, setShowConformal] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/uncertainty-quantification/ensemble.json')
      .then((r) => r.json())
      .then((data) => { if (!cancelled) setPayload(data); })
      .catch((err) => console.error('[DistributionShiftDegradation] fetch failed:', err));
    return () => { cancelled = true; };
  }, []);

  const curves = useMemo(() => {
    if (!payload) return null;
    const rng = mulberry32(20260514 + 40);
    // Train fold for conformal cal: use first 100 of payload.X as train, 50 as cal.
    const idx = Array.from({ length: payload.X.length }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const Xcal = idx.slice(100, 150).map((i) => payload.X[i]);
    const yCal = idx.slice(100, 150).map((i) => payload.y[i]);
    // Refit Laplace on all of payload.X.
    const post = bayesPolyPosterior(payload.X, payload.y, DEGREE, payload.X.map(sigmaTrue));
    const members = payload.members.slice(0, 20) as unknown as MlpCoefs[];
    members.forEach((m) => { m.activation = 'tanh'; });

    const laplaceMu = (X: number[]) => bayesPolyPredict(X, post, X.map(sigmaTrue)).fMean;
    const laplaceSd = (X: number[]) =>
      bayesPolyPredict(X, post, X.map(sigmaTrue)).totalVar.map(Math.sqrt);
    const ensembleMu = (X: number[]) => ensemblePredict(X, members).mean;
    const ensembleSd = (X: number[]) => {
      const pred = ensemblePredict(X, members);
      const ale = interpAleatoric(payload, X);
      return pred.variance.map((v, i) => Math.sqrt(v + ale[i]));
    };

    const lapCov: number[] = [];
    const ensCov: number[] = [];
    const confCov: number[] = [];

    for (const s of SHIFT_GRID) {
      const test = shiftedSample(s, 800, rng);
      const z = 1.645; // 95% Gaussian quantile.
      // Laplace.
      const muL = laplaceMu(test.X);
      const sdL = laplaceSd(test.X);
      lapCov.push(coverage(muL.map((m, i) => m - z * sdL[i]),
        muL.map((m, i) => m + z * sdL[i]), test.y));
      // Ensemble.
      const muE = ensembleMu(test.X);
      const sdE = ensembleSd(test.X);
      ensCov.push(coverage(muE.map((m, i) => m - z * sdE[i]),
        muE.map((m, i) => m + z * sdE[i]), test.y));
      // LW-conformal on ensemble.
      const conf = splitConformalLocallyWeighted(Xcal, yCal, test.X, ensembleMu, ensembleSd, ALPHA);
      confCov.push(coverage(conf.lo, conf.hi, test.y));
    }
    return { lapCov, ensCov, confCov };
  }, [payload]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 24, bottom: 48, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0 || !curves) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 2]).range([0, W]);
      const yScale = d3.scaleLinear().domain([0.4, 1]).range([H, 0]);

      g.append('line').attr('x1', 0).attr('x2', W)
        .attr('y1', yScale(1 - ALPHA)).attr('y2', yScale(1 - ALPHA))
        .style('stroke', '#94a3b8').attr('stroke-dasharray', '5 3').attr('stroke-width', 1.4);
      g.append('text').attr('x', W - 8).attr('y', yScale(1 - ALPHA) - 6).attr('text-anchor', 'end')
        .style('font-size', '11px').style('fill', '#94a3b8').text(`target 1 − α = ${(1 - ALPHA).toFixed(2)}`);

      const drawCurve = (values: number[], color: string, label: string, on: boolean) => {
        if (!on) return;
        const lineGen = d3.line<number>()
          .x((_, i) => xScale(SHIFT_GRID[i])).y((d) => yScale(d));
        g.append('path').datum(values).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', color).attr('stroke-width', 2);
        g.selectAll('.dot-' + label).data(values).enter().append('circle')
          .attr('cx', (_, i) => xScale(SHIFT_GRID[i]))
          .attr('cy', (d) => yScale(d as number))
          .attr('r', 4).style('fill', color);
      };

      drawCurve(curves.lapCov, '#8b5cf6', 'lap', showLaplace);
      drawCurve(curves.ensCov, '#f59e0b', 'ens', showEnsemble);
      drawCurve(curves.confCov, '#06b6d4', 'conf', showConformal);

      g.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6, '.2f'))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('covariate-shift magnitude s');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -40)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('marginal coverage');
    },
    [containerWidth, curves, showLaplace, showEnsemble, showConformal],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      {payload ? (
        <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
          aria-label="Coverage degradation under covariate shift for Laplace, deep ensemble, and locally-weighted conformal on the ensemble." />
      ) : (
        <div style={{ height: HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--color-text-secondary)' }}>
          Loading pre-trained ensemble …
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showLaplace} onChange={(e) => setShowLaplace(e.target.checked)} />
          <span style={{ color: '#8b5cf6', fontWeight: 600 }}>Laplace</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showEnsemble} onChange={(e) => setShowEnsemble(e.target.checked)} />
          <span style={{ color: '#f59e0b', fontWeight: 600 }}>Deep ensemble</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showConformal} onChange={(e) => setShowConformal(e.target.checked)} />
          <span style={{ color: '#06b6d4', fontWeight: 600 }}>LW-conformal on ensemble</span>
        </label>
      </div>
    </div>
  );
}

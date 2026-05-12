import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  aicPenalty,
  bartlettMendelsonPenalty,
  bicPenalty,
  mulberry32,
  plugInSigmaSq,
  polynomialUnitBallRademacher,
  sampleSinTarget,
  srmPickFromArrays,
  trainingMseByDegree,
  vapnikPenalty,
} from './shared/structural-risk-minimization';

// =============================================================================
// InfoCriteriaComparison — interactive companion to §8.5.
//
// Two sliders: n (25-500), σ (0.05-0.5). Computes training MSE and four
// penalty curves (AIC, BIC, Vapnik C=1, Bartlett-Mendelson Rademacher) on the
// §1 toy. Plots all four on a log y-scale (penalties span ~50×) and tabulates
// picked $\hat k$ for each rule. Commit-on-release for both n and σ — the
// Rademacher path uses B = 150 MC draws per k and is too heavy for live drag.
//
// Static fallback: public/images/topics/structural-risk-minimization/08_info_criteria_comparison.png
// =============================================================================

const HEIGHT = 380;
const K_MAX = 15;
const SM_BREAKPOINT = 640;
const DELTA = 0.05;

export default function InfoCriteriaComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nDisplay, setNDisplay] = useState(50);
  const [n, setN] = useState(50);
  const [sigmaDisplay, setSigmaDisplay] = useState(0.2);
  const [sigma, setSigma] = useState(0.2);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const commitN = () => setN(nDisplay);
  const commitSigma = () => setSigma(sigmaDisplay);

  const { trainMse, aic, bic, vapnik, rad, picks, ks } = useMemo(() => {
    const rng = mulberry32(20260512 + n + Math.round(sigma * 1000));
    const { X, Y } = sampleSinTarget(n, sigma, rng);
    const train = trainingMseByDegree(X, Y, K_MAX);
    const sigmaSq = plugInSigmaSq(X, Y, K_MAX);
    const aic = new Float64Array(K_MAX + 1);
    const bic = new Float64Array(K_MAX + 1);
    const vap = new Float64Array(K_MAX + 1);
    const radPen = new Float64Array(K_MAX + 1);
    for (let k = 0; k <= K_MAX; k++) {
      const d = k + 1;
      aic[k] = aicPenalty(d, sigmaSq, n);
      bic[k] = bicPenalty(d, sigmaSq, n);
      vap[k] = vapnikPenalty(d, n, k, DELTA, 1);
      const rEst = polynomialUnitBallRademacher(X, k, 150, rng);
      radPen[k] = bartlettMendelsonPenalty(rEst.mean, n, k, DELTA);
    }
    return {
      trainMse: train,
      aic,
      bic,
      vapnik: vap,
      rad: radPen,
      picks: {
        AIC: srmPickFromArrays(train, aic),
        BIC: srmPickFromArrays(train, bic),
        Vapnik: srmPickFromArrays(train, vap),
        Rademacher: srmPickFromArrays(train, radPen),
      },
      ks: Array.from({ length: K_MAX + 1 }, (_, k) => k),
    };
  }, [n, sigma]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 16, right: 24, bottom: 36, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0, K_MAX]).range([0, W]);
      const allVals = [...aic, ...bic, ...vapnik, ...rad].filter((v) => v > 0);
      const yMin = Math.max(d3.min(allVals) || 1e-3, 1e-4);
      const yMax = d3.max(allVals) || 1;
      const y = d3.scaleLog().domain([yMin / 2, yMax * 1.5]).range([H, 0]).clamp(true);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(8));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('polynomial degree k');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -40).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('penalty (log scale)');
      const lineGen = (vals: Float64Array) => d3.line<number>().x((k) => x(k)).y((k) => y(Math.max(vals[k], yMin / 2)));
      const series = [
        { vals: aic, color: '#3b82f6', name: 'AIC' },
        { vals: bic, color: '#10b981', name: 'BIC' },
        { vals: vapnik, color: '#ef4444', name: 'Vapnik (C=1)', dash: '5 3' },
        { vals: rad, color: '#f59e0b', name: 'Rademacher', dash: '2 3' },
      ];
      series.forEach((s) => {
        g.append('path').datum(ks).attr('d', lineGen(s.vals)).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 2).attr('stroke-dasharray', s.dash || '');
      });
      const legend = g.append('g').attr('transform', `translate(${W - 130}, 6)`);
      series.forEach((s, i) => {
        legend.append('line').attr('x1', 0).attr('y1', i * 14 + 4).attr('x2', 14).attr('y2', i * 14 + 4).attr('stroke', s.color).attr('stroke-width', 2).attr('stroke-dasharray', s.dash || '');
        legend.append('text').attr('x', 20).attr('y', i * 14 + 7).style('fill', 'var(--color-text)').style('font-size', '11px').text(s.name);
      });
    },
    [aic, bic, vapnik, rad, ks, containerWidth],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="grid gap-3 mb-3 text-xs" style={{ gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)' }}>
        <label className="flex flex-col gap-1">
          <span><strong>n</strong>: <span className="tabular-nums">{nDisplay}</span></span>
          <input type="range" min={25} max={500} step={5} value={nDisplay} onChange={(e) => setNDisplay(Number(e.target.value))} onMouseUp={commitN} onTouchEnd={commitN} onKeyUp={commitN} aria-label="sample size n" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>σ</strong> (noise std): <span className="tabular-nums">{sigmaDisplay.toFixed(2)}</span></span>
          <input type="range" min={0.05} max={0.5} step={0.01} value={sigmaDisplay} onChange={(e) => setSigmaDisplay(Number(e.target.value))} onMouseUp={commitSigma} onTouchEnd={commitSigma} onKeyUp={commitSigma} aria-label="noise standard deviation" />
        </label>
      </div>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
      <div className="text-xs mt-2 grid gap-1" style={{ color: 'var(--color-text-secondary)', gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <div>AIC pick: <strong>{picks.AIC}</strong></div>
        <div>BIC pick: <strong>{picks.BIC}</strong></div>
        <div>Vapnik pick: <strong>{picks.Vapnik}</strong></div>
        <div>Rademacher pick: <strong>{picks.Rademacher}</strong></div>
      </div>
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        Same SRM template, different calibrations, different picks. AIC/BIC are parametric/linear; Vapnik/Rademacher are non-parametric/square-root and dominate AIC/BIC by 1-2 orders of magnitude.
      </div>
    </div>
  );
}

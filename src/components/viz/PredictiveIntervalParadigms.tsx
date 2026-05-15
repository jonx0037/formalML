import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bayesPolyPosterior,
  bayesPolyPredict,
  coverage,
  fitOlsPoly,
  fTrue,
  gaussianFrom,
  linspace,
  meanWidth,
  mulberry32,
  normCdf,
  predictOlsPoly,
  sampleHetero,
  sigmaTrue,
  splitConformalConstant,
} from './shared/uncertainty-quantification';

// =============================================================================
// PredictiveIntervalParadigms — §5
//
// Three PI paradigms on the §2 toy at user-set α: Bayesian PP, plug-in
// (homoscedastic — deliberately misspecified), split conformal (constant width).
// Per-method live coverage / mean width readout. Quantile regression is left to
// the prediction-intervals topic; its construction is paradigm-defining rather
// than methodology-defining, so showing three here is faithful to the §5 spec.
// Static fallback: public/images/topics/uncertainty-quantification/fig_05_paradigm_overlay.png
// =============================================================================

const HEIGHT = 480;
const DEGREE = 7;

function inverseNormalCdfApprox(p: number): number {
  // Beasley-Springer-Moro algorithm via golden-section bracket; here we use
  // an iterative bisection on normCdf since we only need ~6 digits.
  if (p <= 0 || p >= 1) return p < 0.5 ? -8 : 8;
  let lo = -8;
  let hi = 8;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (normCdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

export default function PredictiveIntervalParadigms() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [alphaDisplay, setAlphaDisplay] = useState(0.10);
  const [alphaCommitted, setAlphaCommitted] = useState(0.10);

  const fit = useMemo(() => {
    const rng = mulberry32(20260514);
    const { X, y } = sampleHetero(200, -3, 3, rng);
    // Split 100/50 train/cal.
    const idx = Array.from({ length: 200 }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const Xtr = idx.slice(0, 100).map((i) => X[i]);
    const yTr = idx.slice(0, 100).map((i) => y[i]);
    const Xcal = idx.slice(100, 150).map((i) => X[i]);
    const yCal = idx.slice(100, 150).map((i) => y[i]);
    // Eval set.
    const evalSet = sampleHetero(2000, -3, 3, rng);
    return { X, y, Xtr, yTr, Xcal, yCal, evalSet };
  }, []);

  const paradigms = useMemo(() => {
    const z = inverseNormalCdfApprox(1 - alphaCommitted / 2);
    const xGrid = linspace(-3.2, 3.2, 220);
    // Bayesian PP.
    const post = bayesPolyPosterior(fit.Xtr, fit.yTr, DEGREE,
      fit.Xtr.map(sigmaTrue));
    const predGrid = bayesPolyPredict(xGrid, post, xGrid.map(sigmaTrue));
    const bayes = {
      mean: predGrid.fMean,
      lo: predGrid.fMean.map((m, i) => m - z * Math.sqrt(predGrid.totalVar[i])),
      hi: predGrid.fMean.map((m, i) => m + z * Math.sqrt(predGrid.totalVar[i])),
    };
    // Coverage on eval.
    const predEval = bayesPolyPredict(fit.evalSet.X, post, fit.evalSet.X.map(sigmaTrue));
    const bayesEvalLo = predEval.fMean.map((m, i) => m - z * Math.sqrt(predEval.totalVar[i]));
    const bayesEvalHi = predEval.fMean.map((m, i) => m + z * Math.sqrt(predEval.totalVar[i]));
    const bayesCov = coverage(bayesEvalLo, bayesEvalHi, fit.evalSet.y);
    const bayesWidth = meanWidth(bayesEvalLo, bayesEvalHi);

    // Plug-in (misspecified homoscedastic).
    const beta = fitOlsPoly(fit.Xtr, fit.yTr, DEGREE);
    const fitted = predictOlsPoly(beta, fit.Xtr, DEGREE);
    const resid = fit.yTr.map((yi, i) => yi - fitted[i]);
    const sigmaHat = Math.sqrt(
      resid.reduce((s, r) => s + r * r, 0) / Math.max(1, fit.yTr.length - DEGREE - 1),
    );
    const fGrid = predictOlsPoly(beta, xGrid, DEGREE);
    const plug = {
      mean: fGrid,
      lo: fGrid.map((m) => m - z * sigmaHat),
      hi: fGrid.map((m) => m + z * sigmaHat),
    };
    const fEval = predictOlsPoly(beta, fit.evalSet.X, DEGREE);
    const plugCov = coverage(fEval.map((m) => m - z * sigmaHat), fEval.map((m) => m + z * sigmaHat),
      fit.evalSet.y);
    const plugWidth = 2 * z * sigmaHat;

    // Split conformal.
    const muFn = (XX: number[]) => predictOlsPoly(beta, XX, DEGREE);
    const conf = splitConformalConstant(fit.Xcal, fit.yCal, xGrid, muFn, alphaCommitted);
    const confEval = splitConformalConstant(fit.Xcal, fit.yCal, fit.evalSet.X, muFn, alphaCommitted);
    const confCov = coverage(confEval.lo, confEval.hi, fit.evalSet.y);
    const confWidth = meanWidth(confEval.lo, confEval.hi);

    return {
      xGrid,
      bayes: { ...bayes, cov: bayesCov, w: bayesWidth },
      plug: { ...plug, cov: plugCov, w: plugWidth },
      conf: { mean: fGrid, lo: conf.lo, hi: conf.hi, cov: confCov, w: confWidth },
    };
  }, [fit, alphaCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 24, bottom: 48, left: 56 };
      const panelW = (w - margin.left - margin.right - 2 * 24) / 3;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      const xScale = d3.scaleLinear().domain([-3.3, 3.3]).range([0, panelW]);
      const yScale = d3.scaleLinear().domain([-4, 4]).range([H, 0]).nice();
      const panels = [
        { title: 'Bayesian PP', data: paradigms.bayes, color: '#8b5cf6' },
        { title: 'Plug-in (homo σ̂)', data: paradigms.plug, color: '#6b7280' },
        { title: 'Split conformal', data: paradigms.conf, color: '#06b6d4' },
      ];

      panels.forEach((panel, k) => {
        const g = svg.append('g')
          .attr('transform', `translate(${margin.left + k * (panelW + 24)},${margin.top})`);
        const area = d3.area<number>()
          .x((_, i) => xScale(paradigms.xGrid[i]))
          .y0((_, i) => yScale(panel.data.lo[i]))
          .y1((_, i) => yScale(panel.data.hi[i]));
        g.append('path').datum(paradigms.xGrid).attr('d', area)
          .style('fill', panel.color).style('opacity', 0.28);
        const lineGen = d3.line<number>()
          .x((_, i) => xScale(paradigms.xGrid[i])).y((d) => yScale(d));
        g.append('path').datum(panel.data.mean).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', panel.color).attr('stroke-width', 2);
        g.append('path').datum(paradigms.xGrid.map(fTrue)).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', '#94a3b8').attr('stroke-dasharray', '5 3');
        g.selectAll('.pt').data(fit.Xtr).enter().append('circle').attr('class', 'pt')
          .attr('cx', (d) => xScale(d as number))
          .attr('cy', (_, i) => yScale(fit.yTr[i]))
          .attr('r', 1.8).style('fill', '#1f2937').style('opacity', 0.45);
        g.append('g').attr('transform', `translate(0,${H})`)
          .call(d3.axisBottom(xScale).ticks(5))
          .selectAll('text').style('fill', 'var(--color-text)');
        g.append('g').call(d3.axisLeft(yScale).ticks(5))
          .selectAll('text').style('fill', 'var(--color-text)');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
        g.append('text').attr('x', panelW / 2).attr('y', -10).attr('text-anchor', 'middle')
          .style('font-size', '12px').style('fill', 'var(--color-text)').text(panel.title);
        g.append('text').attr('x', panelW / 2).attr('y', H + 36).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
          .text(`coverage ${panel.data.cov.toFixed(3)}  ·  width ${panel.data.w.toFixed(2)}`);
      });
    },
    [containerWidth, paradigms, fit],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Three predictive-interval paradigms (Bayesian, plug-in, split conformal) on the heteroscedastic toy, with live coverage and width annotations." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 240 }}>
          <span>α (miscoverage target): <strong>{alphaDisplay.toFixed(2)}</strong>  (target {(1 - alphaDisplay).toFixed(2)})</span>
          <input type="range" min={0.02} max={0.30} step={0.01} value={alphaDisplay}
            onChange={(e) => setAlphaDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setAlphaCommitted(alphaDisplay)} onTouchEnd={() => setAlphaCommitted(alphaDisplay)}
            onKeyUp={() => setAlphaCommitted(alphaDisplay)} aria-label="Miscoverage target α" />
        </label>
      </div>
      <p style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        At α = 0.10 all three target 0.90 marginal coverage. Bayesian PP is input-adaptive (wider at the
        edges). Plug-in is constant-width and undercovers because its homoscedastic σ̂ misses heteroscedasticity.
        Conformal is constant-width but distribution-free.
      </p>
    </div>
  );
}

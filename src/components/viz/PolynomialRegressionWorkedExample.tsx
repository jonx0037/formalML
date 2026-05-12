import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  aicPick,
  bicPick,
  biasVarianceMC,
  cvPickHistogram,
  logspace,
  mulberry32,
  pacBayesPick,
  polyfitDegree,
  polyvalIncreasing,
  rademacherPick,
  sampleSinTarget,
  targetSin,
  vapnikPick,
} from './shared/structural-risk-minimization';

// =============================================================================
// PolynomialRegressionWorkedExample — interactive companion to §11.5
// (flagship). Four sliders: n, σ, K_folds, δ. Three panels:
//   1. Agreement matrix as annotated cells (six rules + oracle)
//   2. Money shot: SRM-picked polynomial fit with ±2σ bias-variance envelope
//   3. Sensitivity bar: each rule's $\hat k$ shown as a vertical bar
//
// Heavy MC so commit-on-release for n and σ.
//
// Static fallback: public/images/topics/structural-risk-minimization/11_worked_example_money_shot.png
// =============================================================================

const HEIGHT_AGREEMENT = 220;
const HEIGHT_MONEY = 380;
const SIGMA_DEFAULT = 0.2;
const K_MAX = 15;
const SM_BREAKPOINT = 640;

const RULES = ['AIC', 'BIC', 'Vapnik', 'Rademacher', 'CV', 'PAC-Bayes', 'Oracle k*'] as const;

export default function PolynomialRegressionWorkedExample() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nDisplay, setNDisplay] = useState(50);
  const [n, setN] = useState(50);
  const [sigmaDisplay, setSigmaDisplay] = useState(SIGMA_DEFAULT);
  const [sigma, setSigma] = useState(SIGMA_DEFAULT);
  const [K, setK] = useState(5);
  const [delta, setDelta] = useState(0.05);
  const [headlineRule, setHeadlineRule] = useState<typeof RULES[number]>('Rademacher');

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const commitN = () => setN(nDisplay);
  const commitSigma = () => setSigma(sigmaDisplay);

  const { sample, picks, biasVar } = useMemo(() => {
    const rng = mulberry32(20260512 + n + Math.round(sigma * 1000));
    const { X, Y } = sampleSinTarget(n, sigma, rng);
    const bv = biasVarianceMC(n, sigma, K_MAX, 80, mulberry32(20260513 + n + Math.round(sigma * 1000)));
    const lambdas = logspace(-6, 4, 30);
    return {
      sample: { X, Y },
      picks: {
        AIC: aicPick(X, Y, K_MAX),
        BIC: bicPick(X, Y, K_MAX),
        Vapnik: vapnikPick(X, Y, K_MAX, delta, 1),
        Rademacher: rademacherPick(X, Y, K_MAX, delta, 200, rng),
        CV: cvPickHistogram(X, Y, K_MAX, K, 40, rng).mode,
        'PAC-Bayes': pacBayesPick(X, Y, K_MAX, lambdas, 1.0, 0.1, delta).pickedDegree,
        'Oracle k*': bv.kStar,
      },
      biasVar: bv,
    };
  }, [n, sigma, K, delta]);

  const xGrid = useMemo(() => {
    const arr = new Float64Array(201);
    for (let i = 0; i < arr.length; i++) arr[i] = -1 + (2 * i) / 200;
    return arr;
  }, []);

  const headlineK = picks[headlineRule];
  const headlineCoefs = useMemo(() => polyfitDegree(sample.X, sample.Y, headlineK), [sample, headlineK]);

  const agreementRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 16, right: 16, bottom: 24, left: 90 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT_AGREEMENT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT_AGREEMENT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const cellH = H / RULES.length;
      const x = d3.scaleLinear().domain([0, K_MAX]).range([0, W]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(8));
      g.append('text').attr('x', W / 2).attr('y', H + 18).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('picked degree k̂');
      const colorBy = (k: number) => d3.interpolateViridis(k / K_MAX);
      RULES.forEach((rule, i) => {
        g.append('text').attr('x', -8).attr('y', i * cellH + cellH / 2 + 4).attr('text-anchor', 'end').style('fill', 'var(--color-text)').style('font-size', '11px').style('font-weight', rule === headlineRule ? '700' : '400').text(rule);
        const k = picks[rule];
        g.append('rect').attr('x', 0).attr('y', i * cellH + 2).attr('width', W).attr('height', cellH - 4).attr('fill', 'var(--color-bg)').attr('stroke', 'var(--color-border)');
        g.append('circle').attr('cx', x(k)).attr('cy', i * cellH + cellH / 2).attr('r', 8).attr('fill', colorBy(k)).attr('stroke', rule === headlineRule ? 'var(--color-text)' : 'none').attr('stroke-width', 2);
        g.append('text').attr('x', x(k)).attr('y', i * cellH + cellH / 2 + 4).attr('text-anchor', 'middle').style('fill', 'white').style('font-size', '10px').style('font-weight', '600').text(k);
      });
    },
    [picks, headlineRule, containerWidth],
  );

  const moneyRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w * 0.6;
      const margin = { top: 16, right: 16, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT_MONEY - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT_MONEY}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([-1, 1]).range([0, W]);
      const y = d3.scaleLinear().domain([-2, 2]).range([H, 0]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', W / 2).attr('y', H + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px').text('x');
      // ±2σ envelope from variance approximation
      const stdAtK = Math.sqrt(biasVar.variance[headlineK]);
      const upper = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(polyvalIncreasing(headlineCoefs, xGrid[i]) + 2 * stdAtK));
      const lower = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(polyvalIncreasing(headlineCoefs, xGrid[i]) - 2 * stdAtK));
      const envelope = d3.area<number>().x((i) => x(xGrid[i])).y0((i) => y(polyvalIncreasing(headlineCoefs, xGrid[i]) - 2 * stdAtK)).y1((i) => y(polyvalIncreasing(headlineCoefs, xGrid[i]) + 2 * stdAtK));
      g.append('path').datum(d3.range(xGrid.length)).attr('d', envelope).attr('fill', '#ef4444').attr('opacity', 0.15);
      // Truth
      const truth = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(targetSin(xGrid[i])));
      g.append('path').datum(d3.range(xGrid.length)).attr('d', truth).attr('fill', 'none').attr('stroke', '#10b981').attr('stroke-width', 2).attr('stroke-dasharray', '5 3');
      // Fit
      const fit = d3.line<number>().x((i) => x(xGrid[i])).y((i) => y(polyvalIncreasing(headlineCoefs, xGrid[i])));
      g.append('path').datum(d3.range(xGrid.length)).attr('d', fit).attr('fill', 'none').attr('stroke', '#ef4444').attr('stroke-width', 2.6);
      // Data
      g.selectAll(null).data(d3.range(n)).enter().append('circle').attr('cx', (i) => x(sample.X[i])).attr('cy', (i) => y(sample.Y[i])).attr('r', 2.4).attr('fill', '#3b82f6').attr('opacity', 0.6);
      g.append('text').attr('x', 8).attr('y', 14).style('fill', 'var(--color-text)').style('font-size', '11px').style('font-weight', '600').text(`${headlineRule}-picked degree ${headlineK} fit + ±2σ envelope`);
    },
    [sample, headlineCoefs, headlineK, headlineRule, biasVar, xGrid, n, containerWidth, isMobile],
  );

  const sigmaRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w * 0.4 - 16;
      const margin = { top: 16, right: 16, bottom: 36, left: 80 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT_MONEY - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT_MONEY}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      // Show picks as horizontal bars
      const ruleNames = [...RULES];
      const y = d3.scaleBand().domain(ruleNames).range([0, H]).padding(0.2);
      const x = d3.scaleLinear().domain([0, K_MAX]).range([0, W]);
      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y));
      g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('picked degree');
      ruleNames.forEach((rule) => {
        const k = picks[rule];
        g.append('rect').attr('x', 0).attr('y', y(rule)!).attr('width', x(k)).attr('height', y.bandwidth()).attr('fill', d3.interpolateViridis(k / K_MAX)).attr('opacity', rule === headlineRule ? 1.0 : 0.55);
        g.append('text').attr('x', x(k) + 4).attr('y', y(rule)! + y.bandwidth() / 2 + 4).style('fill', 'var(--color-text)').style('font-size', '11px').text(String(k));
      });
    },
    [picks, headlineRule, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="my-6 border rounded-lg p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}>
      <div className="grid gap-3 mb-3 text-xs" style={{ gridTemplateColumns: isMobile ? '1fr' : 'repeat(4, 1fr)' }}>
        <label className="flex flex-col gap-1">
          <span><strong>n</strong>: <span className="tabular-nums">{nDisplay}</span></span>
          <input type="range" min={25} max={500} step={5} value={nDisplay} onChange={(e) => setNDisplay(Number(e.target.value))} onMouseUp={commitN} onTouchEnd={commitN} onKeyUp={commitN} aria-label="sample size n" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>σ</strong>: <span className="tabular-nums">{sigmaDisplay.toFixed(2)}</span></span>
          <input type="range" min={0.05} max={0.5} step={0.01} value={sigmaDisplay} onChange={(e) => setSigmaDisplay(Number(e.target.value))} onMouseUp={commitSigma} onTouchEnd={commitSigma} onKeyUp={commitSigma} aria-label="noise std" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>K</strong> (folds): <span className="tabular-nums">{K}</span></span>
          <input type="range" min={2} max={20} step={1} value={K} onChange={(e) => setK(Number(e.target.value))} aria-label="number of folds K" />
        </label>
        <label className="flex flex-col gap-1">
          <span><strong>δ</strong>: <span className="tabular-nums">{delta.toFixed(2)}</span></span>
          <input type="range" min={0.01} max={0.5} step={0.01} value={delta} onChange={(e) => setDelta(Number(e.target.value))} aria-label="confidence delta" />
        </label>
      </div>
      <label className="flex items-center gap-2 text-xs mb-3">
        <span>headline rule:</span>
        <select value={headlineRule} onChange={(e) => setHeadlineRule(e.target.value as typeof RULES[number])} className="border rounded px-2 py-1" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)', color: 'var(--color-text)' }}>
          {RULES.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
      </label>
      <div className="text-xs mb-2" style={{ color: 'var(--color-text-secondary)' }}>Agreement matrix</div>
      <svg ref={agreementRef} style={{ width: '100%', height: HEIGHT_AGREEMENT }} />
      <div className="text-xs mt-3 mb-2" style={{ color: 'var(--color-text-secondary)' }}>Money shot + sensitivity bar</div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '16px' }}>
        <svg ref={moneyRef} style={{ width: '100%', height: HEIGHT_MONEY }} />
        <svg ref={sigmaRef} style={{ width: '100%', height: HEIGHT_MONEY }} />
      </div>
      <div className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
        Every rule, every pick, at this (n, σ, K, δ). Set the headline-rule selector to swap whose fit drives the money shot. The agreement matrix reveals when rules disagree (small n) and when they cluster (large n).
      </div>
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, logBetaMixtureEvalue, binaryKL } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 8.1 — The e-process catalogue.
//
// Single panel, log-y. On a single Bernoulli(μ_true) stream of length n_max,
// plot five processes testing H_0: μ = μ_0:
//   (1) Wald LR with correct alternative (= μ_true)
//   (2) Wald LR with mis-specified alternative
//   (3) sub-Bernoulli Beta-mixture
//   (4) PrPl-EB betting
//   (5) Fixed-n p-value: -log p (NOT an e-process, gray dotted)
// Horizontal threshold at log(1/δ).
// -----------------------------------------------------------------------------

const MARGIN = { top: 28, right: 18, bottom: 38, left: 54 };
const PANEL_H = 360;
const N_MAX = 1000;
const SEED = 20261201;

const COLOR_THRESH = 'var(--color-accent)';

interface Catalog {
  ng: number[];
  walLR_correct: number[];
  walLR_misspec: number[];
  betaMix: number[];
  prplEB: number[];
  fixedNeglogP: number[];
}

function compute(muTrue: number, mu0: number, muMisspec: number, delta: number): Catalog {
  const rng = mulberry32(SEED);
  // Generate one stream
  const stream = new Int8Array(N_MAX);
  for (let i = 0; i < N_MAX; i++) stream[i] = rng() < muTrue ? 1 : 0;

  const ng: number[] = [];
  const walLR_correct: number[] = [];
  const walLR_misspec: number[] = [];
  const betaMix: number[] = [];
  const prplEB: number[] = [];
  const fixedNeglogP: number[] = [];

  let logLRCorrect = 0;
  let logLRMisspec = 0;
  let logPrPl = 0;
  let S = 0;
  // logIncrement: log[ (μ_alt/μ_0)^x · ((1-μ_alt)/(1-μ_0))^(1-x) ]
  const lrInc = (x: number, mu_alt: number) =>
    x === 1 ? Math.log(mu_alt / mu0) : Math.log((1 - mu_alt) / (1 - mu0));

  for (let n = 1; n <= N_MAX; n++) {
    const x = stream[n - 1];
    S += x;
    logLRCorrect += lrInc(x, muTrue);
    logLRMisspec += lrInc(x, muMisspec);

    // PrPl-EB: log K_n with empirical Kelly bet on H_0: μ = μ_0
    // λ_n = (bar X_{n-1} − μ_0) / σ̂²_{n-1}, with bar X_0 set to μ_true to seed.
    let lambda: number;
    if (n === 1) {
      lambda = 0.0;
    } else {
      const meanPrev = (S - x) / (n - 1);
      const varPrev = Math.max(meanPrev * (1 - meanPrev), 0.01);
      lambda = (meanPrev - mu0) / varPrev;
      lambda = Math.max(-0.5 / Math.max(1 - mu0, mu0), Math.min(0.5 / Math.max(1 - mu0, mu0), lambda));
    }
    logPrPl += Math.log1p(lambda * (x - mu0));

    // Beta-mixture (5.14)
    const logBeta = logBetaMixtureEvalue(S, n, mu0);

    // Fixed-n p-value under H_0: μ = μ_0
    // Normal approximation: z = (bar X_n − μ_0) / √(μ_0(1−μ_0)/n).
    // Two-sided p = 2 (1 − Φ(|z|)).
    const meanN = S / n;
    const se = Math.sqrt(Math.max(mu0 * (1 - mu0) / n, 1e-12));
    const z = (meanN - mu0) / se;
    // 1 − Φ(|z|) approximation: erfc(|z|/√2) / 2
    const erfcAbsZ = 1 - erf(Math.abs(z) / Math.SQRT2);
    const pTwoSided = Math.max(Math.min(erfcAbsZ, 1), 1e-300);
    const neglogP = -Math.log(pTwoSided);

    ng.push(n);
    walLR_correct.push(logLRCorrect);
    walLR_misspec.push(logLRMisspec);
    betaMix.push(logBeta);
    prplEB.push(logPrPl);
    fixedNeglogP.push(neglogP);
  }

  return { ng, walLR_correct, walLR_misspec, betaMix, prplEB, fixedNeglogP };
}

// Abramowitz–Stegun erf approximation (max error ~1.5e-7).
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - ((((a5 * t + a4) * t) + a3) * t + a2) * t * t * Math.exp(-ax * ax) - ((a5 * t + a4) * t * t * t * Math.exp(-ax * ax));
  // simpler closed form:
  const yClean = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-ax * ax);
  return sign * yClean;
}

export default function EProcessCatalogue() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [mu0, setMu0] = useState(0.10);
  const [delta, setDelta] = useState(0.05);
  const muTrue = 0.05; // fixed: data come from Bern(0.05)
  const muMisspec = 0.20;

  const data = useMemo(() => compute(muTrue, mu0, muMisspec, delta), [muTrue, mu0, muMisspec, delta]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));
  const logThr = Math.log(1 / delta);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([1, N_MAX]).range([0, innerW]);
      const allY = [...data.walLR_correct, ...data.walLR_misspec, ...data.betaMix, ...data.prplEB, ...data.fixedNeglogP];
      const yMax = Math.max(Math.max(...allY), logThr + 2);
      const yMin = Math.min(Math.min(...allY), -2);
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(4, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(logThr)).attr('y2', yScale(logThr)).style('stroke', COLOR_THRESH).attr('stroke-width', 1.4).attr('stroke-dasharray', '6 3');
      g.append('text').attr('x', innerW - 4).attr('y', yScale(logThr) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_THRESH).text(`log(1/δ) = ${logThr.toFixed(2)}`);

      const series = [
        { values: data.walLR_correct, color: '#356fa3', label: `Wald LR (alt = ${muTrue}, correct)` },
        { values: data.walLR_misspec, color: '#c44545', label: `Wald LR (alt = ${muMisspec}, wrong)` },
        { values: data.betaMix, color: '#7c3aed', label: 'Beta-mixture (5.14)' },
        { values: data.prplEB, color: '#0f6e56', label: 'PrPl-EB (6.7)' },
        { values: data.fixedNeglogP, color: 'var(--color-text-secondary)', label: '−log p (NOT e-process)' },
      ];
      series.forEach((s, i) => {
        const line = d3.line<number>().x((_, idx) => xScale(idx + 1)).y((v) => yScale(v));
        g.append('path').datum(s.values).attr('d', line as never).style('fill', 'none').style('stroke', s.color).attr('stroke-width', 1.8).attr('stroke-dasharray', i === 4 ? '3 3' : 'none');
        g.append('text').attr('x', innerW - 4).attr('y', 12 + i * 14).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', s.color).text(s.label);
      });

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('n (log scale)');
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('log E_n');
    },
    [data, delta, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>Null μ₀ = <strong>{mu0.toFixed(2)}</strong> (stream is Bern(0.05))</span>
          <input type="range" min={0.05} max={0.25} step={0.01} value={mu0} aria-label="Null mean" onChange={(e) => setMu0(Number(e.target.value))} />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '140px' }}>
          <span>δ = <strong>{delta.toFixed(3)}</strong></span>
          <input type="range" min={0.01} max={0.10} step={0.01} value={delta} aria-label="Level" onChange={(e) => setDelta(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
          KL(0.05 ‖ {mu0.toFixed(2)}) = <strong>{binaryKL(0.05, mu0).toFixed(4)}</strong> per step (theoretical Wald drift).
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

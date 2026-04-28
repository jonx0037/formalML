import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  blockMaxima,
  gevMle,
  gevPdf,
  gpdMle,
  gpdPdf,
  mulberry32,
  potVar,
  potVarSeDelta,
  returnLevel,
  returnLevelSeDelta,
  sampleParent,
  type ParentPreset,
} from '../../data/extreme-value-theory';

// =============================================================================
// BlockMaximaVsPOT — embedded after Theorem 7 (Pickands–Balkema–de Haan) in
// §5.2. Side-by-side comparison of the §4 GEV-on-block-maxima fit (left) and
// the §5 GPD-on-exceedances fit (right) applied to the SAME simulated dataset.
//
// The key teaching point: for the same total N raw observations, the POT
// framework typically uses 5–10× more data points (N_u exceedances above the
// τ-quantile threshold) than block-maxima fits 1 × N/m. Both panels report
// the fitted ξ̂, the fitted scale, and a T = 100 return-level estimate; the
// data-efficiency advantage of POT is visible directly in the readouts and
// in the much-tighter standard errors at the same N.
//
// Simulation seed is shared between panels — the underlying X_i are the same;
// only the post-processing differs. Resample button reseeds.
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 768;

// Slider ranges per brief §11.2
const N_MIN = 1000;
const N_MAX = 100_000;
const N_STEP = 1000;
const M_MIN = 10;
const M_MAX = 500;
const M_STEP = 10;
const TAU_MIN = 0.9;
const TAU_MAX = 0.995;
const TAU_STEP = 0.005;

// Color palette
const COLOR_BLOCK = '#534AB7';   // purple — block-maxima
const COLOR_POT = '#0F6E56';     // teal   — POT
const COLOR_DATA = '#9CA3AF';    // gray bars

const PARENT_LABELS: Record<'normal' | 'pareto2' | 't3', string> = {
  normal: 'Normal (light tail, ξ_true = 0)',
  pareto2: 'Pareto(α=2) (heavy tail, ξ_true = 0.5)',
  t3: 'Student-t₃ (heavy tail, ξ_true ≈ 0.33)',
};

const fmt = (x: number, digits = 3) => x.toFixed(digits);

export default function BlockMaximaVsPOT() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [parent, setParent] = useState<'normal' | 'pareto2' | 't3'>('pareto2');
  const [N, setN] = useState(20_000);
  const [m, setM] = useState(100);
  const [tau, setTau] = useState(0.95);
  const [seed, setSeed] = useState(42);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const leftWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.5);
  const rightWidth = isStacked ? containerWidth : containerWidth - leftWidth;

  // ── Simulate once, fit twice ────────────────────────────────────
  const fits = useMemo(() => {
    const rng = mulberry32(seed);
    const samples = sampleParent(parent, N, rng);

    // Block-maxima (left panel)
    const B = Math.floor(N / m);
    const blocks = blockMaxima(samples.subarray(0, B * m), m);
    let gevFit: ReturnType<typeof gevMle> | null = null;
    let gevReturn100: number = NaN;
    let gevReturn100Se: number = NaN;
    if (blocks.length >= 5) {
      try {
        gevFit = gevMle(blocks);
        gevReturn100 = returnLevel(gevFit.theta, 100);
        gevReturn100Se = returnLevelSeDelta(gevFit.theta, gevFit.cov, 100);
      } catch {
        gevFit = null;
      }
    }

    // POT (right panel) — same samples, exceedances above τ-quantile.
    // (a, b) => a - b sorts ASCENDING; the τ-quantile is the τ-fraction-from-the-bottom index.
    const sortedAsc = samples.slice().sort();
    const u = sortedAsc[Math.floor(tau * sortedAsc.length)];
    const exceedances: number[] = [];
    for (const v of samples) if (v > u) exceedances.push(v - u);
    const Nu = exceedances.length;
    let gpdFit: ReturnType<typeof gpdMle> | null = null;
    let potVar100: number = NaN;
    let potVar100Se: number = NaN;
    if (Nu >= 30) {
      try {
        gpdFit = gpdMle(Float64Array.from(exceedances));
        // T = 100 means "1-in-100-block event"; with block size m, that's
        // equivalent to a tail probability 1/(T·m) per individual observation.
        // The POT return level matching the GEV's x_T is VaR_{1 - 1/(T·m)}.
        const alphaPot = 1 - 1 / (100 * m);
        potVar100 = potVar(gpdFit.theta, u, Nu, N, alphaPot);
        potVar100Se = potVarSeDelta(gpdFit.theta, gpdFit.cov, u, Nu, N, alphaPot);
      } catch {
        gpdFit = null;
      }
    }

    return {
      blocks,
      B,
      gevFit,
      gevReturn100,
      gevReturn100Se,
      u,
      Nu,
      exceedances: Float64Array.from(exceedances),
      gpdFit,
      potVar100,
      potVar100Se,
    };
  }, [parent, N, m, tau, seed]);

  // Shared x-range based on union of block-max and exceedance ranges
  const yMaxBlock = useMemo(() => {
    if (!fits.gevFit) return 1;
    const t = fits.gevFit.theta;
    const peak = Math.max(...Array.from(fits.blocks).map((b) => gevPdf(b, t.xi, t.mu, t.sigma)));
    return peak * 1.3;
  }, [fits.gevFit, fits.blocks]);

  const yMaxPot = useMemo(() => {
    if (!fits.gpdFit) return 1;
    const t = fits.gpdFit.theta;
    return gpdPdf(0, t.xi, t.beta) * 1.15;
  }, [fits.gpdFit]);

  // ── Left panel: GEV fit on block maxima ─────────────────────────
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (leftWidth <= 0) return;
      const margin = { top: 28, right: 12, bottom: 36, left: 44 };
      const w = leftWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Clip x-axis to the 95th percentile of block maxima so the bulk of
      // the histogram is visible — heavy-tailed parents (Pareto, t₃) put a
      // few outliers far above the bulk, which would otherwise compress the
      // histogram against the y-axis.
      const sortedBlocks = fits.blocks.slice().sort();
      const dataMin = sortedBlocks[0] ?? 0;
      const dataMax = sortedBlocks[Math.floor(0.95 * sortedBlocks.length)] ?? 1;
      const pad = 0.1 * (dataMax - dataMin || 1);
      const xScale = d3.scaleLinear().domain([dataMin - pad, dataMax + pad]).range([0, w]);
      const yScale = d3.scaleLinear().domain([0, yMaxBlock]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');

      // Histogram
      const hist = d3.histogram().domain(xScale.domain() as [number, number]).thresholds(20);
      const bins = hist(Array.from(fits.blocks));
      for (const bin of bins) {
        const density = bin.length / (fits.blocks.length * (bin.x1! - bin.x0!));
        g.append('rect')
          .attr('x', xScale(bin.x0!))
          .attr('y', yScale(density))
          .attr('width', Math.max(0, xScale(bin.x1!) - xScale(bin.x0!) - 1))
          .attr('height', Math.max(0, h - yScale(density)))
          .style('fill', COLOR_DATA).style('opacity', 0.4);
      }

      // Fitted GEV density
      if (fits.gevFit) {
        const M = 200;
        const t = fits.gevFit.theta;
        const xs = new Float64Array(M);
        const ys = new Float64Array(M);
        const [xMin, xMax] = xScale.domain();
        for (let i = 0; i < M; i++) {
          xs[i] = xMin + ((xMax - xMin) * i) / (M - 1);
          ys[i] = gevPdf(xs[i], t.xi, t.mu, t.sigma);
        }
        const line = d3.line<number>()
          .defined((i) => Number.isFinite(ys[i]) && ys[i] > 0)
          .x((i) => xScale(xs[i]))
          .y((i) => yScale(ys[i]));
        g.append('path')
          .datum(d3.range(M)).attr('d', line)
          .style('fill', 'none').style('stroke', COLOR_BLOCK).style('stroke-width', 2.2);
      }

      svg.append('text').attr('x', leftWidth / 2).attr('y', 14)
        .style('fill', 'var(--color-text)').style('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600')
        .text(`§4 Block-maxima fit  —  B = ${fits.B} blocks`);
    },
    [leftWidth, fits, yMaxBlock],
  );

  // ── Right panel: GPD fit on exceedances ─────────────────────────
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (rightWidth <= 0) return;
      const margin = { top: 28, right: 12, bottom: 36, left: 44 };
      const w = rightWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Clip x-axis to the 95th percentile of exceedances (same rationale as
      // the left panel — heavy-tailed parents produce a few large outliers).
      const sortedExc = fits.exceedances.slice().sort();
      const dataMax = sortedExc[Math.floor(0.95 * sortedExc.length)] ?? 1;
      const xScale = d3.scaleLinear().domain([0, dataMax * 1.05]).range([0, w]);
      const yScale = d3.scaleLinear().domain([0, yMaxPot]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');

      // Histogram of exceedances
      const hist = d3.histogram().domain(xScale.domain() as [number, number]).thresholds(20);
      const bins = hist(Array.from(fits.exceedances));
      for (const bin of bins) {
        const density = bin.length / (fits.exceedances.length * (bin.x1! - bin.x0!));
        g.append('rect')
          .attr('x', xScale(bin.x0!))
          .attr('y', yScale(density))
          .attr('width', Math.max(0, xScale(bin.x1!) - xScale(bin.x0!) - 1))
          .attr('height', Math.max(0, h - yScale(density)))
          .style('fill', COLOR_DATA).style('opacity', 0.4);
      }

      // Fitted GPD density
      if (fits.gpdFit) {
        const M = 200;
        const t = fits.gpdFit.theta;
        const xs = new Float64Array(M);
        const ys = new Float64Array(M);
        for (let i = 0; i < M; i++) {
          xs[i] = (i * dataMax * 1.05) / (M - 1);
          ys[i] = gpdPdf(xs[i], t.xi, t.beta);
        }
        const line = d3.line<number>()
          .defined((i) => Number.isFinite(ys[i]) && ys[i] > 0)
          .x((i) => xScale(xs[i]))
          .y((i) => yScale(ys[i]));
        g.append('path')
          .datum(d3.range(M)).attr('d', line)
          .style('fill', 'none').style('stroke', COLOR_POT).style('stroke-width', 2.2);
      }

      svg.append('text').attr('x', rightWidth / 2).attr('y', 14)
        .style('fill', 'var(--color-text)').style('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600')
        .text(`§5 POT fit  —  N_u = ${fits.Nu} exceedances above u = ${fmt(fits.u, 2)}`);
    },
    [rightWidth, fits, yMaxPot],
  );

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3">
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg
          role="img"
          aria-label={`Left panel: GEV density fitted to B = ${fits.B} block maxima of size m = ${m} from a parent ${PARENT_LABELS[parent]} sample of size N = ${N}.`}
          ref={leftRef} width={leftWidth} height={HEIGHT}
        />
        <svg
          role="img"
          aria-label={`Right panel: GPD density fitted to N_u = ${fits.Nu} exceedances above the empirical ${(tau * 100).toFixed(1)}% quantile of the same parent sample.`}
          ref={rightRef} width={rightWidth} height={HEIGHT}
        />
      </div>

      <div className={`mt-3 grid grid-cols-1 ${isStacked ? '' : 'sm:grid-cols-2'} gap-3 text-xs`}>
        <div className="rounded border p-2" style={{ borderColor: COLOR_BLOCK + '55' }}>
          <div className="font-medium" style={{ color: COLOR_BLOCK }}>§4 Block-maxima readout</div>
          {fits.gevFit ? (
            <>
              <div className="font-mono mt-1">ξ̂ = {fmt(fits.gevFit.theta.xi, 3)} ± {fmt(Math.sqrt(Math.max(0, fits.gevFit.cov[0])), 3)}</div>
              <div className="font-mono">σ̂ = {fmt(fits.gevFit.theta.sigma, 2)} ± {fmt(Math.sqrt(Math.max(0, fits.gevFit.cov[8])), 2)}</div>
              <div className="font-mono">x_100 = {fmt(fits.gevReturn100, 2)} ± {fmt(fits.gevReturn100Se, 2)}</div>
              <div className="text-[var(--color-text-secondary)] mt-1">Fit observations: B = {fits.B} (= N/m)</div>
            </>
          ) : (
            <div className="text-[var(--color-text-secondary)] italic mt-1">Too few blocks (B &lt; 5) — increase N or decrease m</div>
          )}
        </div>

        <div className="rounded border p-2" style={{ borderColor: COLOR_POT + '55' }}>
          <div className="font-medium" style={{ color: COLOR_POT }}>§5 Peaks-over-threshold readout</div>
          {fits.gpdFit ? (
            <>
              <div className="font-mono mt-1">ξ̂ = {fmt(fits.gpdFit.theta.xi, 3)} ± {fmt(Math.sqrt(Math.max(0, fits.gpdFit.cov[0])), 3)}</div>
              <div className="font-mono">β̂ = {fmt(fits.gpdFit.theta.beta, 3)} ± {fmt(Math.sqrt(Math.max(0, fits.gpdFit.cov[3])), 3)}</div>
              <div className="font-mono">x_100 (= VaR_{(1 - 1 / (100 * m)).toFixed(4)}) = {fmt(fits.potVar100, 2)} ± {fmt(fits.potVar100Se, 2)}</div>
              <div className="text-[var(--color-text-secondary)] mt-1">
                Fit observations: N_u = {fits.Nu} ({(fits.Nu / fits.B).toFixed(1)}× the block-fit count)
              </div>
            </>
          ) : (
            <div className="text-[var(--color-text-secondary)] italic mt-1">Too few exceedances (N_u &lt; 30) — lower τ or increase N</div>
          )}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span>Parent</span>
          <select
            value={parent}
            onChange={(e) => setParent(e.target.value as 'normal' | 'pareto2' | 't3')}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-[var(--color-text)]"
          >
            {(Object.keys(PARENT_LABELS) as ('normal' | 'pareto2' | 't3')[]).map((k) => (
              <option key={k} value={k}>{PARENT_LABELS[k]}</option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span>N</span>
          <input
            type="range" min={N_MIN} max={N_MAX} step={N_STEP}
            value={N} onChange={(e) => setN(parseInt(e.target.value, 10))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-16 text-right">{N.toLocaleString()}</span>
        </label>

        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span style={{ color: COLOR_BLOCK }}>m (block)</span>
          <input
            type="range" min={M_MIN} max={M_MAX} step={M_STEP}
            value={m} onChange={(e) => setM(parseInt(e.target.value, 10))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{m}</span>
        </label>

        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span style={{ color: COLOR_POT }}>τ (threshold)</span>
          <input
            type="range" min={TAU_MIN} max={TAU_MAX} step={TAU_STEP}
            value={tau} onChange={(e) => setTau(parseFloat(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{tau.toFixed(3)}</span>
        </label>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
        >
          Resample
        </button>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Same N raw observations on both sides. Block-maxima (purple, §4) fits a GEV to B = N/m block maxima — discarding everything except the per-block max. Peaks-over-threshold (teal, §5) keeps every observation above the empirical τ-quantile and fits a GPD to the exceedances. The data-efficiency ratio N_u / B is the headline number — typically 5–10× more usable data for tail estimation at the same N. The return level x_100 (left, GEV's 99th-percentile of block maxima) and the POT VaR_0.99 (right, the 99th-percentile of X) are different objects, but both extrapolate to roughly the same physical "1-in-100-block" event when the block size and threshold are matched. Standard errors at the same N are visibly tighter in the POT readout, reflecting the larger fit-observation count.
      </p>
    </div>
  );
}

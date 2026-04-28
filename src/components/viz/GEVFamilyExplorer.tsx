import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  blockMaxima,
  gevCdf,
  gevMle,
  gevPdf,
  mulberry32,
  sampleParent,
  trueXi,
  type ParentPreset,
} from '../../data/extreme-value-theory';

// =============================================================================
// GEVFamilyExplorer — embedded after Theorem 2 in §2.4.
//
// Two-panel display of the GEV density (left) and CDF (right) on a shared
// x-axis. Shape ξ slider colours the curve by regime: green (ξ < 0,
// reverse-Weibull, bounded support), blue (ξ ≈ 0, Gumbel, unbounded), red
// (ξ > 0, Fréchet, polynomial right tail). Support boundaries drawn as
// vertical dashed lines for ξ ≠ 0.
//
// Parent-distribution preset dropdown drives a live MLE fit:
//   - 'normal'  → Gumbel domain (truth ξ = 0)
//   - 'pareto2' → Fréchet domain (truth ξ = 1/2)
//   - 'uniform' → reverse-Weibull domain (truth ξ = -1)
//   - 'custom'  → no fit; sliders directly control the density.
//
// In preset mode: simulate N = 50,000 raw observations, group into B = 1000
// blocks of m = 50, fit GEV via Nelder-Mead MLE, overlay fitted density on
// empirical histogram. Sliders auto-snap to (ξ̂, μ̂, σ̂); user can drag away
// to compare. Standard errors come from the inverse Hessian (delta-method).
//
// Data module: src/data/extreme-value-theory.ts. SciPy sign-convention is
// sidestepped because the densities are written in the standard convention
// directly.
// =============================================================================

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 340;
const SM_BREAKPOINT = 768;

// Slider ranges per brief §11.1
const XI_MIN = -1;
const XI_MAX = 1;
const XI_STEP = 0.01;
const MU_MIN = -2;
const MU_MAX = 2;
const MU_STEP = 0.02;
const SIGMA_MIN = 0.2;
const SIGMA_MAX = 3;
const SIGMA_STEP = 0.02;

// Block-maxima parameters for preset MLE
const PRESET_N_RAW = 50_000;
const PRESET_BLOCK_SIZE = 50;

// Trichotomy colors — matches notebook Cell 2 palette
const COLOR_WEIBULL = '#16A34A'; // green   ξ < 0
const COLOR_GUMBEL = '#2563EB';  // blue    ξ = 0
const COLOR_FRECHET = '#DC2626'; // red     ξ > 0

const fmt = (x: number, digits = 3) => x.toFixed(digits);

function regimeColor(xi: number): string {
  if (xi < -0.02) return COLOR_WEIBULL;
  if (xi > 0.02) return COLOR_FRECHET;
  return COLOR_GUMBEL;
}

function regimeLabel(xi: number): string {
  if (xi < -0.02) return 'reverse-Weibull';
  if (xi > 0.02) return 'Fréchet';
  return 'Gumbel';
}

const PRESET_LABELS: Record<'custom' | ParentPreset, string> = {
  custom: 'Custom parameters',
  normal: 'Normal → Gumbel (ξ ≈ 0)',
  pareto2: 'Pareto(α=2) → Fréchet (ξ ≈ 0.5)',
  uniform: 'Uniform(0,1) → reverse-Weibull (ξ ≈ -1)',
  t3: 'Student-t₃ → Fréchet (ξ ≈ 0.33)',
};

interface FitData {
  blocks: Float64Array;
  xiHat: number;
  muHat: number;
  sigmaHat: number;
  seXi: number;
  seMu: number;
  seSigma: number;
  truthXi: number;
}

// ── Component ────────────────────────────────────────────────────────

export default function GEVFamilyExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [preset, setPreset] = useState<'custom' | ParentPreset>('custom');
  const [seed, setSeed] = useState(42);
  const [xi, setXi] = useState(0);
  const [mu, setMu] = useState(0);
  const [sigma, setSigma] = useState(1);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const leftWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.5);
  const rightWidth = isStacked ? containerWidth : containerWidth - leftWidth;

  // ── Live MLE fit on preset/seed change ──────────────────────────
  const fitData: FitData | null = useMemo(() => {
    if (preset === 'custom') return null;
    const rng = mulberry32(seed);
    const samples = sampleParent(preset, PRESET_N_RAW, rng);
    const blocks = blockMaxima(samples, PRESET_BLOCK_SIZE);
    const result = gevMle(blocks);
    const seXi = Math.sqrt(Math.max(0, result.cov[0]));
    const seMu = Math.sqrt(Math.max(0, result.cov[4]));
    const seSigma = Math.sqrt(Math.max(0, result.cov[8]));
    return {
      blocks,
      xiHat: result.theta.xi,
      muHat: result.theta.mu,
      sigmaHat: result.theta.sigma,
      seXi,
      seMu,
      seSigma,
      truthXi: trueXi(preset),
    };
  }, [preset, seed]);

  // Snap sliders to fitted values when fit data arrives. Sliders may saturate
  // at their bounds (Pareto's μ̂ ≈ 30 exceeds MU_MAX = 2) — that's expected;
  // the preset overlay always uses the FITTED θ̂ for its curve regardless of
  // slider position.
  useEffect(() => {
    if (fitData) {
      setXi(Math.max(XI_MIN, Math.min(XI_MAX, fitData.xiHat)));
      setMu(Math.max(MU_MIN, Math.min(MU_MAX, fitData.muHat)));
      setSigma(Math.max(SIGMA_MIN, Math.min(SIGMA_MAX, fitData.sigmaHat)));
    }
  }, [fitData]);

  // ── x-axis range: covers density + (in preset mode) fitted-θ histogram ──
  const xRange = useMemo<[number, number]>(() => {
    if (fitData) {
      const dataMin = d3.min(fitData.blocks) ?? 0;
      const dataMax = d3.max(fitData.blocks) ?? 1;
      const pad = 0.1 * (dataMax - dataMin || 1);
      return [dataMin - pad, dataMax + pad];
    }
    // Custom mode: x-range driven by current slider ξ regime.
    if (xi > 0.02) return [-2, 5];   // Fréchet has bounded left
    if (xi < -0.02) return [-3, 3];  // reverse-Weibull bounded right
    return [-3, 5];                  // Gumbel
  }, [fitData, xi]);

  // ── Density curve for the manual θ (slider values) ──────────────
  const densityCurve = useMemo(() => {
    const M = 250;
    const [xMin, xMax] = xRange;
    const xs = new Float64Array(M);
    const ys = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      const x = xMin + ((xMax - xMin) * i) / (M - 1);
      xs[i] = x;
      ys[i] = gevPdf(x, xi, mu, sigma);
    }
    return { xs, ys };
  }, [xi, mu, sigma, xRange]);

  // ── Density curve for the fitted θ̂ (preset mode only) ──────────
  const fittedDensityCurve = useMemo(() => {
    if (!fitData) return null;
    const M = 250;
    const [xMin, xMax] = xRange;
    const xs = new Float64Array(M);
    const ys = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      const x = xMin + ((xMax - xMin) * i) / (M - 1);
      xs[i] = x;
      ys[i] = gevPdf(x, fitData.xiHat, fitData.muHat, fitData.sigmaHat);
    }
    return { xs, ys };
  }, [fitData, xRange]);

  // ── Histogram bins for fitted-data overlay ──────────────────────
  const histogramBins = useMemo(() => {
    if (!fitData) return null;
    const data = Array.from(fitData.blocks);
    const [xMin, xMax] = xRange;
    const generator = d3.histogram().domain([xMin, xMax]).thresholds(30);
    const bins = generator(data);
    return bins.map((bin) => ({
      x0: bin.x0!,
      x1: bin.x1!,
      density: bin.length / (data.length * (bin.x1! - bin.x0!)),
    }));
  }, [fitData, xRange]);

  // ── Empirical CDF for the fitted block maxima ───────────────────
  const empiricalCdf = useMemo(() => {
    if (!fitData) return null;
    const sorted = Float64Array.from(fitData.blocks).sort();
    const xs = new Float64Array(sorted.length);
    const ys = new Float64Array(sorted.length);
    for (let i = 0; i < sorted.length; i++) {
      xs[i] = sorted[i];
      ys[i] = (i + 1) / sorted.length;
    }
    return { xs, ys };
  }, [fitData]);

  // ── Support boundary for current θ ──────────────────────────────
  const supportBoundary = useMemo(() => {
    if (Math.abs(xi) < 0.02) return null;
    return mu - sigma / xi;
  }, [xi, mu, sigma]);

  const yMaxDensity = useMemo(() => {
    let m = d3.max(densityCurve.ys) ?? 1;
    if (fittedDensityCurve) m = Math.max(m, d3.max(fittedDensityCurve.ys) ?? 0);
    if (histogramBins) {
      const histMax = d3.max(histogramBins, (b) => b.density) ?? 0;
      m = Math.max(m, histMax);
    }
    return m * 1.15;
  }, [densityCurve, fittedDensityCurve, histogramBins]);

  const curColor = regimeColor(xi);

  // ── Left panel: PDF + (preset) histogram + fitted density ───────
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (leftWidth <= 0) return;

      const margin = { top: 28, right: 12, bottom: 36, left: 44 };
      const w = leftWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain(xRange).range([0, w]);
      const yScale = d3.scaleLinear().domain([0, yMaxDensity]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');

      g.append('text')
        .attr('x', w / 2).attr('y', h + 30)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .text('x');
      g.append('text')
        .attr('x', -34).attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .attr('transform', `rotate(-90,-34,${h / 2})`)
        .text('density');

      // Histogram (preset mode)
      if (histogramBins) {
        for (const bin of histogramBins) {
          const x0 = xScale(bin.x0);
          const x1 = xScale(bin.x1);
          g.append('rect')
            .attr('x', x0).attr('y', yScale(bin.density))
            .attr('width', Math.max(0, x1 - x0 - 1))
            .attr('height', h - yScale(bin.density))
            .style('fill', 'var(--color-text-secondary)')
            .style('opacity', 0.18);
        }
      }

      // Support boundary
      if (supportBoundary !== null) {
        const sx = xScale(supportBoundary);
        if (sx >= 0 && sx <= w) {
          g.append('line')
            .attr('x1', sx).attr('x2', sx)
            .attr('y1', 0).attr('y2', h)
            .style('stroke', curColor).style('stroke-dasharray', '4 3')
            .style('stroke-width', 1).style('opacity', 0.55);
        }
      }

      // Fitted density (dashed) when in preset mode
      if (fittedDensityCurve) {
        const line = d3.line<number>()
          .defined((i) => Number.isFinite(fittedDensityCurve.ys[i]))
          .x((i) => xScale(fittedDensityCurve.xs[i]))
          .y((i) => yScale(fittedDensityCurve.ys[i]));
        g.append('path')
          .datum(d3.range(fittedDensityCurve.xs.length))
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', regimeColor(fitData!.xiHat))
          .style('stroke-width', 2.2)
          .style('stroke-dasharray', '6 3');
      }

      // Manual density (solid) — always rendered, in current ξ regime color
      const manualLine = d3.line<number>()
        .defined((i) => Number.isFinite(densityCurve.ys[i]) && densityCurve.ys[i] > 0)
        .x((i) => xScale(densityCurve.xs[i]))
        .y((i) => yScale(densityCurve.ys[i]));
      g.append('path')
        .datum(d3.range(densityCurve.xs.length))
        .attr('d', manualLine)
        .style('fill', 'none')
        .style('stroke', curColor)
        .style('stroke-width', 2.2);

      // Title
      svg.append('text')
        .attr('x', leftWidth / 2).attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text(`GEV density — ${regimeLabel(xi)} (ξ = ${fmt(xi, 2)})`);
    },
    [leftWidth, densityCurve, fittedDensityCurve, histogramBins, xRange, yMaxDensity, supportBoundary, curColor, fitData, xi],
  );

  // ── Right panel: CDF + (preset) empirical CDF ───────────────────
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (rightWidth <= 0) return;

      const margin = { top: 28, right: 12, bottom: 36, left: 44 };
      const w = rightWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain(xRange).range([0, w]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');

      g.append('text')
        .attr('x', w / 2).attr('y', h + 30)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .text('x');
      g.append('text')
        .attr('x', -34).attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .attr('transform', `rotate(-90,-34,${h / 2})`)
        .text('CDF');

      // Empirical CDF (step function) in preset mode
      if (empiricalCdf) {
        const line = d3.line<number>()
          .x((i) => xScale(empiricalCdf.xs[i]))
          .y((i) => yScale(empiricalCdf.ys[i]))
          .curve(d3.curveStepAfter);
        g.append('path')
          .datum(d3.range(empiricalCdf.xs.length))
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', 'var(--color-text-secondary)')
          .style('stroke-width', 1.2)
          .style('opacity', 0.75);
      }

      // Support boundary
      if (supportBoundary !== null) {
        const sx = xScale(supportBoundary);
        if (sx >= 0 && sx <= w) {
          g.append('line')
            .attr('x1', sx).attr('x2', sx)
            .attr('y1', 0).attr('y2', h)
            .style('stroke', curColor).style('stroke-dasharray', '4 3')
            .style('stroke-width', 1).style('opacity', 0.55);
        }
      }

      // Manual CDF
      const M = 250;
      const xs = new Float64Array(M);
      const ys = new Float64Array(M);
      const [xMin, xMax] = xRange;
      for (let i = 0; i < M; i++) {
        const x = xMin + ((xMax - xMin) * i) / (M - 1);
        xs[i] = x;
        ys[i] = gevCdf(x, xi, mu, sigma);
      }
      const manualLine = d3.line<number>()
        .x((i) => xScale(xs[i]))
        .y((i) => yScale(ys[i]));
      g.append('path')
        .datum(d3.range(M))
        .attr('d', manualLine)
        .style('fill', 'none')
        .style('stroke', curColor)
        .style('stroke-width', 2.2);

      // Fitted CDF (dashed) in preset mode
      if (fitData) {
        const fittedYs = new Float64Array(M);
        for (let i = 0; i < M; i++) {
          fittedYs[i] = gevCdf(xs[i], fitData.xiHat, fitData.muHat, fitData.sigmaHat);
        }
        const fittedLine = d3.line<number>()
          .x((i) => xScale(xs[i]))
          .y((i) => yScale(fittedYs[i]));
        g.append('path')
          .datum(d3.range(M))
          .attr('d', fittedLine)
          .style('fill', 'none')
          .style('stroke', regimeColor(fitData.xiHat))
          .style('stroke-width', 2.2)
          .style('stroke-dasharray', '6 3');
      }

      svg.append('text')
        .attr('x', rightWidth / 2).attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text('GEV CDF');
    },
    [rightWidth, xi, mu, sigma, xRange, empiricalCdf, fitData, supportBoundary, curColor],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3"
    >
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg
          role="img"
          aria-label={`Left panel: GEV probability density (${regimeLabel(xi)}, ξ = ${fmt(xi, 2)}) on the current parameter triple. ${fitData ? 'Empirical histogram of B = 1000 block maxima from the selected parent (gray bars) overlaid with the fitted GEV density (dashed).' : 'Slider-controlled density only.'}`}
          ref={leftRef}
          width={leftWidth}
          height={HEIGHT}
        />
        <svg
          role="img"
          aria-label={`Right panel: GEV cumulative distribution function. ${fitData ? 'Empirical step CDF of block maxima (gray) overlaid with fitted GEV CDF (dashed).' : 'Slider-controlled CDF only.'}`}
          ref={rightRef}
          width={rightWidth}
          height={HEIGHT}
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-medium">Parent</span>
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value as 'custom' | ParentPreset)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-[var(--color-text)]"
          >
            {(Object.keys(PRESET_LABELS) as ('custom' | ParentPreset)[]).map((k) => (
              <option key={k} value={k}>{PRESET_LABELS[k]}</option>
            ))}
          </select>
        </label>

        {preset !== 'custom' && (
          <button
            type="button"
            onClick={() => setSeed((s) => s + 1)}
            className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
          >
            Resample
          </button>
        )}
      </div>

      <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono w-4" style={{ color: curColor }}>ξ</span>
          <input
            type="range" min={XI_MIN} max={XI_MAX} step={XI_STEP}
            value={xi}
            onChange={(e) => setXi(parseFloat(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{fmt(xi, 2)}</span>
        </label>

        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono w-4">μ</span>
          <input
            type="range" min={MU_MIN} max={MU_MAX} step={MU_STEP}
            value={mu}
            onChange={(e) => setMu(parseFloat(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{fmt(mu, 2)}</span>
        </label>

        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span className="font-mono w-4">σ</span>
          <input
            type="range" min={SIGMA_MIN} max={SIGMA_MAX} step={SIGMA_STEP}
            value={sigma}
            onChange={(e) => setSigma(parseFloat(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{fmt(sigma, 2)}</span>
        </label>
      </div>

      {fitData && (
        <div className="mt-3 rounded border border-[var(--color-border)] bg-[var(--color-bg)] p-2 text-xs text-[var(--color-text-secondary)]">
          <span className="font-medium text-[var(--color-text)]">MLE on B = 1000 blocks of m = 50:</span>
          {' '}
          <span style={{ color: regimeColor(fitData.xiHat) }} className="font-mono">
            ξ̂ = {fmt(fitData.xiHat, 3)} ± {fmt(fitData.seXi, 3)}
          </span>
          {', '}
          <span className="font-mono">μ̂ = {fmt(fitData.muHat, 2)} ± {fmt(fitData.seMu, 2)}</span>
          {', '}
          <span className="font-mono">σ̂ = {fmt(fitData.sigmaHat, 2)} ± {fmt(fitData.seSigma, 2)}</span>
          {Number.isFinite(fitData.truthXi) && (
            <span className="ml-2">(truth ξ = {fmt(fitData.truthXi, 2)})</span>
          )}
          {(fitData.muHat < MU_MIN || fitData.muHat > MU_MAX || fitData.sigmaHat < SIGMA_MIN || fitData.sigmaHat > SIGMA_MAX) && (
            <span className="ml-2 italic">(some fitted parameters fall outside the slider ranges; the dashed curve uses the unclipped fitted θ̂)</span>
          )}
        </div>
      )}

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Drag the ξ slider through the three regime zones — green (reverse-Weibull, bounded support), blue (Gumbel, unbounded), red (Fréchet, polynomial right tail). The vertical dashed line marks the support boundary at $μ - σ/ξ$ when ξ ≠ 0. Selecting a parent preset simulates N = {PRESET_N_RAW.toLocaleString()} raw observations, forms B = 1000 block maxima of size m = 50, and fits a GEV via maximum likelihood; the dashed curve is the fitted density (drawn at θ̂, not at the slider values), and the gray histogram is the empirical block-maxima distribution.
      </p>
    </div>
  );
}

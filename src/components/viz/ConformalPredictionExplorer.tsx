import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  fitPredictRidge,
  mulberry32,
  splitConformalInterval,
  synthHeteroscedastic,
} from './shared/nonparametric-ml';

// =============================================================================
// ConformalPredictionExplorer — embedded in §3 after Theorem 1.
//
// Three vertically-stacked panels at any width, side-by-side at desktop:
//   Top     : single representative dataset with ridge band + test in/out colors
//   Middle  : histogram of per-trial coverage proportions across T trials, with
//             the [1 − α, 1 − α + 1/(n_cal+1)] reference band overlaid
//   Bottom  : running cumulative mean coverage over the T trials, log x-axis,
//             converging to the predicted band as t → T
//
// Each "trial" runs a fresh split-conformal procedure on a fresh synthetic
// heteroscedastic dataset (n_train fixed at 300, n_cal slider-controlled,
// n_test = 20 per trial → coverage proportion in [0, 1]). T = 500 trials per
// recompute.
// =============================================================================

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 280;
const SM_BREAKPOINT = 900;
const N_TRAIN = 300;
const N_TEST_PER_TRIAL = 20;
const T = 500;
const N_CAL_OPTIONS = [20, 50, 100, 200, 500, 1000] as const;
const ALPHA_MIN = 0.01;
const ALPHA_MAX = 0.30;
const ALPHA_STEP = 0.01;
const HIST_MIN = 0.7;
const HIST_MAX = 1.0;
const HIST_BINS = 25;

// Match notebook colors so the live viz aligns visually with the static fallback
const BLUE = '#2563EB';
const GREEN = '#059669';
const RED = '#DC2626';
const AMBER = '#D97706';
const SLATE = '#475569';
const LIGHT_BLUE = '#DBEAFE';

const fmt = (x: number, digits = 4) => x.toFixed(digits);

// ── Component ────────────────────────────────────────────────────────

export default function ConformalPredictionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [alpha, setAlpha] = useState(0.1);
  const [nCal, setNCal] = useState(200);
  const [seed, setSeed] = useState(42);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = containerWidth || 0;

  // ── Heavy computation — T trials of split conformal ──────────────
  const sim = useMemo(() => {
    const rng = mulberry32(seed);
    const total = N_TRAIN + nCal + N_TEST_PER_TRIAL;

    // Demo dataset (trial 0) for the top panel
    const { x: xDemo, y: yDemo } = synthHeteroscedastic(total, rng);
    const demoXTrain = xDemo.subarray(0, N_TRAIN);
    const demoYTrain = yDemo.subarray(0, N_TRAIN);
    const demoXCal = xDemo.subarray(N_TRAIN, N_TRAIN + nCal);
    const demoYCal = yDemo.subarray(N_TRAIN, N_TRAIN + nCal);
    const demoXTest = xDemo.subarray(N_TRAIN + nCal);
    const demoYTest = yDemo.subarray(N_TRAIN + nCal);
    const demoResult = splitConformalInterval(
      demoXTrain,
      demoYTrain,
      demoXCal,
      demoYCal,
      demoXTest,
      alpha,
      fitPredictRidge,
    );
    const xGrid = Float64Array.from({ length: 200 }, (_, i) => -2.2 + (4.4 * i) / 199);
    const yGridPred = fitPredictRidge(demoXTrain, demoYTrain, xGrid);
    const demoTestInBand = new Uint8Array(demoYTest.length);
    for (let i = 0; i < demoYTest.length; i++) {
      demoTestInBand[i] =
        demoYTest[i] >= demoResult.lower[i] && demoYTest[i] <= demoResult.upper[i] ? 1 : 0;
    }

    // T trials
    const perTrial = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      const { x, y } = synthHeteroscedastic(total, rng);
      const xTr = x.subarray(0, N_TRAIN);
      const yTr = y.subarray(0, N_TRAIN);
      const xCa = x.subarray(N_TRAIN, N_TRAIN + nCal);
      const yCa = y.subarray(N_TRAIN, N_TRAIN + nCal);
      const xTe = x.subarray(N_TRAIN + nCal);
      const yTe = y.subarray(N_TRAIN + nCal);
      const { lower, upper } = splitConformalInterval(
        xTr,
        yTr,
        xCa,
        yCa,
        xTe,
        alpha,
        fitPredictRidge,
      );
      let cov = 0;
      for (let k = 0; k < N_TEST_PER_TRIAL; k++) {
        if (yTe[k] >= lower[k] && yTe[k] <= upper[k]) cov++;
      }
      perTrial[t] = cov / N_TEST_PER_TRIAL;
    }

    // Running cumulative mean
    const runMean = new Float64Array(T);
    let sum = 0;
    for (let t = 0; t < T; t++) {
      sum += perTrial[t];
      runMean[t] = sum / (t + 1);
    }

    // Histogram bins (fixed across [0.7, 1.0] for visual stability)
    const binCounts = new Array(HIST_BINS).fill(0) as number[];
    const binWidth = (HIST_MAX - HIST_MIN) / HIST_BINS;
    for (let t = 0; t < T; t++) {
      const v = perTrial[t];
      if (v < HIST_MIN || v > HIST_MAX) continue;
      const idx = Math.min(HIST_BINS - 1, Math.floor((v - HIST_MIN) / binWidth));
      binCounts[idx]++;
    }
    const meanCoverage = perTrial.reduce((a, b) => a + b, 0) / T;
    const lowerBound = 1 - alpha;
    const upperBound = 1 - alpha + 1 / (nCal + 1);

    return {
      demo: {
        xTrain: demoXTrain,
        yTrain: demoYTrain,
        xCal: demoXCal,
        yCal: demoYCal,
        xTest: demoXTest,
        yTest: demoYTest,
        xGrid,
        yGridPred,
        qHat: demoResult.qHat,
        inBand: demoTestInBand,
      },
      perTrial,
      runMean,
      binCounts,
      meanCoverage,
      lowerBound,
      upperBound,
    };
  }, [alpha, nCal, seed]);

  // ── Top panel: representative dataset + ridge band ───────────────
  const topRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const margin = { top: 24, right: 16, bottom: 36, left: 44 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-2.2, 2.2]).range([0, w]);
      const yScale = d3.scaleLinear().domain([-3.5, 3.5]).range([h, 0]);

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
        .attr('x', w / 2)
        .attr('y', h + 30)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('x');
      g.append('text')
        .attr('x', -28)
        .attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .attr('transform', `rotate(-90,-28,${h / 2})`)
        .text('y');

      // Prediction band μ̂(x) ± q̂
      const area = d3.area<number>()
        .x((_, i) => xScale(sim.demo.xGrid[i]))
        .y0((_, i) => yScale(sim.demo.yGridPred[i] - sim.demo.qHat))
        .y1((_, i) => yScale(sim.demo.yGridPred[i] + sim.demo.qHat));
      g.append('path')
        .datum(d3.range(sim.demo.xGrid.length))
        .attr('d', area)
        .style('fill', GREEN)
        .style('opacity', 0.18);

      // Ridge predictor curve
      const line = d3.line<number>()
        .x((_, i) => xScale(sim.demo.xGrid[i]))
        .y((_, i) => yScale(sim.demo.yGridPred[i]));
      g.append('path')
        .datum(d3.range(sim.demo.xGrid.length))
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', BLUE)
        .style('stroke-width', 1.8);

      // Training points (down-sampled for clarity if n_train is large)
      const trainStep = Math.max(1, Math.floor(N_TRAIN / 120));
      for (let i = 0; i < N_TRAIN; i += trainStep) {
        g.append('circle')
          .attr('cx', xScale(sim.demo.xTrain[i]))
          .attr('cy', yScale(sim.demo.yTrain[i]))
          .attr('r', 2)
          .style('fill', SLATE)
          .style('opacity', 0.5);
      }

      // Test points colored by in/out band
      for (let i = 0; i < sim.demo.xTest.length; i++) {
        const inside = sim.demo.inBand[i] === 1;
        g.append('circle')
          .attr('cx', xScale(sim.demo.xTest[i]))
          .attr('cy', yScale(sim.demo.yTest[i]))
          .attr('r', 4)
          .style('fill', inside ? GREEN : RED)
          .style('stroke', '#fff')
          .style('stroke-width', 1);
      }

      // Title with q̂ value
      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`Prediction band: μ̂(x) ± q̂ = ${fmt(sim.demo.qHat, 3)}`);
    },
    [panelWidth, sim],
  );

  // ── Middle panel: histogram of per-trial coverage with bound band ──
  const middleRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const margin = { top: 24, right: 16, bottom: 36, left: 44 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([HIST_MIN, HIST_MAX]).range([0, w]);
      const yMaxCount = Math.max(1, d3.max(sim.binCounts) || 1);
      const yScale = d3.scaleLinear().domain([0, yMaxCount * 1.1]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('per-trial coverage');
      g.append('text')
        .attr('x', -28)
        .attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .attr('transform', `rotate(-90,-28,${h / 2})`)
        .text('count');

      // Theoretical bound band
      const bandLeft = xScale(sim.lowerBound);
      const bandRight = xScale(Math.min(HIST_MAX, sim.upperBound));
      if (bandRight > bandLeft) {
        g.append('rect')
          .attr('x', bandLeft)
          .attr('y', 0)
          .attr('width', bandRight - bandLeft)
          .attr('height', h)
          .style('fill', AMBER)
          .style('opacity', 0.10);
      }

      // Histogram bars
      const binWidth = (HIST_MAX - HIST_MIN) / HIST_BINS;
      for (let i = 0; i < HIST_BINS; i++) {
        const x0 = HIST_MIN + i * binWidth;
        g.append('rect')
          .attr('x', xScale(x0))
          .attr('y', yScale(sim.binCounts[i]))
          .attr('width', xScale(x0 + binWidth) - xScale(x0) - 1)
          .attr('height', h - yScale(sim.binCounts[i]))
          .style('fill', LIGHT_BLUE)
          .style('stroke', BLUE)
          .style('stroke-width', 0.8);
      }

      // Lower bound (red dashed)
      g.append('line')
        .attr('x1', xScale(sim.lowerBound))
        .attr('x2', xScale(sim.lowerBound))
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', RED)
        .style('stroke-width', 1.8)
        .style('stroke-dasharray', '5,3');

      // Upper bound (amber dotted) — only if in plot range
      if (sim.upperBound <= HIST_MAX) {
        g.append('line')
          .attr('x1', xScale(sim.upperBound))
          .attr('x2', xScale(sim.upperBound))
          .attr('y1', 0)
          .attr('y2', h)
          .style('stroke', AMBER)
          .style('stroke-width', 1.5)
          .style('stroke-dasharray', '2,3');
      }

      // Mean of trials (green solid line)
      g.append('line')
        .attr('x1', xScale(Math.max(HIST_MIN, Math.min(HIST_MAX, sim.meanCoverage))))
        .attr('x2', xScale(Math.max(HIST_MIN, Math.min(HIST_MAX, sim.meanCoverage))))
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', GREEN)
        .style('stroke-width', 1.5);

      // Title
      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`T = ${T} trials, mean coverage = ${fmt(sim.meanCoverage, 3)}`);

      // Legend
      g.append('text')
        .attr('x', w - 4)
        .attr('y', 8)
        .style('fill', RED)
        .style('text-anchor', 'end')
        .style('font-size', '10px')
        .text(`1 − α = ${fmt(sim.lowerBound, 3)}`);
      g.append('text')
        .attr('x', w - 4)
        .attr('y', 22)
        .style('fill', AMBER)
        .style('text-anchor', 'end')
        .style('font-size', '10px')
        .text(`1 − α + 1/(n_cal+1) = ${fmt(sim.upperBound, 4)}`);
      g.append('text')
        .attr('x', w - 4)
        .attr('y', 36)
        .style('fill', GREEN)
        .style('text-anchor', 'end')
        .style('font-size', '10px')
        .text(`mean = ${fmt(sim.meanCoverage, 3)}`);
    },
    [panelWidth, sim],
  );

  // ── Bottom panel: running cumulative mean ─────────────────────────
  const bottomRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const margin = { top: 24, right: 16, bottom: 36, left: 44 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([1, T]).range([0, w]);
      // Tighter y-range than histogram to show convergence clearly
      const minY = Math.min(0.75, sim.lowerBound - 0.05);
      const maxY = Math.min(1.0, Math.max(sim.upperBound + 0.05, 0.95));
      const yScale = d3.scaleLinear().domain([minY, maxY]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(4, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('trial t (log scale)');
      g.append('text')
        .attr('x', -28)
        .attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .attr('transform', `rotate(-90,-28,${h / 2})`)
        .text('running mean');

      // Theoretical bound band
      const bandTop = yScale(Math.min(maxY, sim.upperBound));
      const bandBottom = yScale(sim.lowerBound);
      g.append('rect')
        .attr('x', 0)
        .attr('y', bandTop)
        .attr('width', w)
        .attr('height', Math.max(0, bandBottom - bandTop))
        .style('fill', AMBER)
        .style('opacity', 0.10);

      // Lower bound line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(sim.lowerBound))
        .attr('y2', yScale(sim.lowerBound))
        .style('stroke', RED)
        .style('stroke-width', 1.5)
        .style('stroke-dasharray', '5,3');

      // Upper bound line (if within range)
      if (sim.upperBound <= maxY) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', w)
          .attr('y1', yScale(sim.upperBound))
          .attr('y2', yScale(sim.upperBound))
          .style('stroke', AMBER)
          .style('stroke-width', 1.2)
          .style('stroke-dasharray', '2,3');
      }

      // Running mean line
      const line = d3.line<number>()
        .x((d) => xScale(d + 1))
        .y((d) => yScale(sim.runMean[d]))
        .curve(d3.curveMonotoneX);
      g.append('path')
        .datum(d3.range(T))
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', BLUE)
        .style('stroke-width', 1.5);

      // Title
      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`Running mean → ${fmt(sim.runMean[T - 1], 3)} (band ${fmt(sim.lowerBound, 3)}, ${fmt(sim.upperBound, 4)})`);
    },
    [panelWidth, sim],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3"
    >
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg
          role="img"
          aria-label="Top panel: heteroscedastic regression scatter with ridge predictor and split-conformal prediction band; test points colored by whether they fall inside the band"
          ref={topRef}
          width={panelWidth}
          height={HEIGHT}
        />
        <svg
          role="img"
          aria-label="Middle panel: histogram of per-trial coverage proportions over T trials, with the theoretical lower and upper bound lines overlaid"
          ref={middleRef}
          width={panelWidth}
          height={HEIGHT}
        />
        <svg
          role="img"
          aria-label="Bottom panel: running cumulative-mean coverage over T trials on a log x-axis, converging into the predicted bound band"
          ref={bottomRef}
          width={panelWidth}
          height={HEIGHT}
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span>α</span>
          <input
            type="range"
            min={ALPHA_MIN}
            max={ALPHA_MAX}
            step={ALPHA_STEP}
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{alpha.toFixed(2)}</span>
        </label>

        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span>n_cal</span>
          <select
            value={nCal}
            onChange={(e) => setNCal(parseInt(e.target.value, 10))}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-xs"
          >
            {N_CAL_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
        >
          Resample
        </button>

        <span className="text-xs text-[var(--color-text-secondary)]">
          T = {T} trials · n_train = {N_TRAIN} · n_test/trial = {N_TEST_PER_TRIAL}
        </span>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Drag α to retarget coverage; switch n_cal to see the upper-bound strip narrow as 1/(n_cal+1) shrinks. The middle-panel histogram concentrates inside the orange [1−α, 1−α+1/(n_cal+1)] band predicted by Theorem 1; the bottom-panel running mean converges into the same band as t → T. The top panel's red test points are the rare misses (~α fraction).
      </p>
    </div>
  );
}

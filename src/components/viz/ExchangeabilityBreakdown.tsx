import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  fitPredictRidge,
  gaussianSampler,
  mulberry32,
  splitConformalInterval,
  weightedSplitConformal,
} from './shared/nonparametric-ml';

// =============================================================================
// ExchangeabilityBreakdown — embedded in §8 after Remark 3 (Covariate Shift).
//
// Two-panel demonstration of split conformal's coverage loss under known
// covariate shift, and the Tibshirani-Barber-Candès-Ramdas 2019 importance-
// weighted recovery:
//
//   Top panel    : a single representative dataset at the user's chosen shift
//                  magnitude Δ = μ_te − μ_tr. Above the scatter, two histograms
//                  of x: blue = train (centered at 0), green = test (shifted by
//                  Δ). Below the histograms, both prediction bands overlaid.
//
//   Bottom panel : marginal test-coverage as a function of Δ on a precomputed
//                  21-point grid {0, 0.1, ..., 2.0}, T = 200 trials per grid
//                  point. Naive curve degrades as Δ grows; weighted curve
//                  holds at the 1−α target. The slider position is rendered as
//                  a vertical marker on this fixed curve.
//
// Importance weights are closed-form for the Gaussian shift:
//   w(x) = N(x; μ_te, 1) / N(x; μ_tr, 1).
// =============================================================================

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 280;
const SM_BREAKPOINT = 768;
const N_TRAIN = 200;
const N_CAL = 200;
const N_TEST = 400;
const T_PER_GRID = 200;
const GRID_POINTS = 21; // {0, 0.1, ..., 2.0}
const ALPHA_MIN = 0.05;
const ALPHA_MAX = 0.20;
const ALPHA_STEP = 0.01;
const SHIFT_MIN = 0;
const SHIFT_MAX = 2.0;
const SHIFT_STEP = 0.05;

const BLUE = '#2563EB';
const GREEN = '#059669';
const RED = '#DC2626';
const SLATE = '#475569';

const fmt = (x: number, digits = 3) => x.toFixed(digits);

type ShowMode = 'both' | 'naive' | 'weighted';

// Synth: x distribution can be shifted (centered at xCenter ± unit variance).
// Y model is the same heteroscedastic regression as the rest of the topic but
// recentered around the shifted mean so we still have a meaningful predictor.
function synthShifted(
  n: number,
  rng: () => number,
  xCenter: number,
): { x: Float64Array; y: Float64Array } {
  const gauss = gaussianSampler(rng);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const xi = xCenter + gauss(); // N(xCenter, 1)
    const sigma = 0.3 + 0.6 * Math.abs(xi);
    x[i] = xi;
    y[i] = 0.5 * xi + Math.sin(1.5 * xi) + sigma * gauss();
  }
  return { x, y };
}

// Closed-form Gaussian density ratio for the shift
function gaussianWeight(x: number, muTrain: number, muTest: number): number {
  // N(x; muTest, 1) / N(x; muTrain, 1) = exp(-0.5*(x-muTest)^2 + 0.5*(x-muTrain)^2)
  const diff = (x - muTrain) ** 2 - (x - muTest) ** 2;
  return Math.exp(0.5 * diff);
}

// ── Component ────────────────────────────────────────────────────────

export default function ExchangeabilityBreakdown() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [shift, setShift] = useState(0.5);
  const [alpha, setAlpha] = useState(0.1);
  const [showMode, setShowMode] = useState<ShowMode>('both');

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = containerWidth || 0;

  // ── Demo dataset for the top panel (depends on shift, alpha) ─────
  const demo = useMemo(() => {
    const rng = mulberry32(1234);
    const muTrain = 0;
    const muTest = shift;
    const train = synthShifted(N_TRAIN, rng, muTrain);
    const cal = synthShifted(N_CAL, rng, muTrain);
    const test = synthShifted(N_TEST, rng, muTest);

    // Common grid for displaying bands across both train and test ranges
    const xLo = Math.min(-3, -3 + shift);
    const xHi = Math.max(3, 3 + shift);
    const xGrid = Float64Array.from({ length: 200 }, (_, i) => xLo + ((xHi - xLo) * i) / 199);

    const naive = splitConformalInterval(
      train.x,
      train.y,
      cal.x,
      cal.y,
      xGrid,
      alpha,
      fitPredictRidge,
    );
    const weighted = weightedSplitConformal(
      train.x,
      train.y,
      cal.x,
      cal.y,
      xGrid,
      alpha,
      (x) => gaussianWeight(x, muTrain, muTest),
      fitPredictRidge,
    );

    // Compute coverage on test set for both procedures
    const naiveTest = splitConformalInterval(
      train.x,
      train.y,
      cal.x,
      cal.y,
      test.x,
      alpha,
      fitPredictRidge,
    );
    const weightedTest = weightedSplitConformal(
      train.x,
      train.y,
      cal.x,
      cal.y,
      test.x,
      alpha,
      (x) => gaussianWeight(x, muTrain, muTest),
      fitPredictRidge,
    );
    let naiveCov = 0;
    let weightedCov = 0;
    for (let i = 0; i < N_TEST; i++) {
      if (test.y[i] >= naiveTest.lower[i] && test.y[i] <= naiveTest.upper[i]) naiveCov++;
      if (test.y[i] >= weightedTest.lower[i] && test.y[i] <= weightedTest.upper[i]) weightedCov++;
    }

    return {
      train,
      test,
      xGrid,
      naiveLower: naive.lower,
      naiveUpper: naive.upper,
      weightedLower: weighted.lower,
      weightedUpper: weighted.upper,
      naiveCoverage: naiveCov / N_TEST,
      weightedCoverage: weightedCov / N_TEST,
      muTrain,
      muTest,
      xLo,
      xHi,
    };
  }, [shift, alpha]);

  // ── Bottom-panel sweep (depends only on alpha; recomputed when alpha changes) ─
  const sweep = useMemo(() => {
    const rng = mulberry32(2024);
    const grid = Float64Array.from({ length: GRID_POINTS }, (_, i) => (i * SHIFT_MAX) / (GRID_POINTS - 1));
    const naiveCovGrid = new Float64Array(GRID_POINTS);
    const weightedCovGrid = new Float64Array(GRID_POINTS);
    for (let g = 0; g < GRID_POINTS; g++) {
      const muTest = grid[g];
      const muTrain = 0;
      let naiveAccum = 0;
      let weightedAccum = 0;
      for (let t = 0; t < T_PER_GRID; t++) {
        const train = synthShifted(N_TRAIN, rng, muTrain);
        const cal = synthShifted(N_CAL, rng, muTrain);
        const test = synthShifted(1, rng, muTest); // single test point per trial
        const xt = test.x.subarray(0, 1);
        const yt = test.y.subarray(0, 1);
        const nv = splitConformalInterval(
          train.x,
          train.y,
          cal.x,
          cal.y,
          xt,
          alpha,
          fitPredictRidge,
        );
        if (yt[0] >= nv.lower[0] && yt[0] <= nv.upper[0]) naiveAccum++;
        const ws = weightedSplitConformal(
          train.x,
          train.y,
          cal.x,
          cal.y,
          xt,
          alpha,
          (x) => gaussianWeight(x, muTrain, muTest),
          fitPredictRidge,
        );
        if (yt[0] >= ws.lower[0] && yt[0] <= ws.upper[0]) weightedAccum++;
      }
      naiveCovGrid[g] = naiveAccum / T_PER_GRID;
      weightedCovGrid[g] = weightedAccum / T_PER_GRID;
    }
    return { grid, naiveCovGrid, weightedCovGrid };
  }, [alpha]);

  // ── Top panel: scatter + bands + train/test histograms ───────────
  const topRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const histH = 38;
      const margin = { top: 20 + histH, right: 12, bottom: 32, left: 44 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([demo.xLo, demo.xHi]).range([0, w]);
      const yScale = d3.scaleLinear().domain([-4, 4]).range([h, 0]);

      // Histograms above the scatter
      const histG = svg
        .append('g')
        .attr('transform', `translate(${margin.left},20)`);
      const binCount = 30;
      const binEdges = d3.range(binCount + 1).map(
        (i) => demo.xLo + ((demo.xHi - demo.xLo) * i) / binCount,
      );
      const binWidth = binEdges[1] - binEdges[0];

      const trainHist = new Array(binCount).fill(0) as number[];
      const testHist = new Array(binCount).fill(0) as number[];
      for (let i = 0; i < demo.train.x.length; i++) {
        const v = demo.train.x[i];
        const idx = Math.min(
          binCount - 1,
          Math.max(0, Math.floor((v - demo.xLo) / binWidth)),
        );
        trainHist[idx]++;
      }
      for (let i = 0; i < demo.test.x.length; i++) {
        const v = demo.test.x[i];
        const idx = Math.min(
          binCount - 1,
          Math.max(0, Math.floor((v - demo.xLo) / binWidth)),
        );
        testHist[idx]++;
      }
      const histMax = Math.max(d3.max(trainHist) || 0, d3.max(testHist) || 0);
      const histScale = d3.scaleLinear().domain([0, histMax * 1.1]).range([histH, 0]);

      for (let i = 0; i < binCount; i++) {
        const xs = xScale(binEdges[i]);
        const xe = xScale(binEdges[i + 1]);
        if (trainHist[i] > 0) {
          histG.append('rect')
            .attr('x', xs)
            .attr('y', histScale(trainHist[i]))
            .attr('width', Math.max(0, xe - xs - 1))
            .attr('height', histH - histScale(trainHist[i]))
            .style('fill', BLUE)
            .style('opacity', 0.5);
        }
        if (testHist[i] > 0) {
          histG.append('rect')
            .attr('x', xs)
            .attr('y', histScale(testHist[i]))
            .attr('width', Math.max(0, xe - xs - 1))
            .attr('height', histH - histScale(testHist[i]))
            .style('fill', GREEN)
            .style('opacity', 0.5);
        }
      }

      // Scatter axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 26)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('x');

      // Bands
      if (showMode !== 'weighted') {
        const naiveArea = d3.area<number>()
          .x((_, i) => xScale(demo.xGrid[i]))
          .y0((_, i) => yScale(Math.max(-4, demo.naiveLower[i])))
          .y1((_, i) => yScale(Math.min(4, demo.naiveUpper[i])));
        g.append('path')
          .datum(d3.range(demo.xGrid.length))
          .attr('d', naiveArea)
          .style('fill', BLUE)
          .style('opacity', 0.18);
      }

      if (showMode !== 'naive') {
        // For weighted bands we need to clamp infinities (which can occur when
        // the weight at the grid point is so small the threshold is +∞)
        const weightedArea = d3.area<number>()
          .defined(
            (_, i) =>
              Number.isFinite(demo.weightedLower[i]) && Number.isFinite(demo.weightedUpper[i]),
          )
          .x((_, i) => xScale(demo.xGrid[i]))
          .y0((_, i) => yScale(Math.max(-4, demo.weightedLower[i])))
          .y1((_, i) => yScale(Math.min(4, demo.weightedUpper[i])));
        g.append('path')
          .datum(d3.range(demo.xGrid.length))
          .attr('d', weightedArea)
          .style('fill', GREEN)
          .style('opacity', 0.22);
      }

      // Training scatter (down-sampled)
      const step = Math.max(1, Math.floor(demo.train.x.length / 100));
      for (let i = 0; i < demo.train.x.length; i += step) {
        g.append('circle')
          .attr('cx', xScale(demo.train.x[i]))
          .attr('cy', yScale(Math.max(-4, Math.min(4, demo.train.y[i]))))
          .attr('r', 1.8)
          .style('fill', SLATE)
          .style('opacity', 0.5);
      }
      // Test scatter
      const stepTe = Math.max(1, Math.floor(demo.test.x.length / 60));
      for (let i = 0; i < demo.test.x.length; i += stepTe) {
        g.append('circle')
          .attr('cx', xScale(demo.test.x[i]))
          .attr('cy', yScale(Math.max(-4, Math.min(4, demo.test.y[i]))))
          .attr('r', 2.4)
          .style('fill', RED)
          .style('opacity', 0.55);
      }

      // Title
      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(
          `Δ = ${shift.toFixed(2)} · naive cov ${fmt(demo.naiveCoverage, 3)} · weighted cov ${fmt(
            demo.weightedCoverage,
            3,
          )}`,
        );
    },
    [panelWidth, demo, showMode, shift],
  );

  // ── Bottom panel: precomputed coverage curves vs shift ───────────
  const bottomRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const margin = { top: 24, right: 12, bottom: 36, left: 44 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, SHIFT_MAX]).range([0, w]);
      const yScale = d3.scaleLinear().domain([0.4, 1.0]).range([h, 0]);

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
        .text('shift Δ = μ_test − μ_train');
      g.append('text')
        .attr('x', -28)
        .attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .attr('transform', `rotate(-90,-28,${h / 2})`)
        .text('marginal coverage');

      // 1 − α reference
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(1 - alpha))
        .attr('y2', yScale(1 - alpha))
        .style('stroke', RED)
        .style('stroke-width', 1.4)
        .style('stroke-dasharray', '5,3');
      g.append('text')
        .attr('x', w - 4)
        .attr('y', yScale(1 - alpha) - 4)
        .style('fill', RED)
        .style('text-anchor', 'end')
        .style('font-size', '10px')
        .text(`1 − α = ${fmt(1 - alpha, 2)}`);

      // Naive curve
      if (showMode !== 'weighted') {
        const naiveLine = d3.line<number>()
          .x((d) => xScale(sweep.grid[d]))
          .y((d) => yScale(sweep.naiveCovGrid[d]));
        g.append('path')
          .datum(d3.range(GRID_POINTS))
          .attr('d', naiveLine)
          .style('fill', 'none')
          .style('stroke', BLUE)
          .style('stroke-width', 1.8);
        for (let i = 0; i < GRID_POINTS; i++) {
          g.append('circle')
            .attr('cx', xScale(sweep.grid[i]))
            .attr('cy', yScale(sweep.naiveCovGrid[i]))
            .attr('r', 2.5)
            .style('fill', BLUE);
        }
      }

      // Weighted curve
      if (showMode !== 'naive') {
        const wLine = d3.line<number>()
          .x((d) => xScale(sweep.grid[d]))
          .y((d) => yScale(sweep.weightedCovGrid[d]));
        g.append('path')
          .datum(d3.range(GRID_POINTS))
          .attr('d', wLine)
          .style('fill', 'none')
          .style('stroke', GREEN)
          .style('stroke-width', 1.8);
        for (let i = 0; i < GRID_POINTS; i++) {
          g.append('rect')
            .attr('x', xScale(sweep.grid[i]) - 2.5)
            .attr('y', yScale(sweep.weightedCovGrid[i]) - 2.5)
            .attr('width', 5)
            .attr('height', 5)
            .style('fill', GREEN);
        }
      }

      // Slider position indicator
      g.append('line')
        .attr('x1', xScale(shift))
        .attr('x2', xScale(shift))
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-width', 1)
        .style('stroke-dasharray', '2,3');
      g.append('text')
        .attr('x', xScale(shift))
        .attr('y', -4)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '9px')
        .text(`Δ = ${shift.toFixed(2)}`);

      // Title
      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`Coverage vs shift Δ (${T_PER_GRID} trials per grid point)`);
    },
    [panelWidth, sweep, showMode, alpha, shift],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3"
    >
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg
          role="img"
          aria-label="Top panel: heteroscedastic regression scatter at the chosen shift, with train/test x-distribution histograms above and naive vs weighted prediction bands overlaid below"
          ref={topRef}
          width={panelWidth}
          height={HEIGHT}
        />
        <svg
          role="img"
          aria-label="Bottom panel: marginal coverage as a function of covariate shift magnitude — naive curve degrades as shift grows; weighted (Tibshirani-Barber-Candès-Ramdas 2019) holds at 1-α; vertical marker shows the current slider position"
          ref={bottomRef}
          width={panelWidth}
          height={HEIGHT}
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <label className="flex items-center gap-2 text-[var(--color-text-secondary)]">
          <span>shift Δ</span>
          <input
            type="range"
            min={SHIFT_MIN}
            max={SHIFT_MAX}
            step={SHIFT_STEP}
            value={shift}
            onChange={(e) => setShift(parseFloat(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{shift.toFixed(2)}</span>
        </label>

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

        <fieldset className="flex items-center gap-3 text-[var(--color-text-secondary)]">
          <legend className="sr-only">Show</legend>
          {(['both', 'naive', 'weighted'] as const).map((m) => (
            <label key={m} className="flex items-center gap-1">
              <input
                type="radio"
                name="exchShowMode"
                checked={showMode === m}
                onChange={() => setShowMode(m)}
                className="accent-[var(--color-accent)]"
              />
              {m === 'both' ? 'Both' : m === 'naive' ? 'Naive only' : 'Weighted only'}
            </label>
          ))}
        </fieldset>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Train distribution: x ~ N(0, 1), blue histogram. Test distribution: x ~ N(Δ, 1), green histogram. Naive split conformal (blue band) was calibrated on the train distribution and progressively under-covers as Δ grows — the bottom-panel blue curve drops well below 1 − α. The TBCR 2019 importance-weighted variant (green band) uses the closed-form Gaussian density ratio w(x) = N(x; Δ, 1) / N(x; 0, 1) to recalibrate; the bottom-panel green curve holds at the 1 − α target across the entire shift range.
      </p>
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cqrInterval,
  fitPredictRidge,
  gaussianSampler,
  mulberry32,
  splitConformalInterval,
} from './shared/nonparametric-ml';

// =============================================================================
// CQRExplorer — embedded in §6 after Definition 4.
//
// Two-panel comparison of naive split conformal vs CQR on heteroscedastic data
// with a user-controlled noise-strength parameter h:
//   sigma(x) = 0.3 + h * |x|     (h slider; default 0.6 matches §2 demo)
//
// Left panel  : training scatter + both prediction bands overlaid.
// Right panel : conditional coverage as a function of x (12 bins) for each
//               method, with the nominal 1 − α reference line.
//
// The locally-adaptive CQR band tracks the noise envelope; the constant-width
// naive band over-covers near x=0 and under-covers in the tails. The right
// panel quantifies the difference: CQR's per-bin coverage is visibly flatter
// than naive's around the 1 − α target.
// =============================================================================

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 360;
const SM_BREAKPOINT = 768;
const N_TRAIN = 400;
const N_CAL = 400;
const N_TEST = 1500;
const N_BINS = 12;
const ALPHA_MIN = 0.05;
const ALPHA_MAX = 0.25;
const ALPHA_STEP = 0.01;
const H_MIN = 0;
const H_MAX = 1.5;
const H_STEP = 0.05;

const BLUE = '#2563EB';
const GREEN = '#059669';
const RED = '#DC2626';
const SLATE = '#475569';

const fmt = (x: number, digits = 3) => x.toFixed(digits);

// Local synth — copies synthHeteroscedastic but with a tunable h parameter.
// (The shared module's synthHeteroscedastic hard-codes h = 0.6.)
function synthHeteroH(
  n: number,
  rng: () => number,
  h: number,
): { x: Float64Array; y: Float64Array } {
  const gauss = gaussianSampler(rng);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const xi = -2 + 4 * rng();
    const sigma = 0.3 + h * Math.abs(xi);
    x[i] = xi;
    y[i] = 0.5 * xi + Math.sin(1.5 * xi) + sigma * gauss();
  }
  return { x, y };
}

type ShowMode = 'both' | 'naive' | 'cqr';

// ── Component ────────────────────────────────────────────────────────

export default function CQRExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [alpha, setAlpha] = useState(0.1);
  const [hetero, setHetero] = useState(0.6);
  const [showMode, setShowMode] = useState<ShowMode>('both');
  const [seed, setSeed] = useState(55);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const leftWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.55);
  const rightWidth = isStacked ? containerWidth : containerWidth - leftWidth;

  const sim = useMemo(() => {
    const rng = mulberry32(seed);
    const total = N_TRAIN + N_CAL;
    const { x, y } = synthHeteroH(total, rng, hetero);
    const xTrain = x.subarray(0, N_TRAIN);
    const yTrain = y.subarray(0, N_TRAIN);
    const xCal = x.subarray(N_TRAIN);
    const yCal = y.subarray(N_TRAIN);

    // Test set for conditional-coverage estimate
    const test = synthHeteroH(N_TEST, rng, hetero);

    // Grid for displaying the bands
    const xGrid = Float64Array.from({ length: 200 }, (_, i) => -2 + (4 * i) / 199);

    const naiveGrid = splitConformalInterval(
      xTrain,
      yTrain,
      xCal,
      yCal,
      xGrid,
      alpha,
      fitPredictRidge,
    );
    const cqrGrid = cqrInterval(xTrain, yTrain, xCal, yCal, xGrid, alpha);

    // Naive predictions on test set for conditional coverage
    const naiveTest = splitConformalInterval(
      xTrain,
      yTrain,
      xCal,
      yCal,
      test.x,
      alpha,
      fitPredictRidge,
    );
    const cqrTest = cqrInterval(xTrain, yTrain, xCal, yCal, test.x, alpha);

    // Per-bin conditional coverage
    const binEdges = Float64Array.from({ length: N_BINS + 1 }, (_, i) => -2 + (4 * i) / N_BINS);
    const binCenters = Float64Array.from({ length: N_BINS }, (_, i) => -2 + 4 * (i + 0.5) / N_BINS);
    const naiveCov = new Float64Array(N_BINS);
    const cqrCov = new Float64Array(N_BINS);
    const binN = new Int32Array(N_BINS);
    for (let i = 0; i < N_TEST; i++) {
      const xi = test.x[i];
      const yi = test.y[i];
      const bin = Math.min(N_BINS - 1, Math.max(0, Math.floor((xi + 2) / (4 / N_BINS))));
      binN[bin]++;
      if (yi >= naiveTest.lower[i] && yi <= naiveTest.upper[i]) naiveCov[bin]++;
      if (yi >= cqrTest.lower[i] && yi <= cqrTest.upper[i]) cqrCov[bin]++;
    }
    for (let b = 0; b < N_BINS; b++) {
      if (binN[b] > 0) {
        naiveCov[b] /= binN[b];
        cqrCov[b] /= binN[b];
      } else {
        naiveCov[b] = NaN;
        cqrCov[b] = NaN;
      }
    }

    // Marginal coverage and mean width on the test set
    let naiveCoveredCount = 0;
    let cqrCoveredCount = 0;
    let naiveWidthSum = 0;
    let cqrWidthSum = 0;
    for (let i = 0; i < N_TEST; i++) {
      if (test.y[i] >= naiveTest.lower[i] && test.y[i] <= naiveTest.upper[i]) {
        naiveCoveredCount++;
      }
      if (test.y[i] >= cqrTest.lower[i] && test.y[i] <= cqrTest.upper[i]) {
        cqrCoveredCount++;
      }
      naiveWidthSum += naiveTest.upper[i] - naiveTest.lower[i];
      cqrWidthSum += cqrTest.upper[i] - cqrTest.lower[i];
    }
    const naiveMarginal = naiveCoveredCount / N_TEST;
    const cqrMarginal = cqrCoveredCount / N_TEST;
    const naiveMeanWidth = naiveWidthSum / N_TEST;
    const cqrMeanWidth = cqrWidthSum / N_TEST;

    return {
      xTrain,
      yTrain,
      xGrid,
      naiveLower: naiveGrid.lower,
      naiveUpper: naiveGrid.upper,
      cqrLower: cqrGrid.lower,
      cqrUpper: cqrGrid.upper,
      binCenters,
      naiveCov,
      cqrCov,
      naiveMarginal,
      cqrMarginal,
      naiveMeanWidth,
      cqrMeanWidth,
    };
  }, [alpha, hetero, seed]);

  // ── Left panel: scatter + bands ─────────────────────────────────
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (leftWidth <= 0) return;

      const margin = { top: 28, right: 12, bottom: 36, left: 44 };
      const w = leftWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-2, 2]).range([0, w]);
      const yMax = Math.min(6, 2 + 1.5 * hetero * 2);
      const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([h, 0]);

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

      // Training points (down-sampled)
      const step = Math.max(1, Math.floor(N_TRAIN / 200));
      for (let i = 0; i < N_TRAIN; i += step) {
        g.append('circle')
          .attr('cx', xScale(sim.xTrain[i]))
          .attr('cy', yScale(sim.yTrain[i]))
          .attr('r', 1.8)
          .style('fill', SLATE)
          .style('opacity', 0.4);
      }

      // Naive band
      if (showMode !== 'cqr') {
        const naiveArea = d3.area<number>()
          .x((_, i) => xScale(sim.xGrid[i]))
          .y0((_, i) => yScale(sim.naiveLower[i]))
          .y1((_, i) => yScale(sim.naiveUpper[i]));
        g.append('path')
          .datum(d3.range(sim.xGrid.length))
          .attr('d', naiveArea)
          .style('fill', BLUE)
          .style('opacity', 0.18);
      }

      // CQR band
      if (showMode !== 'naive') {
        const cqrArea = d3.area<number>()
          .x((_, i) => xScale(sim.xGrid[i]))
          .y0((_, i) => yScale(sim.cqrLower[i]))
          .y1((_, i) => yScale(sim.cqrUpper[i]));
        g.append('path')
          .datum(d3.range(sim.xGrid.length))
          .attr('d', cqrArea)
          .style('fill', GREEN)
          .style('opacity', 0.22);
      }

      // Title
      svg.append('text')
        .attr('x', leftWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`Prediction bands (α = ${alpha.toFixed(2)})`);

      // Legend
      let legendY = 30;
      if (showMode !== 'cqr') {
        g.append('rect')
          .attr('x', 6)
          .attr('y', legendY - 8)
          .attr('width', 14)
          .attr('height', 8)
          .style('fill', BLUE)
          .style('opacity', 0.4);
        g.append('text')
          .attr('x', 24)
          .attr('y', legendY)
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '10px')
          .text(`Naive split: width ${fmt(sim.naiveMeanWidth, 2)}, cov ${fmt(sim.naiveMarginal, 3)}`);
        legendY += 14;
      }
      if (showMode !== 'naive') {
        g.append('rect')
          .attr('x', 6)
          .attr('y', legendY - 8)
          .attr('width', 14)
          .attr('height', 8)
          .style('fill', GREEN)
          .style('opacity', 0.5);
        g.append('text')
          .attr('x', 24)
          .attr('y', legendY)
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '10px')
          .text(`CQR: width ${fmt(sim.cqrMeanWidth, 2)}, cov ${fmt(sim.cqrMarginal, 3)}`);
      }
    },
    [leftWidth, sim, showMode, alpha, hetero],
  );

  // ── Right panel: per-bin conditional coverage ─────────────────────
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (rightWidth <= 0) return;

      const margin = { top: 28, right: 12, bottom: 36, left: 44 };
      const w = rightWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-2, 2]).range([0, w]);
      const yScale = d3.scaleLinear().domain([0.5, 1.0]).range([h, 0]);

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
        .text('cond. coverage');

      // 1 − α reference
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(1 - alpha))
        .attr('y2', yScale(1 - alpha))
        .style('stroke', RED)
        .style('stroke-width', 1.5)
        .style('stroke-dasharray', '5,3');
      g.append('text')
        .attr('x', w - 4)
        .attr('y', yScale(1 - alpha) - 4)
        .style('fill', RED)
        .style('text-anchor', 'end')
        .style('font-size', '10px')
        .text(`1 − α = ${fmt(1 - alpha, 2)}`);

      // Naive coverage curve
      if (showMode !== 'cqr') {
        const naiveLine = d3.line<number>()
          .defined((d) => Number.isFinite(sim.naiveCov[d]))
          .x((d) => xScale(sim.binCenters[d]))
          .y((d) => yScale(sim.naiveCov[d]));
        g.append('path')
          .datum(d3.range(N_BINS))
          .attr('d', naiveLine)
          .style('fill', 'none')
          .style('stroke', BLUE)
          .style('stroke-width', 1.8);
        for (let b = 0; b < N_BINS; b++) {
          if (Number.isFinite(sim.naiveCov[b])) {
            g.append('circle')
              .attr('cx', xScale(sim.binCenters[b]))
              .attr('cy', yScale(sim.naiveCov[b]))
              .attr('r', 3.5)
              .style('fill', BLUE);
          }
        }
      }

      // CQR coverage curve
      if (showMode !== 'naive') {
        const cqrLine = d3.line<number>()
          .defined((d) => Number.isFinite(sim.cqrCov[d]))
          .x((d) => xScale(sim.binCenters[d]))
          .y((d) => yScale(sim.cqrCov[d]));
        g.append('path')
          .datum(d3.range(N_BINS))
          .attr('d', cqrLine)
          .style('fill', 'none')
          .style('stroke', GREEN)
          .style('stroke-width', 1.8);
        for (let b = 0; b < N_BINS; b++) {
          if (Number.isFinite(sim.cqrCov[b])) {
            g.append('rect')
              .attr('x', xScale(sim.binCenters[b]) - 3)
              .attr('y', yScale(sim.cqrCov[b]) - 3)
              .attr('width', 6)
              .attr('height', 6)
              .style('fill', GREEN);
          }
        }
      }

      svg.append('text')
        .attr('x', rightWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`Per-bin conditional coverage (${N_BINS} bins, ${N_TEST} test points)`);
    },
    [rightWidth, sim, showMode, alpha],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3"
    >
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg
          role="img"
          aria-label="Left panel: heteroscedastic regression scatter with naive split-conformal prediction band (blue, constant width) and CQR prediction band (green, locally adaptive) overlaid"
          ref={leftRef}
          width={leftWidth}
          height={HEIGHT}
        />
        <svg
          role="img"
          aria-label="Right panel: per-bin conditional coverage as a function of x for naive split conformal (blue circles) and CQR (green squares), with the 1 minus alpha reference line"
          ref={rightRef}
          width={rightWidth}
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
          <span>noise h</span>
          <input
            type="range"
            min={H_MIN}
            max={H_MAX}
            step={H_STEP}
            value={hetero}
            onChange={(e) => setHetero(parseFloat(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{hetero.toFixed(2)}</span>
        </label>

        <fieldset className="flex items-center gap-3 text-[var(--color-text-secondary)]">
          <legend className="sr-only">Show</legend>
          {(['both', 'naive', 'cqr'] as const).map((m) => (
            <label key={m} className="flex items-center gap-1">
              <input
                type="radio"
                name="cqrShowMode"
                checked={showMode === m}
                onChange={() => setShowMode(m)}
                className="accent-[var(--color-accent)]"
              />
              {m === 'both' ? 'Both' : m === 'naive' ? 'Naive only' : 'CQR only'}
            </label>
          ))}
        </fieldset>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border border-[var(--color-accent)] bg-[var(--color-accent)] px-3 py-1 text-xs font-medium text-white transition hover:opacity-90"
        >
          Resample
        </button>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Heteroscedasticity: σ(x) = 0.3 + h · |x|. At h = 0 the noise is constant and both bands are equivalent. As h grows, naive split-conformal's constant-width band over-covers near x = 0 and under-covers in the tails (visible in the right panel as a U-shaped dip below 1 − α). CQR's per-x quantile estimates track the noise envelope, keeping the right-panel curve much flatter — approximate conditional coverage as a side benefit of locally adaptive width.
      </p>
    </div>
  );
}

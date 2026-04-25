import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  apsPredictionSetDeterministic,
  apsScoreDeterministic,
  fitPredictLogisticRegression2D,
  mulberry32,
  synth3Class,
} from './shared/nonparametric-ml';

// =============================================================================
// APSForClassification — embedded in §7 after Definition 5.
//
// Three-panel visualization of Adaptive Prediction Sets on a 2D 3-class
// classification problem (Gaussian blobs):
//
//   Left   — classifier softmax: argmax color, alpha modulated by max prob;
//            training points overlaid.
//   Middle — APS region map: each grid cell colored by the predicted set
//            size (green = 1, amber = 2, red = 3). Training points overlaid.
//   Right  — bar chart of empirical set-size distribution on a held-out test
//            set, plus per-set-size empirical coverage as paired bars; 1−α
//            reference line.
//
// A 100×100 grid (10,000 cells) is precomputed for the heatmaps. Compute is
// dominated by the multinomial logistic regression fit (~50ms via Newton-
// Raphson at n_train=600) and the grid prediction (~10ms). Total well within
// budget.
// =============================================================================

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 280;
const SM_BREAKPOINT = 1100;
const N_TRAIN = 600;
const N_CAL = 399;
const N_TEST = 600;
const K = 3;
const GRID_RESOLUTION = 100;
const ALPHA_MIN = 0.05;
const ALPHA_MAX = 0.30;
const ALPHA_STEP = 0.01;
const SIGMA_MIN = 0.5;
const SIGMA_MAX = 1.5;
const SIGMA_STEP = 0.05;

// Match notebook colors so the live viz aligns with the static fallback PNG
const CLASS_COLORS = ['#2563EB', '#DC2626', '#059669']; // blue, red, green
const SIZE_COLORS = ['#D1FAE5', '#FEF3C7', '#FEE2E2']; // light green, amber, red
const RED = '#DC2626';
const BLUE = '#2563EB';
const GREEN = '#059669';

const fmt = (x: number, digits = 3) => x.toFixed(digits);

// ── Component ────────────────────────────────────────────────────────

export default function APSForClassification() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [alpha, setAlpha] = useState(0.1);
  const [sigma, setSigma] = useState(1.0);
  const [seed, setSeed] = useState(77);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth
    : Math.floor(containerWidth / 3);

  const sim = useMemo(() => {
    const rng = mulberry32(seed);
    const train = synth3Class(N_TRAIN, rng, sigma);
    const cal = synth3Class(N_CAL, rng, sigma);
    const test = synth3Class(N_TEST, rng, sigma);

    // Fit multinomial logistic regression
    const probsCal = fitPredictLogisticRegression2D(train.X, train.y, cal.X, K);
    const calScores = apsScoreDeterministic(probsCal, cal.y, cal.y.length, K);
    const nCalActual = cal.y.length;
    let k = Math.ceil((1 - alpha) * (nCalActual + 1));
    if (k > nCalActual) k = nCalActual;
    if (k < 1) k = 1;
    const sortedCal = Float64Array.from(calScores).sort();
    const threshold = sortedCal[k - 1];

    // APS on test set
    const probsTest = fitPredictLogisticRegression2D(train.X, train.y, test.X, K);
    const inSetTest = apsPredictionSetDeterministic(probsTest, threshold, test.y.length, K);

    const setSizeCounts = [0, 0, 0, 0]; // index by size
    let coveredCount = 0;
    const coveredBySize = [0, 0, 0, 0];
    const totalBySize = [0, 0, 0, 0];
    for (let i = 0; i < test.y.length; i++) {
      let size = 0;
      for (let c = 0; c < K; c++) size += inSetTest[i * K + c];
      setSizeCounts[size]++;
      totalBySize[size]++;
      const covered = inSetTest[i * K + test.y[i]] === 1;
      if (covered) {
        coveredCount++;
        coveredBySize[size]++;
      }
    }
    const marginalCoverage = coveredCount / test.y.length;
    const coverageBySize = [0, 1, 2, 3].map((s) =>
      totalBySize[s] > 0 ? coveredBySize[s] / totalBySize[s] : NaN,
    );

    // Grid for heatmaps: input range [-3.5, 3.5] x [-2.5, 3.5]
    const xMin = -3.5, xMax = 3.5, yMin = -2.5, yMax = 3.5;
    const gridX = new Float64Array(GRID_RESOLUTION * GRID_RESOLUTION * 2);
    for (let i = 0; i < GRID_RESOLUTION; i++) {
      const yv = yMin + ((yMax - yMin) * i) / (GRID_RESOLUTION - 1);
      for (let j = 0; j < GRID_RESOLUTION; j++) {
        const xv = xMin + ((xMax - xMin) * j) / (GRID_RESOLUTION - 1);
        const idx = i * GRID_RESOLUTION + j;
        gridX[idx * 2] = xv;
        gridX[idx * 2 + 1] = yv;
      }
    }
    const gridProbs = fitPredictLogisticRegression2D(train.X, train.y, gridX, K);
    const gridArgmax = new Int32Array(GRID_RESOLUTION * GRID_RESOLUTION);
    const gridMaxProb = new Float64Array(GRID_RESOLUTION * GRID_RESOLUTION);
    const gridInSet = apsPredictionSetDeterministic(
      gridProbs,
      threshold,
      GRID_RESOLUTION * GRID_RESOLUTION,
      K,
    );
    const gridSetSize = new Int32Array(GRID_RESOLUTION * GRID_RESOLUTION);
    for (let i = 0; i < GRID_RESOLUTION * GRID_RESOLUTION; i++) {
      let argmax = 0;
      let maxP = gridProbs[i * K];
      for (let c = 1; c < K; c++) {
        if (gridProbs[i * K + c] > maxP) {
          maxP = gridProbs[i * K + c];
          argmax = c;
        }
      }
      gridArgmax[i] = argmax;
      gridMaxProb[i] = maxP;
      let size = 0;
      for (let c = 0; c < K; c++) size += gridInSet[i * K + c];
      gridSetSize[i] = size;
    }

    return {
      train,
      gridArgmax,
      gridMaxProb,
      gridSetSize,
      xMin, xMax, yMin, yMax,
      threshold,
      marginalCoverage,
      setSizeCounts,
      coverageBySize,
    };
  }, [alpha, sigma, seed]);

  // ── Helper: render scatter of training points colored by class ─────
  function drawTrainingPoints(g: d3.Selection<SVGGElement, unknown, null, undefined>, xScale: d3.ScaleLinear<number, number>, yScale: d3.ScaleLinear<number, number>) {
    const step = Math.max(1, Math.floor(N_TRAIN / 200));
    for (let i = 0; i < sim.train.y.length; i += step) {
      g.append('circle')
        .attr('cx', xScale(sim.train.X[i * 2]))
        .attr('cy', yScale(sim.train.X[i * 2 + 1]))
        .attr('r', 2.2)
        .style('fill', CLASS_COLORS[sim.train.y[i]])
        .style('opacity', 0.7);
    }
  }

  // ── Left panel: classifier softmax (argmax + confidence shading) ─
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const margin = { top: 28, right: 8, bottom: 32, left: 36 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([sim.xMin, sim.xMax]).range([0, w]);
      const yScale = d3.scaleLinear().domain([sim.yMin, sim.yMax]).range([h, 0]);
      const cellW = w / GRID_RESOLUTION;
      const cellH = h / GRID_RESOLUTION;

      // Heatmap
      for (let i = 0; i < GRID_RESOLUTION; i++) {
        for (let j = 0; j < GRID_RESOLUTION; j++) {
          const idx = i * GRID_RESOLUTION + j;
          g.append('rect')
            .attr('x', j * cellW)
            .attr('y', h - (i + 1) * cellH)
            .attr('width', cellW + 1)
            .attr('height', cellH + 1)
            .style('fill', CLASS_COLORS[sim.gridArgmax[idx]])
            .style('opacity', 0.18 * sim.gridMaxProb[idx])
            .style('shape-rendering', 'crispEdges');
        }
      }

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');

      drawTrainingPoints(g, xScale, yScale);

      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Classifier softmax (argmax + confidence)');
    },
    [panelWidth, sim],
  );

  // ── Middle panel: APS region map ─────────────────────────────────
  const middleRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const margin = { top: 28, right: 8, bottom: 32, left: 36 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([sim.xMin, sim.xMax]).range([0, w]);
      const yScale = d3.scaleLinear().domain([sim.yMin, sim.yMax]).range([h, 0]);
      const cellW = w / GRID_RESOLUTION;
      const cellH = h / GRID_RESOLUTION;

      for (let i = 0; i < GRID_RESOLUTION; i++) {
        for (let j = 0; j < GRID_RESOLUTION; j++) {
          const idx = i * GRID_RESOLUTION + j;
          const size = sim.gridSetSize[idx];
          if (size < 1) continue;
          g.append('rect')
            .attr('x', j * cellW)
            .attr('y', h - (i + 1) * cellH)
            .attr('width', cellW + 1)
            .attr('height', cellH + 1)
            .style('fill', SIZE_COLORS[size - 1])
            .style('opacity', 0.85)
            .style('shape-rendering', 'crispEdges');
        }
      }

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');

      drawTrainingPoints(g, xScale, yScale);

      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`APS region (q̂ = ${fmt(sim.threshold, 3)})`);

      // Tiny legend at bottom-right
      const legendItems = [
        { label: '|set|=1', color: SIZE_COLORS[0] },
        { label: '|set|=2', color: SIZE_COLORS[1] },
        { label: '|set|=3', color: SIZE_COLORS[2] },
      ];
      legendItems.forEach((item, i) => {
        g.append('rect')
          .attr('x', w - 56)
          .attr('y', i * 12 + 2)
          .attr('width', 10)
          .attr('height', 8)
          .style('fill', item.color)
          .style('stroke', '#444')
          .style('stroke-width', 0.5);
        g.append('text')
          .attr('x', w - 42)
          .attr('y', i * 12 + 9)
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '9px')
          .text(item.label);
      });
    },
    [panelWidth, sim],
  );

  // ── Right panel: set-size distribution + per-size coverage ───────
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const margin = { top: 28, right: 36, bottom: 32, left: 40 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const sizes = [1, 2, 3];
      const xBand = d3.scaleBand<number>().domain(sizes).range([0, w]).padding(0.25);
      const yScale = d3.scaleLinear().domain([0, N_TEST]).range([h, 0]);
      const yScaleCov = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xBand).tickFormat((d) => `${d}`))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text')
        .style('fill', BLUE);
      g.append('g')
        .attr('transform', `translate(${w},0)`)
        .call(d3.axisRight(yScaleCov).ticks(4).tickFormat((d) => `${(d as number).toFixed(2)}`))
        .selectAll('text')
        .style('fill', GREEN);
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-muted-border)');

      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 26)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('prediction-set size');

      const halfW = xBand.bandwidth() / 2;
      sizes.forEach((s) => {
        const xs = xBand(s) || 0;
        // Frequency bar (left half)
        g.append('rect')
          .attr('x', xs)
          .attr('y', yScale(sim.setSizeCounts[s]))
          .attr('width', halfW - 1)
          .attr('height', h - yScale(sim.setSizeCounts[s]))
          .style('fill', BLUE)
          .style('opacity', 0.6);
        g.append('text')
          .attr('x', xs + halfW / 2)
          .attr('y', yScale(sim.setSizeCounts[s]) - 3)
          .style('fill', BLUE)
          .style('text-anchor', 'middle')
          .style('font-size', '9px')
          .text(sim.setSizeCounts[s]);

        // Per-size coverage bar (right half)
        const cov = sim.coverageBySize[s];
        if (Number.isFinite(cov)) {
          g.append('rect')
            .attr('x', xs + halfW)
            .attr('y', yScaleCov(cov))
            .attr('width', halfW - 1)
            .attr('height', h - yScaleCov(cov))
            .style('fill', GREEN)
            .style('opacity', 0.6);
          g.append('text')
            .attr('x', xs + halfW + halfW / 2)
            .attr('y', yScaleCov(cov) - 3)
            .style('fill', GREEN)
            .style('text-anchor', 'middle')
            .style('font-size', '9px')
            .text(fmt(cov, 2));
        }
      });

      // 1 − α reference line on the coverage axis
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScaleCov(1 - alpha))
        .attr('y2', yScaleCov(1 - alpha))
        .style('stroke', RED)
        .style('stroke-width', 1.4)
        .style('stroke-dasharray', '4,3');

      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', 14)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`Coverage = ${fmt(sim.marginalCoverage, 3)} (target ${fmt(1 - alpha, 2)})`);

      // Axis labels
      g.append('text')
        .attr('x', -28)
        .attr('y', h / 2)
        .style('fill', BLUE)
        .style('text-anchor', 'middle')
        .style('font-size', '10px')
        .attr('transform', `rotate(-90,-28,${h / 2})`)
        .text('count');
      g.append('text')
        .attr('x', w + 22)
        .attr('y', h / 2)
        .style('fill', GREEN)
        .style('text-anchor', 'middle')
        .style('font-size', '10px')
        .attr('transform', `rotate(90,${w + 22},${h / 2})`)
        .text('coverage');
    },
    [panelWidth, sim, alpha],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted-bg)] p-3"
    >
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg
          role="img"
          aria-label="Left panel: classifier softmax over a 2D feature space — argmax-class color modulated by max probability, training points overlaid"
          ref={leftRef}
          width={panelWidth}
          height={HEIGHT}
        />
        <svg
          role="img"
          aria-label="Middle panel: APS region map — each grid cell colored by predicted set size (light green=1, amber=2, red=3); training points overlaid"
          ref={middleRef}
          width={panelWidth}
          height={HEIGHT}
        />
        <svg
          role="img"
          aria-label="Right panel: bar chart of empirical set-size distribution on test set, with per-set-size empirical coverage as paired bars and 1-α reference line"
          ref={rightRef}
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
          <span>class overlap σ</span>
          <input
            type="range"
            min={SIGMA_MIN}
            max={SIGMA_MAX}
            step={SIGMA_STEP}
            value={sigma}
            onChange={(e) => setSigma(parseFloat(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
          <span className="font-mono w-12 text-right">{sigma.toFixed(2)}</span>
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
        Three Gaussian blobs at fixed centers; σ controls within-class spread (smaller = more separable, larger = more overlap). At small σ the classifier is confident almost everywhere and APS produces singletons; at large σ the decision boundaries get thick and the set sizes grow to 2 or 3 there. Marginal coverage tracks 1 − α regardless of σ; per-set-size coverage stays close to nominal — APS is not over-extracting coverage from any one set-size bucket.
      </p>
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  synthSinusoidWithWiggle,
  trueFunction,
  blrFitPosterior,
  blrPredictPoint,
  polyFeatures,
  gpCandidatePredictive,
  paletteStacking,
} from './shared/bayesian-ml-stacking';

// =============================================================================
// FourCandidatesExplorer — embedded alongside §1.1.
// Renders the §1 ground truth on [0, 1] with sampled data and four candidate
// point fits (BLR, BPR-4, GP, BART-stand-in via shallow random forest), updating
// in real time as the reader varies n and σ. Each candidate's inductive bias is
// fixed regardless of n — the reader should observe that "more data" does not
// solve the M-open problem.
// =============================================================================

const PANEL_HEIGHT = 380;
const N_OPTIONS = [20, 40, 60, 100, 150, 200, 300] as const;
const SIGMA_OPTIONS = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5] as const;

type CandidateKey = 'blr' | 'bpr' | 'gp' | 'bart';

interface CandidateFit {
  key: CandidateKey;
  label: string;
  color: string;
  yhat: Float64Array;
}

const N_GRID = 200;

function bartSurrogate(
  xTrain: Float64Array,
  yTrain: Float64Array,
  xGrid: Float64Array,
): Float64Array {
  // Depth-3 stump-of-stumps surrogate: piecewise-constant means over equal-width
  // bins. 16 bins on [0, 1] gives a visual stand-in for BART's posterior mean
  // with the right inductive bias (piecewise-constant on axis-aligned splits).
  const nBins = 16;
  const sums = new Float64Array(nBins);
  const counts = new Int32Array(nBins);
  for (let i = 0; i < xTrain.length; i++) {
    const b = Math.min(nBins - 1, Math.floor(xTrain[i] * nBins));
    sums[b] += yTrain[i];
    counts[b] += 1;
  }
  const means = new Float64Array(nBins);
  for (let b = 0; b < nBins; b++) means[b] = counts[b] > 0 ? sums[b] / counts[b] : 0;
  const out = new Float64Array(xGrid.length);
  for (let i = 0; i < xGrid.length; i++) {
    const b = Math.min(nBins - 1, Math.floor(xGrid[i] * nBins));
    out[i] = means[b];
  }
  return out;
}

export default function FourCandidatesExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [n, setN] = useState<number>(100);
  const [sigma, setSigma] = useState<number>(0.25);
  const [seed, setSeed] = useState<number>(20260430);
  const [visible, setVisible] = useState<Record<CandidateKey, boolean>>({
    blr: true,
    bpr: true,
    gp: true,
    bart: true,
  });

  const xGrid = useMemo<Float64Array>(() => {
    const g = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) g[i] = i / (N_GRID - 1);
    return g;
  }, []);

  const truthGrid = useMemo<Float64Array>(() => {
    const t = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) t[i] = trueFunction(xGrid[i]);
    return t;
  }, [xGrid]);

  const data = useMemo(() => {
    const rng = mulberry32(seed);
    return synthSinusoidWithWiggle(n, sigma, rng);
  }, [n, sigma, seed]);

  const fits = useMemo<CandidateFit[]>(() => {
    const xArr = data.x;
    const yArr = data.y;

    // BLR.
    const Xblr = polyFeatures(xArr, 1);
    const blrPost = blrFitPosterior(Xblr, yArr);
    const yhatBlr = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      yhatBlr[i] = blrPredictPoint(blrPost, [1, xGrid[i]]).loc;
    }

    // BPR-4.
    const Xbpr = polyFeatures(xArr, 4);
    const bprPost = blrFitPosterior(Xbpr, yArr);
    const yhatBpr = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      const xv = xGrid[i];
      const row = [1, xv, xv * xv, xv * xv * xv, xv * xv * xv * xv];
      yhatBpr[i] = blrPredictPoint(bprPost, row).loc;
    }

    // GP.
    const gpResult = gpCandidatePredictive(xArr, yArr, xGrid, {
      lengthScale: 0.1,
      outputVar: 1.0,
      noiseVar: sigma * sigma,
    });

    // BART stand-in.
    const yhatBart = bartSurrogate(xArr, yArr, xGrid);

    return [
      { key: 'blr', label: 'Linear (BLR)', color: paletteStacking.blr, yhat: yhatBlr },
      { key: 'bpr', label: 'Polynomial-4 (BPR)', color: paletteStacking.bpr, yhat: yhatBpr },
      { key: 'gp', label: 'GP (RBF)', color: paletteStacking.gp, yhat: gpResult.loc },
      { key: 'bart', label: 'Tree ensemble (BART surrogate)', color: paletteStacking.bart, yhat: yhatBart },
    ];
  }, [data, xGrid, sigma]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const height = PANEL_HEIGHT;
      const margin = { top: 16, right: 18, bottom: 36, left: 44 };
      svg.attr('width', w).attr('height', height);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-2.0, 2.0]).range([innerH, 0]);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      g.append('text')
        .attr('transform', `translate(${innerW / 2},${innerH + 30})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('x');
      g.append('text')
        .attr('transform', `translate(-32,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('y');

      // Discontinuity marker at x = 0.5.
      g.append('line')
        .attr('x1', xScale(0.5))
        .attr('x2', xScale(0.5))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#888')
        .attr('stroke-dasharray', '3 3')
        .attr('stroke-width', 0.6);

      // Data points.
      g.append('g')
        .selectAll('circle')
        .data(Array.from({ length: data.x.length }, (_, i) => ({ x: data.x[i], y: data.y[i] })))
        .join('circle')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', 2.4)
        .attr('fill', '#7f7f7f')
        .attr('opacity', 0.55);

      // Truth.
      const lineGen = d3
        .line<number>()
        .x((_, i) => xScale(xGrid[i]))
        .y((d) => yScale(d));

      g.append('path')
        .datum(Array.from(truthGrid))
        .attr('fill', 'none')
        .attr('stroke', paletteStacking.truth)
        .attr('stroke-width', 1.6)
        .attr('d', lineGen);

      // Candidate fits.
      fits.forEach((fit) => {
        if (!visible[fit.key]) return;
        g.append('path')
          .datum(Array.from(fit.yhat))
          .attr('fill', 'none')
          .attr('stroke', fit.color)
          .attr('stroke-width', 1.8)
          .attr('opacity', 0.92)
          .attr('d', lineGen);
      });

      // Legend.
      const legendItems = [{ label: 'Truth', color: paletteStacking.truth }, ...fits.map((f) => ({ label: f.label, color: f.color }))];
      const legend = g.append('g').attr('transform', `translate(${innerW - 200},10)`);
      legendItems.forEach((item, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        row.append('rect').attr('width', 14).attr('height', 3).attr('y', 5).attr('fill', item.color);
        row.append('text').attr('x', 18).attr('y', 9).attr('font-size', 10).text(item.label);
      });
    },
    [data, fits, truthGrid, visible, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap gap-3 text-sm">
        <label>
          n:&nbsp;
          <select value={n} onChange={(e) => setN(Number(e.target.value))} className="rounded border px-1">
            {N_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          σ:&nbsp;
          <select value={sigma} onChange={(e) => setSigma(Number(e.target.value))} className="rounded border px-1">
            {SIGMA_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border bg-white px-2 py-0.5 hover:bg-gray-50"
        >
          Reseed
        </button>
        {(['blr', 'bpr', 'gp', 'bart'] as CandidateKey[]).map((k) => (
          <label key={k} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={visible[k]}
              onChange={(e) => setVisible((v) => ({ ...v, [k]: e.target.checked }))}
            />
            <span style={{ color: { blr: paletteStacking.blr, bpr: paletteStacking.bpr, gp: paletteStacking.gp, bart: paletteStacking.bart }[k] }}>
              {{ blr: 'Linear', bpr: 'Poly-4', gp: 'GP', bart: 'BART' }[k]}
            </span>
          </label>
        ))}
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
    </div>
  );
}

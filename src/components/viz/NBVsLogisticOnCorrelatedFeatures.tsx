import { useState, useMemo, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  paletteNB,
  mulberry32,
  gaussianSampler,
  gaussianNBFit,
  gaussianNBPredictProba,
  logisticFit,
  logisticPredictProba,
  brierScore,
  expectedCalibrationError,
} from './shared/naive-bayes';

// =============================================================================
// NBVsLogisticOnCorrelatedFeatures — §10.3 companion. Generate two-class
// labeled data where K features are noisy copies of a single signal feature,
// adjustable by the correlation slider ρ ∈ [0, 0.99]. Fit Gaussian NB and
// (unregularized) logistic regression; plot two diagnostics:
//
//   (a) Calibration histograms — distribution of predicted P(y = 1 | x). At
//       high ρ, NB collapses to 0/1 (the redundant evidence multiplies under
//       independence); LR pools the duplicates into a single effective signal.
//
//   (b) Per-method accuracy + ECE — accuracy is nearly identical (NB's ranking
//       is good), but ECE diverges sharply.
// =============================================================================

const N_SIGNAL = 1;
const N_COPIES = 8; // number of noisy duplicates of the signal feature
const N = 400;
const SIG_BASE = 1.0;

interface Data {
  X: Float64Array;
  y: Int32Array;
  rho: number;
}

function generateData(rho: number, seed: number): Data {
  const rng = mulberry32(seed);
  const gauss = gaussianSampler(rng);
  const d = N_SIGNAL + N_COPIES;
  const X = new Float64Array(N * d);
  const y = new Int32Array(N);
  // The signal feature: x_0 = (2y − 1) + N(0, σ²).
  // Each copy: x_k = ρ x_0 + sqrt(1 − ρ²) N(0, σ²) for k ≥ 1.
  for (let i = 0; i < N; i++) {
    y[i] = rng() < 0.5 ? 0 : 1;
    const signal = (2 * y[i] - 1) + SIG_BASE * gauss();
    X[i * d + 0] = signal;
    for (let k = 1; k <= N_COPIES; k++) {
      X[i * d + k] = rho * signal + Math.sqrt(1 - rho * rho) * gauss() * SIG_BASE;
    }
  }
  return { X, y, rho };
}

export default function NBVsLogisticOnCorrelatedFeatures() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [displayRho, setDisplayRho] = useState(0.9);
  const [committedRho, setCommittedRho] = useState(0.9);
  const [seed, setSeed] = useState(42);

  const data = useMemo(() => generateData(committedRho, seed), [committedRho, seed]);

  const fits = useMemo(() => {
    const d = N_SIGNAL + N_COPIES;
    // Train/test 70/30 split by index.
    const nTrain = Math.floor(N * 0.7);
    const Xtr = new Float64Array(nTrain * d);
    const ytr = new Int32Array(nTrain);
    const Xte = new Float64Array((N - nTrain) * d);
    const yte = new Int32Array(N - nTrain);
    for (let i = 0; i < nTrain; i++) {
      ytr[i] = data.y[i];
      for (let j = 0; j < d; j++) Xtr[i * d + j] = data.X[i * d + j];
    }
    for (let i = nTrain; i < N; i++) {
      yte[i - nTrain] = data.y[i];
      for (let j = 0; j < d; j++) Xte[(i - nTrain) * d + j] = data.X[i * d + j];
    }
    const gnb = gaussianNBFit(Xtr, ytr, d);
    const gnbProba = gaussianNBPredictProba(Xte, gnb);
    const gnbP1: number[] = new Array(yte.length);
    for (let i = 0; i < yte.length; i++) gnbP1[i] = gnbProba[i * 2 + 1];
    const yteI8 = new Int8Array(yte.length);
    for (let i = 0; i < yte.length; i++) yteI8[i] = yte[i] > 0 ? 1 : 0;
    const ytrI8 = new Int8Array(ytr.length);
    for (let i = 0; i < ytr.length; i++) ytrI8[i] = ytr[i] > 0 ? 1 : 0;
    const beta = logisticFit(Xtr, ytrI8, ytr.length, d, { ridge: 1e-3 });
    const lrP1 = Array.from(logisticPredictProba(beta, Xte, yte.length, d));
    const accGnb = gnbP1.reduce((s: number, p: number, i: number) => s + ((p > 0.5 ? 1 : 0) === yteI8[i] ? 1 : 0), 0) / yte.length;
    const accLr = lrP1.reduce((s: number, p: number, i: number) => s + ((p > 0.5 ? 1 : 0) === yteI8[i] ? 1 : 0), 0) / yte.length;
    const labels = Array.from(yteI8);
    return {
      gnb: {
        proba: gnbP1,
        acc: accGnb,
        ece: expectedCalibrationError(gnbP1, labels, 10),
        brier: brierScore(gnbP1, labels),
      },
      lr: {
        proba: lrP1,
        acc: accLr,
        ece: expectedCalibrationError(lrP1, labels, 10),
        brier: brierScore(lrP1, labels),
      },
    };
  }, [data]);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 280;
  const margin = { top: 14, right: 14, bottom: 36, left: 36 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const histogram = useMemo(() => {
    const binEdges = Array.from({ length: 21 }, (_, i) => i / 20);
    const histGnb = new Array(binEdges.length - 1).fill(0);
    const histLr = new Array(binEdges.length - 1).fill(0);
    for (const p of fits.gnb.proba) {
      const b = Math.min(histGnb.length - 1, Math.floor(p * (histGnb.length - 1e-9)));
      histGnb[b] += 1;
    }
    for (const p of fits.lr.proba) {
      const b = Math.min(histLr.length - 1, Math.floor(p * (histLr.length - 1e-9)));
      histLr[b] += 1;
    }
    return { edges: binEdges, gnb: histGnb, lr: histLr };
  }, [fits]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const maxY = Math.max(...histogram.gnb, ...histogram.lr) * 1.1 || 1;
      const y = d3.scaleLinear().domain([0, maxY]).range([innerH, 0]);
      const bw = innerW / histogram.gnb.length;

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .style('fill', 'currentColor')
        .text('Predicted P(y = 1 | x)');
      g.append('text')
        .attr('transform', `translate(-28,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .style('fill', 'currentColor')
        .text('Test count');

      // GNB bars (red), LR bars (blue), interleaved.
      for (let b = 0; b < histogram.gnb.length; b++) {
        g.append('rect')
          .attr('x', x(histogram.edges[b]))
          .attr('y', y(histogram.gnb[b]))
          .attr('width', bw / 2)
          .attr('height', innerH - y(histogram.gnb[b]))
          .attr('fill-opacity', 0.7)
          .style('fill', paletteNB.neg);
        g.append('rect')
          .attr('x', x(histogram.edges[b]) + bw / 2)
          .attr('y', y(histogram.lr[b]))
          .attr('width', bw / 2)
          .attr('height', innerH - y(histogram.lr[b]))
          .attr('fill-opacity', 0.7)
          .style('fill', paletteNB.pos);
      }

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 130},6)`);
      [
        { label: 'Gaussian NB', color: paletteNB.neg },
        { label: 'Logistic reg', color: paletteNB.pos },
      ].forEach((item, i) => {
        legend.append('rect').attr('x', 0).attr('y', i * 14 - 8).attr('width', 12).attr('height', 6).style('fill', item.color);
        legend
          .append('text')
          .attr('x', 16)
          .attr('y', i * 14)
          .attr('font-size', 11)
          .style('fill', 'currentColor')
          .text(item.label);
      });
    },
    [histogram, W, H, innerW, innerH, margin.left, margin.top],
  );

  // Commit-on-release pattern.
  useEffect(() => {
    const id = window.setTimeout(() => setCommittedRho(displayRho), 200);
    return () => window.clearTimeout(id);
  }, [displayRho]);

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="flex flex-wrap items-center gap-3 mb-2 text-sm">
        <label className="flex items-center gap-2">
          Feature correlation ρ = {displayRho.toFixed(2)}
          <input
            type="range"
            min={0}
            max={0.99}
            step={0.01}
            value={displayRho}
            onChange={(e) => setDisplayRho(Number(e.target.value))}
            aria-label="Feature correlation rho"
          />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="ml-auto rounded border px-2 py-1 text-xs"
          aria-label="Resample data"
        >
          Resample (seed {seed})
        </button>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Predicted probability histograms for NB vs LR under correlated features"/>
      <div className="grid grid-cols-2 gap-3 mt-2 text-xs">
        {[
          { name: 'Gaussian NB', color: paletteNB.neg, m: fits.gnb },
          { name: 'Logistic regression', color: paletteNB.pos, m: fits.lr },
        ].map((v) => (
          <div key={v.name} className="text-center">
            <div className="font-semibold" style={{ color: v.color }}>{v.name}</div>
            <div>Accuracy = {v.m.acc.toFixed(3)}</div>
            <div>ECE = {v.m.ece.toFixed(4)}</div>
            <div>Brier = {v.m.brier.toFixed(4)}</div>
          </div>
        ))}
      </div>
      <div className="text-xs opacity-75 mt-2 text-center">
        At high ρ, NB treats the {N_COPIES} duplicate features as {N_COPIES} independent
        votes, multiplying its evidence — predicted probabilities collapse to 0 / 1 and ECE
        balloons. LR pools the duplicates into a single effective coefficient.
      </div>
    </div>
  );
}

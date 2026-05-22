import { useState, useEffect, useMemo, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  paletteNB,
  plattScalingApply,
  expectedCalibrationError,
  brierScore,
} from './shared/naive-bayes';

// =============================================================================
// CalibrationReliabilityDiagram — §8.3 companion. Loads
//   /sample-data/naive-bayes/calibration_scores.json
// (raw NB log-odds on a 20-Newsgroups held-out split + Platt parameters +
// isotonic step-function breakpoints) and lets the reader recompute reliability
// bins at any bin count. ECE and Brier are computed live for raw, Platt, and
// isotonic — including the characteristic NB-overconfidence on raw probabilities.
// =============================================================================

interface CalibPayload {
  seed: number;
  dataset: string;
  alpha: number;
  scores_test: number[];
  raw_proba_test: number[];
  labels_test: number[];
  platt_params: { A: number; B: number };
  isotonic_breakpoints: Array<[number, number]>;
}

function applyIsotonic(scores: number[], breakpoints: Array<[number, number]>): number[] {
  // Step-function with linear interpolation between breakpoints (matches sklearn).
  if (breakpoints.length === 0) return scores.map(() => 0.5);
  const xs = breakpoints.map((b) => b[0]);
  const ys = breakpoints.map((b) => b[1]);
  return scores.map((s) => {
    if (s <= xs[0]) return ys[0];
    if (s >= xs[xs.length - 1]) return ys[ys.length - 1];
    // Binary search for the bracket.
    let lo = 0;
    let hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= s) lo = mid;
      else hi = mid;
    }
    const t = (s - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  });
}

function reliabilityBinsQuantile(probs: number[], labels: number[], nBins: number) {
  const n = probs.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[a] - probs[b]);
  const bins: Array<{ mean_pred: number; frac_pos: number; count: number }> = [];
  let start = 0;
  for (let b = 0; b < nBins; b++) {
    const end = Math.round(((b + 1) * n) / nBins);
    if (end <= start) continue;
    let pSum = 0;
    let yMean = 0;
    for (let i = start; i < end; i++) {
      pSum += probs[order[i]];
      yMean += labels[order[i]] > 0 ? 1 : 0;
    }
    const m = end - start;
    bins.push({ mean_pred: pSum / m, frac_pos: yMean / m, count: m });
    start = end;
  }
  return bins;
}

export default function CalibrationReliabilityDiagram() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<CalibPayload | null>(null);
  const [showError, setShowError] = useState<string | null>(null);
  const [nBins, setNBins] = useState(10);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/naive-bayes/calibration_scores.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: CalibPayload) => {
        if (!cancelled) setPayload(j);
      })
      .catch((e) => {
        if (!cancelled) setShowError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const W = Math.max(360, Math.min(width || 600, 600));
  const H = 420;
  const margin = { top: 14, right: 16, bottom: 38, left: 50 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const series = useMemo(() => {
    if (!payload) return null;
    const labels = payload.labels_test;
    const raw = payload.raw_proba_test;
    const plattArr = Array.from(plattScalingApply(payload.scores_test, payload.platt_params));
    const isoArr = applyIsotonic(payload.scores_test, payload.isotonic_breakpoints);
    return {
      raw: {
        probs: raw,
        bins: reliabilityBinsQuantile(raw, labels, nBins),
        ece: expectedCalibrationError(raw, labels, nBins),
        brier: brierScore(raw, labels),
      },
      platt: {
        probs: plattArr,
        bins: reliabilityBinsQuantile(plattArr, labels, nBins),
        ece: expectedCalibrationError(plattArr, labels, nBins),
        brier: brierScore(plattArr, labels),
      },
      isotonic: {
        probs: isoArr,
        bins: reliabilityBinsQuantile(isoArr, labels, nBins),
        ece: expectedCalibrationError(isoArr, labels, nBins),
        brier: brierScore(isoArr, labels),
      },
    };
  }, [payload, nBins]);

  useEffect(() => {
    if (!svgRef.current || !series) return;
    const svg = d3.select(svgRef.current);
    svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6));
    g.append('g').call(d3.axisLeft(y).ticks(6));

    // Diagonal = perfect calibration.
    g.append('line')
      .attr('x1', x(0))
      .attr('y1', y(0))
      .attr('x2', x(1))
      .attr('y2', y(1))
      .attr('stroke-dasharray', '4 4')
      .attr('opacity', 0.5)
      .style('stroke', paletteNB.data);

    const variants = [
      { name: 'Raw NB', color: paletteNB.neg, data: series.raw },
      { name: 'Platt', color: paletteNB.pos, data: series.platt },
      { name: 'Isotonic', color: paletteNB.ok, data: series.isotonic },
    ];

    variants.forEach((v) => {
      const line = d3
        .line<typeof v.data.bins[number]>()
        .x((d) => x(d.mean_pred))
        .y((d) => y(d.frac_pos));
      g.append('path')
        .datum(v.data.bins)
        .attr('fill', 'none')
        .attr('stroke-width', 2)
        .attr('d', line)
        .style('stroke', v.color);
      g.append('g')
        .selectAll('circle')
        .data(v.data.bins)
        .join('circle')
        .attr('cx', (d) => x(d.mean_pred))
        .attr('cy', (d) => y(d.frac_pos))
        .attr('r', 4)
        .style('fill', v.color);
    });

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 32)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .style('fill', 'currentColor')
      .text('Mean predicted probability');
    g.append('text')
      .attr('transform', `translate(-36,${innerH / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .style('fill', 'currentColor')
      .text('Fraction of positives');
  }, [series, W, H, innerW, innerH, margin.left, margin.top]);

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="flex flex-wrap items-center gap-3 mb-2 text-sm">
        <label className="flex items-center gap-2">
          Bins:
          <input
            type="range"
            min={5}
            max={20}
            step={1}
            value={nBins}
            onChange={(e) => setNBins(Number(e.target.value))}
            aria-label="Number of quantile bins"
          />
          <span className="tabular-nums w-6 text-right">{nBins}</span>
        </label>
      </div>
      {showError ? (
        <div className="text-sm" style={{ color: paletteNB.neg }}>
          Failed to load /sample-data/naive-bayes/calibration_scores.json: {showError}
        </div>
      ) : !series ? (
        <div className="text-sm opacity-60" style={{ minHeight: H }}>Loading calibration data...</div>
      ) : (
        <>
          <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Reliability diagram"/>
          <div className="grid grid-cols-3 gap-3 mt-2 text-xs">
            {[
              { name: 'Raw NB', color: paletteNB.neg, m: series.raw },
              { name: 'Platt', color: paletteNB.pos, m: series.platt },
              { name: 'Isotonic', color: paletteNB.ok, m: series.isotonic },
            ].map((v) => (
              <div key={v.name} className="text-center">
                <div className="font-semibold" style={{ color: v.color }}>{v.name}</div>
                <div>ECE = {v.m.ece.toFixed(4)}</div>
                <div>Brier = {v.m.brier.toFixed(4)}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

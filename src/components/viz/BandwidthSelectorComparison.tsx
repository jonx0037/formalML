import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gcvScore,
  hStarAmiseUni,
  kGaussian,
  looCvScore,
  mulberry32,
  paletteKR,
  sampleToyUni,
  silvermanRule,
} from './shared/kernel-regression';
import { useEffect, useRef } from 'react';

// =============================================================================
// BandwidthSelectorComparison — interactive companion to §5.5's selector-
// stability figure. Two synchronized panels:
//
//   Top:    single-sample comparison — LOO-CV and GCV objective curves over a
//           bandwidth grid, with vertical markers at h_Silverman, h_LOO-CV,
//           h_GCV, and the analytical h^*.
//
//   Bottom: stability across B replicates — histogram (or strip) of selected
//           bandwidths for each enabled selector, with the analytical h^* as a
//           reference line.
//
// Reader controls: n ∈ {100, 200, 500},  B ∈ {30, 60, 100}, selector toggles.
//
// Numerical anchor: at n = 200, B = 100 (notebook §5.5):
//   Silverman  median 0.1056, IQR 0.0037   (essentially deterministic)
//   LOO-CV     median 0.0272, IQR 0.0077   (slow n^(-3/10) rate ⇒ visible spread)
//   GCV        median 0.0272, IQR 0.0077   (matches LOO-CV)
//   AMISE-opt  h^* = 0.0373                (deterministic property of the DGP)
//
// Static fallback: public/images/topics/kernel-regression/13_selector_comparison.png
// =============================================================================

const TOP_HEIGHT = 280;
const BOTTOM_HEIGHT = 200;
const SM_BREAKPOINT = 640;
const SIGMA = 0.2;
const N_OPTIONS = [100, 200, 500];
const B_OPTIONS = [30, 60, 100];
const H_GRID_SIZE = 30;

type SelectorKey = 'silverman' | 'loocv' | 'gcv';

const SELECTOR_COLORS: Record<SelectorKey, string> = {
  silverman: paletteKR.band,
  loocv: paletteKR.posterior,
  gcv: paletteKR.alt,
};

const SELECTOR_LABELS: Record<SelectorKey, string> = {
  silverman: 'Silverman',
  loocv: 'LOO-CV',
  gcv: 'GCV',
};

export default function BandwidthSelectorComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(1); // default 200
  const [bIdx, setBIdx] = useState(1); // default 60
  const [enabled, setEnabled] = useState<Record<SelectorKey, boolean>>({
    silverman: true,
    loocv: true,
    gcv: true,
  });

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;
  const N = N_OPTIONS[nIdx];
  const B = B_OPTIONS[bIdx];

  const hGrid = useMemo(() => {
    const arr = new Float64Array(H_GRID_SIZE);
    const lo = Math.log(0.005);
    const hi = Math.log(0.3);
    for (let i = 0; i < H_GRID_SIZE; i++) {
      arr[i] = Math.exp(lo + ((hi - lo) * i) / (H_GRID_SIZE - 1));
    }
    return arr;
  }, []);

  // Single canonical sample for the top panel + B replicates for the bottom.
  const { topPanel, hStar, hSilvermanCanonical, selectedSilverman, selectedCv, selectedGcv } =
    useMemo(() => {
      const rngTop = mulberry32(2026);
      const { X: Xc, Y: Yc } = sampleToyUni(N, SIGMA, rngTop);
      const cvScores = new Float64Array(H_GRID_SIZE);
      const gcvScores = new Float64Array(H_GRID_SIZE);
      for (let k = 0; k < H_GRID_SIZE; k++) {
        cvScores[k] = looCvScore(Xc, Yc, hGrid[k], kGaussian);
        gcvScores[k] = gcvScore(Xc, Yc, hGrid[k], kGaussian);
      }

      // Replicate sweep for the bottom panel.
      const rngRep = mulberry32(20260901);
      const silv = new Float64Array(B);
      const cv = new Float64Array(B);
      const gcv = new Float64Array(B);
      for (let b = 0; b < B; b++) {
        const { X, Y } = sampleToyUni(N, SIGMA, rngRep);
        silv[b] = silvermanRule(X);

        let bestCv = Infinity;
        let bestCvIdx = 0;
        let bestGcv = Infinity;
        let bestGcvIdx = 0;
        for (let k = 0; k < H_GRID_SIZE; k++) {
          const sc = looCvScore(X, Y, hGrid[k], kGaussian);
          const sg = gcvScore(X, Y, hGrid[k], kGaussian);
          if (sc < bestCv) {
            bestCv = sc;
            bestCvIdx = k;
          }
          if (sg < bestGcv) {
            bestGcv = sg;
            bestGcvIdx = k;
          }
        }
        cv[b] = hGrid[bestCvIdx];
        gcv[b] = hGrid[bestGcvIdx];
      }

      const median = (arr: Float64Array) => {
        const sorted = Array.from(arr).slice().sort((a, b) => a - b);
        const m = sorted.length;
        return m % 2 === 0 ? (sorted[m / 2 - 1] + sorted[m / 2]) / 2 : sorted[Math.floor(m / 2)];
      };
      return {
        topPanel: { Xc, Yc, cvScores, gcvScores },
        hStar: hStarAmiseUni(N, SIGMA),
        hSilvermanCanonical: silvermanRule(Xc),
        selectedSilverman: silv,
        selectedCv: cv,
        selectedGcv: gcv,
        medianSilverman: median(silv),
        medianCv: median(cv),
        medianGcv: median(gcv),
      };
    }, [N, B, hGrid]);

  // Render top panel.
  const topRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!topRef.current || w <= 0) return;
    const svg = d3.select(topRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 16, bottom: 38, left: 60 };
    const innerW = w - margin.left - margin.right;
    const innerH = TOP_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLog().domain([hGrid[0], hGrid[hGrid.length - 1]]).range([0, innerW]);
    // Normalize each curve by its min so the y-axis is "× minimum".
    const cvMin = topPanel.cvScores.reduce((a, b) => Math.min(a, b), Infinity);
    const gcvMin = topPanel.gcvScores.reduce((a, b) => Math.min(a, b), Infinity);
    const cvNorm: number[] = Array.from(topPanel.cvScores).map((v) => v / cvMin);
    const gcvNorm: number[] = Array.from(topPanel.gcvScores).map((v) => v / gcvMin);
    const yMax = Math.min(2.5, Math.max(...cvNorm, ...gcvNorm) * 1.05);
    const yScale = d3.scaleLinear().domain([1, yMax]).range([innerH, 0]);

    const draw = (data: number[], color: string) => {
      const line = d3
        .line<number>()
        .defined((d) => Number.isFinite(d))
        .x((_, i) => xScale(hGrid[i]))
        .y((d) => yScale(Math.min(d, yMax)));
      g.append('path')
        .datum(data)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', color)
        .style('stroke-width', 1.6);
    };
    if (enabled.loocv) draw(cvNorm, paletteKR.posterior);
    if (enabled.gcv) draw(gcvNorm, paletteKR.alt);

    const cvIdx = cvNorm.indexOf(Math.min(...cvNorm));
    const gcvIdx = gcvNorm.indexOf(Math.min(...gcvNorm));
    const hCv = hGrid[cvIdx];
    const hGcv = hGrid[gcvIdx];

    const verticalMarker = (hVal: number, color: string, dash: string, label: string) => {
      g.append('line')
        .attr('x1', xScale(hVal))
        .attr('y1', 0)
        .attr('x2', xScale(hVal))
        .attr('y2', innerH)
        .style('stroke', color)
        .style('stroke-dasharray', dash)
        .style('stroke-width', 1.4);
      g.append('text')
        .attr('x', xScale(hVal) + 4)
        .attr('y', 14)
        .style('fill', color)
        .style('font-size', '10px')
        .style('font-weight', '500')
        .text(label);
    };
    verticalMarker(hStar, paletteKR.truth, '3 3', `h^* = ${hStar.toFixed(3)}`);
    if (enabled.silverman) verticalMarker(hSilvermanCanonical, paletteKR.band, '4 2', `h_S = ${hSilvermanCanonical.toFixed(3)}`);
    if (enabled.loocv) verticalMarker(hCv, paletteKR.posterior, '2 2', `h_CV = ${hCv.toFixed(3)}`);
    if (enabled.gcv) verticalMarker(hGcv, paletteKR.alt, '2 2', `h_GCV = ${hGcv.toFixed(3)}`);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6, '.2g'))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .text('bandwidth h (log scale)');
    g.append('text')
      .attr('x', -42)
      .attr('y', innerH / 2)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .attr('transform', `rotate(-90,-42,${innerH / 2})`)
      .text('objective / minimum');

    svg
      .append('text')
      .attr('x', w / 2)
      .attr('y', 18)
      .style('fill', 'var(--color-text)')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`Single-sample selector landscape (n = ${N})`);
  }, [w, hGrid, topPanel, hStar, hSilvermanCanonical, enabled, N]);

  // Render bottom panel — strip plot of selected h's across replicates.
  const bottomRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!bottomRef.current || w <= 0) return;
    const svg = d3.select(bottomRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 16, bottom: 36, left: 110 };
    const innerW = w - margin.left - margin.right;
    const innerH = BOTTOM_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const visible: SelectorKey[] = (['silverman', 'loocv', 'gcv'] as SelectorKey[]).filter((k) => enabled[k]);

    const xScale = d3.scaleLog().domain([0.005, 0.3]).range([0, innerW]);
    const yScale = d3.scalePoint<string>().domain(visible).range([20, innerH - 20]).padding(0.5);

    // h* vertical line for reference.
    g.append('line')
      .attr('x1', xScale(hStar))
      .attr('y1', 0)
      .attr('x2', xScale(hStar))
      .attr('y2', innerH)
      .style('stroke', paletteKR.truth)
      .style('stroke-dasharray', '3 3')
      .style('stroke-width', 1.4);

    const drawStrip = (data: Float64Array, key: SelectorKey) => {
      const yC = yScale(key)!;
      g.selectAll(`circle.${key}`)
        .data(Array.from(data))
        .enter()
        .append('circle')
        .attr('class', key)
        .attr('cx', (d) => xScale(d))
        .attr('cy', () => yC + (Math.random() - 0.5) * 12)
        .attr('r', 3.0)
        .style('fill', SELECTOR_COLORS[key])
        .style('opacity', 0.5);
    };
    if (enabled.silverman) drawStrip(selectedSilverman, 'silverman');
    if (enabled.loocv) drawStrip(selectedCv, 'loocv');
    if (enabled.gcv) drawStrip(selectedGcv, 'gcv');

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6, '.2g'))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)');
    g.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '12px');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .style('fill', 'var(--color-text-secondary)')
      .style('text-anchor', 'middle')
      .style('font-size', '11px')
      .text('selected bandwidth h (log scale)');

    svg
      .append('text')
      .attr('x', w / 2)
      .attr('y', 18)
      .style('fill', 'var(--color-text)')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`Stability across B = ${B} replicates`);
  }, [w, enabled, B, hStar, selectedSilverman, selectedCv, selectedGcv]);

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
    >
      <div
        className="flex flex-wrap items-center gap-4 mb-4 text-sm"
        style={{ flexDirection: isMobile ? 'column' : 'row', alignItems: isMobile ? 'stretch' : 'center' }}
      >
        <label className="flex items-center gap-2">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">n: {N}</span>
          <input
            type="range"
            min={0}
            max={N_OPTIONS.length - 1}
            step={1}
            value={nIdx}
            onChange={(e) => setNIdx(Number(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">B: {B}</span>
          <input
            type="range"
            min={0}
            max={B_OPTIONS.length - 1}
            step={1}
            value={bIdx}
            onChange={(e) => setBIdx(Number(e.target.value))}
            className="accent-[var(--color-accent)]"
          />
        </label>
        <div className="flex items-center gap-3 ml-auto text-xs">
          {(['silverman', 'loocv', 'gcv'] as SelectorKey[]).map((k) => (
            <label key={k} className="flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                checked={enabled[k]}
                onChange={(e) => setEnabled({ ...enabled, [k]: e.target.checked })}
                className="accent-[var(--color-accent)]"
              />
              <span style={{ color: SELECTOR_COLORS[k], fontWeight: 500 }}>
                {SELECTOR_LABELS[k]}
              </span>
            </label>
          ))}
        </div>
      </div>

      <svg ref={topRef} width={w} height={TOP_HEIGHT} />
      <svg ref={bottomRef} width={w} height={BOTTOM_HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Top: a single sample's LOO-CV and GCV objective curves nearly coincide; their
        minimizers (vertical markers) sit close to the analytical h^* (dashed red). Silverman's
        rule lands well to the right — oversmoothed because it targets density estimation, not
        regression. Bottom: stability across B replicates. Silverman is essentially
        deterministic; LOO-CV and GCV inherit the slow n^(-3/10) variance and spread out, but
        their median tracks h^*.
      </p>
    </div>
  );
}

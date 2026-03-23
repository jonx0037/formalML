import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { getRobustPCAResults } from '../../data/robust-pca-data';
import type { RobustPCAResult } from '../../data/robust-pca-data';

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const MARGIN = { top: 24, right: 8, bottom: 24, left: 32 };
const HEATMAP_GAP = 12;
const SV_HEIGHT = 140;

// ─── Available parameter values ───

const RANKS = [1, 2, 3, 5, 8];
const CORRUPTIONS = [0.05, 0.10, 0.15, 0.20, 0.30];

// ─── Helpers ───

function findNearest(results: RobustPCAResult[], rank: number, corruption: number): RobustPCAResult {
  let best = results[0];
  let bestDist = Infinity;
  for (const r of results) {
    const d = Math.abs(r.rank - rank) + Math.abs(r.corruption - corruption) * 20;
    if (d < bestDist) {
      bestDist = d;
      best = r;
    }
  }
  return best;
}

// ─── Component ───

export default function RobustPCADemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const heatmapRef = useRef<SVGSVGElement>(null);
  const spectrumRef = useRef<SVGSVGElement>(null);

  const [rank, setRank] = useState(3);
  const [corruption, setCorruption] = useState(0.10);

  const handleRankChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setRank(RANKS[val] ?? 3);
  }, []);

  const handleCorruptionChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setCorruption(CORRUPTIONS[val] ?? 0.10);
  }, []);

  // Find the nearest pre-computed result
  const results = useMemo(() => getRobustPCAResults(), []);
  const result = useMemo(() => findNearest(results, rank, corruption), [results, rank, corruption]);

  const isSmall = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // ─── Matrix extent for shared color scale ───
  const matrixExtent = useMemo(() => {
    let maxAbs = 0;
    for (const row of result.matrixX) {
      for (const v of row) {
        maxAbs = Math.max(maxAbs, Math.abs(v));
      }
    }
    return maxAbs || 1;
  }, [result]);

  const sparseExtent = useMemo(() => {
    let maxAbs = 0;
    for (const row of result.matrixS) {
      for (const v of row) {
        maxAbs = Math.max(maxAbs, Math.abs(v));
      }
    }
    return maxAbs || 1;
  }, [result]);

  // ─── Heatmap rendering ───
  useEffect(() => {
    if (!heatmapRef.current || containerWidth === 0) return;
    const svg = d3.select(heatmapRef.current);
    svg.selectAll('*').remove();

    const nRows = result.matrixX.length;
    const nCols = result.matrixX[0]?.length ?? 0;
    if (nRows === 0 || nCols === 0) return;

    const matrices = [
      { data: result.matrixX, label: 'X = L + S (observed)' },
      { data: result.matrixL, label: 'L (low-rank)' },
      { data: result.matrixS, label: 'S (sparse)' },
    ];

    const availW = containerWidth - 2 * MARGIN.left;
    const panelW = isSmall
      ? availW
      : Math.floor((availW - 2 * HEATMAP_GAP) / 3);
    const cellW = Math.max(1, Math.floor(panelW / nCols));
    const cellH = Math.max(1, Math.floor((cellW * nRows) / nCols));
    const actualPanelW = cellW * nCols;
    const actualPanelH = cellH * nRows;

    const totalW = isSmall ? actualPanelW + 2 * MARGIN.left : actualPanelW * 3 + 2 * HEATMAP_GAP + 2 * MARGIN.left;
    const totalH = isSmall
      ? (actualPanelH + MARGIN.top + 8) * 3
      : actualPanelH + MARGIN.top + MARGIN.bottom;

    svg.attr('width', totalW).attr('height', totalH);

    // Diverging color scale (blue–white–red)
    const colorMain = d3.scaleSequential(d3.interpolateRdBu)
      .domain([matrixExtent, -matrixExtent]);

    const colorSparse = d3.scaleSequential(d3.interpolateRdBu)
      .domain([sparseExtent, -sparseExtent]);

    matrices.forEach((mat, mi) => {
      const offsetX = isSmall ? MARGIN.left : MARGIN.left + mi * (actualPanelW + HEATMAP_GAP);
      const offsetY = isSmall ? mi * (actualPanelH + MARGIN.top + 8) + MARGIN.top : MARGIN.top;
      const colorFn = mi === 2 ? colorSparse : colorMain;

      const g = svg.append('g').attr('transform', `translate(${offsetX}, ${offsetY})`);

      // Title
      g.append('text')
        .attr('x', actualPanelW / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .attr('font-size', 10)
        .attr('font-weight', 600)
        .style('fill', 'var(--color-text)')
        .text(mat.label);

      // Cells
      for (let r = 0; r < nRows; r++) {
        for (let c = 0; c < nCols; c++) {
          g.append('rect')
            .attr('x', c * cellW)
            .attr('y', r * cellH)
            .attr('width', cellW)
            .attr('height', cellH)
            .attr('fill', colorFn(mat.data[r][c]));
        }
      }
    });
  }, [result, containerWidth, isSmall, matrixExtent, sparseExtent]);

  // ─── Singular value spectrum rendering ───
  useEffect(() => {
    if (!spectrumRef.current || containerWidth === 0) return;
    const svg = d3.select(spectrumRef.current);
    svg.selectAll('*').remove();

    const svX = result.singularValuesX;
    const svL = result.singularValuesL;
    const maxSV = Math.max(...svX, ...svL, 1);
    const n = Math.max(svX.length, svL.length);

    const w = Math.min(containerWidth - 2 * MARGIN.left, 500);
    const h = SV_HEIGHT;

    svg.attr('width', w + MARGIN.left + MARGIN.right)
       .attr('height', h + MARGIN.top + MARGIN.bottom);

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain([1, n]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, maxSV * 1.1]).range([h, 0]);

    // Axes
    const xTicks = d3.range(1, n + 1, Math.max(1, Math.floor(n / 8)));
    xTicks.forEach((t) => {
      g.append('line')
        .attr('x1', xScale(t)).attr('y1', 0)
        .attr('x2', xScale(t)).attr('y2', h)
        .style('stroke', 'var(--color-border)').attr('stroke-opacity', 0.2);
      g.append('text')
        .attr('x', xScale(t)).attr('y', h + 14)
        .attr('text-anchor', 'middle')
        .attr('font-size', 9)
        .style('fill', 'var(--color-text)')
        .attr('opacity', 0.5)
        .text(t);
    });

    // Y-axis ticks
    const yTicks = yScale.ticks(4);
    yTicks.forEach((t) => {
      g.append('line')
        .attr('x1', 0).attr('y1', yScale(t))
        .attr('x2', w).attr('y2', yScale(t))
        .style('stroke', 'var(--color-border)').attr('stroke-opacity', 0.2);
      g.append('text')
        .attr('x', -6).attr('y', yScale(t) + 3)
        .attr('text-anchor', 'end')
        .attr('font-size', 9)
        .style('fill', 'var(--color-text)')
        .attr('opacity', 0.5)
        .text(t.toFixed(1));
    });

    // Plot singular values of X
    const lineX = d3.line<number>()
      .x((_, i) => xScale(i + 1))
      .y((d) => yScale(d));

    g.append('path')
      .datum(svX)
      .attr('d', lineX)
      .attr('fill', 'none')
      .attr('stroke', '#6b7280')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,3');

    g.selectAll('.dot-x')
      .data(svX)
      .enter()
      .append('circle')
      .attr('cx', (_, i) => xScale(i + 1))
      .attr('cy', (d) => yScale(d))
      .attr('r', 3)
      .attr('fill', '#6b7280');

    // Plot singular values of L
    g.append('path')
      .datum(svL)
      .attr('d', lineX)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2);

    g.selectAll('.dot-l')
      .data(svL)
      .enter()
      .append('circle')
      .attr('cx', (_, i) => xScale(i + 1))
      .attr('cy', (d) => yScale(d))
      .attr('r', 3)
      .attr('fill', '#3b82f6');

    // True rank line
    g.append('line')
      .attr('x1', xScale(result.rank + 0.5))
      .attr('y1', 0)
      .attr('x2', xScale(result.rank + 0.5))
      .attr('y2', h)
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.7);

    // Labels
    g.append('text')
      .attr('x', w / 2).attr('y', h + 28)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .style('fill', 'var(--color-text)')
      .text('Component index');

    // Legend
    const legendY = -8;
    g.append('line').attr('x1', w - 160).attr('y1', legendY).attr('x2', w - 140).attr('y2', legendY)
      .attr('stroke', '#6b7280').attr('stroke-width', 2).attr('stroke-dasharray', '4,3');
    g.append('text').attr('x', w - 136).attr('y', legendY + 3).attr('font-size', 9)
      .style('fill', 'var(--color-text)').text('X (corrupted)');

    g.append('line').attr('x1', w - 80).attr('y1', legendY).attr('x2', w - 60).attr('y2', legendY)
      .attr('stroke', '#3b82f6').attr('stroke-width', 2);
    g.append('text').attr('x', w - 56).attr('y', legendY + 3).attr('font-size', 9)
      .style('fill', 'var(--color-text)').text('L (recovered)');

  }, [result, containerWidth]);

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-end">
        <div className="flex-1 min-w-[200px]">
          <label
            className="text-xs font-medium block mb-1"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            True rank: {rank}
          </label>
          <input
            type="range"
            min={0}
            max={RANKS.length - 1}
            step={1}
            value={RANKS.indexOf(rank)}
            onChange={handleRankChange}
            className="w-full"
          />
          <div className="flex justify-between text-[9px] opacity-40" style={{ fontFamily: 'var(--font-sans)' }}>
            {RANKS.map((r) => (
              <span key={r}>{r}</span>
            ))}
          </div>
        </div>

        <div className="flex-1 min-w-[200px]">
          <label
            className="text-xs font-medium block mb-1"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Corruption: {(corruption * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min={0}
            max={CORRUPTIONS.length - 1}
            step={1}
            value={CORRUPTIONS.indexOf(corruption)}
            onChange={handleCorruptionChange}
            className="w-full"
          />
          <div className="flex justify-between text-[9px] opacity-40" style={{ fontFamily: 'var(--font-sans)' }}>
            {CORRUPTIONS.map((c) => (
              <span key={c}>{(c * 100).toFixed(0)}%</span>
            ))}
          </div>
        </div>
      </div>

      {/* Heatmaps */}
      <div className="overflow-x-auto">
        <svg ref={heatmapRef} />
      </div>

      {/* Singular value spectrum */}
      <div className="overflow-x-auto">
        <svg ref={spectrumRef} />
      </div>

      {/* Readout */}
      <div
        className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <div
          className="rounded-md px-3 py-2"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
        >
          <div className="opacity-50 mb-0.5">Recovered rank</div>
          <div className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
            {result.recoveredRank}
            <span className="opacity-40 font-normal"> / {rank} true</span>
          </div>
        </div>

        <div
          className="rounded-md px-3 py-2"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
        >
          <div className="opacity-50 mb-0.5">Nonzero in S</div>
          <div className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
            {result.nnzS}
            <span className="opacity-40 font-normal"> / {30 * 25} entries</span>
          </div>
        </div>

        <div
          className="rounded-md px-3 py-2"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
        >
          <div className="opacity-50 mb-0.5">Corruption</div>
          <div className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
            {(result.corruption * 100).toFixed(0)}%
          </div>
        </div>

        <div
          className="rounded-md px-3 py-2"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
        >
          <div className="opacity-50 mb-0.5">Relative error</div>
          <div
            className="text-sm font-semibold"
            style={{ color: result.relativeError < 0.1 ? '#16a34a' : result.relativeError < 0.3 ? '#ca8a04' : '#dc2626' }}
          >
            {(result.relativeError * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}

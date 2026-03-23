import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { getCPResults, type CPResult } from '../../data/tensor-decompositions-data';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const HEAT_MARGIN = { top: 20, right: 4, bottom: 4, left: 4 };
const BAR_MARGIN = { top: 20, right: 16, bottom: 28, left: 40 };
const BAR_HEIGHT = 180;

// ─── Component ───

export default function CPDecompositionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [rank, setRank] = useState(1);
  const [sliceIndex, setSliceIndex] = useState(0);

  const heatOrigRef = useRef<SVGSVGElement>(null);
  const heatApproxRef = useRef<SVGSVGElement>(null);
  const heatResidRef = useRef<SVGSVGElement>(null);
  const barRef = useRef<SVGSVGElement>(null);

  // ─── Data ───

  const results = useMemo(() => getCPResults(), []);
  const fullResult = results[results.length - 1]; // rank-5 = "original"
  const currentResult = results[rank - 1];

  const numSlices = fullResult.reconstruction[0][0].length;
  const numRows = fullResult.reconstruction.length;
  const numCols = fullResult.reconstruction[0].length;

  // Extract frontal slices
  const originalSlice = useMemo(() => {
    return fullResult.reconstruction.map((row) => row.map((col) => col[sliceIndex]));
  }, [fullResult, sliceIndex]);

  const approxSlice = useMemo(() => {
    return currentResult.reconstruction.map((row) => row.map((col) => col[sliceIndex]));
  }, [currentResult, sliceIndex]);

  const residualSlice = useMemo(() => {
    return originalSlice.map((row, i) =>
      row.map((v, j) => v - approxSlice[i][j]),
    );
  }, [originalSlice, approxSlice]);

  // ─── Heatmap sizing ───

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 200;
    return containerWidth >= SM_BREAKPOINT
      ? Math.floor((containerWidth - 24) / 2)
      : containerWidth;
  }, [containerWidth]);

  const cellSize = useMemo(() => {
    if (!containerWidth) return 10;
    const available = panelWidth - HEAT_MARGIN.left - HEAT_MARGIN.right - 16;
    const maxCells = Math.max(numRows, numCols);
    return Math.max(2, Math.min(Math.floor(available / maxCells), 20));
  }, [containerWidth, panelWidth, numRows, numCols]);

  const heatW = cellSize * numCols + HEAT_MARGIN.left + HEAT_MARGIN.right;
  const heatH = cellSize * numRows + HEAT_MARGIN.top + HEAT_MARGIN.bottom;

  // ─── Bar chart sizing ───

  const barWidth = useMemo(() => {
    if (!containerWidth) return 200;
    return Math.min(panelWidth - 16, 300);
  }, [containerWidth, panelWidth]);

  // ─── Color scale ───

  const colorExtent = useMemo(() => {
    const allVals = fullResult.reconstruction.flat(2);
    const absMax = Math.max(Math.abs(Math.min(...allVals)), Math.abs(Math.max(...allVals)));
    return absMax || 1;
  }, [fullResult]);

  const heatmapColor = useCallback(
    (v: number) => {
      // Reversed RdBu: negative = blue, positive = red
      const t = (v + colorExtent) / (2 * colorExtent);
      return d3.interpolateRdBu(1 - t);
    },
    [colorExtent],
  );

  // ─── Heatmap rendering ───

  const renderHeatmap = useCallback(
    (
      ref: React.RefObject<SVGSVGElement | null>,
      data: number[][],
      title: string,
      colorFn: (v: number) => string,
    ) => {
      if (!ref.current) return;
      const svg = d3.select(ref.current);
      svg.selectAll('*').remove();

      const g = svg
        .append('g')
        .attr('transform', `translate(${HEAT_MARGIN.left},${HEAT_MARGIN.top})`);

      // Cells
      data.forEach((row, i) => {
        row.forEach((v, j) => {
          g.append('rect')
            .attr('x', j * cellSize)
            .attr('y', i * cellSize)
            .attr('width', cellSize - 0.5)
            .attr('height', cellSize - 0.5)
            .attr('fill', colorFn(v));
        });
      });

      // Title
      g.append('text')
        .attr('x', (numCols * cellSize) / 2)
        .attr('y', -6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('font-weight', 500)
        .text(title);
    },
    [cellSize, numCols],
  );

  // ─── Residual color scale (separate, symmetric around 0) ───

  const residualColor = useCallback(
    (v: number) => {
      const flat = residualSlice.flat();
      const ext = Math.max(
        Math.abs(Math.min(...flat)),
        Math.abs(Math.max(...flat)),
        0.01,
      );
      const t = (v + ext) / (2 * ext);
      return d3.interpolateRdBu(1 - t);
    },
    [residualSlice],
  );

  // ─── Render heatmaps ───

  useEffect(() => {
    renderHeatmap(heatOrigRef, originalSlice, `Original (slice ${sliceIndex + 1})`, heatmapColor);
    renderHeatmap(heatApproxRef, approxSlice, `Rank-${rank} approx`, heatmapColor);
    renderHeatmap(heatResidRef, residualSlice, 'Residual', residualColor);
  }, [
    rank, sliceIndex, originalSlice, approxSlice, residualSlice,
    renderHeatmap, heatmapColor, residualColor,
  ]);

  // ─── Error bar chart ───

  useEffect(() => {
    if (!barRef.current || barWidth === 0) return;

    const svg = d3.select(barRef.current);
    svg.selectAll('*').remove();

    const w = barWidth - BAR_MARGIN.left - BAR_MARGIN.right;
    const h = BAR_HEIGHT - BAR_MARGIN.top - BAR_MARGIN.bottom;
    const g = svg
      .append('g')
      .attr('transform', `translate(${BAR_MARGIN.left},${BAR_MARGIN.top})`);

    const errors = results.map((r) => r.relativeError);
    const maxError = Math.max(...errors) * 1.1;

    const xScale = d3
      .scaleBand<number>()
      .domain(results.map((_, i) => i))
      .range([0, w])
      .padding(0.2);

    const yScale = d3.scaleLinear().domain([0, maxError]).range([h, 0]);

    // Bars
    results.forEach((r, i) => {
      g.append('rect')
        .attr('x', xScale(i)!)
        .attr('y', yScale(r.relativeError))
        .attr('width', xScale.bandwidth())
        .attr('height', h - yScale(r.relativeError))
        .attr('rx', 2)
        .style(
          'fill',
          i === rank - 1 ? 'var(--color-accent)' : 'var(--color-muted)',
        )
        .attr('opacity', i === rank - 1 ? 0.9 : 0.3);
    });

    // X axis labels
    results.forEach((r, i) => {
      g.append('text')
        .attr('x', xScale(i)! + xScale.bandwidth() / 2)
        .attr('y', h + 14)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-muted)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 9)
        .text(`R=${r.rank}`);
    });

    // Y axis
    const yAxis = d3.axisLeft(yScale).ticks(5).tickSize(-w);
    g.append('g')
      .call(yAxis)
      .call((sel) => sel.select('.domain').remove())
      .call((sel) =>
        sel
          .selectAll('.tick line')
          .style('stroke', 'var(--color-muted)')
          .attr('opacity', 0.2),
      )
      .call((sel) =>
        sel
          .selectAll('.tick text')
          .style('fill', 'var(--color-muted)')
          .attr('font-size', 9),
      );

    // X axis label
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 24)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .text('CP Rank');

    // Title
    g.append('text')
      .attr('x', w / 2)
      .attr('y', -6)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Relative Error by Rank');
  }, [rank, barWidth, results]);

  // ─── Render ───

  const onRankChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => setRank(parseInt(e.target.value)),
    [],
  );

  const onSliceChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => setSliceIndex(parseInt(e.target.value)),
    [],
  );

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Top row: Original + Approximation */}
      <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
        <div className="flex flex-col items-center">
          <svg
            ref={heatOrigRef}
            width={heatW}
            height={heatH}
            className="rounded border border-[var(--color-border)]"
          />
          <span
            className="mt-1 text-[10px]"
            style={{ color: 'var(--color-muted)', fontFamily: 'var(--font-sans)' }}
          >
            range: [{(-colorExtent).toFixed(1)}, {colorExtent.toFixed(1)}]
          </span>
        </div>
        <div className="flex flex-col items-center">
          <svg
            ref={heatApproxRef}
            width={heatW}
            height={heatH}
            className="rounded border border-[var(--color-border)]"
          />
          <span
            className="mt-1 text-[10px]"
            style={{ color: 'var(--color-muted)', fontFamily: 'var(--font-sans)' }}
          >
            rank-{rank} reconstruction
          </span>
        </div>
      </div>

      {/* Bottom row: Residual + Error chart */}
      <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
        <div className="flex flex-col items-center">
          <svg
            ref={heatResidRef}
            width={heatW}
            height={heatH}
            className="rounded border border-[var(--color-border)]"
          />
          <span
            className="mt-1 text-[10px]"
            style={{ color: 'var(--color-muted)', fontFamily: 'var(--font-sans)' }}
          >
            original − approximation
          </span>
        </div>
        <div className="flex flex-col items-center">
          <svg
            ref={barRef}
            width={barWidth}
            height={BAR_HEIGHT}
            className="rounded-lg border border-[var(--color-border)]"
          />
        </div>
      </div>

      {/* Error readout */}
      <div
        className="flex flex-wrap gap-x-6 gap-y-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-xs"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <span>
          Relative error: <strong>{(currentResult.relativeError * 100).toFixed(2)}%</strong>
        </span>
        <span>
          Rank-1 components: <strong>{rank}</strong> of 5
        </span>
        <span>
          Frontal slice: <strong>{sliceIndex + 1}</strong> of {numSlices}
        </span>
      </div>

      {/* Controls */}
      <div className="flex flex-col gap-2 sm:flex-row sm:gap-6">
        <div className="flex flex-1 items-center gap-3">
          <label
            className="min-w-[80px] whitespace-nowrap text-xs font-medium"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Rank R = {rank}
          </label>
          <input
            type="range"
            min={1}
            max={5}
            step={1}
            value={rank}
            onChange={onRankChange}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
        <div className="flex flex-1 items-center gap-3">
          <label
            className="min-w-[80px] whitespace-nowrap text-xs font-medium"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Slice k = {sliceIndex + 1}
          </label>
          <input
            type="range"
            min={0}
            max={numSlices - 1}
            step={1}
            value={sliceIndex}
            onChange={onSliceChange}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
      </div>
    </div>
  );
}

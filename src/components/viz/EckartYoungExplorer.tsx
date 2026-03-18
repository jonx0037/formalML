import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  ROWS, COLS, singularValues, originalMatrix, computeMatrix,
} from '../../data/svd-eckart-young-data';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const BAR_HEIGHT = 200;
const HEATMAP_SIZE = 180;
const BAR_MARGIN = { top: 20, right: 12, bottom: 28, left: 36 };
const HEAT_MARGIN = { top: 16, right: 4, bottom: 4, left: 4 };

// ─── Color scales ───

function sequentialScale(min: number, max: number) {
  return d3.scaleSequential(d3.interpolateYlOrRd).domain([min, max]);
}

function divergingScale(extent: number) {
  return d3.scaleSequential(d3.interpolateRdBu).domain([extent, -extent]);
}

// ─── Component ───

export default function EckartYoungExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const barRef = useRef<SVGSVGElement>(null);
  const heatOrigRef = useRef<SVGSVGElement>(null);
  const heatApproxRef = useRef<SVGSVGElement>(null);
  const heatDiffRef = useRef<SVGSVGElement>(null);

  const [rank, setRank] = useState(4);

  // ─── Computed data ───

  const approxMatrix = useMemo(() => computeMatrix(rank), [rank]);

  const diffMatrix = useMemo(
    () =>
      originalMatrix.map((row, i) =>
        row.map((v, j) => v - approxMatrix[i][j]),
      ),
    [approxMatrix],
  );

  // Statistics (computed directly from singular values — no matrix needed)
  const stats = useMemo(() => {
    const totalEnergy = singularValues.reduce((s, v) => s + v * v, 0);
    const retainedEnergy = singularValues.slice(0, rank).reduce((s, v) => s + v * v, 0);
    const frobError = Math.sqrt(
      singularValues.slice(rank).reduce((s, v) => s + v * v, 0),
    );
    const spectralError = rank < COLS ? singularValues[rank] : 0;
    const storageOriginal = ROWS * COLS;
    const storageTruncated = rank * (ROWS + COLS + 1);

    return {
      energyPct: totalEnergy > 0 ? (retainedEnergy / totalEnergy) * 100 : 100,
      frobError,
      spectralError,
      storageOriginal,
      storageTruncated,
      compressionRatio: storageTruncated > 0 ? storageOriginal / storageTruncated : Infinity,
    };
  }, [rank]);

  // ─── Bar chart width ───

  const barWidth = useMemo(() => {
    if (!containerWidth) return 200;
    return Math.min(
      containerWidth >= SM_BREAKPOINT ? Math.floor(containerWidth * 0.3) : containerWidth - 16,
      260,
    );
  }, [containerWidth]);

  // ─── Heatmap cell size ───

  const cellSize = useMemo(() => {
    if (!containerWidth) return 10;
    const available = containerWidth >= SM_BREAKPOINT
      ? (containerWidth - barWidth - 40) / 3
      : containerWidth - 16;
    const maxCells = Math.max(ROWS, COLS);
    return Math.max(1, Math.min(Math.floor((Math.min(available, HEATMAP_SIZE) - HEAT_MARGIN.left - HEAT_MARGIN.right) / maxCells), 14));
  }, [containerWidth, barWidth]);

  const heatW = cellSize * COLS + HEAT_MARGIN.left + HEAT_MARGIN.right;
  const heatH = cellSize * ROWS + HEAT_MARGIN.top + HEAT_MARGIN.bottom;

  // ─── Singular value bar chart ───

  useEffect(() => {
    if (!barRef.current || barWidth === 0) return;

    const svg = d3.select(barRef.current);
    svg.selectAll('*').remove();

    const w = barWidth - BAR_MARGIN.left - BAR_MARGIN.right;
    const h = BAR_HEIGHT - BAR_MARGIN.top - BAR_MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${BAR_MARGIN.left},${BAR_MARGIN.top})`);

    const xScale = d3.scaleBand<number>()
      .domain(singularValues.map((_, i) => i))
      .range([0, w])
      .padding(0.15);

    const yScale = d3.scaleLinear()
      .domain([0, singularValues[0] * 1.05])
      .range([h, 0]);

    // Bars
    singularValues.forEach((s, i) => {
      g.append('rect')
        .attr('x', xScale(i)!)
        .attr('y', yScale(s))
        .attr('width', xScale.bandwidth())
        .attr('height', h - yScale(s))
        .attr('rx', 1)
        .style('fill', i < rank ? 'var(--color-accent)' : 'var(--color-muted)')
        .attr('opacity', i < rank ? 0.9 : 0.3);
    });

    // Rank divider line
    if (rank < COLS) {
      const xPos = xScale(rank)! - xScale.step() * xScale.paddingInner() / 2;
      g.append('line')
        .attr('x1', xPos).attr('y1', 0)
        .attr('x2', xPos).attr('y2', h)
        .attr('stroke', '#ef4444')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3');
    }

    // Y axis
    const yAxis = d3.axisLeft(yScale).ticks(5).tickSize(-w);
    g.append('g')
      .call(yAxis)
      .call((g) => g.select('.domain').remove())
      .call((g) => g.selectAll('.tick line').style('stroke', 'var(--color-muted)').attr('opacity', 0.2))
      .call((g) => g.selectAll('.tick text').style('fill', 'var(--color-muted)').attr('font-size', 9));

    // X axis label
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 20)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .text('Singular value index');

    // Title
    g.append('text')
      .attr('x', w / 2)
      .attr('y', -6)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text(`σ₁…σ₁₂  (rank ${rank})`);
  }, [rank, barWidth]);

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

      const g = svg.append('g').attr('transform', `translate(${HEAT_MARGIN.left},${HEAT_MARGIN.top})`);

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
        .attr('x', (COLS * cellSize) / 2)
        .attr('y', -4)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('font-weight', 500)
        .text(title);
    },
    [cellSize],
  );

  // Original matrix range is constant — memoize once
  const [origMin, origMax] = useMemo(() => {
    const flat = originalMatrix.flat();
    return [Math.min(...flat), Math.max(...flat)];
  }, []);

  useEffect(() => {
    const diffFlat = diffMatrix.flat();
    const diffExtent = Math.max(Math.abs(Math.min(...diffFlat)), Math.abs(Math.max(...diffFlat)), 0.01);

    const origColor = sequentialScale(origMin, origMax);
    const approxColor = sequentialScale(origMin, origMax);
    const diffColor = divergingScale(diffExtent);

    renderHeatmap(heatOrigRef, originalMatrix, 'Original A', (v) => origColor(v));
    renderHeatmap(heatApproxRef, approxMatrix, `Rank-${rank} Aₖ`, (v) => approxColor(v));
    renderHeatmap(heatDiffRef, diffMatrix, 'Difference A − Aₖ', (v) => diffColor(v));
  }, [rank, approxMatrix, diffMatrix, renderHeatmap, origMin, origMax]);

  // ─── Render ───

  return (
    <div ref={containerRef} className="w-full space-y-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        {/* Bar chart */}
        <svg
          ref={barRef}
          width={barWidth}
          height={BAR_HEIGHT}
          className="shrink-0 rounded-lg border border-[var(--color-border)]"
        />

        {/* Heatmaps */}
        <div className="flex flex-wrap justify-center gap-2">
          <svg ref={heatOrigRef} width={heatW} height={heatH} className="rounded border border-[var(--color-border)]" />
          <svg ref={heatApproxRef} width={heatW} height={heatH} className="rounded border border-[var(--color-border)]" />
          <svg ref={heatDiffRef} width={heatW} height={heatH} className="rounded border border-[var(--color-border)]" />
        </div>
      </div>

      {/* Statistics */}
      <div
        className="flex flex-wrap gap-x-6 gap-y-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-xs"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <span>‖A − A<sub>k</sub>‖<sub>F</sub> = <strong>{stats.frobError.toFixed(3)}</strong></span>
        <span>‖A − A<sub>k</sub>‖<sub>2</sub> = <strong>{stats.spectralError.toFixed(3)}</strong></span>
        <span>Energy retained: <strong>{stats.energyPct.toFixed(1)}%</strong></span>
        <span>Storage: <strong>{stats.storageTruncated}</strong> / {stats.storageOriginal} ({stats.compressionRatio.toFixed(1)}× compression)</span>
      </div>

      {/* Rank slider */}
      <div className="flex items-center gap-3">
        <label
          className="min-w-[100px] whitespace-nowrap text-xs font-medium"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Truncation rank k = {rank}
        </label>
        <input
          type="range"
          min={1}
          max={COLS}
          step={1}
          value={rank}
          onChange={useCallback((e: React.ChangeEvent<HTMLInputElement>) => setRank(parseInt(e.target.value)), [])}
          className="w-full accent-[var(--color-accent)]"
        />
      </div>
    </div>
  );
}

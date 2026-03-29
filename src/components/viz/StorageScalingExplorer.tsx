import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 20, right: 24, bottom: 48, left: 64 };
const SVG_HEIGHT = 360;

const FORMATS = [
  { key: 'full', label: 'Full', color: '#94a3b8' },
  { key: 'tucker', label: 'Tucker', color: '#f97316' },
  { key: 'cp', label: 'CP', color: '#22c55e' },
  { key: 'tt', label: 'Tensor Train', color: '#8b5cf6' },
] as const;

type FormatKey = (typeof FORMATS)[number]['key'];

function computeStorage(format: FormatKey, N: number, I: number, r: number): number {
  switch (format) {
    case 'full': return Math.pow(I, N);
    case 'tucker': return Math.pow(r, N) + N * I * r;
    case 'cp': return N * I * r;
    case 'tt': return N * I * r * r;
  }
}

function formatTick(value: number): string {
  if (value >= 1e12) return `${(value / 1e12).toPrecision(2)}T`;
  if (value >= 1e9) return `${(value / 1e9).toPrecision(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toPrecision(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toPrecision(2)}K`;
  return String(Math.round(value));
}

function formatExact(value: number): string {
  if (value >= 1e12) return `${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toLocaleString();
}

const N_VALUES = d3.range(3, 13); // 3 to 12

export default function StorageScalingExplorer() {
  const { ref, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [modeSize, setModeSize] = useState(10);
  const [rank, setRank] = useState(3);
  const [visible, setVisible] = useState<Record<FormatKey, boolean>>({
    full: true, tucker: true, cp: true, tt: true,
  });
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; format: string; N: number; value: number;
  } | null>(null);

  const totalWidth = containerWidth || 700;
  const innerW = totalWidth - MARGIN.left - MARGIN.right;
  const innerH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  const seriesData = useMemo(() => {
    return FORMATS.map((fmt) => ({
      ...fmt,
      points: N_VALUES.map((N) => ({
        N,
        value: computeStorage(fmt.key, N, modeSize, rank),
      })),
    }));
  }, [modeSize, rank]);

  const toggleFormat = useCallback((key: FormatKey) => {
    setVisible((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  // D3 chart rendering
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    if (!svgRef.current || innerW <= 0) return;
    svg.selectAll('*').remove();

    // Scales
    const xScale = d3.scaleLinear().domain([3, 12]).range([MARGIN.left, MARGIN.left + innerW]);

    const visibleSeries = seriesData.filter((s) => visible[s.key]);
    const allValues = visibleSeries.flatMap((s) => s.points.map((p) => p.value));
    const yMin = Math.max(1, d3.min(allValues) ?? 1);
    const yMax = d3.max(allValues) ?? 1e6;

    const yScale = d3.scaleLog().domain([yMin * 0.5, yMax * 2]).range([MARGIN.top + innerH, MARGIN.top]).nice();

    const g = svg.append('g');

    // Grid lines (y-axis, log scale)
    const yTicks = yScale.ticks(6);
    yTicks.forEach((t) => {
      g.append('line')
        .attr('x1', MARGIN.left)
        .attr('x2', MARGIN.left + innerW)
        .attr('y1', yScale(t))
        .attr('y2', yScale(t))
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', 0.3);
    });

    // Grid lines (x-axis)
    N_VALUES.forEach((n) => {
      g.append('line')
        .attr('x1', xScale(n))
        .attr('x2', xScale(n))
        .attr('y1', MARGIN.top)
        .attr('y2', MARGIN.top + innerH)
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', 0.15);
    });

    // Lines and dots for each visible format
    visibleSeries.forEach((series) => {
      const lineGen = d3.line<{ N: number; value: number }>()
        .x((d) => xScale(d.N))
        .y((d) => yScale(d.value));

      g.append('path')
        .datum(series.points)
        .attr('d', lineGen)
        .style('fill', 'none')
        .style('stroke', series.color)
        .style('stroke-width', 2);

      g.selectAll(`.dot-${series.key}`)
        .data(series.points)
        .join('circle')
        .attr('cx', (d) => xScale(d.N))
        .attr('cy', (d) => yScale(d.value))
        .attr('r', 4)
        .style('fill', series.color)
        .style('cursor', 'pointer')
        .each(function (d) {
          d3.select(this)
            .on('mouseenter', (event: MouseEvent) => {
              setTooltip({
                x: event.offsetX,
                y: event.offsetY,
                format: series.label,
                N: d.N,
                value: d.value,
              });
            })
            .on('mouseleave', () => setTooltip(null));
        });
    });

    // X-axis tick labels
    N_VALUES.forEach((n) => {
      g.append('text')
        .attr('x', xScale(n))
        .attr('y', MARGIN.top + innerH + 16)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text)')
        .style('opacity', 0.6)
        .style('font-family', 'var(--font-sans)')
        .text(n);
    });

    // Y-axis tick labels
    yTicks.forEach((t) => {
      g.append('text')
        .attr('x', MARGIN.left - 6)
        .attr('y', yScale(t) + 3)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text)')
        .style('opacity', 0.6)
        .style('font-family', 'var(--font-sans)')
        .text(formatTick(t));
    });

    // Axis labels
    g.append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', MARGIN.top + innerH + 38)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.8)
      .style('font-family', 'var(--font-sans)')
      .text('Number of modes (N)');

    g.append('text')
      .attr('x', 14)
      .attr('y', MARGIN.top + innerH / 2)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, 14, ${MARGIN.top + innerH / 2})`)
      .style('font-size', '11px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.8)
      .style('font-family', 'var(--font-sans)')
      .text('Storage (entries)');

    // Legend
    const legendX = MARGIN.left + 8;
    const legendY = MARGIN.top + 4;
    FORMATS.forEach((fmt, i) => {
      const opacity = visible[fmt.key] ? 1 : 0.25;
      g.append('circle')
        .attr('cx', legendX)
        .attr('cy', legendY + i * 18)
        .attr('r', 5)
        .style('fill', fmt.color)
        .style('opacity', opacity);
      g.append('text')
        .attr('x', legendX + 10)
        .attr('y', legendY + i * 18 + 4)
        .style('font-size', '10px')
        .style('fill', 'var(--color-text)')
        .style('opacity', opacity * 0.8)
        .style('font-family', 'var(--font-sans)')
        .text(fmt.label);
    });
  }, [seriesData, visible, innerW, innerH]);

  return (
    <div ref={ref} className="w-full space-y-3">
      {/* Chart */}
      <div className="relative">
        <svg role="img" aria-label="Storage scaling explorer visualization"
          ref={svgRef}
          width={totalWidth}
          height={SVG_HEIGHT}
          className="rounded-lg border"
          style={{ borderColor: 'var(--color-border)' }}
        />
        {tooltip && (
          <div
            className="absolute pointer-events-none rounded px-2 py-1 text-xs"
            style={{
              left: tooltip.x + 12,
              top: tooltip.y - 28,
              background: 'var(--color-surface)',
              border: '1px solid var(--color-border)',
              color: 'var(--color-text)',
              fontFamily: 'var(--font-sans)',
              whiteSpace: 'nowrap',
            }}
          >
            <strong>{tooltip.format}</strong> at N={tooltip.N}:{' '}
            {formatExact(tooltip.value)} entries
          </div>
        )}
      </div>

      {/* Controls */}
      <div
        className="flex gap-6 flex-wrap"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        {/* Mode size slider */}
        <label className="flex flex-col gap-1 text-xs min-w-[140px]">
          <span style={{ color: 'var(--color-text)', opacity: 0.8 }}>
            Mode size <em>I</em> = {modeSize}
          </span>
          <input
            type="range"
            min={4}
            max={20}
            step={1}
            value={modeSize}
            onChange={(e) => setModeSize(Number(e.target.value))}
            className="w-full"
          />
        </label>

        {/* Rank slider */}
        <label className="flex flex-col gap-1 text-xs min-w-[140px]">
          <span style={{ color: 'var(--color-text)', opacity: 0.8 }}>
            Rank <em>r</em> = {rank}
          </span>
          <input
            type="range"
            min={2}
            max={10}
            step={1}
            value={rank}
            onChange={(e) => setRank(Number(e.target.value))}
            className="w-full"
          />
        </label>
      </div>

      {/* Format checkboxes */}
      <div
        className="flex gap-4 flex-wrap text-xs"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        {FORMATS.map((fmt) => (
          <label
            key={fmt.key}
            className="flex items-center gap-1.5 cursor-pointer"
            style={{ color: 'var(--color-text)', opacity: visible[fmt.key] ? 1 : 0.5 }}
          >
            <input
              type="checkbox"
              checked={visible[fmt.key]}
              onChange={() => toggleFormat(fmt.key)}
              style={{ accentColor: fmt.color }}
            />
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: fmt.color }}
            />
            {fmt.label}
          </label>
        ))}
      </div>
    </div>
  );
}

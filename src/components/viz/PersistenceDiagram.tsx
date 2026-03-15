import { useMemo } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import type { PersistenceInterval } from './shared/types';
import * as d3 from 'd3';

interface PersistenceDiagramProps {
  intervals: PersistenceInterval[];
  currentEpsilon?: number;
  showDiagonal?: boolean;
  highlightDimension?: number | null;
  mode?: 'diagram' | 'barcode';
}

export default function PersistenceDiagram({
  intervals,
  currentEpsilon,
  showDiagonal = true,
  highlightDimension = null,
  mode = 'diagram',
}: PersistenceDiagramProps) {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 400;
  const height = mode === 'barcode' ? Math.max(200, intervals.length * 16 + 60) : 400;
  const margin = { top: 30, right: 30, bottom: 50, left: 50 };

  const maxVal = useMemo(() => {
    const deaths = intervals.map((i) => (i.death === Infinity ? 0 : i.death));
    const births = intervals.map((i) => i.birth);
    return Math.max(...deaths, ...births) * 1.15 || 1;
  }, [intervals]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();

      if (mode === 'diagram') {
        renderDiagram(svg);
      } else {
        renderBarcode(svg);
      }
    },
    [intervals, currentEpsilon, highlightDimension, mode, width, height, maxVal],
  );

  function renderDiagram(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const xScale = d3.scaleLinear().domain([0, maxVal]).range([margin.left, width - margin.right]);
    const yScale = d3.scaleLinear().domain([0, maxVal]).range([height - margin.bottom, margin.top]);

    // Diagonal line
    if (showDiagonal) {
      svg
        .append('line')
        .attr('x1', xScale(0))
        .attr('y1', yScale(0))
        .attr('x2', xScale(maxVal))
        .attr('y2', yScale(maxVal))
        .attr('stroke', '#ccc')
        .attr('stroke-dasharray', '4,4');
    }

    // Axes
    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('font-size', '11px');

    svg
      .append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('font-size', '11px');

    // Axis labels
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', height - 8)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-family', 'var(--font-sans)')
      .text('Birth');

    svg
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-family', 'var(--font-sans)')
      .text('Death');

    // Current epsilon sweep line
    if (currentEpsilon !== undefined) {
      svg
        .append('line')
        .attr('x1', xScale(currentEpsilon))
        .attr('y1', yScale(0))
        .attr('x2', xScale(currentEpsilon))
        .attr('y2', yScale(maxVal))
        .attr('stroke', '#0F6E56')
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.5)
        .attr('stroke-dasharray', '6,3');

      svg
        .append('text')
        .attr('x', xScale(currentEpsilon))
        .attr('y', margin.top - 8)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', '#0F6E56')
        .text(`ε = ${currentEpsilon.toFixed(2)}`);
    }

    // Points
    const finiteIntervals = intervals.filter((i) => i.death !== Infinity);
    svg
      .selectAll('.pd-point')
      .data(finiteIntervals)
      .join('circle')
      .attr('class', 'pd-point')
      .attr('cx', (d) => xScale(d.birth))
      .attr('cy', (d) => yScale(d.death))
      .attr('r', 5)
      .attr('fill', (d) => dimensionColors[d.dimension] ?? dimensionColors[0])
      .attr('fill-opacity', (d) =>
        highlightDimension === null || highlightDimension === d.dimension ? 0.8 : 0.15,
      )
      .attr('stroke', (d) => dimensionColors[d.dimension] ?? dimensionColors[0])
      .attr('stroke-width', 1);

    // Infinite intervals — draw as triangles at top
    const infiniteIntervals = intervals.filter((i) => i.death === Infinity);
    svg
      .selectAll('.pd-inf')
      .data(infiniteIntervals)
      .join('path')
      .attr('class', 'pd-inf')
      .attr('d', (d) => {
        const x = xScale(d.birth);
        const y = margin.top + 5;
        return `M${x},${y - 5}L${x + 5},${y + 5}L${x - 5},${y + 5}Z`;
      })
      .attr('fill', (d) => dimensionColors[d.dimension] ?? dimensionColors[0])
      .attr('fill-opacity', 0.8);

    // Legend
    const dims = [...new Set(intervals.map((i) => i.dimension))].sort();
    const legend = svg
      .append('g')
      .attr('transform', `translate(${width - margin.right - 80}, ${margin.top})`);

    dims.forEach((dim, i) => {
      const g = legend.append('g').attr('transform', `translate(0, ${i * 20})`);
      g.append('circle').attr('r', 4).attr('cx', 0).attr('cy', 0).attr('fill', dimensionColors[dim]);
      g.append('text')
        .attr('x', 10)
        .attr('y', 4)
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .text(`H${dim}`);
    });
  }

  function renderBarcode(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const xScale = d3.scaleLinear().domain([0, maxVal]).range([margin.left, width - margin.right]);
    const barHeight = 10;
    const gap = 4;

    // Sort by dimension, then by birth
    const sorted = [...intervals].sort((a, b) => a.dimension - b.dimension || a.birth - b.birth);

    svg
      .append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5));

    // Bars
    sorted.forEach((interval, i) => {
      const y = margin.top + i * (barHeight + gap);
      const deathVal = interval.death === Infinity ? maxVal : interval.death;

      svg
        .append('rect')
        .attr('x', xScale(interval.birth))
        .attr('y', y)
        .attr('width', xScale(deathVal) - xScale(interval.birth))
        .attr('height', barHeight)
        .attr('fill', dimensionColors[interval.dimension] ?? dimensionColors[0])
        .attr('fill-opacity', highlightDimension === null || highlightDimension === interval.dimension ? 0.7 : 0.15)
        .attr('rx', 2);
    });

    // Sweep line
    if (currentEpsilon !== undefined) {
      svg
        .append('line')
        .attr('x1', xScale(currentEpsilon))
        .attr('y1', margin.top - 5)
        .attr('x2', xScale(currentEpsilon))
        .attr('y2', height - margin.bottom)
        .attr('stroke', '#0F6E56')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '6,3');
    }
  }

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} width={width} height={height} className="rounded-lg border border-gray-200" />
    </div>
  );
}

import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  financialData,
  financialMapperNodes,
  financialMapperEdges,
  type Regime,
} from '../../data/mapper-financial-data';

const REGIME_COLORS: Record<Regime, string> = {
  bull: '#1f77b4',
  bear: '#d62728',
  transition: '#2ca02c',
};

const REGIME_LABELS: Record<Regime, string> = {
  bull: 'Bull',
  bear: 'Bear',
  transition: 'Transition',
};

interface PanelProps {
  width: number;
  height: number;
  highlightIndices: Set<number> | null;
  onHoverIndices: (indices: number[] | null) => void;
}

/** Left panel: Returns time series colored by regime */
function TimeSeriesPanel({ width, height, highlightIndices, onHoverIndices }: PanelProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || width === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 12, bottom: 24, left: 36 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, financialData.length - 1]).range([0, innerW]);
    const yExtent = d3.extent(financialData, (d) => d.returns);
    if (yExtent[0] == null) return;
    const yPad = (yExtent[1]! - yExtent[0]) * 0.1;
    const yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - yPad, yExtent[1]! + yPad])
      .range([innerH, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat((d) => `t=${d}`))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .attr('font-size', 8);

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.1%')))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .attr('font-size', 8);

    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    // Title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Returns');

    // Data points
    g.selectAll('.dot')
      .data(financialData)
      .join('circle')
      .attr('class', 'dot')
      .attr('cx', (d) => xScale(d.t))
      .attr('cy', (d) => yScale(d.returns))
      .attr('r', 2.5)
      .attr('fill', (d) => REGIME_COLORS[d.regime])
      .attr('opacity', (d) =>
        highlightIndices === null || highlightIndices.has(d.t) ? 0.8 : 0.1,
      )
      .on('mouseenter', (_, d) => onHoverIndices([d.t]))
      .on('mouseleave', () => onHoverIndices(null));
  }, [width, height, highlightIndices, onHoverIndices]);

  return <svg role="img" aria-label="Market regime mapper visualization (panel 1 of 3)" ref={svgRef} width={width} height={height} />;
}

/** Center panel: Mapper graph with nodes colored by regime */
function MapperGraphPanel({ width, height, highlightIndices, onHoverIndices }: PanelProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || width === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 12, bottom: 12, left: 12 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Scale node positions to panel
    const xExtent = d3.extent(financialMapperNodes, (n) => n.x);
    const yExtent = d3.extent(financialMapperNodes, (n) => n.y);
    if (xExtent[0] == null || yExtent[0] == null) return;
    const xScale = d3.scaleLinear().domain([xExtent[0], xExtent[1]!]).range([20, innerW - 20]);
    const yScale = d3.scaleLinear().domain([yExtent[0], yExtent[1]!]).range([20, innerH - 20]);

    // Title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Mapper Graph');

    // Edges
    g.selectAll('.edge')
      .data(financialMapperEdges)
      .join('line')
      .attr('class', 'edge')
      .attr('x1', ([s]) => xScale(financialMapperNodes[s].x))
      .attr('y1', ([s]) => yScale(financialMapperNodes[s].y))
      .attr('x2', ([, t]) => xScale(financialMapperNodes[t].x))
      .attr('y2', ([, t]) => yScale(financialMapperNodes[t].y))
      .style('stroke', 'var(--color-text)')
      .attr('stroke-opacity', 0.2)
      .attr('stroke-width', 2);

    // Nodes
    g.selectAll('.node')
      .data(financialMapperNodes)
      .join('circle')
      .attr('class', 'node')
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('r', (d) => Math.max(6, Math.min(16, d.size * 0.8)))
      .attr('fill', (d) => REGIME_COLORS[d.dominantRegime])
      .style('stroke', 'var(--color-surface)')
      .attr('stroke-width', 2)
      .attr('opacity', (d) => {
        if (highlightIndices === null) return 0.85;
        return d.members.some((m) => highlightIndices.has(m)) ? 0.95 : 0.15;
      })
      .style('cursor', 'pointer')
      .on('mouseenter', (_, d) => onHoverIndices(d.members))
      .on('mouseleave', () => onHoverIndices(null));

    // Node labels
    g.selectAll('.nlabel')
      .data(financialMapperNodes)
      .join('text')
      .attr('class', 'nlabel')
      .attr('x', (d) => xScale(d.x))
      .attr('y', (d) => yScale(d.y) + 3)
      .attr('text-anchor', 'middle')
      .attr('fill', '#fff')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 8)
      .attr('font-weight', 600)
      .attr('pointer-events', 'none')
      .text((d) => d.size);
  }, [width, height, highlightIndices, onHoverIndices]);

  return <svg role="img" aria-label="Market regime mapper visualization (panel 2 of 3)" ref={svgRef} width={width} height={height} />;
}

/** Right panel: PCA scatter colored by regime */
function PCAPanel({ width, height, highlightIndices, onHoverIndices }: PanelProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || width === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 12, bottom: 24, left: 32 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xExtent = d3.extent(financialData, (d) => d.pcaX);
    const yExtent = d3.extent(financialData, (d) => d.pcaY);
    if (xExtent[0] == null || yExtent[0] == null) return;
    const xPad = (xExtent[1]! - xExtent[0]) * 0.1;
    const yPad = (yExtent[1]! - yExtent[0]) * 0.1;

    const xScale = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1]! + xPad]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1]! + yPad]).range([innerH, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(4))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .attr('font-size', 8);

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .attr('font-size', 8);

    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    // Title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Feature Space (PCA)');

    // Data points
    g.selectAll('.dot')
      .data(financialData)
      .join('circle')
      .attr('class', 'dot')
      .attr('cx', (d) => xScale(d.pcaX))
      .attr('cy', (d) => yScale(d.pcaY))
      .attr('r', 2.5)
      .attr('fill', (d) => REGIME_COLORS[d.regime])
      .attr('opacity', (d) =>
        highlightIndices === null || highlightIndices.has(d.t) ? 0.8 : 0.1,
      )
      .on('mouseenter', (_, d) => onHoverIndices([d.t]))
      .on('mouseleave', () => onHoverIndices(null));
  }, [width, height, highlightIndices, onHoverIndices]);

  return <svg role="img" aria-label="Market regime mapper visualization (panel 3 of 3)" ref={svgRef} width={width} height={height} />;
}

export default function MarketRegimeMapper() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [hoveredIndices, setHoveredIndices] = useState<number[] | null>(null);

  const highlightSet = useMemo(
    () => (hoveredIndices ? new Set(hoveredIndices) : null),
    [hoveredIndices],
  );

  const handleHover = useCallback((indices: number[] | null) => {
    setHoveredIndices(indices);
  }, []);

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 220;
    if (containerWidth < 600) return containerWidth - 16;
    return Math.max(180, (containerWidth - 32) / 3);
  }, [containerWidth]);

  const panelHeight = useMemo(() => Math.min(panelWidth * 0.9, 280), [panelWidth]);

  return (
    <div ref={containerRef} className="w-full space-y-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-lg border border-[var(--color-border)]">
          <TimeSeriesPanel
            width={panelWidth}
            height={panelHeight}
            highlightIndices={highlightSet}
            onHoverIndices={handleHover}
          />
        </div>
        <div className="rounded-lg border border-[var(--color-border)]">
          <MapperGraphPanel
            width={panelWidth}
            height={panelHeight}
            highlightIndices={highlightSet}
            onHoverIndices={handleHover}
          />
        </div>
        <div className="rounded-lg border border-[var(--color-border)]">
          <PCAPanel
            width={panelWidth}
            height={panelHeight}
            highlightIndices={highlightSet}
            onHoverIndices={handleHover}
          />
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs" style={{ fontFamily: 'var(--font-sans)' }}>
        {(Object.entries(REGIME_COLORS) as [Regime, string][]).map(([regime, color]) => (
          <span key={regime} className="flex items-center gap-1.5 opacity-70">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: color }}
            />
            {REGIME_LABELS[regime]}
          </span>
        ))}
        <span className="opacity-50 ml-2">Hover nodes to highlight corresponding data points</span>
      </div>
    </div>
  );
}

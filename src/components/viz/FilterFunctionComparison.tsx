import { useState, useMemo, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  figureEightPoints,
  filterVariants,
} from '../../data/mapper-filter-data';

export default function FilterFunctionComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const scatterRef = useRef<SVGSVGElement>(null);
  const graphRef = useRef<SVGSVGElement>(null);
  const [activeFilter, setActiveFilter] = useState(0);

  const variant: FilterVariant = filterVariants[activeFilter];

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 300;
    if (containerWidth < 600) return containerWidth - 16;
    return Math.max(260, (containerWidth - 24) / 2);
  }, [containerWidth]);
  const panelHeight = 300;

  // ─── Scatter plot colored by filter value ───
  useEffect(() => {
    if (!scatterRef.current || panelWidth === 0) return;

    const svg = d3.select(scatterRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 24, right: 16, bottom: 16, left: 16 };
    const innerW = panelWidth - margin.left - margin.right;
    const innerH = panelHeight - margin.top - margin.bottom;

    const xExtent = d3.extent(figureEightPoints, (p) => p.x);
    const yExtent = d3.extent(figureEightPoints, (p) => p.y);
    if (xExtent[0] == null || yExtent[0] == null) return;
    const pad = 0.1;

    const xScale = d3.scaleLinear().domain([xExtent[0] - pad, xExtent[1]! + pad]).range([margin.left, margin.left + innerW]);
    const yScale = d3.scaleLinear().domain([yExtent[0] - pad, yExtent[1]! + pad]).range([margin.top + innerH, margin.top]);

    const fExtent = d3.extent(variant.filterValues);
    if (fExtent[0] == null) return;
    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([fExtent[0], fExtent[1]!]);

    // Title
    svg
      .append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--color-text)')
      .attr('font-family', 'var(--font-sans)')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .text(`Colored by ${variant.name}`);

    const g = svg.append('g');

    g.selectAll('.point')
      .data(figureEightPoints)
      .join('circle')
      .attr('class', 'point')
      .attr('cx', (p) => xScale(p.x))
      .attr('cy', (p) => yScale(p.y))
      .attr('r', 3.5)
      .attr('fill', (_, i) => colorScale(variant.filterValues[i]))
      .attr('stroke', 'var(--color-surface)')
      .attr('stroke-width', 0.5)
      .attr('opacity', 0.85);
  }, [variant, panelWidth]);

  // ─── Mapper graph for active filter ───
  useEffect(() => {
    if (!graphRef.current || panelWidth === 0) return;

    const svg = d3.select(graphRef.current);
    svg.selectAll('*').remove();

    const graph = variant.graph;
    const gW = panelWidth;
    const gH = panelHeight;

    // Title
    svg
      .append('text')
      .attr('x', gW / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--color-text)')
      .attr('font-family', 'var(--font-sans)')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .text('Mapper Graph');

    // Force layout
    const simNodes = graph.nodes.map((n) => ({
      ...n,
      x: gW / 2 + (Math.cos((n.id / graph.nodes.length) * Math.PI * 2)) * gW * 0.25,
      y: gH / 2 + (Math.sin((n.id / graph.nodes.length) * Math.PI * 2)) * gH * 0.25,
    }));

    const nodeMap = new Map(simNodes.map((n) => [n.id, n]));

    const simEdges = graph.edges
      .map(([s, t]) => ({ source: nodeMap.get(s)!, target: nodeMap.get(t)! }))
      .filter((e) => e.source && e.target);

    const simulation = d3
      .forceSimulation(simNodes as any)
      .force('link', d3.forceLink(simEdges as any).id((d: any) => d.id).distance(35))
      .force('charge', d3.forceManyBody().strength(-60))
      .force('center', d3.forceCenter(gW / 2, gH / 2))
      .force('collision', d3.forceCollide(10));

    for (let i = 0; i < 150; i++) simulation.tick();
    simulation.stop();

    // Clamp
    for (const n of simNodes) {
      (n as any).x = Math.max(20, Math.min(gW - 20, (n as any).x));
      (n as any).y = Math.max(30, Math.min(gH - 20, (n as any).y));
    }

    const fExtent = d3.extent(graph.nodes, (n) => n.filterValue);
    if (fExtent[0] == null) return;
    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([fExtent[0], fExtent[1]!]);

    const g = svg.append('g');

    // Edges
    g.selectAll('.edge')
      .data(simEdges)
      .join('line')
      .attr('class', 'edge')
      .attr('x1', (d: any) => d.source.x)
      .attr('y1', (d: any) => d.source.y)
      .attr('x2', (d: any) => d.target.x)
      .attr('y2', (d: any) => d.target.y)
      .attr('stroke', 'var(--color-text)')
      .attr('stroke-opacity', 0.2)
      .attr('stroke-width', 2);

    // Nodes
    g.selectAll('.node')
      .data(simNodes)
      .join('circle')
      .attr('class', 'node')
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)
      .attr('r', (d: any) => Math.max(5, Math.min(12, d.size * 0.8)))
      .attr('fill', (d: any) => colorScale(d.filterValue))
      .attr('stroke', 'var(--color-surface)')
      .attr('stroke-width', 2)
      .attr('opacity', 0.85);

    // Stats
    svg
      .append('text')
      .attr('x', gW / 2)
      .attr('y', gH - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--color-text)')
      .attr('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .attr('opacity', 0.5)
      .text(`${graph.nodes.length} nodes, ${graph.edges.length} edges`);
  }, [variant, panelWidth]);

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Filter tabs */}
      <div className="flex flex-wrap gap-1.5">
        {filterVariants.map((v, i) => (
          <button
            key={i}
            onClick={() => setActiveFilter(i)}
            className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors"
            style={{
              fontFamily: 'var(--font-sans)',
              background: i === activeFilter ? 'var(--color-accent)' : 'var(--color-surface)',
              color: i === activeFilter ? '#fff' : 'var(--color-text)',
              border: `1px solid ${i === activeFilter ? 'var(--color-accent)' : 'var(--color-border)'}`,
            }}
          >
            {v.name}
          </button>
        ))}
      </div>

      {/* Description */}
      <p className="text-sm opacity-70" style={{ fontFamily: 'var(--font-sans)' }}>
        {variant.description}
      </p>

      {/* Panels */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <svg
          ref={scatterRef}
          width={panelWidth}
          height={panelHeight}
          className="rounded-lg border border-[var(--color-border)]"
        />
        <svg
          ref={graphRef}
          width={panelWidth}
          height={panelHeight}
          className="rounded-lg border border-[var(--color-border)]"
        />
      </div>
    </div>
  );
}

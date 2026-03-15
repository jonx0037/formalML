import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { domainColorScale } from './shared/colorScales';
import type { DAGNode, DAGEdge } from './shared/types';

interface DAGGraphProps {
  nodes: DAGNode[];
  edges: DAGEdge[];
  highlightNode?: string;
  onNodeClick?: (id: string) => void;
  layout?: 'force' | 'layered';
}

export default function DAGGraph({
  nodes,
  edges,
  highlightNode,
  onNodeClick,
  layout = 'force',
}: DAGGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth || 600;
    const height = 400;

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    // Create simulation
    const simNodes = nodes.map((n) => ({ ...n })) as (DAGNode & d3.SimulationNodeDatum)[];
    const simEdges = edges.map((e) => ({ ...e })) as (DAGEdge & d3.SimulationLinkDatum<DAGNode & d3.SimulationNodeDatum>)[];

    const simulation = d3
      .forceSimulation(simNodes)
      .force('link', d3.forceLink(simEdges).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('y', d3.forceY(height / 2).strength(0.05));

    // Arrow markers
    svg
      .append('defs')
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#999');

    // Edges
    const link = svg
      .append('g')
      .selectAll('line')
      .data(simEdges)
      .join('line')
      .attr('stroke', '#ccc')
      .attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrowhead)');

    // Nodes
    const node = svg
      .append('g')
      .selectAll('g')
      .data(simNodes)
      .join('g')
      .style('cursor', onNodeClick ? 'pointer' : 'default')
      .on('click', (_, d) => onNodeClick?.(d.id));

    node
      .append('circle')
      .attr('r', (d) => (d.id === highlightNode ? 12 : 8))
      .attr('fill', (d) => domainColorScale(d.domain))
      .attr('stroke', (d) => (d.id === highlightNode ? '#1A1A1A' : 'white'))
      .attr('stroke-width', (d) => (d.id === highlightNode ? 3 : 2))
      .attr('opacity', (d) => (d.status === 'draft' ? 0.4 : 1));

    node
      .append('text')
      .text((d) => d.label)
      .attr('dy', -14)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-sans)')
      .style('fill', 'var(--color-text)')
      .style('pointer-events', 'none');

    // Drag behavior
    const drag = d3
      .drag<SVGGElement, DAGNode & d3.SimulationNodeDatum>()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    node.call(drag);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => simulation.stop();
  }, [nodes, edges, highlightNode, onNodeClick, layout]);

  return <svg ref={svgRef} className="w-full rounded-lg" style={{ minHeight: 400 }} />;
}

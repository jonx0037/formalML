import { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { sensitivityGridData, type SensitivityEntry } from '../../data/mapper-sensitivity-data';

interface CellProps {
  entry: SensitivityEntry;
  width: number;
  height: number;
}

function SensitivityCell({ entry, width, height }: CellProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || width === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = 8;
    const innerW = width - margin * 2;
    const innerH = height - margin * 2 - 24; // leave room for label

    // Create node copies for simulation
    const simNodes = entry.nodes.map((n) => ({
      ...n,
      x: innerW / 2 + (Math.random() - 0.5) * innerW * 0.3,
      y: innerH / 2 + (Math.random() - 0.5) * innerH * 0.3,
    }));

    const nodeMap = new Map(simNodes.map((n) => [n.id, n]));

    const simEdges = entry.edges
      .map(([s, t]) => ({
        source: nodeMap.get(s),
        target: nodeMap.get(t),
      }))
      .filter((e) => e.source && e.target);

    const g = svg.append('g').attr('transform', `translate(${margin},${margin})`);

    // Force simulation
    const simulation = d3
      .forceSimulation(simNodes as any)
      .force(
        'link',
        d3
          .forceLink(simEdges as any)
          .id((d: any) => d.id)
          .distance(Math.min(innerW, innerH) * 0.2),
      )
      .force('charge', d3.forceManyBody().strength(-30))
      .force('center', d3.forceCenter(innerW / 2, innerH / 2))
      .force('collision', d3.forceCollide(6));

    // Edges
    const links = g
      .selectAll('.edge')
      .data(simEdges)
      .join('line')
      .attr('class', 'edge')
      .attr('stroke', 'var(--color-text)')
      .attr('stroke-opacity', 0.3)
      .attr('stroke-width', 1.5);

    // Nodes
    const nodes = g
      .selectAll('.node')
      .data(simNodes)
      .join('circle')
      .attr('class', 'node')
      .attr('r', (d: any) => Math.max(3, Math.min(7, d.size * 0.6)))
      .attr('fill', entry.isGoldilocks ? '#0F6E56' : 'var(--color-text)')
      .attr('fill-opacity', entry.isGoldilocks ? 0.8 : 0.4)
      .attr('stroke', 'var(--color-surface)')
      .attr('stroke-width', 1);

    // Label at bottom
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', height - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--color-text)')
      .attr('font-family', 'var(--font-sans)')
      .attr('font-size', 9)
      .attr('opacity', 0.7)
      .text(entry.label);

    // Tick updates
    simulation.on('tick', () => {
      links
        .attr('x1', (d: any) => Math.max(4, Math.min(innerW - 4, d.source.x)))
        .attr('y1', (d: any) => Math.max(4, Math.min(innerH - 4, d.source.y)))
        .attr('x2', (d: any) => Math.max(4, Math.min(innerW - 4, d.target.x)))
        .attr('y2', (d: any) => Math.max(4, Math.min(innerH - 4, d.target.y)));

      nodes
        .attr('cx', (d: any) => {
          d.x = Math.max(4, Math.min(innerW - 4, d.x));
          return d.x;
        })
        .attr('cy', (d: any) => {
          d.y = Math.max(4, Math.min(innerH - 4, d.y));
          return d.y;
        });
    });

    // Run simulation to completion quickly
    simulation.alpha(1).restart();
    for (let i = 0; i < 120; i++) simulation.tick();
    simulation.stop();

    // Final position update
    links
      .attr('x1', (d: any) => d.source.x)
      .attr('y1', (d: any) => d.source.y)
      .attr('x2', (d: any) => d.target.x)
      .attr('y2', (d: any) => d.target.y);

    nodes.attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y);

    return () => simulation.stop();
  }, [entry, width, height]);

  return (
    <div
      className="rounded-lg border"
      style={{
        borderColor: entry.isGoldilocks ? '#0F6E56' : 'var(--color-border)',
        borderWidth: entry.isGoldilocks ? 2 : 1,
        background: entry.isGoldilocks
          ? 'color-mix(in srgb, #0F6E56 5%, var(--color-surface))'
          : 'var(--color-surface)',
      }}
    >
      <svg ref={svgRef} width={width} height={height} />
    </div>
  );
}

export default function ParameterSensitivityGrid() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const data = sensitivityGridData;

  const cellWidth = useMemo(() => {
    if (!containerWidth) return 140;
    const cols = containerWidth >= 600 ? 4 : 2;
    const gap = 8;
    return Math.max(100, (containerWidth - gap * (cols - 1)) / cols);
  }, [containerWidth]);

  const cellHeight = useMemo(() => Math.min(cellWidth * 0.85, 130), [cellWidth]);

  if (data.length === 0) return null;

  const row1 = data.filter((_, i) => i < 4);
  const row2 = data.filter((_, i) => i >= 4);

  return (
    <div ref={containerRef} className="w-full space-y-4">
      {/* Row 1: Varying n_intervals */}
      <div>
        <p
          className="text-xs font-medium mb-2 opacity-70"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Varying n_intervals (overlap = 0.35)
        </p>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {row1.map((entry, i) => (
            <SensitivityCell key={i} entry={entry} width={cellWidth} height={cellHeight} />
          ))}
        </div>
      </div>

      {/* Row 2: Varying overlap */}
      <div>
        <p
          className="text-xs font-medium mb-2 opacity-70"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Varying overlap (n_intervals = 12)
        </p>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {row2.map((entry, i) => (
            <SensitivityCell key={i + 4} entry={entry} width={cellWidth} height={cellHeight} />
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs opacity-60" style={{ fontFamily: 'var(--font-sans)' }}>
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm border-2"
            style={{ borderColor: '#0F6E56', background: 'color-mix(in srgb, #0F6E56 15%, transparent)' }}
          />
          Stable topology (Goldilocks zone)
        </span>
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { domainColorScale } from './shared/colorScales';
import type { DAGNode, DAGEdge } from './shared/types';

interface CurriculumGraphProps {
  nodes: DAGNode[];
  edges: DAGEdge[];
}

/** Compute topological depth via BFS from root nodes (no incoming edges). */
function computeDepths(nodes: DAGNode[], edges: DAGEdge[]): Map<string, number> {
  const incoming = new Set(edges.map((e) => e.target));
  const children = new Map<string, string[]>();
  for (const e of edges) {
    if (!children.has(e.source)) children.set(e.source, []);
    children.get(e.source)!.push(e.target);
  }

  const depths = new Map<string, number>();
  const roots = nodes.filter((n) => !incoming.has(n.id));
  const queue: { id: string; depth: number }[] = roots.map((r) => ({ id: r.id, depth: 0 }));

  while (queue.length > 0) {
    const { id, depth } = queue.shift()!;
    if (depths.has(id) && depths.get(id)! >= depth) continue;
    depths.set(id, depth);
    for (const child of children.get(id) ?? []) {
      queue.push({ id: child, depth: depth + 1 });
    }
  }

  // Nodes with no edges get depth 0
  for (const n of nodes) {
    if (!depths.has(n.id)) depths.set(n.id, 0);
  }
  return depths;
}

/** Build adjacency lookup for hover highlighting. */
function buildAdjacency(edges: DAGEdge[]) {
  const predecessors = new Map<string, Set<string>>();
  const successors = new Map<string, Set<string>>();
  for (const e of edges) {
    if (!successors.has(e.source)) successors.set(e.source, new Set());
    successors.get(e.source)!.add(e.target);
    if (!predecessors.has(e.target)) predecessors.set(e.target, new Set());
    predecessors.get(e.target)!.add(e.source);
  }
  return { predecessors, successors };
}

export default function CurriculumGraph({ nodes, edges }: CurriculumGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string } | null>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = containerRef.current.clientWidth || 800;
    const height = 500;
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    // Precompute depths and adjacency
    const depths = computeDepths(nodes, edges);
    const maxDepth = Math.max(...depths.values(), 1);
    const depthScale = d3.scaleLinear().domain([0, maxDepth]).range([80, width - 80]);
    const { predecessors, successors } = buildAdjacency(edges);

    // Read theme colors
    const style = getComputedStyle(document.documentElement);
    const textColor = style.getPropertyValue('--color-text').trim() || '#1A1A1A';
    const surfaceColor = style.getPropertyValue('--color-surface').trim() || '#FFFFFF';
    const borderColor = style.getPropertyValue('--color-border').trim() || '#E5E5E0';

    // Simulation nodes/edges
    type SimNode = DAGNode & d3.SimulationNodeDatum & { depth: number };
    type SimEdge = DAGEdge & d3.SimulationLinkDatum<SimNode>;

    const simNodes: SimNode[] = nodes.map((n) => ({
      ...n,
      depth: depths.get(n.id) ?? 0,
    }));
    const simEdges: SimEdge[] = edges.map((e) => ({ ...e }));

    // Force simulation with left-to-right bias
    const simulation = d3
      .forceSimulation(simNodes)
      .force(
        'link',
        d3
          .forceLink<SimNode, SimEdge>(simEdges)
          .id((d) => d.id)
          .distance(90)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('x', d3.forceX<SimNode>((d) => depthScale(d.depth)).strength(0.4))
      .force('y', d3.forceY(height / 2).strength(0.08))
      .force('collision', d3.forceCollide(30));

    // Zoom container
    const g = svg.append('g');
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.4, 3])
      .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);

    // Arrow marker
    g.append('defs')
      .append('marker')
      .attr('id', 'curriculum-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 18)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 5)
      .attr('markerHeight', 5)
      .append('path')
      .attr('d', 'M0,-4L10,0L0,4')
      .attr('fill', borderColor);

    // Edges
    const link = g
      .append('g')
      .selectAll('line')
      .data(simEdges)
      .join('line')
      .attr('stroke', borderColor)
      .attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#curriculum-arrow)');

    // Node groups
    const node = g
      .append('g')
      .selectAll<SVGGElement, SimNode>('g')
      .data(simNodes)
      .join('g')
      .style('cursor', (d) => (d.status === 'published' ? 'pointer' : 'default'));

    // Node circles
    node
      .append('circle')
      .attr('r', 8)
      .attr('fill', (d) =>
        d.status === 'published' ? domainColorScale(d.domain) : 'transparent'
      )
      .attr('stroke', (d) => domainColorScale(d.domain))
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', (d) => (d.status === 'published' ? 'none' : '4,2'))
      .attr('opacity', (d) => (d.status === 'published' ? 1 : 0.5));

    // Node labels
    node
      .append('text')
      .text((d) => d.label)
      .attr('dy', -14)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('font-family', 'var(--font-sans)')
      .style('fill', textColor)
      .style('opacity', (d) => (d.status === 'published' ? 1 : 0.5))
      .style('pointer-events', 'none');

    // Hover highlighting
    const nodeIds = new Set(nodes.map((n) => n.id));
    node
      .on('mouseenter', (_, d) => {
        const neighbors = new Set<string>([d.id]);
        for (const p of predecessors.get(d.id) ?? []) neighbors.add(p);
        for (const s of successors.get(d.id) ?? []) neighbors.add(s);

        node.attr('opacity', (n) => (neighbors.has(n.id) ? 1 : 0.12));
        link.attr('opacity', (l: SimEdge) => {
          const src = typeof l.source === 'string' ? l.source : (l.source as SimNode).id;
          const tgt = typeof l.target === 'string' ? l.target : (l.target as SimNode).id;
          return neighbors.has(src) && neighbors.has(tgt) ? 1 : 0.08;
        });
      })
      .on('mouseleave', () => {
        node.attr('opacity', 1);
        link.attr('opacity', 1);
      });

    // Click behavior
    node.on('click', (event, d) => {
      if (d.status === 'published' && d.url) {
        window.location.href = d.url;
      } else {
        const rect = svgRef.current!.getBoundingClientRect();
        setTooltip({
          x: event.clientX - rect.left,
          y: event.clientY - rect.top - 30,
          label: d.label,
        });
        setTimeout(() => setTooltip(null), 2000);
      }
    });

    // Drag behavior (stops zoom propagation)
    const drag = d3
      .drag<SVGGElement, SimNode>()
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

    // Tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: { source: SimNode; target: SimNode }) => d.source.x!)
        .attr('y1', (d: { source: SimNode; target: SimNode }) => d.source.y!)
        .attr('x2', (d: { source: SimNode; target: SimNode }) => d.target.x!)
        .attr('y2', (d: { source: SimNode; target: SimNode }) => d.target.y!);
      node.attr('transform', (d: SimNode) => `translate(${d.x},${d.y})`);
    });

    return () => simulation.stop();
  }, [nodes, edges]);

  // Domain legend data
  const domains = [...new Set(nodes.map((n) => n.domain))];

  return (
    <div ref={containerRef} className="relative">
      <svg role="img" aria-label="Curriculum graph visualization"
        ref={svgRef}
        className="w-full rounded-lg border border-[var(--color-border)]"
        style={{ minHeight: 500 }}
      />

      {/* Tooltip for planned topics */}
      {tooltip && (
        <div
          className="absolute pointer-events-none rounded px-2 py-1 text-xs"
          style={{
            left: tooltip.x,
            top: tooltip.y,
            background: 'var(--color-surface)',
            color: 'var(--color-text-secondary)',
            border: '1px solid var(--color-border)',
            transform: 'translateX(-50%)',
          }}
        >
          {tooltip.label} — Coming soon
        </div>
      )}

      {/* Legend */}
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-[var(--color-text-secondary)]">
        {domains.map((domain) => (
          <span key={domain} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: domainColorScale(domain) }}
            />
            {domain.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
          </span>
        ))}
        <span className="ml-2 flex items-center gap-3 border-l border-[var(--color-border)] pl-3">
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: 'var(--color-accent)' }}
            />
            Published
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full border border-current"
              style={{ background: 'transparent' }}
            />
            Planned
          </span>
        </span>
      </div>
    </div>
  );
}

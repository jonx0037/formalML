import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { runMapper, createIntervalCover } from './shared/mapper';
import type { MapperPoint } from './shared/types';
import { circlePoints } from '../../data/mapper-pipeline-data';

const STEP_LABELS = [
  '1. Filter',
  '2. Cover',
  '3. Pullback',
  '4. Cluster',
  '5. Graph',
];

const STEP_DESCRIPTIONS = [
  'Color each point by its filter value (x-coordinate).',
  'Divide the filter range into overlapping intervals.',
  'Pull back each interval to find which points belong.',
  'Cluster within each pullback set independently.',
  'Build the Mapper graph: one node per cluster, edges where clusters share points.',
];

const INTERVAL_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
];

export default function MapperPipelineExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const graphRef = useRef<SVGSVGElement>(null);

  const [step, setStep] = useState(0);
  const [nIntervals, setNIntervals] = useState(10);
  const [overlap, setOverlap] = useState(0.35);

  const points: MapperPoint[] = circlePoints;

  const mapperResult = useMemo(
    () => runMapper(points, { nIntervals, overlap }),
    [points, nIntervals, overlap],
  );

  const filterExtent = useMemo(() => {
    const vals = points.map((p) => p.filterValue);
    return [Math.min(...vals), Math.max(...vals)] as [number, number];
  }, [points]);

  const intervals = useMemo(
    () => createIntervalCover(filterExtent[0], filterExtent[1], nIntervals, overlap),
    [filterExtent, nIntervals, overlap],
  );

  // Point cloud panel dimensions
  const panelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    return Math.min(containerWidth - 16, 600);
  }, [containerWidth]);
  const panelHeight = 340;
  const margin = { top: 24, right: 16, bottom: 40, left: 16 };

  // Scales
  const xExtent = useMemo(
    () => d3.extent(points, (p) => p.x) as [number, number],
    [points],
  );
  const yExtent = useMemo(
    () => d3.extent(points, (p) => p.y) as [number, number],
    [points],
  );

  // ─── Point cloud rendering ───
  useEffect(() => {
    if (!svgRef.current || panelWidth === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerW = panelWidth - margin.left - margin.right;
    const innerH = panelHeight - margin.top - margin.bottom;

    const pad = 0.15;
    const xScale = d3
      .scaleLinear()
      .domain([xExtent[0] - pad, xExtent[1] + pad])
      .range([margin.left, margin.left + innerW]);
    const yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - pad, yExtent[1] + pad])
      .range([margin.top + innerH, margin.top]);

    const g = svg.append('g');

    // Filter color scale
    const filterColorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain(filterExtent);

    // ── Step 1: Filter ─ color by filter value
    if (step >= 0) {
      g.selectAll('.point')
        .data(points)
        .join('circle')
        .attr('class', 'point')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', 4)
        .attr('fill', (p) => {
          if (step <= 1) return filterColorScale(p.filterValue);
          if (step >= 2) {
            // Color by interval membership
            for (let i = 0; i < mapperResult.pullbackAssignments.length; i++) {
              if (mapperResult.pullbackAssignments[i].includes(p.id)) {
                return INTERVAL_COLORS[i % INTERVAL_COLORS.length];
              }
            }
            return '#999';
          }
          return filterColorScale(p.filterValue);
        })
        .attr('stroke', (p) => {
          // Highlight overlap points in step 2+
          if (step >= 2) {
            let count = 0;
            for (const pb of mapperResult.pullbackAssignments) {
              if (pb.includes(p.id)) count++;
            }
            if (count > 1) return '#fff';
          }
          return 'var(--color-surface)';
        })
        .attr('stroke-width', (p) => {
          if (step >= 2) {
            let count = 0;
            for (const pb of mapperResult.pullbackAssignments) {
              if (pb.includes(p.id)) count++;
            }
            if (count > 1) return 2;
          }
          return 1;
        })
        .attr('opacity', 0.85);
    }

    // ── Step 4: Cluster ─ draw cluster hulls
    if (step >= 3) {
      const clusterGroups = mapperResult.clusters;
      for (let ci = 0; ci < clusterGroups.length; ci++) {
        const cluster = clusterGroups[ci];
        if (cluster.members.length < 3) continue;

        const hullPoints = cluster.members.map((idx) => [
          xScale(points[idx].x),
          yScale(points[idx].y),
        ] as [number, number]);

        const hull = d3.polygonHull(hullPoints);
        if (hull) {
          g.append('polygon')
            .attr('points', hull.map((p) => p.join(',')).join(' '))
            .attr('fill', INTERVAL_COLORS[cluster.intervalIdx % INTERVAL_COLORS.length])
            .attr('fill-opacity', 0.1)
            .attr('stroke', INTERVAL_COLORS[cluster.intervalIdx % INTERVAL_COLORS.length])
            .attr('stroke-opacity', 0.4)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '4,2');
        }
      }
    }

    // ── Step 2: Cover ─ draw interval bars below the scatter plot
    if (step >= 1) {
      const barY = margin.top + innerH + 8;
      const barH = 16;
      const fScale = d3.scaleLinear().domain(filterExtent).range([margin.left, margin.left + innerW]);

      for (let i = 0; i < intervals.length; i++) {
        const [lo, hi] = intervals[i];
        const x1 = fScale(lo);
        const x2 = fScale(hi);

        g.append('rect')
          .attr('x', x1)
          .attr('y', barY + (i % 2) * 2)
          .attr('width', x2 - x1)
          .attr('height', barH)
          .attr('fill', INTERVAL_COLORS[i % INTERVAL_COLORS.length])
          .attr('fill-opacity', 0.25)
          .attr('stroke', INTERVAL_COLORS[i % INTERVAL_COLORS.length])
          .attr('stroke-opacity', 0.6)
          .attr('stroke-width', 1)
          .attr('rx', 2);
      }

      // Filter axis label
      g.append('text')
        .attr('x', margin.left + innerW / 2)
        .attr('y', barY + barH + 14)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--color-text)')
        .attr('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('opacity', 0.6)
        .text('Filter range (x-coordinate)');
    }
  }, [step, points, panelWidth, panelHeight, mapperResult, intervals, filterExtent, xExtent, yExtent]);

  // ─── Mapper graph rendering (step 5) ───
  useEffect(() => {
    if (!graphRef.current || step < 4 || panelWidth === 0) return;

    const svg = d3.select(graphRef.current);
    svg.selectAll('*').remove();

    const gW = Math.min(panelWidth, 400);
    const gH = 260;

    // Force layout
    const simNodes = mapperResult.nodes.map((n) => ({
      ...n,
      x: gW / 2 + (Math.cos((n.id / mapperResult.nodes.length) * Math.PI * 2)) * gW * 0.3,
      y: gH / 2 + (Math.sin((n.id / mapperResult.nodes.length) * Math.PI * 2)) * gH * 0.3,
    }));

    const nodeMap = new Map(simNodes.map((n) => [n.id, n]));

    const simEdges = mapperResult.edges
      .map(([s, t]) => ({ source: nodeMap.get(s)!, target: nodeMap.get(t)! }))
      .filter((e) => e.source && e.target);

    const simulation = d3
      .forceSimulation(simNodes as any)
      .force('link', d3.forceLink(simEdges as any).id((d: any) => d.id).distance(40))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(gW / 2, gH / 2))
      .force('collision', d3.forceCollide(12));

    // Run to completion
    for (let i = 0; i < 150; i++) simulation.tick();
    simulation.stop();

    // Clamp positions
    for (const n of simNodes) {
      (n as any).x = Math.max(20, Math.min(gW - 20, (n as any).x));
      (n as any).y = Math.max(20, Math.min(gH - 20, (n as any).y));
    }

    const g = svg.append('g');

    // Title
    svg
      .append('text')
      .attr('x', gW / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--color-text)')
      .attr('font-family', 'var(--font-sans)')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .text('Mapper Graph');

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
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', 2);

    // Nodes
    g.selectAll('.node')
      .data(simNodes)
      .join('circle')
      .attr('class', 'node')
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)
      .attr('r', (d) => Math.max(5, Math.min(14, d.size * 1.2)))
      .attr('fill', '#0F6E56')
      .attr('fill-opacity', 0.7)
      .attr('stroke', 'var(--color-surface)')
      .attr('stroke-width', 2);

    // Node labels
    g.selectAll('.nlabel')
      .data(simNodes)
      .join('text')
      .attr('class', 'nlabel')
      .attr('x', (d: any) => d.x)
      .attr('y', (d: any) => d.y + 3)
      .attr('text-anchor', 'middle')
      .attr('fill', '#fff')
      .attr('font-family', 'var(--font-sans)')
      .attr('font-size', 8)
      .attr('font-weight', 600)
      .attr('pointer-events', 'none')
      .text((d) => d.size);

    // Stats
    svg
      .append('text')
      .attr('x', gW / 2)
      .attr('y', gH - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--color-text)')
      .attr('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .attr('opacity', 0.5)
      .text(`${mapperResult.nodes.length} nodes, ${mapperResult.edges.length} edges`);
  }, [step, mapperResult, panelWidth]);

  const handleNIntervalsChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setNIntervals(parseInt(e.target.value, 10));
  }, []);

  const handleOverlapChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setOverlap(parseFloat(e.target.value));
  }, []);

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Step indicator */}
      <div className="flex flex-wrap gap-1.5">
        {STEP_LABELS.map((label, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className="rounded-md px-2.5 py-1 text-xs font-medium transition-colors"
            style={{
              fontFamily: 'var(--font-sans)',
              background: i === step ? 'var(--color-accent)' : 'var(--color-surface)',
              color: i === step ? '#fff' : 'var(--color-text)',
              border: `1px solid ${i === step ? 'var(--color-accent)' : 'var(--color-border)'}`,
              opacity: i <= step ? 1 : 0.5,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Step description */}
      <p
        className="text-sm opacity-70"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        {STEP_DESCRIPTIONS[step]}
      </p>

      {/* Main visualization */}
      <div className="flex flex-col items-center gap-3 md:flex-row md:items-start">
        <svg
          ref={svgRef}
          width={panelWidth}
          height={panelHeight}
          className="rounded-lg border border-[var(--color-border)]"
        />
        {step >= 4 && (
          <svg
            ref={graphRef}
            width={Math.min(panelWidth, 400)}
            height={260}
            className="rounded-lg border border-[var(--color-border)]"
          />
        )}
      </div>

      {/* Parameter controls */}
      <div className="flex flex-col gap-3 sm:flex-row sm:gap-6">
        <div className="flex items-center gap-2 flex-1">
          <label
            className="text-xs font-medium whitespace-nowrap min-w-[100px]"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Intervals: {nIntervals}
          </label>
          <input
            type="range"
            min={3}
            max={20}
            step={1}
            value={nIntervals}
            onChange={handleNIntervalsChange}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
        <div className="flex items-center gap-2 flex-1">
          <label
            className="text-xs font-medium whitespace-nowrap min-w-[100px]"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Overlap: {(overlap * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min={0.05}
            max={0.8}
            step={0.05}
            value={overlap}
            onChange={handleOverlapChange}
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
      </div>
    </div>
  );
}

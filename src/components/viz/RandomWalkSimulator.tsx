import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pathGraph, cycleGraph, completeGraph, starGraph, barbellGraph, gridGraph,
  stationaryDistribution, totalVariationDistance,
  type Graph,
} from './shared/graphTheory';
import type { WalkState } from './shared/types';

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const GRAPH_PANEL_HEIGHT = 360;
const BAR_CHART_HEIGHT = 200;
const TV_CHART_HEIGHT = 180;
const MARGIN = { top: 28, right: 16, bottom: 36, left: 44 };

// ─── Colors (formalML palette) ───

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const SLATE = '#6B6B6B';

// ─── Types ───

type WalkType = 'standard' | 'lazy';

interface Preset {
  label: string;
  build: () => Graph;
}

const PRESETS: Preset[] = [
  { label: 'Path(8)', build: () => pathGraph(8) },
  { label: 'Cycle(8)', build: () => cycleGraph(8) },
  { label: 'Complete(6)', build: () => completeGraph(6) },
  { label: 'Barbell(4-4)', build: () => barbellGraph(4) },
  { label: 'Grid(3×3)', build: () => gridGraph(3) },
  { label: 'Star(7)', build: () => starGraph(7) },
];

// ─── Helpers ───

/** Build an edge list from the upper triangle of the adjacency matrix. */
function edgesFromAdj(adj: number[][]): { source: number; target: number }[] {
  const edges: { source: number; target: number }[] = [];
  for (let i = 0; i < adj.length; i++) {
    for (let j = i + 1; j < adj.length; j++) {
      if (adj[i][j] !== 0) edges.push({ source: i, target: j });
    }
  }
  return edges;
}

/** Get neighbors of vertex v from adjacency matrix. */
function neighbors(adj: number[][], v: number): number[] {
  const nbrs: number[] = [];
  for (let j = 0; j < adj[v].length; j++) {
    if (adj[v][j] !== 0) nbrs.push(j);
  }
  return nbrs;
}

/** Create an initial walk state. */
function createWalkState(n: number, startVertex: number): WalkState {
  const visitCounts = new Array(n).fill(0);
  visitCounts[startVertex] = 1;
  return {
    currentVertex: startVertex,
    visitCounts,
    totalSteps: 1,
    trajectory: [startVertex],
  };
}

// ─── Component ───

export default function RandomWalkSimulator() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const graphSvgRef = useRef<SVGSVGElement>(null);
  const barSvgRef = useRef<SVGSVGElement>(null);
  const tvSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───

  const [presetIdx, setPresetIdx] = useState(0);
  const [graph, setGraph] = useState<Graph>(() => PRESETS[0].build());
  const [walkState, setWalkState] = useState<WalkState>(() =>
    createWalkState(PRESETS[0].build().n, 0),
  );
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(5);
  const [walkType, setWalkType] = useState<WalkType>('standard');
  const [startVertex, setStartVertex] = useState(0);
  const [tvHistory, setTvHistory] = useState<number[]>([]);

  const n = graph.n;
  const adj = graph.adjacency;

  // Node positions managed outside React state for the force simulation.
  const positionsRef = useRef<{ x: number; y: number }[]>([]);
  const simRef = useRef<d3.Simulation<{ x: number; y: number }, undefined> | null>(null);
  const [tick, setTick] = useState(0);

  // ─── Derived computations ───

  const stationaryDist = useMemo(() => stationaryDistribution(adj), [adj]);

  // Empirical visit frequency
  const empiricalDist = useMemo(() => {
    if (walkState.totalSteps === 0) return new Array(n).fill(0);
    return walkState.visitCounts.map((c) => c / walkState.totalSteps);
  }, [walkState.visitCounts, walkState.totalSteps, n]);

  // ─── Layout calculations ───

  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const graphPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    if (isNarrow) return containerWidth - 16;
    return Math.floor((containerWidth - 32) * 0.55);
  }, [containerWidth, isNarrow]);

  const rightPanelWidth = useMemo(() => {
    if (!containerWidth) return 340;
    if (isNarrow) return containerWidth - 16;
    return Math.floor((containerWidth - 32) * 0.45);
  }, [containerWidth, isNarrow]);

  // ─── Force simulation management ───

  const initSimulation = useCallback(
    (nodeCount: number, adjacency: number[][]) => {
      if (simRef.current) simRef.current.stop();

      const nodes: { x: number; y: number }[] = Array.from(
        { length: nodeCount },
        (_, i) => ({
          x: graphPanelWidth / 2 + Math.cos((2 * Math.PI * i) / nodeCount) * 80,
          y: GRAPH_PANEL_HEIGHT / 2 + Math.sin((2 * Math.PI * i) / nodeCount) * 80,
        }),
      );
      positionsRef.current = nodes;

      const links = edgesFromAdj(adjacency);

      const sim = d3
        .forceSimulation(nodes)
        .force(
          'link',
          d3
            .forceLink<{ x: number; y: number }, { source: number; target: number }>(links)
            .id((_, i) => i)
            .distance(60),
        )
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(graphPanelWidth / 2, GRAPH_PANEL_HEIGHT / 2))
        .force('collide', d3.forceCollide(20))
        .alphaDecay(0.03)
        .on('tick', () => {
          const pad = 24;
          for (const node of nodes) {
            node.x = Math.max(pad, Math.min(graphPanelWidth - pad, node.x));
            node.y = Math.max(pad, Math.min(GRAPH_PANEL_HEIGHT - pad, node.y));
          }
          setTick((t) => t + 1);
        });

      simRef.current = sim;
    },
    [graphPanelWidth],
  );

  // Re-initialize simulation when graph changes
  const prevPreset = useRef(presetIdx);

  useEffect(() => {
    const needsReinit =
      prevPreset.current !== presetIdx ||
      positionsRef.current.length !== n;

    if (needsReinit) {
      initSimulation(n, adj);
      prevPreset.current = presetIdx;
    }

    return () => {
      if (simRef.current) simRef.current.stop();
    };
  }, [adj, n, presetIdx, initSimulation]);

  // ─── Walk reset helper ───

  const resetWalk = useCallback(
    (vertex: number) => {
      setIsPlaying(false);
      setWalkState(createWalkState(n, vertex));
      setTvHistory([]);
    },
    [n],
  );

  // ─── Interaction handlers ───

  const handlePresetChange = useCallback(
    (idx: number) => {
      setPresetIdx(idx);
      const g = PRESETS[idx].build();
      setGraph(g);
      setStartVertex(0);
      setIsPlaying(false);
      setWalkState(createWalkState(g.n, 0));
      setTvHistory([]);
    },
    [],
  );

  const handleNodeClick = useCallback(
    (nodeIdx: number) => {
      setStartVertex(nodeIdx);
      resetWalk(nodeIdx);
    },
    [resetWalk],
  );

  const handleReset = useCallback(() => {
    resetWalk(startVertex);
  }, [resetWalk, startVertex]);

  // ─── Walk animation ───

  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setWalkState((prev) => {
        const nbrs = neighbors(adj, prev.currentVertex);
        if (nbrs.length === 0) return prev; // isolated vertex

        let nextVertex: number;
        if (walkType === 'lazy' && Math.random() < 0.5) {
          // Lazy walk: stay with probability 0.5
          nextVertex = prev.currentVertex;
        } else {
          // Uniform random neighbor
          nextVertex = nbrs[Math.floor(Math.random() * nbrs.length)];
        }

        const newVisitCounts = [...prev.visitCounts];
        newVisitCounts[nextVertex] += 1;
        const newTotalSteps = prev.totalSteps + 1;
        const newTrajectory = [...prev.trajectory, nextVertex];

        // Compute empirical distribution and TV distance
        const empirical = newVisitCounts.map((c) => c / newTotalSteps);
        const tv = totalVariationDistance(empirical, stationaryDist);

        setTvHistory((prevTv) => [...prevTv, tv]);

        return {
          currentVertex: nextVertex,
          visitCounts: newVisitCounts,
          totalSteps: newTotalSteps,
          trajectory: newTrajectory,
        };
      });
    }, 1000 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed, walkType, adj, stationaryDist]);

  // ─── Visit frequency color scale ───

  const visitColorScale = useMemo(() => {
    const maxFreq = Math.max(...empiricalDist, 0.01);
    return d3.scaleSequential((t: number) => d3.interpolateRgb('#e0f2f1', '#0F6E56')(t)).domain([0, maxFreq]);
  }, [empiricalDist]);

  // ─── Graph panel rendering ───

  useEffect(() => {
    if (!graphSvgRef.current || graphPanelWidth === 0) return;

    const svg = d3.select(graphSvgRef.current);
    svg.selectAll('*').remove();

    const positions = positionsRef.current;
    if (positions.length !== n) return;

    const edges = edgesFromAdj(adj);

    // Define glow filter for the walker
    const walkerGlowId = `walker-glow-${Math.random().toString(36).slice(2)}`;
    const defs = svg.append('defs');
    const filter = defs.append('filter').attr('id', walkerGlowId);
    filter
      .append('feGaussianBlur')
      .attr('stdDeviation', '4')
      .attr('result', 'coloredBlur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Draw edges
    const edgeG = svg.append('g');
    edgeG
      .selectAll('line')
      .data(edges)
      .join('line')
      .attr('x1', (d) => positions[d.source]?.x ?? 0)
      .attr('y1', (d) => positions[d.source]?.y ?? 0)
      .attr('x2', (d) => positions[d.target]?.x ?? 0)
      .attr('y2', (d) => positions[d.target]?.y ?? 0)
      .style('stroke', 'var(--color-border)')
      .style('stroke-width', '2')
      .style('stroke-opacity', '0.6');

    // Draw nodes (visit-frequency heat map)
    const nodeG = svg.append('g');
    nodeG
      .selectAll<SVGCircleElement, number>('circle.node')
      .data(d3.range(n))
      .join('circle')
      .attr('class', 'node')
      .attr('cx', (i) => positions[i]?.x ?? 0)
      .attr('cy', (i) => positions[i]?.y ?? 0)
      .attr('r', 14)
      .style('fill', (i) => visitColorScale(empiricalDist[i]))
      .style('stroke', 'var(--color-text)')
      .style('stroke-width', '1.5')
      .style('cursor', 'pointer')
      .on('click', function (event, i) {
        event.stopPropagation();
        handleNodeClick(i);
      });

    // Node labels
    nodeG
      .selectAll('text')
      .data(d3.range(n))
      .join('text')
      .attr('x', (i) => positions[i]?.x ?? 0)
      .attr('y', (i) => (positions[i]?.y ?? 0) + 4.5)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('pointer-events', 'none')
      .style('fill', (i) =>
        empiricalDist[i] > (d3.max(empiricalDist) || 0.01) * 0.5
          ? 'white'
          : 'var(--color-text)',
      )
      .text((i) => i);

    // Walker highlight — larger circle with glow
    const walkerPos = positions[walkState.currentVertex];
    if (walkerPos) {
      const walkerG = svg.append('g');

      // Glow circle
      walkerG
        .append('circle')
        .attr('cx', walkerPos.x)
        .attr('cy', walkerPos.y)
        .attr('r', 20)
        .style('fill', AMBER)
        .style('fill-opacity', '0.35')
        .attr('filter', `url(#${walkerGlowId})`);

      // Walker circle
      walkerG
        .append('circle')
        .attr('cx', walkerPos.x)
        .attr('cy', walkerPos.y)
        .attr('r', 10)
        .style('fill', AMBER)
        .style('stroke', 'white')
        .style('stroke-width', '2.5');
    }

    // Panel label
    svg
      .append('text')
      .attr('x', graphPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Graph — click a node to set start vertex');

    // Step counter
    svg
      .append('text')
      .attr('x', graphPanelWidth / 2)
      .attr('y', GRAPH_PANEL_HEIGHT - 8)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', 500)
      .style('fill', 'var(--color-text-secondary)')
      .text(`Step ${walkState.totalSteps - 1}  ·  Current vertex: ${walkState.currentVertex}`);
  }, [
    adj, n, graphPanelWidth, walkState.currentVertex, walkState.totalSteps,
    empiricalDist, visitColorScale, tick, handleNodeClick,
  ]);

  // ─── Bar chart rendering (empirical vs stationary) ───

  useEffect(() => {
    if (!barSvgRef.current || rightPanelWidth === 0) return;

    const svg = d3.select(barSvgRef.current);
    svg.selectAll('*').remove();

    const innerW = rightPanelWidth - MARGIN.left - MARGIN.right;
    const innerH = BAR_CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

    const maxProb = Math.max(
      d3.max(empiricalDist) || 0,
      d3.max(stationaryDist) || 0,
      0.01,
    ) * 1.15;

    const xScale = d3
      .scaleBand<number>()
      .domain(d3.range(n))
      .range([0, innerW])
      .padding(0.2);

    const yScale = d3
      .scaleLinear()
      .domain([0, maxProb])
      .range([innerH, 0]);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Title
    svg
      .append('text')
      .attr('x', rightPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Distribution comparison');

    // Empirical bars (filled)
    g.selectAll('rect.empirical')
      .data(empiricalDist)
      .join('rect')
      .attr('class', 'empirical')
      .attr('x', (_, i) => xScale(i)!)
      .attr('y', (d) => yScale(d))
      .attr('width', xScale.bandwidth())
      .attr('height', (d) => Math.max(0, innerH - yScale(d)))
      .attr('rx', 2)
      .attr('fill', TEAL)
      .style('opacity', '0.75');

    // Stationary bars (outline only)
    g.selectAll('rect.stationary')
      .data(stationaryDist)
      .join('rect')
      .attr('class', 'stationary')
      .attr('x', (_, i) => xScale(i)!)
      .attr('y', (d) => yScale(d))
      .attr('width', xScale.bandwidth())
      .attr('height', (d) => Math.max(0, innerH - yScale(d)))
      .attr('rx', 2)
      .attr('fill', 'none')
      .style('stroke', PURPLE)
      .style('stroke-width', '2')
      .style('stroke-dasharray', '4,2');

    // X-axis
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).tickFormat((d) => String(d)))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .attr('font-size', 10);

    g.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Y-axis
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.2f')))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .attr('font-size', 10);

    g.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Legend
    const legendG = g.append('g').attr('transform', `translate(${innerW - 120}, -10)`);

    legendG.append('rect').attr('x', 0).attr('y', 0).attr('width', 12).attr('height', 8)
      .attr('fill', TEAL).style('opacity', '0.75');
    legendG.append('text').attr('x', 16).attr('y', 7).attr('font-size', 9)
      .style('fill', 'var(--color-text-secondary)').text('Empirical');

    legendG.append('rect').attr('x', 68).attr('y', 0).attr('width', 12).attr('height', 8)
      .attr('fill', 'none').style('stroke', PURPLE).style('stroke-width', '2');
    legendG.append('text').attr('x', 84).attr('y', 7).attr('font-size', 9)
      .style('fill', 'var(--color-text-secondary)').text('Stationary');
  }, [empiricalDist, stationaryDist, n, rightPanelWidth]);

  // ─── TV distance trace rendering ───

  // Find first crossing of the threshold
  const TV_THRESHOLD = 0.25;
  const firstCrossingStep = useMemo(() => {
    for (let i = 0; i < tvHistory.length; i++) {
      if (tvHistory[i] <= TV_THRESHOLD) return i;
    }
    return -1;
  }, [tvHistory]);

  useEffect(() => {
    if (!tvSvgRef.current || rightPanelWidth === 0) return;

    const svg = d3.select(tvSvgRef.current);
    svg.selectAll('*').remove();

    const innerW = rightPanelWidth - MARGIN.left - MARGIN.right;
    const innerH = TV_CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

    const maxSteps = Math.max(tvHistory.length, 10);
    const maxTV = Math.max(d3.max(tvHistory) || 0.5, 0.5);

    const xScale = d3.scaleLinear().domain([0, maxSteps]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, maxTV * 1.05]).range([innerH, 0]);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Title
    svg
      .append('text')
      .attr('x', rightPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Total variation distance');

    // Threshold dashed line
    g.append('line')
      .attr('x1', 0)
      .attr('y1', yScale(TV_THRESHOLD))
      .attr('x2', innerW)
      .attr('y2', yScale(TV_THRESHOLD))
      .style('stroke', SLATE)
      .style('stroke-width', '1.5')
      .style('stroke-dasharray', '6,3')
      .style('opacity', '0.7');

    // Threshold label
    g.append('text')
      .attr('x', innerW - 2)
      .attr('y', yScale(TV_THRESHOLD) - 5)
      .attr('text-anchor', 'end')
      .attr('font-size', 9)
      .style('fill', SLATE)
      .text(`ε = ${TV_THRESHOLD}`);

    // TV distance line
    if (tvHistory.length > 1) {
      const lineGen = d3
        .line<number>()
        .x((_, i) => xScale(i))
        .y((d) => yScale(d))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(tvHistory)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .style('stroke', AMBER)
        .style('stroke-width', '2');
    }

    // Annotate first crossing
    if (firstCrossingStep >= 0) {
      const cx = xScale(firstCrossingStep);
      const cy = yScale(tvHistory[firstCrossingStep]);

      g.append('circle')
        .attr('cx', cx)
        .attr('cy', cy)
        .attr('r', 4)
        .style('fill', AMBER)
        .style('stroke', 'white')
        .style('stroke-width', '1.5');

      g.append('text')
        .attr('x', cx + 6)
        .attr('y', cy - 6)
        .attr('font-size', 9)
        .attr('font-weight', 600)
        .style('fill', AMBER)
        .text(`t = ${firstCrossingStep}`);
    }

    // X-axis
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('d')))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .attr('font-size', 10);

    g.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Y-axis
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.2f')))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .attr('font-size', 10);

    g.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // X-axis label
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .style('fill', 'var(--color-text-secondary)')
      .text('Step');
  }, [tvHistory, rightPanelWidth, firstCrossingStep]);

  // ─── Render ───

  return (
    <div
      ref={containerRef}
      className="w-full rounded-lg border p-4"
      style={{
        borderColor: 'var(--color-border)',
        backgroundColor: 'var(--color-surface)',
      }}
    >
      {/* Controls */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {/* Preset dropdown */}
        <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Graph:</span>
          <select
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              backgroundColor: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
            value={presetIdx}
            onChange={(e) => handlePresetChange(Number(e.target.value))}
          >
            {PRESETS.map((p, i) => (
              <option key={p.label} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>

        {/* Walk type toggle */}
        <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Walk:</span>
          <select
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              backgroundColor: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
            value={walkType}
            onChange={(e) => {
              setWalkType(e.target.value as WalkType);
              resetWalk(startVertex);
            }}
          >
            <option value="standard">Standard</option>
            <option value="lazy">Lazy</option>
          </select>
        </label>

        {/* Speed slider */}
        <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Speed:</span>
          <input
            type="range"
            min={1}
            max={20}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-20"
          />
          <span className="tabular-nums" style={{ color: 'var(--color-text-secondary)', minWidth: '2.5ch' }}>
            {speed}
          </span>
        </label>

        {/* Play / Pause */}
        <button
          className="rounded border px-3 py-1 text-sm font-medium"
          style={{
            borderColor: 'var(--color-border)',
            backgroundColor: isPlaying ? TEAL : 'var(--color-bg)',
            color: isPlaying ? 'white' : 'var(--color-text)',
          }}
          onClick={() => setIsPlaying((v) => !v)}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        {/* Reset */}
        <button
          className="rounded border px-3 py-1 text-sm font-medium"
          style={{
            borderColor: 'var(--color-border)',
            backgroundColor: 'var(--color-bg)',
            color: 'var(--color-text)',
          }}
          onClick={handleReset}
        >
          Reset
        </button>
      </div>

      {/* Main panels */}
      <div
        className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}
        style={{ alignItems: 'flex-start' }}
      >
        {/* Left: force-directed graph */}
        <div style={{ flexShrink: 0 }}>
          <svg
            ref={graphSvgRef}
            width={graphPanelWidth}
            height={GRAPH_PANEL_HEIGHT}
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: 8,
              backgroundColor: 'var(--color-bg)',
            }}
          />
        </div>

        {/* Right: distribution chart + TV trace (stacked) */}
        <div style={{ flexShrink: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <svg
            ref={barSvgRef}
            width={rightPanelWidth}
            height={BAR_CHART_HEIGHT}
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: 8,
              backgroundColor: 'var(--color-bg)',
            }}
          />
          <svg
            ref={tvSvgRef}
            width={rightPanelWidth}
            height={TV_CHART_HEIGHT}
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: 8,
              backgroundColor: 'var(--color-bg)',
            }}
          />
        </div>
      </div>

      {/* Legend */}
      <div
        className="mt-3 flex flex-wrap gap-4 text-xs"
        style={{ color: 'var(--color-text-secondary)' }}
      >
        <span><strong>Click node</strong> to set start vertex</span>
        <span><strong>Node color</strong> encodes visit frequency</span>
        <span>
          <span style={{ color: AMBER, fontWeight: 600 }}>Walker</span> = current position
        </span>
        <span>
          <span style={{ color: TEAL, fontWeight: 600 }}>Filled bars</span> = empirical &nbsp;
          <span style={{ color: PURPLE, fontWeight: 600 }}>Outlined bars</span> = stationary
        </span>
      </div>
    </div>
  );
}

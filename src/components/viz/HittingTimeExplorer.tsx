import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pathGraph, cycleGraph, completeGraph, starGraph, barbellGraph, gridGraph,
  analyzeHittingTimes, stationaryDistribution, degrees, bfsDistance,
  type Graph, type HittingTimeResult,
} from './shared/graphTheory';

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const MAX_NODES = 15;
const GRAPH_PANEL_HEIGHT = 380;
const HEATMAP_CELL = 32;
const BAR_PANEL_HEIGHT = 220;
const MARGIN = { top: 28, right: 16, bottom: 32, left: 120 };

type ViewMode = 'hitting' | 'resistance';

interface Preset {
  label: string;
  build: () => Graph;
}

const PRESETS: Preset[] = [
  { label: 'Path(8)', build: () => pathGraph(8) },
  { label: 'Cycle(8)', build: () => cycleGraph(8) },
  { label: 'Star(7)', build: () => starGraph(7) },
  { label: 'Barbell(4-4)', build: () => barbellGraph(4) },
  { label: 'Complete(6)', build: () => completeGraph(6) },
  { label: 'Grid(3×3)', build: () => gridGraph(3) },
];

// ─── Colors ───

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const SLATE = '#6B6B6B';
const SOURCE_BLUE = '#2563EB';
const TARGET_RED = '#E11D48';

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

/** Format a numeric value: use scientific notation for large values. */
function fmt(v: number): string {
  if (!isFinite(v)) return '\u221E';
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(1);
}

/** Format for bar labels with more precision. */
function fmtBar(v: number): string {
  if (!isFinite(v)) return '\u221E';
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(2);
}

// ─── Component ───

export default function HittingTimeExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const graphSvgRef = useRef<SVGSVGElement>(null);
  const heatmapSvgRef = useRef<SVGSVGElement>(null);
  const barSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───

  const [presetIdx, setPresetIdx] = useState(0);
  const [source, setSource] = useState<number | null>(0);
  const [target, setTarget] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('hitting');

  const graph = useMemo(() => PRESETS[presetIdx].build(), [presetIdx]);
  const n = graph.n;
  const adjacency = graph.adjacency;

  // Node positions managed outside React state for the force simulation.
  const positionsRef = useRef<{ x: number; y: number }[]>([]);
  const simRef = useRef<d3.Simulation<{ x: number; y: number }, undefined> | null>(null);
  const [tick, setTick] = useState(0);

  // ─── Derived computations ───

  const hitResult: HittingTimeResult = useMemo(
    () => analyzeHittingTimes(graph),
    [graph],
  );

  const pi = useMemo(() => stationaryDistribution(adjacency), [adjacency]);
  const deg = useMemo(() => degrees(adjacency), [adjacency]);
  const twoM = useMemo(() => deg.reduce((s, d) => s + d, 0), [deg]);

  // Return times: h(i,i) = 1/π_i = 2m/d_i
  const returnTimes = useMemo(() => {
    return pi.map((p) => (p > 0 ? 1 / p : Infinity));
  }, [pi]);

  // BFS distance for selected pair
  const bfsDist = useMemo(() => {
    if (source === null || target === null) return Infinity;
    return bfsDistance(graph, source, target);
  }, [graph, source, target]);

  // ─── Layout calculations ───

  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const graphPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    if (isNarrow) return containerWidth - 16;
    return Math.floor((containerWidth - 32) * 0.5);
  }, [containerWidth, isNarrow]);

  const heatmapPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    if (isNarrow) return containerWidth - 16;
    return Math.floor((containerWidth - 32) * 0.5);
  }, [containerWidth, isNarrow]);

  const barPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    return containerWidth - 16;
  }, [containerWidth]);

  // ─── Force simulation management ───

  const initSimulation = useCallback(
    (nodeCount: number, adj: number[][]) => {
      if (simRef.current) simRef.current.stop();

      const nodes: { x: number; y: number }[] = Array.from(
        { length: nodeCount },
        (_, i) => ({
          x: graphPanelWidth / 2 + Math.cos((2 * Math.PI * i) / nodeCount) * 80,
          y: GRAPH_PANEL_HEIGHT / 2 + Math.sin((2 * Math.PI * i) / nodeCount) * 80,
        }),
      );
      positionsRef.current = nodes;

      const links = edgesFromAdj(adj);

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
    if (prevPreset.current !== presetIdx || positionsRef.current.length !== n) {
      initSimulation(n, adjacency);
      prevPreset.current = presetIdx;
    }

    return () => {
      if (simRef.current) simRef.current.stop();
    };
  }, [adjacency, n, presetIdx, initSimulation]);

  // Initial mount
  useEffect(() => {
    initSimulation(n, adjacency);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── Interaction handlers ───

  const handlePresetChange = useCallback((idx: number) => {
    setPresetIdx(idx);
    setSource(0);
    setTarget(null);
  }, []);

  const handleNodeClick = useCallback(
    (nodeIdx: number) => {
      if (source === nodeIdx) {
        // Deselect source
        setSource(null);
        setTarget(null);
      } else if (source === null) {
        setSource(nodeIdx);
        setTarget(null);
      } else {
        // Source already set, this click sets or changes target
        if (target === nodeIdx) {
          setTarget(null);
        } else {
          setTarget(nodeIdx);
        }
      }
    },
    [source, target],
  );

  // ─── Graph panel rendering ───

  useEffect(() => {
    if (!graphSvgRef.current || graphPanelWidth === 0) return;

    const svg = d3.select(graphSvgRef.current);
    svg.selectAll('*').remove();

    const positions = positionsRef.current;
    if (positions.length !== n) return;

    const edges = edgesFromAdj(adjacency);

    // Edge width scale: effective resistance → strokeWidth [1, 6]
    const rEffValues = edges.map(
      (e) => hitResult.effectiveResistance[e.source]?.[e.target] ?? 0,
    );
    const rEffExtent = d3.extent(rEffValues) as [number, number];
    const edgeWidthScale = d3
      .scaleLinear()
      .domain([rEffExtent[0] || 0, rEffExtent[1] || 1])
      .range([1, 6])
      .clamp(true);

    // Node size scale: return time → radius [6, 18]
    const returnExtent = d3.extent(returnTimes.filter(isFinite)) as [number, number];
    const nodeRadiusScale = d3
      .scaleLinear()
      .domain([returnExtent[0] || 1, returnExtent[1] || 2])
      .range([6, 18])
      .clamp(true);

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
      .style('stroke', '#94A3B8')
      .style('stroke-width', (d) =>
        String(edgeWidthScale(hitResult.effectiveResistance[d.source]?.[d.target] ?? 0)),
      )
      .style('stroke-opacity', '0.7')
      .style('stroke-linecap', 'round');

    // Node fill
    const nodeFill = (i: number): string => {
      if (i === source) return SOURCE_BLUE;
      if (i === target) return TARGET_RED;
      return '#94A3B8';
    };

    // Draw nodes
    const nodeG = svg.append('g');
    const nodeCircles = nodeG
      .selectAll<SVGCircleElement, number>('circle')
      .data(d3.range(n))
      .join('circle')
      .attr('cx', (i) => positions[i]?.x ?? 0)
      .attr('cy', (i) => positions[i]?.y ?? 0)
      .attr('r', (i) => {
        const rt = returnTimes[i];
        return isFinite(rt) ? nodeRadiusScale(rt) : 6;
      })
      .style('fill', (i) => nodeFill(i))
      .style('stroke', (i) => {
        if (i === source) return '#1D4ED8';
        if (i === target) return '#BE123C';
        return 'var(--color-text)';
      })
      .style('stroke-width', (i) => (i === source || i === target ? '2.5' : '1.5'))
      .style('cursor', 'pointer');

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
      .style('fill', (i) => {
        if (i === source || i === target) return 'white';
        return 'var(--color-text)';
      })
      .text((i) => i);

    // Drag behavior
    const drag = d3
      .drag<SVGCircleElement, number>()
      .on('start', function (_event, i) {
        if (simRef.current) {
          simRef.current.alphaTarget(0.3).restart();
        }
        const pos = positions[i];
        if (pos) {
          (pos as any).fx = pos.x;
          (pos as any).fy = pos.y;
        }
      })
      .on('drag', function (event, i) {
        const pos = positions[i];
        if (pos) {
          (pos as any).fx = event.x;
          (pos as any).fy = event.y;
        }
      })
      .on('end', function (_event, i) {
        if (simRef.current) {
          simRef.current.alphaTarget(0);
        }
        const pos = positions[i];
        if (pos) {
          (pos as any).fx = null;
          (pos as any).fy = null;
        }
      });

    nodeCircles.call(drag);

    // Click handler for nodes
    nodeCircles.on('click', function (event, i) {
      event.stopPropagation();
      handleNodeClick(i);
    });

    // Panel label
    svg
      .append('text')
      .attr('x', graphPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Graph (edge width \u221D R_eff, node size \u221D return time)');
  }, [
    adjacency, n, graphPanelWidth, source, target, hitResult,
    returnTimes, tick, handleNodeClick,
  ]);

  // ─── Heatmap panel rendering ───

  const heatmapSvgHeight = useMemo(() => {
    const cellSize = Math.min(
      HEATMAP_CELL,
      Math.floor((heatmapPanelWidth - 60) / Math.max(n, 1)),
    );
    // Top label (28) + row labels margin (24) + grid + bottom margin (8)
    return 28 + 24 + cellSize * n + 8;
  }, [heatmapPanelWidth, n]);

  useEffect(() => {
    if (!heatmapSvgRef.current || heatmapPanelWidth === 0) return;

    const svg = d3.select(heatmapSvgRef.current);
    svg.selectAll('*').remove();

    const cellSize = Math.min(
      HEATMAP_CELL,
      Math.floor((heatmapPanelWidth - 60) / Math.max(n, 1)),
    );
    const showLabels = n <= 9;

    const matrix =
      viewMode === 'hitting'
        ? hitResult.hittingTimes
        : hitResult.effectiveResistance;

    // Compute extent (excluding diagonal zeros for hitting times)
    const flatVals: number[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j && isFinite(matrix[i][j])) {
          flatVals.push(matrix[i][j]);
        }
      }
    }
    const maxVal = d3.max(flatVals) ?? 1;
    const minVal = d3.min(flatVals) ?? 0;

    const colorScale = d3
      .scaleSequential(d3.interpolateYlOrRd)
      .domain([minVal, maxVal]);

    const gridWidth = cellSize * n;
    const xStart = Math.max(28, (heatmapPanelWidth - gridWidth) / 2);
    const yStart = 48;

    // Title
    svg
      .append('text')
      .attr('x', heatmapPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text(viewMode === 'hitting' ? 'Hitting Times h(i, j)' : 'Effective Resistance R_eff(i, j)');

    const matG = svg
      .append('g')
      .attr('transform', `translate(${xStart}, ${yStart})`);

    // Column labels
    for (let j = 0; j < n; j++) {
      matG
        .append('text')
        .attr('x', j * cellSize + cellSize / 2)
        .attr('y', -6)
        .attr('text-anchor', 'middle')
        .attr('font-size', 9)
        .attr('font-weight', 500)
        .style('fill', j === target ? TARGET_RED : 'var(--color-text-secondary)')
        .text(j);
    }

    // Row labels
    for (let i = 0; i < n; i++) {
      matG
        .append('text')
        .attr('x', -8)
        .attr('y', i * cellSize + cellSize / 2 + 3.5)
        .attr('text-anchor', 'end')
        .attr('font-size', 9)
        .attr('font-weight', 500)
        .style('fill', i === source ? SOURCE_BLUE : 'var(--color-text-secondary)')
        .text(i);
    }

    // Cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const val = matrix[i][j];
        const isOnDiag = i === j;

        matG
          .append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('rx', 2)
          .attr('fill', isOnDiag ? 'var(--color-bg)' : colorScale(isFinite(val) ? val : maxVal))
          .style('stroke', 'var(--color-border)')
          .style('stroke-width', '0.5');

        if (showLabels && !isOnDiag) {
          matG
            .append('text')
            .attr('x', j * cellSize + cellSize / 2)
            .attr('y', i * cellSize + cellSize / 2 + 3.5)
            .attr('text-anchor', 'middle')
            .attr('font-size', Math.min(9, cellSize * 0.35))
            .attr('font-weight', 500)
            .style('fill', () => {
              const normalized = (val - minVal) / (maxVal - minVal || 1);
              return normalized > 0.55 ? 'white' : 'var(--color-text)';
            })
            .text(fmt(val));
        }
      }
    }

    // Highlight source row
    if (source !== null) {
      matG
        .append('rect')
        .attr('x', -1)
        .attr('y', source * cellSize - 1)
        .attr('width', n * cellSize + 1)
        .attr('height', cellSize + 1)
        .attr('fill', 'none')
        .style('stroke', SOURCE_BLUE)
        .style('stroke-width', '2')
        .attr('rx', 2)
        .attr('pointer-events', 'none');
    }

    // Highlight target column
    if (target !== null) {
      matG
        .append('rect')
        .attr('x', target * cellSize - 1)
        .attr('y', -1)
        .attr('width', cellSize + 1)
        .attr('height', n * cellSize + 1)
        .attr('fill', 'none')
        .style('stroke', TARGET_RED)
        .style('stroke-width', '2')
        .attr('rx', 2)
        .attr('pointer-events', 'none');
    }

    // Highlight intersection cell
    if (source !== null && target !== null) {
      matG
        .append('rect')
        .attr('x', target * cellSize - 1)
        .attr('y', source * cellSize - 1)
        .attr('width', cellSize + 1)
        .attr('height', cellSize + 1)
        .attr('fill', 'none')
        .style('stroke', PURPLE)
        .style('stroke-width', '2.5')
        .attr('rx', 2)
        .attr('pointer-events', 'none');
    }
  }, [hitResult, viewMode, n, heatmapPanelWidth, source, target]);

  // ─── Comparison bar chart rendering ───

  useEffect(() => {
    if (!barSvgRef.current || barPanelWidth === 0) return;

    const svg = d3.select(barSvgRef.current);
    svg.selectAll('*').remove();

    if (source === null || target === null) {
      svg
        .append('text')
        .attr('x', barPanelWidth / 2)
        .attr('y', BAR_PANEL_HEIGHT / 2)
        .attr('text-anchor', 'middle')
        .attr('font-size', 13)
        .style('fill', 'var(--color-text-secondary)')
        .text('Select a source (blue) and target (red) node to compare metrics');
      return;
    }

    const hST = hitResult.hittingTimes[source]?.[target] ?? 0;
    const hTS = hitResult.hittingTimes[target]?.[source] ?? 0;
    const kappa = hitResult.commuteTimes[source]?.[target] ?? 0;
    const rEff = hitResult.effectiveResistance[source]?.[target] ?? 0;

    const bars = [
      { label: `Shortest path d(${source}, ${target})`, value: bfsDist, color: TEAL },
      { label: `Hitting time h(${source}, ${target})`, value: hST, color: PURPLE },
      { label: `Hitting time h(${target}, ${source})`, value: hTS, color: PURPLE },
      { label: `Commute time \u03BA(${source}, ${target})`, value: kappa, color: AMBER },
      { label: `R_eff(${source}, ${target})`, value: rEff, color: SLATE },
    ];

    const innerW = barPanelWidth - MARGIN.left - MARGIN.right;
    const innerH = BAR_PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;

    const finiteValues = bars.map((b) => b.value).filter(isFinite);
    const maxBarVal = (d3.max(finiteValues) as number) || 1;

    const xScale = d3.scaleLinear().domain([0, maxBarVal * 1.15]).range([0, innerW]);

    const yScale = d3
      .scaleBand<number>()
      .domain(d3.range(bars.length))
      .range([0, innerH])
      .padding(0.2);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Title
    svg
      .append('text')
      .attr('x', barPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text(`Metrics: ${source} \u2192 ${target}`);

    // Bars
    g.selectAll('rect')
      .data(bars)
      .join('rect')
      .attr('x', 0)
      .attr('y', (_, i) => yScale(i)!)
      .attr('width', (d) => (isFinite(d.value) ? Math.max(1, xScale(d.value)) : innerW))
      .attr('height', yScale.bandwidth())
      .attr('rx', 3)
      .attr('fill', (d) => d.color)
      .style('opacity', '0.85');

    // Bar value labels
    g.selectAll<SVGTextElement, (typeof bars)[number]>('.bar-val')
      .data(bars)
      .join('text')
      .attr('class', 'bar-val')
      .attr('x', (d) => {
        const w = isFinite(d.value) ? Math.max(1, xScale(d.value)) : innerW;
        return w + 6;
      })
      .attr('y', (_, i) => (yScale(i)!) + yScale.bandwidth() / 2 + 4)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text)')
      .text((d) => fmtBar(d.value));

    // Row labels
    g.selectAll<SVGTextElement, (typeof bars)[number]>('.bar-label')
      .data(bars)
      .join('text')
      .attr('class', 'bar-label')
      .attr('x', -8)
      .attr('y', (_, i) => (yScale(i)!) + yScale.bandwidth() / 2 + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', 11)
      .style('fill', 'var(--color-text)')
      .text((d) => d.label);

    // X-axis
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .attr('font-size', 10);

    g.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Asymmetry annotation
    if (Math.abs(hST - hTS) > 0.01) {
      const asymmetryPct = Math.abs(hST - hTS) / Math.max(hST, hTS) * 100;
      g.append('text')
        .attr('x', 0)
        .attr('y', innerH + 28)
        .attr('font-size', 11)
        .attr('font-weight', 500)
        .style('fill', PURPLE)
        .text(
          `Asymmetry: h(${source},${target}) \u2212 h(${target},${source}) = ${fmtBar(hST - hTS)}` +
          ` (${asymmetryPct.toFixed(1)}%)`,
        );
    } else {
      g.append('text')
        .attr('x', 0)
        .attr('y', innerH + 28)
        .attr('font-size', 11)
        .attr('font-weight', 500)
        .style('fill', 'var(--color-text-secondary)')
        .text('Symmetric: h(s,t) = h(t,s) (regular graph)');
    }
  }, [hitResult, source, target, bfsDist, barPanelWidth]);

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

        {/* View toggle */}
        <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Heatmap:</span>
          <select
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              backgroundColor: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as ViewMode)}
          >
            <option value="hitting">Hitting Times</option>
            <option value="resistance">Effective Resistance</option>
          </select>
        </label>

        {/* Selection indicator */}
        <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>
            Source:{' '}
            <strong style={{ color: source !== null ? SOURCE_BLUE : 'var(--color-text-secondary)' }}>
              {source !== null ? source : '\u2014'}
            </strong>
          </span>
          <span>
            Target:{' '}
            <strong style={{ color: target !== null ? TARGET_RED : 'var(--color-text-secondary)' }}>
              {target !== null ? target : '\u2014'}
            </strong>
          </span>
        </div>
      </div>

      {/* Main panels: graph + heatmap */}
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

        {/* Right: heatmap */}
        <div style={{ flexShrink: 1, minWidth: 0, overflow: 'auto' }}>
          <svg
            ref={heatmapSvgRef}
            width={heatmapPanelWidth}
            height={heatmapSvgHeight}
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: 8,
              backgroundColor: 'var(--color-bg)',
            }}
          />
        </div>
      </div>

      {/* Bottom: comparison bar chart */}
      <div className="mt-4">
        <svg
          ref={barSvgRef}
          width={barPanelWidth}
          height={BAR_PANEL_HEIGHT}
          style={{
            border: '1px solid var(--color-border)',
            borderRadius: 8,
            backgroundColor: 'var(--color-bg)',
          }}
        />
      </div>

      {/* Legend */}
      <div
        className="mt-3 flex flex-wrap gap-4 text-xs"
        style={{ color: 'var(--color-text-secondary)' }}
      >
        <span><strong>Click</strong> node: set source (blue)</span>
        <span><strong>Click</strong> another: set target (red)</span>
        <span><strong>Click</strong> source again: deselect</span>
        <span><strong>Drag</strong> node: reposition</span>
      </div>
    </div>
  );
}

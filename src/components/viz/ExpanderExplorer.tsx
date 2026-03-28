import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pathGraph, cycleGraph, completeGraph, barbellGraph, petersenGraph,
  hypercubeGraph, randomRegularGraph, cayleyCirculantGraph,
  getEdges, isBridge,
  adjacencySpectrum, alonBoppanaBound,
  vertexExpansion, edgeExpansionFull, analyzeExpansion,
  createRng,
  type Graph,
} from './shared/graphTheory';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRAPH_PANEL_HEIGHT = 340;
const METRIC_PANEL_HEIGHT = 260;
const SPECTRUM_HEIGHT = 110;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 20, right: 20, bottom: 30, left: 20 };

// Colors
const TEAL = '#14b8a6';
const AMBER = '#f59e0b';
const PURPLE = '#a855f7';
const CUT_STROKE = '#e377c2';
const RED = '#ef4444';
const GREEN_CHECK = '#4daf4a';

// ---------------------------------------------------------------------------
// Graph presets
// ---------------------------------------------------------------------------

type Preset =
  | 'petersen'
  | 'hypercube3'
  | 'hypercube4'
  | 'cycle12'
  | 'path12'
  | 'barbell5'
  | 'random3reg'
  | 'random4reg'
  | 'complete8'
  | 'cayley15';

interface PresetConfig {
  label: string;
  build: (size: number) => Graph;
  defaultSize: number;
  sizeRange: [number, number] | null; // null = size slider disabled
}

const PRESETS: Record<Preset, PresetConfig> = {
  petersen: {
    label: 'Petersen',
    build: () => petersenGraph(),
    defaultSize: 10,
    sizeRange: null,
  },
  hypercube3: {
    label: 'Hypercube Q\u2083',
    build: () => hypercubeGraph(3),
    defaultSize: 8,
    sizeRange: null,
  },
  hypercube4: {
    label: 'Hypercube Q\u2084',
    build: () => hypercubeGraph(4),
    defaultSize: 16,
    sizeRange: null,
  },
  cycle12: {
    label: 'Cycle(n)',
    build: (n) => cycleGraph(n),
    defaultSize: 12,
    sizeRange: [6, 16],
  },
  path12: {
    label: 'Path(n)',
    build: (n) => pathGraph(n),
    defaultSize: 12,
    sizeRange: [6, 16],
  },
  barbell5: {
    label: 'Barbell(k+k)',
    build: (n) => barbellGraph(Math.floor(n / 2)),
    defaultSize: 10,
    sizeRange: [6, 16],
  },
  random3reg: {
    label: 'Random 3-regular',
    build: (n) => randomRegularGraph(n, 3, 42),
    defaultSize: 12,
    sizeRange: [6, 16],
  },
  random4reg: {
    label: 'Random 4-regular',
    build: (n) => randomRegularGraph(n, 4, 42),
    defaultSize: 12,
    sizeRange: [8, 16],
  },
  complete8: {
    label: 'Complete(n)',
    build: (n) => completeGraph(n),
    defaultSize: 8,
    sizeRange: [6, 16],
  },
  cayley15: {
    label: 'Cayley \u2124\u2099 {\u00B11,\u00B12}',
    build: (n) => cayleyCirculantGraph(n, [1, 2]),
    defaultSize: 15,
    sizeRange: [8, 16],
  },
};

// ---------------------------------------------------------------------------
// Deep-clone a graph so mutations don't affect the original
// ---------------------------------------------------------------------------

function cloneGraph(g: Graph): Graph {
  return {
    n: g.n,
    adjacency: g.adjacency.map((row) => [...row]),
    labels: g.labels ? [...g.labels] : undefined,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ExpanderExplorer() {
  // Responsive container
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // SVG refs
  const graphSvgRef = useRef<SVGSVGElement>(null);
  const spectrumSvgRef = useRef<SVGSVGElement>(null);

  // State
  const [preset, setPreset] = useState<Preset>('petersen');
  const [size, setSize] = useState<number>(PRESETS.petersen.defaultSize);
  const [graph, setGraph] = useState<Graph>(() => cloneGraph(PRESETS.petersen.build(PRESETS.petersen.defaultSize)));
  const rngRef = useRef(createRng(42));

  // Derived metrics
  const metrics = useMemo(() => analyzeExpansion(graph), [graph]);
  const edgeCut = useMemo(() => edgeExpansionFull(graph), [graph]);
  const vExp = useMemo(() => vertexExpansion(graph), [graph]);
  const spectrum = useMemo(() => adjacencySpectrum(graph), [graph]);
  const edges = useMemo(() => getEdges(graph), [graph]);

  // Check if all edges are bridges (disable remove button)
  const allBridges = useMemo(() => {
    return edges.every(([u, v]) => isBridge(graph, u, v));
  }, [graph, edges]);

  // Non-edge count (disable add button when complete)
  const nonEdgeCount = useMemo(() => {
    const { n, adjacency } = graph;
    let count = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (adjacency[i][j] === 0) count++;
      }
    }
    return count;
  }, [graph]);

  // Rebuild graph from preset and size
  const rebuildGraph = useCallback((p: Preset, s: number) => {
    const config = PRESETS[p];
    const newGraph = cloneGraph(config.build(s));
    setGraph(newGraph);
  }, []);

  const handlePresetChange = useCallback((newPreset: Preset) => {
    setPreset(newPreset);
    const config = PRESETS[newPreset];
    setSize(config.defaultSize);
    rebuildGraph(newPreset, config.defaultSize);
  }, [rebuildGraph]);

  const handleSizeChange = useCallback((newSize: number) => {
    setSize(newSize);
    rebuildGraph(preset, newSize);
  }, [preset, rebuildGraph]);

  // Add a random edge
  const addRandomEdge = useCallback(() => {
    const { n, adjacency } = graph;
    const nonEdges: [number, number][] = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (adjacency[i][j] === 0) nonEdges.push([i, j]);
      }
    }
    if (nonEdges.length === 0) return;
    const [u, v] = nonEdges[Math.floor(rngRef.current() * nonEdges.length)];
    const newGraph = cloneGraph(graph);
    newGraph.adjacency[u][v] = 1;
    newGraph.adjacency[v][u] = 1;
    setGraph(newGraph);
  }, [graph]);

  // Remove a random non-bridge edge
  const removeRandomEdge = useCallback(() => {
    const currentEdges = getEdges(graph);
    const removable = currentEdges.filter(([u, v]) => !isBridge(graph, u, v));
    if (removable.length === 0) return;
    const [u, v] = removable[Math.floor(rngRef.current() * removable.length)];
    const newGraph = cloneGraph(graph);
    newGraph.adjacency[u][v] = 0;
    newGraph.adjacency[v][u] = 0;
    setGraph(newGraph);
  }, [graph]);

  // =========================================================================
  // Panel 1: Graph force layout with minimum-cut partition highlighted
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(graphSvgRef.current);
    if (!graphSvgRef.current || containerWidth === 0) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.52);
    const panelHeight = GRAPH_PANEL_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    const { n, adjacency } = graph;
    const edgeList = getEdges(graph);

    // Build partition sets for coloring
    const optimalSet = new Set(edgeCut.optimalSet);
    const cutEdgeSet = new Set(
      edgeCut.cutEdges.map(([u, v]) => `${Math.min(u, v)}-${Math.max(u, v)}`)
    );

    // Prepare D3 simulation data
    type SimNode = d3.SimulationNodeDatum & { id: number; inS: boolean };
    const nodes: SimNode[] = Array.from({ length: n }, (_, i) => ({
      id: i,
      inS: optimalSet.has(i),
    }));
    const links = edgeList.map(([s, t]) => ({
      source: s,
      target: t,
      isCut: cutEdgeSet.has(`${Math.min(s, t)}-${Math.max(s, t)}`),
    }));

    const innerW = panelWidth - MARGIN.left - MARGIN.right;
    const innerH = panelHeight - MARGIN.top - MARGIN.bottom - 20;

    const simulation = d3.forceSimulation<SimNode>(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(40).strength(0.8))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(MARGIN.left + innerW / 2, MARGIN.top + 10 + innerH / 2))
      .force('collision', d3.forceCollide(12))
      .stop();

    // Run simulation synchronously
    for (let i = 0; i < 200; i++) simulation.tick();

    // Clamp positions
    for (const node of nodes as any[]) {
      node.x = Math.max(MARGIN.left + 10, Math.min(MARGIN.left + innerW - 10, node.x));
      node.y = Math.max(MARGIN.top + 20, Math.min(MARGIN.top + innerH, node.y));
    }

    const g = svg.append('g');

    // Draw edges
    g.selectAll('.edge')
      .data(links)
      .join('line')
      .attr('class', 'edge')
      .attr('x1', (d: any) => d.source.x)
      .attr('y1', (d: any) => d.source.y)
      .attr('x2', (d: any) => d.target.x)
      .attr('y2', (d: any) => d.target.y)
      .style('stroke', (d: any) => d.isCut ? CUT_STROKE : 'var(--color-text)')
      .style('stroke-opacity', (d: any) => d.isCut ? '0.9' : '0.3')
      .style('stroke-width', (d: any) => d.isCut ? '3' : '1.5')
      .style('stroke-dasharray', (d: any) => d.isCut ? '6,4' : 'none');

    // Draw nodes
    g.selectAll('.node')
      .data(nodes)
      .join('circle')
      .attr('class', 'node')
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)
      .attr('r', 7)
      .style('fill', (d) => d.inS ? TEAL : AMBER)
      .style('fill-opacity', '0.9')
      .style('stroke', 'var(--color-surface)')
      .style('stroke-width', '1.5');

    // Node labels for small graphs
    if (n <= 16) {
      g.selectAll('.label')
        .data(nodes)
        .join('text')
        .attr('class', 'label')
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y - 11)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '9px')
        .style('text-anchor', 'middle')
        .text((d) => String(d.id));
    }

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 14)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`Graph (${n} nodes, ${edgeList.length} edges)`);

    // Annotation at bottom
    const sSize = edgeCut.optimalSet.length;
    const cutCount = edgeCut.cutEdges.length;
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', panelHeight - 6)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-family', 'var(--font-sans)')
      .style('font-size', '11px')
      .text(`|S*| = ${sSize}   |E(S*, S\u0304*)| = ${cutCount}   h(G) = ${edgeCut.expansion.toFixed(3)}`);
  }, [graph, edgeCut, containerWidth, isNarrow]);

  // =========================================================================
  // Panel 3: Eigenvalue spectrum number line
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(spectrumSvgRef.current);
    if (!spectrumSvgRef.current || containerWidth === 0) return;
    svg.selectAll('*').remove();

    const panelWidth = containerWidth;
    const panelHeight = SPECTRUM_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    if (spectrum.length === 0) return;

    const d = metrics.degree;
    const ramBound = metrics.isRegular ? alonBoppanaBound(d) : NaN;

    // Eigenvalue range
    const minEig = Math.min(...spectrum);
    const maxEig = Math.max(...spectrum);
    const pad = Math.max((maxEig - minEig) * 0.1, 0.5);

    const xScale = d3.scaleLinear()
      .domain([minEig - pad, maxEig + pad])
      .range([MARGIN.left + 30, panelWidth - MARGIN.right - 10]);

    const yMid = panelHeight / 2 + 8;

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 14)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('Adjacency Eigenvalue Spectrum');

    // Ramanujan window shading (if regular)
    if (metrics.isRegular && !isNaN(ramBound)) {
      const windowLeft = xScale(-ramBound);
      const windowRight = xScale(ramBound);
      svg.append('rect')
        .attr('x', windowLeft)
        .attr('y', yMid - 18)
        .attr('width', Math.max(0, windowRight - windowLeft))
        .attr('height', 36)
        .style('fill', TEAL)
        .style('fill-opacity', '0.1')
        .attr('rx', 4);

      // Dashed boundary lines for Ramanujan window
      [ramBound, -ramBound].forEach((val) => {
        svg.append('line')
          .attr('x1', xScale(val))
          .attr('y1', yMid - 20)
          .attr('x2', xScale(val))
          .attr('y2', yMid + 20)
          .style('stroke', TEAL)
          .style('stroke-width', '1.5')
          .style('stroke-dasharray', '4,3')
          .style('stroke-opacity', '0.6');
      });

      // Label for Ramanujan window
      svg.append('text')
        .attr('x', xScale(ramBound) + 4)
        .attr('y', yMid - 22)
        .style('fill', TEAL)
        .style('font-size', '9px')
        .style('font-family', 'var(--font-sans)')
        .text(`2\u221A(d\u22121) = ${ramBound.toFixed(2)}`);
    }

    // Axis line
    svg.append('line')
      .attr('x1', MARGIN.left + 30)
      .attr('y1', yMid)
      .attr('x2', panelWidth - MARGIN.right - 10)
      .attr('y2', yMid)
      .style('stroke', 'var(--color-border)')
      .style('stroke-width', '1');

    // Tick marks
    const xAxis = d3.axisBottom(xScale).ticks(8).tickSize(5);
    svg.append('g')
      .attr('transform', `translate(0,${yMid})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '9px');
    svg.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Eigenvalue dots
    const lambda1 = spectrum[0]; // largest eigenvalue = d for regular
    spectrum.forEach((eig, i) => {
      const isLambda1 = i === 0;
      const isOutsideWindow = metrics.isRegular && !isNaN(ramBound) && Math.abs(eig) > ramBound + 1e-9 && !isLambda1;

      svg.append('circle')
        .attr('cx', xScale(eig))
        .attr('cy', yMid)
        .attr('r', isLambda1 ? 6 : 4)
        .style('fill', isLambda1 ? PURPLE : isOutsideWindow ? RED : 'var(--color-text-secondary)')
        .style('fill-opacity', isLambda1 ? '0.9' : isOutsideWindow ? '0.9' : '0.6')
        .style('stroke', 'var(--color-surface)')
        .style('stroke-width', isLambda1 ? '2' : '1');
    });

    // Label for lambda1
    svg.append('text')
      .attr('x', xScale(lambda1))
      .attr('y', yMid - 14)
      .style('text-anchor', 'middle')
      .style('fill', PURPLE)
      .style('font-size', '10px')
      .style('font-weight', '600')
      .style('font-family', 'var(--font-sans)')
      .text(`\u03BB\u2081 = ${lambda1.toFixed(2)}`);

  }, [spectrum, metrics, containerWidth, isNarrow]);

  // =========================================================================
  // Metric bars (rendered as HTML, not SVG)
  // =========================================================================

  const config = PRESETS[preset];
  const hasSizeSlider = config.sizeRange !== null;

  // Compute max for metric bars scaling
  const maxMetricVal = Math.max(
    vExp.expansion,
    edgeCut.expansion,
    metrics.spectralParameter,
    metrics.isRegular ? metrics.ramanujanBound : 0,
    metrics.degree,
    1
  );

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '12px',
          alignItems: 'center',
          marginBottom: '12px',
          padding: '10px 12px',
          borderRadius: '8px',
          border: '1px solid var(--color-border)',
          background: 'var(--color-surface)',
        }}
      >
        {/* Preset selector */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
          <span style={{ fontWeight: 600 }}>Graph:</span>
          <select
            value={preset}
            onChange={(e) => handlePresetChange(e.target.value as Preset)}
            style={{
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg)',
              color: 'var(--color-text)',
              fontSize: '13px',
            }}
          >
            {(Object.keys(PRESETS) as Preset[]).map((p) => (
              <option key={p} value={p}>{PRESETS[p].label}</option>
            ))}
          </select>
        </label>

        {/* Size slider */}
        {hasSizeSlider && (
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
            <span style={{ fontWeight: 600 }}>n:</span>
            <input
              type="range"
              min={config.sizeRange![0]}
              max={config.sizeRange![1]}
              step={preset === 'barbell5' ? 2 : 1}
              value={size}
              onChange={(e) => handleSizeChange(parseInt(e.target.value))}
              style={{ width: '100px' }}
            />
            <span style={{ minWidth: '28px', textAlign: 'right', color: 'var(--color-text-secondary)', fontSize: '12px' }}>
              {size}
            </span>
          </label>
        )}

        {/* Edge buttons */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={addRandomEdge}
            disabled={nonEdgeCount === 0}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: nonEdgeCount === 0 ? 'var(--color-surface)' : 'var(--color-bg)',
              color: nonEdgeCount === 0 ? 'var(--color-text-secondary)' : 'var(--color-text)',
              fontSize: '12px',
              cursor: nonEdgeCount === 0 ? 'not-allowed' : 'pointer',
              opacity: nonEdgeCount === 0 ? 0.5 : 1,
            }}
          >
            + Add edge
          </button>
          <button
            onClick={removeRandomEdge}
            disabled={allBridges || edges.length === 0}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: (allBridges || edges.length === 0) ? 'var(--color-surface)' : 'var(--color-bg)',
              color: (allBridges || edges.length === 0) ? 'var(--color-text-secondary)' : 'var(--color-text)',
              fontSize: '12px',
              cursor: (allBridges || edges.length === 0) ? 'not-allowed' : 'pointer',
              opacity: (allBridges || edges.length === 0) ? 0.5 : 1,
            }}
          >
            &minus; Remove edge
          </button>
        </div>
      </div>

      {/* Top row: graph + metrics */}
      <div
        style={{
          display: 'flex',
          flexDirection: isNarrow ? 'column' : 'row',
          gap: '12px',
          marginBottom: '12px',
        }}
      >
        {/* Left: graph panel */}
        <div
          style={{
            flex: isNarrow ? undefined : '0 0 52%',
            border: '1px solid var(--color-border)',
            borderRadius: '8px',
            overflow: 'hidden',
            background: 'var(--color-surface)',
          }}
        >
          <svg ref={graphSvgRef} />
        </div>

        {/* Right: metric bars + Ramanujan check */}
        <div
          style={{
            flex: isNarrow ? undefined : '1 1 0%',
            border: '1px solid var(--color-border)',
            borderRadius: '8px',
            padding: '16px',
            background: 'var(--color-surface)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            gap: '16px',
            minHeight: isNarrow ? undefined : METRIC_PANEL_HEIGHT,
          }}
        >
          {/* Title */}
          <div
            style={{
              fontSize: '13px',
              fontWeight: 600,
              color: 'var(--color-text)',
              fontFamily: 'var(--font-sans)',
              marginBottom: '4px',
            }}
          >
            Expansion Metrics
          </div>

          {/* Vertex expansion bar */}
          <MetricBar
            label="h_V(G)"
            sublabel="Vertex expansion"
            value={vExp.expansion}
            maxVal={maxMetricVal}
            color={TEAL}
            isInfinite={vExp.expansion === Infinity}
          />

          {/* Edge expansion bar */}
          <MetricBar
            label="h(G)"
            sublabel="Edge expansion"
            value={edgeCut.expansion}
            maxVal={maxMetricVal}
            color={AMBER}
          />

          {/* Spectral parameter bar with Ramanujan threshold */}
          <MetricBar
            label={'\u03BB(G)'}
            sublabel="Spectral parameter"
            value={metrics.spectralParameter}
            maxVal={maxMetricVal}
            color={PURPLE}
            threshold={metrics.isRegular ? metrics.ramanujanBound : undefined}
            thresholdLabel={metrics.isRegular ? `2\u221A(d\u22121)` : undefined}
          />

          {/* Ramanujan check */}
          <div
            style={{
              fontSize: '12px',
              color: 'var(--color-text)',
              fontFamily: 'var(--font-sans)',
              padding: '8px 10px',
              borderRadius: '6px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg)',
            }}
          >
            <span style={{ fontWeight: 600 }}>Ramanujan: </span>
            {!metrics.isRegular ? (
              <span style={{ color: 'var(--color-text-secondary)' }}>N/A (irregular)</span>
            ) : metrics.isRamanujan ? (
              <span style={{ color: GREEN_CHECK, fontWeight: 600 }}>
                &#10003; Yes &mdash; \u03BB(G) = {metrics.spectralParameter.toFixed(3)} &le; {metrics.ramanujanBound.toFixed(3)} = 2&radic;(d&minus;1)
              </span>
            ) : (
              <span style={{ color: RED, fontWeight: 600 }}>
                &#10007; No &mdash; \u03BB(G) = {metrics.spectralParameter.toFixed(3)} &gt; {metrics.ramanujanBound.toFixed(3)} = 2&radic;(d&minus;1)
              </span>
            )}
            {metrics.isRegular && (
              <div style={{ marginTop: '4px', color: 'var(--color-text-secondary)', fontSize: '11px' }}>
                d = {metrics.degree}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom: eigenvalue spectrum number line */}
      <div
        style={{
          border: '1px solid var(--color-border)',
          borderRadius: '8px',
          overflow: 'hidden',
          background: 'var(--color-surface)',
        }}
      >
        <svg ref={spectrumSvgRef} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MetricBar sub-component
// ---------------------------------------------------------------------------

interface MetricBarProps {
  label: string;
  sublabel: string;
  value: number;
  maxVal: number;
  color: string;
  threshold?: number;
  thresholdLabel?: string;
  isInfinite?: boolean;
}

function MetricBar({ label, sublabel, value, maxVal, color, threshold, thresholdLabel, isInfinite }: MetricBarProps) {
  const barMax = Math.max(maxVal, threshold ?? 0) * 1.15;
  const widthPct = isInfinite ? 100 : Math.min((value / barMax) * 100, 100);
  const thresholdPct = threshold !== undefined ? Math.min((threshold / barMax) * 100, 100) : undefined;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
        <div style={{ fontSize: '12px', fontFamily: 'var(--font-sans)' }}>
          <span style={{ fontWeight: 600, color: 'var(--color-text)' }}>{label}</span>
          <span style={{ color: 'var(--color-text-secondary)', marginLeft: '6px', fontSize: '11px' }}>{sublabel}</span>
        </div>
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-text)', fontFamily: 'var(--font-sans)' }}>
          {isInfinite ? '\u221E' : value.toFixed(3)}
        </span>
      </div>
      <div
        style={{
          position: 'relative',
          height: '14px',
          background: 'var(--color-bg)',
          borderRadius: '4px',
          overflow: 'visible',
          border: '1px solid var(--color-border)',
        }}
      >
        {/* Value bar */}
        <div
          style={{
            height: '100%',
            width: `${widthPct}%`,
            background: color,
            borderRadius: '3px',
            opacity: 0.7,
            transition: 'width 0.3s ease',
          }}
        />
        {/* Threshold marker */}
        {thresholdPct !== undefined && (
          <div
            style={{
              position: 'absolute',
              left: `${thresholdPct}%`,
              top: '-3px',
              bottom: '-3px',
              width: '2px',
              background: 'var(--color-text-secondary)',
              borderRadius: '1px',
            }}
            title={`${thresholdLabel} = ${threshold!.toFixed(3)}`}
          >
            <div
              style={{
                position: 'absolute',
                top: '-14px',
                left: '50%',
                transform: 'translateX(-50%)',
                fontSize: '9px',
                color: 'var(--color-text-secondary)',
                whiteSpace: 'nowrap',
                fontFamily: 'var(--font-sans)',
              }}
            >
              {thresholdLabel}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

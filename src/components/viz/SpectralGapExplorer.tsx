import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pathGraph, cycleGraph, completeGraph, barbellGraph, gridGraph,
  erdosRenyiGraph,
  analyzeSpectrum, getEdges, isConnected, isBridge,
  createRng,
  type Graph,
} from './shared/graphTheory';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRAPH_PANEL_HEIGHT = 340;
const SPECTRUM_HEIGHT = 100;
const EVOLUTION_HEIGHT = 180;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 20, right: 20, bottom: 35, left: 45 };
const MAX_HISTORY = 50;

// ---------------------------------------------------------------------------
// Graph families
// ---------------------------------------------------------------------------

type GraphFamily = 'path' | 'cycle' | 'complete' | 'barbell' | 'grid' | 'er';

interface FamilyConfig {
  label: string;
  paramLabel: string;
  min: number;
  max: number;
  step: number;
  default: number;
  build: (param: number) => Graph;
}

const FAMILIES: Record<GraphFamily, FamilyConfig> = {
  path: {
    label: 'Path(n)',
    paramLabel: 'n',
    min: 4, max: 12, step: 1, default: 6,
    build: (n) => pathGraph(n),
  },
  cycle: {
    label: 'Cycle(n)',
    paramLabel: 'n',
    min: 4, max: 12, step: 1, default: 6,
    build: (n) => cycleGraph(n),
  },
  complete: {
    label: 'Complete(n)',
    paramLabel: 'n',
    min: 4, max: 12, step: 1, default: 6,
    build: (n) => completeGraph(n),
  },
  barbell: {
    label: 'Barbell(k–k)',
    paramLabel: 'k',
    min: 3, max: 6, step: 1, default: 3,
    build: (k) => barbellGraph(k),
  },
  grid: {
    label: 'Grid(m×m)',
    paramLabel: 'm',
    min: 2, max: 4, step: 1, default: 3,
    build: (m) => gridGraph(m),
  },
  er: {
    label: 'Erdős–Rényi(n,p)',
    paramLabel: 'p',
    min: 0.1, max: 0.9, step: 0.05, default: 0.3,
    build: (p) => {
      // Try seeds until we get a connected graph
      for (let seed = 42; seed < 142; seed++) {
        const g = erdosRenyiGraph(10, p, seed);
        if (isConnected(g)) return g;
      }
      // Fallback: return the seed-42 graph even if disconnected
      return erdosRenyiGraph(10, p, 42);
    },
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

export default function SpectralGapExplorer() {
  // Responsive container
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // SVG refs — one per panel
  const graphSvgRef = useRef<SVGSVGElement>(null);
  const spectrumSvgRef = useRef<SVGSVGElement>(null);
  const evolutionSvgRef = useRef<SVGSVGElement>(null);

  // State
  const [family, setFamily] = useState<GraphFamily>('path');
  const [param, setParam] = useState<number>(FAMILIES.path.default);
  const [graph, setGraph] = useState<Graph>(() => cloneGraph(FAMILIES.path.build(FAMILIES.path.default)));
  const [history, setHistory] = useState<number[]>([]);
  const rngRef = useRef(createRng(42));

  // Derived
  const spectrum = useMemo(() => analyzeSpectrum(graph), [graph]);
  const edges = useMemo(() => getEdges(graph), [graph]);
  const lambda2 = spectrum.fiedlerValue;

  // Check if all edges are bridges (disable remove button)
  const allBridges = useMemo(() => {
    return edges.every(([u, v]) => isBridge(graph, u, v));
  }, [graph, edges]);

  // Seed the history with the initial lambda2 on graph rebuild
  const rebuildGraph = useCallback((fam: GraphFamily, p: number) => {
    const config = FAMILIES[fam];
    const newGraph = cloneGraph(config.build(p));
    const sp = analyzeSpectrum(newGraph);
    setGraph(newGraph);
    setHistory([sp.fiedlerValue]);
  }, []);

  // On family change, reset param and rebuild
  const handleFamilyChange = useCallback((newFamily: GraphFamily) => {
    setFamily(newFamily);
    const config = FAMILIES[newFamily];
    setParam(config.default);
    rebuildGraph(newFamily, config.default);
  }, [rebuildGraph]);

  // On param change, rebuild
  const handleParamChange = useCallback((newParam: number) => {
    setParam(newParam);
    rebuildGraph(family, newParam);
  }, [family, rebuildGraph]);

  // Add a random edge
  const addRandomEdge = useCallback(() => {
    const { n, adjacency } = graph;
    // Collect all non-edges
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
    const sp = analyzeSpectrum(newGraph);
    setGraph(newGraph);
    setHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), sp.fiedlerValue]);
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
    const sp = analyzeSpectrum(newGraph);
    setGraph(newGraph);
    setHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), sp.fiedlerValue]);
  }, [graph]);

  // Initialize history on mount
  useEffect(() => {
    setHistory([lambda2]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // =========================================================================
  // Panel 1: Graph force layout
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(graphSvgRef.current);
    if (!graphSvgRef.current || containerWidth === 0) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.48);
    const panelHeight = GRAPH_PANEL_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    const { n, adjacency } = graph;
    const edgeList = getEdges(graph);

    // Prepare D3 simulation data
    type SimNode = d3.SimulationNodeDatum & { index: number };
    const nodes: SimNode[] = Array.from({ length: n }, (_, i) => ({ index: i }));
    const links = edgeList.map(([s, t]) => ({ source: s, target: t }));

    const simulation = d3.forceSimulation<SimNode>(nodes)
      .force('link', d3.forceLink(links).id((_, i) => i).distance(40).strength(0.8))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(panelWidth / 2, panelHeight / 2))
      .force('collision', d3.forceCollide(12));

    const g = svg.append('g');

    const linkElements = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .style('stroke', 'var(--color-text-secondary)')
      .style('stroke-opacity', '0.4')
      .style('stroke-width', '1.5');

    const nodeElements = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', 6)
      .style('fill', 'var(--color-text)')
      .style('fill-opacity', '0.8')
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', '1.5');

    // Labels for small graphs
    const labelElements = n <= 12
      ? g.append('g')
          .selectAll('text')
          .data(nodes)
          .join('text')
          .text((d) => String(d.index))
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '9px')
          .style('text-anchor', 'middle')
          .attr('dy', -10)
      : null;

    simulation.on('tick', () => {
      linkElements
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);
      nodeElements
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);
      if (labelElements) {
        labelElements
          .attr('x', (d: any) => d.x)
          .attr('y', (d: any) => d.y);
      }
    });

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 14)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`Graph (${n} nodes, ${edgeList.length} edges)`);

    return () => {
      simulation.stop();
    };
  }, [graph, containerWidth, isNarrow]);

  // =========================================================================
  // Panel 2: Eigenvalue spectrum number line
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(spectrumSvgRef.current);
    if (!spectrumSvgRef.current || containerWidth === 0) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.48);
    const panelHeight = SPECTRUM_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    const eigenvalues = spectrum.eigen.eigenvalues;
    const maxEig = Math.max(...eigenvalues) * 1.1 || 1;

    const xScale = d3.scaleLinear()
      .domain([0, maxEig])
      .range([MARGIN.left, panelWidth - MARGIN.right]);

    const yMid = panelHeight / 2 + 5;

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 14)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('Eigenvalue Spectrum');

    // Axis line
    svg.append('line')
      .attr('x1', MARGIN.left)
      .attr('y1', yMid)
      .attr('x2', panelWidth - MARGIN.right)
      .attr('y2', yMid)
      .style('stroke', 'var(--color-border)')
      .style('stroke-width', '1');

    // Tick marks
    const xAxis = d3.axisBottom(xScale).ticks(5).tickSize(5);
    svg.append('g')
      .attr('transform', `translate(0,${yMid})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');
    svg.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Eigenvalue dots (skip λ₂ for separate rendering)
    eigenvalues.forEach((eig, i) => {
      if (i === 1 && eigenvalues.length > 1) return; // skip λ₂
      svg.append('circle')
        .attr('cx', xScale(eig))
        .attr('cy', yMid)
        .attr('r', 4)
        .style('fill', 'var(--color-text-secondary)')
        .style('fill-opacity', '0.6');
    });

    // λ₂ dot — highlighted
    if (eigenvalues.length > 1) {
      const l2 = eigenvalues[1];

      // Vertical dotted line
      svg.append('line')
        .attr('x1', xScale(l2))
        .attr('y1', yMid - 25)
        .attr('x2', xScale(l2))
        .attr('y2', yMid + 20)
        .style('stroke', '#ef4444')
        .style('stroke-width', '1.5')
        .style('stroke-dasharray', '4,3');

      // Larger dot
      svg.append('circle')
        .attr('cx', xScale(l2))
        .attr('cy', yMid)
        .attr('r', 7)
        .style('fill', '#ef4444')
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', '2');

      // Label
      svg.append('text')
        .attr('x', xScale(l2))
        .attr('y', yMid - 30)
        .style('text-anchor', 'middle')
        .style('fill', '#ef4444')
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(`λ₂ = ${l2.toFixed(3)}`);
    }
  }, [spectrum, containerWidth, isNarrow]);

  // =========================================================================
  // Panel 3: λ₂ evolution line chart
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(evolutionSvgRef.current);
    if (!evolutionSvgRef.current || containerWidth === 0) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.48);
    const panelHeight = EVOLUTION_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    if (history.length === 0) return;

    const xMax = Math.max(history.length - 1, 1);
    const yMax = Math.max(...history) * 1.2 || 1;

    const xScale = d3.scaleLinear()
      .domain([0, xMax])
      .range([MARGIN.left, panelWidth - MARGIN.right]);

    const yScale = d3.scaleLinear()
      .domain([0, yMax])
      .range([panelHeight - MARGIN.bottom, MARGIN.top]);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 14)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('λ₂ Evolution');

    // Axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(Math.min(history.length, 10))
      .tickFormat(d3.format('d'));
    const yAxis = d3.axisLeft(yScale).ticks(5);

    svg.append('g')
      .attr('transform', `translate(0,${panelHeight - MARGIN.bottom})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    svg.append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(yAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    svg.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // X-axis label
    svg.append('text')
      .attr('x', (MARGIN.left + panelWidth - MARGIN.right) / 2)
      .attr('y', panelHeight - 3)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('Step');

    // Y-axis label
    svg.append('text')
      .attr('transform', `rotate(-90)`)
      .attr('x', -(MARGIN.top + panelHeight - MARGIN.bottom) / 2)
      .attr('y', 13)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('λ₂');

    // Line
    const line = d3.line<number>()
      .x((_, i) => xScale(i))
      .y((d) => yScale(d));

    svg.append('path')
      .datum(history)
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', '#ef4444')
      .style('stroke-width', '2');

    // Dots
    svg.selectAll('circle.evo-dot')
      .data(history)
      .join('circle')
      .attr('class', 'evo-dot')
      .attr('cx', (_, i) => xScale(i))
      .attr('cy', (d) => yScale(d))
      .attr('r', 3)
      .style('fill', '#ef4444')
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', '1');
  }, [history, containerWidth, isNarrow]);

  // =========================================================================
  // Check if we can add edges (complete graph check)
  // =========================================================================

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

  // =========================================================================
  // Render
  // =========================================================================

  const config = FAMILIES[family];

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
        {/* Family selector */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
          <span style={{ fontWeight: 600 }}>Family:</span>
          <select
            value={family}
            onChange={(e) => handleFamilyChange(e.target.value as GraphFamily)}
            style={{
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg)',
              color: 'var(--color-text)',
              fontSize: '13px',
            }}
          >
            {(Object.keys(FAMILIES) as GraphFamily[]).map((f) => (
              <option key={f} value={f}>{FAMILIES[f].label}</option>
            ))}
          </select>
        </label>

        {/* Size parameter slider */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
          <span style={{ fontWeight: 600 }}>{config.paramLabel}:</span>
          <input
            type="range"
            min={config.min}
            max={config.max}
            step={config.step}
            value={param}
            onChange={(e) => handleParamChange(parseFloat(e.target.value))}
            style={{ width: '100px' }}
          />
          <span style={{ minWidth: '28px', textAlign: 'right', color: 'var(--color-text-secondary)', fontSize: '12px' }}>
            {family === 'er' ? param.toFixed(2) : param}
          </span>
        </label>

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
            − Remove edge
          </button>
        </div>
      </div>

      {/* Panels */}
      <div
        style={{
          display: 'flex',
          flexDirection: isNarrow ? 'column' : 'row',
          gap: '12px',
          justifyContent: 'space-between',
        }}
      >
        {/* Left: graph */}
        <div
          style={{
            flex: isNarrow ? undefined : '0 0 48%',
            border: '1px solid var(--color-border)',
            borderRadius: '8px',
            overflow: 'hidden',
            background: 'var(--color-surface)',
          }}
        >
          <svg ref={graphSvgRef} />
        </div>

        {/* Right: spectrum + evolution stacked */}
        <div
          style={{
            flex: isNarrow ? undefined : '0 0 48%',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
          }}
        >
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
          <div
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              overflow: 'hidden',
              background: 'var(--color-surface)',
            }}
          >
            <svg ref={evolutionSvgRef} />
          </div>
        </div>
      </div>
    </div>
  );
}

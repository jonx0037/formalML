import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  petersenGraph, cycleGraph, randomRegularGraph,
  getEdges, degrees, adjacencySpectrum, spectralParameter,
  expanderMixingLemma, emlAllSubsetPairs,
  createRng,
  type Graph, type EMLResult,
} from './shared/graphTheory';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRAPH_PANEL_HEIGHT = 360;
const BAR_CHART_HEIGHT = 180;
const HISTOGRAM_HEIGHT = 200;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 24, right: 20, bottom: 35, left: 50 };

const COLOR_TEAL = '#0d9488';
const COLOR_AMBER = '#d97706';
const COLOR_PURPLE = '#7c3aed';
const COLOR_GRAY = '#6b7280';

// ---------------------------------------------------------------------------
// Graph Presets
// ---------------------------------------------------------------------------

interface PresetConfig {
  id: string;
  label: string;
  build: () => Graph;
}

const PRESETS: PresetConfig[] = [
  { id: 'petersen', label: 'Petersen graph', build: () => petersenGraph() },
  { id: 'cycle10', label: 'Cycle(10)', build: () => cycleGraph(10) },
  { id: 'random3', label: 'Random 3-regular(10)', build: () => randomRegularGraph(10, 3, 42) },
  { id: 'random4', label: 'Random 4-regular(10)', build: () => randomRegularGraph(10, 4, 42) },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function cloneGraph(g: Graph): Graph {
  return {
    n: g.n,
    adjacency: g.adjacency.map((row) => [...row]),
    labels: g.labels ? [...g.labels] : undefined,
  };
}

/** Pick k random distinct indices from [0, n). */
function randomSubset(n: number, k: number, rng: () => number): number[] {
  const pool = Array.from({ length: n }, (_, i) => i);
  const result: number[] = [];
  for (let i = 0; i < k && pool.length > 0; i++) {
    const idx = Math.floor(rng() * pool.length);
    result.push(pool[idx]);
    pool.splice(idx, 1);
  }
  return result.sort((a, b) => a - b);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MixingLemmaExplorer() {
  // Responsive container
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // SVG refs
  const graphSvgRef = useRef<SVGSVGElement>(null);
  const barSvgRef = useRef<SVGSVGElement>(null);
  const histSvgRef = useRef<SVGSVGElement>(null);

  // State
  const [presetId, setPresetId] = useState('petersen');
  const [graph, setGraph] = useState<Graph>(() => cloneGraph(PRESETS[0].build()));
  const [setS, setSetS] = useState<number[]>([]);
  const [setT, setSetT] = useState<number[]>([]);
  const [addingTo, setAddingTo] = useState<'S' | 'T'>('S');
  const [subsetSize, setSubsetSize] = useState(3);
  const rngRef = useRef(createRng(77));

  // Derived: EML result for current S, T
  const emlResult: EMLResult | null = useMemo(() => {
    if (setS.length === 0 || setT.length === 0) return null;
    return expanderMixingLemma(graph, setS, setT);
  }, [graph, setS, setT]);

  // Derived: spectral parameter
  const lambda = useMemo(() => spectralParameter(graph), [graph]);

  // Derived: degree (average or regular)
  const avgDegree = useMemo(() => {
    const deg = degrees(graph.adjacency);
    return deg.reduce((s, d) => s + d, 0) / (deg.length || 1);
  }, [graph]);

  // Derived: histogram data (expensive, cached on graph + subsetSize)
  const histogramData = useMemo(() => {
    return emlAllSubsetPairs(graph, subsetSize);
  }, [graph, subsetSize]);

  // Handle preset change
  const handlePresetChange = useCallback((id: string) => {
    const preset = PRESETS.find((p) => p.id === id);
    if (!preset) return;
    setPresetId(id);
    setGraph(cloneGraph(preset.build()));
    setSetS([]);
    setSetT([]);
    setAddingTo('S');
  }, []);

  // Handle node click (toggle in S or T)
  const handleNodeClick = useCallback((nodeIdx: number) => {
    if (addingTo === 'S') {
      setSetS((prev) =>
        prev.includes(nodeIdx)
          ? prev.filter((v) => v !== nodeIdx)
          : [...prev, nodeIdx].sort((a, b) => a - b)
      );
    } else {
      setSetT((prev) =>
        prev.includes(nodeIdx)
          ? prev.filter((v) => v !== nodeIdx)
          : [...prev, nodeIdx].sort((a, b) => a - b)
      );
    }
  }, [addingTo]);

  // Random S, T
  const handleRandom = useCallback(() => {
    const k = Math.min(subsetSize, graph.n);
    const s = randomSubset(graph.n, k, rngRef.current);
    const t = randomSubset(graph.n, k, rngRef.current);
    setSetS(s);
    setSetT(t);
  }, [graph, subsetSize]);

  // Clear selection
  const handleClear = useCallback(() => {
    setSetS([]);
    setSetT([]);
    setAddingTo('S');
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

    const { n } = graph;
    const edgeList = getEdges(graph);

    const sSet = new Set(setS);
    const tSet = new Set(setT);

    // Prepare D3 simulation data
    type SimNode = d3.SimulationNodeDatum & { index: number };
    const nodes: SimNode[] = Array.from({ length: n }, (_, i) => ({ index: i }));
    const links = edgeList.map(([s, t]) => ({ source: s, target: t }));

    const simulation = d3.forceSimulation<SimNode>(nodes)
      .force('link', d3.forceLink(links).id((_, i) => i).distance(50).strength(0.8))
      .force('charge', d3.forceManyBody().strength(-150))
      .force('center', d3.forceCenter(panelWidth / 2, panelHeight / 2 + 10))
      .force('collision', d3.forceCollide(14));

    const g = svg.append('g');

    // Determine if an edge is between S and T
    const isSTEdge = (s: number, t: number) =>
      (sSet.has(s) && tSet.has(t)) || (sSet.has(t) && tSet.has(s));

    const hasSelection = setS.length > 0 || setT.length > 0;

    const linkElements = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .style('stroke', (d: any) => {
        const si = typeof d.source === 'object' ? d.source.index : d.source;
        const ti = typeof d.target === 'object' ? d.target.index : d.target;
        return isSTEdge(si, ti) ? COLOR_PURPLE : 'var(--color-text-secondary)';
      })
      .style('stroke-opacity', (d: any) => {
        if (!hasSelection) return '0.4';
        const si = typeof d.source === 'object' ? d.source.index : d.source;
        const ti = typeof d.target === 'object' ? d.target.index : d.target;
        return isSTEdge(si, ti) ? '0.9' : '0.15';
      })
      .style('stroke-width', (d: any) => {
        const si = typeof d.source === 'object' ? d.source.index : d.source;
        const ti = typeof d.target === 'object' ? d.target.index : d.target;
        return isSTEdge(si, ti) ? '3' : '1.5';
      });

    // Node color based on set membership
    const nodeColor = (idx: number) => {
      const inS = sSet.has(idx);
      const inT = tSet.has(idx);
      if (inS && inT) return COLOR_PURPLE;
      if (inS) return COLOR_TEAL;
      if (inT) return COLOR_AMBER;
      return 'var(--color-text)';
    };

    const nodeOpacity = (idx: number) => {
      if (!hasSelection) return '0.8';
      const inS = sSet.has(idx);
      const inT = tSet.has(idx);
      return (inS || inT) ? '1' : '0.25';
    };

    const nodeElements = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', 8)
      .style('fill', (d) => nodeColor(d.index))
      .style('fill-opacity', (d) => nodeOpacity(d.index))
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', '2')
      .style('cursor', 'pointer')
      .on('click', (_, d) => {
        handleNodeClick(d.index);
      });

    // Labels
    const labelElements = n <= 15
      ? g.append('g')
          .selectAll('text')
          .data(nodes)
          .join('text')
          .text((d) => String(d.index))
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '9px')
          .style('text-anchor', 'middle')
          .style('pointer-events', 'none')
          .attr('dy', -13)
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
      .text(`Graph (n=${n}, d=${avgDegree.toFixed(1)}, \u03BB=${lambda.toFixed(3)})`);

    // Annotations when we have a selection
    if (emlResult) {
      const annoY = panelHeight - 8;
      const annoLines = [
        `|S|=${setS.length}  |T|=${setT.length}  |E(S,T)|=${emlResult.actualEdges}`,
        `expected=${emlResult.expectedEdges.toFixed(2)}  bound=\u03BB\u221A(|S||T|)=${emlResult.emlBound.toFixed(2)}`,
      ];
      annoLines.forEach((line, i) => {
        svg.append('text')
          .attr('x', panelWidth / 2)
          .attr('y', annoY - (annoLines.length - 1 - i) * 14)
          .style('text-anchor', 'middle')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '11px')
          .text(line);
      });
    }

    return () => {
      simulation.stop();
    };
  }, [graph, containerWidth, isNarrow, setS, setT, emlResult, lambda, avgDegree, handleNodeClick]);

  // =========================================================================
  // Panel 2: Deviation bar chart
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(barSvgRef.current);
    if (!barSvgRef.current || containerWidth === 0) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.48);
    const panelHeight = BAR_CHART_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('Edge Count vs. EML Bound');

    if (!emlResult) {
      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', panelHeight / 2)
        .style('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Select nodes for S and T to see comparison');
      return;
    }

    const { actualEdges, expectedEdges, emlBound, withinBound } = emlResult;

    // Bar chart: actual, expected, and bound band
    const upperBound = expectedEdges + emlBound;
    const lowerBound = Math.max(0, expectedEdges - emlBound);
    const yMax = Math.max(actualEdges, upperBound, expectedEdges) * 1.2 || 1;

    const chartLeft = MARGIN.left + 10;
    const chartRight = panelWidth - MARGIN.right - 10;
    const chartTop = MARGIN.top + 8;
    const chartBottom = panelHeight - MARGIN.bottom;

    const yScale = d3.scaleLinear()
      .domain([0, yMax])
      .range([chartBottom, chartTop]);

    // Y axis
    const yAxis = d3.axisLeft(yScale).ticks(5);
    svg.append('g')
      .attr('transform', `translate(${chartLeft},0)`)
      .call(yAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');
    svg.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    const barWidth = Math.min(60, (chartRight - chartLeft) / 5);
    const cx1 = chartLeft + (chartRight - chartLeft) * 0.3;
    const cx2 = chartLeft + (chartRight - chartLeft) * 0.7;

    // EML bound band (shaded region around expected)
    svg.append('rect')
      .attr('x', cx2 - barWidth * 0.7)
      .attr('y', yScale(upperBound))
      .attr('width', barWidth * 1.4)
      .attr('height', Math.max(0, yScale(lowerBound) - yScale(upperBound)))
      .style('fill', COLOR_PURPLE)
      .style('fill-opacity', '0.12')
      .style('stroke', COLOR_PURPLE)
      .style('stroke-width', '1')
      .style('stroke-dasharray', '4,3')
      .style('stroke-opacity', '0.5');

    // Expected bar (gray)
    svg.append('rect')
      .attr('x', cx2 - barWidth / 2)
      .attr('y', yScale(expectedEdges))
      .attr('width', barWidth)
      .attr('height', Math.max(0, yScale(0) - yScale(expectedEdges)))
      .style('fill', COLOR_GRAY)
      .style('fill-opacity', '0.4');

    // Actual bar (teal)
    svg.append('rect')
      .attr('x', cx1 - barWidth / 2)
      .attr('y', yScale(actualEdges))
      .attr('width', barWidth)
      .attr('height', Math.max(0, yScale(0) - yScale(actualEdges)))
      .style('fill', COLOR_TEAL)
      .style('fill-opacity', '0.8');

    // Labels under bars
    svg.append('text')
      .attr('x', cx1)
      .attr('y', chartBottom + 14)
      .style('text-anchor', 'middle')
      .style('fill', COLOR_TEAL)
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(`|E(S,T)| = ${actualEdges}`);

    svg.append('text')
      .attr('x', cx2)
      .attr('y', chartBottom + 14)
      .style('text-anchor', 'middle')
      .style('fill', COLOR_GRAY)
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(`expected = ${expectedEdges.toFixed(1)}`);

    // Check mark or X
    const checkX = panelWidth - 30;
    const checkY = chartTop + 10;
    svg.append('text')
      .attr('x', checkX)
      .attr('y', checkY)
      .style('text-anchor', 'middle')
      .style('font-size', '18px')
      .style('fill', withinBound ? '#22c55e' : '#ef4444')
      .text(withinBound ? '\u2713' : '\u2717');

    svg.append('text')
      .attr('x', checkX)
      .attr('y', checkY + 14)
      .style('text-anchor', 'middle')
      .style('font-size', '9px')
      .style('fill', 'var(--color-text-secondary)')
      .text(withinBound ? 'within' : 'exceeds');
  }, [emlResult, containerWidth, isNarrow]);

  // =========================================================================
  // Panel 3: Histogram of all deviations
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(histSvgRef.current);
    if (!histSvgRef.current || containerWidth === 0) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.48);
    const panelHeight = HISTOGRAM_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text(`All C(n,${subsetSize})\u00B2 Subset-Pair Deviations`);

    if (histogramData.length === 0) {
      svg.append('text')
        .attr('x', panelWidth / 2)
        .attr('y', panelHeight / 2)
        .style('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('No data');
      return;
    }

    const deviations = histogramData.map((d) => d.deviation);
    const bound = histogramData.length > 0 ? histogramData[0].bound : 0;
    const maxDev = Math.max(...deviations, bound * 1.2) || 1;

    const chartLeft = MARGIN.left;
    const chartRight = panelWidth - MARGIN.right;
    const chartTop = MARGIN.top + 4;
    const chartBottom = panelHeight - MARGIN.bottom;

    const xScale = d3.scaleLinear()
      .domain([0, maxDev])
      .range([chartLeft, chartRight]);

    // Build histogram bins
    const binGenerator = d3.bin()
      .domain([0, maxDev] as [number, number])
      .thresholds(d3.range(0, maxDev, maxDev / 25));
    const bins = binGenerator(deviations);

    const yMax = d3.max(bins, (b) => b.length) || 1;
    const yScale = d3.scaleLinear()
      .domain([0, yMax])
      .range([chartBottom, chartTop]);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(6);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    svg.append('g')
      .attr('transform', `translate(0,${chartBottom})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    svg.append('g')
      .attr('transform', `translate(${chartLeft},0)`)
      .call(yAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    svg.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // X-axis label
    svg.append('text')
      .attr('x', (chartLeft + chartRight) / 2)
      .attr('y', panelHeight - 2)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('deviation |E(S,T) \u2212 expected|');

    // Y-axis label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(chartTop + chartBottom) / 2)
      .attr('y', 13)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('count');

    // Bars
    svg.selectAll('rect.hist-bar')
      .data(bins)
      .join('rect')
      .attr('class', 'hist-bar')
      .attr('x', (d) => xScale(d.x0 ?? 0) + 1)
      .attr('y', (d) => yScale(d.length))
      .attr('width', (d) => Math.max(0, xScale(d.x1 ?? 0) - xScale(d.x0 ?? 0) - 2))
      .attr('height', (d) => Math.max(0, chartBottom - yScale(d.length)))
      .style('fill', COLOR_TEAL)
      .style('fill-opacity', '0.7');

    // EML bound line (dashed red)
    svg.append('line')
      .attr('x1', xScale(bound))
      .attr('y1', chartTop - 4)
      .attr('x2', xScale(bound))
      .attr('y2', chartBottom)
      .style('stroke', '#ef4444')
      .style('stroke-width', '2')
      .style('stroke-dasharray', '6,4');

    svg.append('text')
      .attr('x', xScale(bound) + 4)
      .attr('y', chartTop + 6)
      .style('fill', '#ef4444')
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(`\u03BB\u221A(|S||T|) = ${bound.toFixed(2)}`);

    // Count how many exceed the bound
    const exceeding = deviations.filter((d) => d > bound + 1e-9).length;
    if (exceeding === 0) {
      svg.append('text')
        .attr('x', chartRight - 4)
        .attr('y', chartTop + 6)
        .style('text-anchor', 'end')
        .style('fill', '#22c55e')
        .style('font-size', '10px')
        .style('font-weight', '600')
        .text('\u2713 All within bound');
    } else {
      svg.append('text')
        .attr('x', chartRight - 4)
        .attr('y', chartTop + 6)
        .style('text-anchor', 'end')
        .style('fill', '#ef4444')
        .style('font-size', '10px')
        .style('font-weight', '600')
        .text(`\u2717 ${exceeding} exceed bound`);
    }
  }, [histogramData, subsetSize, containerWidth, isNarrow]);

  // =========================================================================
  // Render
  // =========================================================================

  const btnStyle = (disabled = false): React.CSSProperties => ({
    padding: '4px 10px',
    borderRadius: '4px',
    border: '1px solid var(--color-border)',
    background: disabled ? 'var(--color-surface)' : 'var(--color-bg)',
    color: disabled ? 'var(--color-text-secondary)' : 'var(--color-text)',
    fontSize: '12px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
  });

  const radioBtnStyle = (active: boolean): React.CSSProperties => ({
    padding: '4px 10px',
    borderRadius: '4px',
    border: `1.5px solid ${active ? (addingTo === 'S' ? COLOR_TEAL : COLOR_AMBER) : 'var(--color-border)'}`,
    background: active ? (addingTo === 'S' ? `${COLOR_TEAL}18` : `${COLOR_AMBER}18`) : 'var(--color-bg)',
    color: active ? (addingTo === 'S' ? COLOR_TEAL : COLOR_AMBER) : 'var(--color-text-secondary)',
    fontSize: '12px',
    cursor: 'pointer',
    fontWeight: active ? 600 : 400,
  });

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
        {/* Graph preset */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
          <span style={{ fontWeight: 600 }}>Graph:</span>
          <select
            value={presetId}
            onChange={(e) => handlePresetChange(e.target.value)}
            style={{
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg)',
              color: 'var(--color-text)',
              fontSize: '13px',
            }}
          >
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        </label>

        {/* Adding to S / T toggle */}
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)', fontWeight: 600 }}>Click adds to:</span>
          <button
            onClick={() => setAddingTo('S')}
            style={radioBtnStyle(addingTo === 'S')}
          >
            S
          </button>
          <button
            onClick={() => setAddingTo('T')}
            style={radioBtnStyle(addingTo === 'T')}
          >
            T
          </button>
        </div>

        {/* Subset size slider */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
          <span style={{ fontWeight: 600 }}>k:</span>
          <input
            type="range"
            min={2}
            max={5}
            step={1}
            value={subsetSize}
            onChange={(e) => setSubsetSize(parseInt(e.target.value, 10))}
            style={{ width: '80px' }}
          />
          <span style={{ minWidth: '20px', textAlign: 'right', color: 'var(--color-text-secondary)', fontSize: '12px' }}>
            {subsetSize}
          </span>
        </label>

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={handleRandom} style={btnStyle()}>
            Random S, T
          </button>
          <button
            onClick={handleClear}
            disabled={setS.length === 0 && setT.length === 0}
            style={btnStyle(setS.length === 0 && setT.length === 0)}
          >
            Clear selection
          </button>
        </div>
      </div>

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '14px',
          marginBottom: '10px',
          fontSize: '11px',
          color: 'var(--color-text-secondary)',
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: COLOR_TEAL }} />
          S ({setS.length})
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: COLOR_AMBER }} />
          T ({setT.length})
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: COLOR_PURPLE }} />
          S \u2229 T
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ display: 'inline-block', width: 16, height: 3, background: COLOR_PURPLE, borderRadius: 2 }} />
          edges between S and T
        </span>
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

        {/* Right: bar chart + histogram stacked */}
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
            <svg ref={barSvgRef} />
          </div>
          <div
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              overflow: 'hidden',
              background: 'var(--color-surface)',
            }}
          >
            <svg ref={histSvgRef} />
          </div>
        </div>
      </div>
    </div>
  );
}

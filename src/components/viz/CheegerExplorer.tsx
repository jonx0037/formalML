import { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  barbellGraph, pathGraph, cycleGraph, gridGraph, twoCliquesBridge,
  randomRegularGraph, erdosRenyiGraph,
  analyzeSpectrum, cheegerConstant, fiedlerPartition, getEdges, degrees,
  type Graph, type CheegerResult,
} from './shared/graphTheory';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PANEL_HEIGHT = 320;
const SCATTER_HEIGHT = 300;
const BOUNDS_HEIGHT = 100;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 24, right: 20, bottom: 40, left: 44 };
const SCATTER_MARGIN = { top: 30, right: 20, bottom: 44, left: 52 };

const BLUE = '#2171b5';
const RED = '#d94801';
const CUT_STROKE = '#e377c2';

type Preset =
  | 'barbell-5'
  | 'path-10'
  | 'cycle-10'
  | 'grid-4'
  | 'two-cliques-5'
  | 'random-3-regular-10';

type CutMode = 'min-cut' | 'fiedler';

interface PresetOption {
  value: Preset;
  label: string;
}

const PRESETS: PresetOption[] = [
  { value: 'barbell-5', label: 'Barbell (5+5)' },
  { value: 'path-10', label: 'Path (10)' },
  { value: 'cycle-10', label: 'Cycle (10)' },
  { value: 'grid-4', label: 'Grid (4×4)' },
  { value: 'two-cliques-5', label: 'Two cliques (5+5)' },
  { value: 'random-3-regular-10', label: 'Random 3-regular (10)' },
];

interface ScatterPoint {
  h: number;
  lambda2: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildGraph(preset: Preset): Graph {
  switch (preset) {
    case 'barbell-5': return barbellGraph(5);
    case 'path-10': return pathGraph(10);
    case 'cycle-10': return cycleGraph(10);
    case 'grid-4': return gridGraph(4);
    case 'two-cliques-5': return twoCliquesBridge(5);
    case 'random-3-regular-10': return randomRegularGraph(10, 3, 42);
  }
}

/** Check if a graph is connected (all nodes reachable from node 0). */
function isConnected(g: Graph): boolean {
  if (g.n === 0) return true;
  const visited = new Set<number>();
  const stack = [0];
  while (stack.length > 0) {
    const u = stack.pop()!;
    if (visited.has(u)) continue;
    visited.add(u);
    for (let v = 0; v < g.n; v++) {
      if (g.adjacency[u][v] > 0 && !visited.has(v)) {
        stack.push(v);
      }
    }
  }
  return visited.size === g.n;
}

/** Generate scatter data: 100 Erdos-Renyi random graphs. */
function generateScatterData(): ScatterPoint[] {
  const points: ScatterPoint[] = [];
  for (let seed = 0; seed < 100; seed++) {
    const p = 0.15 + (seed / 99) * (0.8 - 0.15);
    const g = erdosRenyiGraph(10, p, seed);
    // Skip disconnected graphs — Cheeger inequality for normalized
    // Laplacian assumes connectivity.
    if (!isConnected(g)) continue;
    const cr = cheegerConstant(g);
    const sp = analyzeSpectrum(g);
    // Use the normalized Laplacian's second-smallest eigenvalue
    const lambda2 = sp.normalizedEigen.eigenvalues[1];
    if (lambda2 > 0 && cr.cheegerConstant > 0) {
      points.push({ h: cr.cheegerConstant, lambda2 });
    }
  }
  return points;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CheegerExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const graphSvgRef = useRef<SVGSVGElement>(null);
  const boundsSvgRef = useRef<SVGSVGElement>(null);
  const scatterSvgRef = useRef<SVGSVGElement>(null);

  const [preset, setPreset] = useState<Preset>('barbell-5');
  const [cutMode, setCutMode] = useState<CutMode>('min-cut');

  // Pre-generate scatter data once on mount
  const scatterDataRef = useRef<ScatterPoint[] | null>(null);
  if (scatterDataRef.current === null) {
    scatterDataRef.current = generateScatterData();
  }

  // Build graph from preset
  const graph = useMemo(() => buildGraph(preset), [preset]);

  // Spectrum (normalized Laplacian eigenvalues)
  const spectrum = useMemo(() => analyzeSpectrum(graph), [graph]);

  // Cheeger result based on cut mode
  const cheegerResult = useMemo<CheegerResult>(() => {
    if (cutMode === 'fiedler') return fiedlerPartition(graph);
    return cheegerConstant(graph);
  }, [graph, cutMode]);

  // Fiedler partition for comparison (always computed for stats)
  const minCutResult = useMemo(() => cheegerConstant(graph), [graph]);

  // Normalized lambda2
  const lambda2 = spectrum.normalizedEigen.eigenvalues[1];
  const h = minCutResult.cheegerConstant;

  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // Panel widths
  const graphPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    if (isNarrow) return Math.min(containerWidth - 16, 560);
    return Math.floor((containerWidth - 24) * 0.5);
  }, [containerWidth, isNarrow]);

  const boundsPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    if (isNarrow) return Math.min(containerWidth - 16, 560);
    return Math.floor((containerWidth - 24) * 0.5);
  }, [containerWidth, isNarrow]);

  const scatterPanelWidth = useMemo(() => {
    if (!containerWidth) return 600;
    return Math.min(containerWidth - 16, 700);
  }, [containerWidth]);

  // -----------------------------------------------------------------------
  // Graph panel (force-directed layout with cut highlighted)
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!graphSvgRef.current || graphPanelWidth === 0) return;

    const svg = d3.select(graphSvgRef.current);
    svg.selectAll('*').remove();

    const w = graphPanelWidth;
    const ht = PANEL_HEIGHT;
    const innerW = w - MARGIN.left - MARGIN.right;
    const innerH = ht - MARGIN.top - MARGIN.bottom;

    // Title
    svg.append('text')
      .attr('x', w / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 13)
      .attr('font-weight', 600)
      .text('Graph & Cut');

    const g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Build partition sets for coloring
    const [sideA, sideB] = cheegerResult.optimalPartition;
    const sideASet = new Set(sideA);
    const cutEdgeSet = new Set(
      cheegerResult.cutEdges.map(([u, v]) => `${Math.min(u, v)}-${Math.max(u, v)}`)
    );

    // Build node and link data for force simulation
    const nodes = Array.from({ length: graph.n }, (_, i) => ({
      id: i,
      side: sideASet.has(i) ? 'A' : 'B',
    }));

    const edges = getEdges(graph);
    const links = edges.map(([source, target]) => ({
      source,
      target,
      isCut: cutEdgeSet.has(`${Math.min(source, target)}-${Math.max(source, target)}`),
    }));

    // Force simulation
    const sim = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force('link', d3.forceLink(links as d3.SimulationLinkDatum<d3.SimulationNodeDatum>[])
        .id((d: any) => d.id)
        .distance(40))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(innerW / 2, innerH / 2))
      .force('collision', d3.forceCollide(12))
      .stop();

    // Run simulation synchronously
    for (let i = 0; i < 200; i++) sim.tick();

    // Clamp positions
    for (const node of nodes as any[]) {
      node.x = Math.max(10, Math.min(innerW - 10, node.x));
      node.y = Math.max(10, Math.min(innerH - 10, node.y));
    }

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
      .attr('stroke-opacity', (d: any) => d.isCut ? 0.9 : 0.3)
      .attr('stroke-width', (d: any) => d.isCut ? 3 : 1.5)
      .attr('stroke-dasharray', (d: any) => d.isCut ? '6,4' : 'none');

    // Draw nodes
    g.selectAll('.node')
      .data(nodes)
      .join('circle')
      .attr('class', 'node')
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)
      .attr('r', 7)
      .attr('fill', (d) => d.side === 'A' ? BLUE : RED)
      .attr('fill-opacity', 0.9)
      .style('stroke', 'var(--color-surface)')
      .attr('stroke-width', 1.5);

    // Cheeger constant label
    svg.append('text')
      .attr('x', w / 2)
      .attr('y', ht - 6)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .text(`h(G) = ${cheegerResult.cheegerConstant.toFixed(4)}`);

  }, [graph, cheegerResult, graphPanelWidth]);

  // -----------------------------------------------------------------------
  // Bounds panel (number line showing h²/2 ≤ λ₂ ≤ 2h)
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!boundsSvgRef.current || boundsPanelWidth === 0) return;

    const svg = d3.select(boundsSvgRef.current);
    svg.selectAll('*').remove();

    const w = boundsPanelWidth;
    const ht = BOUNDS_HEIGHT;
    const pad = 20;
    const lineY = ht / 2 + 4;

    // Title
    svg.append('text')
      .attr('x', w / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 13)
      .attr('font-weight', 600)
      .text('Cheeger Inequality Bounds');

    const lowerBound = (h * h) / 2;
    const upperBound = 2 * h;

    // Domain: from 0 to max of (2h, lambda2) with some padding
    const domMax = Math.max(upperBound, lambda2, 0.01) * 1.15;
    const xScale = d3.scaleLinear()
      .domain([0, domMax])
      .range([MARGIN.left + pad, w - MARGIN.right - pad]);

    // Shaded valid region
    svg.append('rect')
      .attr('x', xScale(lowerBound))
      .attr('y', lineY - 12)
      .attr('width', Math.max(0, xScale(upperBound) - xScale(lowerBound)))
      .attr('height', 24)
      .attr('fill', '#4daf4a')
      .attr('fill-opacity', 0.15)
      .attr('rx', 4);

    // Base line
    svg.append('line')
      .attr('x1', xScale(0))
      .attr('y1', lineY)
      .attr('x2', xScale(domMax))
      .attr('y2', lineY)
      .style('stroke', 'var(--color-border)')
      .attr('stroke-width', 1.5);

    // Marker helper
    function drawMarker(value: number, label: string, color: string, above: boolean) {
      const x = xScale(value);
      // Tick
      svg.append('line')
        .attr('x1', x)
        .attr('y1', lineY - 8)
        .attr('x2', x)
        .attr('y2', lineY + 8)
        .style('stroke', color)
        .attr('stroke-width', 2.5);

      // Label
      svg.append('text')
        .attr('x', x)
        .attr('y', above ? lineY - 14 : lineY + 22)
        .attr('text-anchor', 'middle')
        .style('fill', color)
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('font-weight', 600)
        .text(`${label} = ${value.toFixed(3)}`);
    }

    drawMarker(lowerBound, 'h\u00B2/2', '#666', true);
    drawMarker(lambda2, '\u03BB\u2082', BLUE, false);
    drawMarker(upperBound, '2h', '#666', true);

    // Inequality text
    svg.append('text')
      .attr('x', w / 2)
      .attr('y', ht - 4)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .text('h\u00B2/2  \u2264  \u03BB\u2082  \u2264  2h');

  }, [lambda2, h, boundsPanelWidth]);

  // -----------------------------------------------------------------------
  // Scatter panel (h vs λ₂ for random graphs, with bound curves)
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!scatterSvgRef.current || scatterPanelWidth === 0) return;
    const scatterData = scatterDataRef.current;
    if (!scatterData || scatterData.length === 0) return;

    const svg = d3.select(scatterSvgRef.current);
    svg.selectAll('*').remove();

    const w = scatterPanelWidth;
    const ht = SCATTER_HEIGHT;
    const innerW = w - SCATTER_MARGIN.left - SCATTER_MARGIN.right;
    const innerH = ht - SCATTER_MARGIN.top - SCATTER_MARGIN.bottom;

    // Title
    svg.append('text')
      .attr('x', w / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 13)
      .attr('font-weight', 600)
      .text('Cheeger Inequality: h(G) vs \u03BB\u2082 for Random Graphs');

    const g = svg.append('g')
      .attr('transform', `translate(${SCATTER_MARGIN.left}, ${SCATTER_MARGIN.top})`);

    // Determine domain
    const allH = scatterData.map((d) => d.h).concat([h]);
    const allL = scatterData.map((d) => d.lambda2).concat([lambda2]);
    const hMax = (d3.max(allH) as number) * 1.1;
    const lMax = Math.min((d3.max(allL) as number) * 1.15, 2.1);

    const xScale = d3.scaleLinear().domain([0, hMax]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, lMax]).range([innerH, 0]);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(6);
    const yAxis = d3.axisLeft(yScale).ticks(6);

    const xAxisG = g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(xAxis);
    xAxisG.selectAll('line').style('stroke', 'var(--color-border)');
    xAxisG.selectAll('path').style('stroke', 'var(--color-border)');
    xAxisG.selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 9);

    const yAxisG = g.append('g').call(yAxis);
    yAxisG.selectAll('line').style('stroke', 'var(--color-border)');
    yAxisG.selectAll('path').style('stroke', 'var(--color-border)');
    yAxisG.selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 9);

    // Grid lines
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6).tickSize(-innerH).tickFormat(() => ''))
      .selectAll('line')
      .style('stroke', 'var(--color-border)')
      .attr('stroke-opacity', 0.15);
    g.select('.domain').remove();

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(6).tickSize(-innerW).tickFormat(() => ''))
      .selectAll('line')
      .style('stroke', 'var(--color-border)')
      .attr('stroke-opacity', 0.15);

    // Axis labels
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 36)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .text('h(G)');

    g.append('text')
      .attr('x', -innerH / 2)
      .attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .text('\u03BB\u2082');

    // Draw bound curves: lower bound y = x²/2, upper bound y = 2x
    const nSteps = 200;
    const curveClip = g.append('clipPath')
      .attr('id', 'cheeger-scatter-clip')
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', innerW)
      .attr('height', innerH);

    const curveGroup = g.append('g')
      .attr('clip-path', 'url(#cheeger-scatter-clip)');

    // Shaded feasible region between the two curves
    const areaPoints: [number, number][] = [];
    for (let i = 0; i <= nSteps; i++) {
      const hVal = (i / nSteps) * hMax;
      const lower = (hVal * hVal) / 2;
      areaPoints.push([hVal, lower]);
    }
    // Upper boundary going backwards
    const areaPointsUpper: [number, number][] = [];
    for (let i = nSteps; i >= 0; i--) {
      const hVal = (i / nSteps) * hMax;
      const upper = 2 * hVal;
      areaPointsUpper.push([hVal, upper]);
    }
    const allAreaPts = areaPoints.concat(areaPointsUpper);

    curveGroup.append('path')
      .datum(allAreaPts)
      .attr('d', d3.line()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(Math.max(0, Math.min(d[1], lMax))))
      )
      .attr('fill', '#4daf4a')
      .attr('fill-opacity', 0.08)
      .attr('stroke', 'none');

    // Lower bound curve: y = x²/2
    const lowerCurve: [number, number][] = [];
    for (let i = 0; i <= nSteps; i++) {
      const hVal = (i / nSteps) * hMax;
      lowerCurve.push([hVal, (hVal * hVal) / 2]);
    }

    curveGroup.append('path')
      .datum(lowerCurve)
      .attr('d', d3.line()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]))
      )
      .attr('fill', 'none')
      .style('stroke', 'var(--color-text-secondary)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,3')
      .attr('stroke-opacity', 0.7);

    // Upper bound curve: y = 2x
    const upperCurve: [number, number][] = [];
    for (let i = 0; i <= nSteps; i++) {
      const hVal = (i / nSteps) * hMax;
      upperCurve.push([hVal, 2 * hVal]);
    }

    curveGroup.append('path')
      .datum(upperCurve)
      .attr('d', d3.line()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]))
      )
      .attr('fill', 'none')
      .style('stroke', 'var(--color-text-secondary)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,3')
      .attr('stroke-opacity', 0.7);

    // Curve labels
    // Lower bound label
    const lbLabelH = Math.min(hMax * 0.7, 1.2);
    const lbLabelY = (lbLabelH * lbLabelH) / 2;
    if (lbLabelY < lMax) {
      curveGroup.append('text')
        .attr('x', xScale(lbLabelH) + 4)
        .attr('y', yScale(lbLabelY) + 14)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('font-weight', 500)
        .text('\u03BB\u2082 = h\u00B2/2');
    }

    // Upper bound label
    const ubLabelH = Math.min(hMax * 0.45, 0.8);
    const ubLabelY = 2 * ubLabelH;
    if (ubLabelY < lMax) {
      curveGroup.append('text')
        .attr('x', xScale(ubLabelH) + 4)
        .attr('y', yScale(ubLabelY) - 6)
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('font-weight', 500)
        .text('\u03BB\u2082 = 2h');
    }

    // Scatter points
    g.selectAll('.scatter-pt')
      .data(scatterData)
      .join('circle')
      .attr('class', 'scatter-pt')
      .attr('cx', (d) => xScale(d.h))
      .attr('cy', (d) => yScale(d.lambda2))
      .attr('r', 3.5)
      .attr('fill', '#999')
      .attr('fill-opacity', 0.5)
      .style('stroke', 'var(--color-surface)')
      .attr('stroke-width', 0.5);

    // Current graph's point (highlighted)
    g.append('circle')
      .attr('cx', xScale(h))
      .attr('cy', yScale(lambda2))
      .attr('r', 7)
      .attr('fill', BLUE)
      .attr('fill-opacity', 0.9)
      .style('stroke', 'var(--color-surface)')
      .attr('stroke-width', 2);

    // Label for current point
    g.append('text')
      .attr('x', xScale(h) + 10)
      .attr('y', yScale(lambda2) + 4)
      .style('fill', BLUE)
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .attr('font-weight', 600)
      .text('current');

  }, [h, lambda2, scatterPanelWidth]);

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------
  return (
    <div ref={containerRef} className="w-full space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Graph preset */}
        <label
          className="flex items-center gap-1.5 text-xs font-medium"
          style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}
        >
          Graph:
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value as Preset)}
            className="rounded-md px-2 py-1 text-xs"
            style={{
              fontFamily: 'var(--font-sans)',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              border: '1px solid var(--color-border)',
            }}
          >
            {PRESETS.map((p) => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </label>

        {/* Cut mode toggle */}
        <div
          className="flex items-center gap-1.5 text-xs font-medium"
          style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}
        >
          Cut:
          <button
            onClick={() => setCutMode('min-cut')}
            className="rounded-md px-2.5 py-1 text-xs font-medium transition-colors"
            style={{
              fontFamily: 'var(--font-sans)',
              background: cutMode === 'min-cut' ? 'var(--color-accent)' : 'var(--color-surface)',
              color: cutMode === 'min-cut' ? '#fff' : 'var(--color-text)',
              border: `1px solid ${cutMode === 'min-cut' ? 'var(--color-accent)' : 'var(--color-border)'}`,
            }}
          >
            Minimum cut
          </button>
          <button
            onClick={() => setCutMode('fiedler')}
            className="rounded-md px-2.5 py-1 text-xs font-medium transition-colors"
            style={{
              fontFamily: 'var(--font-sans)',
              background: cutMode === 'fiedler' ? 'var(--color-accent)' : 'var(--color-surface)',
              color: cutMode === 'fiedler' ? '#fff' : 'var(--color-text)',
              border: `1px solid ${cutMode === 'fiedler' ? 'var(--color-accent)' : 'var(--color-border)'}`,
            }}
          >
            Fiedler partition
          </button>
        </div>
      </div>

      {/* Top row: graph + bounds panels */}
      <div
        className="flex gap-3"
        style={{ flexDirection: isNarrow ? 'column' : 'row' }}
      >
        {/* Graph panel */}
        <svg
          ref={graphSvgRef}
          width={graphPanelWidth}
          height={PANEL_HEIGHT}
          className="rounded-lg border border-[var(--color-border)]"
        />

        {/* Bounds panel */}
        <div className="flex flex-col gap-2" style={{ width: isNarrow ? undefined : boundsPanelWidth }}>
          <svg
            ref={boundsSvgRef}
            width={boundsPanelWidth}
            height={BOUNDS_HEIGHT}
            className="rounded-lg border border-[var(--color-border)]"
          />
          {/* Stats */}
          <div
            className="rounded-lg border px-3 py-2 text-xs leading-relaxed"
            style={{
              fontFamily: 'var(--font-sans)',
              color: 'var(--color-text-secondary)',
              borderColor: 'var(--color-border)',
              background: 'var(--color-surface)',
            }}
          >
            <div><strong style={{ color: 'var(--color-text)' }}>h(G)</strong> = {h.toFixed(4)}</div>
            <div><strong style={{ color: 'var(--color-text)' }}>{'\u03BB\u2082'}</strong> = {lambda2.toFixed(4)}</div>
            <div>
              <strong style={{ color: 'var(--color-text)' }}>Bounds:</strong>{' '}
              {((h * h) / 2).toFixed(4)} {'\u2264'} {lambda2.toFixed(4)} {'\u2264'} {(2 * h).toFixed(4)}
              {lambda2 >= (h * h) / 2 - 1e-6 && lambda2 <= 2 * h + 1e-6
                ? <span style={{ color: '#4daf4a', marginLeft: 6 }}>&#10003;</span>
                : <span style={{ color: RED, marginLeft: 6 }}>&#10007;</span>}
            </div>
            {cutMode === 'fiedler' && (
              <div style={{ marginTop: 4, opacity: 0.8 }}>
                Fiedler h = {cheegerResult.cheegerConstant.toFixed(4)}
                {cheegerResult.cheegerConstant > minCutResult.cheegerConstant + 1e-6 && (
                  <span> (suboptimal vs min-cut {minCutResult.cheegerConstant.toFixed(4)})</span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Scatter panel */}
      <svg
        ref={scatterSvgRef}
        width={scatterPanelWidth}
        height={SCATTER_HEIGHT}
        className="rounded-lg border border-[var(--color-border)]"
      />
    </div>
  );
}

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  cycleGraph, completeGraph, petersenGraph,
  wlColorRefinement, wlDistinguishes,
  createRng,
  type Graph, type WLResult,
} from './shared/graphTheory';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRAPH_PANEL_HEIGHT = 260;
const HIST_HEIGHT = 160;
const TRACE_HEIGHT = 130;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 16, right: 16, bottom: 28, left: 36 };

const WL_COLORS = [
  '#14b8a6', '#f59e0b', '#a855f7', '#ef4444', '#3b82f6',
  '#ec4899', '#84cc16', '#06b6d4', '#f97316', '#8b5cf6',
  '#10b981', '#e11d48',
];

// ---------------------------------------------------------------------------
// Graph pair presets
// ---------------------------------------------------------------------------

type PairId = 'c6-vs-2k3' | 'k33-vs-prism' | 'petersen-vs-3reg';

function buildC6(): Graph {
  return cycleGraph(6);
}

function buildTwoK3(): Graph {
  // Two disjoint triangles (6 nodes)
  const n = 6;
  const adj = Array.from({ length: n }, () => new Array(n).fill(0));
  // Triangle 1: 0-1-2
  adj[0][1] = adj[1][0] = 1;
  adj[1][2] = adj[2][1] = 1;
  adj[0][2] = adj[2][0] = 1;
  // Triangle 2: 3-4-5
  adj[3][4] = adj[4][3] = 1;
  adj[4][5] = adj[5][4] = 1;
  adj[3][5] = adj[5][3] = 1;
  return { n, adjacency: adj };
}

function buildK33(): Graph {
  // Complete bipartite K_{3,3} (6 nodes)
  const n = 6;
  const adj = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < 3; i++)
    for (let j = 3; j < 6; j++) {
      adj[i][j] = 1;
      adj[j][i] = 1;
    }
  return { n, adjacency: adj };
}

function buildPrism(): Graph {
  // Prism graph (triangular prism, 6 nodes, 3-regular)
  const n = 6;
  const adj = Array.from({ length: n }, () => new Array(n).fill(0));
  // Top triangle: 0-1-2
  adj[0][1] = adj[1][0] = 1;
  adj[1][2] = adj[2][1] = 1;
  adj[0][2] = adj[2][0] = 1;
  // Bottom triangle: 3-4-5
  adj[3][4] = adj[4][3] = 1;
  adj[4][5] = adj[5][4] = 1;
  adj[3][5] = adj[5][3] = 1;
  // Vertical edges
  adj[0][3] = adj[3][0] = 1;
  adj[1][4] = adj[4][1] = 1;
  adj[2][5] = adj[5][2] = 1;
  return { n, adjacency: adj };
}

function buildRandom3Reg10(): Graph {
  // Deterministic 3-regular graph on 10 nodes (not Petersen)
  const n = 10;
  const adj = Array.from({ length: n }, () => new Array(n).fill(0));
  const edges: [number, number][] = [
    [0,1],[0,4],[0,5],[1,2],[1,6],[2,3],[2,7],[3,4],[3,8],[4,9],[5,7],[5,8],[6,8],[6,9],[7,9],
  ];
  for (const [u, v] of edges) {
    adj[u][v] = 1;
    adj[v][u] = 1;
  }
  return { n, adjacency: adj };
}

interface PairConfig {
  label: string;
  g1Label: string;
  g2Label: string;
  build: () => [Graph, Graph];
  expectDistinguish: boolean;
}

const PAIRS: Record<PairId, PairConfig> = {
  'c6-vs-2k3': {
    label: 'C\u2086 vs Two K\u2083',
    g1Label: 'C\u2086',
    g2Label: '2\u00d7K\u2083',
    build: () => [buildC6(), buildTwoK3()],
    expectDistinguish: true,
  },
  'k33-vs-prism': {
    label: 'K\u2083,\u2083 vs Prism',
    g1Label: 'K\u2083,\u2083',
    g2Label: 'Prism',
    build: () => [buildK33(), buildPrism()],
    expectDistinguish: false,
  },
  'petersen-vs-3reg': {
    label: 'Petersen vs 3-reg(10)',
    g1Label: 'Petersen',
    g2Label: '3-reg(10)',
    build: () => [petersenGraph(), buildRandom3Reg10()],
    expectDistinguish: false,
  },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function WLExpressivenessExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const graphSvg1Ref = useRef<SVGSVGElement>(null);
  const graphSvg2Ref = useRef<SVGSVGElement>(null);
  const histSvgRef = useRef<SVGSVGElement>(null);
  const traceSvgRef = useRef<SVGSVGElement>(null);

  const [pairId, setPairId] = useState<PairId>('c6-vs-2k3');
  const [step, setStep] = useState(0);
  const [autoPlaying, setAutoPlaying] = useState(false);

  const isSmall = containerWidth < SM_BREAKPOINT;

  // --- Compute WL results on disjoint union ---
  const { g1, g2, wlCombined, verdict, maxStep } = useMemo(() => {
    const [graph1, graph2] = PAIRS[pairId].build();
    const n1 = graph1.n;
    const n2 = graph2.n;
    const nTotal = n1 + n2;
    const combinedAdj = Array.from({ length: nTotal }, () => new Array(nTotal).fill(0));
    for (let i = 0; i < n1; i++)
      for (let j = 0; j < n1; j++)
        combinedAdj[i][j] = graph1.adjacency[i][j];
    for (let i = 0; i < n2; i++)
      for (let j = 0; j < n2; j++)
        combinedAdj[n1 + i][n1 + j] = graph2.adjacency[i][j];
    const combined: Graph = { n: nTotal, adjacency: combinedAdj };
    const wl = wlColorRefinement(combined);
    const result = wlDistinguishes(graph1, graph2);
    return {
      g1: graph1,
      g2: graph2,
      wlCombined: wl,
      verdict: result,
      maxStep: wl.colorHistory.length - 1,
    };
  }, [pairId]);

  // Reset step on pair change
  useEffect(() => {
    setStep(0);
    setAutoPlaying(false);
  }, [pairId]);

  // Auto-refine animation
  useEffect(() => {
    if (!autoPlaying) return;
    const interval = setInterval(() => {
      setStep((prev) => {
        if (prev >= maxStep) {
          setAutoPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 600);
    return () => clearInterval(interval);
  }, [autoPlaying, maxStep]);

  // --- Force positions for a graph ---
  const computePositions = useCallback((graph: Graph) => {
    const n = graph.n;
    const nodes = Array.from({ length: n }, (_, i) => ({ id: i, x: 0, y: 0 }));
    const links: { source: number; target: number }[] = [];
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (graph.adjacency[i][j] > 0) links.push({ source: i, target: j });

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(40))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(0, 0))
      .stop();
    for (let i = 0; i < 200; i++) sim.tick();
    return nodes.map((nd) => ({ x: nd.x!, y: nd.y! }));
  }, []);

  const pos1 = useMemo(() => computePositions(g1), [g1, computePositions]);
  const pos2 = useMemo(() => computePositions(g2), [g2, computePositions]);

  // --- Draw graph helper ---
  const drawGraph = useCallback((
    svgRef: React.RefObject<SVGSVGElement | null>,
    graph: Graph,
    positions: { x: number; y: number }[],
    colors: number[],
    title: string
  ) => {
    const svg = d3.select(svgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.24);
    const h = GRAPH_PANEL_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const xs = positions.map((p) => p.x);
    const ys = positions.map((p) => p.y);
    const xScale = d3.scaleLinear().domain(d3.extent(xs) as [number, number]).range([24, w - 24]);
    const yScale = d3.scaleLinear().domain(d3.extent(ys) as [number, number]).range([28, h - 20]);

    const g = svg.append('g');

    for (let i = 0; i < graph.n; i++)
      for (let j = i + 1; j < graph.n; j++)
        if (graph.adjacency[i][j] > 0)
          g.append('line')
            .attr('x1', xScale(positions[i].x)).attr('y1', yScale(positions[i].y))
            .attr('x2', xScale(positions[j].x)).attr('y2', yScale(positions[j].y))
            .style('stroke', '#475569').style('stroke-opacity', 0.5).style('stroke-width', 1.5);

    for (let i = 0; i < graph.n; i++)
      g.append('circle')
        .attr('cx', xScale(positions[i].x))
        .attr('cy', yScale(positions[i].y))
        .attr('r', 8)
        .style('fill', WL_COLORS[colors[i] % WL_COLORS.length])
        .style('stroke', '#1e293b')
        .style('stroke-width', 1.5);

    svg.append('text')
      .attr('x', w / 2).attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8').style('font-size', '12px')
      .text(title);
  }, [containerWidth, isSmall]);

  // --- Draw both graphs ---
  useEffect(() => {
    const colors = wlCombined.colorHistory[Math.min(step, maxStep)];
    const colors1 = colors.slice(0, g1.n);
    const colors2 = colors.slice(g1.n, g1.n + g2.n);
    drawGraph(graphSvg1Ref, g1, pos1, colors1, PAIRS[pairId].g1Label);
    drawGraph(graphSvg2Ref, g2, pos2, colors2, PAIRS[pairId].g2Label);
  }, [g1, g2, pos1, pos2, wlCombined, step, maxStep, pairId, drawGraph]);

  // --- Draw histogram ---
  useEffect(() => {
    const svg = d3.select(histSvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.48);
    const h = HIST_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const colors = wlCombined.colorHistory[Math.min(step, maxStep)];
    const colors1 = colors.slice(0, g1.n);
    const colors2 = colors.slice(g1.n, g1.n + g2.n);

    const hist1 = new Map<number, number>();
    const hist2 = new Map<number, number>();
    for (const c of colors1) hist1.set(c, (hist1.get(c) ?? 0) + 1);
    for (const c of colors2) hist2.set(c, (hist2.get(c) ?? 0) + 1);

    const allColors = [...new Set([...hist1.keys(), ...hist2.keys()])].sort((a, b) => a - b);
    const maxCount = Math.max(
      ...allColors.map((c) => Math.max(hist1.get(c) ?? 0, hist2.get(c) ?? 0)),
      1
    );

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
    const pw = w - MARGIN.left - MARGIN.right;
    const ph = h - MARGIN.top - MARGIN.bottom;

    const xScale = d3.scaleBand<number>()
      .domain(allColors)
      .range([0, pw])
      .padding(0.3);
    const yScale = d3.scaleLinear().domain([0, maxCount]).range([ph, 0]);

    g.append('g').attr('transform', `translate(0,${ph})`)
      .call(d3.axisBottom(xScale).tickFormat((d) => `c${d}`))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.selectAll('.domain, .tick line').style('stroke', '#475569');

    const barW = xScale.bandwidth() / 2.3;
    for (const c of allColors) {
      const xPos = xScale(c)!;
      // Graph 1 bar
      g.append('rect')
        .attr('x', xPos)
        .attr('y', yScale(hist1.get(c) ?? 0))
        .attr('width', barW)
        .attr('height', ph - yScale(hist1.get(c) ?? 0))
        .style('fill', '#3b82f6')
        .style('opacity', 0.8);
      // Graph 2 bar
      g.append('rect')
        .attr('x', xPos + barW + 1)
        .attr('y', yScale(hist2.get(c) ?? 0))
        .attr('width', barW)
        .attr('height', ph - yScale(hist2.get(c) ?? 0))
        .style('fill', '#f59e0b')
        .style('opacity', 0.8);
    }

    // Legend
    g.append('rect').attr('x', pw - 80).attr('y', 0).attr('width', 8).attr('height', 8).style('fill', '#3b82f6');
    g.append('text').attr('x', pw - 68).attr('y', 8).style('fill', '#94a3b8').style('font-size', '9px').text(PAIRS[pairId].g1Label);
    g.append('rect').attr('x', pw - 80).attr('y', 14).attr('width', 8).attr('height', 8).style('fill', '#f59e0b');
    g.append('text').attr('x', pw - 68).attr('y', 22).style('fill', '#94a3b8').style('font-size', '9px').text(PAIRS[pairId].g2Label);

    svg.append('text')
      .attr('x', w / 2).attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8').style('font-size', '11px')
      .text(`Color histogram — step ${step}`);
  }, [g1, g2, wlCombined, step, maxStep, pairId, containerWidth, isSmall]);

  // --- Draw refinement trace ---
  useEffect(() => {
    const svg = d3.select(traceSvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.48);
    const h = TRACE_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
    const pw = w - MARGIN.left - MARGIN.right;
    const ph = h - MARGIN.top - MARGIN.bottom;

    const numColors = wlCombined.numColorsHistory;
    const xScale = d3.scaleLinear().domain([0, numColors.length - 1]).range([0, pw]);
    const yScale = d3.scaleLinear().domain([0, d3.max(numColors)! + 1]).range([ph, 0]);

    g.append('g').attr('transform', `translate(0,${ph})`)
      .call(d3.axisBottom(xScale).ticks(Math.min(numColors.length, 6)).tickFormat((d) => `${d}`))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.selectAll('.domain, .tick line').style('stroke', '#475569');

    const line = d3.line<number>()
      .x((_, i) => xScale(i))
      .y((d) => yScale(d));

    g.append('path')
      .datum(numColors)
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', '#14b8a6')
      .style('stroke-width', 2);

    g.selectAll('.trace-dot')
      .data(numColors)
      .join('circle')
      .attr('cx', (_, i) => xScale(i))
      .attr('cy', (d) => yScale(d))
      .attr('r', 3)
      .style('fill', '#14b8a6');

    // Current step marker
    g.append('line')
      .attr('x1', xScale(step)).attr('x2', xScale(step))
      .attr('y1', 0).attr('y2', ph)
      .style('stroke', '#f59e0b')
      .style('stroke-width', 1.5)
      .style('stroke-dasharray', '4,3');

    svg.append('text')
      .attr('x', w / 2).attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8').style('font-size', '11px')
      .text('Distinct colors per step');

    g.append('text')
      .attr('x', pw / 2).attr('y', ph + 24)
      .attr('text-anchor', 'middle')
      .style('fill', '#64748b').style('font-size', '10px')
      .text('WL iteration');
  }, [wlCombined, step, containerWidth, isSmall]);

  const verdictColor = verdict.distinguishes ? '#4ade80' : '#ef4444';
  const verdictText = verdict.distinguishes
    ? `1-WL distinguishes \u2713 (step ${verdict.step})`
    : `1-WL cannot distinguish \u2717`;

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-slate-700 bg-slate-900 p-4">
      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-1.5 text-sm text-slate-300">
          Graph pair
          <select
            value={pairId}
            onChange={(e) => setPairId(e.target.value as PairId)}
            className="rounded bg-slate-800 px-2 py-1 text-sm text-slate-200"
          >
            {Object.entries(PAIRS).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-1.5 text-sm text-slate-300">
          Step {step}
          <input
            type="range"
            min={0}
            max={maxStep}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            className="w-24 accent-teal-500"
          />
        </label>

        <button
          onClick={() => {
            if (step >= maxStep) setStep(0);
            setAutoPlaying(!autoPlaying);
          }}
          className="rounded bg-slate-700 px-3 py-1 text-sm text-slate-200 hover:bg-slate-600"
        >
          {autoPlaying ? 'Pause' : 'Auto-refine'}
        </button>

        <span className="text-sm font-medium" style={{ color: verdictColor }}>
          {verdictText}
        </span>
      </div>

      {/* Panels */}
      <div className={isSmall ? 'space-y-2' : 'flex gap-2'}>
        <div className={isSmall ? 'flex gap-2' : 'flex flex-col gap-2'}>
          <svg ref={graphSvg1Ref} />
          <svg ref={graphSvg2Ref} />
        </div>
        <div className="flex flex-col gap-2">
          <svg ref={histSvgRef} />
          <svg ref={traceSvgRef} />
        </div>
      </div>
    </div>
  );
}

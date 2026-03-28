import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pathGraph, cycleGraph, barbellGraph, petersenGraph, gridGraph,
  karateClubGraph, laplacian, propagateFeatures, createRng,
  type Graph, type MPNNConfig, type PropagationResult,
} from './shared/graphTheory';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRAPH_PANEL_HEIGHT = 360;
const HEATMAP_HEIGHT = 180;
const ENERGY_HEIGHT = 160;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 16, right: 16, bottom: 28, left: 40 };

const TEAL = '#14b8a6';
const AMBER = '#f59e0b';
const PURPLE = '#a855f7';
const GRAY_EDGE = '#64748b';

// ---------------------------------------------------------------------------
// Graph presets
// ---------------------------------------------------------------------------

type Preset = 'karate' | 'barbell5' | 'grid4' | 'petersen' | 'cycle10' | 'path10';
type Architecture = 'gcn' | 'graphsage' | 'gin';

const PRESETS: Record<Preset, { label: string; build: () => Graph }> = {
  karate: { label: 'Karate Club (34)', build: karateClubGraph },
  barbell5: { label: 'Barbell(5)', build: () => barbellGraph(5) },
  grid4: { label: 'Grid 4\u00d74', build: () => gridGraph(4) },
  petersen: { label: 'Petersen', build: petersenGraph },
  cycle10: { label: 'Cycle(10)', build: () => cycleGraph(10) },
  path10: { label: 'Path(10)', build: () => pathGraph(10) },
};

const ARCH_LABELS: Record<Architecture, string> = {
  gcn: 'GCN',
  graphsage: 'GraphSAGE',
  gin: 'GIN',
};

const MAX_LAYERS = 20;
const FEATURE_DIM = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateInitialFeatures(n: number): number[][] {
  const rng = createRng(42);
  return Array.from({ length: n }, () =>
    Array.from({ length: FEATURE_DIM }, () => rng() * 2 - 1)
  );
}

/** Project nd features to 1D via first principal direction (simplified). */
function projectTo1D(H: number[][]): number[] {
  const n = H.length;
  const d = H[0].length;
  // Use first column as a simple projection
  // (full PCA is overkill for viz; column 0 captures sufficient variance)
  const col = H.map((row) => row[0]);
  const minV = Math.min(...col);
  const maxV = Math.max(...col);
  const range = maxV - minV || 1;
  return col.map((v) => (v - minV) / range);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MessagePassingExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const graphSvgRef = useRef<SVGSVGElement>(null);
  const heatmapSvgRef = useRef<SVGSVGElement>(null);
  const energySvgRef = useRef<SVGSVGElement>(null);

  const [preset, setPreset] = useState<Preset>('barbell5');
  const [architecture, setArchitecture] = useState<Architecture>('gcn');
  const [layer, setLayer] = useState(0);
  const [playing, setPlaying] = useState(false);

  const isSmall = containerWidth < SM_BREAKPOINT;

  // --- Compute graph + propagation ---
  const { graph, result } = useMemo(() => {
    const g = PRESETS[preset].build();
    const H0 = generateInitialFeatures(g.n);
    const config: MPNNConfig = { architecture, layers: MAX_LAYERS };
    const r = propagateFeatures(g, H0, config);
    return { graph: g, result: r };
  }, [preset, architecture]);

  // Reset layer on preset/arch change
  useEffect(() => {
    setLayer(0);
    setPlaying(false);
  }, [preset, architecture]);

  // Animation
  useEffect(() => {
    if (!playing) return;
    const interval = setInterval(() => {
      setLayer((prev) => {
        if (prev >= MAX_LAYERS) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 400);
    return () => clearInterval(interval);
  }, [playing]);

  // --- Node color scale ---
  const nodeColors = useMemo(() => {
    const H = result.features[layer];
    const proj = projectTo1D(H);
    const colorScale = d3.scaleDiverging(d3.interpolateRdBu).domain([0, 0.5, 1]);
    return proj.map((v) => colorScale(v));
  }, [result, layer]);

  // --- Force simulation positions ---
  const positions = useMemo(() => {
    const n = graph.n;
    const nodes = Array.from({ length: n }, (_, i) => ({ id: i, x: 0, y: 0 }));
    const links: { source: number; target: number }[] = [];
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (graph.adjacency[i][j] > 0) links.push({ source: i, target: j });

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(30))
      .force('charge', d3.forceManyBody().strength(-60))
      .force('center', d3.forceCenter(0, 0))
      .stop();

    for (let i = 0; i < 300; i++) sim.tick();
    return nodes.map((nd) => ({ x: nd.x!, y: nd.y! }));
  }, [graph]);

  // --- Draw graph ---
  useEffect(() => {
    const svg = d3.select(graphSvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.45);
    const h = GRAPH_PANEL_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const g = svg.append('g');
    // Fit positions into viewport
    const xs = positions.map((p) => p.x);
    const ys = positions.map((p) => p.y);
    const xExt = d3.extent(xs) as [number, number];
    const yExt = d3.extent(ys) as [number, number];
    const xScale = d3.scaleLinear().domain(xExt).range([30, w - 30]);
    const yScale = d3.scaleLinear().domain(yExt).range([30, h - 30]);

    // Edges
    for (let i = 0; i < graph.n; i++) {
      for (let j = i + 1; j < graph.n; j++) {
        if (graph.adjacency[i][j] > 0) {
          g.append('line')
            .attr('x1', xScale(positions[i].x))
            .attr('y1', yScale(positions[i].y))
            .attr('x2', xScale(positions[j].x))
            .attr('y2', yScale(positions[j].y))
            .style('stroke', GRAY_EDGE)
            .style('stroke-opacity', 0.3)
            .style('stroke-width', 1);
        }
      }
    }

    // Nodes
    const nodeRadius = graph.n > 20 ? 5 : 7;
    for (let i = 0; i < graph.n; i++) {
      g.append('circle')
        .attr('cx', xScale(positions[i].x))
        .attr('cy', yScale(positions[i].y))
        .attr('r', nodeRadius)
        .style('fill', nodeColors[i])
        .style('stroke', '#1e293b')
        .style('stroke-width', 1);
    }

    // Title
    svg.append('text')
      .attr('x', w / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8')
      .style('font-size', '12px')
      .text(`Layer ${layer} — ${ARCH_LABELS[architecture]}`);
  }, [graph, positions, nodeColors, layer, architecture, containerWidth, isSmall]);

  // --- Draw heatmap ---
  useEffect(() => {
    const svg = d3.select(heatmapSvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.5);
    const h = HEATMAP_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const H = result.features[layer];
    const n = H.length;
    const d = H[0].length;

    const allVals = H.flat();
    const ext = d3.extent(allVals) as [number, number];
    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain(ext);

    const cellW = Math.min(30, (w - MARGIN.left - MARGIN.right) / d);
    const cellH = Math.min(8, (h - MARGIN.top - MARGIN.bottom) / n);

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    for (let i = 0; i < n; i++) {
      for (let k = 0; k < d; k++) {
        g.append('rect')
          .attr('x', k * cellW)
          .attr('y', i * cellH)
          .attr('width', cellW - 0.5)
          .attr('height', cellH - 0.5)
          .style('fill', colorScale(H[i][k]));
      }
    }

    svg.append('text')
      .attr('x', w / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8')
      .style('font-size', '11px')
      .text(`Feature heatmap H(${layer}) — ${n} nodes × ${d} dims`);
  }, [result, layer, containerWidth, isSmall]);

  // --- Draw energy trace ---
  useEffect(() => {
    const svg = d3.select(energySvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.5);
    const h = ENERGY_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const energyData = result.dirichletEnergy;
    const E0 = energyData[0] || 1;
    const normalized = energyData.map((e) => e / E0);

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
    const pw = w - MARGIN.left - MARGIN.right;
    const ph = h - MARGIN.top - MARGIN.bottom;

    const xScale = d3.scaleLinear().domain([0, MAX_LAYERS]).range([0, pw]);
    const yScale = d3.scaleLog().domain([Math.max(1e-6, d3.min(normalized)!), 1]).range([ph, 0]).clamp(true);

    // Axes
    g.append('g').attr('transform', `translate(0,${ph})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat((d) => `${d}`))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4, '.0e'))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '10px');
    g.selectAll('.domain, .tick line').style('stroke', '#475569');

    // Energy line
    const line = d3.line<number>()
      .x((_, i) => xScale(i))
      .y((d) => yScale(Math.max(1e-6, d)));

    g.append('path')
      .datum(normalized)
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', TEAL)
      .style('stroke-width', 2);

    // Dots
    g.selectAll('.energy-dot')
      .data(normalized)
      .join('circle')
      .attr('cx', (_, i) => xScale(i))
      .attr('cy', (d) => yScale(Math.max(1e-6, d)))
      .attr('r', 2.5)
      .style('fill', TEAL);

    // Current layer marker
    g.append('line')
      .attr('x1', xScale(layer))
      .attr('x2', xScale(layer))
      .attr('y1', 0)
      .attr('y2', ph)
      .style('stroke', AMBER)
      .style('stroke-width', 1.5)
      .style('stroke-dasharray', '4,3');

    // Over-smoothing depth marker
    if (result.overSmoothingDepth < MAX_LAYERS) {
      g.append('line')
        .attr('x1', xScale(result.overSmoothingDepth))
        .attr('x2', xScale(result.overSmoothingDepth))
        .attr('y1', 0)
        .attr('y2', ph)
        .style('stroke', '#ef4444')
        .style('stroke-width', 1)
        .style('stroke-dasharray', '2,2');

      g.append('text')
        .attr('x', xScale(result.overSmoothingDepth) + 4)
        .attr('y', 10)
        .style('fill', '#ef4444')
        .style('font-size', '9px')
        .text(`1% @ L=${result.overSmoothingDepth}`);
    }

    // Labels
    svg.append('text')
      .attr('x', w / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8')
      .style('font-size', '11px')
      .text('Dirichlet energy E(H) / E(H₀) — log scale');

    g.append('text')
      .attr('x', pw / 2)
      .attr('y', ph + 24)
      .attr('text-anchor', 'middle')
      .style('fill', '#64748b')
      .style('font-size', '10px')
      .text('Layer ℓ');
  }, [result, layer, containerWidth, isSmall]);

  // --- Info bar ---
  const spectralGapText = result.spectralGap.toFixed(4);
  const depthText = result.overSmoothingDepth < MAX_LAYERS
    ? `${result.overSmoothingDepth}`
    : `>${MAX_LAYERS}`;

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-slate-700 bg-slate-900 p-4">
      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-1.5 text-sm text-slate-300">
          Graph
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value as Preset)}
            className="rounded bg-slate-800 px-2 py-1 text-sm text-slate-200"
          >
            {Object.entries(PRESETS).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-1.5 text-sm text-slate-300">
          Architecture
          <select
            value={architecture}
            onChange={(e) => setArchitecture(e.target.value as Architecture)}
            className="rounded bg-slate-800 px-2 py-1 text-sm text-slate-200"
          >
            {Object.entries(ARCH_LABELS).map(([k, v]) => (
              <option key={k} value={k}>{v}</option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-1.5 text-sm text-slate-300">
          Layer {layer}
          <input
            type="range"
            min={0}
            max={MAX_LAYERS}
            value={layer}
            onChange={(e) => setLayer(Number(e.target.value))}
            className="w-28 accent-teal-500"
          />
        </label>

        <button
          onClick={() => {
            if (layer >= MAX_LAYERS) setLayer(0);
            setPlaying(!playing);
          }}
          className="rounded bg-slate-700 px-3 py-1 text-sm text-slate-200 hover:bg-slate-600"
        >
          {playing ? 'Pause' : 'Animate'}
        </button>
      </div>

      {/* Info bar */}
      <div className="mb-3 flex flex-wrap gap-4 text-xs text-slate-400">
        <span>Spectral gap γ = <span className="text-teal-400">{spectralGapText}</span></span>
        <span>Over-smoothing depth: <span className="text-amber-400">{depthText}</span></span>
        <span>Nodes: {graph.n}</span>
      </div>

      {/* Panels */}
      <div className={isSmall ? 'space-y-2' : 'flex gap-2'}>
        <div className={isSmall ? '' : 'shrink-0'}>
          <svg ref={graphSvgRef} />
        </div>
        <div className="flex flex-col gap-2">
          <svg ref={heatmapSvgRef} />
          <svg ref={energySvgRef} />
        </div>
      </div>
    </div>
  );
}

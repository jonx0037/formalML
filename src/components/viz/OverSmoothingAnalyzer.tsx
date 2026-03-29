import { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pathGraph, cycleGraph, completeGraph, barbellGraph, petersenGraph,
  gridGraph, karateClubGraph, propagateFeatures, fosrRewire, createRng,
  type Graph, type MPNNConfig, type PropagationResult,
} from './shared/graphTheory';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRAPH_PANEL_HEIGHT = 300;
const ENERGY_HEIGHT = 200;
const SCATTER_HEIGHT = 160;
const REWIRE_HEIGHT = 160;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 16, right: 16, bottom: 28, left: 44 };
const MAX_LAYERS = 30;
const FEATURE_DIM = 4;

const FAMILY_COLORS: Record<string, string> = {
  path10: '#3b82f6',
  cycle10: '#14b8a6',
  grid3: '#f59e0b',
  barbell5: '#ef4444',
  petersen: '#a855f7',
  complete8: '#ec4899',
  karate: '#84cc16',
};

// ---------------------------------------------------------------------------
// Graph family presets
// ---------------------------------------------------------------------------

type FamilyId = 'path10' | 'cycle10' | 'grid3' | 'barbell5' | 'petersen' | 'complete8' | 'karate';

const FAMILIES: Record<FamilyId, { label: string; build: () => Graph }> = {
  path10: { label: 'Path(10)', build: () => pathGraph(10) },
  cycle10: { label: 'Cycle(10)', build: () => cycleGraph(10) },
  grid3: { label: 'Grid 3\u00d73', build: () => gridGraph(3) },
  barbell5: { label: 'Barbell(5)', build: () => barbellGraph(5) },
  petersen: { label: 'Petersen', build: petersenGraph },
  complete8: { label: 'Complete(8)', build: () => completeGraph(8) },
  karate: { label: 'Karate Club', build: karateClubGraph },
};

const ALL_FAMILY_IDS: FamilyId[] = Object.keys(FAMILIES) as FamilyId[];
const DEFAULT_SELECTED: FamilyId[] = ['path10', 'barbell5', 'petersen', 'complete8'];

function generateH0(n: number): number[][] {
  const rng = createRng(42);
  return Array.from({ length: n }, () =>
    Array.from({ length: FEATURE_DIM }, () => rng() * 2 - 1)
  );
}

function projectTo1D(H: number[][]): number[] {
  const col = H.map((row) => row[0]);
  const minV = Math.min(...col);
  const maxV = Math.max(...col);
  const range = maxV - minV || 1;
  return col.map((v) => (v - minV) / range);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function OverSmoothingAnalyzer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const graphSvgRef = useRef<SVGSVGElement>(null);
  const energySvgRef = useRef<SVGSVGElement>(null);
  const scatterSvgRef = useRef<SVGSVGElement>(null);
  const rewireSvgRef = useRef<SVGSVGElement>(null);

  const [selectedFamilies, setSelectedFamilies] = useState<Set<FamilyId>>(new Set(DEFAULT_SELECTED));
  const [activeGraph, setActiveGraph] = useState<FamilyId>('barbell5');
  const [layer, setLayer] = useState(0);
  const [showRewiring, setShowRewiring] = useState(false);
  const [rewiringEdges, setRewiringEdges] = useState(5);

  const isSmall = containerWidth < SM_BREAKPOINT;

  // --- Precompute propagation for all families ---
  const allResults = useMemo(() => {
    const results: Record<string, { graph: Graph; result: PropagationResult }> = {};
    const config: MPNNConfig = { architecture: 'gcn', layers: MAX_LAYERS };
    for (const id of ALL_FAMILY_IDS) {
      const g = FAMILIES[id].build();
      const H0 = generateH0(g.n);
      results[id] = { graph: g, result: propagateFeatures(g, H0, config) };
    }
    return results;
  }, []);

  // --- Rewiring result ---
  const rewireResult = useMemo(() => {
    if (!showRewiring) return null;
    const { graph } = allResults[activeGraph];
    const rr = fosrRewire(graph, rewiringEdges);
    const H0 = generateH0(rr.graph.n);
    const config: MPNNConfig = { architecture: 'gcn', layers: MAX_LAYERS };
    return {
      rewiredGraph: rr.graph,
      gapHistory: rr.gapHistory,
      addedEdges: rr.addedEdges,
      propagation: propagateFeatures(rr.graph, H0, config),
    };
  }, [showRewiring, activeGraph, rewiringEdges, allResults]);

  // --- Force positions for active graph ---
  const positions = useMemo(() => {
    const graph = allResults[activeGraph].graph;
    const n = graph.n;
    const nodes = Array.from({ length: n }, (_, i) => ({ id: i, x: 0, y: 0 }));
    const links: { source: number; target: number }[] = [];
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (graph.adjacency[i][j] > 0) links.push({ source: i, target: j });

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink<typeof nodes[number], typeof links[number]>(links).distance(30))
      .force('charge', d3.forceManyBody().strength(-60))
      .force('center', d3.forceCenter(0, 0))
      .stop();
    for (let i = 0; i < 300; i++) sim.tick();
    return nodes.map((nd) => ({ x: nd.x!, y: nd.y! }));
  }, [activeGraph, allResults]);

  // --- Draw active graph ---
  useEffect(() => {
    const svg = d3.select(graphSvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.4);
    const h = GRAPH_PANEL_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const graph = allResults[activeGraph].graph;
    const result = allResults[activeGraph].result;
    const H = result.features[Math.min(layer, MAX_LAYERS)];
    const proj = projectTo1D(H);
    const colorScale = d3.scaleDiverging(d3.interpolateRdBu).domain([0, 0.5, 1]);

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
            .style('stroke', '#475569').style('stroke-opacity', 0.3).style('stroke-width', 1);

    const nodeRadius = graph.n > 20 ? 5 : 7;
    for (let i = 0; i < graph.n; i++)
      g.append('circle')
        .attr('cx', xScale(positions[i].x))
        .attr('cy', yScale(positions[i].y))
        .attr('r', nodeRadius)
        .style('fill', colorScale(proj[i]))
        .style('stroke', '#1e293b')
        .style('stroke-width', 1);

    svg.append('text')
      .attr('x', w / 2).attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8').style('font-size', '12px')
      .text(`${FAMILIES[activeGraph].label} — layer ${layer}`);
  }, [activeGraph, allResults, positions, layer, containerWidth, isSmall]);

  // --- Draw energy decay curves ---
  useEffect(() => {
    const svg = d3.select(energySvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.56);
    const h = ENERGY_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
    const pw = w - MARGIN.left - MARGIN.right;
    const ph = h - MARGIN.top - MARGIN.bottom;

    const xScale = d3.scaleLinear().domain([0, MAX_LAYERS]).range([0, pw]);

    // Collect all normalized energies for y-domain
    let minE = 1;
    const selected = [...selectedFamilies];
    for (const id of selected) {
      const E = allResults[id].result.dirichletEnergy;
      const E0 = E[0] || 1;
      for (const e of E) {
        const ne = e / E0;
        if (ne > 0 && ne < minE) minE = ne;
      }
    }

    const yScale = d3.scaleLog()
      .domain([Math.max(1e-8, minE * 0.5), 1])
      .range([ph, 0])
      .clamp(true);

    g.append('g').attr('transform', `translate(0,${ph})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4, '.0e'))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.selectAll('.domain, .tick line').style('stroke', '#475569');

    // Draw curves for selected families
    for (const id of selected) {
      const E = allResults[id].result.dirichletEnergy;
      const E0 = E[0] || 1;
      const normalized = E.map((e) => e / E0);

      const line = d3.line<number>()
        .x((_, i) => xScale(i))
        .y((d) => yScale(Math.max(1e-8, d)));

      g.append('path')
        .datum(normalized)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', FAMILY_COLORS[id])
        .style('stroke-width', 2);
    }

    // Current layer marker
    g.append('line')
      .attr('x1', xScale(layer)).attr('x2', xScale(layer))
      .attr('y1', 0).attr('y2', ph)
      .style('stroke', '#f59e0b').style('stroke-width', 1).style('stroke-dasharray', '4,3');

    // Legend
    const legendG = g.append('g').attr('transform', `translate(${pw - 100}, 4)`);
    selected.forEach((id, idx) => {
      legendG.append('line')
        .attr('x1', 0).attr('y1', idx * 14 + 4).attr('x2', 14).attr('y2', idx * 14 + 4)
        .style('stroke', FAMILY_COLORS[id]).style('stroke-width', 2);
      legendG.append('text')
        .attr('x', 18).attr('y', idx * 14 + 7)
        .style('fill', '#94a3b8').style('font-size', '9px')
        .text(FAMILIES[id].label);
    });

    svg.append('text')
      .attr('x', w / 2).attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8').style('font-size', '11px')
      .text('Dirichlet energy E/E\u2080 — log scale');

    g.append('text')
      .attr('x', pw / 2).attr('y', ph + 24)
      .attr('text-anchor', 'middle')
      .style('fill', '#64748b').style('font-size', '10px')
      .text('Layer \u2113');
  }, [selectedFamilies, allResults, layer, containerWidth, isSmall]);

  // --- Draw scatter plot: spectral gap vs over-smoothing depth ---
  useEffect(() => {
    const svg = d3.select(scatterSvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.56);
    const h = SCATTER_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
    const pw = w - MARGIN.left - MARGIN.right;
    const ph = h - MARGIN.top - MARGIN.bottom;

    const data = ALL_FAMILY_IDS.map((id) => ({
      id,
      gap: allResults[id].result.spectralGap,
      depth: allResults[id].result.overSmoothingDepth,
    }));

    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.gap)! * 1.1])
      .range([0, pw]);
    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, (d) => d.depth)! * 1.1])
      .range([ph, 0]);

    g.append('g').attr('transform', `translate(0,${ph})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.selectAll('.domain, .tick line').style('stroke', '#475569');

    for (const d of data) {
      g.append('circle')
        .attr('cx', xScale(d.gap))
        .attr('cy', yScale(d.depth))
        .attr('r', 6)
        .style('fill', FAMILY_COLORS[d.id])
        .style('stroke', selectedFamilies.has(d.id) ? '#fff' : '#475569')
        .style('stroke-width', selectedFamilies.has(d.id) ? 2 : 1)
        .style('cursor', 'pointer')
        .on('click', () => setActiveGraph(d.id));

      g.append('text')
        .attr('x', xScale(d.gap) + 8)
        .attr('y', yScale(d.depth) + 3)
        .style('fill', '#94a3b8')
        .style('font-size', '8px')
        .text(FAMILIES[d.id].label);
    }

    svg.append('text')
      .attr('x', w / 2).attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8').style('font-size', '11px')
      .text('Spectral gap \u03b3 vs over-smoothing depth');

    g.append('text')
      .attr('x', pw / 2).attr('y', ph + 24)
      .attr('text-anchor', 'middle')
      .style('fill', '#64748b').style('font-size', '10px')
      .text('Spectral gap \u03b3');
  }, [allResults, selectedFamilies, containerWidth, isSmall]);

  // --- Draw rewiring comparison ---
  useEffect(() => {
    const svg = d3.select(rewireSvgRef.current);
    if (!svg.node()) return;
    const w = isSmall ? (containerWidth || 300) : ((containerWidth || 600) * 0.56);
    const h = REWIRE_HEIGHT;
    svg.attr('width', w).attr('height', h);
    svg.selectAll('*').remove();

    if (!showRewiring || !rewireResult) {
      svg.append('text')
        .attr('x', w / 2).attr('y', h / 2)
        .attr('text-anchor', 'middle')
        .style('fill', '#475569').style('font-size', '12px')
        .text('Enable rewiring to compare');
      return;
    }

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
    const pw = w - MARGIN.left - MARGIN.right;
    const ph = h - MARGIN.top - MARGIN.bottom;

    const origE = allResults[activeGraph].result.dirichletEnergy;
    const rewiredE = rewireResult.propagation.dirichletEnergy;
    const origE0 = origE[0] || 1;
    const rewiredE0 = rewiredE[0] || 1;
    const origN = origE.map((e) => e / origE0);
    const rewiredN = rewiredE.map((e) => e / rewiredE0);

    const allVals = [...origN, ...rewiredN].filter((v) => v > 0);
    const minVal = Math.max(1e-8, Math.min(...allVals) * 0.5);

    const xScale = d3.scaleLinear().domain([0, MAX_LAYERS]).range([0, pw]);
    const yScale = d3.scaleLog().domain([minVal, 1]).range([ph, 0]).clamp(true);

    g.append('g').attr('transform', `translate(0,${ph})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(3, '.0e'))
      .selectAll('text').style('fill', '#94a3b8').style('font-size', '9px');
    g.selectAll('.domain, .tick line').style('stroke', '#475569');

    const line = d3.line<number>()
      .x((_, i) => xScale(i))
      .y((d) => yScale(Math.max(1e-8, d)));

    // Original (solid)
    g.append('path')
      .datum(origN)
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', FAMILY_COLORS[activeGraph])
      .style('stroke-width', 2);

    // Rewired (dashed)
    g.append('path')
      .datum(rewiredN)
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', FAMILY_COLORS[activeGraph])
      .style('stroke-width', 2)
      .style('stroke-dasharray', '6,3');

    // Legend
    g.append('line').attr('x1', pw - 110).attr('y1', 6).attr('x2', pw - 90).attr('y2', 6)
      .style('stroke', FAMILY_COLORS[activeGraph]).style('stroke-width', 2);
    g.append('text').attr('x', pw - 86).attr('y', 9)
      .style('fill', '#94a3b8').style('font-size', '9px').text('Original');

    g.append('line').attr('x1', pw - 110).attr('y1', 20).attr('x2', pw - 90).attr('y2', 20)
      .style('stroke', FAMILY_COLORS[activeGraph]).style('stroke-width', 2).style('stroke-dasharray', '6,3');
    g.append('text').attr('x', pw - 86).attr('y', 23)
      .style('fill', '#94a3b8').style('font-size', '9px').text(`+${rewiringEdges} FoSR`);

    svg.append('text')
      .attr('x', w / 2).attr('y', 12)
      .attr('text-anchor', 'middle')
      .style('fill', '#94a3b8').style('font-size', '11px')
      .text('Rewiring comparison — original vs FoSR');
  }, [showRewiring, rewireResult, allResults, activeGraph, rewiringEdges, containerWidth, isSmall]);

  const toggleFamily = (id: FamilyId) => {
    setSelectedFamilies((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-slate-700 bg-slate-900 p-4">
      {/* Controls */}
      <div className="mb-3 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-slate-400">Families:</span>
          {ALL_FAMILY_IDS.map((id) => (
            <label key={id} className="flex items-center gap-1 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={selectedFamilies.has(id)}
                onChange={() => toggleFamily(id)}
                className="accent-teal-500"
              />
              <span style={{ color: FAMILY_COLORS[id] }}>{FAMILIES[id].label}</span>
            </label>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1.5 text-sm text-slate-300">
            Active graph
            <select
              value={activeGraph}
              onChange={(e) => setActiveGraph(e.target.value as FamilyId)}
              className="rounded bg-slate-800 px-2 py-1 text-sm text-slate-200"
            >
              {ALL_FAMILY_IDS.map((id) => (
                <option key={id} value={id}>{FAMILIES[id].label}</option>
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

          <label className="flex items-center gap-1.5 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={showRewiring}
              onChange={() => setShowRewiring(!showRewiring)}
              className="accent-teal-500"
            />
            Show FoSR rewiring
          </label>

          {showRewiring && (
            <label className="flex items-center gap-1.5 text-sm text-slate-300">
              Edges: {rewiringEdges}
              <input
                type="range"
                min={2}
                max={10}
                value={rewiringEdges}
                onChange={(e) => setRewiringEdges(Number(e.target.value))}
                className="w-20 accent-teal-500"
              />
            </label>
          )}
        </div>
      </div>

      {/* Panels */}
      <div className={isSmall ? 'space-y-2' : 'flex gap-2'}>
        <div className="shrink-0">
          <svg role="img" aria-label="Over smoothing analyzer visualization (panel 1 of 4)" ref={graphSvgRef} />
        </div>
        <div className="flex flex-col gap-2">
          <svg role="img" aria-label="Over smoothing analyzer visualization (panel 2 of 4)" ref={energySvgRef} />
          <svg role="img" aria-label="Over smoothing analyzer visualization (panel 3 of 4)" ref={scatterSvgRef} />
          <svg role="img" aria-label="Over smoothing analyzer visualization (panel 4 of 4)" ref={rewireSvgRef} />
        </div>
      </div>
    </div>
  );
}

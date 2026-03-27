import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pathGraph, cycleGraph, completeGraph, barbellGraph, gridGraph, hypercubeGraph,
  mixingProfile, analyzeTransitionMatrix, degrees,
  jacobiEigen,
  type Graph, type MixingProfile, type TransitionResult,
} from './shared/graphTheory';

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const GRAPH_PANEL_HEIGHT = 340;
const TV_CHART_HEIGHT = 280;
const BAR_CHART_HEIGHT = 200;
const MARGIN = { top: 32, right: 20, bottom: 40, left: 52 };
const MAX_TIME = 200;
const EPSILON = 0.25;

// ─── Graph families ───

interface FamilyDef {
  key: string;
  label: string;
  build: (n: number) => Graph;
}

const FAMILIES: FamilyDef[] = [
  { key: 'path', label: 'Path', build: (n) => pathGraph(n) },
  { key: 'cycle', label: 'Cycle', build: (n) => cycleGraph(n) },
  { key: 'complete', label: 'Complete', build: (n) => completeGraph(n) },
  { key: 'barbell', label: 'Barbell', build: (n) => barbellGraph(Math.max(2, Math.floor(n / 2))) },
  { key: 'grid', label: 'Grid', build: (n) => gridGraph(Math.max(2, Math.round(Math.sqrt(n)))) },
  { key: 'hypercube', label: 'Hypercube', build: (n) => hypercubeGraph(Math.min(Math.max(2, Math.floor(Math.log2(n))), 4)) },
];

const FAMILY_COLORS = d3.schemeTableau10;

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

/** Compute the walk distribution P^t(x0, ·) via matrix powering. */
function computeDistributionAtTime(
  graph: Graph,
  x0: number,
  t: number,
): number[] {
  const A = graph.adjacency;
  const n = graph.n;
  const deg = degrees(A);

  // Build symmetric matrix S = D^{-1/2} A D^{-1/2}
  const S: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (A[i][j] !== 0 && deg[i] > 0 && deg[j] > 0) {
        S[i][j] = A[i][j] / Math.sqrt(deg[i] * deg[j]);
      }
    }
  }

  const eigen = jacobiEigen(S);
  const mu = eigen.eigenvalues;
  const V = eigen.eigenvectors;

  // phi[k][x] = D^{-1/2}_x * v_k(x), psi[k][y] = D^{1/2}_y * v_k(y)
  const phi: number[][] = [];
  const psi: number[][] = [];
  for (let k = 0; k < n; k++) {
    const pk: number[] = new Array(n);
    const sk: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const sqrtD = Math.sqrt(deg[i] || 1);
      pk[i] = V[k][i] / sqrtD;
      sk[i] = V[k][i] * sqrtD;
    }
    phi.push(pk);
    psi.push(sk);
  }

  // P^t(x0, y) = Σ_k μ_k^t * φ_k(x0) * ψ_k(y)
  const dist: number[] = new Array(n).fill(0);
  for (let y = 0; y < n; y++) {
    let val = 0;
    for (let k = 0; k < n; k++) {
      val += Math.pow(mu[k], t) * phi[k][x0] * psi[k][y];
    }
    dist[y] = Math.max(0, val); // clamp small negatives from numerical error
  }

  return dist;
}

// ─── Cached computation result ───

interface FamilyResult {
  key: string;
  graph: Graph;
  profile: MixingProfile;
  transition: TransitionResult;
}

// ─── Component ───

export default function MixingTimeExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const graphSvgRef = useRef<SVGSVGElement>(null);
  const tvSvgRef = useRef<SVGSVGElement>(null);
  const barSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───

  const [selectedFamilies, setSelectedFamilies] = useState<Set<string>>(
    () => new Set(['path', 'cycle', 'complete']),
  );
  const [sizeParam, setSizeParam] = useState(8);
  const [timeStep, setTimeStep] = useState(0);
  const [highlightedFamily, setHighlightedFamily] = useState<string | null>('path');

  // Force simulation refs
  const positionsRef = useRef<{ x: number; y: number }[]>([]);
  const simRef = useRef<d3.Simulation<{ x: number; y: number }, undefined> | null>(null);
  const [tick, setTick] = useState(0);

  // ─── Layout calculations ───

  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const leftPanelWidth = useMemo(() => {
    if (!containerWidth) return 360;
    if (isNarrow) return containerWidth - 16;
    return Math.floor((containerWidth - 32) * 0.4);
  }, [containerWidth, isNarrow]);

  const rightPanelWidth = useMemo(() => {
    if (!containerWidth) return 440;
    if (isNarrow) return containerWidth - 16;
    return Math.floor((containerWidth - 32) * 0.6);
  }, [containerWidth, isNarrow]);

  // ─── Derived computations ───

  const familyResults: Map<string, FamilyResult> = useMemo(() => {
    const results = new Map<string, FamilyResult>();
    for (const fam of FAMILIES) {
      if (!selectedFamilies.has(fam.key)) continue;
      const graph = fam.build(sizeParam);
      const profile = mixingProfile(graph, MAX_TIME, EPSILON);
      const transition = analyzeTransitionMatrix(graph);
      results.set(fam.key, { key: fam.key, graph, profile, transition });
    }
    return results;
  }, [selectedFamilies, sizeParam]);

  // The highlighted family's graph and distribution at time t
  const highlightedResult = highlightedFamily
    ? familyResults.get(highlightedFamily) ?? null
    : null;

  const highlightedDistribution = useMemo(() => {
    if (!highlightedResult) return null;
    const { graph, profile } = highlightedResult;
    return computeDistributionAtTime(graph, profile.worstStartVertex, timeStep);
  }, [highlightedResult, timeStep]);

  // Family index for stable color mapping
  const familyColorIndex = useMemo(() => {
    const map = new Map<string, number>();
    FAMILIES.forEach((f, i) => map.set(f.key, i));
    return map;
  }, []);

  // ─── Interaction handlers ───

  const handleFamilyToggle = useCallback((key: string) => {
    setSelectedFamilies((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        if (next.size <= 1) return prev; // keep at least one
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  const handleFamilyHighlight = useCallback((key: string) => {
    setHighlightedFamily((prev) => (prev === key ? prev : key));
  }, []);

  // ─── Force simulation management ───

  const initSimulation = useCallback(
    (graph: Graph, panelWidth: number) => {
      if (simRef.current) simRef.current.stop();

      const n = graph.n;
      const nodes: { x: number; y: number }[] = Array.from(
        { length: n },
        (_, i) => ({
          x: panelWidth / 2 + Math.cos((2 * Math.PI * i) / n) * 80,
          y: GRAPH_PANEL_HEIGHT / 2 + Math.sin((2 * Math.PI * i) / n) * 80,
        }),
      );
      positionsRef.current = nodes;

      const links = edgesFromAdj(graph.adjacency);

      const sim = d3
        .forceSimulation(nodes)
        .force(
          'link',
          d3
            .forceLink<{ x: number; y: number }, { source: number; target: number }>(links)
            .id((_, i) => i)
            .distance(50),
        )
        .force('charge', d3.forceManyBody().strength(-160))
        .force('center', d3.forceCenter(panelWidth / 2, GRAPH_PANEL_HEIGHT / 2))
        .force('collide', d3.forceCollide(16))
        .alphaDecay(0.03)
        .on('tick', () => {
          const pad = 20;
          for (const node of nodes) {
            node.x = Math.max(pad, Math.min(panelWidth - pad, node.x));
            node.y = Math.max(pad, Math.min(GRAPH_PANEL_HEIGHT - pad, node.y));
          }
          setTick((t) => t + 1);
        });

      simRef.current = sim;
    },
    [],
  );

  // Re-init simulation when highlighted graph changes
  const prevHighlightKey = useRef<string | null>(null);
  const prevSize = useRef(sizeParam);

  useEffect(() => {
    const needsReinit =
      prevHighlightKey.current !== highlightedFamily ||
      prevSize.current !== sizeParam;

    if (needsReinit && highlightedResult) {
      initSimulation(highlightedResult.graph, leftPanelWidth);
      prevHighlightKey.current = highlightedFamily;
      prevSize.current = sizeParam;
    }

    return () => {
      if (simRef.current) simRef.current.stop();
    };
  }, [highlightedResult, highlightedFamily, sizeParam, leftPanelWidth, initSimulation]);

  // Initial mount
  useEffect(() => {
    if (highlightedResult) {
      initSimulation(highlightedResult.graph, leftPanelWidth);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── Graph panel rendering (left) ───

  useEffect(() => {
    if (!graphSvgRef.current || leftPanelWidth === 0) return;

    const svg = d3.select(graphSvgRef.current);
    svg.selectAll('*').remove();

    if (!highlightedResult || !highlightedDistribution) {
      svg
        .append('text')
        .attr('x', leftPanelWidth / 2)
        .attr('y', GRAPH_PANEL_HEIGHT / 2)
        .attr('text-anchor', 'middle')
        .attr('font-size', 13)
        .style('fill', 'var(--color-text-secondary)')
        .text('Select a family to view');
      return;
    }

    const { graph, profile } = highlightedResult;
    const dist = highlightedDistribution;
    const n = graph.n;
    const positions = positionsRef.current;
    if (positions.length !== n) return;

    const edges = edgesFromAdj(graph.adjacency);

    // Color scale: light → dark teal based on distribution
    const maxProb = d3.max(dist) || 1;
    // Light → dark teal color scale for distribution intensity
    const colorScale = d3
      .scaleSequential((t: number) => d3.interpolateRgb('#e0f2f1', '#0F6E56')(t))
      .domain([0, maxProb]);

    // Panel label
    const familyDef = FAMILIES.find((f) => f.key === highlightedFamily);
    svg
      .append('text')
      .attr('x', leftPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text(`${familyDef?.label ?? ''} (n=${n}) — distribution at t=${timeStep}`);

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
      .style('stroke-width', '1.5')
      .style('stroke-opacity', '0.5');

    // Draw nodes
    const nodeG = svg.append('g');
    nodeG
      .selectAll<SVGCircleElement, number>('circle')
      .data(d3.range(n))
      .join('circle')
      .attr('cx', (i) => positions[i]?.x ?? 0)
      .attr('cy', (i) => positions[i]?.y ?? 0)
      .attr('r', Math.max(8, Math.min(14, 120 / n)))
      .style('fill', (i) => colorScale(dist[i]))
      .style('stroke', (i) =>
        i === profile.worstStartVertex ? '#D97706' : 'var(--color-text)',
      )
      .style('stroke-width', (i) =>
        i === profile.worstStartVertex ? '2.5' : '1',
      );

    // Node labels (only when not too crowded)
    if (n <= 16) {
      nodeG
        .selectAll('text')
        .data(d3.range(n))
        .join('text')
        .attr('x', (i) => positions[i]?.x ?? 0)
        .attr('y', (i) => (positions[i]?.y ?? 0) + 4)
        .attr('text-anchor', 'middle')
        .attr('font-size', Math.max(8, Math.min(11, 100 / n)))
        .attr('font-weight', 600)
        .attr('pointer-events', 'none')
        .style('fill', (i) => (dist[i] > maxProb * 0.5 ? 'white' : 'var(--color-text)'))
        .text((i) => i);
    }

    // Start vertex indicator
    const sx = positions[profile.worstStartVertex]?.x ?? 0;
    const sy = positions[profile.worstStartVertex]?.y ?? 0;
    svg
      .append('text')
      .attr('x', sx)
      .attr('y', sy - Math.max(10, Math.min(16, 120 / n)) - 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('font-weight', 600)
      .style('fill', '#D97706')
      .text('start');

    // Color legend
    const legendW = 120;
    const legendH = 10;
    const legendX = leftPanelWidth - legendW - 12;
    const legendY = GRAPH_PANEL_HEIGHT - 24;

    const defs = svg.append('defs');
    const gradientId = 'dist-gradient';
    const gradient = defs
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '100%');
    gradient.append('stop').attr('offset', '0%').attr('stop-color', colorScale(0));
    gradient.append('stop').attr('offset', '100%').attr('stop-color', colorScale(maxProb));

    svg
      .append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendW)
      .attr('height', legendH)
      .attr('rx', 3)
      .attr('fill', `url(#${gradientId})`);

    svg
      .append('text')
      .attr('x', legendX)
      .attr('y', legendY - 3)
      .attr('font-size', 9)
      .style('fill', 'var(--color-text-secondary)')
      .text('0');
    svg
      .append('text')
      .attr('x', legendX + legendW)
      .attr('y', legendY - 3)
      .attr('text-anchor', 'end')
      .attr('font-size', 9)
      .style('fill', 'var(--color-text-secondary)')
      .text(maxProb.toFixed(3));
    svg
      .append('text')
      .attr('x', legendX + legendW / 2)
      .attr('y', legendY - 3)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .style('fill', 'var(--color-text-secondary)')
      .text('P(y)');
  }, [
    highlightedResult, highlightedDistribution, highlightedFamily,
    leftPanelWidth, timeStep, tick,
  ]);

  // ─── TV distance curves rendering (right top) ───

  useEffect(() => {
    if (!tvSvgRef.current || rightPanelWidth === 0) return;

    const svg = d3.select(tvSvgRef.current);
    svg.selectAll('*').remove();

    const innerW = rightPanelWidth - MARGIN.left - MARGIN.right;
    const innerH = TV_CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

    if (familyResults.size === 0) return;

    // Find the max TV distance across all selected families (should be ~0.5-1)
    let maxTV = 0.5;
    for (const [, result] of familyResults) {
      const localMax = d3.max(result.profile.tvDistances) ?? 0;
      if (localMax > maxTV) maxTV = localMax;
    }

    const xScale = d3.scaleLinear().domain([0, MAX_TIME]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, Math.min(1, maxTV * 1.1)]).range([innerH, 0]);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Title
    svg
      .append('text')
      .attr('x', rightPanelWidth / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Total variation distance vs. time');

    // Axes
    const xAxis = g
      .append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(8));
    xAxis.selectAll('text').style('fill', 'var(--color-text-secondary)').attr('font-size', 10);
    xAxis.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    const yAxis = g.append('g').call(d3.axisLeft(yScale).ticks(6));
    yAxis.selectAll('text').style('fill', 'var(--color-text-secondary)').attr('font-size', 10);
    yAxis.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    // Axis labels
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 34)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .style('fill', 'var(--color-text-secondary)')
      .text('Time step t');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .style('fill', 'var(--color-text-secondary)')
      .text('TV distance');

    // Epsilon threshold dashed line
    if (yScale.domain()[1] >= EPSILON) {
      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(EPSILON))
        .attr('x2', innerW)
        .attr('y2', yScale(EPSILON))
        .style('stroke', '#9CA3AF')
        .style('stroke-width', '1.5')
        .style('stroke-dasharray', '6,4')
        .style('opacity', '0.8');

      g.append('text')
        .attr('x', innerW - 4)
        .attr('y', yScale(EPSILON) - 5)
        .attr('text-anchor', 'end')
        .attr('font-size', 10)
        .style('fill', '#9CA3AF')
        .text(`ε = ${EPSILON}`);
    }

    // Time step indicator
    g.append('line')
      .attr('x1', xScale(timeStep))
      .attr('y1', 0)
      .attr('x2', xScale(timeStep))
      .attr('y2', innerH)
      .style('stroke', '#D97706')
      .style('stroke-width', '1.5')
      .style('stroke-dasharray', '4,3')
      .style('opacity', '0.7');

    g.append('text')
      .attr('x', xScale(timeStep))
      .attr('y', -6)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('font-weight', 600)
      .style('fill', '#D97706')
      .text(`t=${timeStep}`);

    // Draw curves and mixing time markers
    const lineGen = d3
      .line<number>()
      .x((_, i) => xScale(i))
      .y((d) => yScale(Math.max(0, d)))
      .curve(d3.curveMonotoneX);

    for (const [key, result] of familyResults) {
      const colorIdx = familyColorIndex.get(key) ?? 0;
      const color = FAMILY_COLORS[colorIdx % FAMILY_COLORS.length];
      const isHighlighted = key === highlightedFamily;
      const { tvDistances, mixingTime } = result.profile;

      // TV curve
      g.append('path')
        .datum(tvDistances)
        .attr('d', lineGen)
        .style('fill', 'none')
        .style('stroke', color)
        .style('stroke-width', isHighlighted ? '2.5' : '1.5')
        .style('opacity', isHighlighted ? '1' : '0.7');

      // Mixing time dot
      if (mixingTime < MAX_TIME) {
        g.append('circle')
          .attr('cx', xScale(mixingTime))
          .attr('cy', yScale(EPSILON))
          .attr('r', 4)
          .style('fill', color)
          .style('stroke', 'var(--color-bg)')
          .style('stroke-width', '1.5');

        // Label
        const familyDef = FAMILIES.find((f) => f.key === key);
        g.append('text')
          .attr('x', xScale(mixingTime))
          .attr('y', yScale(EPSILON) + 14)
          .attr('text-anchor', 'middle')
          .attr('font-size', 9)
          .attr('font-weight', 600)
          .style('fill', color)
          .text(`${familyDef?.label}: t=${mixingTime}`);
      }
    }

    // Legend
    const legendEntries = Array.from(familyResults.entries());
    const legendY = innerH + 16;
    let legendX = 0;
    for (const [key] of legendEntries) {
      const colorIdx = familyColorIndex.get(key) ?? 0;
      const color = FAMILY_COLORS[colorIdx % FAMILY_COLORS.length];
      const familyDef = FAMILIES.find((f) => f.key === key);
      const label = familyDef?.label ?? key;

      g.append('line')
        .attr('x1', legendX)
        .attr('y1', legendY + 5)
        .attr('x2', legendX + 16)
        .attr('y2', legendY + 5)
        .style('stroke', color)
        .style('stroke-width', '2');

      g.append('text')
        .attr('x', legendX + 20)
        .attr('y', legendY + 9)
        .attr('font-size', 10)
        .style('fill', 'var(--color-text-secondary)')
        .text(label);

      legendX += label.length * 7 + 30;
    }
  }, [familyResults, rightPanelWidth, timeStep, highlightedFamily, familyColorIndex]);

  // ─── Spectral gap bar chart rendering (right bottom) ───

  useEffect(() => {
    if (!barSvgRef.current || rightPanelWidth === 0) return;

    const svg = d3.select(barSvgRef.current);
    svg.selectAll('*').remove();

    const innerW = rightPanelWidth - MARGIN.left - MARGIN.right;
    const innerH = BAR_CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

    if (familyResults.size === 0) return;

    // Sort by spectral gap ascending
    const entries = Array.from(familyResults.entries())
      .map(([key, result]) => ({
        key,
        label: FAMILIES.find((f) => f.key === key)?.label ?? key,
        spectralGap: result.profile.spectralGap,
      }))
      .sort((a, b) => a.spectralGap - b.spectralGap);

    const maxGap = d3.max(entries, (d) => d.spectralGap) ?? 1;

    const xScale = d3
      .scaleLinear()
      .domain([0, Math.min(1, maxGap * 1.2)])
      .range([0, innerW]);

    const yScale = d3
      .scaleBand<string>()
      .domain(entries.map((d) => d.key))
      .range([0, innerH])
      .padding(0.2);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Title
    svg
      .append('text')
      .attr('x', rightPanelWidth / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Spectral gap γ = 1 − λ*');

    // Bars
    g.selectAll('rect')
      .data(entries)
      .join('rect')
      .attr('x', 0)
      .attr('y', (d) => yScale(d.key)!)
      .attr('width', (d) => Math.max(2, xScale(Math.max(0, d.spectralGap))))
      .attr('height', yScale.bandwidth())
      .attr('rx', 3)
      .attr('fill', (d) => {
        const colorIdx = familyColorIndex.get(d.key) ?? 0;
        return FAMILY_COLORS[colorIdx % FAMILY_COLORS.length];
      })
      .style('opacity', (d) => (d.key === highlightedFamily ? '1' : '0.7'));

    // Bar value labels
    g.selectAll<SVGTextElement, typeof entries[number]>('.bar-val')
      .data(entries)
      .join('text')
      .attr('class', 'bar-val')
      .attr('x', (d) => Math.max(2, xScale(Math.max(0, d.spectralGap))) + 6)
      .attr('y', (d) => (yScale(d.key)!) + yScale.bandwidth() / 2 + 4)
      .attr('font-size', 10)
      .attr('font-weight', 500)
      .style('fill', 'var(--color-text)')
      .text((d) => d.spectralGap.toFixed(4));

    // Y-axis labels
    g.selectAll<SVGTextElement, typeof entries[number]>('.y-label')
      .data(entries)
      .join('text')
      .attr('class', 'y-label')
      .attr('x', -8)
      .attr('y', (d) => (yScale(d.key)!) + yScale.bandwidth() / 2 + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', 11)
      .attr('font-weight', (d) => (d.key === highlightedFamily ? 700 : 500))
      .style('fill', (d) => {
        const colorIdx = familyColorIndex.get(d.key) ?? 0;
        return FAMILY_COLORS[colorIdx % FAMILY_COLORS.length];
      })
      .text((d) => d.label);

    // X-axis
    const xAxis = g
      .append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5));
    xAxis.selectAll('text').style('fill', 'var(--color-text-secondary)').attr('font-size', 10);
    xAxis.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
  }, [familyResults, rightPanelWidth, highlightedFamily, familyColorIndex]);

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
      <div className="mb-4 flex flex-wrap items-start gap-4">
        {/* Family checkboxes */}
        <fieldset className="flex flex-wrap gap-2">
          <legend
            className="mb-1 text-sm font-medium"
            style={{ color: 'var(--color-text)' }}
          >
            Graph families
          </legend>
          {FAMILIES.map((fam, idx) => {
            const checked = selectedFamilies.has(fam.key);
            const color = FAMILY_COLORS[idx % FAMILY_COLORS.length];
            const isHighlighted = fam.key === highlightedFamily;
            return (
              <label
                key={fam.key}
                className="flex items-center gap-1 rounded px-2 py-1 text-sm cursor-pointer select-none"
                style={{
                  borderWidth: 1,
                  borderStyle: 'solid',
                  borderColor: isHighlighted ? color : 'var(--color-border)',
                  backgroundColor: isHighlighted
                    ? `${color}18`
                    : 'var(--color-bg)',
                  color: checked ? color : 'var(--color-text-secondary)',
                  fontWeight: isHighlighted ? 600 : 400,
                }}
                onClick={(e) => {
                  // Click on the label area (not the checkbox) highlights the family
                  if ((e.target as HTMLElement).tagName !== 'INPUT') {
                    if (checked) {
                      handleFamilyHighlight(fam.key);
                    }
                  }
                }}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => handleFamilyToggle(fam.key)}
                  style={{ accentColor: color }}
                />
                <span>{fam.label}</span>
              </label>
            );
          })}
        </fieldset>

        {/* Size slider */}
        <label
          className="flex flex-col gap-1 text-sm"
          style={{ color: 'var(--color-text)' }}
        >
          <span className="font-medium">Size n = {sizeParam}</span>
          <input
            type="range"
            min={6}
            max={16}
            step={1}
            value={sizeParam}
            onChange={(e) => {
              setSizeParam(Number(e.target.value));
              setTimeStep(0);
            }}
            className="w-28"
            style={{ accentColor: '#0F6E56' }}
          />
        </label>

        {/* Time step slider */}
        <label
          className="flex flex-col gap-1 text-sm"
          style={{ color: 'var(--color-text)' }}
        >
          <span className="font-medium">Time step t = {timeStep}</span>
          <input
            type="range"
            min={0}
            max={MAX_TIME}
            step={1}
            value={timeStep}
            onChange={(e) => setTimeStep(Number(e.target.value))}
            className="w-36"
            style={{ accentColor: '#D97706' }}
          />
        </label>
      </div>

      {/* Main panels */}
      <div
        className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}
        style={{ alignItems: 'flex-start' }}
      >
        {/* Left: force-directed graph with distribution heatmap */}
        <div style={{ flexShrink: 0 }}>
          <svg
            ref={graphSvgRef}
            width={leftPanelWidth}
            height={GRAPH_PANEL_HEIGHT}
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: 8,
              backgroundColor: 'var(--color-bg)',
            }}
          />
        </div>

        {/* Right: stacked charts */}
        <div style={{ flexShrink: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* TV distance curves */}
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

          {/* Spectral gap bar chart */}
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
        </div>
      </div>

      {/* Annotations */}
      <div
        className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs"
        style={{ color: 'var(--color-text-secondary)' }}
      >
        <span>
          <strong>Checkboxes</strong> toggle families on/off
        </span>
        <span>
          <strong>Click label</strong> to highlight family in graph panel
        </span>
        <span>
          <strong>Size slider</strong> adjusts node count for all families
        </span>
        <span>
          <strong>Time slider</strong> scrubs the walk distribution on the graph
        </span>
        <span style={{ color: '#D97706' }}>
          <strong>Amber node</strong> = worst-case start vertex
        </span>
      </div>

      {/* Summary row */}
      {familyResults.size > 0 && (
        <div
          className="mt-3 rounded px-3 py-2 text-sm"
          style={{
            backgroundColor: 'var(--color-bg)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text)',
          }}
        >
          {Array.from(familyResults.entries()).map(([key, result]) => {
            const familyDef = FAMILIES.find((f) => f.key === key);
            const colorIdx = familyColorIndex.get(key) ?? 0;
            const color = FAMILY_COLORS[colorIdx % FAMILY_COLORS.length];
            const { mixingTime, spectralGap } = result.profile;
            const nActual = result.graph.n;
            return (
              <span
                key={key}
                className="mr-4 inline-block"
                style={{ color }}
              >
                <strong>{familyDef?.label}</strong> (n={nActual}):
                t_mix = {mixingTime < MAX_TIME ? mixingTime : `>${MAX_TIME}`},
                {' '}γ = {spectralGap.toFixed(4)}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

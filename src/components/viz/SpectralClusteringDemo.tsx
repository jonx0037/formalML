import { useState, useMemo, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  twoMoons, twoCircles, threeBlobs, spiral,
  knnGraph, epsilonBallGraph,
  normalizedLaplacian, jacobiEigen,
  kMeans,
  type Graph,
} from './shared/graphTheory';

type DatasetName = 'Two Moons' | 'Two Circles' | 'Three Blobs' | 'Spiral';
type GraphMode = 'knn' | 'epsilon';

const DATASETS: DatasetName[] = ['Two Moons', 'Two Circles', 'Three Blobs', 'Spiral'];
const K_OPTIONS = [2, 3, 4];
const N_POINTS = 100;
const SEED = 42;
const KNN_K = 7;
const MAX_EDGES_SHOWN = 500;
const MARGIN = { top: 28, right: 12, bottom: 12, left: 12 };
const PANEL_TITLES = ['Input Data', 'Similarity Graph', 'Spectral Embedding', 'Clustering Result'];
const MOBILE_BREAKPOINT = 640;

function generatePoints(name: DatasetName, noise: number): [number, number][] {
  switch (name) {
    case 'Two Moons': return twoMoons(N_POINTS, noise, SEED);
    case 'Two Circles': return twoCircles(N_POINTS, noise, SEED);
    case 'Three Blobs': return threeBlobs(N_POINTS, noise, SEED);
    case 'Spiral': return spiral(N_POINTS, noise, SEED);
  }
}

function computeMedianPairwiseDist(points: [number, number][]): number {
  const dists: number[] = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      dists.push(Math.sqrt(dx * dx + dy * dy));
    }
  }
  dists.sort((a, b) => a - b);
  return dists[Math.floor(dists.length / 2)];
}

export default function SpectralClusteringDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const svgRef1 = useRef<SVGSVGElement>(null);
  const svgRef2 = useRef<SVGSVGElement>(null);
  const svgRef3 = useRef<SVGSVGElement>(null);
  const svgRef4 = useRef<SVGSVGElement>(null);

  const [datasetName, setDatasetName] = useState<DatasetName>('Two Moons');
  const [kClusters, setKClusters] = useState(2);
  const [graphMode, setGraphMode] = useState<GraphMode>('knn');
  const [noise, setNoise] = useState(1.0);

  const isMobile = containerWidth > 0 && containerWidth < MOBILE_BREAKPOINT;

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 280;
    if (isMobile) return Math.min(containerWidth - 16, 480);
    return Math.floor((containerWidth - 24) / 2);
  }, [containerWidth, isMobile]);

  const panelHeight = useMemo(() => Math.min(panelWidth, 320), [panelWidth]);

  // --- Pipeline computation ---

  const points = useMemo(
    () => generatePoints(datasetName, noise),
    [datasetName, noise],
  );

  const graph: Graph = useMemo(() => {
    if (graphMode === 'knn') {
      return knnGraph(points, KNN_K);
    }
    const median = computeMedianPairwiseDist(points);
    const epsilon = median * 0.5;
    return epsilonBallGraph(points, epsilon);
  }, [points, graphMode]);

  const eigenResult = useMemo(() => {
    const Lnorm = normalizedLaplacian(graph.adjacency);
    return jacobiEigen(Lnorm);
  }, [graph]);

  const spectralEmbedding = useMemo(() => {
    // eigenvectors[i] is the i-th eigenvector (sorted by ascending eigenvalue).
    // Skip eigenvector 0 (trivial), take eigenvectors 1..kClusters.
    const k = Math.min(kClusters, eigenResult.eigenvectors.length - 1);
    const n = points.length;
    const rows: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 1; j <= k; j++) {
        row.push(eigenResult.eigenvectors[j][i]);
      }
      rows.push(row);
    }

    // Ng-Jordan-Weiss: normalize each row to unit length
    for (let i = 0; i < n; i++) {
      let norm = 0;
      for (let d = 0; d < rows[i].length; d++) norm += rows[i][d] * rows[i][d];
      norm = Math.sqrt(norm);
      if (norm > 1e-10) {
        for (let d = 0; d < rows[i].length; d++) rows[i][d] /= norm;
      }
    }

    return rows;
  }, [eigenResult, kClusters, points.length]);

  const clusterAssignments = useMemo(
    () => kMeans(spectralEmbedding, kClusters, 20, SEED),
    [spectralEmbedding, kClusters],
  );

  const clusterColors = useMemo(() => d3.schemeCategory10, []);

  // Edges for Panel 2 — subsample if too many
  const visibleEdges = useMemo(() => {
    const allEdges: { i: number; j: number; w: number }[] = [];
    const adj = graph.adjacency;
    const n = graph.n;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (adj[i][j] > 0) {
          allEdges.push({ i, j, w: adj[i][j] });
        }
      }
    }
    if (allEdges.length <= MAX_EDGES_SHOWN) return allEdges;
    // Keep the strongest edges
    allEdges.sort((a, b) => b.w - a.w);
    return allEdges.slice(0, MAX_EDGES_SHOWN);
  }, [graph]);

  // --- Rendering ---

  // Panel 1: Input data (neutral gray)
  useEffect(() => {
    const svg = d3.select(svgRef1.current);
    if (!svgRef1.current || panelWidth === 0) return;
    svg.selectAll('*').remove();

    const innerW = panelWidth - MARGIN.left - MARGIN.right;
    const innerH = panelHeight - MARGIN.top - MARGIN.bottom;

    const xExtent = d3.extent(points, (p) => p[0]) as [number, number];
    const yExtent = d3.extent(points, (p) => p[1]) as [number, number];
    const pad = 0.1;
    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - pad, xExtent[1] + pad])
      .range([MARGIN.left, MARGIN.left + innerW]);
    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - pad, yExtent[1] + pad])
      .range([MARGIN.top + innerH, MARGIN.top]);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .text(PANEL_TITLES[0]);

    // Points in neutral gray
    svg.selectAll('.pt')
      .data(points)
      .join('circle')
      .attr('class', 'pt')
      .attr('cx', (p) => xScale(p[0]))
      .attr('cy', (p) => yScale(p[1]))
      .attr('r', 3.5)
      .attr('fill', '#999')
      .attr('fill-opacity', 0.75)
      .style('stroke', 'var(--color-surface)')
      .attr('stroke-width', 0.5);
  }, [points, panelWidth, panelHeight]);

  // Panel 2: Similarity graph
  useEffect(() => {
    const svg = d3.select(svgRef2.current);
    if (!svgRef2.current || panelWidth === 0) return;
    svg.selectAll('*').remove();

    const innerW = panelWidth - MARGIN.left - MARGIN.right;
    const innerH = panelHeight - MARGIN.top - MARGIN.bottom;

    const xExtent = d3.extent(points, (p) => p[0]) as [number, number];
    const yExtent = d3.extent(points, (p) => p[1]) as [number, number];
    const pad = 0.1;
    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - pad, xExtent[1] + pad])
      .range([MARGIN.left, MARGIN.left + innerW]);
    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - pad, yExtent[1] + pad])
      .range([MARGIN.top + innerH, MARGIN.top]);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .text(PANEL_TITLES[1]);

    // Max weight for opacity scaling
    const maxW = d3.max(visibleEdges, (e) => e.w) ?? 1;

    // Edges
    svg.selectAll('.edge')
      .data(visibleEdges)
      .join('line')
      .attr('class', 'edge')
      .attr('x1', (e) => xScale(points[e.i][0]))
      .attr('y1', (e) => yScale(points[e.i][1]))
      .attr('x2', (e) => xScale(points[e.j][0]))
      .attr('y2', (e) => yScale(points[e.j][1]))
      .style('stroke', 'var(--color-text)')
      .attr('stroke-opacity', (e) => 0.05 + 0.5 * (e.w / maxW))
      .attr('stroke-width', (e) => 0.3 + 0.7 * (e.w / maxW));

    // Points
    svg.selectAll('.pt')
      .data(points)
      .join('circle')
      .attr('class', 'pt')
      .attr('cx', (p) => xScale(p[0]))
      .attr('cy', (p) => yScale(p[1]))
      .attr('r', 3)
      .attr('fill', '#999')
      .attr('fill-opacity', 0.8)
      .style('stroke', 'var(--color-surface)')
      .attr('stroke-width', 0.5);
  }, [points, visibleEdges, panelWidth, panelHeight]);

  // Panel 3: Spectral embedding
  useEffect(() => {
    const svg = d3.select(svgRef3.current);
    if (!svgRef3.current || panelWidth === 0) return;
    svg.selectAll('*').remove();

    const innerW = panelWidth - MARGIN.left - MARGIN.right;
    const innerH = panelHeight - MARGIN.top - MARGIN.bottom;

    // Use first two dimensions of the embedding for x,y
    const embX = spectralEmbedding.map((r) => r[0]);
    const embY = spectralEmbedding.map((r) => r.length > 1 ? r[1] : 0);

    const xExtent = d3.extent(embX) as [number, number];
    const yExtent = d3.extent(embY) as [number, number];
    const padX = (xExtent[1] - xExtent[0]) * 0.1 || 0.1;
    const padY = (yExtent[1] - yExtent[0]) * 0.1 || 0.1;

    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - padX, xExtent[1] + padX])
      .range([MARGIN.left, MARGIN.left + innerW]);
    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - padY, yExtent[1] + padY])
      .range([MARGIN.top + innerH, MARGIN.top]);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .text(PANEL_TITLES[2]);

    // Axis labels
    svg.append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', MARGIN.top + innerH + 10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .text('v\u2082');

    svg.append('text')
      .attr('x', 6)
      .attr('y', MARGIN.top + innerH / 2)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, 6, ${MARGIN.top + innerH / 2})`)
      .style('fill', 'var(--color-text-secondary)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .text('v\u2083');

    // Points colored by cluster
    svg.selectAll('.pt')
      .data(spectralEmbedding)
      .join('circle')
      .attr('class', 'pt')
      .attr('cx', (_, i) => xScale(embX[i]))
      .attr('cy', (_, i) => yScale(embY[i]))
      .attr('r', 3.5)
      .attr('fill', (_, i) => clusterColors[clusterAssignments[i] % 10])
      .attr('fill-opacity', 0.85)
      .style('stroke', 'var(--color-surface)')
      .attr('stroke-width', 0.5);
  }, [spectralEmbedding, clusterAssignments, clusterColors, panelWidth, panelHeight]);

  // Panel 4: Clustering result (original positions, colored)
  useEffect(() => {
    const svg = d3.select(svgRef4.current);
    if (!svgRef4.current || panelWidth === 0) return;
    svg.selectAll('*').remove();

    const innerW = panelWidth - MARGIN.left - MARGIN.right;
    const innerH = panelHeight - MARGIN.top - MARGIN.bottom;

    const xExtent = d3.extent(points, (p) => p[0]) as [number, number];
    const yExtent = d3.extent(points, (p) => p[1]) as [number, number];
    const pad = 0.1;
    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - pad, xExtent[1] + pad])
      .range([MARGIN.left, MARGIN.left + innerW]);
    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - pad, yExtent[1] + pad])
      .range([MARGIN.top + innerH, MARGIN.top]);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .text(PANEL_TITLES[3]);

    // Points colored by cluster assignment
    svg.selectAll('.pt')
      .data(points)
      .join('circle')
      .attr('class', 'pt')
      .attr('cx', (p) => xScale(p[0]))
      .attr('cy', (p) => yScale(p[1]))
      .attr('r', 3.5)
      .attr('fill', (_, i) => clusterColors[clusterAssignments[i] % 10])
      .attr('fill-opacity', 0.85)
      .style('stroke', 'var(--color-surface)')
      .attr('stroke-width', 0.5);
  }, [points, clusterAssignments, clusterColors, panelWidth, panelHeight]);

  return (
    <div ref={containerRef} className="w-full space-y-4">
      {/* Controls */}
      <div
        className="flex flex-wrap items-center gap-3"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        {/* Dataset selector */}
        <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
          Dataset
          <select
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value as DatasetName)}
            className="rounded-md px-2 py-1 text-xs"
            style={{
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              border: '1px solid var(--color-border)',
            }}
          >
            {DATASETS.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>

        {/* k clusters */}
        <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
          Clusters k
          <div className="flex gap-1">
            {K_OPTIONS.map((k) => (
              <button
                key={k}
                onClick={() => setKClusters(k)}
                className="rounded-md px-2 py-1 text-xs font-medium transition-colors"
                style={{
                  background: k === kClusters ? 'var(--color-accent)' : 'var(--color-surface)',
                  color: k === kClusters ? '#fff' : 'var(--color-text)',
                  border: `1px solid ${k === kClusters ? 'var(--color-accent)' : 'var(--color-border)'}`,
                }}
              >
                {k}
              </button>
            ))}
          </div>
        </label>

        {/* Graph mode toggle */}
        <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
          Graph
          <div className="flex gap-1">
            {(['knn', 'epsilon'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setGraphMode(mode)}
                className="rounded-md px-2 py-1 text-xs font-medium transition-colors"
                style={{
                  background: mode === graphMode ? 'var(--color-accent)' : 'var(--color-surface)',
                  color: mode === graphMode ? '#fff' : 'var(--color-text)',
                  border: `1px solid ${mode === graphMode ? 'var(--color-accent)' : 'var(--color-border)'}`,
                }}
              >
                {mode === 'knn' ? 'k-NN' : '\u03B5-ball'}
              </button>
            ))}
          </div>
        </label>

        {/* Noise slider */}
        <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
          Noise {noise.toFixed(1)}
          <input
            type="range"
            min={0.5}
            max={3.0}
            step={0.1}
            value={noise}
            onChange={(e) => setNoise(parseFloat(e.target.value))}
            className="w-20"
          />
        </label>
      </div>

      {/* 2x2 panel grid */}
      <div
        className="gap-3"
        style={{
          display: 'grid',
          gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
        }}
      >
        {[svgRef1, svgRef2, svgRef3, svgRef4].map((ref, idx) => (
          <svg role="img" aria-label={`Spectral clustering — step ${idx + 1} of 4`}
            key={idx}
            ref={ref}
            width={panelWidth}
            height={panelHeight}
            className="rounded-lg border"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
          />
        ))}
      </div>
    </div>
  );
}

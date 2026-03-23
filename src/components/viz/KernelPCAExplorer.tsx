import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { kernelPCADatasets, GAMMAS } from '../../data/kernel-pca-data';

const MARGIN = { top: 20, right: 16, bottom: 32, left: 40 };
const SVG_HEIGHT = 300;
const MOBILE_BREAKPOINT = 640;
const HEATMAP_N = 20;
const POINT_RADIUS = 3;

type KernelType = 'rbf' | 'linear';

export default function KernelPCAExplorer() {
  const { ref, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetIdx, setDatasetIdx] = useState(0);
  const [gammaIdx, setGammaIdx] = useState(4); // default gamma=1
  const [kernelType, setKernelType] = useState<KernelType>('rbf');

  const leftSvgRef = useRef<SVGSVGElement>(null);
  const centerSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const dataset = kernelPCADatasets[datasetIdx];
  const gamma = GAMMAS[gammaIdx];

  // Find the nearest pre-computed embedding for the current gamma
  const embedding = useMemo(() => {
    if (kernelType === 'linear') {
      // Use the lowest gamma embedding as a proxy for linear PCA
      return dataset.embeddings[0];
    }
    // Find exact match
    const match = dataset.embeddings.find((e) => e.gamma === gamma);
    if (match) return match;
    // Find nearest
    let bestIdx = 0;
    let bestDist = Math.abs(Math.log(dataset.embeddings[0].gamma) - Math.log(gamma));
    for (let i = 1; i < dataset.embeddings.length; i++) {
      const dist = Math.abs(Math.log(dataset.embeddings[i].gamma) - Math.log(gamma));
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    return dataset.embeddings[bestIdx];
  }, [dataset, gamma, kernelType]);

  // Color scale for labels
  const colorScale = useMemo(() => {
    const labels = dataset.points.map((p) => p.label);
    const uniqueLabels = [...new Set(labels)];
    if (uniqueLabels.length === 2) {
      // Binary: red/blue
      return (label: number) => (label === uniqueLabels[0] ? '#e74c3c' : '#3498db');
    }
    // Continuous or many labels: viridis
    const extent = d3.extent(labels) as [number, number];
    const scale = d3.scaleSequential(d3.interpolateViridis).domain(extent);
    return (label: number) => scale(label);
  }, [dataset]);

  // Compute kernel matrix for heatmap (first HEATMAP_N points)
  const kernelMatrix = useMemo(() => {
    const pts = dataset.points.slice(0, HEATMAP_N);
    const n = pts.length;
    const K: number[][] = [];

    for (let i = 0; i < n; i++) {
      K[i] = [];
      for (let j = 0; j < n; j++) {
        if (kernelType === 'linear') {
          K[i][j] = pts[i].x * pts[j].x + pts[i].y * pts[j].y;
        } else {
          // RBF kernel
          const dx = pts[i].x - pts[j].x;
          const dy = pts[i].y - pts[j].y;
          K[i][j] = Math.exp(-gamma * (dx * dx + dy * dy));
        }
      }
    }
    return K;
  }, [dataset, gamma, kernelType]);

  const isMobile = (containerWidth || 900) < MOBILE_BREAKPOINT;
  const totalWidth = containerWidth || 900;
  const panelWidth = isMobile ? totalWidth : Math.floor((totalWidth - 32) / 3);
  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  const handleDatasetChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setDatasetIdx(Number(e.target.value));
  }, []);

  const handleGammaChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setGammaIdx(Number(e.target.value));
  }, []);

  const handleKernelChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setKernelType(e.target.value as KernelType);
  }, []);

  // ─── Left panel: Original data scatter ───
  useEffect(() => {
    const svg = d3.select(leftSvgRef.current);
    if (!leftSvgRef.current || innerW <= 0) return;
    svg.selectAll('*').remove();

    const pts = dataset.points;
    const xExtent = d3.extent(pts, (p) => p.x) as [number, number];
    const yExtent = d3.extent(pts, (p) => p.y) as [number, number];

    // Equal aspect ratio: use the larger range for both axes
    const xRange = xExtent[1] - xExtent[0];
    const yRange = yExtent[1] - yExtent[0];
    const maxRange = Math.max(xRange, yRange) * 1.1;
    const xMid = (xExtent[0] + xExtent[1]) / 2;
    const yMid = (yExtent[0] + yExtent[1]) / 2;

    const xScale = d3
      .scaleLinear()
      .domain([xMid - maxRange / 2, xMid + maxRange / 2])
      .range([MARGIN.left, MARGIN.left + innerW]);

    const yScale = d3
      .scaleLinear()
      .domain([yMid - maxRange / 2, yMid + maxRange / 2])
      .range([MARGIN.top + innerH, MARGIN.top]);

    const g = svg.append('g');

    // Grid
    xScale.ticks(4).forEach((t) => {
      g.append('line')
        .attr('x1', xScale(t))
        .attr('x2', xScale(t))
        .attr('y1', MARGIN.top)
        .attr('y2', MARGIN.top + innerH)
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', 0.2);
    });
    yScale.ticks(4).forEach((t) => {
      g.append('line')
        .attr('x1', MARGIN.left)
        .attr('x2', MARGIN.left + innerW)
        .attr('y1', yScale(t))
        .attr('y2', yScale(t))
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', 0.2);
    });

    // Points
    g.selectAll('.pt')
      .data(pts)
      .join('circle')
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('r', POINT_RADIUS)
      .style('fill', (d) => colorScale(d.label))
      .style('fill-opacity', 0.75)
      .style('stroke', 'var(--color-surface)')
      .style('stroke-width', 0.5);

    // Axis labels
    g.append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', MARGIN.top + innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '9px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.5)
      .style('font-family', 'var(--font-sans)')
      .text('x');

    g.append('text')
      .attr('x', 12)
      .attr('y', MARGIN.top + innerH / 2)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, 12, ${MARGIN.top + innerH / 2})`)
      .style('font-size', '9px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.5)
      .style('font-family', 'var(--font-sans)')
      .text('y');
  }, [dataset, innerW, innerH, colorScale]);

  // ─── Center panel: Kernel PCA embedding ───
  useEffect(() => {
    const svg = d3.select(centerSvgRef.current);
    if (!centerSvgRef.current || innerW <= 0) return;
    svg.selectAll('*').remove();

    const pts = embedding.points;
    const xExtent = d3.extent(pts, (p) => p.x) as [number, number];
    const yExtent = d3.extent(pts, (p) => p.y) as [number, number];

    const xRange = xExtent[1] - xExtent[0] || 1;
    const yRange = yExtent[1] - yExtent[0] || 1;
    const maxRange = Math.max(xRange, yRange) * 1.15;
    const xMid = (xExtent[0] + xExtent[1]) / 2;
    const yMid = (yExtent[0] + yExtent[1]) / 2;

    const xScale = d3
      .scaleLinear()
      .domain([xMid - maxRange / 2, xMid + maxRange / 2])
      .range([MARGIN.left, MARGIN.left + innerW]);

    const yScale = d3
      .scaleLinear()
      .domain([yMid - maxRange / 2, yMid + maxRange / 2])
      .range([MARGIN.top + innerH, MARGIN.top]);

    const g = svg.append('g');

    // Grid
    xScale.ticks(4).forEach((t) => {
      g.append('line')
        .attr('x1', xScale(t))
        .attr('x2', xScale(t))
        .attr('y1', MARGIN.top)
        .attr('y2', MARGIN.top + innerH)
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', 0.2);
    });
    yScale.ticks(4).forEach((t) => {
      g.append('line')
        .attr('x1', MARGIN.left)
        .attr('x2', MARGIN.left + innerW)
        .attr('y1', yScale(t))
        .attr('y2', yScale(t))
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', 0.2);
    });

    // Points — use original dataset labels for coloring
    g.selectAll('.pt')
      .data(pts)
      .join('circle')
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('r', POINT_RADIUS)
      .style('fill', (_, i) => colorScale(dataset.points[i]?.label ?? 0))
      .style('fill-opacity', 0.75)
      .style('stroke', 'var(--color-surface)')
      .style('stroke-width', 0.5);

    // Axis labels
    g.append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', MARGIN.top + innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '9px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.5)
      .style('font-family', 'var(--font-sans)')
      .text('PC 1');

    g.append('text')
      .attr('x', 12)
      .attr('y', MARGIN.top + innerH / 2)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, 12, ${MARGIN.top + innerH / 2})`)
      .style('font-size', '9px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.5)
      .style('font-family', 'var(--font-sans)')
      .text('PC 2');
  }, [embedding, dataset, innerW, innerH, colorScale]);

  // ─── Right panel: Kernel matrix heatmap ───
  useEffect(() => {
    const svg = d3.select(rightSvgRef.current);
    if (!rightSvgRef.current || innerW <= 0) return;
    svg.selectAll('*').remove();

    const n = kernelMatrix.length;
    const heatSize = Math.min(innerW, innerH);
    const cellSize = heatSize / n;
    const offsetX = MARGIN.left + (innerW - heatSize) / 2;
    const offsetY = MARGIN.top + (innerH - heatSize) / 2;

    // Determine value range for color scale
    let vMin = Infinity;
    let vMax = -Infinity;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (kernelMatrix[i][j] < vMin) vMin = kernelMatrix[i][j];
        if (kernelMatrix[i][j] > vMax) vMax = kernelMatrix[i][j];
      }
    }

    const colorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([vMin, vMax]);

    const g = svg.append('g');

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        g.append('rect')
          .attr('x', offsetX + j * cellSize)
          .attr('y', offsetY + i * cellSize)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .style('fill', colorScale(kernelMatrix[i][j]));
      }
    }

    // Border around the heatmap
    g.append('rect')
      .attr('x', offsetX)
      .attr('y', offsetY)
      .attr('width', heatSize)
      .attr('height', heatSize)
      .style('fill', 'none')
      .style('stroke', 'var(--color-border)')
      .style('stroke-width', 0.5);

    // Label
    g.append('text')
      .attr('x', offsetX + heatSize / 2)
      .attr('y', MARGIN.top + innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '9px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.5)
      .style('font-family', 'var(--font-sans)')
      .text(`K (${n} x ${n} subsample)`);
  }, [kernelMatrix, innerW, innerH]);

  return (
    <div ref={ref} className="w-full space-y-3">
      {/* Controls */}
      <div
        className="flex flex-wrap items-center gap-4"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        {/* Dataset selector */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium whitespace-nowrap">Dataset</label>
          <select
            value={datasetIdx}
            onChange={handleDatasetChange}
            className="text-xs rounded border px-2 py-1"
            style={{
              borderColor: 'var(--color-border)',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
            }}
          >
            {kernelPCADatasets.map((ds, i) => (
              <option key={ds.name} value={i}>
                {ds.name}
              </option>
            ))}
          </select>
        </div>

        {/* Kernel type radio buttons */}
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium">Kernel</span>
          {(['linear', 'rbf'] as KernelType[]).map((kt) => (
            <label key={kt} className="flex items-center gap-1 text-xs">
              <input
                type="radio"
                name="kernelType"
                value={kt}
                checked={kernelType === kt}
                onChange={handleKernelChange}
                className="accent-[var(--color-accent)]"
              />
              {kt === 'rbf' ? 'RBF' : 'Linear'}
            </label>
          ))}
        </div>

        {/* Gamma slider */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium whitespace-nowrap">
            {'\u03B3'} = {gamma}
          </label>
          <input
            type="range"
            min={0}
            max={GAMMAS.length - 1}
            step={1}
            value={gammaIdx}
            onChange={handleGammaChange}
            disabled={kernelType === 'linear'}
            className="w-24 accent-[var(--color-accent)]"
          />
        </div>
      </div>

      {/* Three-panel chart area */}
      <div
        className="flex gap-4"
        style={{ flexDirection: isMobile ? 'column' : 'row' }}
      >
        {/* Left: Original data */}
        <div className="flex-1 min-w-0">
          <p
            className="text-xs font-medium mb-1 opacity-70"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Original data
          </p>
          <svg
            ref={leftSvgRef}
            width={panelWidth}
            height={SVG_HEIGHT}
            className="rounded-lg border"
            style={{ borderColor: 'var(--color-border)' }}
          />
        </div>

        {/* Center: Kernel PCA embedding */}
        <div className="flex-1 min-w-0">
          <p
            className="text-xs font-medium mb-1 opacity-70"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            {kernelType === 'linear' ? 'Linear PCA' : `RBF Kernel PCA (\u03B3=${gamma})`}
          </p>
          <svg
            ref={centerSvgRef}
            width={panelWidth}
            height={SVG_HEIGHT}
            className="rounded-lg border"
            style={{ borderColor: 'var(--color-border)' }}
          />
        </div>

        {/* Right: Kernel matrix heatmap */}
        <div className="flex-1 min-w-0">
          <p
            className="text-xs font-medium mb-1 opacity-70"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Kernel matrix
          </p>
          <svg
            ref={rightSvgRef}
            width={panelWidth}
            height={SVG_HEIGHT}
            className="rounded-lg border"
            style={{ borderColor: 'var(--color-border)' }}
          />
        </div>
      </div>

      {/* Readout */}
      <p
        className="text-xs opacity-60"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        {dataset.points.length} points &nbsp;|&nbsp; Kernel:{' '}
        {kernelType === 'linear'
          ? 'Linear (x\u1D40y)'
          : `RBF exp(-\u03B3\u2016x\u2212y\u2016\u00B2), \u03B3=${gamma}`}
      </p>
    </div>
  );
}

import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pathGraph, cycleGraph, completeGraph, starGraph, barbellGraph,
  gridGraph, petersenGraph,
  degreeMatrix, laplacian, degrees,
  jacobiEigen, countComponents,
  type Graph, type EigenResult,
} from './shared/graphTheory';

// ─── Layout constants ───

const SM_BREAKPOINT = 640;
const MAX_NODES = 15;
const GRAPH_PANEL_HEIGHT = 360;
const MATRIX_CELL = 28;
const EIGEN_BAR_HEIGHT = 340;
const MARGIN = { top: 28, right: 16, bottom: 32, left: 44 };

type ColoringMode = 'fiedler' | 'degree' | 'none';

interface Preset {
  label: string;
  build: () => Graph;
}

const PRESETS: Preset[] = [
  { label: 'Path(6)', build: () => pathGraph(6) },
  { label: 'Cycle(6)', build: () => cycleGraph(6) },
  { label: 'Complete(6)', build: () => completeGraph(6) },
  { label: 'Star(6)', build: () => starGraph(6) },
  { label: 'Barbell(3-3)', build: () => barbellGraph(3) },
  { label: 'Petersen', build: () => petersenGraph() },
  { label: 'Grid(3×3)', build: () => gridGraph(3) },
];

// ─── Helpers ───

/** Clone an adjacency matrix. */
function cloneAdj(adj: number[][]): number[][] {
  return adj.map((row) => [...row]);
}

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

// ─── Component ───

export default function GraphLaplacianExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const graphSvgRef = useRef<SVGSVGElement>(null);
  const matrixSvgRef = useRef<SVGSVGElement>(null);
  const eigenSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───

  const [adjacency, setAdjacency] = useState<number[][]>(() => pathGraph(6).adjacency);
  const [presetIdx, setPresetIdx] = useState(0);
  const [coloringMode, setColoringMode] = useState<ColoringMode>('fiedler');
  const [showMatrices, setShowMatrices] = useState(true);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [warning, setWarning] = useState<string | null>(null);

  // Node positions managed outside React state for the force simulation.
  // We keep a ref so the simulation can mutate them in place and we
  // re-render via a tick counter.
  const positionsRef = useRef<{ x: number; y: number }[]>([]);
  const simRef = useRef<d3.Simulation<{ x: number; y: number }, undefined> | null>(null);
  const [tick, setTick] = useState(0);

  const n = adjacency.length;

  // ─── Derived computations ───

  const graph: Graph = useMemo(() => ({ n, adjacency }), [n, adjacency]);

  const D = useMemo(() => degreeMatrix(adjacency), [adjacency]);
  const L = useMemo(() => laplacian(adjacency), [adjacency]);
  const eigenResult: EigenResult = useMemo(() => jacobiEigen(L), [L]);
  const deg = useMemo(() => degrees(adjacency), [adjacency]);

  const fiedlerVector = useMemo(() => {
    if (n <= 1) return [0];
    return eigenResult.eigenvectors[1];
  }, [eigenResult, n]);

  const fiedlerValue = useMemo(() => {
    if (n <= 1) return 0;
    return eigenResult.eigenvalues[1];
  }, [eigenResult, n]);

  const numComponents = useMemo(
    () => countComponents(eigenResult.eigenvalues),
    [eigenResult],
  );

  // ─── Layout calculations ───

  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const graphPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    if (isNarrow) return containerWidth - 16;
    // When matrices are visible, graph takes ~55% of space
    return showMatrices
      ? Math.floor((containerWidth - 32) * 0.55)
      : containerWidth - 16;
  }, [containerWidth, isNarrow, showMatrices]);

  const matrixPanelWidth = useMemo(() => {
    if (!containerWidth) return 260;
    if (isNarrow) return containerWidth - 16;
    return Math.floor((containerWidth - 32) * 0.45);
  }, [containerWidth, isNarrow]);

  const eigenPanelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    return containerWidth - 16;
  }, [containerWidth]);

  // ─── Color scales ───

  const fiedlerColorScale = useMemo(() => {
    if (fiedlerVector.length === 0) return () => '#888';
    const ext = d3.extent(fiedlerVector) as [number, number];
    const absMax = Math.max(Math.abs(ext[0]), Math.abs(ext[1])) || 1;
    return d3.scaleDiverging(d3.interpolateRdBu).domain([absMax, 0, -absMax]);
  }, [fiedlerVector]);

  const degreeColorScale = useMemo(() => {
    const maxDeg = d3.max(deg) || 1;
    return d3.scaleSequential(d3.interpolateBlues).domain([0, maxDeg]);
  }, [deg]);

  // ─── Force simulation management ───

  const initSimulation = useCallback(
    (nodeCount: number, adj: number[][]) => {
      // Stop previous simulation
      if (simRef.current) simRef.current.stop();

      const nodes: { x: number; y: number }[] = Array.from(
        { length: nodeCount },
        (_, i) => ({
          x: graphPanelWidth / 2 + (Math.cos((2 * Math.PI * i) / nodeCount)) * 80,
          y: GRAPH_PANEL_HEIGHT / 2 + (Math.sin((2 * Math.PI * i) / nodeCount)) * 80,
        }),
      );
      positionsRef.current = nodes;

      const links = edgesFromAdj(adj);

      const sim = d3
        .forceSimulation(nodes)
        .force(
          'link',
          d3
            .forceLink<{ x: number; y: number }, { source: number; target: number }>(links)
            .id((_, i) => i)
            .distance(60),
        )
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(graphPanelWidth / 2, GRAPH_PANEL_HEIGHT / 2))
        .force('collide', d3.forceCollide(20))
        .alphaDecay(0.03)
        .on('tick', () => {
          // Clamp positions to panel bounds
          const pad = 24;
          for (const node of nodes) {
            node.x = Math.max(pad, Math.min(graphPanelWidth - pad, node.x));
            node.y = Math.max(pad, Math.min(GRAPH_PANEL_HEIGHT - pad, node.y));
          }
          setTick((t) => t + 1);
        });

      simRef.current = sim;
    },
    [graphPanelWidth],
  );

  // Re-initialize simulation when adjacency changes structurally
  // (i.e., node count changes or preset is loaded).
  const prevNodeCount = useRef(n);
  const prevPreset = useRef(presetIdx);

  useEffect(() => {
    const needsReinit =
      prevNodeCount.current !== n ||
      prevPreset.current !== presetIdx ||
      positionsRef.current.length !== n;

    if (needsReinit) {
      initSimulation(n, adjacency);
      prevNodeCount.current = n;
      prevPreset.current = presetIdx;
    } else {
      // Just update links in existing simulation
      if (simRef.current) {
        const links = edgesFromAdj(adjacency);
        (simRef.current.force('link') as d3.ForceLink<{ x: number; y: number }, { source: number; target: number }>)
          .links(links);
        simRef.current.alpha(0.3).restart();
      }
    }

    return () => {
      if (simRef.current) simRef.current.stop();
    };
  }, [adjacency, n, presetIdx, initSimulation]);

  // Initial mount
  useEffect(() => {
    initSimulation(n, adjacency);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── Interaction handlers ───

  const handlePresetChange = useCallback(
    (idx: number) => {
      setPresetIdx(idx);
      setSelectedNode(null);
      setWarning(null);
      const g = PRESETS[idx].build();
      setAdjacency(g.adjacency);
    },
    [],
  );

  const handleAddNode = useCallback(
    (x: number, y: number) => {
      if (n >= MAX_NODES) {
        setWarning(`Maximum of ${MAX_NODES} nodes reached.`);
        return;
      }
      setWarning(null);
      setAdjacency((prev) => {
        const newN = prev.length + 1;
        const newAdj = prev.map((row) => [...row, 0]);
        newAdj.push(new Array(newN).fill(0));
        return newAdj;
      });
      // Add a node position
      positionsRef.current.push({ x, y });
    },
    [n],
  );

  const handleDeleteNode = useCallback(
    (nodeIdx: number) => {
      if (n <= 1) return;
      setSelectedNode(null);
      setWarning(null);
      setAdjacency((prev) => {
        const newAdj = prev
          .filter((_, i) => i !== nodeIdx)
          .map((row) => row.filter((_, j) => j !== nodeIdx));
        return newAdj;
      });
      positionsRef.current.splice(nodeIdx, 1);
    },
    [n],
  );

  const handleToggleEdge = useCallback(
    (i: number, j: number) => {
      if (i === j) return;
      setWarning(null);
      setAdjacency((prev) => {
        const newAdj = cloneAdj(prev);
        const val = newAdj[i][j] === 0 ? 1 : 0;
        newAdj[i][j] = val;
        newAdj[j][i] = val;
        return newAdj;
      });
    },
    [],
  );

  const handleNodeClick = useCallback(
    (nodeIdx: number) => {
      if (selectedNode === null) {
        setSelectedNode(nodeIdx);
      } else if (selectedNode === nodeIdx) {
        setSelectedNode(null);
      } else {
        handleToggleEdge(selectedNode, nodeIdx);
        setSelectedNode(null);
      }
    },
    [selectedNode, handleToggleEdge],
  );

  // ─── Graph panel rendering ───

  useEffect(() => {
    if (!graphSvgRef.current || graphPanelWidth === 0) return;

    const svg = d3.select(graphSvgRef.current);
    svg.selectAll('*').remove();

    const positions = positionsRef.current;
    if (positions.length !== n) return;

    const edges = edgesFromAdj(adjacency);

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
      .style('stroke-width', '2')
      .style('stroke-opacity', '0.7');

    // Determine node fill
    const nodeFill = (i: number): string => {
      if (coloringMode === 'fiedler' && fiedlerVector.length > i) {
        return fiedlerColorScale(fiedlerVector[i]);
      }
      if (coloringMode === 'degree') {
        return degreeColorScale(deg[i]);
      }
      return 'var(--color-text-secondary)';
    };

    // Draw nodes
    const nodeG = svg.append('g');
    const nodeCircles = nodeG
      .selectAll<SVGCircleElement, number>('circle')
      .data(d3.range(n))
      .join('circle')
      .attr('cx', (i) => positions[i]?.x ?? 0)
      .attr('cy', (i) => positions[i]?.y ?? 0)
      .attr('r', 14)
      .attr('fill', (i) => nodeFill(i))
      .style('stroke', (i) =>
        i === selectedNode ? '#f59e0b' : 'var(--color-text)',
      )
      .style('stroke-width', (i) => (i === selectedNode ? '3' : '1.5'))
      .style('cursor', 'pointer');

    // Node labels
    nodeG
      .selectAll('text')
      .data(d3.range(n))
      .join('text')
      .attr('x', (i) => positions[i]?.x ?? 0)
      .attr('y', (i) => (positions[i]?.y ?? 0) + 4.5)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('pointer-events', 'none')
      .style('fill', (i) => {
        if (coloringMode === 'none') return 'var(--color-bg)';
        // Dark text on light backgrounds, light text on dark
        if (coloringMode === 'fiedler' && fiedlerVector.length > i) {
          return Math.abs(fiedlerVector[i]) < 0.3 ? 'var(--color-text)' : 'white';
        }
        return deg[i] > (d3.max(deg) || 1) * 0.5 ? 'white' : 'var(--color-text)';
      })
      .text((i) => i);

    // ─── Drag behavior ───

    const drag = d3
      .drag<SVGCircleElement, number>()
      .on('start', function (event, i) {
        if (simRef.current) {
          simRef.current.alphaTarget(0.3).restart();
        }
        const pos = positions[i];
        if (pos) {
          (pos as any).fx = pos.x;
          (pos as any).fy = pos.y;
        }
      })
      .on('drag', function (event, i) {
        const pos = positions[i];
        if (pos) {
          (pos as any).fx = event.x;
          (pos as any).fy = event.y;
        }
      })
      .on('end', function (event, i) {
        if (simRef.current) {
          simRef.current.alphaTarget(0);
        }
        const pos = positions[i];
        if (pos) {
          (pos as any).fx = null;
          (pos as any).fy = null;
        }
      });

    nodeCircles.call(drag);

    // Click handler for nodes (edge toggling / selection)
    nodeCircles.on('click', function (event, i) {
      event.stopPropagation();
      handleNodeClick(i);
    });

    // Double-click to delete node
    nodeCircles.on('dblclick', function (event, i) {
      event.stopPropagation();
      event.preventDefault();
      handleDeleteNode(i);
    });

    // Click empty space to add a node
    svg.on('click', function (event) {
      const [mx, my] = d3.pointer(event);
      handleAddNode(mx, my);
    });

    // Panel label
    svg
      .append('text')
      .attr('x', graphPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Graph (click to add node, click two nodes to toggle edge)');
  }, [
    adjacency, n, graphPanelWidth, coloringMode, fiedlerVector,
    fiedlerColorScale, degreeColorScale, deg, selectedNode,
    tick, handleNodeClick, handleDeleteNode, handleAddNode,
  ]);

  // ─── Matrix panel rendering ───

  useEffect(() => {
    if (!matrixSvgRef.current || !showMatrices || matrixPanelWidth === 0) return;

    const svg = d3.select(matrixSvgRef.current);
    svg.selectAll('*').remove();

    const showValues = n <= 8;
    const cellSize = Math.min(MATRIX_CELL, Math.floor((matrixPanelWidth - 60) / n));

    const matrices: { label: string; data: number[][]; colorDomain: [number, number] }[] = [
      {
        label: 'A (adjacency)',
        data: adjacency,
        colorDomain: [0, 1],
      },
      {
        label: 'D (degree)',
        data: D,
        colorDomain: [0, d3.max(deg) || 1],
      },
      {
        label: 'L = D − A (Laplacian)',
        data: L,
        colorDomain: [
          d3.min(L.flat()) as number,
          d3.max(L.flat()) as number,
        ],
      },
    ];

    let yOffset = 8;

    for (const { label, data, colorDomain } of matrices) {
      const matrixWidth = cellSize * n;
      const matrixHeight = cellSize * n;
      const xStart = Math.max(8, (matrixPanelWidth - matrixWidth) / 2);

      // Label
      svg
        .append('text')
        .attr('x', matrixPanelWidth / 2)
        .attr('y', yOffset + 14)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .style('fill', 'var(--color-text-secondary)')
        .text(label);

      yOffset += 22;

      // Color scale
      const colorScale = d3
        .scaleSequential(d3.interpolateYlOrRd)
        .domain(colorDomain);

      const matG = svg.append('g').attr('transform', `translate(${xStart}, ${yOffset})`);

      // Cells
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const val = data[i][j];
          matG
            .append('rect')
            .attr('x', j * cellSize)
            .attr('y', i * cellSize)
            .attr('width', cellSize - 1)
            .attr('height', cellSize - 1)
            .attr('rx', 2)
            .attr('fill', colorScale(val))
            .style('stroke', 'var(--color-border)')
            .style('stroke-width', '0.5');

          if (showValues) {
            matG
              .append('text')
              .attr('x', j * cellSize + cellSize / 2)
              .attr('y', i * cellSize + cellSize / 2 + 3.5)
              .attr('text-anchor', 'middle')
              .attr('font-size', Math.min(10, cellSize * 0.4))
              .attr('font-weight', 500)
              .style('fill', val > (colorDomain[1] - colorDomain[0]) * 0.6 + colorDomain[0] ? 'white' : 'var(--color-text)')
              .text(Number.isInteger(val) ? val : val.toFixed(1));
          }
        }
      }

      yOffset += matrixHeight + 12;
    }
  }, [adjacency, D, L, n, deg, showMatrices, matrixPanelWidth]);

  // ─── Eigenvalue bar chart rendering ───

  useEffect(() => {
    if (!eigenSvgRef.current || eigenPanelWidth === 0) return;

    const svg = d3.select(eigenSvgRef.current);
    svg.selectAll('*').remove();

    const innerW = eigenPanelWidth - MARGIN.left - MARGIN.right;
    const innerH = EIGEN_BAR_HEIGHT - MARGIN.top - MARGIN.bottom;

    const eigenvals = eigenResult.eigenvalues;
    const maxVal = (d3.max(eigenvals) as number) || 1;

    const xScale = d3
      .scaleLinear()
      .domain([0, maxVal * 1.1])
      .range([0, innerW]);

    const yScale = d3
      .scaleBand<number>()
      .domain(d3.range(eigenvals.length))
      .range([0, innerH])
      .padding(0.15);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Title
    svg
      .append('text')
      .attr('x', eigenPanelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text-secondary)')
      .text('Laplacian eigenvalues');

    // Bars
    g.selectAll('rect')
      .data(eigenvals)
      .join('rect')
      .attr('x', 0)
      .attr('y', (_, i) => yScale(i)!)
      .attr('width', (d) => Math.max(1, xScale(Math.max(0, d))))
      .attr('height', yScale.bandwidth())
      .attr('rx', 3)
      .attr('fill', (_, i) => (i === 1 ? '#f59e0b' : '#3b82f6'))
      .style('opacity', (_, i) => (i === 1 ? '1' : '0.7'));

    // Bar labels
    g.selectAll<SVGTextElement, number>('.bar-label')
      .data(eigenvals)
      .join('text')
      .attr('class', 'bar-label')
      .attr('x', (d) => Math.max(1, xScale(Math.max(0, d))) + 4)
      .attr('y', (_, i) => (yScale(i)!) + yScale.bandwidth() / 2 + 4)
      .attr('font-size', 10)
      .style('fill', 'var(--color-text)')
      .text((d, i) => `λ${sub(i + 1)} = ${d.toFixed(3)}`);

    // Y-axis labels
    g.selectAll<SVGTextElement, number>('.y-label')
      .data(eigenvals)
      .join('text')
      .attr('class', 'y-label')
      .attr('x', -6)
      .attr('y', (_, i) => (yScale(i)!) + yScale.bandwidth() / 2 + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', 10)
      .style('fill', 'var(--color-text-secondary)')
      .text((_, i) => `λ${sub(i + 1)}`);

    // X-axis
    g.append('g')
      .attr('transform', `translate(0, ${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .attr('font-size', 10);

    g.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Annotation: algebraic connectivity
    const annotationY = innerH + 28;
    const connected = numComponents === 1;
    g.append('text')
      .attr('x', 0)
      .attr('y', annotationY)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .style('fill', 'var(--color-text)')
      .text(
        `Algebraic connectivity λ₂ = ${fiedlerValue.toFixed(4)}  ·  ` +
        `${numComponents} component${numComponents > 1 ? 's' : ''}  ·  ` +
        (connected ? 'Connected' : 'Disconnected'),
      );
  }, [eigenResult, eigenPanelWidth, numComponents, fiedlerValue]);

  // ─── Total matrix panel SVG height ───

  const matrixSvgHeight = useMemo(() => {
    if (!showMatrices) return 0;
    const cellSize = Math.min(MATRIX_CELL, Math.floor((matrixPanelWidth - 60) / n));
    // 3 matrices, each has label (22px) + grid (cellSize*n) + gap (12px)
    return 3 * (22 + cellSize * n + 12) + 8;
  }, [showMatrices, matrixPanelWidth, n]);

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
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {/* Preset dropdown */}
        <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Preset:</span>
          <select
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              backgroundColor: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
            value={presetIdx}
            onChange={(e) => handlePresetChange(Number(e.target.value))}
          >
            {PRESETS.map((p, i) => (
              <option key={p.label} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>

        {/* Coloring mode */}
        <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Color:</span>
          <select
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              backgroundColor: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
            value={coloringMode}
            onChange={(e) => setColoringMode(e.target.value as ColoringMode)}
          >
            <option value="fiedler">Fiedler vector</option>
            <option value="degree">Degree</option>
            <option value="none">None</option>
          </select>
        </label>

        {/* Matrix toggle */}
        <button
          className="rounded border px-3 py-1 text-sm font-medium"
          style={{
            borderColor: 'var(--color-border)',
            backgroundColor: showMatrices ? 'var(--color-text)' : 'var(--color-bg)',
            color: showMatrices ? 'var(--color-bg)' : 'var(--color-text)',
          }}
          onClick={() => setShowMatrices((v) => !v)}
        >
          {showMatrices ? 'Hide matrices' : 'Show matrices'}
        </button>
      </div>

      {/* Warning */}
      {warning && (
        <div
          className="mb-3 rounded px-3 py-1.5 text-sm font-medium"
          style={{ backgroundColor: '#fef3c7', color: '#92400e' }}
        >
          {warning}
        </div>
      )}

      {/* Main panels */}
      <div
        className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}
        style={{ alignItems: 'flex-start' }}
      >
        {/* Left: force-directed graph */}
        <div style={{ flexShrink: 0 }}>
          <svg
            ref={graphSvgRef}
            width={graphPanelWidth}
            height={GRAPH_PANEL_HEIGHT}
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: 8,
              backgroundColor: 'var(--color-bg)',
            }}
          />
        </div>

        {/* Right: matrices */}
        {showMatrices && (
          <div style={{ flexShrink: 1, minWidth: 0, overflow: 'auto' }}>
            <svg
              ref={matrixSvgRef}
              width={matrixPanelWidth}
              height={matrixSvgHeight}
              style={{
                border: '1px solid var(--color-border)',
                borderRadius: 8,
                backgroundColor: 'var(--color-bg)',
              }}
            />
          </div>
        )}
      </div>

      {/* Bottom: eigenvalue bar chart */}
      <div className="mt-4">
        <svg
          ref={eigenSvgRef}
          width={eigenPanelWidth}
          height={EIGEN_BAR_HEIGHT}
          style={{
            border: '1px solid var(--color-border)',
            borderRadius: 8,
            backgroundColor: 'var(--color-bg)',
          }}
        />
      </div>

      {/* Legend */}
      <div
        className="mt-3 flex flex-wrap gap-4 text-xs"
        style={{ color: 'var(--color-text-secondary)' }}
      >
        <span><strong>Click</strong> empty space: add node</span>
        <span><strong>Click</strong> node: select for edge toggle</span>
        <span><strong>Click</strong> second node: toggle edge</span>
        <span><strong>Double-click</strong> node: delete</span>
        <span><strong>Drag</strong> node: reposition</span>
      </div>
    </div>
  );
}

// ─── Subscript helper ───

function sub(n: number): string {
  const subs = '₀₁₂₃₄₅₆₇₈₉';
  return String(n)
    .split('')
    .map((c) => subs[parseInt(c, 10)] ?? c)
    .join('');
}

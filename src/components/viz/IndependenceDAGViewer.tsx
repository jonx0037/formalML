import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteNB } from './shared/naive-bayes';

// =============================================================================
// IndependenceDAGViewer — §2.3 companion. Three Bayesian-network structures:
//
//   "naive"  Y → X_1, Y → X_2, ..., Y → X_d   (d edges)
//   "chain"  Y → X_1 → X_2 → ... → X_d        (d edges, but features depend
//            sequentially — TAN-style structure)
//   "full"   Y → X_i for all i AND X_i → X_j for all j > i  (the joint)
//
// Reader chooses structure and feature count d ∈ [3, 8]; the viewer displays
// the DAG and the parameter count (for binary features and a binary Y).
//
// Parameter counts (binary features, binary Y):
//   naive  = 2d + 1   (d means + d variances, plus class prior)
//   chain  = 3d - 1   (each feature has 2 parents → 4 conditionals minus root)
//   full   = 2 · (2^d − 1) + 1   (full joint table per class)
// =============================================================================

const STRUCTURES = ['naive', 'chain', 'full'] as const;
type Structure = (typeof STRUCTURES)[number];

const STRUCTURE_LABELS: Record<Structure, string> = {
  naive: 'Naive Bayes (Y ⊥ X_i given other Xs)',
  chain: 'Chain / TAN-like (Y → X_1 → X_2 → ...)',
  full: 'Full joint (every X_i connected)',
};

function paramCount(structure: Structure, d: number): number {
  switch (structure) {
    case 'naive':
      return 2 * d + 1;
    case 'chain':
      return 3 * d - 1;
    case 'full':
      return 2 * (Math.pow(2, d) - 1) + 1;
  }
}

interface Node {
  id: string;
  label: string;
  cx: number;
  cy: number;
  isClass: boolean;
}

interface Edge {
  source: string;
  target: string;
}

function buildLayout(structure: Structure, d: number, width: number, height: number) {
  const cy = height * 0.32;
  const fy = height * 0.74;
  const yNode: Node = { id: 'Y', label: 'Y', cx: width / 2, cy, isClass: true };
  const featureNodes: Node[] = [];
  for (let i = 0; i < d; i++) {
    const x = width * (0.1 + (0.8 * i) / Math.max(1, d - 1));
    featureNodes.push({ id: `X${i + 1}`, label: `X_${i + 1}`, cx: x, cy: fy, isClass: false });
  }
  const edges: Edge[] = [];
  if (structure === 'naive') {
    for (let i = 0; i < d; i++) edges.push({ source: 'Y', target: `X${i + 1}` });
  } else if (structure === 'chain') {
    edges.push({ source: 'Y', target: 'X1' });
    for (let i = 0; i < d - 1; i++) edges.push({ source: `X${i + 1}`, target: `X${i + 2}` });
  } else {
    for (let i = 0; i < d; i++) edges.push({ source: 'Y', target: `X${i + 1}` });
    for (let i = 0; i < d; i++) {
      for (let j = i + 1; j < d; j++) edges.push({ source: `X${i + 1}`, target: `X${j + 1}` });
    }
  }
  return { nodes: [yNode, ...featureNodes], edges };
}

export default function IndependenceDAGViewer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [structure, setStructure] = useState<Structure>('naive');
  const [d, setD] = useState<number>(5);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 320;
  const layout = useMemo(() => buildLayout(structure, d, W, H), [structure, d, W, H]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const nodeById = new Map(layout.nodes.map((n) => [n.id, n]));
      // Arrowhead marker.
      const defs = svg.append('defs');
      defs
        .append('marker')
        .attr('id', 'arr-nb')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 24)
        .attr('refY', 0)
        .attr('markerWidth', 7)
        .attr('markerHeight', 7)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-4L8,0L0,4')
        .style('fill', 'currentColor');

      const edgeG = svg.append('g').attr('class', 'edges');
      edgeG
        .selectAll('line')
        .data(layout.edges)
        .join('line')
        .attr('x1', (e) => nodeById.get(e.source)!.cx)
        .attr('y1', (e) => nodeById.get(e.source)!.cy)
        .attr('x2', (e) => nodeById.get(e.target)!.cx)
        .attr('y2', (e) => nodeById.get(e.target)!.cy)
        .attr('stroke-width', 1.4)
        .attr('opacity', 0.7)
        .attr('marker-end', 'url(#arr-nb)')
        .style('stroke', paletteNB.data);

      const nodeG = svg.append('g').attr('class', 'nodes');
      const gs = nodeG.selectAll('g').data(layout.nodes).join('g');
      gs.append('circle')
        .attr('cx', (n) => n.cx)
        .attr('cy', (n) => n.cy)
        .attr('r', 18)
        .attr('stroke-width', 1.5)
        .style('fill', (n) => (n.isClass ? paletteNB.highlight : paletteNB.pos))
        .style('stroke', (n) => (n.isClass ? paletteNB.highlight : paletteNB.pos))
        .style('fill-opacity', 0.18);
      gs.append('text')
        .attr('x', (n) => n.cx)
        .attr('y', (n) => n.cy)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('font-size', 13)
        .style('fill', 'currentColor')
        .text((n) => n.label);
    },
    [layout, W, H],
  );

  const params = paramCount(structure, d);

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="flex flex-wrap items-center gap-3 mb-2 text-sm">
        <label className="flex items-center gap-2">
          Structure:
          <select
            value={structure}
            onChange={(e) => setStructure(e.target.value as Structure)}
            className="rounded border px-2 py-1 bg-transparent"
            aria-label="Bayesian network structure"
          >
            {STRUCTURES.map((s) => (
              <option key={s} value={s}>
                {STRUCTURE_LABELS[s]}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          Feature count d = {d}
          <input
            type="range"
            min={3}
            max={8}
            step={1}
            value={d}
            onChange={(e) => setD(Number(e.target.value))}
            aria-label="Number of features"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label={`${STRUCTURE_LABELS[structure]} DAG`}/>
      <div className="text-xs opacity-75 mt-1 text-center">
        Parameters to specify (binary features, binary class): <strong>{params.toLocaleString()}</strong>.
        At d = {d}, the full joint requires {paramCount('full', d).toLocaleString()} parameters;
        the naive factorization requires only {paramCount('naive', d).toLocaleString()}.
      </div>
    </div>
  );
}

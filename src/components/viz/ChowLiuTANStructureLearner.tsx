import { useState, useMemo } from 'react';
import {
  paletteNB,
  mulberry32,
  chowLiuTree,
} from './shared/naive-bayes';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

// =============================================================================
// ChowLiuTANStructureLearner — §9.1 companion. A small (d = 5–7) discrete-feature
// dataset with planted dependencies; the reader chooses which features are
// near-duplicates of which (the dependency probability slider), and watches the
// Chow–Liu maximum-mutual-information spanning tree adapt. The viz overlays the
// learned tree on the NB DAG, making explicit which dependencies TAN restores
// that NB ignores.
// =============================================================================

const D = 6;
const N = 500;
const CATS = 3;

type DepPair = [number, number];

const PLANTED_PAIRS: DepPair[] = [
  [0, 1],
  [2, 3],
];

function generateData(depProb: number, seed: number): Uint8Array {
  const rng = mulberry32(seed);
  const X = new Uint8Array(N * D);
  // Independent random discrete features.
  for (let i = 0; i < N; i++) for (let j = 0; j < D; j++) X[i * D + j] = Math.floor(rng() * CATS);
  // Inject the planted dependencies (with probability depProb feature q copies feature p).
  for (const [p, q] of PLANTED_PAIRS) {
    for (let i = 0; i < N; i++) {
      if (rng() < depProb) X[i * D + q] = X[i * D + p];
    }
  }
  return X;
}

export default function ChowLiuTANStructureLearner() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [depProb, setDepProb] = useState(0.8);
  const [seed, setSeed] = useState(42);

  const data = useMemo(() => generateData(depProb, seed), [depProb, seed]);
  const result = useMemo(() => {
    const Xnum = new Float64Array(N * D);
    for (let k = 0; k < N * D; k++) Xnum[k] = data[k];
    return chowLiuTree(Xnum, N, D, 1.0);
  }, [data]);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 360;
  const cy = H * 0.32;
  const fy = H * 0.74;

  const nodes = useMemo(() => {
    const arr = [{ id: 'Y', cx: W / 2, cy, label: 'Y' }];
    for (let i = 0; i < D; i++) {
      const x = W * (0.08 + (0.84 * i) / Math.max(1, D - 1));
      arr.push({ id: `X${i}`, cx: x, cy: fy, label: `X_${i}` });
    }
    return arr;
  }, [W, cy, fy]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const nodeById = new Map(nodes.map((n) => [n.id, n]));

      const defs = svg.append('defs');
      defs
        .append('marker')
        .attr('id', 'arr-cl')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 22)
        .attr('refY', 0)
        .attr('markerWidth', 7)
        .attr('markerHeight', 7)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-4L8,0L0,4')
        .style('fill', 'currentColor');

      // NB edges (Y → X_i).
      const nbEdges = svg.append('g').attr('class', 'nb-edges');
      for (let i = 0; i < D; i++) {
        const t = nodeById.get(`X${i}`)!;
        nbEdges
          .append('line')
          .attr('x1', W / 2)
          .attr('y1', cy)
          .attr('x2', t.cx)
          .attr('y2', t.cy)
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '3 3')
          .attr('opacity', 0.45)
          .attr('marker-end', 'url(#arr-cl)')
          .style('stroke', paletteNB.data);
      }

      // Chow–Liu edges (X_i — X_j).
      const clEdges = svg.append('g').attr('class', 'cl-edges');
      result.edges.forEach(([i, j]) => {
        const a = nodeById.get(`X${i}`)!;
        const b = nodeById.get(`X${j}`)!;
        const mi = result.miMatrix[i * D + j];
        const thickness = Math.max(1.5, Math.min(5, 1.5 + mi * 6));
        clEdges
          .append('line')
          .attr('x1', a.cx)
          .attr('y1', a.cy)
          .attr('x2', b.cx)
          .attr('y2', b.cy)
          .attr('stroke-width', thickness)
          .attr('opacity', 0.85)
          .style('stroke', paletteNB.highlight);
        clEdges
          .append('text')
          .attr('x', (a.cx + b.cx) / 2)
          .attr('y', (a.cy + b.cy) / 2 - 6)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .style('fill', paletteNB.highlight)
          .text(`MI = ${mi.toFixed(2)}`);
      });

      // Nodes.
      const gs = svg.append('g').selectAll('g').data(nodes).join('g');
      gs.append('circle')
        .attr('cx', (n) => n.cx)
        .attr('cy', (n) => n.cy)
        .attr('r', 16)
        .attr('stroke-width', 1.5)
        .style('fill', (n) => (n.id === 'Y' ? paletteNB.highlight : paletteNB.pos))
        .style('stroke', (n) => (n.id === 'Y' ? paletteNB.highlight : paletteNB.pos))
        .style('fill-opacity', 0.2);
      gs.append('text')
        .attr('x', (n) => n.cx)
        .attr('y', (n) => n.cy)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('font-size', 12)
        .style('fill', 'currentColor')
        .text((n) => n.label);
    },
    [nodes, result, W, H, cy],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="flex flex-wrap items-center gap-3 mb-2 text-sm">
        <label className="flex items-center gap-2">
          Planted-dependency probability = {depProb.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.02}
            value={depProb}
            onChange={(e) => setDepProb(Number(e.target.value))}
            aria-label="Planted dependency probability"
          />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="ml-auto rounded border px-2 py-1 text-xs"
          aria-label="Resample data"
        >
          Resample (seed {seed})
        </button>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Chow-Liu maximum-mutual-information spanning tree"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        Dashed gray = NB's assumed independence (Y → X_i only). Solid orange = the Chow–Liu
        tree (edge thickness ∝ pairwise MI). Planted dependencies: X_0–X_1 and X_2–X_3 with
        probability {depProb.toFixed(2)}; the tree should recover these as high-MI edges.
      </div>
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  synthSinusoidWithWiggle,
  fitThreeCandidatesLOO,
  stackingObjective,
  stackingObjectiveOnSimplex3,
  fitStackingWeights,
  paletteStacking,
} from './shared/bayesian-ml-stacking';

// =============================================================================
// StackingSimplexExplorer — embedded alongside §3.4.
// Renders the stacking objective as a heatmap on the 2-simplex of three
// closed-form candidates (Linear, Polynomial-degree-2, GP-fixed-hypers). The
// reader drags a weight point on the simplex; a side panel shows the per-x
// predictive uncertainty contributed by each candidate at the chosen weights.
// "Snap to optimum" animates the dragged point to w*. As n grows, the contour
// lines tighten around the interior optimum, but the optimum itself is stable —
// stacking does not collapse onto a single candidate.
// =============================================================================

const PANEL_HEIGHT = 380;
const N_OPTIONS = [40, 60, 80, 120, 160, 240, 320] as const;
const SIM_RES = 40; // 40 → ~820 grid points
const SIDE = 360;

// Convert barycentric (w1, w2, w3) with w1+w2+w3 = 1 to Cartesian on the unit
// equilateral triangle. Vertex 1 = bottom-left, vertex 2 = bottom-right, vertex 3 = top.
function baryToCartesian(w1: number, w2: number, w3: number): [number, number] {
  const xv1 = 0;
  const yv1 = 0;
  const xv2 = 1;
  const yv2 = 0;
  const xv3 = 0.5;
  const yv3 = Math.sqrt(3) / 2;
  return [w1 * xv1 + w2 * xv2 + w3 * xv3, w1 * yv1 + w2 * yv2 + w3 * yv3];
}

function cartesianToBary(x: number, y: number): [number, number, number] {
  // Inverse of baryToCartesian on the unit equilateral triangle.
  const w3 = y / (Math.sqrt(3) / 2);
  const w2 = x - 0.5 * w3;
  const w1 = 1 - w2 - w3;
  return [w1, w2, w3];
}

export default function StackingSimplexExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [n, setN] = useState<number>(80);
  const [seed, setSeed] = useState<number>(20260430);
  const [w, setW] = useState<[number, number, number]>([1 / 3, 1 / 3, 1 / 3]);

  const data = useMemo(() => {
    const rng = mulberry32(seed);
    return synthSinusoidWithWiggle(n, 0.25, rng);
  }, [n, seed]);

  const looMatrix = useMemo(() => fitThreeCandidatesLOO(data.x, data.y, 2), [data]);

  const grid = useMemo(() => stackingObjectiveOnSimplex3(looMatrix.L, looMatrix.n, SIM_RES), [looMatrix]);

  const optWeights = useMemo<[number, number, number]>(() => {
    const result = fitStackingWeights(looMatrix.L, looMatrix.n, looMatrix.K);
    return [result.weights[0], result.weights[1], result.weights[2]];
  }, [looMatrix]);

  const currentObj = useMemo(
    () => stackingObjective(new Float64Array(w), looMatrix.L, looMatrix.n, looMatrix.K),
    [w, looMatrix],
  );

  const ref = useD3(
    (svg) => {
      const panelW = width || 720;
      const height = PANEL_HEIGHT;
      svg.attr('width', panelW).attr('height', height);
      svg.selectAll('*').remove();

      const triSize = Math.min(panelW - SIDE - 40, height - 40);
      const margin = { top: 20, left: 20 };

      const sx = (cx: number) => margin.left + cx * triSize;
      const sy = (cy: number) => margin.top + (1 - cy / (Math.sqrt(3) / 2)) * triSize;

      const root = svg.append('g');

      // Triangle outline.
      const v1 = baryToCartesian(1, 0, 0);
      const v2 = baryToCartesian(0, 1, 0);
      const v3 = baryToCartesian(0, 0, 1);
      root
        .append('path')
        .attr('d', `M ${sx(v1[0])},${sy(v1[1])} L ${sx(v2[0])},${sy(v2[1])} L ${sx(v3[0])},${sy(v3[1])} Z`)
        .attr('fill', '#fff')
        .attr('stroke', '#444')
        .attr('stroke-width', 1.2);

      // Heatmap via small circles at each grid point.
      const minObj = d3.min(grid.values) as number;
      const maxObj = d3.max(grid.values) as number;
      const colorScale = d3
        .scaleSequential(d3.interpolateViridis)
        .domain([minObj, maxObj]);

      for (let i = 0; i <= SIM_RES; i++) {
        for (let j = 0; j <= SIM_RES - i; j++) {
          const w2 = i / SIM_RES;
          const w3 = j / SIM_RES;
          const w1 = 1 - w2 - w3;
          const [cx, cy] = baryToCartesian(w1, w2, w3);
          // Index using the same gridIndex as the math utility.
          let off = 0;
          for (let k = 0; k < i; k++) off += SIM_RES - k + 1;
          const idx = off + j;
          const v = grid.values[idx];
          root
            .append('circle')
            .attr('cx', sx(cx))
            .attr('cy', sy(cy))
            .attr('r', 4)
            .attr('fill', colorScale(v))
            .attr('opacity', 0.9);
        }
      }

      // Vertex labels.
      const vertexLabels = [
        { pos: v1, label: 'Linear', color: paletteStacking.blr, dx: -8, dy: 18 },
        { pos: v2, label: 'Poly-2', color: paletteStacking.bpr, dx: 8, dy: 18 },
        { pos: v3, label: 'GP', color: paletteStacking.gp, dx: 0, dy: -8 },
      ];
      vertexLabels.forEach(({ pos, label, color, dx, dy }) => {
        root
          .append('text')
          .attr('x', sx(pos[0]) + dx)
          .attr('y', sy(pos[1]) + dy)
          .attr('text-anchor', 'middle')
          .attr('font-size', 11)
          .attr('font-weight', 600)
          .attr('fill', color)
          .text(label);
      });

      // Optimum marker.
      const [oxC, oyC] = baryToCartesian(optWeights[0], optWeights[1], optWeights[2]);
      root
        .append('circle')
        .attr('cx', sx(oxC))
        .attr('cy', sy(oyC))
        .attr('r', 7)
        .attr('fill', 'none')
        .attr('stroke', '#000')
        .attr('stroke-width', 1.6);
      root
        .append('text')
        .attr('x', sx(oxC) + 10)
        .attr('y', sy(oyC) - 6)
        .attr('font-size', 10)
        .text('w*');

      // Current weight marker.
      const [wxC, wyC] = baryToCartesian(w[0], w[1], w[2]);
      const dragHandle = root
        .append('circle')
        .attr('cx', sx(wxC))
        .attr('cy', sy(wyC))
        .attr('r', 8)
        .attr('fill', '#fff')
        .attr('stroke', '#d62728')
        .attr('stroke-width', 2.4)
        .style('cursor', 'grab');

      const drag = d3
        .drag<SVGCircleElement, unknown>()
        .on('drag', (event) => {
          const cx = (event.x - margin.left) / triSize;
          const cy = (1 - (event.y - margin.top) / triSize) * (Math.sqrt(3) / 2);
          const [w1n, w2n, w3n] = cartesianToBary(cx, cy);
          if (w1n >= 0 && w2n >= 0 && w3n >= 0 && w1n <= 1 && w2n <= 1 && w3n <= 1) {
            setW([w1n, w2n, w3n]);
          }
        });
      dragHandle.call(drag as any);

      // Side panel: weight readout.
      const side = svg.append('g').attr('transform', `translate(${triSize + 60},20)`);
      side.append('text').attr('font-size', 12).attr('font-weight', 600).text('Current weights');
      const ws: [string, number, string][] = [
        ['Linear (BLR)', w[0], paletteStacking.blr],
        ['Polynomial-2 (BPR)', w[1], paletteStacking.bpr],
        ['GP', w[2], paletteStacking.gp],
      ];
      ws.forEach(([label, value, col], i) => {
        side
          .append('text')
          .attr('x', 0)
          .attr('y', 22 + i * 16)
          .attr('font-size', 11)
          .attr('fill', col)
          .text(`${label}: ${value.toFixed(3)}`);
      });
      side
        .append('text')
        .attr('x', 0)
        .attr('y', 22 + 4 * 16)
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .text(`S_n(w) = ${currentObj.toFixed(4)}`);
      side
        .append('text')
        .attr('x', 0)
        .attr('y', 22 + 5 * 16)
        .attr('font-size', 11)
        .text(`max S_n  = ${grid.maxValue.toFixed(4)}  (at w*)`);
      side
        .append('text')
        .attr('x', 0)
        .attr('y', 22 + 7 * 16)
        .attr('font-size', 10)
        .attr('fill', '#444')
        .text('Drag the white circle to vary w.');
      side
        .append('text')
        .attr('x', 0)
        .attr('y', 22 + 8 * 16)
        .attr('font-size', 10)
        .attr('fill', '#444')
        .text('Black ring = optimum w* found by SLSQP.');
    },
    [grid, w, optWeights, currentObj, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap gap-3 text-sm">
        <label>
          n:&nbsp;
          <select value={n} onChange={(e) => setN(Number(e.target.value))} className="rounded border px-1">
            {N_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setW([optWeights[0], optWeights[1], optWeights[2]])}
          className="rounded border bg-white px-2 py-0.5 hover:bg-gray-50"
        >
          Snap to optimum
        </button>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border bg-white px-2 py-0.5 hover:bg-gray-50"
        >
          Reseed
        </button>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
    </div>
  );
}

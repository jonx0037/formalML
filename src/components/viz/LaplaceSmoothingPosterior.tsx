import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { paletteNB, dirichletPosteriorMean } from './shared/naive-bayes';

// =============================================================================
// LaplaceSmoothingPosterior — §6.2 companion. A 3-category Dirichlet posterior
// on the 2-simplex (the equilateral triangle in barycentric coordinates).
//
// The reader controls (a) the per-category observed counts (preset = "leaning
// toward category 1"), (b) the smoothing parameter α, and (c) the multiplicative
// resampling factor that scales n. As α → 0 the posterior mean collapses to the
// MLE (the empirical frequencies); as α → ∞ the posterior mean converges to the
// uniform prior at the simplex centroid. At α = 1 we recover Laplace smoothing.
// =============================================================================

// Base count pattern — heavily skewed toward category 0 with a zero at cat 1.
const BASE_COUNTS = [3, 0, 5];

// Triangle vertices for the 2-simplex (barycentric).
const TRI = [
  [0.5, 0.08], // top vertex — cat 0
  [0.08, 0.92], // bottom-left — cat 1
  [0.92, 0.92], // bottom-right — cat 2
];

function bary([a, b, c]: number[], W: number, H: number): [number, number] {
  // Convert barycentric (a, b, c) ∈ Δ² to pixel coords.
  const x = a * TRI[0][0] + b * TRI[1][0] + c * TRI[2][0];
  const y = a * TRI[0][1] + b * TRI[1][1] + c * TRI[2][1];
  return [x * W, y * H];
}

export default function LaplaceSmoothingPosterior() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [alpha, setAlpha] = useState(1.0);
  const [scale, setScale] = useState(1.0);

  const W = Math.max(360, Math.min(width || 540, 540));
  const H = Math.round(W * 0.92);

  const counts = useMemo(() => BASE_COUNTS.map((c) => c * scale), [scale]);
  const n = counts.reduce((s, v) => s + v, 0);
  const mle = useMemo(() => (n === 0 ? [1 / 3, 1 / 3, 1 / 3] : counts.map((c) => c / n)), [counts, n]);
  const posteriorMean = useMemo(
    () => Array.from(dirichletPosteriorMean(counts, alpha)),
    [counts, alpha],
  );
  const uniform = [1 / 3, 1 / 3, 1 / 3];

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();

      // Triangle outline.
      const tri = TRI.map((v) => [v[0] * W, v[1] * H]);
      svg
        .append('polygon')
        .attr('points', tri.map((p) => p.join(',')).join(' '))
        .attr('stroke-width', 1.2)
        .attr('opacity', 0.7)
        .style('fill', 'none')
        .style('stroke', paletteNB.data);

      // Vertex labels.
      const labels = ['cat 0', 'cat 1', 'cat 2'];
      tri.forEach(([px, py], i) => {
        svg
          .append('text')
          .attr('x', px)
          .attr('y', py + (i === 0 ? -8 : 14))
          .attr('text-anchor', 'middle')
          .attr('font-size', 11)
          .style('fill', 'currentColor')
          .text(labels[i]);
      });

      // Plot a small simplex grid (gridlines parallel to each edge).
      const grid = svg.append('g').attr('opacity', 0.18);
      for (let t = 1; t < 5; t++) {
        const f = t / 5;
        const a = [f, 1 - f, 0];
        const b = [f, 0, 1 - f];
        grid
          .append('line')
          .attr('x1', bary(a, W, H)[0])
          .attr('y1', bary(a, W, H)[1])
          .attr('x2', bary(b, W, H)[0])
          .attr('y2', bary(b, W, H)[1])
          .attr('stroke-width', 1)
          .style('stroke', paletteNB.data);
        const c = [0, f, 1 - f];
        const d = [1 - f, f, 0];
        grid
          .append('line')
          .attr('x1', bary(c, W, H)[0])
          .attr('y1', bary(c, W, H)[1])
          .attr('x2', bary(d, W, H)[0])
          .attr('y2', bary(d, W, H)[1])
          .attr('stroke-width', 1)
          .style('stroke', paletteNB.data);
        const e = [0, 1 - f, f];
        const ff = [1 - f, 0, f];
        grid
          .append('line')
          .attr('x1', bary(e, W, H)[0])
          .attr('y1', bary(e, W, H)[1])
          .attr('x2', bary(ff, W, H)[0])
          .attr('y2', bary(ff, W, H)[1])
          .attr('stroke-width', 1)
          .style('stroke', paletteNB.data);
      }

      // Annotate the three points: MLE (red), posterior mean (blue), uniform prior (gray dashed).
      const points = [
        { p: uniform, label: 'uniform (α → ∞)', color: paletteNB.data, r: 5 },
        { p: mle, label: `MLE (α → 0): (${mle.map((v) => v.toFixed(2)).join(', ')})`, color: paletteNB.neg, r: 6 },
        {
          p: posteriorMean,
          label: `posterior mean at α = ${alpha.toFixed(2)}: (${posteriorMean.map((v) => v.toFixed(2)).join(', ')})`,
          color: paletteNB.pos,
          r: 7,
        },
      ];
      const g = svg.append('g');
      points.forEach((pt) => {
        const [cx, cy] = bary(pt.p, W, H);
        g.append('circle')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', pt.r)
          .attr('stroke-width', 1.5)
          .style('fill', pt.color)
          .style('stroke', pt.color)
          .style('fill-opacity', 0.55);
      });

      // Trajectory from MLE → posterior → uniform as α grows: dashed connector.
      const path: Array<[number, number]> = [];
      for (let k = 0; k <= 30; k++) {
        const a = Math.pow(10, -2 + k * 0.15); // α from 0.01 to ~316
        const mean = Array.from(dirichletPosteriorMean(counts, a));
        path.push(bary(mean, W, H));
      }
      const line = d3
        .line<[number, number]>()
        .x((d) => d[0])
        .y((d) => d[1]);
      g.append('path')
        .datum(path)
        .attr('fill', 'none')
        .attr('stroke-width', 1.2)
        .attr('stroke-dasharray', '3 3')
        .attr('opacity', 0.55)
        .attr('d', line)
        .style('stroke', paletteNB.accent);
    },
    [counts, mle, posteriorMean, uniform, alpha, W, H],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Smoothing α = {alpha.toFixed(2)}</span>
          <input
            type="range"
            min={0.01}
            max={20}
            step={0.01}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            aria-label="Smoothing parameter"
          />
        </label>
        <label className="flex flex-col">
          <span>Sample-size scale ×{scale.toFixed(2)} → n = {n.toFixed(0)}</span>
          <input
            type="range"
            min={0.5}
            max={20}
            step={0.5}
            value={scale}
            onChange={(e) => setScale(Number(e.target.value))}
            aria-label="Sample-size scale"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Dirichlet posterior on the 2-simplex"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        Counts = ({counts.map((c) => c.toFixed(0)).join(', ')}). The orange dashed curve traces
        the posterior mean as α sweeps from 10⁻² to 10². At α = 1 we get Laplace smoothing.
      </div>
    </div>
  );
}

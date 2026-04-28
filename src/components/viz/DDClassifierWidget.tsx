import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  DEPTH_COLORS,
  IRIS_VERSICOLOR,
  IRIS_VIRGINICA,
  tukeyDepth2D,
  type Point2D,
} from '../../data/statistical-depth';

// =============================================================================
// DDClassifierWidget — anchors §5.1 DD-classifier on Versicolor vs Virginica.
//
// Two views toggled by a button row:
//   "DD-plot"      — scatter of (D_0, D_1) with the y = x decision boundary,
//   "Feature space" — petal-length × petal-width with the depth-based
//                     decision regions overlaid as a coarse grid.
//
// Live training-accuracy readout reproduces the notebook's printed value
// (0.960 on Versicolor vs Virginica) — the ultimate correctness signal,
// since Iris is a fixed dataset and Tukey HD is deterministic on a fixed
// sample.
//
// Static fallback: public/images/topics/statistical-depth/07_ddclassifier.png
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const GRID_RES = 28; // Each cell = one Tukey HD pair → 2 × GRID_RES² queries

type ViewMode = 'dd' | 'feature';

interface DDPlotPoint {
  d0: number;
  d1: number;
  cls: 0 | 1;
  petal: Point2D;
}

function classifyByDD(d0: number, d1: number): 0 | 1 {
  return d1 > d0 ? 1 : 0;
}

export default function DDClassifierWidget() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [view, setView] = useState<ViewMode>('dd');

  const { points, accuracy, gridPreds, xExtent, yExtent } = useMemo(() => {
    // (D_0, D_1) for every training point.
    const points: DDPlotPoint[] = [];
    let correct = 0;

    for (const p of IRIS_VERSICOLOR) {
      const d0 = tukeyDepth2D(p, IRIS_VERSICOLOR);
      const d1 = tukeyDepth2D(p, IRIS_VIRGINICA);
      const pred = classifyByDD(d0, d1);
      if (pred === 0) correct++;
      points.push({ d0, d1, cls: 0, petal: p });
    }
    for (const p of IRIS_VIRGINICA) {
      const d0 = tukeyDepth2D(p, IRIS_VERSICOLOR);
      const d1 = tukeyDepth2D(p, IRIS_VIRGINICA);
      const pred = classifyByDD(d0, d1);
      if (pred === 1) correct++;
      points.push({ d0, d1, cls: 1, petal: p });
    }
    const accuracy = correct / points.length;

    // Feature-space decision-region grid.
    const allPetals = [...IRIS_VERSICOLOR, ...IRIS_VIRGINICA];
    const xMin = Math.min(...allPetals.map((p) => p[0])) - 0.4;
    const xMax = Math.max(...allPetals.map((p) => p[0])) + 0.4;
    const yMin = Math.min(...allPetals.map((p) => p[1])) - 0.2;
    const yMax = Math.max(...allPetals.map((p) => p[1])) + 0.2;

    const gridPreds = new Uint8Array(GRID_RES * GRID_RES);
    const xStep = (xMax - xMin) / (GRID_RES - 1);
    const yStep = (yMax - yMin) / (GRID_RES - 1);
    for (let i = 0; i < GRID_RES; i++) {
      const y = yMin + i * yStep;
      for (let j = 0; j < GRID_RES; j++) {
        const x = xMin + j * xStep;
        const q: Point2D = [x, y];
        const d0 = tukeyDepth2D(q, IRIS_VERSICOLOR);
        const d1 = tukeyDepth2D(q, IRIS_VIRGINICA);
        gridPreds[i * GRID_RES + j] = classifyByDD(d0, d1);
      }
    }

    return {
      points,
      accuracy,
      gridPreds,
      xExtent: [xMin, xMax] as [number, number],
      yExtent: [yMin, yMax] as [number, number],
    };
  }, []);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const ddRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0 || view !== 'dd') return;

      const margin = { top: 28, right: 16, bottom: 42, left: 50 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const dMax = Math.max(
        ...points.map((p) => Math.max(p.d0, p.d1)),
        0.01,
      ) * 1.1;

      const xScale = d3.scaleLinear().domain([0, dMax]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, dMax]).range([innerH, 0]);

      // Decision-line: y = x
      g.append('line')
        .attr('x1', xScale(0)).attr('y1', yScale(0))
        .attr('x2', xScale(dMax)).attr('y2', yScale(dMax))
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-dasharray', '4 3')
        .style('stroke-width', 1.0);

      // Half-plane shading
      g.append('polygon')
        .attr('points', `${xScale(0)},${yScale(0)} ${xScale(dMax)},${yScale(0)} ${xScale(dMax)},${yScale(dMax)}`)
        .style('fill', DEPTH_COLORS.tukey)
        .style('opacity', 0.06);
      g.append('polygon')
        .attr('points', `${xScale(0)},${yScale(0)} ${xScale(0)},${yScale(dMax)} ${xScale(dMax)},${yScale(dMax)}`)
        .style('fill', DEPTH_COLORS.mahalanobis)
        .style('opacity', 0.06);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 32)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .text('D₀(z) — depth in Versicolor sample');
      g.append('text')
        .attr('x', -36).attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .attr('transform', `rotate(-90,-36,${innerH / 2})`)
        .text('D₁(z) — depth in Virginica sample');

      // Points
      g.append('g')
        .selectAll('circle')
        .data(points)
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.d0))
        .attr('cy', (d) => yScale(d.d1))
        .attr('r', 4.2)
        .style('fill', (d) => (d.cls === 0 ? DEPTH_COLORS.tukey : DEPTH_COLORS.mahalanobis))
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', 1.0)
        .style('opacity', 0.85);

      // Title
      svg.append('text')
        .attr('x', w / 2).attr('y', 16)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text('DD-plot — Tukey halfspace depth (Iris: Versicolor vs Virginica)');
    },
    [w, view, points],
  );

  const featureRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0 || view !== 'feature') return;

      const margin = { top: 28, right: 16, bottom: 42, left: 50 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain(xExtent).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(yExtent).range([innerH, 0]);

      // Decision-region shading via grid cells
      const cellW = innerW / (GRID_RES - 1);
      const cellH = innerH / (GRID_RES - 1);
      for (let i = 0; i < GRID_RES; i++) {
        for (let j = 0; j < GRID_RES; j++) {
          const cls = gridPreds[i * GRID_RES + j];
          const cx = j * cellW;
          const cy = innerH - i * cellH;
          g.append('rect')
            .attr('x', cx - cellW / 2)
            .attr('y', cy - cellH / 2)
            .attr('width', cellW)
            .attr('height', cellH)
            .style('fill', cls === 0 ? DEPTH_COLORS.tukey : DEPTH_COLORS.mahalanobis)
            .style('opacity', 0.18)
            .style('stroke', 'none');
        }
      }

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 32)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .text('petal length (cm)');
      g.append('text')
        .attr('x', -36).attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .attr('transform', `rotate(-90,-36,${innerH / 2})`)
        .text('petal width (cm)');

      // Sample points
      g.append('g')
        .selectAll('circle')
        .data(IRIS_VERSICOLOR.map((p) => ({ p, cls: 0 as const })).concat(
          IRIS_VIRGINICA.map((p) => ({ p, cls: 1 as const })),
        ))
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.p[0]))
        .attr('cy', (d) => yScale(d.p[1]))
        .attr('r', 4.2)
        .style('fill', (d) => (d.cls === 0 ? DEPTH_COLORS.tukey : DEPTH_COLORS.mahalanobis))
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', 1.0)
        .style('opacity', 0.92);

      // Title
      svg.append('text')
        .attr('x', w / 2).attr('y', 16)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text('Feature-space decision regions (DD-classifier on Iris petal features)');
    },
    [w, view, gridPreds, xExtent, yExtent],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
    >
      <div
        className="flex flex-wrap items-center gap-3 mb-3"
        style={{ flexDirection: isMobile ? 'column' : 'row', alignItems: isMobile ? 'stretch' : 'center' }}
      >
        <div className="inline-flex rounded border border-[var(--color-border)] overflow-hidden text-sm">
          <button
            type="button"
            onClick={() => setView('dd')}
            className={`px-3 py-1.5 transition-colors ${view === 'dd' ? 'bg-[var(--color-text)] text-[var(--color-bg)]' : 'bg-[var(--color-bg)] text-[var(--color-text)]'}`}
          >
            DD-plot
          </button>
          <button
            type="button"
            onClick={() => setView('feature')}
            className={`px-3 py-1.5 border-l border-[var(--color-border)] transition-colors ${view === 'feature' ? 'bg-[var(--color-text)] text-[var(--color-bg)]' : 'bg-[var(--color-bg)] text-[var(--color-text)]'}`}
          >
            Feature space
          </button>
        </div>
        <div className="ml-auto flex items-center gap-3 text-sm">
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full" style={{ background: DEPTH_COLORS.tukey }} />
            Versicolor (n=50)
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full" style={{ background: DEPTH_COLORS.mahalanobis }} />
            Virginica (n=50)
          </span>
          <span className="text-[var(--color-text-secondary)]">|</span>
          <span>
            training accuracy: <span className="font-mono font-semibold">{accuracy.toFixed(3)}</span>
          </span>
        </div>
      </div>
      <svg
        ref={view === 'dd' ? ddRef : featureRef}
        width={w}
        height={HEIGHT}
        role="img"
        aria-label={view === 'dd' ? 'Iris DD-plot for Tukey halfspace depth' : 'DD-classifier decision regions in petal feature space'}
      />
      <p className="text-xs text-[var(--color-text-secondary)] mt-2">
        {view === 'dd'
          ? 'In the DD-plot, the diagonal D₀ = D₁ is the decision boundary. Points above the line are classified as Virginica; points below as Versicolor. The two clusters separate cleanly because the Iris petal subspace is nearly linearly separable.'
          : 'The decision regions are derived from the DD-rule applied at every grid cell. The boundary curves where Versicolor and Virginica halfspace-depth contours cross — a non-parametric alternative to the linear separator from logistic regression.'}
      </p>
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  DEPTH_COLORS,
  depthContourGrid,
  getSampleA,
  getSampleB,
  getSampleC,
  tukeyDepth2D,
  tukeyMedianFromGrid,
  type Point2D,
} from '../../data/statistical-depth';

// =============================================================================
// DepthContourExplorer — anchors §2.5 Tukey halfspace depth contours.
//
// Single-panel SVG. Sample selector (A: Gaussian / B: Cauchy / C: contaminated)
// switches the underlying point cloud. Draggable query point reads live depth.
// Grid-resolution slider trades accuracy against wait time (the §4 NP-hardness
// preview — at $d = 2$ the cost is per-query, not per-dimension, but the grid
// scan is still gridSize² queries).
//
// Static-figure fallback: public/images/topics/statistical-depth/02_tukey_contours.png
// (rendered separately in the MDX above the widget).
// =============================================================================

const HEIGHT = 480;
const SM_BREAKPOINT = 640;
const CONTOUR_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.45];

type SampleKey = 'A' | 'B' | 'C';

const SAMPLE_LABELS: Record<SampleKey, string> = {
  A: 'Sample A — Gaussian',
  B: 'Sample B — Cauchy',
  C: 'Sample C — contaminated',
};

const SAMPLE_DESCRIPTIONS: Record<SampleKey, string> = {
  A: 'iid bivariate Gaussian; all four §1 centres agree near the origin.',
  B: 'bivariate Cauchy with the same scale matrix; the population mean does not exist.',
  C: '90% Gaussian + 10% outliers at (8, 8) — sample mean drifts; depth median stays put.',
};

function clipForDisplay(X: Point2D[], key: SampleKey): Point2D[] {
  if (key !== 'B') return X;
  // Cauchy view: clip the most extreme draws so the axes don't blow up.
  return X.filter(([x, y]) => x * x + y * y < 30 * 30);
}

export default function DepthContourExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sampleKey, setSampleKey] = useState<SampleKey>('A');
  const [gridSize, setGridSize] = useState(40);
  const [queryX, setQueryX] = useState(0);
  const [queryY, setQueryY] = useState(0);

  const sample = useMemo<Point2D[]>(() => {
    if (sampleKey === 'A') return getSampleA();
    if (sampleKey === 'B') return getSampleB();
    return getSampleC().points;
  }, [sampleKey]);

  // Visible sample (used for plotting + bounding box).
  const visible = useMemo(() => clipForDisplay(sample, sampleKey), [sample, sampleKey]);

  // Depth value at the live query — recomputed every drag.
  const queryDepth = useMemo(
    () => tukeyDepth2D([queryX, queryY], sample),
    [queryX, queryY, sample],
  );

  // Depth grid (cached on sample + gridSize).
  const grid = useMemo(
    () => depthContourGrid(visible, (q) => tukeyDepth2D(q, sample), gridSize, sampleKey === 'B' ? 2 : 1.5),
    [visible, sample, sampleKey, gridSize],
  );
  const tMedian = useMemo(() => tukeyMedianFromGrid(grid), [grid]);
  const maxDepth = useMemo(() => {
    let m = 0;
    for (let k = 0; k < grid.Z.length; k++) {
      if (grid.Z[k] > m) m = grid.Z[k];
    }
    return m;
  }, [grid]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;

      const margin = { top: 28, right: 16, bottom: 38, left: 48 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xExtent: [number, number] = [grid.xs[0], grid.xs[grid.xs.length - 1]];
      const yExtent: [number, number] = [grid.ys[0], grid.ys[grid.ys.length - 1]];

      const xScale = d3.scaleLinear().domain(xExtent).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(yExtent).range([innerH, 0]);

      // ---- Filled contour heatmap via d3.contours ----
      const contourGen = d3
        .contours()
        .size([grid.ncols, grid.nrows])
        .thresholds(20);
      const filled = contourGen(Array.from(grid.Z));

      // Density colour scale matching the notebook's 'Blues' colormap.
      const blues = d3.interpolateBlues;
      const fillScale = d3.scaleLinear<number>().domain([0, maxDepth]).range([0.05, 0.85]);

      const x0 = grid.xs[0];
      const x1 = grid.xs[grid.xs.length - 1];
      const y0 = grid.ys[0];
      const y1 = grid.ys[grid.ys.length - 1];
      const ncols = grid.ncols;
      const nrows = grid.nrows;
      const contourPath = d3
        .geoPath()
        .projection(
          d3.geoTransform({
            point(gx: number, gy: number) {
              const x = xScale(x0 + (x1 - x0) * (gx / (ncols - 1)));
              const y = yScale(y0 + (y1 - y0) * (gy / (nrows - 1)));
              (this as unknown as { stream: d3.GeoStream }).stream.point(x, y);
            },
          }),
        );

      g.append('g')
        .attr('class', 'depth-fill')
        .selectAll('path')
        .data(filled)
        .enter()
        .append('path')
        .attr('d', contourPath as unknown as string)
        .style('fill', (d) => blues(fillScale(d.value)))
        .style('opacity', 0.55)
        .style('stroke', 'none');

      // ---- Line contours at the canonical levels ----
      const validLevels = CONTOUR_LEVELS.filter((lv) => lv < maxDepth);
      const lineGen = d3.contours().size([grid.ncols, grid.nrows]).thresholds(validLevels);
      const lines = lineGen(Array.from(grid.Z));

      g.append('g')
        .attr('class', 'depth-lines')
        .selectAll('path')
        .data(lines)
        .enter()
        .append('path')
        .attr('d', contourPath as unknown as string)
        .style('fill', 'none')
        .style('stroke', DEPTH_COLORS.tukey)
        .style('stroke-width', 1.0)
        .style('opacity', 0.85);

      // Contour labels — stagger across levels so they don't overlap.
      validLevels.forEach((lv, i) => {
        const sample = lines[i];
        if (!sample || !sample.coordinates.length) return;
        const ring = sample.coordinates[0]?.[0];
        if (!ring || ring.length === 0) return;
        const [gx, gy] = ring[Math.floor(ring.length / 2)];
        const x = xScale(grid.xs[0] + (grid.xs[grid.xs.length - 1] - grid.xs[0]) * (gx / (grid.ncols - 1)));
        const y = yScale(grid.ys[0] + (grid.ys[grid.ys.length - 1] - grid.ys[0]) * (gy / (grid.nrows - 1)));
        g.append('text')
          .attr('x', x).attr('y', y)
          .style('fill', DEPTH_COLORS.tukey)
          .style('font-size', '9px')
          .style('font-family', 'var(--font-mono)')
          .style('paint-order', 'stroke')
          .style('stroke', 'var(--color-bg)')
          .style('stroke-width', '3px')
          .text(lv.toFixed(2));
      });

      // ---- Sample points ----
      g.append('g')
        .selectAll('circle')
        .data(visible)
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('r', 2.4)
        .style('fill', 'var(--color-text-secondary)')
        .style('opacity', 0.55);

      // ---- Tukey-median estimate (gold star) ----
      g.append('text')
        .attr('x', xScale(tMedian[0]))
        .attr('y', yScale(tMedian[1]))
        .style('fill', '#F4B400')
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', '0.6px')
        .style('font-size', '24px')
        .style('text-anchor', 'middle')
        .style('dominant-baseline', 'central')
        .text('★');

      // ---- Axes ----
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 30)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .text('x₁');
      g.append('text')
        .attr('x', -34).attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .attr('transform', `rotate(-90,-34,${innerH / 2})`)
        .text('x₂');

      // ---- Draggable query crosshair ----
      const queryGroup = g.append('g').attr('class', 'query-crosshair').style('cursor', 'grab');
      const cx = xScale(queryX);
      const cy = yScale(queryY);
      queryGroup
        .append('circle')
        .attr('cx', cx).attr('cy', cy).attr('r', 18)
        .style('fill', 'transparent')
        .style('stroke', 'none');
      queryGroup
        .append('line')
        .attr('x1', cx - 10).attr('x2', cx + 10)
        .attr('y1', cy).attr('y2', cy)
        .style('stroke', DEPTH_COLORS.query).style('stroke-width', 2.0);
      queryGroup
        .append('line')
        .attr('x1', cx).attr('x2', cx)
        .attr('y1', cy - 10).attr('y2', cy + 10)
        .style('stroke', DEPTH_COLORS.query).style('stroke-width', 2.0);
      queryGroup
        .append('circle')
        .attr('cx', cx).attr('cy', cy).attr('r', 5)
        .style('fill', 'var(--color-bg)')
        .style('stroke', DEPTH_COLORS.query).style('stroke-width', 1.6);

      const drag = d3
        .drag<SVGGElement, unknown>()
        .on('start', function () {
          d3.select(this).style('cursor', 'grabbing');
        })
        .on('drag', (event) => {
          const ex = Math.max(0, Math.min(innerW, event.x));
          const ey = Math.max(0, Math.min(innerH, event.y));
          setQueryX(xScale.invert(ex));
          setQueryY(yScale.invert(ey));
        })
        .on('end', function () {
          d3.select(this).style('cursor', 'grab');
        });
      queryGroup.call(drag);

      // ---- Title ----
      svg.append('text')
        .attr('x', w / 2).attr('y', 16)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text(`Tukey halfspace depth — ${SAMPLE_LABELS[sampleKey]}`);
    },
    [w, sampleKey, sample, visible, grid, maxDepth, queryX, queryY, tMedian],
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
        <div className="flex items-center gap-2">
          <label htmlFor="sd-sample" className="text-sm text-[var(--color-text-secondary)]">
            Sample:
          </label>
          <select
            id="sd-sample"
            value={sampleKey}
            onChange={(e) => {
              setSampleKey(e.target.value as SampleKey);
              setQueryX(0);
              setQueryY(0);
            }}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-sm"
          >
            <option value="A">A — Gaussian</option>
            <option value="B">B — Cauchy</option>
            <option value="C">C — contaminated</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor="sd-grid" className="text-sm text-[var(--color-text-secondary)]">
            Grid:
          </label>
          <input
            id="sd-grid"
            type="range"
            min={20}
            max={70}
            step={5}
            value={gridSize}
            onChange={(e) => setGridSize(parseInt(e.target.value, 10))}
            className="w-32"
          />
          <span className="text-xs font-mono text-[var(--color-text-secondary)]">{gridSize}×{gridSize}</span>
        </div>
        <div className="ml-auto flex items-center gap-3 text-sm">
          <span className="text-[var(--color-text-secondary)]">
            query <span className="font-mono">({queryX.toFixed(2)}, {queryY.toFixed(2)})</span>
          </span>
          <span>
            HDₙ = <span className="font-mono font-semibold" style={{ color: DEPTH_COLORS.tukey }}>
              {queryDepth.toFixed(3)}
            </span>
          </span>
        </div>
      </div>
      <p className="text-xs text-[var(--color-text-secondary)] mb-2">{SAMPLE_DESCRIPTIONS[sampleKey]}</p>
      <svg ref={svgRef} width={w} height={HEIGHT} role="img" aria-label="Tukey halfspace depth contours" />
      <p className="text-xs text-[var(--color-text-secondary)] mt-2">
        ★ marks the empirical Tukey median (centroid of the deepest grid cell). Drag the pink crosshair to read depth at any query point. Grid resolution trades accuracy against wait time — gridSize² Tukey-depth queries per redraw.
      </p>
    </div>
  );
}

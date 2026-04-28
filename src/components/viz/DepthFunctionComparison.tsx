import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  DEPTH_COLORS,
  depthContourGrid,
  fitMahalanobis,
  mahalanobisDepth,
  projectionDepth2D,
  sampleBanana,
  sampleGaussian,
  spatialDepth,
  tukeyDepth2D,
  type DepthGrid,
  type Point2D,
} from '../../data/statistical-depth';

// =============================================================================
// DepthFunctionComparison — anchors §3.5 zoo comparison and the §3.5
// banana-sample contrast (figures 03 + 04).
//
// Two-panel always-Tukey-on-the-left + user-selected depth on the right. The
// reference Tukey panel is the affine-invariant canonical against which the
// other depth functions agree (on Gaussian) or visibly diverge (on banana,
// where Mahalanobis stays elliptical even though the bulk is not).
//
// Inline implementation — simplicial_depth_2d at $n \le 100$, per the
// topic-specific notes. The $\binom{n}{3}$ enumeration is unavoidable; the
// triples list is precomputed once per sample to keep the inner loop fast.
//
// Static fallbacks: 03_depth_zoo_comparison.png and 04_mahalanobis_failure.png
// =============================================================================

const HEIGHT = 380;
const SM_BREAKPOINT = 720;
const SAMPLE_N = 60;
const GRID_SIZE_FAST = 28; // for non-simplicial depths
const GRID_SIZE_SLOW = 22; // simplicial (O(n^3) per query)

type SampleKey = 'gaussian' | 'banana';
type DepthKey = 'mahalanobis' | 'projection' | 'spatial' | 'simplicial';

const DEPTH_LABELS: Record<DepthKey, string> = {
  mahalanobis: 'Mahalanobis',
  projection: 'Projection (K=200)',
  spatial: 'Spatial / L¹',
  simplicial: 'Simplicial',
};
const DEPTH_PALETTE: Record<DepthKey | 'tukey', string> = {
  tukey: DEPTH_COLORS.tukey,
  mahalanobis: DEPTH_COLORS.mahalanobis,
  projection: DEPTH_COLORS.projection,
  spatial: DEPTH_COLORS.spatial,
  simplicial: DEPTH_COLORS.simplicial,
};

const SAMPLE_LABELS: Record<SampleKey, string> = {
  gaussian: 'Gaussian (Σ tilted)',
  banana: 'Banana (x₂ = 0.5 x₁² − 1.5 + noise)',
};

// -----------------------------------------------------------------------------
// Inline simplicial_depth_2d — kept here per the topic-specific notes.
// Triples precomputed once per sample so the per-query cost is the sign-test
// loop only (still O(n³) per query on the grid scan).
// -----------------------------------------------------------------------------

function buildTriples(n: number): Int32Array {
  const m = (n * (n - 1) * (n - 2)) / 6;
  const out = new Int32Array(m * 3);
  let idx = 0;
  for (let i = 0; i < n - 2; i++) {
    for (let j = i + 1; j < n - 1; j++) {
      for (let k = j + 1; k < n; k++) {
        out[idx++] = i;
        out[idx++] = j;
        out[idx++] = k;
      }
    }
  }
  return out;
}

function simplicialDepth2D(query: Point2D, X: Point2D[], triples: Int32Array): number {
  const m = triples.length / 3;
  if (m === 0) return 0;
  let inside = 0;
  const qx = query[0];
  const qy = query[1];
  for (let t = 0; t < m; t++) {
    const i = triples[3 * t];
    const j = triples[3 * t + 1];
    const k = triples[3 * t + 2];
    const ax = X[i][0], ay = X[i][1];
    const bx = X[j][0], by = X[j][1];
    const cx = X[k][0], cy = X[k][1];
    // Signed-area test: sign(Q, A, B), sign(Q, B, C), sign(Q, C, A)
    const s1 = (qx - bx) * (ay - by) - (ax - bx) * (qy - by);
    const s2 = (qx - cx) * (by - cy) - (bx - cx) * (qy - cy);
    const s3 = (qx - ax) * (cy - ay) - (cx - ax) * (qy - ay);
    const hasNeg = s1 < -1e-12 || s2 < -1e-12 || s3 < -1e-12;
    const hasPos = s1 > 1e-12 || s2 > 1e-12 || s3 > 1e-12;
    if (!(hasNeg && hasPos)) inside++;
  }
  return inside / m;
}

// -----------------------------------------------------------------------------

export default function DepthFunctionComparison() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sampleKey, setSampleKey] = useState<SampleKey>('gaussian');
  const [depthKey, setDepthKey] = useState<DepthKey>('mahalanobis');

  const sample = useMemo<Point2D[]>(() => {
    if (sampleKey === 'gaussian') return sampleGaussian(SAMPLE_N, 52);
    return sampleBanana(SAMPLE_N, 72);
  }, [sampleKey]);

  // Precompute Mahalanobis params once per sample.
  const mahalParams = useMemo(() => fitMahalanobis(sample), [sample]);
  // Precompute simplicial triples once per sample.
  const triples = useMemo(() => buildTriples(SAMPLE_N), []);

  // Tukey contour grid — always rendered on the left.
  const tukeyGrid = useMemo<DepthGrid>(
    () => depthContourGrid(sample, (q) => tukeyDepth2D(q, sample), GRID_SIZE_FAST, 1.2),
    [sample],
  );

  // Right-panel grid for the user-selected depth.
  const otherGrid = useMemo<DepthGrid>(() => {
    const gridSize = depthKey === 'simplicial' ? GRID_SIZE_SLOW : GRID_SIZE_FAST;
    let fn: (q: Point2D) => number;
    switch (depthKey) {
      case 'mahalanobis':
        fn = (q) => mahalanobisDepth(q, mahalParams);
        break;
      case 'projection':
        fn = (q) => projectionDepth2D(q, sample, 200, 62);
        break;
      case 'spatial':
        fn = (q) => spatialDepth(q, sample);
        break;
      case 'simplicial':
        fn = (q) => simplicialDepth2D(q, sample, triples);
        break;
    }
    return depthContourGrid(sample, fn, gridSize, 1.2);
  }, [sample, depthKey, mahalParams, triples]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelW = containerWidth <= 0
    ? 0
    : isMobile
      ? containerWidth
      : Math.floor((containerWidth - 12) / 2);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => paintPanel(svg, panelW, tukeyGrid, sample, DEPTH_PALETTE.tukey, 'Tukey halfspace'),
    [panelW, tukeyGrid, sample],
  );
  const rightRef = useD3<SVGSVGElement>(
    (svg) => paintPanel(svg, panelW, otherGrid, sample, DEPTH_PALETTE[depthKey], DEPTH_LABELS[depthKey]),
    [panelW, otherGrid, sample, depthKey],
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
          <label htmlFor="dfc-sample" className="text-sm text-[var(--color-text-secondary)]">Sample:</label>
          <select
            id="dfc-sample"
            value={sampleKey}
            onChange={(e) => setSampleKey(e.target.value as SampleKey)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-sm"
          >
            <option value="gaussian">Gaussian</option>
            <option value="banana">Banana</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor="dfc-depth" className="text-sm text-[var(--color-text-secondary)]">Compare to:</label>
          <select
            id="dfc-depth"
            value={depthKey}
            onChange={(e) => setDepthKey(e.target.value as DepthKey)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1 text-sm"
          >
            <option value="mahalanobis">Mahalanobis</option>
            <option value="projection">Projection (K=200)</option>
            <option value="spatial">Spatial / L¹</option>
            <option value="simplicial">Simplicial (slow at n=60)</option>
          </select>
        </div>
        <p className="text-xs text-[var(--color-text-secondary)] ml-auto max-w-md">
          {SAMPLE_LABELS[sampleKey]} · n = {SAMPLE_N}
        </p>
      </div>
      <div
        className="flex gap-3"
        style={{ flexDirection: isMobile ? 'column' : 'row' }}
      >
        <svg ref={leftRef} width={panelW} height={HEIGHT} role="img" aria-label="Tukey halfspace depth contours" />
        <svg ref={rightRef} width={panelW} height={HEIGHT} role="img" aria-label={`${DEPTH_LABELS[depthKey]} depth contours`} />
      </div>
      <p className="text-xs text-[var(--color-text-secondary)] mt-2">
        On Gaussian data all five depths produce nearly affine images of one another — the elliptical bulk doesn't punish any choice. On the banana sample, Mahalanobis depth's elliptical commitment shows: contours stay aligned with the sample covariance even though the bulk bends. Tukey halfspace depth tracks the bend; the others vary.
      </p>
    </div>
  );
}

function paintPanel(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  w: number,
  grid: DepthGrid,
  sample: Point2D[],
  lineColor: string,
  title: string,
) {
  svg.selectAll('*').remove();
  if (w <= 0) return;

  const margin = { top: 26, right: 10, bottom: 24, left: 28 };
  const innerW = w - margin.left - margin.right;
  const innerH = HEIGHT - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleLinear()
    .domain([grid.xs[0], grid.xs[grid.xs.length - 1]])
    .range([0, innerW]);
  const yScale = d3.scaleLinear()
    .domain([grid.ys[0], grid.ys[grid.ys.length - 1]])
    .range([innerH, 0]);

  let zMax = 0;
  for (let k = 0; k < grid.Z.length; k++) {
    if (grid.Z[k] > zMax) zMax = grid.Z[k];
  }
  if (zMax <= 0) zMax = 1;

  const fillThresholds = d3.range(0.05, 0.96, 0.05).map((f) => f * zMax);
  const lineThresholds = d3.range(0.1, 0.91, 0.2).map((f) => f * zMax);

  const fillGen = d3.contours().size([grid.ncols, grid.nrows]).thresholds(fillThresholds);
  const filled = fillGen(Array.from(grid.Z));
  const lineGen = d3.contours().size([grid.ncols, grid.nrows]).thresholds(lineThresholds);
  const lines = lineGen(Array.from(grid.Z));

  const blueColor = d3.interpolateBlues;
  const fillScale = d3.scaleLinear<number>().domain([0, zMax]).range([0.05, 0.85]);

  const x0 = grid.xs[0];
  const x1 = grid.xs[grid.xs.length - 1];
  const y0 = grid.ys[0];
  const y1 = grid.ys[grid.ys.length - 1];
  const ncols = grid.ncols;
  const nrows = grid.nrows;
  const path = d3.geoPath().projection(
    d3.geoTransform({
      point(gx: number, gy: number) {
        const x = xScale(x0 + (x1 - x0) * (gx / (ncols - 1)));
        const y = yScale(y0 + (y1 - y0) * (gy / (nrows - 1)));
        (this as unknown as { stream: d3.GeoStream }).stream.point(x, y);
      },
    }),
  );

  g.append('g')
    .selectAll('path')
    .data(filled)
    .enter().append('path')
    .attr('d', path as unknown as string)
    .style('fill', (d) => blueColor(fillScale(d.value)))
    .style('opacity', 0.5)
    .style('stroke', 'none');

  g.append('g')
    .selectAll('path')
    .data(lines)
    .enter().append('path')
    .attr('d', path as unknown as string)
    .style('fill', 'none')
    .style('stroke', lineColor)
    .style('stroke-width', 1.0)
    .style('opacity', 0.85);

  g.append('g')
    .selectAll('circle')
    .data(sample)
    .enter().append('circle')
    .attr('cx', (d) => xScale(d[0]))
    .attr('cy', (d) => yScale(d[1]))
    .attr('r', 1.8)
    .style('fill', 'var(--color-text-secondary)')
    .style('opacity', 0.6);

  g.append('g')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(xScale).ticks(4))
    .selectAll('text')
    .style('fill', 'var(--color-text-secondary)')
    .style('font-size', '9px');
  g.append('g')
    .call(d3.axisLeft(yScale).ticks(4))
    .selectAll('text')
    .style('fill', 'var(--color-text-secondary)')
    .style('font-size', '9px');
  g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

  svg.append('text')
    .attr('x', w / 2).attr('y', 14)
    .style('fill', 'var(--color-text)')
    .style('text-anchor', 'middle').style('font-size', '11px')
    .style('font-weight', '600')
    .text(`${title}  (max = ${zMax.toFixed(2)})`);
}

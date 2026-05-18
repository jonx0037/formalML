// =============================================================================
// BananaMetricGeodesicExplorer.tsx
//
// §3 Riemannian geometry of probability. Two coordinated panels showing what
// the Fisher metric DOES to a Bayesian posterior with non-constant curvature:
//
//   (a) Metric ellipses {v : v^T G(θ) v = c} at a 5×5 grid of θ over the banana
//       support. At b = 0 the metric is constant (ellipses are circles); as b
//       grows the ellipses rotate and stretch along the banana's ridge.
//   (b) Geodesic vs Euclidean straight-line trajectories from a common
//       starting point and initial velocity. The geodesic respects the
//       manifold's geometry; the straight line ignores it.
//
// Naming note: the brief's viz design intent §3 calls this `FisherMetricExplorer`,
// but that name is already taken by the information-geometry topic's
// Fisher-metric explorer (covers Gaussian/Bernoulli/Exponential families).
// Renamed to BananaMetricGeodesicExplorer for the RMHMC version.
//
// **NB.** The geodesic uses the CORRECTED Christoffel symbols from
// shared/riemann-hmc.ts. The notebook's banana_christoffel() has a bracket-
// formula bug that flips the geodesic's curvature direction (notebook cell 11
// prints "Geodesic on G: 7.292" vs Euclidean 2.003 — the opposite of the brief's
// "geodesic curves toward the ridge" claim). The corrected TS implementation
// gives Γ^2_{11} = 2b as the ONLY nonzero symbol, which yields the EXACT
// ridge-following geodesic when the initial velocity is tangent to the ridge.
//
// Controls: b ∈ [0, 1.5] slider (display-vs-committed). Starting-position
// preset selector. "Ridge-tangent IC" toggle.
//
// Computation: in-browser via shared/riemann-hmc.ts.
// Static fallback: /images/topics/riemann-manifold-hmc/03_banana_metric_ellipses_and_geodesic.png
// (degraded — uses the notebook's buggy Christoffel; the live viz here is the
// authoritative behavior).
// =============================================================================

import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bananaLogDensity,
  bananaMetric,
  geodesicTrajectory,
  paletteRMHMC,
} from './shared/riemann-hmc';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const DEFAULT_B = 1.0;
const X_MIN = -3.5;
const X_MAX = 3.5;
const Y_MIN = -6;
const Y_MAX = 4;
const GRID_N = 5; // 5×5 metric ellipse grid
const ELLIPSE_C = 0.6; // contour level for ellipses (visible but not overlapping)

interface StartPreset {
  label: string;
  theta0: [number, number];
  velAngle: number; // radians, used when ridge-tangent toggle is OFF
}

const STARTS: StartPreset[] = [
  { label: '(-2, -3) — on ridge', theta0: [-2, -3], velAngle: 0 },
  { label: '(-2.5, -2) — above ridge', theta0: [-2.5, -2], velAngle: 0 },
  { label: '(0, -1) — near center', theta0: [0, -1], velAngle: 0.3 },
];

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/** Ridge of the banana with params (a=1, b): θ_2 = -b(θ_1² - 1). */
function ridgeTheta2(t1: number, b: number, a: number = 1): number {
  return -b * (t1 * t1 - a * a);
}

/** Tangent dθ_2/dθ_1 of the ridge at θ_1: -2 b θ_1. */
function ridgeTangentSlope(t1: number, b: number): number {
  return -2 * b * t1;
}

interface MetricEllipse {
  cx: number;
  cy: number;
  semiA: number;
  semiB: number;
  angle: number; // radians
}

function computeEllipse(theta: [number, number], b: number, c: number): MetricEllipse {
  const G = bananaMetric([theta[0], theta[1]], { a: 1, b });
  const a11 = G[0][0];
  const a12 = G[0][1];
  const a22 = G[1][1];
  const tr = a11 + a22;
  const det = a11 * a22 - a12 * a12;
  const disc = Math.sqrt(Math.max(tr * tr - 4 * det, 0));
  const lam1 = 0.5 * (tr + disc);
  const lam2 = 0.5 * (tr - disc);
  // Semi-axes: v^T G v = c with v along eigenvector λ_k ⇒ ||v||² = c/λ_k.
  // Smaller eigenvalue ⇒ longer axis.
  const semiBig = Math.sqrt(c / Math.max(lam2, 1e-12));
  const semiSmall = Math.sqrt(c / Math.max(lam1, 1e-12));
  // Angle of eigenvector for the LARGER eigenvalue (shorter axis direction).
  let angle: number;
  if (Math.abs(a12) < 1e-10) {
    angle = a11 >= a22 ? 0 : Math.PI / 2;
  } else {
    angle = Math.atan2(lam1 - a11, a12);
  }
  // We return semiA as the longer axis (perpendicular to the larger-eigenvalue direction).
  return { cx: theta[0], cy: theta[1], semiA: semiBig, semiB: semiSmall, angle };
}

interface ContourData {
  field: number[];
  nx: number;
  ny: number;
  thresholds: number[];
}

function computeContours(b: number): ContourData {
  const nx = 80;
  const ny = 100;
  const field: number[] = new Array(nx * ny);
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = 0; j < ny; j++) {
    const y = Y_MIN + ((Y_MAX - Y_MIN) * j) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / (nx - 1);
      const v = bananaLogDensity([x, y], { a: 1, b });
      field[j * nx + i] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const nLevels = 8;
  const thresholds: number[] = [];
  for (let i = 0; i < nLevels; i++) thresholds.push(lo + ((hi - lo) * (i + 1)) / (nLevels + 1));
  return { field, nx, ny, thresholds };
}

// -----------------------------------------------------------------------------
// Rendering helpers
// -----------------------------------------------------------------------------

interface PanelLayout {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
}

function drawContoursAndAxes(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  innerW: number,
  innerH: number,
  xScale: d3.ScaleLinear<number, number>,
  yScale: d3.ScaleLinear<number, number>,
  contours: ContourData,
  title: string,
): void {
  const contourGen = d3.contours().size([contours.nx, contours.ny]).thresholds(contours.thresholds);
  const cs = contourGen(contours.field);
  const colorScale = d3
    .scaleLinear<string>()
    .domain([contours.thresholds[0], contours.thresholds[contours.thresholds.length - 1]])
    .range(['rgba(44, 62, 80, 0.04)', 'rgba(44, 62, 80, 0.28)']);
  const geoPath = d3.geoPath(
    d3.geoTransform({
      point(this: { stream: { point: (x: number, y: number) => void } }, x: number, y: number): void {
        const px = (x / (contours.nx - 1)) * (X_MAX - X_MIN) + X_MIN;
        const py = (y / (contours.ny - 1)) * (Y_MAX - Y_MIN) + Y_MIN;
        this.stream.point(xScale(px), yScale(py));
      },
    }),
  );
  g.append('g')
    .selectAll('path')
    .data(cs)
    .join('path')
    .attr('d', geoPath as unknown as (d: d3.ContourMultiPolygon) => string)
    .style('fill', (d) => colorScale(d.value))
    .style('stroke', 'var(--color-text-secondary, #6B6B6B)')
    .style('stroke-width', 0.3)
    .style('opacity', 0.85);
  g.append('g')
    .attr('transform', `translate(0, ${innerH})`)
    .call(d3.axisBottom(xScale).ticks(5))
    .selectAll('text')
    .style('fill', 'var(--color-text, #1A1A1A)');
  g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text, #1A1A1A)');
  g.append('text')
    .attr('x', innerW / 2)
    .attr('y', -8)
    .attr('text-anchor', 'middle')
    .style('font-size', '13px')
    .style('font-weight', '600')
    .style('fill', 'var(--color-text, #1A1A1A)')
    .text(title);
  g.append('text')
    .attr('x', innerW / 2)
    .attr('y', innerH + 32)
    .attr('text-anchor', 'middle')
    .style('font-size', '11px')
    .style('fill', 'var(--color-text-secondary, #6B6B6B)')
    .text('θ₁');
  g.append('text')
    .attr('x', -innerH / 2)
    .attr('y', -36)
    .attr('text-anchor', 'middle')
    .attr('transform', 'rotate(-90)')
    .style('font-size', '11px')
    .style('fill', 'var(--color-text-secondary, #6B6B6B)')
    .text('θ₂');
}

function drawRidge(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  xScale: d3.ScaleLinear<number, number>,
  yScale: d3.ScaleLinear<number, number>,
  b: number,
): void {
  const ridgePath = d3
    .line<number>()
    .x((t1) => xScale(t1))
    .y((t1) => yScale(ridgeTheta2(t1, b)));
  const ridgePts: number[] = [];
  for (let i = 0; i < 100; i++) ridgePts.push(X_MIN + ((X_MAX - X_MIN) * i) / 99);
  g.append('path')
    .datum(ridgePts)
    .attr('d', ridgePath)
    .style('fill', 'none')
    .style('stroke', paletteRMHMC.target)
    .style('stroke-width', 1)
    .style('stroke-dasharray', '3 3')
    .style('opacity', 0.5);
}

function renderEllipsePanel(svg: SVGSVGElement, layout: PanelLayout, b: number, contours: ContourData): void {
  const sel = d3.select(svg);
  sel.selectAll('*').remove();
  const innerW = layout.width - layout.margin.left - layout.margin.right;
  const innerH = layout.height - layout.margin.top - layout.margin.bottom;
  if (innerW <= 0 || innerH <= 0) return;
  const g = sel.append('g').attr('transform', `translate(${layout.margin.left}, ${layout.margin.top})`);
  const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([Y_MIN, Y_MAX]).range([innerH, 0]);
  drawContoursAndAxes(g, innerW, innerH, xScale, yScale, contours, 'Fisher metric ellipses');
  drawRidge(g, xScale, yScale, b);

  // Ellipses at GRID_N × GRID_N grid
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < GRID_N; i++) xs.push(X_MIN + ((X_MAX - X_MIN) * (i + 0.5)) / GRID_N);
  for (let j = 0; j < GRID_N; j++) ys.push(Y_MIN + ((Y_MAX - Y_MIN) * (j + 0.5)) / GRID_N);
  for (const x of xs) {
    for (const y of ys) {
      const e = computeEllipse([x, y], b, ELLIPSE_C);
      const rxPix = Math.max((e.semiA / (X_MAX - X_MIN)) * innerW, 0.5);
      const ryPix = Math.max((e.semiB / (Y_MAX - Y_MIN)) * innerH, 0.5);
      const cxPix = xScale(e.cx);
      const cyPix = yScale(e.cy);
      g.append('ellipse')
        .attr('cx', cxPix)
        .attr('cy', cyPix)
        .attr('rx', rxPix)
        .attr('ry', ryPix)
        .attr('transform', `rotate(${(-e.angle * 180) / Math.PI}, ${cxPix}, ${cyPix})`)
        .style('fill', 'none')
        .style('stroke', paletteRMHMC.metric)
        .style('stroke-width', 1)
        .style('opacity', 0.85);
      g.append('circle')
        .attr('cx', cxPix)
        .attr('cy', cyPix)
        .attr('r', 1.4)
        .style('fill', paletteRMHMC.metric);
    }
  }
}

function renderGeodesicPanel(
  svg: SVGSVGElement,
  layout: PanelLayout,
  b: number,
  contours: ContourData,
  startPreset: StartPreset,
  ridgeTangent: boolean,
): void {
  const sel = d3.select(svg);
  sel.selectAll('*').remove();
  const innerW = layout.width - layout.margin.left - layout.margin.right;
  const innerH = layout.height - layout.margin.top - layout.margin.bottom;
  if (innerW <= 0 || innerH <= 0) return;
  const g = sel.append('g').attr('transform', `translate(${layout.margin.left}, ${layout.margin.top})`);
  const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([Y_MIN, Y_MAX]).range([innerH, 0]);
  drawContoursAndAxes(g, innerW, innerH, xScale, yScale, contours, 'Geodesic vs Euclidean line');
  drawRidge(g, xScale, yScale, b);

  const [t10, t20] = startPreset.theta0;
  let vel0: [number, number];
  if (ridgeTangent) {
    const slope = ridgeTangentSlope(t10, b);
    const norm = Math.hypot(1, slope);
    vel0 = [1 / norm, slope / norm];
  } else {
    vel0 = [Math.cos(startPreset.velAngle), Math.sin(startPreset.velAngle)];
  }
  const T = 4;
  const nSteps = 400;
  const geo = geodesicTrajectory([t10, t20], [vel0[0], vel0[1]], T, nSteps, { a: 1, b });
  const geoPts: [number, number][] = geo.map((s) => [s.theta[0], s.theta[1]]);

  const dt = T / nSteps;
  const linePts: [number, number][] = [];
  for (let n = 0; n <= nSteps; n++) linePts.push([t10 + n * dt * vel0[0], t20 + n * dt * vel0[1]]);

  const inBounds = (p: [number, number]): boolean =>
    p[0] >= X_MIN - 0.3 && p[0] <= X_MAX + 0.3 && p[1] >= Y_MIN - 0.5 && p[1] <= Y_MAX + 0.5;
  const geoClip = geoPts.filter(inBounds);
  const lineClip = linePts.filter(inBounds);

  const line = d3
    .line<[number, number]>()
    .x((p) => xScale(p[0]))
    .y((p) => yScale(p[1]));
  g.append('path')
    .datum(lineClip)
    .attr('d', line)
    .style('fill', 'none')
    .style('stroke', paletteRMHMC.hmc)
    .style('stroke-width', 1.6)
    .style('opacity', 0.85);
  g.append('path')
    .datum(geoClip)
    .attr('d', line)
    .style('fill', 'none')
    .style('stroke', paletteRMHMC.geodesic)
    .style('stroke-width', 2)
    .style('opacity', 0.9);

  g.append('circle')
    .attr('cx', xScale(t10))
    .attr('cy', yScale(t20))
    .attr('r', 4)
    .style('fill', paletteRMHMC.target)
    .style('stroke', 'var(--color-surface, #fff)')
    .style('stroke-width', 1.5);

  const legend = g.append('g').attr('transform', `translate(${innerW - 130}, 8)`);
  const legendItems = [
    { color: paletteRMHMC.geodesic, label: 'Geodesic on G', dashed: false },
    { color: paletteRMHMC.hmc, label: 'Euclidean line', dashed: false },
    { color: paletteRMHMC.target, label: 'Ridge', dashed: true },
  ];
  legendItems.forEach((item, i) => {
    legend
      .append('line')
      .attr('x1', 0)
      .attr('x2', 18)
      .attr('y1', i * 16)
      .attr('y2', i * 16)
      .style('stroke', item.color)
      .style('stroke-width', 1.8)
      .style('stroke-dasharray', item.dashed ? '3 3' : 'none');
    legend
      .append('text')
      .attr('x', 22)
      .attr('y', i * 16 + 4)
      .style('font-size', '10.5px')
      .style('fill', 'var(--color-text, #1A1A1A)')
      .text(item.label);
  });
}

// -----------------------------------------------------------------------------
// Main component
// -----------------------------------------------------------------------------

export default function BananaMetricGeodesicExplorer() {
  const [displayB, setDisplayB] = useState(DEFAULT_B);
  const [committedB, setCommittedB] = useState(DEFAULT_B);
  const [startIdx, setStartIdx] = useState(0);
  const [ridgeTangent, setRidgeTangent] = useState(true);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 800;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.floor((containerWidth - 16) / 2);
  const panelHeight = isMobile ? 320 : 380;
  const layout: PanelLayout = useMemo(
    () => ({ width: panelWidth, height: panelHeight, margin: { top: 30, right: 20, bottom: 40, left: 50 } }),
    [panelWidth, panelHeight],
  );

  const contours = useMemo(() => computeContours(committedB), [committedB]);

  const ellipseRef = useMemo(() => ({ current: null as SVGSVGElement | null }), []);
  const geoRef = useMemo(() => ({ current: null as SVGSVGElement | null }), []);

  useEffect(() => {
    if (ellipseRef.current) renderEllipsePanel(ellipseRef.current, layout, committedB, contours);
    if (geoRef.current)
      renderGeodesicPanel(geoRef.current, layout, committedB, contours, STARTS[startIdx], ridgeTangent);
  }, [layout, committedB, contours, startIdx, ridgeTangent, ellipseRef, geoRef]);

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 3: Fisher metric ellipses and geodesic-vs-Euclidean trajectories on the banana"
      style={{ margin: '1.5rem 0' }}
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem 1.25rem',
          alignItems: 'center',
          fontSize: '0.875rem',
          color: 'var(--color-text, #333)',
          marginBottom: '0.75rem',
        }}
      >
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>banana curvature b:</span>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={displayB}
            onChange={(e) => setDisplayB(Number(e.target.value))}
            onMouseUp={(e) => setCommittedB(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedB(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedB(Number((e.target as HTMLInputElement).value))}
            style={{ width: '160px' }}
            aria-label="Banana curvature parameter b"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>b = {displayB.toFixed(2)}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>start:</span>
          <select
            value={startIdx}
            onChange={(e) => setStartIdx(Number(e.target.value))}
            aria-label="Starting position preset"
            style={{ fontSize: '0.85rem' }}
          >
            {STARTS.map((s, i) => (
              <option key={i} value={i}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
          <input
            type="checkbox"
            checked={ridgeTangent}
            onChange={(e) => setRidgeTangent(e.target.checked)}
            aria-label="Use ridge-tangent initial velocity"
          />
          <span>ridge-tangent initial velocity</span>
        </label>
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: '0.5rem',
          justifyContent: 'center',
        }}
      >
        <svg
          ref={(n) => {
            ellipseRef.current = n;
          }}
          width={panelWidth}
          height={panelHeight}
          role="img"
          aria-label="Fisher metric ellipses over the banana support"
        />
        <svg
          ref={(n) => {
            geoRef.current = n;
          }}
          width={panelWidth}
          height={panelHeight}
          role="img"
          aria-label="Geodesic on the Fisher metric versus Euclidean straight line"
        />
      </div>
      <figcaption
        style={{
          marginTop: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-text-secondary, #6B6B6B)',
          textAlign: 'center',
        }}
      >
        Figure 3. Left: Fisher-metric ellipses at a 5×5 grid; at b = 0 the metric is constant
        and ellipses are circles, but as b grows they rotate and stretch along the banana&apos;s ridge.
        Right: with ridge-tangent initial velocity, the geodesic on (Θ, G) traces the ridge exactly
        (Γ²₁₁ = 2b yields d²θ₂/dθ₁² = -2b — matching the ridge curvature). Toggle off &quot;ridge-tangent&quot;
        to see how the geodesic deviates when the initial velocity is generic.
      </figcaption>
    </figure>
  );
}

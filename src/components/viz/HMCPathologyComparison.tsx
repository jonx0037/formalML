// =============================================================================
// HMCPathologyComparison.tsx
//
// §1 Overview & Motivation. Two side-by-side panels demonstrating where
// standard (constant-mass) HMC breaks on position-dependent curvature:
//
//   (a) Neal's funnel — hierarchical prior θ | τ ~ N(0, e^τ) with τ ~ N(0, 9).
//       Log-density contour over (τ, θ) ∈ [-4, 4] × [-15, 15] with three
//       representative constant-mass HMC trajectories at user-set ε.
//   (b) The banana — Gauss-Newton posterior from likelihood
//       y | θ ~ N(θ_2 + b(θ_1² - a²), 1), y_obs = 0, prior θ_1 ~ N(0, a²),
//       (a, b) = (1, 1). Log-density contour over [-3.5, 3.5] × [-6, 4] with
//       three constant-mass HMC trajectories at the same ε.
//
// Reader takeaway: no single ε works everywhere. Small ε stalls in the loose
// regions; large ε shoots through the tight regions. The §2-onward Riemannian
// fix promotes the mass matrix to a function of position.
//
// Slider: ε ∈ [0.02, 0.40]. Display-vs-committed per CLAUDE.md "Slider perf:
// commit-on-release for heavy MC" — live label on onChange, trajectory
// recompute on onMouseUp / onTouchEnd / onKeyUp.
//
// Computation: in-browser via shared/riemann-hmc.ts. No precompute.
// Static fallback: /images/topics/riemann-manifold-hmc/01_hmc_pathologies.png.
// =============================================================================

import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bananaLogDensity,
  bananaGradU,
  standardLeapfrogStep,
  mulberry32,
  makeGaussian,
  paletteRMHMC,
} from './shared/riemann-hmc';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const DEFAULT_EPS = 0.15;
const L_LEAPFROG = 40;
const TRAJECTORY_SEED = 11;
const SM_BREAKPOINT = 640;

// -----------------------------------------------------------------------------
// Funnel (Neal): τ ~ N(0, 9), θ | τ ~ N(0, e^τ)
// -----------------------------------------------------------------------------

function funnelLogDensity(state: number[]): number {
  const tau = state[0];
  const theta = state[1];
  // -½ τ² / 9 - ½ θ² / e^τ - ½ τ
  return -0.5 * ((tau * tau) / 9) - 0.5 * ((theta * theta) / Math.exp(tau)) - 0.5 * tau;
}

function funnelGradU(state: number[]): number[] {
  const tau = state[0];
  const theta = state[1];
  const eTau = Math.exp(tau);
  // U = -log π = ½ τ²/9 + ½ θ²/e^τ + ½ τ
  // ∂U/∂τ = τ/9 - ½ θ²/e^τ + ½
  // ∂U/∂θ = θ/e^τ
  return [tau / 9 - 0.5 * (theta * theta) / eTau + 0.5, theta / eTau];
}

// -----------------------------------------------------------------------------
// Compute trajectories
// -----------------------------------------------------------------------------

interface Trajectory {
  points: number[][];
}

function computeFunnelTrajectories(eps: number): Trajectory[] {
  const rng = mulberry32(TRAJECTORY_SEED);
  const gauss = makeGaussian(rng);
  const starts: number[][] = [
    [-3, 5],
    [0, -1],
    [3, 8],
  ];
  return starts.map((start) => {
    const points: number[][] = [start.slice()];
    let theta = start.slice();
    let mom = [gauss(), gauss()];
    for (let l = 0; l < L_LEAPFROG; l++) {
      const r = standardLeapfrogStep(theta, mom, eps, funnelGradU, [1, 1]);
      theta = r.theta;
      mom = r.mom;
      points.push(theta.slice());
    }
    return { points };
  });
}

function computeBananaTrajectories(eps: number): Trajectory[] {
  const rng = mulberry32(TRAJECTORY_SEED + 1);
  const gauss = makeGaussian(rng);
  const starts: number[][] = [
    [-2.5, -2],
    [0, 0.5],
    [2, -3.5],
  ];
  const bananaP = { a: 1, b: 1 };
  return starts.map((start) => {
    const points: number[][] = [start.slice()];
    let theta = start.slice();
    let mom = [gauss(), gauss()];
    for (let l = 0; l < L_LEAPFROG; l++) {
      const r = standardLeapfrogStep(theta, mom, eps, (th) => bananaGradU(th, bananaP), [1, 1]);
      theta = r.theta;
      mom = r.mom;
      points.push(theta.slice());
    }
    return { points };
  });
}

// -----------------------------------------------------------------------------
// Contour helpers
// -----------------------------------------------------------------------------

interface ContourSpec {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  nx: number;
  ny: number;
  logDensity: (state: number[]) => number;
}

function computeContours(spec: ContourSpec): { thresholds: number[]; densityField: number[]; minLog: number; maxLog: number } {
  const { xMin, xMax, yMin, yMax, nx, ny, logDensity } = spec;
  const field: number[] = new Array(nx * ny);
  let minLog = Infinity;
  let maxLog = -Infinity;
  for (let j = 0; j < ny; j++) {
    const y = yMin + ((yMax - yMin) * j) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const x = xMin + ((xMax - xMin) * i) / (nx - 1);
      const v = logDensity([x, y]);
      field[j * nx + i] = v;
      if (Number.isFinite(v)) {
        if (v < minLog) minLog = v;
        if (v > maxLog) maxLog = v;
      }
    }
  }
  // 8 evenly spaced contour levels between the 10th and 99th percentile
  const sorted = field.filter((v) => Number.isFinite(v)).slice().sort((a, b) => a - b);
  const q = (p: number) => sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor(p * sorted.length)))];
  const lo = q(0.4);
  const hi = q(0.99);
  const nLevels = 8;
  const thresholds: number[] = [];
  for (let i = 0; i < nLevels; i++) thresholds.push(lo + ((hi - lo) * i) / (nLevels - 1));
  return { thresholds, densityField: field, minLog, maxLog };
}

// -----------------------------------------------------------------------------
// Panel renderer
// -----------------------------------------------------------------------------

interface PanelConfig {
  spec: ContourSpec;
  trajectories: Trajectory[];
  width: number;
  height: number;
  title: string;
  xLabel: string;
  yLabel: string;
}

function renderPanel(svg: SVGSVGElement, cfg: PanelConfig): void {
  const sel = d3.select(svg);
  sel.selectAll('*').remove();
  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const innerW = cfg.width - margin.left - margin.right;
  const innerH = cfg.height - margin.top - margin.bottom;
  if (innerW <= 0 || innerH <= 0) return;

  const g = sel.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

  const xScale = d3.scaleLinear().domain([cfg.spec.xMin, cfg.spec.xMax]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([cfg.spec.yMin, cfg.spec.yMax]).range([innerH, 0]);

  // Contour fill via d3.contours
  const { thresholds, densityField } = computeContours(cfg.spec);
  const contourGen = d3
    .contours()
    .size([cfg.spec.nx, cfg.spec.ny])
    .thresholds(thresholds);
  const contours = contourGen(densityField);
  const colorScale = d3
    .scaleLinear<string>()
    .domain([thresholds[0], thresholds[thresholds.length - 1]])
    .range(['rgba(44, 62, 80, 0.04)', 'rgba(44, 62, 80, 0.28)']);

  // Project contours via geoPath with a custom transform
  const geoPath = d3.geoPath(
    d3.geoTransform({
      point(this: { stream: { point: (x: number, y: number) => void } }, x: number, y: number): void {
        const px = (x / (cfg.spec.nx - 1)) * (cfg.spec.xMax - cfg.spec.xMin) + cfg.spec.xMin;
        const py = (y / (cfg.spec.ny - 1)) * (cfg.spec.yMax - cfg.spec.yMin) + cfg.spec.yMin;
        this.stream.point(xScale(px), yScale(py));
      },
    }),
  );

  g.append('g')
    .selectAll('path')
    .data(contours)
    .join('path')
    .attr('d', geoPath as unknown as (d: d3.ContourMultiPolygon) => string)
    .style('fill', (d) => colorScale(d.value))
    .style('stroke', 'var(--color-text-secondary, #6B6B6B)')
    .style('stroke-width', 0.4)
    .style('opacity', 0.85);

  // Axes
  g.append('g')
    .attr('transform', `translate(0, ${innerH})`)
    .call(d3.axisBottom(xScale).ticks(5))
    .selectAll('text')
    .style('fill', 'var(--color-text, #1A1A1A)');
  g.append('g')
    .call(d3.axisLeft(yScale).ticks(5))
    .selectAll('text')
    .style('fill', 'var(--color-text, #1A1A1A)');

  g.append('text')
    .attr('x', innerW / 2)
    .attr('y', -8)
    .attr('text-anchor', 'middle')
    .style('font-size', '13px')
    .style('font-weight', '600')
    .style('fill', 'var(--color-text, #1A1A1A)')
    .text(cfg.title);
  g.append('text')
    .attr('x', innerW / 2)
    .attr('y', innerH + 32)
    .attr('text-anchor', 'middle')
    .style('font-size', '11px')
    .style('fill', 'var(--color-text-secondary, #6B6B6B)')
    .text(cfg.xLabel);
  g.append('text')
    .attr('x', -innerH / 2)
    .attr('y', -36)
    .attr('text-anchor', 'middle')
    .attr('transform', 'rotate(-90)')
    .style('font-size', '11px')
    .style('fill', 'var(--color-text-secondary, #6B6B6B)')
    .text(cfg.yLabel);

  // Trajectories
  const line = d3
    .line<number[]>()
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]));
  for (const traj of cfg.trajectories) {
    // Clip to panel viewport so wild trajectories don't paint off-canvas
    const clipped = traj.points.filter(
      (p) => p[0] >= cfg.spec.xMin - 0.5 && p[0] <= cfg.spec.xMax + 0.5 && p[1] >= cfg.spec.yMin - 0.5 && p[1] <= cfg.spec.yMax + 0.5,
    );
    g.append('path')
      .datum(clipped)
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', paletteRMHMC.hmc)
      .style('stroke-width', 1.4)
      .style('opacity', 0.85);
    if (clipped.length > 0) {
      g.append('circle')
        .attr('cx', xScale(clipped[0][0]))
        .attr('cy', yScale(clipped[0][1]))
        .attr('r', 3.5)
        .style('fill', paletteRMHMC.hmc)
        .style('stroke', 'var(--color-surface, #fff)')
        .style('stroke-width', 1);
    }
  }
}

// -----------------------------------------------------------------------------
// Main component
// -----------------------------------------------------------------------------

export default function HMCPathologyComparison() {
  const [displayEps, setDisplayEps] = useState(DEFAULT_EPS);
  const [committedEps, setCommittedEps] = useState(DEFAULT_EPS);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 800;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.floor((containerWidth - 16) / 2);
  const panelHeight = isMobile ? 320 : 380;

  // Heavy compute: only on committedEps change
  const funnelTraj = useMemo(() => computeFunnelTrajectories(committedEps), [committedEps]);
  const bananaTraj = useMemo(() => computeBananaTrajectories(committedEps), [committedEps]);

  const funnelSvgRef = useMemo(() => ({ current: null as SVGSVGElement | null }), []);
  const bananaSvgRef = useMemo(() => ({ current: null as SVGSVGElement | null }), []);

  useEffect(() => {
    if (funnelSvgRef.current) {
      renderPanel(funnelSvgRef.current, {
        spec: {
          xMin: -4,
          xMax: 4,
          yMin: -15,
          yMax: 15,
          nx: 80,
          ny: 100,
          logDensity: funnelLogDensity,
        },
        trajectories: funnelTraj,
        width: panelWidth,
        height: panelHeight,
        title: "Neal's funnel",
        xLabel: 'τ',
        yLabel: 'θ',
      });
    }
    if (bananaSvgRef.current) {
      renderPanel(bananaSvgRef.current, {
        spec: {
          xMin: -3.5,
          xMax: 3.5,
          yMin: -6,
          yMax: 4,
          nx: 80,
          ny: 100,
          logDensity: (s) => bananaLogDensity(s, { a: 1, b: 1 }),
        },
        trajectories: bananaTraj,
        width: panelWidth,
        height: panelHeight,
        title: 'Banana',
        xLabel: 'θ₁',
        yLabel: 'θ₂',
      });
    }
  }, [funnelTraj, bananaTraj, panelWidth, panelHeight, funnelSvgRef, bananaSvgRef]);

  const commitEps = (v: number): void => setCommittedEps(v);

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 1: Constant-mass HMC trajectories on Neal's funnel and the banana distribution"
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
          <span>step size ε:</span>
          <input
            type="range"
            min={0.02}
            max={0.4}
            step={0.01}
            value={displayEps}
            onChange={(e) => setDisplayEps(Number(e.target.value))}
            onMouseUp={(e) => commitEps(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => commitEps(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => commitEps(Number((e.target as HTMLInputElement).value))}
            style={{ width: '160px' }}
            aria-label="Leapfrog step size epsilon"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>
            ε = {displayEps.toFixed(2)}
          </span>
        </label>
        <span style={{ color: 'var(--color-text-secondary, #6B6B6B)', fontSize: '0.8rem' }}>
          L = {L_LEAPFROG} leapfrog steps, identity mass matrix
        </span>
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
          ref={(node) => {
            funnelSvgRef.current = node;
          }}
          width={panelWidth}
          height={panelHeight}
          role="img"
          aria-label="Neal's funnel with constant-mass HMC trajectories"
        />
        <svg
          ref={(node) => {
            bananaSvgRef.current = node;
          }}
          width={panelWidth}
          height={panelHeight}
          role="img"
          aria-label="Banana with constant-mass HMC trajectories"
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
        Figure 1. Three constant-mass-matrix HMC trajectories on each pathology. At small ε the trajectory crawls in the loose regions and never reaches the tight ones; at large ε it overshoots in the tight regions. A single ε cannot serve both. The Riemannian fix (§§3 onward) replaces M with a function of position.
      </figcaption>
    </figure>
  );
}

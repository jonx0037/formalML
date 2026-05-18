// =============================================================================
// RiemannianHamiltonianFlow.tsx
//
// §4 The Riemannian Hamiltonian. Two coordinated panels showing continuous-time
// RMHMC dynamics on the banana — the "ground truth" against which §5's discrete
// generalized leapfrog will be compared:
//
//   (a) Banana log-density contour with one or more RMHMC trajectories
//       (RK4 on Hamilton's equations) from clickable starting points, plus
//       optional standard-HMC comparison trajectories (constant Minv, RK4)
//       from the same starts. Reader sees: RMHMC follows the ridge, HMC
//       overshoots and bounces.
//   (b) For the currently selected trajectory, three time series U(θ(t)),
//       K(θ(t), p(t)), V(θ(t)) plus their sum H — RK4 conserves H exactly
//       (panel-b H line is visibly flat), which is the §4 takeaway.
//
// On the banana, V(θ) = -log a is CONSTANT (det G = 1/a² identically), so
// the V time series is a flat line. That's the brief's §4.3 "V is free
// pedagogy on banana" remark made visual. §10's LCP exercises a non-trivial V.
//
// Controls: T ∈ [0.5, 6.0] integration time (display-vs-committed),
// click-to-place starting points (up to 3), HMC-comparison toggle.
//
// Computation: in-browser via shared/riemann-hmc.ts.
// Static fallback: /images/topics/riemann-manifold-hmc/04_hamiltonian_flow_and_energy.png
// =============================================================================

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bananaLogDensity,
  bananaHamiltonian,
  bananaGradU,
  bananaMetric,
  rmhmcRK4Step,
  standardLeapfrogStep,
  mulberry32,
  makeGaussian,
  paletteRMHMC,
} from './shared/riemann-hmc';

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const DEFAULT_T = 3;
const X_MIN = -3.5;
const X_MAX = 3.5;
const Y_MIN = -6;
const Y_MAX = 4;
const BANANA = { a: 1, b: 1 };
const SEED = 41;

const DEFAULT_STARTS: [number, number][] = [
  [-2, -3], // on ridge
  [0, 0.5], // near origin
  [1.5, -2], // descending arm
];

// -----------------------------------------------------------------------------
// Trajectory computation
// -----------------------------------------------------------------------------

interface TrajectoryStep {
  theta: [number, number];
  mom: [number, number];
  t: number;
  U: number;
  K: number;
  V: number;
  H: number;
}

function rmhmcTrajectory(theta0: [number, number], mom0: [number, number], T: number, nSteps: number): TrajectoryStep[] {
  const dt = T / nSteps;
  const path: TrajectoryStep[] = [];
  const h0 = bananaHamiltonian([theta0[0], theta0[1]], [mom0[0], mom0[1]], BANANA);
  path.push({
    theta: [theta0[0], theta0[1]],
    mom: [mom0[0], mom0[1]],
    t: 0,
    U: h0.U,
    K: h0.K,
    V: h0.V,
    H: h0.H,
  });
  let theta = theta0.slice();
  let mom = mom0.slice();
  for (let n = 1; n <= nSteps; n++) {
    const r = rmhmcRK4Step(theta, mom, dt, BANANA);
    theta = r.theta;
    mom = r.mom;
    const h = bananaHamiltonian(theta, mom, BANANA);
    path.push({
      theta: [theta[0], theta[1]],
      mom: [mom[0], mom[1]],
      t: n * dt,
      U: h.U,
      K: h.K,
      V: h.V,
      H: h.H,
    });
  }
  return path;
}

function hmcTrajectory(theta0: [number, number], mom0: [number, number], T: number, nSteps: number): [number, number][] {
  // Standard leapfrog with identity Minv (constant-mass HMC). Step size = T/nSteps.
  const dt = T / nSteps;
  const path: [number, number][] = [[theta0[0], theta0[1]]];
  let theta: number[] = theta0.slice();
  let mom: number[] = mom0.slice();
  for (let n = 1; n <= nSteps; n++) {
    const r = standardLeapfrogStep(theta, mom, dt, (th) => bananaGradU(th, BANANA), [1, 1]);
    theta = r.theta;
    mom = r.mom;
    path.push([theta[0], theta[1]]);
  }
  return path;
}

// -----------------------------------------------------------------------------
// Contour grid
// -----------------------------------------------------------------------------

interface ContourData {
  field: number[];
  nx: number;
  ny: number;
  thresholds: number[];
}

function makeBananaContours(): ContourData {
  const nx = 80;
  const ny = 100;
  const field: number[] = new Array(nx * ny);
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = 0; j < ny; j++) {
    const y = Y_MIN + ((Y_MAX - Y_MIN) * j) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / (nx - 1);
      const v = bananaLogDensity([x, y], BANANA);
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
// Main component
// -----------------------------------------------------------------------------

export default function RiemannianHamiltonianFlow() {
  const [displayT, setDisplayT] = useState(DEFAULT_T);
  const [committedT, setCommittedT] = useState(DEFAULT_T);
  const [starts, setStarts] = useState<[number, number][]>(DEFAULT_STARTS);
  const [showHMC, setShowHMC] = useState(true);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 800;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.floor((containerWidth - 16) / 2);
  const panelHeight = isMobile ? 320 : 380;

  const contours = useMemo(() => makeBananaContours(), []);

  // Compute trajectories from each start. Initial momentum drawn deterministically per start.
  const rmhmcPaths = useMemo(() => {
    const rng = mulberry32(SEED);
    const gauss = makeGaussian(rng);
    const nSteps = 300;
    return starts.map((th0) => {
      // Draw momentum from N(0, G(θ_0)) so the trajectory has a "typical" speed.
      const G = bananaMetric([th0[0], th0[1]], BANANA);
      // 2×2 Cholesky
      const l00 = Math.sqrt(G[0][0]);
      const l10 = G[1][0] / l00;
      const l11 = Math.sqrt(Math.max(G[1][1] - l10 * l10, 1e-12));
      const z0 = gauss();
      const z1 = gauss();
      const mom0: [number, number] = [l00 * z0, l10 * z0 + l11 * z1];
      return rmhmcTrajectory(th0, mom0, committedT, nSteps);
    });
  }, [starts, committedT]);

  const hmcPaths = useMemo(() => {
    if (!showHMC) return null;
    const rng = mulberry32(SEED + 1);
    const gauss = makeGaussian(rng);
    const nSteps = 300;
    return starts.map((th0) => {
      const mom0: [number, number] = [gauss(), gauss()];
      return hmcTrajectory(th0, mom0, committedT, nSteps);
    });
  }, [starts, committedT, showHMC]);

  // Refs for the two panels
  const trajRef = useRef<SVGSVGElement | null>(null);
  const energyRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    // ────────── Trajectory panel ──────────
    if (trajRef.current) {
      const svg = d3.select(trajRef.current);
      svg.selectAll('*').remove();
      const margin = { top: 30, right: 20, bottom: 40, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW > 0 && innerH > 0) {
        const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
        const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
        const yScale = d3.scaleLinear().domain([Y_MIN, Y_MAX]).range([innerH, 0]);

        // Contour fills
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

        // Axes
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
          .text('Continuous-time RMHMC vs HMC');
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

        // HMC trajectories (drawn first, behind RMHMC)
        if (hmcPaths) {
          const line = d3
            .line<[number, number]>()
            .x((p) => xScale(p[0]))
            .y((p) => yScale(p[1]));
          for (const path of hmcPaths) {
            const clipped = path.filter(
              (p) => p[0] >= X_MIN - 0.3 && p[0] <= X_MAX + 0.3 && p[1] >= Y_MIN - 0.5 && p[1] <= Y_MAX + 0.5,
            );
            g.append('path')
              .datum(clipped)
              .attr('d', line)
              .style('fill', 'none')
              .style('stroke', paletteRMHMC.hmc)
              .style('stroke-width', 1.2)
              .style('opacity', 0.7);
          }
        }

        // RMHMC trajectories
        const line = d3
          .line<TrajectoryStep>()
          .x((s) => xScale(s.theta[0]))
          .y((s) => yScale(s.theta[1]));
        rmhmcPaths.forEach((path, i) => {
          const clipped = path.filter(
            (s) =>
              s.theta[0] >= X_MIN - 0.3 &&
              s.theta[0] <= X_MAX + 0.3 &&
              s.theta[1] >= Y_MIN - 0.5 &&
              s.theta[1] <= Y_MAX + 0.5,
          );
          g.append('path')
            .datum(clipped)
            .attr('d', line)
            .style('fill', 'none')
            .style('stroke', paletteRMHMC.rmhmc)
            .style('stroke-width', i === selectedIdx ? 2.4 : 1.4)
            .style('opacity', i === selectedIdx ? 0.95 : 0.7);
          // Start marker
          g.append('circle')
            .attr('cx', xScale(starts[i][0]))
            .attr('cy', yScale(starts[i][1]))
            .attr('r', 4)
            .style('fill', i === selectedIdx ? paletteRMHMC.target : paletteRMHMC.rmhmc)
            .style('stroke', 'var(--color-surface, #fff)')
            .style('stroke-width', 1.5)
            .style('cursor', 'pointer')
            .on('click', () => setSelectedIdx(i));
        });

        // Click-to-add overlay
        g.append('rect')
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', innerW)
          .attr('height', innerH)
          .style('fill', 'none')
          .style('pointer-events', 'all')
          .on('click', function (event) {
            const [px, py] = d3.pointer(event);
            const t1 = xScale.invert(px);
            const t2 = yScale.invert(py);
            if (t1 >= X_MIN && t1 <= X_MAX && t2 >= Y_MIN && t2 <= Y_MAX) {
              setStarts((prev) => {
                const next = prev.length >= 3 ? prev.slice(1) : prev.slice();
                next.push([t1, t2]);
                return next;
              });
              setSelectedIdx(starts.length >= 3 ? 2 : starts.length);
            }
          })
          .lower(); // place behind the trajectories so they receive clicks first
      }
    }

    // ────────── Energy panel ──────────
    if (energyRef.current) {
      const svg = d3.select(energyRef.current);
      svg.selectAll('*').remove();
      const margin = { top: 30, right: 20, bottom: 40, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW > 0 && innerH > 0) {
        const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
        const path = rmhmcPaths[selectedIdx] ?? rmhmcPaths[0];
        if (path && path.length > 1) {
          const xScale = d3.scaleLinear().domain([0, committedT]).range([0, innerW]);
          // Y domain spans U, K, H magnitudes
          const allVals = path.flatMap((s) => [s.U, s.K, s.V, s.H]);
          const yMin = Math.min(...allVals);
          const yMax = Math.max(...allVals);
          const pad = 0.1 * (yMax - yMin + 1e-6);
          const yScale = d3.scaleLinear().domain([yMin - pad, yMax + pad]).range([innerH, 0]);

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
            .text(`Energy decomposition (trajectory ${selectedIdx + 1})`);
          g.append('text')
            .attr('x', innerW / 2)
            .attr('y', innerH + 32)
            .attr('text-anchor', 'middle')
            .style('font-size', '11px')
            .style('fill', 'var(--color-text-secondary, #6B6B6B)')
            .text('t');

          const seriesDefs: Array<{
            key: 'U' | 'K' | 'V' | 'H';
            color: string;
            label: string;
            dashed?: boolean;
            width?: number;
          }> = [
            { key: 'U', color: paletteRMHMC.hmc, label: 'U(θ)', width: 1.8 },
            { key: 'K', color: paletteRMHMC.rmhmc, label: 'K(θ,p)', width: 1.8 },
            { key: 'V', color: paletteRMHMC.metric, label: 'V(θ) = const', width: 1.4 },
            { key: 'H', color: paletteRMHMC.target, label: 'H = U+K+V', width: 1.6, dashed: true },
          ];
          for (const d of seriesDefs) {
            const line = d3
              .line<TrajectoryStep>()
              .x((s) => xScale(s.t))
              .y((s) => yScale(s[d.key]));
            g.append('path')
              .datum(path)
              .attr('d', line)
              .style('fill', 'none')
              .style('stroke', d.color)
              .style('stroke-width', d.width ?? 1.8)
              .style('stroke-dasharray', d.dashed ? '4 3' : 'none');
          }
          // Legend
          const legend = g.append('g').attr('transform', `translate(${innerW - 110}, 8)`);
          seriesDefs.forEach((d, i) => {
            legend
              .append('line')
              .attr('x1', 0)
              .attr('x2', 18)
              .attr('y1', i * 16)
              .attr('y2', i * 16)
              .style('stroke', d.color)
              .style('stroke-width', 1.8)
              .style('stroke-dasharray', d.dashed ? '4 3' : 'none');
            legend
              .append('text')
              .attr('x', 22)
              .attr('y', i * 16 + 4)
              .style('font-size', '10.5px')
              .style('fill', 'var(--color-text, #1A1A1A)')
              .text(d.label);
          });

          // Annotation: H conservation magnitude
          const H0 = path[0].H;
          const Hfinal = path[path.length - 1].H;
          const drift = Math.abs(Hfinal - H0);
          g.append('text')
            .attr('x', 8)
            .attr('y', 14)
            .style('font-size', '10.5px')
            .style('fill', 'var(--color-text-secondary, #6B6B6B)')
            .text(`|H(T) - H(0)| = ${drift.toExponential(2)}`);
        }
      }
    }
  }, [contours, rmhmcPaths, hmcPaths, panelWidth, panelHeight, selectedIdx, committedT, starts]);

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 4: Continuous-time RMHMC vs HMC on the banana, with energy decomposition"
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
          <span>integration time T:</span>
          <input
            type="range"
            min={0.5}
            max={6}
            step={0.1}
            value={displayT}
            onChange={(e) => setDisplayT(Number(e.target.value))}
            onMouseUp={(e) => setCommittedT(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedT(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedT(Number((e.target as HTMLInputElement).value))}
            style={{ width: '160px' }}
            aria-label="Integration time T"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>T = {displayT.toFixed(1)}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showHMC} onChange={(e) => setShowHMC(e.target.checked)} />
          <span>show HMC comparison</span>
        </label>
        <button
          type="button"
          onClick={() => {
            setStarts(DEFAULT_STARTS);
            setSelectedIdx(0);
          }}
          style={{
            fontSize: '0.85rem',
            padding: '0.2rem 0.6rem',
            border: '1px solid var(--color-border, #ccc)',
            background: 'var(--color-surface, #fff)',
            color: 'var(--color-text, #1A1A1A)',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          reset starts
        </button>
        <span style={{ color: 'var(--color-text-secondary, #6B6B6B)', fontSize: '0.8rem' }}>
          Click panel (a) to add a start (rolls oldest off).
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
        <svg ref={trajRef} width={panelWidth} height={panelHeight} role="img" aria-label="RMHMC and HMC trajectories on the banana" />
        <svg
          ref={energyRef}
          width={panelWidth}
          height={panelHeight}
          role="img"
          aria-label="Energy decomposition U, K, V, H along the selected trajectory"
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
        Figure 4. Left: continuous-time RMHMC trajectories (olive, RK4 on Hamilton&apos;s equations)
        follow the banana ridge; constant-mass HMC trajectories (blue) overshoot. Right:
        the selected RMHMC trajectory&apos;s U, K, V time series with their sum H. On the banana
        V = -log a is constant (V line is flat) and H is conserved to RK4 tolerance — the §4
        takeaway. Click the trajectory panel to add a new start, drag the slider to change T.
      </figcaption>
    </figure>
  );
}

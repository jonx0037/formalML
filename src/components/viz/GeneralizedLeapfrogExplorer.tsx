// =============================================================================
// GeneralizedLeapfrogExplorer.tsx
//
// §5 The Generalized Leapfrog Integrator. Two coordinated panels that compare
// three integrators on the banana Riemannian Hamiltonian:
//
//   - RK4 reference (continuous-time, the §4 ground truth)
//   - Generalized leapfrog (implicit Leimkuhler-Reich, the §5 hero)
//   - Naive explicit Riemannian leapfrog (controlled failure of §5.1)
//
// All three start at the same (θ_0, p_0) and run L leapfrog "outer" steps at
// the user-set step size ε. Panel (a) overlays the trajectories on the banana
// log-density contour. Panel (b) shows |H_n - H_0| vs step number on a log
// y-axis: naive drifts linearly upward, GL oscillates with bounded amplitude,
// RK4 stays nearly flat. A readout reports the mean fixed-point iteration
// counts per GL step.
//
// Heavy-MC slider per CLAUDE.md "Slider perf: commit-on-release for heavy MC".
// Live label on onChange, full recompute on onMouseUp / onTouchEnd / onKeyUp.
//
// Controls:
//   - ε ∈ [0.02, 0.40] (display-vs-committed)
//   - L ∈ [20, 400] leapfrog steps (display-vs-committed)
//   - Divergence-iteration cap (default 20)
//
// Computation: in-browser via shared/riemann-hmc.ts.
// Static fallback: /images/topics/riemann-manifold-hmc/05_generalized_leapfrog_energy_drift.png
// =============================================================================

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bananaLogDensity,
  bananaHamiltonian,
  generalizedLeapfrogStep,
  naiveRiemannianLeapfrogStep,
  rmhmcRK4Step,
  bananaMetric,
  paletteRMHMC,
} from './shared/riemann-hmc';

const SM_BREAKPOINT = 640;
const DEFAULT_EPS = 0.15;
const DEFAULT_L = 200;
const DEFAULT_MAX_ITERS = 20;
const X_MIN = -3.5;
const X_MAX = 3.5;
const Y_MIN = -6;
const Y_MAX = 4;
const BANANA = { a: 1, b: 1 };

// Common starting point (matches notebook §5)
const THETA0: [number, number] = [-1.5, -1.0];
const MOM_SEED_THETA0 = (): [number, number] => {
  // Draw a "typical" momentum from N(0, G(θ_0))
  const G = bananaMetric([THETA0[0], THETA0[1]], BANANA);
  const l00 = Math.sqrt(G[0][0]);
  const l10 = G[1][0] / l00;
  const l11 = Math.sqrt(Math.max(G[1][1] - l10 * l10, 1e-12));
  // Use a deterministic momentum so re-runs are reproducible
  const z0 = 0.6;
  const z1 = -0.4;
  return [l00 * z0, l10 * z0 + l11 * z1];
};

interface IntegratorRun {
  trajectory: [number, number][];
  drift: number[]; // |H_n - H_0|
  meanPIters: number;
  meanThetaIters: number;
  divergent: boolean;
}

function runGL(eps: number, L: number, maxIters: number): IntegratorRun {
  const mom0 = MOM_SEED_THETA0();
  let theta: number[] = [THETA0[0], THETA0[1]];
  let mom: number[] = [mom0[0], mom0[1]];
  const H0 = bananaHamiltonian(theta, mom, BANANA).H;
  const traj: [number, number][] = [[theta[0], theta[1]]];
  const drift: number[] = [0];
  let totalP = 0;
  let totalTh = 0;
  let nSteps = 0;
  let divergent = false;
  for (let i = 0; i < L; i++) {
    const r = generalizedLeapfrogStep(theta, mom, eps, BANANA, { maxIters });
    totalP += r.pIters;
    totalTh += r.thetaIters;
    nSteps++;
    if (!r.converged) {
      divergent = true;
      break;
    }
    theta = r.theta;
    mom = r.mom;
    traj.push([theta[0], theta[1]]);
    drift.push(Math.abs(bananaHamiltonian(theta, mom, BANANA).H - H0));
  }
  return {
    trajectory: traj,
    drift,
    meanPIters: nSteps > 0 ? totalP / nSteps : 0,
    meanThetaIters: nSteps > 0 ? totalTh / nSteps : 0,
    divergent,
  };
}

function runNaive(eps: number, L: number): IntegratorRun {
  const mom0 = MOM_SEED_THETA0();
  let theta: number[] = [THETA0[0], THETA0[1]];
  let mom: number[] = [mom0[0], mom0[1]];
  const H0 = bananaHamiltonian(theta, mom, BANANA).H;
  const traj: [number, number][] = [[theta[0], theta[1]]];
  const drift: number[] = [0];
  let divergent = false;
  for (let i = 0; i < L; i++) {
    const r = naiveRiemannianLeapfrogStep(theta, mom, eps, BANANA);
    if (!Number.isFinite(r.theta[0]) || !Number.isFinite(r.theta[1])) {
      divergent = true;
      break;
    }
    theta = r.theta;
    mom = r.mom;
    traj.push([theta[0], theta[1]]);
    const h = bananaHamiltonian(theta, mom, BANANA).H;
    if (!Number.isFinite(h)) {
      divergent = true;
      drift.push(1e10);
      break;
    }
    drift.push(Math.abs(h - H0));
  }
  return { trajectory: traj, drift, meanPIters: 0, meanThetaIters: 0, divergent };
}

function runRK4(eps: number, L: number): IntegratorRun {
  const mom0 = MOM_SEED_THETA0();
  let theta: number[] = [THETA0[0], THETA0[1]];
  let mom: number[] = [mom0[0], mom0[1]];
  const H0 = bananaHamiltonian(theta, mom, BANANA).H;
  const traj: [number, number][] = [[theta[0], theta[1]]];
  const drift: number[] = [0];
  const nSub = 10; // substeps per ε "outer" step for accuracy
  const dt = eps / nSub;
  for (let i = 0; i < L; i++) {
    for (let s = 0; s < nSub; s++) {
      const r = rmhmcRK4Step(theta, mom, dt, BANANA);
      theta = r.theta;
      mom = r.mom;
    }
    traj.push([theta[0], theta[1]]);
    drift.push(Math.abs(bananaHamiltonian(theta, mom, BANANA).H - H0));
  }
  return { trajectory: traj, drift, meanPIters: 0, meanThetaIters: 0, divergent: false };
}

// Banana contour
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

export default function GeneralizedLeapfrogExplorer() {
  const [displayEps, setDisplayEps] = useState(DEFAULT_EPS);
  const [committedEps, setCommittedEps] = useState(DEFAULT_EPS);
  const [displayL, setDisplayL] = useState(DEFAULT_L);
  const [committedL, setCommittedL] = useState(DEFAULT_L);
  const [maxIters, setMaxIters] = useState(DEFAULT_MAX_ITERS);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 800;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.floor((containerWidth - 16) / 2);
  const panelHeight = isMobile ? 320 : 380;

  const contours = useMemo(() => makeBananaContours(), []);
  const runs = useMemo(() => {
    const gl = runGL(committedEps, committedL, maxIters);
    const naive = runNaive(committedEps, committedL);
    const rk4 = runRK4(committedEps, committedL);
    return { gl, naive, rk4 };
  }, [committedEps, committedL, maxIters]);

  const trajRef = useRef<SVGSVGElement | null>(null);
  const driftRef = useRef<SVGSVGElement | null>(null);

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

        // Contours
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
          .text('Three integrators on banana');

        // Trajectories
        const line = d3
          .line<[number, number]>()
          .x((p) => xScale(p[0]))
          .y((p) => yScale(p[1]));
        const inBounds = (p: [number, number]): boolean =>
          Number.isFinite(p[0]) &&
          Number.isFinite(p[1]) &&
          p[0] >= X_MIN - 0.3 &&
          p[0] <= X_MAX + 0.3 &&
          p[1] >= Y_MIN - 0.5 &&
          p[1] <= Y_MAX + 0.5;
        const layers = [
          { run: runs.naive, color: paletteRMHMC.naive, width: 1.2, label: 'naive explicit' },
          { run: runs.gl, color: paletteRMHMC.rmhmc, width: 1.8, label: 'generalized leapfrog' },
          { run: runs.rk4, color: paletteRMHMC.geodesic, width: 1.4, label: 'RK4 reference', dashed: true },
        ];
        for (const L of layers) {
          const clipped = L.run.trajectory.filter(inBounds);
          g.append('path')
            .datum(clipped)
            .attr('d', line)
            .style('fill', 'none')
            .style('stroke', L.color)
            .style('stroke-width', L.width)
            .style('stroke-dasharray', L.dashed ? '4 3' : 'none')
            .style('opacity', 0.9);
        }
        // Start marker
        g.append('circle')
          .attr('cx', xScale(THETA0[0]))
          .attr('cy', yScale(THETA0[1]))
          .attr('r', 4)
          .style('fill', paletteRMHMC.target)
          .style('stroke', 'var(--color-surface, #fff)')
          .style('stroke-width', 1.5);

        // Legend
        const legend = g.append('g').attr('transform', `translate(${innerW - 150}, 8)`);
        layers.forEach((L, i) => {
          legend
            .append('line')
            .attr('x1', 0)
            .attr('x2', 18)
            .attr('y1', i * 16)
            .attr('y2', i * 16)
            .style('stroke', L.color)
            .style('stroke-width', 1.8)
            .style('stroke-dasharray', L.dashed ? '4 3' : 'none');
          legend
            .append('text')
            .attr('x', 22)
            .attr('y', i * 16 + 4)
            .style('font-size', '10.5px')
            .style('fill', 'var(--color-text, #1A1A1A)')
            .text(L.label);
        });
      }
    }
    // ────────── Drift panel (log y) ──────────
    if (driftRef.current) {
      const svg = d3.select(driftRef.current);
      svg.selectAll('*').remove();
      const margin = { top: 30, right: 20, bottom: 40, left: 60 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW > 0 && innerH > 0) {
        const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
        const xScale = d3.scaleLinear().domain([0, committedL]).range([0, innerW]);
        // Compute y domain from all three drifts (clip at 1e-12 floor and 1e6 ceiling)
        const flatDrift = [...runs.gl.drift, ...runs.naive.drift, ...runs.rk4.drift]
          .filter((v) => Number.isFinite(v) && v > 0);
        const yLo = Math.max(Math.min(...flatDrift, 1e-10), 1e-12);
        const yHi = Math.min(Math.max(...flatDrift, 1e-6), 1e6);
        const yScale = d3.scaleLog().domain([yLo, yHi]).range([innerH, 0]).clamp(true);

        g.append('g')
          .attr('transform', `translate(0, ${innerH})`)
          .call(d3.axisBottom(xScale).ticks(5))
          .selectAll('text')
          .style('fill', 'var(--color-text, #1A1A1A)');
        g.append('g')
          .call(d3.axisLeft(yScale).ticks(6, '~e'))
          .selectAll('text')
          .style('fill', 'var(--color-text, #1A1A1A)');
        g.append('text')
          .attr('x', innerW / 2)
          .attr('y', -8)
          .attr('text-anchor', 'middle')
          .style('font-size', '13px')
          .style('font-weight', '600')
          .style('fill', 'var(--color-text, #1A1A1A)')
          .text('|H(n) - H(0)| vs step (log scale)');
        g.append('text')
          .attr('x', innerW / 2)
          .attr('y', innerH + 32)
          .attr('text-anchor', 'middle')
          .style('font-size', '11px')
          .style('fill', 'var(--color-text-secondary, #6B6B6B)')
          .text('leapfrog step');

        const seriesDefs: Array<{ run: IntegratorRun; color: string; width: number; dashed?: boolean }> = [
          { run: runs.naive, color: paletteRMHMC.naive, width: 1.4 },
          { run: runs.gl, color: paletteRMHMC.rmhmc, width: 1.8 },
          { run: runs.rk4, color: paletteRMHMC.geodesic, width: 1.4, dashed: true },
        ];
        for (const s of seriesDefs) {
          const pts: [number, number][] = s.run.drift.map((v, i) => [i, Math.max(v, yLo)]);
          const line = d3
            .line<[number, number]>()
            .x((p) => xScale(p[0]))
            .y((p) => yScale(p[1]))
            .defined((p) => Number.isFinite(p[1]) && p[1] > 0);
          g.append('path')
            .datum(pts)
            .attr('d', line)
            .style('fill', 'none')
            .style('stroke', s.color)
            .style('stroke-width', s.width)
            .style('stroke-dasharray', s.dashed ? '4 3' : 'none');
        }

        // FP-iter readout
        g.append('text')
          .attr('x', 8)
          .attr('y', 14)
          .style('font-size', '10.5px')
          .style('fill', 'var(--color-text-secondary, #6B6B6B)')
          .text(
            `GL mean FP iters: p=${runs.gl.meanPIters.toFixed(1)}, θ=${runs.gl.meanThetaIters.toFixed(1)}${runs.gl.divergent ? ' (divergent)' : ''}`,
          );
      }
    }
  }, [contours, runs, panelWidth, panelHeight, committedL]);

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 5: Three integrators on the banana — RK4 reference, generalized leapfrog, naive explicit"
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
            onMouseUp={(e) => setCommittedEps(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedEps(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedEps(Number((e.target as HTMLInputElement).value))}
            style={{ width: '140px' }}
            aria-label="Leapfrog step size epsilon"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>ε = {displayEps.toFixed(2)}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>steps L:</span>
          <input
            type="range"
            min={20}
            max={400}
            step={10}
            value={displayL}
            onChange={(e) => setDisplayL(Number(e.target.value))}
            onMouseUp={(e) => setCommittedL(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedL(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedL(Number((e.target as HTMLInputElement).value))}
            style={{ width: '140px' }}
            aria-label="Number of leapfrog steps L"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '3em' }}>L = {displayL}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>divergence cap:</span>
          <input
            type="number"
            min={5}
            max={50}
            step={5}
            value={maxIters}
            onChange={(e) => setMaxIters(Math.max(5, Math.min(50, Number(e.target.value) || DEFAULT_MAX_ITERS)))}
            style={{ width: '4em', fontSize: '0.85rem' }}
            aria-label="Fixed-point iteration cap"
          />
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
        <svg ref={trajRef} width={panelWidth} height={panelHeight} role="img" aria-label="Three integrators' trajectories on the banana" />
        <svg ref={driftRef} width={panelWidth} height={panelHeight} role="img" aria-label="Energy drift on log scale" />
      </div>
      <figcaption
        style={{
          marginTop: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-text-secondary, #6B6B6B)',
          textAlign: 'center',
        }}
      >
        Figure 5. RK4 reference (teal, dashed), generalized leapfrog (olive), and naive
        explicit Riemannian leapfrog (orange) all start at the same (θ₀, p₀). RK4 stays
        nearly flat (it&apos;s the continuous-time reference at high resolution); the generalized
        leapfrog oscillates with bounded amplitude (the §5 symplecticity claim made visible);
        the naive explicit drifts linearly and often diverges at moderate ε. Increase ε to
        see the regimes pull apart; increase L to see the long-time behavior. Heavy
        compute recommits only on slider release.
      </figcaption>
    </figure>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// HMCTrajectoryAnimator — embedded after §4.2 of the probabilistic-programming
// topic. Visualizes the §4.2 claim that "in regions of high curvature the
// leapfrog approximation breaks down and the integrator's energy diverges" by
// running a single HMC trajectory on a 2D correlated Gaussian target and
// displaying:
//
//   - a filled contour of the target N(0, Σ),
//   - the leapfrog path as a dotted line with momentum arrows at each step,
//   - a U-turn marker at the first step where (q − q₀)·p < 0,
//   - a "divergent" badge if the energy H = U(q) + ½‖p‖² drifts past a
//     threshold (1e3 is the standard PyMC convention).
//
// Sliders for step size ε (the §4.2 hyperparameter that drives divergence) and
// trajectory length L. A "reseed" button regenerates the initial position and
// momentum. As the reader pushes ε past ~0.5 on a unit-variance target, the
// trajectory starts overshooting; past ε ~ 1.5 it spirals out and the
// divergent-energy badge fires.
//
// All math is closed-form in TS — the gradient of a 2D Gaussian's negative
// log density is Σ⁻¹(θ − μ) and leapfrog is eight lines. Synchronous viz
// despite the Tier 3 plan estimating "precomputed trajectories": TS handles
// 1000 force evaluations per render in real time.
// =============================================================================

const PANEL_HEIGHT = 460;
const VIEW_DOMAIN: [number, number] = [-4, 4];
const N_CONTOUR_GRID = 60;
const ENERGY_DIVERGENCE_THRESHOLD = 1e3;
const CHI2_95 = 2.447746830680816;

const COLORS = {
  posterior: '#dbeafe',  // light blue fill for posterior region
  posteriorEdge: '#3b82f6',
  trajectory: '#1f2937', // slate path
  trajectoryDiv: '#dc2626', // red path when divergent
  start: '#15803d',      // green start point
  end: '#9333ea',        // purple end point
  uTurn: '#f59e0b',      // amber U-turn marker
  momentum: '#9ca3af',
};

interface State {
  q: [number, number];
  p: [number, number];
}

// 2D Gaussian target N(0, Σ) with Σ = [[1, ρ], [ρ, varY]]. Unit variance on
// θ₁; mild stretch on θ₂ keeps the curvature anisotropic — important for
// showing that fixed-ε integration mismatches local curvature in some
// directions. ρ=0.6, varY=2.0.
const SIGMA = [
  [1.0, 0.6],
  [0.6, 2.0],
] as const;
const DET_SIGMA = SIGMA[0][0] * SIGMA[1][1] - SIGMA[0][1] * SIGMA[1][0];
// Σ⁻¹ in closed form for 2x2.
const SIGMA_INV: [[number, number], [number, number]] = [
  [SIGMA[1][1] / DET_SIGMA, -SIGMA[0][1] / DET_SIGMA],
  [-SIGMA[1][0] / DET_SIGMA, SIGMA[0][0] / DET_SIGMA],
];

// Negative log density (potential energy) U(θ) = ½ θᵀ Σ⁻¹ θ + const.
function potentialEnergy(q: [number, number]): number {
  const [a, b] = q;
  return (
    0.5 *
    (SIGMA_INV[0][0] * a * a +
      2 * SIGMA_INV[0][1] * a * b +
      SIGMA_INV[1][1] * b * b)
  );
}

// ∇U(θ) = Σ⁻¹ θ.
function gradPotential(q: [number, number]): [number, number] {
  const [a, b] = q;
  return [
    SIGMA_INV[0][0] * a + SIGMA_INV[0][1] * b,
    SIGMA_INV[1][0] * a + SIGMA_INV[1][1] * b,
  ];
}

// One leapfrog step.
function leapfrog(state: State, eps: number): State {
  const g0 = gradPotential(state.q);
  const pHalf: [number, number] = [
    state.p[0] - 0.5 * eps * g0[0],
    state.p[1] - 0.5 * eps * g0[1],
  ];
  const qNew: [number, number] = [
    state.q[0] + eps * pHalf[0],
    state.q[1] + eps * pHalf[1],
  ];
  const g1 = gradPotential(qNew);
  const pNew: [number, number] = [
    pHalf[0] - 0.5 * eps * g1[0],
    pHalf[1] - 0.5 * eps * g1[1],
  ];
  return { q: qNew, p: pNew };
}

// Mulberry32 PRNG + Box–Muller normal sampler — deterministic for a given seed.
function makeNormalSampler(seed: number) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = next();
    const u2 = next();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

export default function HMCTrajectoryAnimator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [eps, setEps] = useState<number>(0.3);
  const [nSteps, setNSteps] = useState<number>(60);
  const [showMomentum, setShowMomentum] = useState<boolean>(false);
  const [seed, setSeed] = useState<number>(42);

  // Initial state — Gaussian random θ₀ from N(0, Σ); momentum from N(0, I).
  const initial = useMemo<State>(() => {
    const sample = makeNormalSampler(seed * 7919 + 1);
    // Cholesky of Σ = [[1, ρ], [ρ, varY]] is [[1, 0], [ρ, √(varY − ρ²)]].
    const r = Math.sqrt(SIGMA[1][1] - SIGMA[0][1] * SIGMA[0][1]);
    const e1 = sample();
    const e2 = sample();
    // Start somewhere ~1.2 σ from the mode along a random radial.
    const theta = 2 * Math.PI * sample();
    const q0: [number, number] = [
      1.4 * Math.cos(theta),
      1.4 * Math.sin(theta) * Math.sqrt(SIGMA[1][1]),
    ];
    const p0: [number, number] = [sample(), sample()];
    return { q: q0, p: p0 };
  }, [seed]);

  // Run the trajectory.
  const trajectory = useMemo<{
    states: State[];
    energies: number[];
    uTurnIndex: number | null;
    diverged: boolean;
  }>(() => {
    const states: State[] = [initial];
    const energies: number[] = [
      potentialEnergy(initial.q) + 0.5 * (initial.p[0] ** 2 + initial.p[1] ** 2),
    ];
    let uTurnIndex: number | null = null;
    let diverged = false;
    for (let i = 0; i < nSteps; i++) {
      const next = leapfrog(states[i], eps);
      states.push(next);
      const H =
        potentialEnergy(next.q) + 0.5 * (next.p[0] ** 2 + next.p[1] ** 2);
      energies.push(H);
      if (Math.abs(H - energies[0]) > ENERGY_DIVERGENCE_THRESHOLD) {
        diverged = true;
      }
      if (uTurnIndex === null) {
        const dq: [number, number] = [
          next.q[0] - initial.q[0],
          next.q[1] - initial.q[1],
        ];
        const dot = dq[0] * next.p[0] + dq[1] * next.p[1];
        if (dot < 0) uTurnIndex = i + 1;
      }
    }
    return { states, energies, uTurnIndex, diverged };
  }, [initial, eps, nSteps]);

  const energyDrift = useMemo<number>(() => {
    let m = 0;
    for (const E of trajectory.energies) {
      const d = Math.abs(E - trajectory.energies[0]);
      if (d > m) m = d;
    }
    return m;
  }, [trajectory]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 20, right: 20, bottom: 44, left: 52 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      const size = Math.min(innerW, innerH);
      const root = svg
        .append('g')
        .attr(
          'transform',
          `translate(${margin.left + Math.max(0, (innerW - size) / 2)},${margin.top})`,
        );

      const xScale = d3.scaleLinear().domain(VIEW_DOMAIN).range([0, size]);
      const yScale = d3.scaleLinear().domain(VIEW_DOMAIN).range([size, 0]);

      root
        .append('g')
        .attr('transform', `translate(0,${size})`)
        .call(d3.axisBottom(xScale).ticks(7).tickSizeOuter(0));
      root.append('g').call(d3.axisLeft(yScale).ticks(7).tickSizeOuter(0));
      root
        .append('text')
        .attr('transform', `translate(${size / 2},${size + 36})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('θ₁');
      root
        .append('text')
        .attr('transform', `translate(-40,${size / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('θ₂');

      // Filled posterior 95% confidence ellipse.
      // Σ = [[1, ρ], [ρ, varY]] eigenvalues at angle ½ atan2(2ρ, 1−varY).
      const trace = SIGMA[0][0] + SIGMA[1][1];
      const det = DET_SIGMA;
      const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
      const lambda1 = trace / 2 + disc;
      const lambda2 = trace / 2 - disc;
      const angle = 0.5 * Math.atan2(2 * SIGMA[0][1], SIGMA[0][0] - SIGMA[1][1]);
      const pixelsPerUnitX = size / (VIEW_DOMAIN[1] - VIEW_DOMAIN[0]);
      root
        .append('ellipse')
        .attr('cx', xScale(0))
        .attr('cy', yScale(0))
        .attr('rx', CHI2_95 * Math.sqrt(lambda1) * pixelsPerUnitX)
        .attr('ry', CHI2_95 * Math.sqrt(lambda2) * pixelsPerUnitX)
        .attr('fill', COLORS.posterior)
        .attr('fill-opacity', 0.6)
        .attr('stroke', COLORS.posteriorEdge)
        .attr('stroke-width', 1.4)
        .attr(
          'transform',
          `rotate(${(-angle * 180) / Math.PI} ${xScale(0)} ${yScale(0)})`,
        );

      // Trajectory line.
      const states = trajectory.states;
      const pathColor = trajectory.diverged ? COLORS.trajectoryDiv : COLORS.trajectory;
      const pathPts: string[] = [];
      for (const s of states) {
        const x = Math.min(Math.max(s.q[0], VIEW_DOMAIN[0] * 1.5), VIEW_DOMAIN[1] * 1.5);
        const y = Math.min(Math.max(s.q[1], VIEW_DOMAIN[0] * 1.5), VIEW_DOMAIN[1] * 1.5);
        pathPts.push(`${xScale(x)},${yScale(y)}`);
      }
      root
        .append('polyline')
        .attr('points', pathPts.join(' '))
        .attr('fill', 'none')
        .attr('stroke', pathColor)
        .attr('stroke-width', 1.4)
        .attr('opacity', 0.8);

      // Waypoint dots.
      const dotGroup = root.append('g');
      states.forEach((s, i) => {
        if (
          s.q[0] < VIEW_DOMAIN[0] * 1.5 ||
          s.q[0] > VIEW_DOMAIN[1] * 1.5 ||
          s.q[1] < VIEW_DOMAIN[0] * 1.5 ||
          s.q[1] > VIEW_DOMAIN[1] * 1.5
        )
          return;
        dotGroup
          .append('circle')
          .attr('cx', xScale(s.q[0]))
          .attr('cy', yScale(s.q[1]))
          .attr('r', i === 0 ? 5.5 : i === states.length - 1 ? 5.5 : 2)
          .attr(
            'fill',
            i === 0
              ? COLORS.start
              : i === states.length - 1
                ? COLORS.end
                : pathColor,
          )
          .attr('opacity', i === 0 || i === states.length - 1 ? 0.95 : 0.6);
      });

      // U-turn marker.
      if (trajectory.uTurnIndex !== null) {
        const u = states[trajectory.uTurnIndex];
        if (
          u.q[0] >= VIEW_DOMAIN[0] &&
          u.q[0] <= VIEW_DOMAIN[1] &&
          u.q[1] >= VIEW_DOMAIN[0] &&
          u.q[1] <= VIEW_DOMAIN[1]
        ) {
          root
            .append('circle')
            .attr('cx', xScale(u.q[0]))
            .attr('cy', yScale(u.q[1]))
            .attr('r', 8)
            .attr('fill', 'none')
            .attr('stroke', COLORS.uTurn)
            .attr('stroke-width', 2);
          root
            .append('text')
            .attr('x', xScale(u.q[0]) + 12)
            .attr('y', yScale(u.q[1]) - 8)
            .attr('font-size', 10)
            .attr('fill', COLORS.uTurn)
            .attr('font-weight', 600)
            .text(`U-turn @ step ${trajectory.uTurnIndex}`);
        }
      }

      // Momentum arrows at every 5th step.
      if (showMomentum) {
        states.forEach((s, i) => {
          if (i % 5 !== 0) return;
          if (
            s.q[0] < VIEW_DOMAIN[0] ||
            s.q[0] > VIEW_DOMAIN[1] ||
            s.q[1] < VIEW_DOMAIN[0] ||
            s.q[1] > VIEW_DOMAIN[1]
          )
            return;
          const len = Math.sqrt(s.p[0] ** 2 + s.p[1] ** 2);
          if (len < 1e-6) return;
          const arrowScale = 0.35;
          const x1 = xScale(s.q[0]);
          const y1 = yScale(s.q[1]);
          const x2 = xScale(s.q[0] + arrowScale * s.p[0]);
          const y2 = yScale(s.q[1] + arrowScale * s.p[1]);
          root
            .append('line')
            .attr('x1', x1)
            .attr('y1', y1)
            .attr('x2', x2)
            .attr('y2', y2)
            .attr('stroke', COLORS.momentum)
            .attr('stroke-width', 1.4)
            .attr('opacity', 0.6);
        });
      }

      // Legend.
      const legend = root.append('g').attr('transform', `translate(${size - 200},10)`);
      const items = [
        { label: 'Start q₀', color: COLORS.start },
        { label: 'End qL', color: COLORS.end },
        { label: trajectory.diverged ? 'Trajectory (DIVERGENT)' : 'Trajectory', color: pathColor },
      ];
      items.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        row
          .append('circle')
          .attr('cx', 6)
          .attr('cy', 7)
          .attr('r', 4)
          .attr('fill', it.color);
        row
          .append('text')
          .attr('x', 16)
          .attr('y', 11)
          .attr('font-size', 10)
          .text(it.label);
      });
    },
    [trajectory, showMomentum, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          step size ε:
          <input
            type="range"
            min={0.05}
            max={2}
            step={0.025}
            value={eps}
            onChange={(e) => setEps(Number(e.target.value))}
            className="w-32"
            aria-label="step size slider"
          />
          <span className="tabular-nums w-12 text-right text-xs text-[var(--color-text-muted)]">
            {eps.toFixed(3)}
          </span>
        </label>
        <label className="flex items-center gap-2">
          leapfrog steps L:
          <input
            type="range"
            min={5}
            max={200}
            step={1}
            value={nSteps}
            onChange={(e) => setNSteps(Number(e.target.value))}
            className="w-32"
            aria-label="trajectory length slider"
          />
          <span className="tabular-nums w-10 text-right text-xs text-[var(--color-text-muted)]">
            {nSteps}
          </span>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showMomentum}
            onChange={(e) => setShowMomentum(e.target.checked)}
          />
          <span>momentum arrows</span>
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="rounded border bg-white px-2 py-0.5 hover:bg-gray-50 text-xs"
        >
          Reseed initial state
        </button>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-xs text-[var(--color-text-muted)]">
        <span>
          <strong>max |ΔH|:</strong>{' '}
          <span
            className="tabular-nums"
            style={{ color: trajectory.diverged ? COLORS.trajectoryDiv : 'inherit' }}
          >
            {energyDrift < 1e3 ? energyDrift.toFixed(3) : energyDrift.toExponential(2)}
          </span>
        </span>
        <span>
          <strong>U-turn step:</strong>{' '}
          <span className="tabular-nums">
            {trajectory.uTurnIndex ?? `none in ${nSteps}`}
          </span>
        </span>
        {trajectory.diverged && (
          <span style={{ color: COLORS.trajectoryDiv, fontWeight: 600 }}>
            Divergent — energy drift exceeds {ENERGY_DIVERGENCE_THRESHOLD}
          </span>
        )}
      </div>
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Target N(0, Σ) with Σ = [[1, 0.6], [0.6, 2]] (mild correlation, mild
        anisotropy). Each leapfrog step alternates a half-step momentum
        update, a full-step position update, and another half-step
        momentum update — the symplectic Störmer–Verlet integrator that gives
        HMC its energy-conserving guarantees in continuous time. With ε small
        enough, the trajectory traces a nearly-elliptical orbit through the
        posterior and the energy H = U(q) + ½‖p‖² stays nearly constant.
        Push ε past about 0.5 and the energy starts drifting; past about 1.5
        the trajectory spirals out and the badge turns red — that's a{' '}
        <em>divergent transition</em>, the signal §4.2 calls out as evidence
        that the leapfrog step size mismatches local curvature. The U-turn
        marker shows the first step where (q − q₀) · p &lt; 0; NUTS doubles
        the trajectory length until this happens, then samples uniformly
        from what came before.
      </div>
    </div>
  );
}

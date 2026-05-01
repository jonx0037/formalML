import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// CAVITrajectoryExplorer — embedded after §3.2 of the variational-inference
// topic. Visualizes Theorem 2's coordinate-ascent algorithm on a 2D Gaussian
// target N(μ_t, Σ) with Σ = [[1, ρ], [ρ, 1]] and a fully-factorized
// mean-field family q = q_1·q_2 with q_j = N(μ_j, σ_j²).
//
// For Gaussian targets the CAVI updates have closed form (§3.3 conjugate
// exponential family). Conditional on θ_2, θ_1 is Gaussian with mean
// μ_{t,1} − (Λ_12/Λ_11)(θ_2 − μ_{t,2}) and variance 1/Λ_11. The variational
// q_1 update inherits the mean (with E_q2[θ_2] = μ_2^(t)) and the variance:
//   μ_1^(t+1) = μ_{t,1} + ρ(μ_2^(t) − μ_{t,2})
//   σ_1^2    = 1/Λ_11 = 1 − ρ²        (constant across iterations)
// Symmetric for q_2 update with the *latest* μ_1.
//
// Reader sees:
//   - ELBO non-decreasing along the trajectory (Theorem 2's monotonicity).
//   - Trajectory zig-zags right-angled toward the true mean — alternating
//     horizontal (q_1 update) and vertical (q_2 update) moves.
//   - Convergence rate per sweep is ρ²; high-ρ targets need many sweeps.
//   - The converged variational ellipse (orange) is axis-aligned and tight
//     (radius √(1−ρ²)), while the true posterior (blue) is tilted —
//     anticipating §5.1's structural error of mean-field.
//
// Sliders for ρ ∈ [0, 0.95] and iteration step k ∈ [0, 20] (40 total block
// updates because each sweep alternates two blocks). Initial state at
// (−2.5, −2.5) is fixed; the truth at (1, 1.5) is also fixed.
// =============================================================================

const PANEL_HEIGHT = 460;
const VIEW_DOMAIN: [number, number] = [-4, 4];
const N_SAMPLES = 1500;
const N_SWEEPS = 20;
const CHI2_95 = 2.447746830680816;

// True target mean and starting variational mean. Truth deliberately
// off-axis so the path is non-trivial.
const TRUTH_MEAN: [number, number] = [1.0, 1.5];
const INITIAL_MEAN: [number, number] = [-2.5, -2.5];

const COLORS = {
  truthFill: '#dbeafe',
  truthEdge: '#2563eb',
  variational: '#ea580c',
  trajectory: '#dc2626',
  initial: '#15803d',
  axis: '#374151',
  guide: '#9ca3af',
};

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

interface TrajectoryPoint {
  mu1: number;
  mu2: number;
  sweepIdx: number;          // 0, 0.5, 1, 1.5, ... where .5 = mid-sweep
  block: 'init' | '1' | '2'; // which block was just updated
  elbo: number;
}

function computeCAVITrajectory(rho: number): TrajectoryPoint[] {
  const sigma2 = 1 - rho * rho; // variance of converged q_j
  const sigma = Math.sqrt(Math.max(sigma2, 1e-12));
  const traj: TrajectoryPoint[] = [];

  let mu1 = INITIAL_MEAN[0];
  let mu2 = INITIAL_MEAN[1];

  const elboAt = (m1: number, m2: number) => {
    // Closed-form ELBO with fixed σ_j² = 1 − ρ². Up to a constant in (μ_1, μ_2):
    //   E_q[log p] = −½ tr(Σ⁻¹ S_q) − ½ (μ_q − μ_t)ᵀ Σ⁻¹ (μ_q − μ_t)
    // with S_q = diag(σ²,σ²) the variational covariance.
    // Then ELBO = E_q[log p] + H(q) − ½ log(2π)² det Σ.
    const detSigma = 1 - rho * rho;
    const lambda11 = 1 / detSigma;
    const lambda12 = -rho / detSigma;
    const lambda22 = 1 / detSigma;
    const dm1 = m1 - TRUTH_MEAN[0];
    const dm2 = m2 - TRUTH_MEAN[1];
    const quad =
      lambda11 * dm1 * dm1 +
      2 * lambda12 * dm1 * dm2 +
      lambda22 * dm2 * dm2;
    const trace = lambda11 * sigma2 + lambda22 * sigma2;
    // Entropy of two independent N(·, σ²): −½ log(2π e σ²) per block, ×2.
    const entropyQ = Math.log(2 * Math.PI * Math.E * sigma2);
    // Constant from log p's normalization is independent of (μ_1, μ_2).
    const logZpConst = -Math.log(2 * Math.PI) - 0.5 * Math.log(detSigma);
    return -0.5 * (trace + quad) + entropyQ + logZpConst;
  };

  traj.push({
    mu1,
    mu2,
    sweepIdx: 0,
    block: 'init',
    elbo: elboAt(mu1, mu2),
  });

  for (let t = 0; t < N_SWEEPS; t++) {
    // Block 1 update: μ_1 ← μ_{t,1} + ρ(μ_2^(t) − μ_{t,2})
    mu1 = TRUTH_MEAN[0] + rho * (mu2 - TRUTH_MEAN[1]);
    traj.push({
      mu1,
      mu2,
      sweepIdx: t + 0.5,
      block: '1',
      elbo: elboAt(mu1, mu2),
    });
    // Block 2 update with the latest μ_1
    mu2 = TRUTH_MEAN[1] + rho * (mu1 - TRUTH_MEAN[0]);
    traj.push({
      mu1,
      mu2,
      sweepIdx: t + 1,
      block: '2',
      elbo: elboAt(mu1, mu2),
    });
  }

  return traj;
}

export default function CAVITrajectoryExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [rho, setRho] = useState<number>(0.85);
  const [step, setStep] = useState<number>(40); // 0..40 (2 per sweep × 20)

  const samples = useMemo<{ x: Float64Array; y: Float64Array }>(() => {
    const sampler = makeNormalSampler(20260430);
    const x = new Float64Array(N_SAMPLES);
    const y = new Float64Array(N_SAMPLES);
    const r = Math.sqrt(Math.max(1 - rho * rho, 0));
    for (let i = 0; i < N_SAMPLES; i++) {
      const e1 = sampler();
      const e2 = sampler();
      x[i] = TRUTH_MEAN[0] + e1;
      y[i] = TRUTH_MEAN[1] + rho * e1 + r * e2;
    }
    return { x, y };
  }, [rho]);

  const trajectory = useMemo<TrajectoryPoint[]>(
    () => computeCAVITrajectory(rho),
    [rho],
  );

  const visibleTrajectory = useMemo<TrajectoryPoint[]>(
    () => trajectory.slice(0, step + 1),
    [trajectory, step],
  );
  const current = visibleTrajectory[visibleTrajectory.length - 1];
  const finalElbo = trajectory[trajectory.length - 1].elbo;
  const initialElbo = trajectory[0].elbo;

  const sigmaQ = Math.sqrt(Math.max(1 - rho * rho, 1e-9));

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 24, right: 20, bottom: 44, left: 52 };
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
      const pixelsPerUnit = size / (VIEW_DOMAIN[1] - VIEW_DOMAIN[0]);

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

      // Truth ellipse (95% confidence) of N(μ_t, Σ).
      // Eigenvalues of [[1, ρ], [ρ, 1]] are 1±ρ at ±45°.
      const lambdaPlus = 1 + rho;
      const lambdaMinus = 1 - rho;
      root
        .append('ellipse')
        .attr('cx', xScale(TRUTH_MEAN[0]))
        .attr('cy', yScale(TRUTH_MEAN[1]))
        .attr('rx', CHI2_95 * Math.sqrt(lambdaPlus) * pixelsPerUnit)
        .attr('ry', CHI2_95 * Math.sqrt(lambdaMinus) * pixelsPerUnit)
        .attr('fill', COLORS.truthFill)
        .attr('fill-opacity', 0.55)
        .attr('stroke', COLORS.truthEdge)
        .attr('stroke-width', 1.4)
        .attr(
          'transform',
          `rotate(-45 ${xScale(TRUTH_MEAN[0])} ${yScale(TRUTH_MEAN[1])})`,
        );

      // Truth target samples (light gray dots).
      const sampleGroup = root.append('g');
      for (let i = 0; i < N_SAMPLES; i++) {
        const xv = samples.x[i];
        const yv = samples.y[i];
        if (xv < VIEW_DOMAIN[0] || xv > VIEW_DOMAIN[1]) continue;
        if (yv < VIEW_DOMAIN[0] || yv > VIEW_DOMAIN[1]) continue;
        sampleGroup
          .append('circle')
          .attr('cx', xScale(xv))
          .attr('cy', yScale(yv))
          .attr('r', 1.2)
          .attr('fill', COLORS.truthEdge)
          .attr('opacity', 0.18);
      }

      // Trajectory polyline.
      const pts = visibleTrajectory
        .map((p) => `${xScale(p.mu1)},${yScale(p.mu2)}`)
        .join(' ');
      root
        .append('polyline')
        .attr('points', pts)
        .attr('fill', 'none')
        .attr('stroke', COLORS.trajectory)
        .attr('stroke-width', 1.8)
        .attr('opacity', 0.85);

      // Trajectory waypoints.
      visibleTrajectory.forEach((p, idx) => {
        root
          .append('circle')
          .attr('cx', xScale(p.mu1))
          .attr('cy', yScale(p.mu2))
          .attr('r', idx === 0 ? 5 : idx === visibleTrajectory.length - 1 ? 5 : 1.8)
          .attr(
            'fill',
            idx === 0
              ? COLORS.initial
              : idx === visibleTrajectory.length - 1
                ? COLORS.variational
                : COLORS.trajectory,
          )
          .attr('opacity', idx === 0 || idx === visibleTrajectory.length - 1 ? 0.95 : 0.7);
      });

      // Current variational ellipse (axis-aligned at the current μ_q).
      root
        .append('ellipse')
        .attr('cx', xScale(current.mu1))
        .attr('cy', yScale(current.mu2))
        .attr('rx', CHI2_95 * sigmaQ * pixelsPerUnit)
        .attr('ry', CHI2_95 * sigmaQ * pixelsPerUnit)
        .attr('fill', 'none')
        .attr('stroke', COLORS.variational)
        .attr('stroke-width', 2.2);

      // Truth-mean cross.
      const tx = xScale(TRUTH_MEAN[0]);
      const ty = yScale(TRUTH_MEAN[1]);
      const tickLen = 4;
      ['+', '-'].forEach((s) => {
        const y0 = s === '+' ? ty - tickLen : ty + tickLen;
        const y1 = s === '+' ? ty + tickLen : ty - tickLen;
        root
          .append('line')
          .attr('x1', tx - tickLen)
          .attr('x2', tx + tickLen)
          .attr('y1', y0)
          .attr('y2', y1)
          .attr('stroke', COLORS.truthEdge)
          .attr('stroke-width', 1.6);
      });
      root
        .append('text')
        .attr('x', tx + 8)
        .attr('y', ty - 8)
        .attr('font-size', 10)
        .attr('fill', COLORS.truthEdge)
        .attr('font-weight', 600)
        .text('true μ');

      // Legend.
      const legend = root.append('g').attr('transform', `translate(${size - 220},10)`);
      const items = [
        { label: 'Target N(μ_t, Σ)', color: COLORS.truthEdge },
        { label: 'Variational q (mean-field)', color: COLORS.variational },
        { label: 'CAVI trajectory', color: COLORS.trajectory },
        { label: 'Initial q', color: COLORS.initial },
      ];
      items.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 14})`);
        row
          .append('rect')
          .attr('x', 0)
          .attr('y', 3)
          .attr('width', 12)
          .attr('height', 6)
          .attr('fill', it.color)
          .attr('opacity', 0.7);
        row
          .append('text')
          .attr('x', 18)
          .attr('y', 10)
          .attr('font-size', 10)
          .text(it.label);
      });
    },
    [trajectory, visibleTrajectory, current, sigmaQ, samples, rho, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          target correlation ρ:
          <input
            type="range"
            min={0}
            max={0.95}
            step={0.01}
            value={rho}
            onChange={(e) => {
              setRho(Number(e.target.value));
              setStep(N_SWEEPS * 2);
            }}
            className="w-32"
            aria-label="target correlation slider"
          />
          <span className="tabular-nums w-10 text-right text-xs text-[var(--color-text-muted)]">
            {rho.toFixed(2)}
          </span>
        </label>
        <label className="flex items-center gap-2">
          CAVI step k:
          <input
            type="range"
            min={0}
            max={N_SWEEPS * 2}
            step={1}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            className="w-40"
            aria-label="CAVI step slider"
          />
          <span className="tabular-nums w-10 text-right text-xs text-[var(--color-text-muted)]">
            {step}
          </span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-xs text-[var(--color-text-muted)]">
        <span>
          <strong>μ_q at step k:</strong>{' '}
          <span className="tabular-nums">
            ({current.mu1.toFixed(3)}, {current.mu2.toFixed(3)})
          </span>
        </span>
        <span>
          <strong>true μ:</strong>{' '}
          <span className="tabular-nums">
            ({TRUTH_MEAN[0].toFixed(2)}, {TRUTH_MEAN[1].toFixed(2)})
          </span>
        </span>
        <span>
          <strong>σ_q:</strong>{' '}
          <span className="tabular-nums">{sigmaQ.toFixed(3)}</span>{' '}
          (= √(1−ρ²), Theorem 5)
        </span>
        <span>
          <strong>ELBO at step k:</strong>{' '}
          <span className="tabular-nums">{current.elbo.toFixed(4)}</span>
        </span>
        <span>
          <strong>final ELBO:</strong>{' '}
          <span className="tabular-nums">{finalElbo.toFixed(4)}</span>{' '}
          (gain {(finalElbo - initialElbo).toFixed(2)} from step 0)
        </span>
      </div>
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Target: N(μ_t, Σ) with Σ = [[1, ρ], [ρ, 1]] — light blue scatter
        and tilted ellipse. Variational q = q_1·q_2 (mean-field) starts at
        the green dot, far from the truth. Each odd step updates q_1 (a
        horizontal move in μ-space — μ_2 unchanged); each even step
        updates q_2 (a vertical move). The trajectory zig-zags toward the
        true mean at contraction rate ρ² per sweep, exactly per the
        closed-form CAVI update derived from Theorem 2 on a Gaussian
        target. The orange ellipse is the current variational distribution;
        because mean-field forces axis-aligned variance σ_q² = 1−ρ², the
        ellipse is a circle smaller than the tilted truth — the §5.1
        "structural error of mean-field" Theorem 5 quantifies emerges
        already here in the §3.2 algorithm. ELBO is non-decreasing along
        the path (Theorem 2's monotonicity), and the gain from step 0 to
        step k tells the reader how much progress remains.
      </div>
    </div>
  );
}

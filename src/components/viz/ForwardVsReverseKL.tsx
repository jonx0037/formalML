import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// ForwardVsReverseKL — embedded in §1.3 of the variational-inference topic.
// One panel: a bimodal Gaussian-mixture target p(θ) = w₁ N(−Δ/2, 1) + (1−w₁)
// N(+Δ/2, 1), overlaid with two Gaussian projections — the forward-KL
// minimizer q_F (closed-form moment matching: mean and variance of p) and a
// reverse-KL minimizer q_R (analytic "lock-on-the-heavier-mode" approximation
// of the local minimizer). Drag the mode-separation slider Δ from 1 to 8 and
// the weight slider w₁ from 0.1 to 0.9. Reader sees:
//   - small Δ: q_F and q_R coincide (target is essentially unimodal)
//   - large Δ: q_F is wide and dips between modes (mode-covering)
//                q_R locks onto one mode (mode-seeking)
//   - asymmetric w₁: q_F shifts toward the heavier mode; q_R sits on it
//
// KL values are displayed numerically — KL(p ∥ q_F) and KL(q_R ∥ p) — computed
// by trapezoidal integration on a fine θ grid.
// =============================================================================

const PANEL_HEIGHT = 360;
const N_GRID = 480;
const X_DOMAIN: [number, number] = [-7, 7];
const SIGMA = 1;

const COLORS = {
  target: '#475569',     // slate — true posterior p
  forwardKL: '#2563eb',  // blue  — forward-KL projection (mode-covering)
  reverseKL: '#ea580c',  // orange — reverse-KL projection (mode-seeking)
};

// Gaussian density.
function normalPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// Two-component Gaussian-mixture density with modes at ±Δ/2 and unit width.
function targetPdf(x: number, sep: number, w1: number): number {
  return (
    w1 * normalPdf(x, -sep / 2, SIGMA) + (1 - w1) * normalPdf(x, sep / 2, SIGMA)
  );
}

// Forward-KL minimizer: moment matching. m* = E_p[θ], s² = Var_p[θ].
function forwardKLFit(sep: number, w1: number): { mu: number; sigma: number } {
  const m = w1 * (-sep / 2) + (1 - w1) * (sep / 2);
  const m1Term = w1 * (SIGMA * SIGMA + (-sep / 2 - m) ** 2);
  const m2Term = (1 - w1) * (SIGMA * SIGMA + (sep / 2 - m) ** 2);
  return { mu: m, sigma: Math.sqrt(m1Term + m2Term) };
}

// Reverse-KL minimizer (mode-locked approximation): center on the heavier
// mode, width ≈ local Gaussian width. For w1 = 0.5 the two modes are
// symmetric optima; we pick the right mode by convention.
function reverseKLFit(sep: number, w1: number): { mu: number; sigma: number } {
  const mu = w1 > 0.5 ? -sep / 2 : sep / 2;
  return { mu, sigma: SIGMA };
}

// Trapezoidal integration of f(x) over x ∈ [a, b] with N intervals.
function trapInt(
  f: (x: number) => number,
  a: number,
  b: number,
  n: number,
): number {
  const h = (b - a) / n;
  let s = 0.5 * (f(a) + f(b));
  for (let i = 1; i < n; i++) s += f(a + i * h);
  return s * h;
}

export default function ForwardVsReverseKL() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [separation, setSeparation] = useState<number>(4);
  const [weight1, setWeight1] = useState<number>(0.5);

  const xGrid = useMemo<Float64Array>(() => {
    const g = new Float64Array(N_GRID);
    const span = X_DOMAIN[1] - X_DOMAIN[0];
    for (let i = 0; i < N_GRID; i++)
      g[i] = X_DOMAIN[0] + (i / (N_GRID - 1)) * span;
    return g;
  }, []);

  const targetCurve = useMemo<Float64Array>(() => {
    const c = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) c[i] = targetPdf(xGrid[i], separation, weight1);
    return c;
  }, [xGrid, separation, weight1]);

  const fwdFit = useMemo(
    () => forwardKLFit(separation, weight1),
    [separation, weight1],
  );
  const revFit = useMemo(
    () => reverseKLFit(separation, weight1),
    [separation, weight1],
  );

  const fwdCurve = useMemo<Float64Array>(() => {
    const c = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++)
      c[i] = normalPdf(xGrid[i], fwdFit.mu, fwdFit.sigma);
    return c;
  }, [xGrid, fwdFit]);

  const revCurve = useMemo<Float64Array>(() => {
    const c = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++)
      c[i] = normalPdf(xGrid[i], revFit.mu, revFit.sigma);
    return c;
  }, [xGrid, revFit]);

  // KL(p ∥ q_F): the divergence forward-KL minimizes.
  const klPQf = useMemo<number>(() => {
    return trapInt(
      (x) => {
        const px = targetPdf(x, separation, weight1);
        if (px <= 0) return 0;
        const qx = normalPdf(x, fwdFit.mu, fwdFit.sigma);
        return px * Math.log(px / Math.max(qx, 1e-300));
      },
      X_DOMAIN[0],
      X_DOMAIN[1],
      N_GRID,
    );
  }, [separation, weight1, fwdFit]);

  // KL(q_R ∥ p): the divergence reverse-KL minimizes.
  const klQrP = useMemo<number>(() => {
    return trapInt(
      (x) => {
        const qx = normalPdf(x, revFit.mu, revFit.sigma);
        if (qx <= 0) return 0;
        const px = targetPdf(x, separation, weight1);
        return qx * Math.log(qx / Math.max(px, 1e-300));
      },
      X_DOMAIN[0],
      X_DOMAIN[1],
      N_GRID,
    );
  }, [separation, weight1, revFit]);

  const yMax = useMemo<number>(() => {
    let m = 0;
    for (let i = 0; i < N_GRID; i++) {
      if (targetCurve[i] > m) m = targetCurve[i];
      if (fwdCurve[i] > m) m = fwdCurve[i];
      if (revCurve[i] > m) m = revCurve[i];
    }
    return m * 1.1;
  }, [targetCurve, fwdCurve, revCurve]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 16, right: 18, bottom: 48, left: 52 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;

      const root = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain(X_DOMAIN).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      root
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(8).tickSizeOuter(0));
      root.append('g').call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0));

      root
        .append('text')
        .attr('transform', `translate(${innerW / 2},${innerH + 36})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('θ');

      root
        .append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('density');

      // Target p as filled area.
      const areaTarget = d3
        .area<number>()
        .x((_, i) => xScale(xGrid[i]))
        .y0(innerH)
        .y1((d) => yScale(d))
        .curve(d3.curveMonotoneX);

      root
        .append('path')
        .datum(Array.from(targetCurve))
        .attr('fill', COLORS.target)
        .attr('fill-opacity', 0.18)
        .attr('stroke', COLORS.target)
        .attr('stroke-width', 1.6)
        .attr('d', areaTarget);

      // Forward-KL fit as solid line.
      const lineFn = (color: string, dash: string | null) =>
        d3
          .line<number>()
          .x((_, i) => xScale(xGrid[i]))
          .y((d) => yScale(d))
          .curve(d3.curveMonotoneX);

      root
        .append('path')
        .datum(Array.from(fwdCurve))
        .attr('fill', 'none')
        .attr('stroke', COLORS.forwardKL)
        .attr('stroke-width', 2.2)
        .attr('d', lineFn(COLORS.forwardKL, null));

      // Reverse-KL fit as dashed line.
      root
        .append('path')
        .datum(Array.from(revCurve))
        .attr('fill', 'none')
        .attr('stroke', COLORS.reverseKL)
        .attr('stroke-width', 2.2)
        .attr('stroke-dasharray', '6 4')
        .attr('d', lineFn(COLORS.reverseKL, '6 4'));

      // Legend.
      const legendItems = [
        { label: 'Target p(θ)', color: COLORS.target, dash: false },
        {
          label: `Forward-KL  q_F = N(${fwdFit.mu.toFixed(2)}, ${fwdFit.sigma.toFixed(
            2,
          )}²)`,
          color: COLORS.forwardKL,
          dash: false,
        },
        {
          label: `Reverse-KL  q_R = N(${revFit.mu.toFixed(2)}, ${revFit.sigma.toFixed(
            2,
          )}²)`,
          color: COLORS.reverseKL,
          dash: true,
        },
      ];
      const legend = root.append('g').attr('transform', `translate(${innerW - 280}, 6)`);
      legendItems.forEach((item, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 18})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 24)
          .attr('y1', 8)
          .attr('y2', 8)
          .attr('stroke', item.color)
          .attr('stroke-width', 2.2)
          .attr('stroke-dasharray', item.dash ? '6 4' : null);
        row
          .append('text')
          .attr('x', 30)
          .attr('y', 11)
          .attr('font-size', 11)
          .text(item.label);
      });
    },
    [
      targetCurve,
      fwdCurve,
      revCurve,
      yMax,
      width,
      fwdFit.mu,
      fwdFit.sigma,
      revFit.mu,
      revFit.sigma,
    ],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          mode separation Δ:
          <input
            type="range"
            min={1}
            max={8}
            step={0.25}
            value={separation}
            onChange={(e) => setSeparation(Number(e.target.value))}
            className="w-32"
            aria-label="mode separation"
          />
          <span className="tabular-nums w-10 text-right">{separation.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-2">
          left-mode weight w₁:
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={weight1}
            onChange={(e) => setWeight1(Number(e.target.value))}
            className="w-32"
            aria-label="left mode weight"
          />
          <span className="tabular-nums w-10 text-right">{weight1.toFixed(2)}</span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-1 text-xs text-[var(--color-text-muted)]">
        <span>
          <strong style={{ color: COLORS.forwardKL }}>KL(p ∥ q_F)</strong> ={' '}
          <span className="tabular-nums">{klPQf.toFixed(3)}</span>{' '}
          — what forward-KL minimizes
        </span>
        <span>
          <strong style={{ color: COLORS.reverseKL }}>KL(q_R ∥ p)</strong> ={' '}
          <span className="tabular-nums">{klQrP.toFixed(3)}</span>{' '}
          — what reverse-KL minimizes
        </span>
      </div>
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Forward-KL <em>moment-matches</em> the target — a wide Gaussian that
        spans both modes and necessarily places density in the trough between
        them. Reverse-KL <em>locks onto a single mode</em> — a narrow Gaussian
        that puts near-zero density on the other mode. As Δ → 0 the two fits
        coincide (target is essentially unimodal); as Δ grows the divergence in
        their qualitative behavior is the {' '}
        <em>mode-covering vs mode-seeking</em> trade-off that defines the
        choice of variational divergence.
      </div>
    </div>
  );
}

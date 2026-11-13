import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// ── Problem definitions (1D constrained problems) ────────────────────

interface DualityProblem {
  name: string;
  /** Objective f₀(x) */
  f0: (x: number) => number;
  /** Constraint function f₁(x) — feasible when f₁(x) ≤ 0 */
  f1: (x: number, constraint: number) => number;
  /** Lagrangian L(x, λ) for given constraint RHS */
  lagrangian: (x: number, lambda: number, constraint: number) => number;
  /** Dual function g(λ) = inf_x L(x, λ) */
  dual: (lambda: number, constraint: number) => number;
  /** Primal optimal p*(constraint) */
  primalOpt: (constraint: number) => number;
  /** Perturbation function p*(u) */
  perturbation: (u: number, constraint: number) => number;
  xRange: [number, number];
  lambdaRange: [number, number];
  constraintRange: [number, number];
  constraintDefault: number;
  /** Initial λ when the problem is selected (default 2). */
  lambdaDefault?: number;
  /** λ values drawn as reference curves in the Lagrangian panel (default 0–5). */
  lagrangianLambdas?: number[];
  /** Isolated λ values where g is finite (e.g. an LP's dual); sampled explicitly. */
  dualAtoms?: number[];
  isConvex: boolean;
}

// ── 1D minimization helpers ─────────────────────────────────────────

const GOLDEN = (Math.sqrt(5) - 1) / 2;

/** Golden-section search for a local minimizer of f on [a, b]. */
function goldenMin(f: (x: number) => number, a: number, b: number, iters = 60): number {
  let c = b - GOLDEN * (b - a);
  let d = a + GOLDEN * (b - a);
  let fc = f(c);
  let fd = f(d);
  for (let i = 0; i < iters; i++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - GOLDEN * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + GOLDEN * (b - a);
      fd = f(d);
    }
  }
  return Math.min(fc, fd, f(a), f(b));
}

/** Global minimum of a smooth f on [lo, hi]: dense grid (endpoints included), then golden refinement. */
function minOnInterval(f: (x: number) => number, lo: number, hi: number, step = 0.01): number {
  const n = Math.max(2, Math.ceil((hi - lo) / step));
  let bestI = 0;
  let best = Infinity;
  for (let i = 0; i <= n; i++) {
    const v = f(lo + ((hi - lo) * i) / n);
    if (v < best) {
      best = v;
      bestI = i;
    }
  }
  const a = lo + ((hi - lo) * Math.max(0, bestI - 1)) / n;
  const b = lo + ((hi - lo) * Math.min(n, bestI + 1)) / n;
  return Math.min(best, goldenMin(f, a, b));
}

const nonconvexF0 = (x: number) => Math.sin(2 * x) + 0.1 * x * x;

/**
 * min f₀(x) over x ≤ c for f₀ = sin 2x + 0.1x². Since f₀ ≥ 0.1x² − 1, any x < −12
 * has f₀ > 13 > f₀(0), so the search can start at −12 for every c ≥ −12.
 */
function nonconvexPrimal(c: number): number {
  return minOnInterval(nonconvexF0, -12, c);
}

const PROBLEMS: DualityProblem[] = [
  {
    name: 'Quadratic + linear (convex)',
    f0: (x) => (x - 3) * (x - 3),
    f1: (x, c) => x - c,
    lagrangian: (x, lam, c) => (x - 3) * (x - 3) + lam * (x - c),
    dual: (lam, c) => {
      // min_x (x-3)^2 + lam*(x-c) → x* = 3 - lam/2
      const xStar = 3 - lam / 2;
      return (xStar - 3) * (xStar - 3) + lam * (xStar - c);
    },
    primalOpt: (c) => (c < 3 ? (c - 3) * (c - 3) : 0),
    perturbation: (u, c) => {
      const effC = c + u;
      return effC < 3 ? (effC - 3) * (effC - 3) : 0;
    },
    xRange: [-1, 6],
    lambdaRange: [0, 8],
    constraintRange: [0.5, 4],
    constraintDefault: 1.5,
    isConvex: true,
  },
  {
    name: 'Non-convex (positive gap)',
    f0: (x) => Math.sin(2 * x) + 0.1 * x * x,
    f1: (x, c) => x - c,
    lagrangian: (x, lam, c) => Math.sin(2 * x) + 0.1 * x * x + lam * (x - c),
    dual: (lam, c) => {
      // inf_x L(x, λ). Stationary points solve 0.2x + λ = −2cos 2x, so |0.2x + λ| ≤ 2:
      // every minimizer lies in [−5λ − 10, −5λ + 10] (it drifts left like −5λ).
      const L = (x: number) => nonconvexF0(x) + lam * (x - c);
      return minOnInterval(L, -5 * lam - 10, -5 * lam + 10);
    },
    primalOpt: (c) => nonconvexPrimal(c),
    perturbation: (u, c) => nonconvexPrimal(c + u),
    xRange: [-5, 3],
    lambdaRange: [0, 1.5],
    // Unconstrained minimizer x ≈ −0.748; for c below it the constraint binds and,
    // because f₀ is non-convex, p* > d*. At c = −1.5: p* ≈ 0.084, d* ≈ −0.613.
    constraintRange: [-2.5, 1.5],
    constraintDefault: -1.5,
    lambdaDefault: 0.5,
    lagrangianLambdas: [0, 0.25, 0.5, 0.75, 1, 1.25],
    isConvex: false,
  },
  {
    name: 'Linear program',
    f0: (x) => -2 * x + 4,
    f1: (x, c) => x - c,
    lagrangian: (x, lam, c) => -2 * x + 4 + lam * (x - c),
    dual: (lam, c) => {
      // inf_x (λ − 2)x + 4 − λc is −∞ unless λ = 2, where it equals 4 − 2c.
      if (Math.abs(lam - 2) < 1e-6) return 4 - 2 * c;
      return -Infinity;
    },
    primalOpt: (c) => -2 * c + 4,
    perturbation: (u, c) => -2 * (c + u) + 4,
    xRange: [-1, 5],
    lambdaRange: [0, 5],
    constraintRange: [0.5, 4],
    constraintDefault: 2,
    dualAtoms: [2],
    isConvex: true,
  },
];

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 320;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 40, left: 50 };

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';

// ── Component ────────────────────────────────────────────────────────

export default function DualityExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');
  const clipLag = `clip-lag-${instanceId}`;
  const clipDual = `clip-dual-${instanceId}`;
  const clipPert = `clip-pert-${instanceId}`;
  const [problemIdx, setProblemIdx] = useState(0);
  const [constraint, setConstraint] = useState(PROBLEMS[0].constraintDefault);
  const [lambda, setLambda] = useState(PROBLEMS[0].lambdaDefault ?? 2.0);

  const problem = PROBLEMS[problemIdx];
  const isStacked = containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth
    : Math.floor((containerWidth - 24) / 3);

  const pStar = problem.primalOpt(constraint);

  // Compute Lagrangian curves for several λ values
  const lagrangianLambdas = problem.lagrangianLambdas ?? [0, 1, 2, 3, 4, 5];
  const lagrangianCurves = useMemo(() => {
    const lambdas = lagrangianLambdas;
    const [xMin, xMax] = problem.xRange;
    return lambdas.map((lam) => ({
      lambda: lam,
      points: Array.from({ length: 200 }, (_, i) => {
        const x = xMin + (xMax - xMin) * (i / 199);
        return { x, y: problem.lagrangian(x, lam, constraint) };
      }),
    }));
  }, [problemIdx, constraint]);

  // Compute dual function curve (plus any isolated λ where g is finite)
  const dualCurve = useMemo(() => {
    const [lMin, lMax] = problem.lambdaRange;
    const lams = Array.from({ length: 200 }, (_, i) => lMin + (lMax - lMin) * (i / 199));
    for (const a of problem.dualAtoms ?? []) if (a >= lMin && a <= lMax) lams.push(a);
    lams.sort((a, b) => a - b);
    return lams.map((lam) => ({ lambda: lam, g: problem.dual(lam, constraint) }));
  }, [problemIdx, constraint]);

  // d* = max_{λ ≥ 0} g(λ): best sample, refined by golden section (g is concave)
  const { dStar, lambdaStar } = useMemo(() => {
    let bi = 0;
    for (let i = 1; i < dualCurve.length; i++) if (dualCurve[i].g > dualCurve[bi].g) bi = i;
    let best = dualCurve[bi];
    const lo = dualCurve[bi - 1];
    const hi = dualCurve[bi + 1];
    if (lo && hi && Number.isFinite(lo.g) && Number.isFinite(hi.g)) {
      const neg = (lam: number) => -problem.dual(lam, constraint);
      let a = lo.lambda;
      let b = hi.lambda;
      for (let k = 0; k < 60; k++) {
        const c1 = b - GOLDEN * (b - a);
        const c2 = a + GOLDEN * (b - a);
        if (neg(c1) < neg(c2)) b = c2;
        else a = c1;
      }
      const lam = (a + b) / 2;
      const g = problem.dual(lam, constraint);
      if (g > best.g) best = { lambda: lam, g };
    }
    return { dStar: best.g, lambdaStar: best.lambda };
  }, [dualCurve, problemIdx, constraint]);

  // Compute perturbation function
  const pertCurve = useMemo(() => {
    const uRange: [number, number] = [-1.5, 1.5];
    return Array.from({ length: 200 }, (_, i) => {
      const u = uRange[0] + (uRange[1] - uRange[0]) * (i / 199);
      return { u, p: problem.perturbation(u, constraint) };
    });
  }, [problemIdx, constraint]);

  // ── Left panel: Lagrangian L(x, λ) ────────────────────────────────

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 80) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const [xMin, xMax] = problem.xRange;
      const allY = lagrangianCurves.flatMap((c) => c.points.map((p) => p.y));
      const yMin = Math.max(d3.min(allY)! - 1, -10);
      const yMax = Math.min(d3.max(allY)! + 1, 30);

      const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, plotW]);
      const yScale = d3.scaleLinear().domain([yMax, yMin]).range([0, plotH]);

      // Clip path
      g.append('defs')
        .append('clipPath')
        .attr('id', clipLag)
        .append('rect')
        .attr('width', plotW)
        .attr('height', plotH);

      const plotArea = g.append('g').attr('clip-path', `url(#${clipLag})`);

      // Feasible region shading
      plotArea
        .append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', xScale(Math.min(constraint, xMax)))
        .attr('height', plotH)
        .style('fill', TEAL)
        .style('opacity', 0.08);

      // Constraint boundary
      g.append('line')
        .attr('x1', xScale(constraint))
        .attr('y1', 0)
        .attr('x2', xScale(constraint))
        .attr('y2', plotH)
        .style('stroke', TEAL)
        .style('stroke-width', '2')
        .style('stroke-dasharray', '5,3');

      // Lagrangian curves
      const line = d3
        .line<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y));

      const lams = lagrangianCurves.map((c) => c.lambda);
      const spacing = lams.length > 1 ? lams[1] - lams[0] : 1;
      const lamLo = lams[0];
      const lamHi = lams[lams.length - 1];
      const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([lamLo, lamHi]);

      lagrangianCurves.forEach((curve) => {
        const isHighlighted = Math.abs(curve.lambda - lambda) < spacing / 2;
        plotArea
          .append('path')
          .datum(curve.points)
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', isHighlighted ? AMBER : colorScale(curve.lambda))
          .style('stroke-width', isHighlighted ? '2.5' : '1')
          .style('opacity', isHighlighted ? 1 : 0.4);
      });

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${plotH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      // Title
      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Lagrangian L(x, λ)');
    },
    [panelWidth, lagrangianCurves, constraint, lambda],
  );

  // ── Center panel: dual function g(λ) ──────────────────────────────

  const centerRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 80) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const [lMin, lMax] = problem.lambdaRange;
      const filteredDual = dualCurve.filter((d) => d.g > -20);
      const gVals = filteredDual.map((d) => d.g);
      const gMin = Math.min(d3.min(gVals) ?? -5, pStar - 2);
      const gMax = Math.max(d3.max(gVals) ?? 10, pStar + 2);

      const xScale = d3.scaleLinear().domain([lMin, lMax]).range([0, plotW]);
      const yScale = d3.scaleLinear().domain([gMax, gMin]).range([0, plotH]);

      g.append('defs')
        .append('clipPath')
        .attr('id', clipDual)
        .append('rect')
        .attr('width', plotW)
        .attr('height', plotH);

      const plotArea = g.append('g').attr('clip-path', `url(#${clipDual})`);

      // p* dashed horizontal line
      plotArea
        .append('line')
        .attr('x1', 0)
        .attr('y1', yScale(pStar))
        .attr('x2', plotW)
        .attr('y2', yScale(pStar))
        .style('stroke', PURPLE)
        .style('stroke-dasharray', '5,3')
        .style('stroke-width', '1.5');

      // p* label
      g.append('text')
        .attr('x', plotW - 4)
        .attr('y', yScale(pStar) - 6)
        .attr('text-anchor', 'end')
        .style('fill', PURPLE)
        .style('font-size', '10px')
        .text(`p* = ${pStar.toFixed(2)}`);

      // Duality gap shading
      const gAtLambda = problem.dual(lambda, constraint);
      if (gAtLambda < pStar && gAtLambda > -20) {
        plotArea
          .append('rect')
          .attr('x', xScale(lambda) - 15)
          .attr('y', yScale(pStar))
          .attr('width', 30)
          .attr('height', Math.max(0, yScale(gAtLambda) - yScale(pStar)))
          .style('fill', AMBER)
          .style('opacity', 0.15);
      }

      // Dual function curve
      const line = d3
        .line<{ lambda: number; g: number }>()
        .defined((d) => d.g > -20)
        .x((d) => xScale(d.lambda))
        .y((d) => yScale(d.g));

      plotArea
        .append('path')
        .datum(dualCurve)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', '2.5');

      // Isolated finite values of g (e.g. the LP's dual is finite only at λ = 2)
      const ok = (i: number) => dualCurve[i] !== undefined && dualCurve[i].g > -20;
      dualCurve.forEach((d, i) => {
        if (ok(i) && !ok(i - 1) && !ok(i + 1)) {
          plotArea
            .append('circle')
            .attr('cx', xScale(d.lambda))
            .attr('cy', yScale(d.g))
            .attr('r', 4)
            .style('fill', TEAL);
        }
      });

      // Current λ marker
      if (gAtLambda > -20) {
        plotArea
          .append('circle')
          .attr('cx', xScale(lambda))
          .attr('cy', yScale(gAtLambda))
          .attr('r', 5)
          .style('fill', AMBER)
          .style('stroke', '#fff')
          .style('stroke-width', '1.5');
      }

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${plotH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      // Labels
      g.append('text')
        .attr('x', plotW / 2)
        .attr('y', plotH + 35)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('λ');

      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Dual Function g(λ)');
    },
    [panelWidth, dualCurve, lambda, pStar, constraint],
  );

  // ── Right panel: perturbation function p*(u) ──────────────────────

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 80) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const pVals = pertCurve.map((d) => d.p);
      const pMin = (d3.min(pVals) ?? 0) - 0.5;
      const pMax = (d3.max(pVals) ?? 10) + 0.5;

      const xScale = d3.scaleLinear().domain([-1.5, 1.5]).range([0, plotW]);
      const yScale = d3.scaleLinear().domain([pMax, pMin]).range([0, plotH]);

      g.append('defs')
        .append('clipPath')
        .attr('id', clipPert)
        .append('rect')
        .attr('width', plotW)
        .attr('height', plotH);

      const plotArea = g.append('g').attr('clip-path', `url(#${clipPert})`);

      // Perturbation curve
      const line = d3
        .line<{ u: number; p: number }>()
        .x((d) => xScale(d.u))
        .y((d) => yScale(d.p));

      plotArea
        .append('path')
        .datum(pertCurve)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', '2.5');

      // Affine minorant p*(u) ≥ d* − λ*u (slope −λ*, intercept d*); it touches
      // p* at u = 0 exactly when there is no duality gap.
      plotArea
        .append('line')
        .attr('x1', xScale(-1.5))
        .attr('y1', yScale(dStar + lambdaStar * 1.5))
        .attr('x2', xScale(1.5))
        .attr('y2', yScale(dStar - lambdaStar * 1.5))
        .style('stroke', AMBER)
        .style('stroke-dasharray', '5,3')
        .style('stroke-width', '1.5')
        .style('opacity', 0.8);

      // p*(0) marker
      plotArea
        .append('circle')
        .attr('cx', xScale(0))
        .attr('cy', yScale(pStar))
        .attr('r', 5)
        .style('fill', PURPLE);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${plotH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      g.append('text')
        .attr('x', plotW / 2)
        .attr('y', plotH + 35)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('u');

      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Perturbation p*(u)');
    },
    [panelWidth, pertCurve, pStar, dStar, lambdaStar],
  );

  const gAtLambda = problem.dual(lambda, constraint);
  const fmtG = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : '−∞');
  const dualityGap = pStar - dStar;

  return (
    <div ref={containerRef} className="w-full my-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="text-xs text-zinc-400 flex items-center gap-2">
          Problem:
          <select
            value={problemIdx}
            onChange={(e) => {
              const idx = Number(e.target.value);
              setProblemIdx(idx);
              setConstraint(PROBLEMS[idx].constraintDefault);
              setLambda(PROBLEMS[idx].lambdaDefault ?? 2.0);
            }}
            className="bg-zinc-800 text-zinc-200 text-xs rounded px-2 py-1 border border-zinc-600"
          >
            {PROBLEMS.map((p, i) => (
              <option key={i} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>

        <label className="text-xs text-zinc-400 flex items-center gap-2">
          Constraint: x ≤ {constraint.toFixed(2)}
          <input
            type="range"
            min={problem.constraintRange[0]}
            max={problem.constraintRange[1]}
            step={0.01}
            value={constraint}
            onChange={(e) => setConstraint(Number(e.target.value))}
            className="w-28 accent-teal-500"
          />
        </label>

        <label className="text-xs text-zinc-400 flex items-center gap-2">
          λ = {lambda.toFixed(2)}
          <input
            type="range"
            min={problem.lambdaRange[0]}
            max={problem.lambdaRange[1]}
            step={0.05}
            value={lambda}
            onChange={(e) => setLambda(Number(e.target.value))}
            className="w-28 accent-amber-500"
          />
        </label>
      </div>

      {/* Three panels */}
      <div className="flex gap-2" style={{ flexDirection: isStacked ? 'column' : 'row' }}>
        <svg role="img" aria-label="Duality explorer visualization (panel 1 of 3)" ref={leftRef} />
        <svg role="img" aria-label="Duality explorer visualization (panel 2 of 3)" ref={centerRef} />
        <svg role="img" aria-label="Duality explorer visualization (panel 3 of 3)" ref={rightRef} />
      </div>

      {/* Readout */}
      <p className="text-xs font-mono text-zinc-500 mt-2">
        p* = {pStar.toFixed(4)} | g(λ={lambda.toFixed(2)}) = {fmtG(gAtLambda)} | d* = max g ={' '}
        {fmtG(dStar)} at λ* ≈ {lambdaStar.toFixed(2)} | duality gap p* − d* ={' '}
        {dualityGap > 5e-4 ? dualityGap.toFixed(4) : '0 (strong duality)'}
        {!problem.isConvex && dualityGap > 5e-4 ? ' ← non-convex: positive gap' : ''}
      </p>
    </div>
  );
}

import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { checkKKT } from './shared/dualityUtils';

// ── Problem setup ────────────────────────────────────────────────────

/** 2D quadratic objective: f₀(x) = (x₁ - 3)² + (x₂ - 2)² */
function f0(x: number[]): number {
  return (x[0] - 3) ** 2 + (x[1] - 2) ** 2;
}

function gradF0(x: number[]): number[] {
  return [2 * (x[0] - 3), 2 * (x[1] - 2)];
}

/** Inequality constraints: fᵢ(x) ≤ 0 */
const CONSTRAINTS = [
  {
    label: 'x₁ + x₂ ≤ 4',
    f: (x: number[]) => x[0] + x[1] - 4,
    grad: (_x: number[]) => [1, 1],
    boundary: (t: number): [number, number] => [t, 4 - t],
    tRange: [-0.5, 5] as [number, number],
  },
  {
    label: 'x₁ ≥ 0',
    f: (x: number[]) => -x[0],
    grad: (_x: number[]) => [-1, 0],
    boundary: (t: number): [number, number] => [0, t],
    tRange: [-0.5, 5] as [number, number],
  },
  {
    label: 'x₂ ≥ 0',
    f: (x: number[]) => -x[1],
    grad: (_x: number[]) => [0, -1],
    boundary: (t: number): [number, number] => [t, 0],
    tRange: [-0.5, 5] as [number, number],
  },
];

/** Compute optimal dual variables for the active constraints via gradient matching */
function computeDualVars(x: number[], activeIndices: number[]): number[] {
  const gf0 = gradF0(x);
  const activeGrads = activeIndices.map((i) => CONSTRAINTS[i].grad(x));
  const n = activeIndices.length;

  if (n === 0) return [];
  if (n === 1) {
    // λ = -∇f₀ · ∇fᵢ / ‖∇fᵢ‖²
    const gi = activeGrads[0];
    const dotProd = gf0[0] * gi[0] + gf0[1] * gi[1];
    const normSq = gi[0] * gi[0] + gi[1] * gi[1];
    return [Math.max(0, -dotProd / normSq)];
  }
  if (n === 2) {
    // Solve: ∇f₀ + λ₁∇f₁ + λ₂∇f₂ = 0 via 2×2 system
    const [g1, g2] = activeGrads;
    const det = g1[0] * g2[1] - g1[1] * g2[0];
    if (Math.abs(det) < 1e-10) return [0, 0];
    const l1 = (-gf0[0] * g2[1] + gf0[1] * g2[0]) / det;
    const l2 = (-g1[0] * gf0[1] + g1[1] * gf0[0]) / det;
    return [Math.max(0, l1), Math.max(0, l2)];
  }
  return activeIndices.map(() => 0);
}

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 50 };

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const RED = '#DC2626';

// ── Component ────────────────────────────────────────────────────────

export default function KKTExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');
  const clipId = `clip-kkt-${instanceId}`;
  const arrowId = `arrow-purple-${instanceId}`;
  const [candidate, setCandidate] = useState<[number, number]>([2.0, 2.0]);
  const [showGradients, setShowGradients] = useState(true);
  const [activeConstraints, setActiveConstraints] = useState([0, 1, 2]); // indices

  const isStacked = containerWidth < SM_BREAKPOINT;
  const leftWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.6);
  const rightWidth = isStacked ? containerWidth : containerWidth - leftWidth - 16;

  // Determine active set and compute KKT
  const kktResult = useMemo(() => {
    const x = candidate;
    const constraints = activeConstraints.map((i) => CONSTRAINTS[i]);
    const fi = constraints.map((c) => c.f(x));
    const gradFi = constraints.map((c) => c.grad(x));

    // Compute optimal lambdas
    const nearActive = activeConstraints.filter(
      (i) => Math.abs(CONSTRAINTS[i].f(x)) < 0.3,
    );
    const lambdas = new Array(activeConstraints.length).fill(0);
    const dualVars = computeDualVars(x, nearActive);
    nearActive.forEach((ai, j) => {
      const idx = activeConstraints.indexOf(ai);
      if (idx >= 0) lambdas[idx] = dualVars[j] ?? 0;
    });

    return checkKKT(gradF0(x), gradFi, lambdas, fi);
  }, [candidate, activeConstraints]);

  // Contour data
  const contourData = useMemo(() => {
    const n = 60;
    const data: number[] = [];
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = -0.5 + (i / (n - 1)) * 5.5;
        const y = -0.5 + (j / (n - 1)) * 5.5;
        data.push(f0([x, y]));
      }
    }
    return { values: data, n };
  }, []);

  // ── Left panel: contour plot ──────────────────────────────────────

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (leftWidth < 100) return;

      const w = leftWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const xScale = d3.scaleLinear().domain([-0.5, 5]).range([0, plotW]);
      const yScale = d3.scaleLinear().domain([-0.5, 5]).range([plotH, 0]);

      g.append('defs')
        .append('clipPath')
        .attr('id', clipId)
        .append('rect')
        .attr('width', plotW)
        .attr('height', plotH);

      const plotArea = g.append('g').attr('clip-path', `url(#${clipId})`);

      // Contour levels
      const contours = d3
        .contours()
        .size([contourData.n, contourData.n])
        .thresholds(d3.range(0.5, 20, 1.5));

      const contourGeo = contours(contourData.values);
      const contourColor = d3.scaleSequential(d3.interpolateBlues).domain([20, 0]);

      const xContour = d3.scaleLinear().domain([0, contourData.n - 1]).range([-0.5, 5]);
      const yContour = d3.scaleLinear().domain([0, contourData.n - 1]).range([-0.5, 5]);

      const pathGen = d3.geoPath(
        d3.geoTransform({
          point(x, y) {
            this.stream.point(xScale(xContour(x)), yScale(yContour(y)));
          },
        }),
      );

      plotArea
        .selectAll('.contour')
        .data(contourGeo)
        .enter()
        .append('path')
        .attr('d', pathGen as any)
        .style('fill', 'none')
        .style('stroke', (d) => contourColor(d.value))
        .style('stroke-width', '0.8')
        .style('opacity', 0.5);

      // Constraint boundaries
      activeConstraints.forEach((ci) => {
        const c = CONSTRAINTS[ci];
        const [tMin, tMax] = c.tRange;
        const pts = Array.from({ length: 100 }, (_, i) => {
          const t = tMin + (tMax - tMin) * (i / 99);
          const [bx, by] = c.boundary(t);
          return [xScale(bx), yScale(by)] as [number, number];
        });

        plotArea
          .append('path')
          .datum(pts)
          .attr(
            'd',
            d3
              .line()
              .x((d) => d[0])
              .y((d) => d[1]),
          )
          .style('fill', 'none')
          .style('stroke', TEAL)
          .style('stroke-width', '2')
          .style('stroke-dasharray', '5,3');
      });

      // Gradient arrows at candidate
      if (showGradients) {
        const gf = gradF0(candidate);
        const arrowScale = 15;

        // -∇f₀ arrow (objective gradient, negated for descent direction)
        plotArea
          .append('line')
          .attr('x1', xScale(candidate[0]))
          .attr('y1', yScale(candidate[1]))
          .attr('x2', xScale(candidate[0]) - gf[0] * arrowScale)
          .attr('y2', yScale(candidate[1]) + gf[1] * arrowScale)
          .style('stroke', PURPLE)
          .style('stroke-width', '2.5')
          .attr('marker-end', `url(#${arrowId})`);

        // Constraint gradient arrows (for active constraints)
        activeConstraints.forEach((ci, idx) => {
          const c = CONSTRAINTS[ci];
          const cVal = c.f(candidate);
          if (Math.abs(cVal) < 1.0) {
            const cg = c.grad(candidate);
            const lam = kktResult.lambdas[idx] ?? 0;
            if (lam > 0.01) {
              plotArea
                .append('line')
                .attr('x1', xScale(candidate[0]))
                .attr('y1', yScale(candidate[1]))
                .attr('x2', xScale(candidate[0]) + cg[0] * lam * arrowScale)
                .attr('y2', yScale(candidate[1]) - cg[1] * lam * arrowScale)
                .style('stroke', AMBER)
                .style('stroke-width', '2')
                .style('opacity', 0.8);
            }
          }
        });

        // Arrow markers
        const defs = svg.select('defs').empty() ? svg.append('defs') : svg.select('defs');
        defs
          .append('marker')
          .attr('id', arrowId)
          .attr('viewBox', '0 0 10 10')
          .attr('refX', 8)
          .attr('refY', 5)
          .attr('markerWidth', 6)
          .attr('markerHeight', 6)
          .attr('orient', 'auto')
          .append('path')
          .attr('d', 'M 0 0 L 10 5 L 0 10 z')
          .style('fill', PURPLE);
      }

      // Candidate point (draggable)
      const candidateCircle = plotArea
        .append('circle')
        .attr('cx', xScale(candidate[0]))
        .attr('cy', yScale(candidate[1]))
        .attr('r', 8)
        .style('fill', kktResult.allSatisfied ? TEAL : RED)
        .style('stroke', '#fff')
        .style('stroke-width', '2')
        .style('cursor', 'grab');

      // Drag behavior
      const drag = d3
        .drag<SVGCircleElement, unknown>()
        .on('drag', (event) => {
          const newX = Math.max(-0.5, Math.min(5, xScale.invert(event.x)));
          const newY = Math.max(-0.5, Math.min(5, yScale.invert(event.y)));
          setCandidate([newX, newY]);
        });

      candidateCircle.call(drag);

      // Optimum marker (analytical: constrained min of quadratic).
      // For f₀(x) = (x₁ - 3)² + (x₂ - 2)² with x₁ + x₂ ≤ 4, x ≥ 0,
      // the unconstrained minimum is (3, 2), whose projection onto x₁ + x₂ = 4 is (2.5, 1.5).
      const xOpt = [2.5, 1.5];
      plotArea
        .append('circle')
        .attr('cx', xScale(xOpt[0]))
        .attr('cy', yScale(xOpt[1]))
        .attr('r', 4)
        .style('fill', 'none')
        .style('stroke', '#d4d4d8')
        .style('stroke-width', '1.5')
        .style('stroke-dasharray', '3,2');

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${plotH})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Contour Plot & KKT Geometry');
    },
    [leftWidth, candidate, showGradients, activeConstraints, contourData, kktResult],
  );

  // ── Right panel: KKT checklist ────────────────────────────────────

  const conditions = [
    {
      name: 'Stationarity',
      satisfied: kktResult.stationarity,
      detail: `‖∇L‖ = ${kktResult.stationarityResidual.toFixed(3)}`,
    },
    {
      name: 'Primal feasibility',
      satisfied: kktResult.primalFeasibility,
      detail: kktResult.constraintValues
        .map((v, i) => `f${i + 1} = ${v.toFixed(3)}`)
        .join(', '),
    },
    {
      name: 'Dual feasibility',
      satisfied: kktResult.dualFeasibility,
      detail: kktResult.lambdas.map((l, i) => `λ${i + 1} = ${l.toFixed(3)}`).join(', '),
    },
    {
      name: 'Complementary slackness',
      satisfied: kktResult.complementarySlackness,
      detail: kktResult.lambdas
        .map((l, i) => `λ${i + 1}f${i + 1} = ${(l * kktResult.constraintValues[i]).toFixed(4)}`)
        .join(', '),
    },
  ];

  return (
    <div ref={containerRef} className="w-full my-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="text-xs text-zinc-400 flex items-center gap-2">
          <input
            type="checkbox"
            checked={showGradients}
            onChange={(e) => setShowGradients(e.target.checked)}
            className="accent-purple-500"
          />
          Show gradient decomposition
        </label>

        <div className="text-xs text-zinc-400 flex items-center gap-2">
          Constraints:
          {CONSTRAINTS.map((c, i) => (
            <label key={i} className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={activeConstraints.includes(i)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setActiveConstraints((prev) => [...prev, i].sort());
                  } else {
                    setActiveConstraints((prev) => prev.filter((j) => j !== i));
                  }
                }}
                className="accent-teal-500"
              />
              <span>{c.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Panels */}
      <div className="flex gap-4" style={{ flexDirection: isStacked ? 'column' : 'row' }}>
        <svg ref={leftRef} />

        {/* KKT checklist */}
        <div
          style={{ width: isStacked ? '100%' : rightWidth, minHeight: HEIGHT }}
          className="bg-zinc-900/50 rounded-lg border border-zinc-700 p-4"
        >
          <h3 className="text-sm font-semibold text-zinc-200 mb-3">KKT Conditions</h3>

          <div className="space-y-3">
            {conditions.map((c, i) => (
              <div key={i} className="flex items-start gap-2">
                <span
                  className={`text-lg font-bold ${c.satisfied ? 'text-green-400' : 'text-red-400'}`}
                >
                  {c.satisfied ? '✓' : '✗'}
                </span>
                <div>
                  <div className="text-xs text-zinc-300 font-medium">{c.name}</div>
                  <div className="text-xs text-zinc-500 font-mono">{c.detail}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-700">
            <div
              className={`text-sm font-semibold ${kktResult.allSatisfied ? 'text-green-400' : 'text-red-400'}`}
            >
              {kktResult.allSatisfied ? 'All KKT conditions satisfied ✓' : 'KKT conditions not all satisfied'}
            </div>
            <div className="text-xs text-zinc-500 mt-1 font-mono">
              x = ({candidate[0].toFixed(2)}, {candidate[1].toFixed(2)}) | f₀(x) ={' '}
              {f0(candidate).toFixed(4)}
            </div>
          </div>

          <p className="text-xs text-zinc-600 mt-3">
            Drag the point to explore KKT conditions at different locations.
          </p>
        </div>
      </div>
    </div>
  );
}

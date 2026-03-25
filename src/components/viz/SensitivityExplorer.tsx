import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// ── Problem definitions ──────────────────────────────────────────────

interface SensitivityProblem {
  name: string;
  /** Optimal value as a function of perturbation u */
  pStar: (u: number) => number;
  /** Shadow prices (one per constraint) */
  shadowPrices: { label: string; value: number; active: boolean }[];
  uRange: [number, number];
}

const PROBLEMS: SensitivityProblem[] = [
  {
    name: 'Resource allocation (linear)',
    // min −0.75x s.t. x ≤ 4+u → p*(u) = −0.75(4+u) = −3 − 0.75u, ∂p*/∂u = −0.75 → λ* = 0.75
    pStar: (u) => -3 - 0.75 * u,
    shadowPrices: [
      { label: 'Material A', value: 0.75, active: true },
      { label: 'Material B', value: 0.5, active: true },
      { label: 'Labor', value: 0.0, active: false },
    ],
    uRange: [-2, 2],
  },
  {
    name: 'Quadratic program',
    // min (x−3)² s.t. x ≤ 1.5+u → p*(u) = (1.5+u−3)², ∂p*/∂u|₀ = 2(−1.5) = −3 → λ* = 3
    pStar: (u) => {
      const c = 1.5 + u;
      return c > 0 ? (c - 3) * (c - 3) : 9;
    },
    shadowPrices: [
      { label: 'x ≤ c', value: 3.0, active: true },
      { label: 'x ≥ 0', value: 0.0, active: false },
    ],
    uRange: [-1.5, 2],
  },
  {
    name: 'Inequality-constrained LS',
    // min (x−3.5)²+0.5 s.t. x ≤ 2+u → p*(u) = (2+u−3.5)²+0.5, ∂p*/∂u|₀ = 2(−1.5) = −3 → λ* = 3
    pStar: (u) => {
      const bound = 2 + u;
      if (bound >= 3.5) return 0.5;
      return (bound - 3.5) * (bound - 3.5) + 0.5;
    },
    shadowPrices: [
      { label: '‖x‖ ≤ b', value: 3.0, active: true },
      { label: 'xᵢ ≥ 0', value: 0.0, active: false },
      { label: 'Σxᵢ ≤ M', value: 0.0, active: false },
    ],
    uRange: [-1.5, 2],
  },
];

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 320;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 55 };

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const GRAY = '#6B6B6B';

// ── Component ────────────────────────────────────────────────────────

export default function SensitivityExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [problemIdx, setProblemIdx] = useState(0);
  const [perturbation, setPerturbation] = useState(0);
  const [showTangent, setShowTangent] = useState(true);

  const problem = PROBLEMS[problemIdx];
  const isStacked = containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth
    : Math.floor((containerWidth - 16) / 2);

  // Compute p*(u) curve
  const curveData = useMemo(() => {
    const [uMin, uMax] = problem.uRange;
    const nPts = 200;
    return Array.from({ length: nPts }, (_, i) => {
      const u = uMin + (uMax - uMin) * (i / (nPts - 1));
      return { u, p: problem.pStar(u) };
    });
  }, [problemIdx]);

  const pAtZero = problem.pStar(0);
  const pAtU = problem.pStar(perturbation);
  const mainShadow = problem.shadowPrices.find((s) => s.active)?.value ?? 0;

  // ── Left panel: perturbation function ──────────────────────────────

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 100) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const [uMin, uMax] = problem.uRange;
      const pVals = curveData.map((d) => d.p);
      const pMin = d3.min(pVals)! - 0.5;
      const pMax = d3.max(pVals)! + 0.5;

      const xScale = d3.scaleLinear().domain([uMin, uMax]).range([0, plotW]);
      const yScale = d3.scaleLinear().domain([pMax, pMin]).range([0, plotH]);

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

      // Axis labels
      g.append('text')
        .attr('x', plotW / 2)
        .attr('y', plotH + 35)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('Perturbation u');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -plotH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('p*(u)');

      // Title
      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Perturbation Function p*(u)');

      // p*(u) curve
      const line = d3
        .line<{ u: number; p: number }>()
        .x((d) => xScale(d.u))
        .y((d) => yScale(d.p));

      g.append('path')
        .datum(curveData)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', '2.5');

      // Tangent line at u = 0
      if (showTangent) {
        const tangentPts = [
          { u: uMin, p: pAtZero - mainShadow * uMin },
          { u: uMax, p: pAtZero - mainShadow * uMax },
        ].map((d) => ({ u: d.u, p: d.p }));

        g.append('line')
          .attr('x1', xScale(tangentPts[0].u))
          .attr('y1', yScale(tangentPts[0].p))
          .attr('x2', xScale(tangentPts[1].u))
          .attr('y2', yScale(tangentPts[1].p))
          .style('stroke', AMBER)
          .style('stroke-width', '1.5')
          .style('stroke-dasharray', '6,3')
          .style('opacity', 0.8);

        // Label
        g.append('text')
          .attr('x', xScale(uMax * 0.6))
          .attr('y', yScale(pAtZero - mainShadow * uMax * 0.6) - 8)
          .style('fill', AMBER)
          .style('font-size', '10px')
          .text(`slope = −λ* = −${mainShadow.toFixed(2)}`);
      }

      // Mark p*(0)
      g.append('circle')
        .attr('cx', xScale(0))
        .attr('cy', yScale(pAtZero))
        .attr('r', 5)
        .style('fill', PURPLE);

      // Mark current perturbation
      g.append('circle')
        .attr('cx', xScale(perturbation))
        .attr('cy', yScale(pAtU))
        .attr('r', 5)
        .style('fill', AMBER)
        .style('stroke', '#fff')
        .style('stroke-width', '1.5');

      // Vertical dashed line at u
      if (Math.abs(perturbation) > 0.01) {
        g.append('line')
          .attr('x1', xScale(perturbation))
          .attr('y1', yScale(pAtU))
          .attr('x2', xScale(perturbation))
          .attr('y2', yScale(pAtZero))
          .style('stroke', '#a1a1aa')
          .style('stroke-dasharray', '3,3')
          .style('opacity', 0.5);
      }
    },
    [panelWidth, curveData, perturbation, showTangent, pAtZero, pAtU, mainShadow],
  );

  // ── Right panel: shadow price bar chart ────────────────────────────

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 100) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const sp = problem.shadowPrices;
      const maxVal = Math.max(1, ...sp.map((s) => s.value)) * 1.2;

      const yBand = d3
        .scaleBand<number>()
        .domain(sp.map((_, i) => i))
        .range([0, plotH])
        .padding(0.3);

      const xScale = d3.scaleLinear().domain([0, maxVal]).range([0, plotW]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${plotH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      g.append('g')
        .call(
          d3
            .axisLeft(yBand)
            .tickFormat((i) => sp[i as number]?.label ?? ''),
        )
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa').style('font-size', '10px'))
        .call((g) => g.selectAll('line').style('stroke', '#52525b'))
        .call((g) => g.select('.domain').style('stroke', '#52525b'));

      // Axis label
      g.append('text')
        .attr('x', plotW / 2)
        .attr('y', plotH + 35)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('Shadow Price λ*');

      // Title
      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Shadow Prices');

      // Bars
      g.selectAll('.bar')
        .data(sp)
        .enter()
        .append('rect')
        .attr('x', 0)
        .attr('y', (_, i) => yBand(i)!)
        .attr('width', (d) => xScale(d.value))
        .attr('height', yBand.bandwidth())
        .style('fill', (d) => (d.active ? TEAL : GRAY))
        .style('opacity', (d) => (d.active ? 1 : 0.4));

      // Value labels
      g.selectAll('.val-label')
        .data(sp)
        .enter()
        .append('text')
        .attr('x', (d) => xScale(d.value) + 6)
        .attr('y', (_, i) => yBand(i)! + yBand.bandwidth() / 2 + 4)
        .style('fill', '#d4d4d8')
        .style('font-size', '11px')
        .text((d) => d.value.toFixed(2));
    },
    [panelWidth, problemIdx],
  );

  return (
    <div ref={containerRef} className="w-full my-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="text-xs text-zinc-400 flex items-center gap-2">
          Problem:
          <select
            value={problemIdx}
            onChange={(e) => {
              setProblemIdx(Number(e.target.value));
              setPerturbation(0);
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
          u = {perturbation.toFixed(2)}
          <input
            type="range"
            min={problem.uRange[0]}
            max={problem.uRange[1]}
            step={0.01}
            value={perturbation}
            onChange={(e) => setPerturbation(Number(e.target.value))}
            className="w-32 accent-amber-500"
          />
        </label>

        <label className="text-xs text-zinc-400 flex items-center gap-2">
          <input
            type="checkbox"
            checked={showTangent}
            onChange={(e) => setShowTangent(e.target.checked)}
            className="accent-amber-500"
          />
          Show tangent line
        </label>
      </div>

      {/* Panels */}
      <div className="flex gap-4" style={{ flexDirection: isStacked ? 'column' : 'row' }}>
        <svg ref={leftRef} />
        <svg ref={rightRef} />
      </div>

      {/* Readout */}
      <p className="text-xs font-mono text-zinc-500 mt-2">
        p*(0) = {pAtZero.toFixed(4)} | p*({perturbation.toFixed(2)}) = {pAtU.toFixed(4)} |
        Δp* ≈ −λ*·u = −{mainShadow.toFixed(2)}×{perturbation.toFixed(2)} ={' '}
        {(-mainShadow * perturbation).toFixed(4)}
      </p>
    </div>
  );
}

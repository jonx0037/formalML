import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  generateLassoProblem,
  solveADMM,
  solveISTA,
  solveFISTA,
} from './shared/proximalUtils';

const SM_BREAKPOINT = 640;
const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const GREEN = '#16A34A';

const margin = { top: 30, right: 20, bottom: 40, left: 55 };
const HEIGHT = 340;

const problem = generateLassoProblem(30, 20, 4, 42);
const LAMBDA = 0.1;
const MAX_ITER = 200;

export default function ADMMExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [rho, setRho] = useState(1.0);
  const [showResiduals, setShowResiduals] = useState(false);
  const [compareMode, setCompareMode] = useState<'admm' | 'compare'>('compare');

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const admmResult = useMemo(
    () => solveADMM(problem.A, problem.b, LAMBDA, rho, MAX_ITER),
    [rho],
  );

  const { istaResult, fistaResult } = useMemo(() => {
    if (compareMode !== 'compare') return { istaResult: null, fistaResult: null };
    return {
      istaResult: solveISTA(problem.A, problem.b, LAMBDA, problem.L, MAX_ITER),
      fistaResult: solveFISTA(problem.A, problem.b, LAMBDA, problem.L, MAX_ITER),
    };
  }, [compareMode]);

  const fStar = useMemo(() => {
    const candidates = [...admmResult.objectives.slice(-20)];
    if (istaResult) candidates.push(...istaResult.objectives.slice(-20));
    if (fistaResult) candidates.push(...fistaResult.objectives.slice(-20));
    return Math.min(...candidates);
  }, [admmResult, istaResult, fistaResult]);

  // Left panel: convergence / residuals
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const panelW = isMobile
        ? containerWidth - margin.left - margin.right
        : Math.floor(containerWidth / 2) - margin.left - margin.right;
      if (panelW <= 0) return;
      const panelH = HEIGHT - margin.top - margin.bottom;

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, MAX_ITER]).range([0, panelW]);

      if (showResiduals) {
        // Primal and dual residuals
        const allRes = [...admmResult.primalResiduals, ...admmResult.dualResiduals].filter(
          (v) => v > 0,
        );
        const maxRes = Math.max(...allRes, 1e-10);
        const minRes = Math.min(...allRes.filter((v) => v > 1e-15), 1e-10);

        const yScale = d3.scaleLog().domain([minRes * 0.1, maxRes * 2]).range([panelH, 0]);

        const line = d3
          .line<number>()
          .x((_, i) => xScale(i))
          .y((d) => yScale(Math.max(d, minRes * 0.1)));

        g.append('path')
          .datum(admmResult.primalResiduals)
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', TEAL)
          .style('stroke-width', 2);

        g.append('path')
          .datum(admmResult.dualResiduals)
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', PURPLE)
          .style('stroke-width', 2);

        // Axes
        g.append('g')
          .attr('transform', `translate(0,${panelH})`)
          .call(d3.axisBottom(xScale).ticks(5))
          .selectAll('text')
          .style('fill', '#a1a1aa');
        g.append('g')
          .call(d3.axisLeft(yScale).ticks(5, '.0e'))
          .selectAll('text')
          .style('fill', '#a1a1aa');

        g.selectAll('.domain, .tick line').style('stroke', '#52525b');

        // Legend
        const items = [
          { color: TEAL, label: 'Primal ‖x−z‖' },
          { color: PURPLE, label: 'Dual ρ‖Δz‖' },
        ];
        items.forEach((d, i) => {
          g.append('line')
            .attr('x1', panelW - 105)
            .attr('y1', 8 + i * 16)
            .attr('x2', panelW - 85)
            .attr('y2', 8 + i * 16)
            .style('stroke', d.color)
            .style('stroke-width', 2);
          g.append('text')
            .attr('x', panelW - 81)
            .attr('y', 8 + i * 16)
            .attr('dominant-baseline', 'central')
            .style('fill', '#d4d4d8')
            .style('font-size', '10px')
            .text(d.label);
        });

        g.append('text')
          .attr('x', panelW / 2)
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .style('fill', '#d4d4d8')
          .style('font-size', '12px')
          .style('font-weight', 'bold')
          .text(`ADMM residuals (ρ = ${rho.toFixed(2)})`);
      } else {
        // Objective gap
        const admmGap = admmResult.objectives.map((v) => Math.max(v - fStar, 1e-15));
        const maxGap = admmGap[0] || 1;

        const yScale = d3.scaleLog().domain([1e-12, maxGap * 2]).range([panelH, 0]);

        const line = d3
          .line<number>()
          .x((_, i) => xScale(i))
          .y((d) => yScale(Math.max(d, 1e-15)));

        g.append('path')
          .datum(admmGap)
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', AMBER)
          .style('stroke-width', 2);

        if (compareMode === 'compare' && istaResult && fistaResult) {
          const istaGap = istaResult.objectives.slice(1).map((v) => Math.max(v - fStar, 1e-15));
          const fistaGap = fistaResult.objectives.slice(1).map((v) => Math.max(v - fStar, 1e-15));

          g.append('path')
            .datum(istaGap)
            .attr('d', line)
            .style('fill', 'none')
            .style('stroke', TEAL)
            .style('stroke-width', 1.5)
            .style('opacity', 0.7);

          g.append('path')
            .datum(fistaGap)
            .attr('d', line)
            .style('fill', 'none')
            .style('stroke', PURPLE)
            .style('stroke-width', 1.5)
            .style('opacity', 0.7);
        }

        // Axes
        g.append('g')
          .attr('transform', `translate(0,${panelH})`)
          .call(d3.axisBottom(xScale).ticks(5))
          .selectAll('text')
          .style('fill', '#a1a1aa');
        g.append('g')
          .call(d3.axisLeft(yScale).ticks(5, '.0e'))
          .selectAll('text')
          .style('fill', '#a1a1aa');

        g.selectAll('.domain, .tick line').style('stroke', '#52525b');

        // Legend
        const items = [{ color: AMBER, label: 'ADMM' }];
        if (compareMode === 'compare') {
          items.push({ color: TEAL, label: 'ISTA' });
          items.push({ color: PURPLE, label: 'FISTA' });
        }
        items.forEach((d, i) => {
          g.append('line')
            .attr('x1', panelW - 75)
            .attr('y1', 8 + i * 16)
            .attr('x2', panelW - 55)
            .attr('y2', 8 + i * 16)
            .style('stroke', d.color)
            .style('stroke-width', 2);
          g.append('text')
            .attr('x', panelW - 51)
            .attr('y', 8 + i * 16)
            .attr('dominant-baseline', 'central')
            .style('fill', '#d4d4d8')
            .style('font-size', '10px')
            .text(d.label);
        });

        g.append('text')
          .attr('x', panelW / 2)
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .style('fill', '#d4d4d8')
          .style('font-size', '12px')
          .style('font-weight', 'bold')
          .text(`Convergence (ρ = ${rho.toFixed(2)})`);
      }

      // x-label
      g.append('text')
        .attr('x', panelW / 2)
        .attr('y', panelH + 35)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('Iteration k');
    },
    [admmResult, istaResult, fistaResult, containerWidth, isMobile, showResiduals, rho, fStar, compareMode],
  );

  // Right panel: recovered sparse vector
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const panelW = isMobile
        ? containerWidth - margin.left - margin.right
        : Math.floor(containerWidth / 2) - margin.left - margin.right;
      if (panelW <= 0) return;
      const panelH = HEIGHT - margin.top - margin.bottom;

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const p = problem.p;
      const MIN_BAR_WIDTH = 2;
      const BAR_GAP = 2; // px between paired truth/recovered bars
      const barW = Math.max(MIN_BAR_WIDTH, (panelW / p - BAR_GAP) / 2);

      const allVals = [...problem.xTrue, ...admmResult.z];
      const yMax = Math.max(...allVals.map(Math.abs), 0.5);

      const xScale = d3
        .scaleBand<number>()
        .domain(d3.range(p))
        .range([0, panelW])
        .padding(0.3);
      const yScale = d3.scaleLinear().domain([-yMax * 1.1, yMax * 1.1]).range([panelH, 0]);

      // Ground truth bars
      g.selectAll('rect.truth')
        .data(problem.xTrue)
        .join('rect')
        .attr('class', 'truth')
        .attr('x', (_, i) => (xScale(i) || 0))
        .attr('y', (d) => (d >= 0 ? yScale(d) : yScale(0)))
        .attr('width', barW)
        .attr('height', (d) => Math.abs(yScale(0) - yScale(d)))
        .style('fill', GREEN)
        .style('opacity', 0.4);

      // ADMM solution bars
      g.selectAll('rect.admm')
        .data(admmResult.z)
        .join('rect')
        .attr('class', 'admm')
        .attr('x', (_, i) => (xScale(i) || 0) + barW)
        .attr('y', (d) => (d >= 0 ? yScale(d) : yScale(0)))
        .attr('width', barW)
        .attr('height', (d) => Math.abs(yScale(0) - yScale(d)))
        .style('fill', AMBER);

      // Zero line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', panelW)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .style('stroke', '#52525b')
        .style('stroke-width', 1);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${panelH})`)
        .call(
          d3
            .axisBottom(xScale)
            .tickValues(d3.range(0, p, Math.max(1, Math.floor(p / 10))))
            .tickFormat((d) => String(d)),
        )
        .selectAll('text')
        .style('fill', '#a1a1aa')
        .style('font-size', '9px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', '#a1a1aa');

      g.selectAll('.domain, .tick line').style('stroke', '#52525b');

      // Legend
      const items = [
        { color: GREEN, label: 'Ground truth', opacity: 0.4 },
        { color: AMBER, label: 'ADMM recovery', opacity: 1 },
      ];
      items.forEach((d, i) => {
        g.append('rect')
          .attr('x', panelW - 110)
          .attr('y', 2 + i * 16)
          .attr('width', 12)
          .attr('height', 10)
          .style('fill', d.color)
          .style('opacity', d.opacity);
        g.append('text')
          .attr('x', panelW - 94)
          .attr('y', 7 + i * 16)
          .attr('dominant-baseline', 'central')
          .style('fill', '#d4d4d8')
          .style('font-size', '10px')
          .text(d.label);
      });

      // Title
      g.append('text')
        .attr('x', panelW / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text('Sparse recovery');

      // x-label
      g.append('text')
        .attr('x', panelW / 2)
        .attr('y', panelH + 35)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('Coordinate index');
    },
    [admmResult, containerWidth, isMobile],
  );

  const halfW = isMobile ? containerWidth : Math.floor(containerWidth / 2);

  return (
    <div ref={containerRef} className="w-full my-6">
      <div className="flex flex-wrap items-center gap-4 mb-4 text-sm">
        <label className="flex items-center gap-2">
          <span className="text-zinc-400">ρ</span>
          <input
            type="range"
            min={-2}
            max={1}
            step={0.1}
            value={Math.log10(rho)}
            onChange={(e) => setRho(Math.pow(10, Number(e.target.value)))}
            className="w-32"
          />
          <span className="text-zinc-300 font-mono w-14">{rho.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-2 text-zinc-400">
          <input
            type="checkbox"
            checked={showResiduals}
            onChange={(e) => setShowResiduals(e.target.checked)}
            className="accent-purple-500"
          />
          Show residuals
        </label>
        <select
          value={compareMode}
          onChange={(e) => setCompareMode(e.target.value as 'admm' | 'compare')}
          className="bg-zinc-800 text-zinc-200 border border-zinc-600 rounded px-2 py-1"
        >
          <option value="admm">ADMM only</option>
          <option value="compare">ADMM vs ISTA vs FISTA</option>
        </select>
      </div>
      <div className={`flex ${isMobile ? 'flex-col' : 'flex-row'} gap-0`}>
        <svg ref={leftRef} width={halfW} height={HEIGHT} className="overflow-visible" />
        <svg ref={rightRef} width={halfW} height={HEIGHT} className="overflow-visible" />
      </div>
    </div>
  );
}

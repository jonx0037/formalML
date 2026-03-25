import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { softThresholdScalar } from './shared/proximalUtils';

const SM_BREAKPOINT = 640;
const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const SLATE = '#6B6B6B';

const margin = { top: 30, right: 20, bottom: 40, left: 50 };
const HEIGHT = 340;

/** Build a 2D quadratic g(x) = (1/2) x^T A x with condition number kappa. */
function makeQuadratic2D(kappa: number) {
  const theta = Math.PI / 6;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const a00 = c * c + kappa * s * s;
  const a01 = (1 - kappa) * c * s;
  const a11 = s * s + kappa * c * c;
  const L = kappa;

  const f = (x: number[]) =>
    0.5 * (a00 * x[0] * x[0] + 2 * a01 * x[0] * x[1] + a11 * x[1] * x[1]);
  const grad = (x: number[]) => [
    a00 * x[0] + a01 * x[1],
    a01 * x[0] + a11 * x[1],
  ];
  return { f, grad, L, A: [[a00, a01], [a01, a11]] };
}

/** Run ISTA on composite g + lambda*||.||_1. */
function runISTA(
  quad: ReturnType<typeof makeQuadratic2D>,
  x0: number[],
  lambda: number,
  maxIter: number,
) {
  const eta = 1 / quad.L;
  const path: number[][] = [x0.slice()];
  const objs: number[] = [quad.f(x0) + lambda * (Math.abs(x0[0]) + Math.abs(x0[1]))];
  let x = x0.slice();

  for (let k = 0; k < maxIter; k++) {
    const g = quad.grad(x);
    const xh = [x[0] - eta * g[0], x[1] - eta * g[1]];
    x = [softThresholdScalar(xh[0], eta * lambda), softThresholdScalar(xh[1], eta * lambda)];
    path.push(x.slice());
    objs.push(quad.f(x) + lambda * (Math.abs(x[0]) + Math.abs(x[1])));
  }
  return { path, objs };
}

/** Run FISTA on composite g + lambda*||.||_1. */
function runFISTA(
  quad: ReturnType<typeof makeQuadratic2D>,
  x0: number[],
  lambda: number,
  maxIter: number,
) {
  const eta = 1 / quad.L;
  const path: number[][] = [x0.slice()];
  const objs: number[] = [quad.f(x0) + lambda * (Math.abs(x0[0]) + Math.abs(x0[1]))];
  const momentumPts: number[][] = [];
  let x = x0.slice();
  let xPrev = x0.slice();
  let tk = 1;

  for (let k = 0; k < maxIter; k++) {
    const tNext = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2;
    const mom = (tk - 1) / tNext;
    const y = [x[0] + mom * (x[0] - xPrev[0]), x[1] + mom * (x[1] - xPrev[1])];
    momentumPts.push(y.slice());

    const g = quad.grad(y);
    const yh = [y[0] - eta * g[0], y[1] - eta * g[1]];
    xPrev = x.slice();
    x = [softThresholdScalar(yh[0], eta * lambda), softThresholdScalar(yh[1], eta * lambda)];
    tk = tNext;
    path.push(x.slice());
    objs.push(quad.f(x) + lambda * (Math.abs(x[0]) + Math.abs(x[1])));
  }
  return { path, objs, momentumPts };
}

export default function FISTAExplorer() {
  const rawId = useId();
  const id = rawId.replace(/:/g, '');
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [kappa, setKappa] = useState(10);
  const [x0, setX0] = useState<[number, number]>([2.5, 2.5]);
  const [showMomentum, setShowMomentum] = useState(false);
  const lambda = 0.3;
  const maxIter = 60;

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { istaResult, fistaResult, quad } = useMemo(() => {
    const q = makeQuadratic2D(kappa);
    const ista = runISTA(q, x0, lambda, maxIter);
    const fista = runFISTA(q, x0, lambda, maxIter);
    return { istaResult: ista, fistaResult: fista, quad: q };
  }, [kappa, x0]);

  // Left panel: contour + trajectories
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const panelW = isMobile
        ? containerWidth - margin.left - margin.right
        : Math.floor(containerWidth / 2) - margin.left - margin.right;
      if (panelW <= 0) return;
      const panelH = HEIGHT - margin.top - margin.bottom;

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xRange = 4;
      const xScale = d3.scaleLinear().domain([-xRange, xRange]).range([0, panelW]);
      const yScale = d3.scaleLinear().domain([-xRange, xRange]).range([panelH, 0]);

      // Clip path
      g.append('defs')
        .append('clipPath')
        .attr('id', `${id}-clip-traj`)
        .append('rect')
        .attr('width', panelW)
        .attr('height', panelH);

      const plotG = g.append('g').attr('clip-path', `url(#${id}-clip-traj)`);

      // Contours
      const gridSize = 80;
      const vals = new Array(gridSize * gridSize);
      for (let iy = 0; iy < gridSize; iy++) {
        for (let ix = 0; ix < gridSize; ix++) {
          const px = -xRange + (ix / (gridSize - 1)) * 2 * xRange;
          const py = -xRange + (iy / (gridSize - 1)) * 2 * xRange;
          vals[iy * gridSize + ix] = quad.f([px, py]);
        }
      }
      const contourGen = d3.contours().size([gridSize, gridSize]).thresholds(15);
      const contours = contourGen(vals);

      const cxScale = d3.scaleLinear().domain([0, gridSize - 1]).range([0, panelW]);
      const cyScale = d3.scaleLinear().domain([0, gridSize - 1]).range([panelH, 0]);

      plotG
        .selectAll('path.contour')
        .data(contours)
        .join('path')
        .attr('class', 'contour')
        .attr('d', d3.geoPath(d3.geoTransform({
          point(px: number, py: number) {
            this.stream.point(cxScale(px), cyScale(py));
          },
        })))
        .style('fill', 'none')
        .style('stroke', SLATE)
        .style('stroke-opacity', 0.25)
        .style('stroke-width', 0.8);

      // L1 diamond
      const diamondR = lambda * 3;
      const diamondPts: [number, number][] = [
        [xScale(diamondR), yScale(0)],
        [xScale(0), yScale(diamondR)],
        [xScale(-diamondR), yScale(0)],
        [xScale(0), yScale(-diamondR)],
      ];
      plotG
        .append('polygon')
        .attr('points', diamondPts.map((p) => p.join(',')).join(' '))
        .style('fill', 'rgba(15,110,86,0.08)')
        .style('stroke', TEAL)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4,3');

      // ISTA trajectory
      const istaLine = d3
        .line<number[]>()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]));

      plotG
        .append('path')
        .datum(istaResult.path)
        .attr('d', istaLine)
        .style('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', 1.5)
        .style('stroke-opacity', 0.8);

      plotG
        .selectAll('circle.ista')
        .data(istaResult.path.slice(0, 30))
        .join('circle')
        .attr('class', 'ista')
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('r', 2)
        .style('fill', TEAL);

      // FISTA trajectory
      plotG
        .append('path')
        .datum(fistaResult.path)
        .attr('d', istaLine)
        .style('fill', 'none')
        .style('stroke', PURPLE)
        .style('stroke-width', 1.5)
        .style('stroke-opacity', 0.8);

      plotG
        .selectAll('circle.fista')
        .data(fistaResult.path.slice(0, 30))
        .join('circle')
        .attr('class', 'fista')
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('r', 2)
        .style('fill', PURPLE);

      // Momentum arrows for FISTA
      if (showMomentum) {
        for (let i = 0; i < Math.min(8, fistaResult.momentumPts.length); i++) {
          const from = fistaResult.path[i + 1];
          const to = fistaResult.momentumPts[i];
          if (!from || !to) continue;
          const dx = xScale(to[0]) - xScale(from[0]);
          const dy = yScale(to[1]) - yScale(from[1]);
          const len = Math.sqrt(dx * dx + dy * dy);
          if (len < 2) continue;
          plotG
            .append('line')
            .attr('x1', xScale(from[0]))
            .attr('y1', yScale(from[1]))
            .attr('x2', xScale(to[0]))
            .attr('y2', yScale(to[1]))
            .style('stroke', AMBER)
            .style('stroke-width', 1.5)
            .style('stroke-dasharray', '3,2')
            .style('opacity', 0.7);
          plotG
            .append('circle')
            .attr('cx', xScale(to[0]))
            .attr('cy', yScale(to[1]))
            .attr('r', 3)
            .style('fill', AMBER)
            .style('opacity', 0.6);
        }
      }

      // Start point
      plotG
        .append('rect')
        .attr('x', xScale(x0[0]) - 5)
        .attr('y', yScale(x0[1]) - 5)
        .attr('width', 10)
        .attr('height', 10)
        .style('fill', AMBER)
        .style('cursor', 'grab');

      // Optimal point
      plotG
        .append('text')
        .attr('x', xScale(0))
        .attr('y', yScale(0))
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .style('fill', '#DC2626')
        .style('font-size', '16px')
        .text('★');

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${panelH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', '#a1a1aa');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', '#a1a1aa');

      g.selectAll('.domain, .tick line').style('stroke', '#52525b');

      // Legend
      const legendData = [
        { color: TEAL, label: 'ISTA' },
        { color: PURPLE, label: 'FISTA' },
      ];
      legendData.forEach((d, i) => {
        g.append('line')
          .attr('x1', panelW - 75)
          .attr('y1', 8 + i * 16)
          .attr('x2', panelW - 60)
          .attr('y2', 8 + i * 16)
          .style('stroke', d.color)
          .style('stroke-width', 2);
        g.append('text')
          .attr('x', panelW - 56)
          .attr('y', 8 + i * 16)
          .attr('dominant-baseline', 'central')
          .style('fill', '#d4d4d8')
          .style('font-size', '11px')
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
        .text('ISTA vs FISTA trajectories');

      // Drag on start point
      const dragBehavior = d3
        .drag<SVGSVGElement, unknown>()
        .on('drag', (event) => {
          const gNode = svg.select('g').node() as SVGGElement;
          const pt = gNode.ownerSVGElement!.createSVGPoint();
          pt.x = event.sourceEvent.offsetX;
          pt.y = event.sourceEvent.offsetY;
          const newX = xScale.invert(pt.x - margin.left);
          const newY = yScale.invert(pt.y - margin.top);
          setX0([
            Math.max(-3.5, Math.min(3.5, newX)),
            Math.max(-3.5, Math.min(3.5, newY)),
          ]);
        });
      svg.call(dragBehavior);
    },
    [istaResult, fistaResult, containerWidth, isMobile, showMomentum, x0, kappa, quad],
  );

  // Right panel: convergence
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const panelW = isMobile
        ? containerWidth - margin.left - margin.right
        : Math.floor(containerWidth / 2) - margin.left - margin.right;
      if (panelW <= 0) return;
      const panelH = HEIGHT - margin.top - margin.bottom;

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const fStar = Math.min(
        ...istaResult.objs.slice(-10),
        ...fistaResult.objs.slice(-10),
      );

      const istaGap = istaResult.objs.map((v) => Math.max(v - fStar, 1e-15));
      const fistaGap = fistaResult.objs.map((v) => Math.max(v - fStar, 1e-15));

      const maxGap = Math.max(istaGap[0], fistaGap[0], 1);

      const xScale = d3.scaleLinear().domain([0, maxIter]).range([0, panelW]);
      const yScale = d3.scaleLog().domain([1e-12, maxGap * 2]).range([panelH, 0]);

      // ISTA line
      const line = d3
        .line<number>()
        .x((_, i) => xScale(i))
        .y((d) => yScale(Math.max(d, 1e-15)));

      g.append('path')
        .datum(istaGap)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', 2);

      // FISTA line
      g.append('path')
        .datum(fistaGap)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', PURPLE)
        .style('stroke-width', 2);

      // Rate envelopes — scale constant so the theoretical O(1/k) / O(1/k²)
      // curves visually bracket the empirical convergence curves.
      const ks = d3.range(1, maxIter + 1);
      const ENVELOPE_SCALE = 3; // visual scaling to sit above empirical curves
      const C1 = istaGap[0] * ENVELOPE_SCALE;

      g.append('path')
        .datum(ks)
        .attr(
          'd',
          d3
            .line<number>()
            .x((k) => xScale(k))
            .y((k) => yScale(Math.max(C1 / k, 1e-15))),
        )
        .style('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4,3')
        .style('opacity', 0.5);

      g.append('path')
        .datum(ks)
        .attr(
          'd',
          d3
            .line<number>()
            .x((k) => xScale(k))
            .y((k) => yScale(Math.max(C1 / (k * k), 1e-15))),
        )
        .style('fill', 'none')
        .style('stroke', PURPLE)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4,3')
        .style('opacity', 0.5);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${panelH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', '#a1a1aa');
      g.append('g')
        .call(
          d3
            .axisLeft(yScale)
            .ticks(5, '.0e'),
        )
        .selectAll('text')
        .style('fill', '#a1a1aa');

      g.selectAll('.domain, .tick line').style('stroke', '#52525b');

      // Labels
      g.append('text')
        .attr('x', panelW / 2)
        .attr('y', panelH + 35)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('Iteration k');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -panelH / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('F(xₖ) − F*');

      // Legend
      const legend = [
        { color: TEAL, label: 'ISTA O(1/k)', dash: '' },
        { color: PURPLE, label: 'FISTA O(1/k²)', dash: '' },
        { color: TEAL, label: 'O(1/k) env.', dash: '4,3' },
        { color: PURPLE, label: 'O(1/k²) env.', dash: '4,3' },
      ];
      legend.forEach((d, i) => {
        g.append('line')
          .attr('x1', panelW - 110)
          .attr('y1', 8 + i * 15)
          .attr('x2', panelW - 90)
          .attr('y2', 8 + i * 15)
          .style('stroke', d.color)
          .style('stroke-width', d.dash ? 1 : 2)
          .style('stroke-dasharray', d.dash);
        g.append('text')
          .attr('x', panelW - 86)
          .attr('y', 8 + i * 15)
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
        .text('Convergence: acceleration gap');
    },
    [istaResult, fistaResult, containerWidth, isMobile],
  );

  const halfW = isMobile ? containerWidth : Math.floor(containerWidth / 2);

  return (
    <div ref={containerRef} className="w-full my-6">
      <div className="flex flex-wrap items-center gap-4 mb-4 text-sm">
        <label className="flex items-center gap-2">
          <span className="text-zinc-400">κ</span>
          <input
            type="range"
            min={2}
            max={100}
            step={1}
            value={kappa}
            onChange={(e) => setKappa(Number(e.target.value))}
            className="w-32"
          />
          <span className="text-zinc-300 font-mono w-10">{kappa}</span>
        </label>
        <label className="flex items-center gap-2 text-zinc-400">
          <input
            type="checkbox"
            checked={showMomentum}
            onChange={(e) => setShowMomentum(e.target.checked)}
            className="accent-amber-500"
          />
          Show momentum
        </label>
      </div>
      <div className={`flex ${isMobile ? 'flex-col' : 'flex-row'} gap-0`}>
        <svg
          ref={leftRef}
          width={halfW}
          height={HEIGHT}
          className="overflow-visible"
        />
        <svg
          ref={rightRef}
          width={halfW}
          height={HEIGHT}
          className="overflow-visible"
        />
      </div>
    </div>
  );
}

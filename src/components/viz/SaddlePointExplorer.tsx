import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// ── Problem: min (x-3)^2 s.t. x ≤ c  → L(x,λ) = (x-3)^2 + λ(x-c)

const C = 1.5; // constraint RHS

function lagrangian(x: number, lam: number): number {
  return (x - 3) ** 2 + lam * (x - C);
}

/** g(λ) = inf_x L(x, λ) → x*(λ) = 3 - λ/2, g(λ) = -λ²/4 + λ(3 - C) - ? */
function dualFunc(lam: number): number {
  const xStar = 3 - lam / 2;
  return lagrangian(xStar, lam);
}

/** h(x) = max_{λ≥0} L(x, λ) */
function primalSlice(x: number): number {
  if (x <= C) return (x - 3) ** 2; // λ*=0 if feasible
  return Infinity; // infeasible → max_λ blows up
}

const X_RANGE: [number, number] = [-1, 6];
const LAM_RANGE: [number, number] = [0, 7];
const SADDLE_X = C; // x* = 1.5
const SADDLE_LAM = 2 * (3 - C); // λ* = 3
const SADDLE_VAL = lagrangian(SADDLE_X, SADDLE_LAM); // L(x*,λ*) = 2.25

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 50 };
const GRID_N = 60;

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';

// ── Component ────────────────────────────────────────────────────────

export default function SaddlePointExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [crosshair, setCrosshair] = useState<[number, number]>([SADDLE_X, SADDLE_LAM]);
  const [showCurves, setShowCurves] = useState(true);

  const isStacked = containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth
    : Math.floor((containerWidth - 16) / 2);

  // Compute heatmap grid
  const heatmapData = useMemo(() => {
    const data: { x: number; lam: number; val: number }[] = [];
    for (let j = 0; j < GRID_N; j++) {
      for (let i = 0; i < GRID_N; i++) {
        const x = X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * (i / (GRID_N - 1));
        const lam = LAM_RANGE[0] + (LAM_RANGE[1] - LAM_RANGE[0]) * (j / (GRID_N - 1));
        data.push({ x, lam, val: lagrangian(x, lam) });
      }
    }
    return data;
  }, []);

  const valExtent = useMemo(() => {
    const vals = heatmapData.map((d) => d.val);
    return [Math.max(d3.min(vals)!, -10), Math.min(d3.max(vals)!, 30)] as [number, number];
  }, [heatmapData]);

  // Slices at crosshair
  const xSlice = useMemo(() => {
    return Array.from({ length: 200 }, (_, i) => {
      const x = X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * (i / 199);
      return { x, val: lagrangian(x, crosshair[1]) };
    });
  }, [crosshair[1]]);

  const lamSlice = useMemo(() => {
    return Array.from({ length: 200 }, (_, i) => {
      const lam = LAM_RANGE[0] + (LAM_RANGE[1] - LAM_RANGE[0]) * (i / 199);
      return { lam, val: lagrangian(crosshair[0], lam) };
    });
  }, [crosshair[0]]);

  // ── Left panel: heatmap ───────────────────────────────────────────

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 100) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const xScale = d3.scaleLinear().domain(X_RANGE).range([0, plotW]);
      const yScale = d3.scaleLinear().domain(LAM_RANGE).range([plotH, 0]);

      const colorScale = d3
        .scaleSequential(d3.interpolateRdYlBu)
        .domain([valExtent[1], valExtent[0]]);

      // Heatmap cells
      const cellW = plotW / GRID_N;
      const cellH = plotH / GRID_N;

      g.selectAll('.heatcell')
        .data(heatmapData)
        .enter()
        .append('rect')
        .attr('x', (d) => xScale(d.x) - cellW / 2)
        .attr('y', (d) => yScale(d.lam) - cellH / 2)
        .attr('width', cellW + 0.5)
        .attr('height', cellH + 0.5)
        .style('fill', (d) => {
          const clamped = Math.max(valExtent[0], Math.min(valExtent[1], d.val));
          return colorScale(clamped);
        })
        .style('stroke', 'none');

      // min-x and max-λ curves
      if (showCurves) {
        // g(λ) = min_x L(x,λ) curve
        const gCurve = Array.from({ length: 200 }, (_, i) => {
          const lam = LAM_RANGE[0] + (LAM_RANGE[1] - LAM_RANGE[0]) * (i / 199);
          const xStar = 3 - lam / 2;
          return { x: xStar, lam, val: dualFunc(lam) };
        });

        const minXLine = d3
          .line<{ x: number; lam: number }>()
          .x((d) => xScale(d.x))
          .y((d) => yScale(d.lam));

        g.append('path')
          .datum(gCurve.filter((d) => d.x >= X_RANGE[0] && d.x <= X_RANGE[1]))
          .attr('d', minXLine)
          .style('fill', 'none')
          .style('stroke', TEAL)
          .style('stroke-width', '2.5')
          .style('stroke-dasharray', '5,3');
      }

      // Crosshairs
      g.append('line')
        .attr('x1', xScale(crosshair[0]))
        .attr('y1', 0)
        .attr('x2', xScale(crosshair[0]))
        .attr('y2', plotH)
        .style('stroke', '#d4d4d8')
        .style('stroke-width', '1')
        .style('opacity', 0.6);

      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(crosshair[1]))
        .attr('x2', plotW)
        .attr('y2', yScale(crosshair[1]))
        .style('stroke', '#d4d4d8')
        .style('stroke-width', '1')
        .style('opacity', 0.6);

      // Saddle point marker
      g.append('circle')
        .attr('cx', xScale(SADDLE_X))
        .attr('cy', yScale(SADDLE_LAM))
        .attr('r', 6)
        .style('fill', 'none')
        .style('stroke', '#fff')
        .style('stroke-width', '2');

      // Crosshair point (draggable)
      const crosshairPt = g
        .append('circle')
        .attr('cx', xScale(crosshair[0]))
        .attr('cy', yScale(crosshair[1]))
        .attr('r', 7)
        .style('fill', AMBER)
        .style('stroke', '#fff')
        .style('stroke-width', '1.5')
        .style('cursor', 'grab');

      const drag = d3
        .drag<SVGCircleElement, unknown>()
        .on('drag', (event) => {
          const newX = Math.max(X_RANGE[0], Math.min(X_RANGE[1], xScale.invert(event.x)));
          const newLam = Math.max(LAM_RANGE[0], Math.min(LAM_RANGE[1], yScale.invert(event.y)));
          setCrosshair([newX, newLam]);
        });

      crosshairPt.call(drag);

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

      g.append('text')
        .attr('x', plotW / 2)
        .attr('y', plotH + 35)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '11px')
        .text('x');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -plotH / 2)
        .attr('y', -38)
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
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('L(x, λ) Heatmap');
    },
    [panelWidth, heatmapData, crosshair, showCurves, valExtent],
  );

  // ── Right panel: 1D slices ────────────────────────────────────────

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 100) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      // Combine both slices for y-axis range
      const allVals = [
        ...xSlice.map((d) => d.val),
        ...lamSlice.map((d) => d.val),
      ].filter((v) => v > -20 && v < 30);
      const yMin = (d3.min(allVals) ?? -2) - 0.5;
      const yMax = (d3.max(allVals) ?? 15) + 0.5;

      // Top half: L(x, λ_fixed) — slice at fixed λ
      const halfH = (plotH - 20) / 2;

      // Top: x-slice
      {
        const xScale = d3.scaleLinear().domain(X_RANGE).range([0, plotW]);
        const yScaleTop = d3.scaleLinear().domain([yMax, yMin]).range([0, halfH]);

        const line = d3
          .line<{ x: number; val: number }>()
          .x((d) => xScale(d.x))
          .y((d) => yScaleTop(d.val));

        g.append('path')
          .datum(xSlice)
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', PURPLE)
          .style('stroke-width', '2');

        // Saddle value horizontal line
        g.append('line')
          .attr('x1', 0)
          .attr('y1', yScaleTop(SADDLE_VAL))
          .attr('x2', plotW)
          .attr('y2', yScaleTop(SADDLE_VAL))
          .style('stroke', '#a1a1aa')
          .style('stroke-dasharray', '3,3')
          .style('opacity', 0.5);

        // Crosshair x marker
        const lAtCross = lagrangian(crosshair[0], crosshair[1]);
        if (lAtCross > yMin && lAtCross < yMax) {
          g.append('circle')
            .attr('cx', xScale(crosshair[0]))
            .attr('cy', yScaleTop(lAtCross))
            .attr('r', 4)
            .style('fill', AMBER);
        }

        g.append('g')
          .attr('transform', `translate(0,${halfH})`)
          .call(d3.axisBottom(xScale).ticks(5))
          .call((g) => g.selectAll('text').style('fill', '#a1a1aa').style('font-size', '9px'))
          .call((g) => g.selectAll('line').style('stroke', '#52525b'))
          .call((g) => g.select('.domain').style('stroke', '#52525b'));

        g.append('g')
          .call(d3.axisLeft(yScaleTop).ticks(4))
          .call((g) => g.selectAll('text').style('fill', '#a1a1aa').style('font-size', '9px'))
          .call((g) => g.selectAll('line').style('stroke', '#52525b'))
          .call((g) => g.select('.domain').style('stroke', '#52525b'));

        g.append('text')
          .attr('x', plotW / 2)
          .attr('y', -4)
          .attr('text-anchor', 'middle')
          .style('fill', PURPLE)
          .style('font-size', '10px')
          .text(`L(x, λ=${crosshair[1].toFixed(1)}) — min over x`);
      }

      // Bottom: λ-slice
      {
        const yOffset = halfH + 20;
        const xScale = d3.scaleLinear().domain(LAM_RANGE).range([0, plotW]);
        const yScaleBot = d3.scaleLinear().domain([yMax, yMin]).range([0, halfH]);

        const gBot = g.append('g').attr('transform', `translate(0,${yOffset})`);

        const line = d3
          .line<{ lam: number; val: number }>()
          .x((d) => xScale(d.lam))
          .y((d) => yScaleBot(d.val));

        gBot
          .append('path')
          .datum(lamSlice)
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', TEAL)
          .style('stroke-width', '2');

        // Saddle value
        gBot
          .append('line')
          .attr('x1', 0)
          .attr('y1', yScaleBot(SADDLE_VAL))
          .attr('x2', plotW)
          .attr('y2', yScaleBot(SADDLE_VAL))
          .style('stroke', '#a1a1aa')
          .style('stroke-dasharray', '3,3')
          .style('opacity', 0.5);

        // Crosshair λ marker
        const lAtCross = lagrangian(crosshair[0], crosshair[1]);
        if (lAtCross > yMin && lAtCross < yMax) {
          gBot
            .append('circle')
            .attr('cx', xScale(crosshair[1]))
            .attr('cy', yScaleBot(lAtCross))
            .attr('r', 4)
            .style('fill', AMBER);
        }

        gBot
          .append('g')
          .attr('transform', `translate(0,${halfH})`)
          .call(d3.axisBottom(xScale).ticks(5))
          .call((g) => g.selectAll('text').style('fill', '#a1a1aa').style('font-size', '9px'))
          .call((g) => g.selectAll('line').style('stroke', '#52525b'))
          .call((g) => g.select('.domain').style('stroke', '#52525b'));

        gBot
          .append('g')
          .call(d3.axisLeft(yScaleBot).ticks(4))
          .call((g) => g.selectAll('text').style('fill', '#a1a1aa').style('font-size', '9px'))
          .call((g) => g.selectAll('line').style('stroke', '#52525b'))
          .call((g) => g.select('.domain').style('stroke', '#52525b'));

        gBot
          .append('text')
          .attr('x', plotW / 2)
          .attr('y', -4)
          .attr('text-anchor', 'middle')
          .style('fill', TEAL)
          .style('font-size', '10px')
          .text(`L(x=${crosshair[0].toFixed(1)}, λ) — max over λ`);
      }

      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Cross-Sections');
    },
    [panelWidth, xSlice, lamSlice, crosshair],
  );

  const currentL = lagrangian(crosshair[0], crosshair[1]);
  const isSaddle =
    Math.abs(crosshair[0] - SADDLE_X) < 0.3 && Math.abs(crosshair[1] - SADDLE_LAM) < 0.5;

  return (
    <div ref={containerRef} className="w-full my-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="text-xs text-zinc-400 flex items-center gap-2">
          <input
            type="checkbox"
            checked={showCurves}
            onChange={(e) => setShowCurves(e.target.checked)}
            className="accent-teal-500"
          />
          Show min-x curve (g(λ))
        </label>
      </div>

      {/* Panels */}
      <div className="flex gap-4" style={{ flexDirection: isStacked ? 'column' : 'row' }}>
        <svg ref={leftRef} />
        <svg ref={rightRef} />
      </div>

      {/* Readout */}
      <p className="text-xs font-mono text-zinc-500 mt-2">
        L({crosshair[0].toFixed(2)}, {crosshair[1].toFixed(2)}) = {currentL.toFixed(4)} |
        Saddle point: L({SADDLE_X}, {SADDLE_LAM}) = {SADDLE_VAL.toFixed(4)}
        {isSaddle ? ' ← you are at the saddle point!' : ''}
      </p>
    </div>
  );
}

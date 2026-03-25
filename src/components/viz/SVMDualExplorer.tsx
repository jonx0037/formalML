import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { solveSimpleSVM, generateSVMDataset } from './shared/dualityUtils';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 320;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 50 };

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const RED = '#DC2626';
const BLUE = '#2563EB';

const N_PER_CLASS = 10;
const INITIAL_DATA = generateSVMDataset(N_PER_CLASS, 42);

// ── Component ────────────────────────────────────────────────────────

export default function SVMDualExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const clipId = `clip-svm-${useId().replace(/:/g, '')}`;
  const [points, setPoints] = useState(INITIAL_DATA.X);
  const [labels] = useState(INITIAL_DATA.y);
  const [showDualVars, setShowDualVars] = useState(true);
  const [showCompSlack, setShowCompSlack] = useState(true);

  const isStacked = containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth
    : Math.floor((containerWidth - 24) / 3);

  // Solve SVM
  const solution = useMemo(() => {
    return solveSimpleSVM(points, labels);
  }, [points, labels]);

  // Compute margins for each point: yᵢ(w·xᵢ + b)
  const margins = useMemo(() => {
    return points.map((xi, i) => {
      const wTx = solution.w[0] * xi[0] + solution.w[1] * xi[1] + solution.b;
      return labels[i] * wTx;
    });
  }, [points, labels, solution]);

  // Data extent
  const extent = useMemo(() => {
    const allX = points.map((p) => p[0]);
    const allY = points.map((p) => p[1]);
    const xMin = (d3.min(allX) ?? -3) - 0.5;
    const xMax = (d3.max(allX) ?? 3) + 0.5;
    const yMin = (d3.min(allY) ?? -3) - 0.5;
    const yMax = (d3.max(allY) ?? 3) + 0.5;
    return { xMin, xMax, yMin, yMax };
  }, [points]);

  // ── Left panel: scatter + decision boundary ───────────────────────

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

      const xScale = d3.scaleLinear().domain([extent.xMin, extent.xMax]).range([0, plotW]);
      const yScale = d3.scaleLinear().domain([extent.yMin, extent.yMax]).range([plotH, 0]);

      g.append('defs')
        .append('clipPath')
        .attr('id', clipId)
        .append('rect')
        .attr('width', plotW)
        .attr('height', plotH);

      const plotArea = g.append('g').attr('clip-path', `url(#${clipId})`);

      const { w: wVec, b: bVal } = solution;
      const wNorm = Math.sqrt(wVec[0] ** 2 + wVec[1] ** 2);

      if (wNorm > 1e-6) {
        // Decision boundary: w·x + b = 0
        // Handle vertical boundary (wVec[1] ≈ 0) by drawing x = const
        const isVertical = Math.abs(wVec[1]) < 1e-8;

        const drawLine = (offset: number, color: string, dashed: boolean) => {
          let lx1: number, ly1: number, lx2: number, ly2: number;

          if (isVertical) {
            // Vertical line: x = -(b + offset) / w₁
            const xConst = -(bVal + offset) / wVec[0];
            lx1 = xConst; ly1 = extent.yMin;
            lx2 = xConst; ly2 = extent.yMax;
          } else {
            // Standard: x₂ = -(w₁x₁ + b + offset) / w₂
            lx1 = extent.xMin;
            lx2 = extent.xMax;
            ly1 = -(wVec[0] * lx1 + bVal + offset) / wVec[1];
            ly2 = -(wVec[0] * lx2 + bVal + offset) / wVec[1];
          }

          plotArea
            .append('line')
            .attr('x1', xScale(lx1))
            .attr('y1', yScale(ly1))
            .attr('x2', xScale(lx2))
            .attr('y2', yScale(ly2))
            .style('stroke', color)
            .style('stroke-width', dashed ? '1.5' : '2')
            .style('stroke-dasharray', dashed ? '5,3' : 'none');
        };

        // Margin boundaries
        drawLine(1, TEAL, true);
        drawLine(-1, TEAL, true);
        // Decision boundary
        drawLine(0, '#d4d4d8', false);

        // Margin band shading (skip for vertical boundaries — polygon degenerates)
        if (!isVertical) {
          const polyPts = [
            { x: extent.xMin, y: -(wVec[0] * extent.xMin + bVal + 1) / wVec[1] },
            { x: extent.xMax, y: -(wVec[0] * extent.xMax + bVal + 1) / wVec[1] },
            { x: extent.xMax, y: -(wVec[0] * extent.xMax + bVal - 1) / wVec[1] },
            { x: extent.xMin, y: -(wVec[0] * extent.xMin + bVal - 1) / wVec[1] },
          ];

          plotArea
            .append('polygon')
            .attr('points', polyPts.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(' '))
            .style('fill', TEAL)
            .style('opacity', 0.08);
        }
      }

      // Data points
      points.forEach((pt, i) => {
        const isSV = solution.svIndices.includes(i);
        const color = labels[i] === 1 ? BLUE : RED;

        // Support vector ring
        if (isSV) {
          plotArea
            .append('circle')
            .attr('cx', xScale(pt[0]))
            .attr('cy', yScale(pt[1]))
            .attr('r', 10)
            .style('fill', 'none')
            .style('stroke', AMBER)
            .style('stroke-width', '2');
        }

        const ptCircle = plotArea
          .append('circle')
          .attr('cx', xScale(pt[0]))
          .attr('cy', yScale(pt[1]))
          .attr('r', 5)
          .style('fill', color)
          .style('stroke', '#fff')
          .style('stroke-width', '1')
          .style('cursor', 'grab');

        // Drag behavior
        const drag = d3
          .drag<SVGCircleElement, unknown>()
          .on('drag', (event) => {
            const newX = xScale.invert(event.x);
            const newY = yScale.invert(event.y);
            setPoints((prev) => {
              const next = prev.map((p) => [...p]);
              next[i] = [newX, newY];
              return next;
            });
          });

        ptCircle.call(drag);
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

      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`SVM Decision Boundary (margin = ${solution.margin.toFixed(2)})`);
    },
    [panelWidth, points, labels, solution, extent],
  );

  // ── Center panel: dual variable bar chart ─────────────────────────

  const centerRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 80 || !showDualVars) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const n = solution.alpha.length;
      const maxAlpha = Math.max(0.1, ...solution.alpha) * 1.2;

      const xScale = d3
        .scaleBand<number>()
        .domain(d3.range(n))
        .range([0, plotW])
        .padding(0.15);

      const yScale = d3.scaleLinear().domain([0, maxAlpha]).range([plotH, 0]);

      // Bars
      g.selectAll('.alpha-bar')
        .data(solution.alpha)
        .enter()
        .append('rect')
        .attr('x', (_, i) => xScale(i)!)
        .attr('y', (d) => yScale(d))
        .attr('width', xScale.bandwidth())
        .attr('height', (d) => plotH - yScale(d))
        .style('fill', (d, i) =>
          solution.svIndices.includes(i)
            ? AMBER
            : labels[i] === 1
              ? BLUE
              : RED,
        )
        .style('opacity', (d) => (d > 1e-4 ? 1 : 0.2));

      // Zero line
      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(0))
        .attr('x2', plotW)
        .attr('y2', yScale(0))
        .style('stroke', '#52525b');

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${plotH})`)
        .call(
          d3.axisBottom(xScale).tickFormat((i) => `${i + 1}`),
        )
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa').style('font-size', '8px'))
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
        .text('Point index');

      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Dual Variables αᵢ');
    },
    [panelWidth, solution, showDualVars, labels],
  );

  // ── Right panel: complementary slackness ──────────────────────────

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 80 || !showCompSlack) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const n = margins.length;
      const maxMargin = Math.max(3, ...margins.map(Math.abs)) * 1.1;

      const xScale = d3
        .scaleBand<number>()
        .domain(d3.range(n))
        .range([0, plotW])
        .padding(0.15);

      const yScale = d3.scaleLinear().domain([0, maxMargin]).range([plotH, 0]);

      // Margin = 1 reference line
      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(1))
        .attr('x2', plotW)
        .attr('y2', yScale(1))
        .style('stroke', AMBER)
        .style('stroke-dasharray', '5,3')
        .style('stroke-width', '1.5');

      g.append('text')
        .attr('x', plotW - 4)
        .attr('y', yScale(1) - 6)
        .attr('text-anchor', 'end')
        .style('fill', AMBER)
        .style('font-size', '10px')
        .text('margin = 1');

      // Margin dots
      g.selectAll('.margin-dot')
        .data(margins)
        .enter()
        .append('circle')
        .attr('cx', (_, i) => xScale(i)! + xScale.bandwidth() / 2)
        .attr('cy', (d) => yScale(Math.max(0, d)))
        .attr('r', 4)
        .style('fill', (d, i) => {
          if (solution.svIndices.includes(i)) return AMBER;
          return labels[i] === 1 ? BLUE : RED;
        })
        .style('opacity', (d, i) => (solution.svIndices.includes(i) ? 1 : 0.6));

      // Stems
      g.selectAll('.margin-stem')
        .data(margins)
        .enter()
        .append('line')
        .attr('x1', (_, i) => xScale(i)! + xScale.bandwidth() / 2)
        .attr('y1', yScale(0))
        .attr('x2', (_, i) => xScale(i)! + xScale.bandwidth() / 2)
        .attr('y2', (d) => yScale(Math.max(0, d)))
        .style('stroke', (d, i) => {
          if (solution.svIndices.includes(i)) return AMBER;
          return '#52525b';
        })
        .style('stroke-width', '1')
        .style('opacity', 0.5);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${plotH})`)
        .call(
          d3.axisBottom(xScale).tickFormat((i) => `${i + 1}`),
        )
        .call((g) => g.selectAll('text').style('fill', '#a1a1aa').style('font-size', '8px'))
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
        .text('Point index');

      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Complementary Slackness');
    },
    [panelWidth, margins, solution, showCompSlack, labels],
  );

  return (
    <div ref={containerRef} className="w-full my-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="text-xs text-zinc-400 flex items-center gap-2">
          <input
            type="checkbox"
            checked={showDualVars}
            onChange={(e) => setShowDualVars(e.target.checked)}
            className="accent-amber-500"
          />
          Show dual variables
        </label>

        <label className="text-xs text-zinc-400 flex items-center gap-2">
          <input
            type="checkbox"
            checked={showCompSlack}
            onChange={(e) => setShowCompSlack(e.target.checked)}
            className="accent-amber-500"
          />
          Show complementary slackness
        </label>

        <button
          onClick={() => {
            const newData = generateSVMDataset(N_PER_CLASS, Math.floor(Math.random() * 10000));
            setPoints(newData.X);
          }}
          className="text-xs bg-zinc-800 text-zinc-300 rounded px-2 py-1 border border-zinc-600 hover:bg-zinc-700"
        >
          Regenerate data
        </button>
      </div>

      {/* Panels */}
      <div className="flex gap-2" style={{ flexDirection: isStacked ? 'column' : 'row' }}>
        <svg ref={leftRef} />
        {showDualVars && <svg ref={centerRef} />}
        {showCompSlack && <svg ref={rightRef} />}
      </div>

      {/* Readout */}
      <p className="text-xs font-mono text-zinc-500 mt-2">
        Support vectors: {solution.svIndices.length} / {points.length} |
        margin = {solution.margin.toFixed(4)} |
        w = [{solution.w.map((v) => v.toFixed(3)).join(', ')}], b = {solution.b.toFixed(3)}
      </p>
    </div>
  );
}

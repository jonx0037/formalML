import { useState, useEffect, useRef, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 30, left: 40 };
const EPS = 0.05;

// ─── Function definitions with analytical subdifferentials ───

interface FuncDef {
  fn: (x: number) => number;
  label: string;
  subdiff: (x: number) => [number, number];
  subdiffPlot: (x: number) => { lower: number; upper: number } | null;
}

const FUNCTIONS: Record<string, FuncDef> = {
  abs: {
    label: '|x|',
    fn: (x: number) => Math.abs(x),
    subdiff: (x: number): [number, number] => {
      if (x > EPS) return [1, 1];
      if (x < -EPS) return [-1, -1];
      return [-1, 1];
    },
    subdiffPlot: (x: number) => {
      if (x > EPS) return { lower: 1, upper: 1 };
      if (x < -EPS) return { lower: -1, upper: -1 };
      return { lower: -1, upper: 1 };
    },
  },
  relu: {
    label: 'max(0, x)',
    fn: (x: number) => Math.max(0, x),
    subdiff: (x: number): [number, number] => {
      if (x > EPS) return [1, 1];
      if (x < -EPS) return [0, 0];
      return [0, 1];
    },
    subdiffPlot: (x: number) => {
      if (x > EPS) return { lower: 1, upper: 1 };
      if (x < -EPS) return { lower: 0, upper: 0 };
      return { lower: 0, upper: 1 };
    },
  },
  absQuad: {
    label: '|x−1| + 0.3x²',
    fn: (x: number) => Math.abs(x - 1) + 0.3 * x * x,
    subdiff: (x: number): [number, number] => {
      const quadPart = 0.6 * x;
      if (x - 1 > EPS) return [1 + quadPart, 1 + quadPart];
      if (x - 1 < -EPS) return [-1 + quadPart, -1 + quadPart];
      return [-1 + quadPart, 1 + quadPart];
    },
    subdiffPlot: (x: number) => {
      const quadPart = 0.6 * x;
      if (x - 1 > EPS) return { lower: 1 + quadPart, upper: 1 + quadPart };
      if (x - 1 < -EPS) return { lower: -1 + quadPart, upper: -1 + quadPart };
      return { lower: -1 + quadPart, upper: 1 + quadPart };
    },
  },
  maxPiece: {
    label: 'max(x², 2|x|−1)',
    fn: (x: number) => Math.max(x * x, 2 * Math.abs(x) - 1),
    subdiff: (x: number): [number, number] => {
      const piece1 = x * x;
      const piece2 = 2 * Math.abs(x) - 1;
      const diff = piece1 - piece2;

      if (Math.abs(diff) < EPS) {
        // At transition: subdifferential is the convex hull of both slopes
        const slope1 = 2 * x;
        const slope2 = x >= 0 ? 2 : -2;
        return [Math.min(slope1, slope2), Math.max(slope1, slope2)];
      }

      if (piece1 > piece2) {
        const slope = 2 * x;
        return [slope, slope];
      }

      // piece2 active: 2|x|-1
      if (x > EPS) return [2, 2];
      if (x < -EPS) return [-2, -2];
      return [-2, 2];
    },
    subdiffPlot: (x: number) => {
      const piece1 = x * x;
      const piece2 = 2 * Math.abs(x) - 1;
      const diff = piece1 - piece2;

      if (Math.abs(diff) < EPS) {
        const slope1 = 2 * x;
        const slope2 = x >= 0 ? 2 : -2;
        return { lower: Math.min(slope1, slope2), upper: Math.max(slope1, slope2) };
      }

      if (piece1 > piece2) {
        const slope = 2 * x;
        return { lower: slope, upper: slope };
      }

      if (x > EPS) return { lower: 2, upper: 2 };
      if (x < -EPS) return { lower: -2, upper: -2 };
      return { lower: -2, upper: 2 };
    },
  },
};

type FuncKey = keyof typeof FUNCTIONS;

// ─── Component ───

export default function SubgradientExplorer() {
  const instanceId = useId();
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const [selectedFunc, setSelectedFunc] = useState<FuncKey>('abs');
  const [xPos, setXPos] = useState(0.0);

  const funcDef = FUNCTIONS[selectedFunc];

  // Domain and range
  const xDomain: [number, number] = [-3, 3];
  const numSamples = 400;

  const functionData = useMemo(() => {
    const step = (xDomain[1] - xDomain[0]) / numSamples;
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= numSamples; i++) {
      const x = xDomain[0] + i * step;
      pts.push({ x, y: funcDef.fn(x) });
    }
    return pts;
  }, [selectedFunc]);

  const yRange = useMemo(() => {
    const ys = functionData.map((d) => d.y);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const pad = (yMax - yMin) * 0.15 || 1;
    return [yMin - pad, yMax + pad] as [number, number];
  }, [functionData]);

  // Subdifferential data for right panel
  const subdiffData = useMemo(() => {
    const step = (xDomain[1] - xDomain[0]) / numSamples;
    const pts: { x: number; lower: number; upper: number }[] = [];
    for (let i = 0; i <= numSamples; i++) {
      const x = xDomain[0] + i * step;
      const sd = funcDef.subdiffPlot(x);
      if (sd) pts.push({ x, ...sd });
    }
    return pts;
  }, [selectedFunc]);

  const subdiffYRange = useMemo(() => {
    const vals = subdiffData.flatMap((d) => [d.lower, d.upper]);
    const mn = Math.min(...vals, -0.5);
    const mx = Math.max(...vals, 0.5);
    const pad = (mx - mn) * 0.15 || 1;
    return [mn - pad, mx + pad] as [number, number];
  }, [subdiffData]);

  // Find minimizer (where 0 is in subdifferential)
  const minimizerX = useMemo(() => {
    const step = 0.001;
    for (let x = xDomain[0]; x <= xDomain[1]; x += step) {
      const sd = funcDef.subdiffPlot(x);
      if (sd && sd.lower <= 0 && sd.upper >= 0) return x;
    }
    return null;
  }, [selectedFunc]);

  // Determine layout
  const isWide = containerWidth >= SM_BREAKPOINT;
  const panelWidth = isWide ? containerWidth / 2 : containerWidth;
  const panelHeight = isWide ? 360 : 280;
  const totalHeight = isWide ? panelHeight : panelHeight * 2;

  // Reset xPos when switching functions
  useEffect(() => {
    setXPos(0.0);
  }, [selectedFunc]);

  // ─── Left panel: function with subgradient fan ───
  useEffect(() => {
    const svg = d3.select(leftSvgRef.current);
    svg.selectAll('*').remove();
    if (!panelWidth || panelWidth < 50) return;

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = panelHeight - MARGIN.top - MARGIN.bottom;

    const g = svg
      .attr('width', panelWidth)
      .attr('height', panelHeight)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain(xDomain).range([0, w]);
    const yScale = d3.scaleLinear().domain(yRange).range([h, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick line').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').style('font-size', '11px'));

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick line').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').style('font-size', '11px'));

    // Title
    svg
      .append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text(`f(x) = ${funcDef.label}`);

    // Clip path
    g.append('defs')
      .append('clipPath')
      .attr('id', `subgrad-left-${instanceId.replace(/:/g, '')}`)
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', w)
      .attr('height', h);

    const plotArea = g.append('g').attr('clip-path', `url(#subgrad-left-${instanceId.replace(/:/g, '')})`);

    // Function curve
    const line = d3
      .line<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y))
      .curve(d3.curveLinear);

    plotArea
      .append('path')
      .datum(functionData)
      .attr('d', line)
      .attr('fill', 'none')
      .style('stroke', 'var(--color-text)')
      .attr('stroke-width', 2);

    // Subgradient fan lines
    const currentY = funcDef.fn(xPos);
    const [gMin, gMax] = funcDef.subdiff(xPos);
    const numLines = 15;

    for (let i = 0; i < numLines; i++) {
      const t = numLines === 1 ? 0.5 : i / (numLines - 1);
      const g_val = gMin + t * (gMax - gMin);

      // Line: y = currentY + g_val * (x - xPos)
      const x0 = xDomain[0];
      const x1 = xDomain[1];
      const y0 = currentY + g_val * (x0 - xPos);
      const y1_val = currentY + g_val * (x1 - xPos);

      // Determine alpha: higher at edges, lower in middle
      const edgeDist = Math.min(t, 1 - t);
      const alpha = 0.25 + 0.55 * (1 - edgeDist * 2);

      plotArea
        .append('line')
        .attr('x1', xScale(x0))
        .attr('y1', yScale(y0))
        .attr('x2', xScale(x1))
        .attr('y2', yScale(y1_val))
        .attr('stroke', `rgba(220, 80, 40, ${alpha})`)
        .attr('stroke-width', 1.2)
        .attr('stroke-dasharray', '4,3');
    }

    // Draggable point
    const dragCircle = plotArea
      .append('circle')
      .attr('cx', xScale(xPos))
      .attr('cy', yScale(currentY))
      .attr('r', 7)
      .attr('fill', '#dc5028')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'ew-resize');

    const drag = d3.drag<SVGCircleElement, unknown>().on('drag', (event) => {
      const newX = Math.max(xDomain[0], Math.min(xDomain[1], xScale.invert(event.x)));
      setXPos(newX);
    });

    dragCircle.call(drag);
  }, [containerWidth, selectedFunc, xPos, functionData, yRange, panelWidth, panelHeight]);

  // ─── Right panel: subdifferential map ───
  useEffect(() => {
    const svg = d3.select(rightSvgRef.current);
    svg.selectAll('*').remove();
    if (!panelWidth || panelWidth < 50) return;

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = panelHeight - MARGIN.top - MARGIN.bottom;

    const g = svg
      .attr('width', panelWidth)
      .attr('height', panelHeight)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain(xDomain).range([0, w]);
    const yScale = d3.scaleLinear().domain(subdiffYRange).range([h, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick line').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').style('font-size', '11px'));

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick line').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').style('font-size', '11px'));

    // Title
    svg
      .append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text('∂f(x) — Subdifferential Map');

    // Clip path
    g.append('defs')
      .append('clipPath')
      .attr('id', `subgrad-right-${instanceId.replace(/:/g, '')}`)
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', w)
      .attr('height', h);

    const plotArea = g.append('g').attr('clip-path', `url(#subgrad-right-${instanceId.replace(/:/g, '')})`);

    // Subdifferential: draw the lower and upper curves, with vertical segments at kinks
    // Split into segments based on where lower !== upper (kink regions)
    const smoothLower: { x: number; y: number }[] = [];
    const smoothUpper: { x: number; y: number }[] = [];
    const kinkSegments: { x: number; lower: number; upper: number }[] = [];

    for (const pt of subdiffData) {
      smoothLower.push({ x: pt.x, y: pt.lower });
      smoothUpper.push({ x: pt.x, y: pt.upper });
      if (Math.abs(pt.upper - pt.lower) > 0.01) {
        kinkSegments.push(pt);
      }
    }

    const lineGen = d3
      .line<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y))
      .curve(d3.curveLinear);

    // Lower subdiff curve
    plotArea
      .append('path')
      .datum(smoothLower)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .attr('stroke', '#6366f1')
      .attr('stroke-width', 2);

    // Upper subdiff curve (only draw where it differs from lower)
    const upperDiffPts = subdiffData.filter((d) => Math.abs(d.upper - d.lower) > 0.01);
    if (upperDiffPts.length > 0) {
      // For set-valued parts, draw upper as separate segments
      plotArea
        .append('path')
        .datum(smoothUpper)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', '#6366f1')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '2,2');
    }

    // Vertical kink segments
    for (const ks of kinkSegments) {
      plotArea
        .append('line')
        .attr('x1', xScale(ks.x))
        .attr('y1', yScale(ks.lower))
        .attr('x2', xScale(ks.x))
        .attr('y2', yScale(ks.upper))
        .attr('stroke', '#6366f1')
        .attr('stroke-width', 2.5)
        .attr('stroke-opacity', 0.6);
    }

    // Horizontal dashed line at y=0 (optimality condition)
    if (subdiffYRange[0] <= 0 && subdiffYRange[1] >= 0) {
      plotArea
        .append('line')
        .attr('x1', 0)
        .attr('y1', yScale(0))
        .attr('x2', w)
        .attr('y2', yScale(0))
        .attr('stroke', '#22c55e')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '6,4')
        .attr('stroke-opacity', 0.8);

      // Label
      g.append('text')
        .attr('x', w - 4)
        .attr('y', yScale(0) - 6)
        .attr('text-anchor', 'end')
        .style('fill', '#22c55e')
        .style('font-size', '11px')
        .text('0 ∈ ∂f(x*)');
    }

    // Minimizer star
    if (minimizerX !== null) {
      const starX = xScale(minimizerX);
      const starY = yScale(0);

      // Five-pointed star path
      const outerR = 8;
      const innerR = 3.5;
      let starPath = '';
      for (let i = 0; i < 10; i++) {
        const angle = (Math.PI / 2) + (i * Math.PI) / 5;
        const r = i % 2 === 0 ? outerR : innerR;
        const px = Math.cos(angle) * r;
        const py = -Math.sin(angle) * r;
        starPath += (i === 0 ? 'M' : 'L') + `${px},${py}`;
      }
      starPath += 'Z';

      plotArea
        .append('path')
        .attr('d', starPath)
        .attr('transform', `translate(${starX},${starY})`)
        .attr('fill', '#22c55e')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);
    }

    // Vertical cursor at current xPos
    const cursorSd = funcDef.subdiffPlot(xPos);
    plotArea
      .append('line')
      .attr('x1', xScale(xPos))
      .attr('y1', 0)
      .attr('x2', xScale(xPos))
      .attr('y2', h)
      .attr('stroke', '#dc5028')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('stroke-opacity', 0.7);

    // Show the subdifferential value(s) at current xPos
    if (cursorSd) {
      if (Math.abs(cursorSd.upper - cursorSd.lower) < 0.01) {
        // Single point
        plotArea
          .append('circle')
          .attr('cx', xScale(xPos))
          .attr('cy', yScale(cursorSd.lower))
          .attr('r', 5)
          .attr('fill', '#dc5028')
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);
      } else {
        // Interval
        plotArea
          .append('line')
          .attr('x1', xScale(xPos))
          .attr('y1', yScale(cursorSd.lower))
          .attr('x2', xScale(xPos))
          .attr('y2', yScale(cursorSd.upper))
          .attr('stroke', '#dc5028')
          .attr('stroke-width', 3)
          .attr('stroke-linecap', 'round');

        plotArea
          .append('circle')
          .attr('cx', xScale(xPos))
          .attr('cy', yScale(cursorSd.lower))
          .attr('r', 4)
          .attr('fill', '#dc5028')
          .attr('stroke', '#fff')
          .attr('stroke-width', 1);

        plotArea
          .append('circle')
          .attr('cx', xScale(xPos))
          .attr('cy', yScale(cursorSd.upper))
          .attr('r', 4)
          .attr('fill', '#dc5028')
          .attr('stroke', '#fff')
          .attr('stroke-width', 1);
      }
    }
  }, [containerWidth, selectedFunc, xPos, subdiffData, subdiffYRange, panelWidth, panelHeight, minimizerX]);

  // Guard for initial render before width measurement
  if (!containerWidth) {
    return <div ref={containerRef} style={{ minHeight: 400 }} />;
  }

  const currentSd = funcDef.subdiff(xPos);
  const sdLabel =
    Math.abs(currentSd[1] - currentSd[0]) < 0.01
      ? `{${currentSd[0].toFixed(2)}}`
      : `[${currentSd[0].toFixed(2)}, ${currentSd[1].toFixed(2)}]`;

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: isWide ? 'row' : 'column',
          width: '100%',
          gap: 0,
        }}
      >
        <svg role="img" aria-label="Subgradient explorer visualization (panel 1 of 2)" ref={leftSvgRef} />
        <svg role="img" aria-label="Subgradient explorer visualization (panel 2 of 2)" ref={rightSvgRef} />
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: isWide ? 'row' : 'column',
          alignItems: isWide ? 'center' : 'flex-start',
          justifyContent: 'space-between',
          gap: 12,
          padding: '8px 4px 0',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label
            htmlFor="func-select"
            style={{
              fontSize: '13px',
              color: 'var(--color-muted)',
              whiteSpace: 'nowrap',
            }}
          >
            Function:
          </label>
          <select
            id="func-select"
            value={selectedFunc}
            onChange={(e) => setSelectedFunc(e.target.value as FuncKey)}
            style={{
              fontSize: '13px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: 'transparent',
              color: 'var(--color-text)',
            }}
          >
            {Object.entries(FUNCTIONS).map(([key, def]) => (
              <option key={key} value={key}>
                {def.label}
              </option>
            ))}
          </select>
        </div>

        <div
          style={{
            fontSize: '13px',
            color: 'var(--color-muted)',
            fontFamily: 'monospace',
          }}
        >
          x = {xPos.toFixed(2)} &nbsp;|&nbsp; ∂f(x) = {sdLabel}
        </div>
      </div>
    </div>
  );
}

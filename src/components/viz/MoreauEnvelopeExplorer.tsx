import { useState, useMemo, useEffect, useRef, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { huber } from './shared/proximalUtils';

const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 50 };

// ─── Colors ───

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const SLATE = '#6B6B6B';
const RED = '#DC2626';

// ─── Function definitions ───

type FuncKey = 'abs' | 'relu' | 'indicator';

interface FuncDef {
  label: string;
  fn: (x: number) => number;
  prox: (v: number, lambda: number) => number;
  subdiff: (xDomain: [number, number]) => { segments: { x: number; yLow: number; yHigh: number }[] };
  xDomain: [number, number];
  yDomain: [number, number];
  gradYDomain: [number, number];
  huberOverlay?: boolean;
}

/** Tiny offset used to separate single-valued from set-valued segments in subdiff rendering. */
const SUBDIFF_KINK_EPSILON = 0.01;
/** Threshold for distinguishing single-valued vs set-valued subdiff segments in the plot. */
const SUBDIFF_SET_VALUED_THRESHOLD = 0.02;
/** Margin from y-axis bounds used to decide if a subdiff endpoint is "finite" (worth a dot). */
const SUBDIFF_FINITE_MARGIN = 0.5;
/** Number of grid points for Moreau envelope numerical minimization. */
const MOREAU_GRID_SIZE = 500;
/** Half-width of the grid search window relative to v. */
const MOREAU_GRID_HALFWIDTH = 5;

const FUNCTIONS: Record<FuncKey, FuncDef> = {
  abs: {
    label: '|x|',
    fn: (x) => Math.abs(x),
    prox: (v, lambda) => Math.sign(v) * Math.max(Math.abs(v) - lambda, 0),
    subdiff: (xDomain: [number, number]) => ({
      segments: [
        // sign(x) for x != 0, [-1,1] at x = 0
        { x: xDomain[0], yLow: -1, yHigh: -1 },
        { x: -SUBDIFF_KINK_EPSILON, yLow: -1, yHigh: -1 },
        { x: 0, yLow: -1, yHigh: 1 },
        { x: SUBDIFF_KINK_EPSILON, yLow: 1, yHigh: 1 },
        { x: xDomain[1], yLow: 1, yHigh: 1 },
      ],
    }),
    xDomain: [-3, 3],
    yDomain: [-0.5, 3.5],
    gradYDomain: [-1.5, 1.5],
    huberOverlay: true,
  },
  relu: {
    label: 'max(0, x)',
    fn: (x) => Math.max(0, x),
    prox: (v, lambda) => {
      if (v < 0) return v;
      if (v <= lambda) return 0;
      return v - lambda;
    },
    subdiff: (xDomain: [number, number]) => ({
      segments: [
        { x: xDomain[0], yLow: 0, yHigh: 0 },
        { x: -SUBDIFF_KINK_EPSILON, yLow: 0, yHigh: 0 },
        { x: 0, yLow: 0, yHigh: 1 },
        { x: SUBDIFF_KINK_EPSILON, yLow: 1, yHigh: 1 },
        { x: xDomain[1], yLow: 1, yHigh: 1 },
      ],
    }),
    xDomain: [-3, 3],
    yDomain: [-0.5, 3.5],
    gradYDomain: [-0.5, 1.5],
  },
  indicator: {
    label: 'ι_{[-1,1]}(x)',
    fn: (x) => (Math.abs(x) <= 1 ? 0 : Infinity),
    prox: (v, _lambda) => Math.max(-1, Math.min(1, v)),
    subdiff: (xDomain: [number, number]) => {
      // Normal cone: 0 in interior, ray outward at boundaries
      const rayExtent = Math.abs(xDomain[1]) * 2.5;
      return {
        segments: [
          { x: -1, yLow: -rayExtent, yHigh: 0 },
          { x: -1 + SUBDIFF_KINK_EPSILON, yLow: 0, yHigh: 0 },
          { x: 0, yLow: 0, yHigh: 0 },
          { x: 1 - SUBDIFF_KINK_EPSILON, yLow: 0, yHigh: 0 },
          { x: 1, yLow: 0, yHigh: rayExtent },
        ],
      };
    },
    xDomain: [-3, 3],
    yDomain: [-0.5, 4],
    gradYDomain: [-3, 3],
  },
};

// ─── Moreau envelope computation ───

function computeMoreauEnvelope(
  fFunc: (x: number) => number,
  v: number,
  lambda: number,
): number {
  const halfW = MOREAU_GRID_HALFWIDTH;
  const grid = Array.from(
    { length: MOREAU_GRID_SIZE },
    (_, i) => v - halfW + (i * 2 * halfW) / (MOREAU_GRID_SIZE - 1),
  );
  let minVal = Infinity;
  for (const x of grid) {
    const val = fFunc(x) + (1 / (2 * lambda)) * (x - v) ** 2;
    if (val < minVal) minVal = val;
  }
  return minVal;
}

function computeMoreauGradient(
  prox: (v: number, lambda: number) => number,
  v: number,
  lambda: number,
): number {
  return (1 / lambda) * (v - prox(v, lambda));
}

// ─── Component ───

export default function MoreauEnvelopeExplorer() {
  const instanceId = useId();
  const clipId = instanceId.replace(/:/g, '');
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const [selectedFunc, setSelectedFunc] = useState<FuncKey>('abs');
  const [lambda, setLambda] = useState(1.0);
  const [showGradient, setShowGradient] = useState(false);

  const funcDef = FUNCTIONS[selectedFunc];
  const numSamples = 400;

  // Precompute function curve and Moreau envelope
  const { funcData, envelopeData, huberData } = useMemo(() => {
    const [xMin, xMax] = funcDef.xDomain;
    const step = (xMax - xMin) / numSamples;
    const fPts: { x: number; y: number }[] = [];
    const mPts: { x: number; y: number }[] = [];
    const hPts: { x: number; y: number }[] = [];

    for (let i = 0; i <= numSamples; i++) {
      const x = xMin + i * step;
      const fVal = funcDef.fn(x);
      // For indicator, skip infinite values in the plot
      if (isFinite(fVal)) {
        fPts.push({ x, y: fVal });
      }
      mPts.push({ x, y: computeMoreauEnvelope(funcDef.fn, x, lambda) });
      if (funcDef.huberOverlay) {
        hPts.push({ x, y: huber(x, lambda) });
      }
    }
    return { funcData: fPts, envelopeData: mPts, huberData: hPts };
  }, [selectedFunc, lambda]);

  // Precompute gradient data
  const gradientData = useMemo(() => {
    const [xMin, xMax] = funcDef.xDomain;
    const step = (xMax - xMin) / numSamples;
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= numSamples; i++) {
      const x = xMin + i * step;
      pts.push({ x, y: computeMoreauGradient(funcDef.prox, x, lambda) });
    }
    return pts;
  }, [selectedFunc, lambda]);

  // Layout
  const isWide = containerWidth >= SM_BREAKPOINT;
  const panelWidth = isWide ? containerWidth / 2 : containerWidth;
  const panelHeight = 320;

  // ─── Left panel: f(x), Moreau envelope, optional Huber ───
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

    const xScale = d3.scaleLinear().domain(funcDef.xDomain).range([0, w]);

    // Dynamic y-range from envelope data
    const allY = [...funcData.map((d) => d.y), ...envelopeData.map((d) => d.y)];
    const yMin = Math.min(...allY, funcDef.yDomain[0]);
    const yMax = Math.max(...allY, funcDef.yDomain[1]);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);

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
      .text('Function & Moreau Envelope');

    // Clip path
    g.append('defs')
      .append('clipPath')
      .attr('id', `moreau-left-${clipId}`)
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', w)
      .attr('height', h);

    const plotArea = g.append('g').attr('clip-path', `url(#moreau-left-${clipId})`);

    const lineGen = d3
      .line<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y))
      .curve(d3.curveLinear);

    // For indicator function, draw vertical dashed lines at boundaries
    if (selectedFunc === 'indicator') {
      for (const bx of [-1, 1]) {
        plotArea
          .append('line')
          .attr('x1', xScale(bx))
          .attr('y1', yScale(yMin))
          .attr('x2', xScale(bx))
          .attr('y2', yScale(yMax))
          .style('stroke', SLATE)
          .style('stroke-width', '1.5')
          .style('stroke-dasharray', '4,3')
          .style('stroke-opacity', '0.6');
      }

      // Draw the zero segment on [-1, 1]
      plotArea
        .append('line')
        .attr('x1', xScale(-1))
        .attr('y1', yScale(0))
        .attr('x2', xScale(1))
        .attr('y2', yScale(0))
        .style('stroke', SLATE)
        .style('stroke-width', '2.5');

      // Dots at endpoints
      for (const bx of [-1, 1]) {
        plotArea
          .append('circle')
          .attr('cx', xScale(bx))
          .attr('cy', yScale(0))
          .attr('r', 4)
          .style('fill', SLATE)
          .style('stroke', '#fff')
          .style('stroke-width', '1.5');
      }

      // Upward arrows at boundaries
      for (const bx of [-1, 1]) {
        plotArea
          .append('line')
          .attr('x1', xScale(bx))
          .attr('y1', yScale(0))
          .attr('x2', xScale(bx))
          .attr('y2', yScale(yMax * 0.85))
          .style('stroke', SLATE)
          .style('stroke-width', '2')
          .style('stroke-dasharray', '3,2');
      }
    } else {
      // Draw f(x) curve
      plotArea
        .append('path')
        .datum(funcData)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .style('stroke', SLATE)
        .style('stroke-width', '2');
    }

    // Moreau envelope curve
    plotArea
      .append('path')
      .datum(envelopeData)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .style('stroke', TEAL)
      .style('stroke-width', '2.5');

    // Huber overlay for |x|
    if (funcDef.huberOverlay && huberData.length > 0) {
      plotArea
        .append('path')
        .datum(huberData)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .style('stroke', AMBER)
        .style('stroke-width', '2')
        .style('stroke-dasharray', '6,3');
    }

    // Legend
    const legendX = w - 10;
    let legendY = 12;
    const legendItems: { color: string; dash: string; label: string }[] = [
      { color: SLATE, dash: '', label: `f(x) = ${funcDef.label}` },
      { color: TEAL, dash: '', label: `Mλf(x), λ = ${lambda.toFixed(1)}` },
    ];
    if (funcDef.huberOverlay) {
      legendItems.push({ color: AMBER, dash: '6,3', label: 'Huber(x, λ)' });
    }

    for (const item of legendItems) {
      g.append('line')
        .attr('x1', legendX - 60)
        .attr('y1', legendY)
        .attr('x2', legendX - 40)
        .attr('y2', legendY)
        .style('stroke', item.color)
        .style('stroke-width', '2')
        .style('stroke-dasharray', item.dash);

      g.append('text')
        .attr('x', legendX - 35)
        .attr('y', legendY + 4)
        .attr('text-anchor', 'start')
        .style('fill', 'var(--color-muted)')
        .style('font-size', '11px')
        .text(item.label);

      legendY += 16;
    }

    // x-axis label (inside the main g element for consistent positioning)
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 34)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-size', '11px')
      .text('x');
  }, [containerWidth, selectedFunc, lambda, funcData, envelopeData, huberData, panelWidth]);

  // ─── Right panel: subdifferential & Moreau gradient ───
  useEffect(() => {
    const svg = d3.select(rightSvgRef.current);
    svg.selectAll('*').remove();
    if (!panelWidth || panelWidth < 50) return;
    if (!showGradient) {
      svg.attr('width', panelWidth).attr('height', panelHeight);
      // Empty state message
      svg
        .append('text')
        .attr('x', panelWidth / 2)
        .attr('y', panelHeight / 2)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-muted)')
        .style('font-size', '13px')
        .text('Enable "Show gradient" to see ∂f vs ∇Mλf');
      return;
    }

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = panelHeight - MARGIN.top - MARGIN.bottom;

    const g = svg
      .attr('width', panelWidth)
      .attr('height', panelHeight)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain(funcDef.xDomain).range([0, w]);

    // y range from gradient data + subdiff bounds
    const gradYs = gradientData.map((d) => d.y);
    const yMin = Math.min(...gradYs, funcDef.gradYDomain[0]);
    const yMax = Math.max(...gradYs, funcDef.gradYDomain[1]);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);

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
      .text('Subdifferential vs Moreau Gradient');

    // Clip path
    g.append('defs')
      .append('clipPath')
      .attr('id', `moreau-right-${clipId}`)
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', w)
      .attr('height', h);

    const plotArea = g.append('g').attr('clip-path', `url(#moreau-right-${clipId})`);

    // Zero line
    if (yMin <= 0 && yMax >= 0) {
      plotArea
        .append('line')
        .attr('x1', 0)
        .attr('y1', yScale(0))
        .attr('x2', w)
        .attr('y2', yScale(0))
        .style('stroke', 'var(--color-border)')
        .style('stroke-width', '1')
        .style('stroke-dasharray', '4,4')
        .style('stroke-opacity', '0.5');
    }

    // Subdifferential — draw as thick segments
    const subdiffInfo = funcDef.subdiff(funcDef.xDomain);
    const segs = subdiffInfo.segments;

    // Draw single-valued parts as lines
    for (let i = 0; i < segs.length - 1; i++) {
      const s0 = segs[i];
      const s1 = segs[i + 1];
      // Only connect if both are single-valued (no jump)
      if (Math.abs(s0.yHigh - s0.yLow) < SUBDIFF_SET_VALUED_THRESHOLD && Math.abs(s1.yHigh - s1.yLow) < SUBDIFF_SET_VALUED_THRESHOLD) {
        plotArea
          .append('line')
          .attr('x1', xScale(s0.x))
          .attr('y1', yScale(s0.yLow))
          .attr('x2', xScale(s1.x))
          .attr('y2', yScale(s1.yLow))
          .style('stroke', PURPLE)
          .style('stroke-width', '3')
          .style('stroke-opacity', '0.7');
      }
    }

    // Draw set-valued parts as thick vertical segments
    for (const seg of segs) {
      if (Math.abs(seg.yHigh - seg.yLow) >= SUBDIFF_SET_VALUED_THRESHOLD) {
        // Clamp to visible range
        const drawLow = Math.max(seg.yLow, yMin);
        const drawHigh = Math.min(seg.yHigh, yMax);
        plotArea
          .append('line')
          .attr('x1', xScale(seg.x))
          .attr('y1', yScale(drawLow))
          .attr('x2', xScale(seg.x))
          .attr('y2', yScale(drawHigh))
          .style('stroke', PURPLE)
          .style('stroke-width', '4')
          .style('stroke-opacity', '0.7')
          .style('stroke-linecap', 'round');

        // Endpoint dots for finite endpoints
        if (seg.yLow > yMin + SUBDIFF_FINITE_MARGIN) {
          plotArea
            .append('circle')
            .attr('cx', xScale(seg.x))
            .attr('cy', yScale(seg.yLow))
            .attr('r', 4)
            .style('fill', PURPLE)
            .style('stroke', '#fff')
            .style('stroke-width', '1.5');
        }
        if (seg.yHigh < yMax - SUBDIFF_FINITE_MARGIN) {
          plotArea
            .append('circle')
            .attr('cx', xScale(seg.x))
            .attr('cy', yScale(seg.yHigh))
            .attr('r', 4)
            .style('fill', PURPLE)
            .style('stroke', '#fff')
            .style('stroke-width', '1.5');
        }
      }
    }

    // Moreau gradient — smooth curve
    const lineGen = d3
      .line<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y((d) => yScale(d.y))
      .curve(d3.curveLinear);

    plotArea
      .append('path')
      .datum(gradientData)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .style('stroke', RED)
      .style('stroke-width', '2');

    // Legend
    const legendX = w - 10;
    let legendY = 12;

    g.append('line')
      .attr('x1', legendX - 70)
      .attr('y1', legendY)
      .attr('x2', legendX - 50)
      .attr('y2', legendY)
      .style('stroke', PURPLE)
      .style('stroke-width', '3')
      .style('stroke-opacity', '0.7');

    g.append('text')
      .attr('x', legendX - 45)
      .attr('y', legendY + 4)
      .attr('text-anchor', 'start')
      .style('fill', 'var(--color-muted)')
      .style('font-size', '11px')
      .text('∂f(x)');

    legendY += 16;

    g.append('line')
      .attr('x1', legendX - 70)
      .attr('y1', legendY)
      .attr('x2', legendX - 50)
      .attr('y2', legendY)
      .style('stroke', RED)
      .style('stroke-width', '2');

    g.append('text')
      .attr('x', legendX - 45)
      .attr('y', legendY + 4)
      .attr('text-anchor', 'start')
      .style('fill', 'var(--color-muted)')
      .style('font-size', '11px')
      .text('∇Mλf(x)');

    // x-axis label (inside the main g element for consistent positioning)
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 34)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-size', '11px')
      .text('x');
  }, [containerWidth, selectedFunc, lambda, showGradient, gradientData, panelWidth]);

  // Guard for initial render before width measurement
  if (!containerWidth) {
    return <div ref={containerRef} style={{ minHeight: 400 }} />;
  }

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-4 text-sm">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label
            htmlFor={`moreau-func-${clipId}`}
            style={{ fontSize: '13px', color: 'var(--color-muted)', whiteSpace: 'nowrap' }}
          >
            Function:
          </label>
          <select
            id={`moreau-func-${clipId}`}
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
            <option value="abs">|x| — Huber connection</option>
            <option value="relu">max(0, x) — ReLU smoothing</option>
            <option value="indicator">ι_[-1,1] — distance-to-set</option>
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label
            htmlFor={`moreau-lambda-${clipId}`}
            style={{ fontSize: '13px', color: 'var(--color-muted)', whiteSpace: 'nowrap' }}
          >
            λ = {lambda.toFixed(1)}
          </label>
          <input
            id={`moreau-lambda-${clipId}`}
            type="range"
            min="0.1"
            max="5"
            step="0.1"
            value={lambda}
            onChange={(e) => setLambda(parseFloat(e.target.value))}
            style={{ width: 120, accentColor: TEAL }}
          />
        </div>

        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: '13px',
            color: 'var(--color-muted)',
            cursor: 'pointer',
          }}
        >
          <input
            type="checkbox"
            checked={showGradient}
            onChange={(e) => setShowGradient(e.target.checked)}
            style={{ accentColor: RED }}
          />
          Show gradient
        </label>
      </div>

      {/* Panels */}
      <div
        style={{
          display: 'flex',
          flexDirection: isWide ? 'row' : 'column',
          width: '100%',
          gap: 0,
        }}
      >
        <svg ref={leftSvgRef} />
        <svg ref={rightSvgRef} />
      </div>
    </div>
  );
}

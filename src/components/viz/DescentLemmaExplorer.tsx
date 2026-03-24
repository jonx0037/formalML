import { useState, useEffect, useRef, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 30, left: 40 };
const X_DOMAIN: [number, number] = [-3, 3];
const NUM_SAMPLES = 500;

// ─── Function library ───

const FUNCTIONS: Record<
  string,
  {
    fn: (x: number) => number;
    df: (x: number) => number;
    trueL: number | ((x: number) => number);
    label: string;
  }
> = {
  x2half: { fn: (x) => (x * x) / 2, df: (x) => x, trueL: 1, label: 'x\u00B2/2' },
  logistic: {
    fn: (x) => Math.log(1 + Math.exp(x)),
    df: (x) => 1 / (1 + Math.exp(-x)),
    trueL: 0.25,
    label: 'log(1+e\u02E3)',
  },
  x4quarter: {
    fn: (x) => Math.pow(x, 4) / 4,
    df: (x) => Math.pow(x, 3),
    trueL: (x: number) => 3 * x * x,
    label: 'x\u2074/4',
  },
};

// ─── Helpers ───

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function sampleFunction(fn: (x: number) => number): { x: number; y: number }[] {
  const step = (X_DOMAIN[1] - X_DOMAIN[0]) / (NUM_SAMPLES - 1);
  return Array.from({ length: NUM_SAMPLES }, (_, i) => {
    const x = X_DOMAIN[0] + i * step;
    return { x, y: fn(x) };
  });
}

// ─── Component ───

export default function DescentLemmaExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId();

  // ─── State ───
  const [xPos, setXPos] = useState(1.5);
  const [L, setL] = useState(2.0);
  const [fnKey, setFnKey] = useState<string>('x2half');
  const [showTangent, setShowTangent] = useState(false);

  const activeFn = FUNCTIONS[fnKey];
  const fn = activeFn.fn;
  const df = activeFn.df;

  // ─── Sampled data ───
  const samples = useMemo(() => sampleFunction(fn), [fn]);

  // ─── Y domain ───
  const yExtent = useMemo(() => {
    const ys = samples.map((d) => d.y);
    // Also include the quadratic bound values around xPos for a reasonable range
    const qMax = fn(xPos) + Math.abs(df(xPos)) * 3 + (L / 2) * 9;
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys, qMax);
    const pad = (yMax - yMin) * 0.15 || 1;
    return [yMin - pad, Math.min(yMax + pad, 20)] as [number, number];
  }, [samples, fn, df, xPos, L]);

  // ─── Panel dimensions ───
  const panelWidth = useMemo(() => {
    if (!containerWidth) return 0;
    return Math.min(containerWidth, 700);
  }, [containerWidth]);
  const panelHeight = 320;
  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = panelHeight - MARGIN.top - MARGIN.bottom;

  // ─── Scales ───
  const xScale = useMemo(
    () => d3.scaleLinear().domain(X_DOMAIN).range([0, innerW]),
    [innerW],
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain(yExtent).range([innerH, 0]),
    [innerH, yExtent],
  );

  // ─── Line generator ───
  const lineGen = useMemo(
    () =>
      d3
        .line<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y)),
    [xScale, yScale],
  );

  // ─── Derived values ───
  const trueL = typeof activeFn.trueL === 'function' ? activeFn.trueL(xPos) : activeFn.trueL;
  const gradVal = df(xPos);
  const fAtAnchor = fn(xPos);
  const nextX = xPos - gradVal / L;
  const decrease = (1 / (2 * L)) * gradVal * gradVal;
  const boundValid = L >= trueL;

  // ─── D3 rendering ───
  useEffect(() => {
    if (!svgRef.current || innerW <= 0 || innerH <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', panelWidth).attr('height', panelHeight);

    // Clip path
    svg
      .append('defs')
      .append('clipPath')
      .attr('id', clipId)
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', innerW)
      .attr('height', innerH);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // ── Axes ──
    const xAxisG = g
      .append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6));
    xAxisG.selectAll('text').style('fill', 'var(--color-muted)');
    xAxisG.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    const yAxisG = g.append('g').call(d3.axisLeft(yScale).ticks(5));
    yAxisG.selectAll('text').style('fill', 'var(--color-muted)');
    yAxisG.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    // Panel title
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text('Descent Lemma — Quadratic Upper Bound');

    const clipped = g.append('g').attr('clip-path', `url(#${clipId})`);

    // ── Quadratic upper bound: q(x) = f(xk) + f'(xk)(x - xk) + (L/2)(x - xk)² ──
    const qFn = (x: number) => fAtAnchor + gradVal * (x - xPos) + (L / 2) * (x - xPos) * (x - xPos);
    const qSamples = sampleFunction(qFn);

    // ── Shade guaranteed decrease region ──
    // Region between f(xPos) horizontal line and the minimum of the upper bound
    const qMin = fAtAnchor - decrease; // value of q at next iterate
    const fillColor = boundValid ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)';

    // Shade: a rectangle from qMin to fAtAnchor, horizontally around the next iterate
    // More precisely, shade where the upper bound is below f(xPos)
    const shadeSamples = qSamples.filter(
      (d) => d.y < fAtAnchor && d.y >= qMin - 0.5,
    );
    if (shadeSamples.length > 1) {
      const shadeArea = d3
        .area<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y0(() => yScale(fAtAnchor))
        .y1((d) => yScale(Math.max(d.y, qMin)));

      clipped
        .append('path')
        .datum(shadeSamples)
        .attr('d', shadeArea)
        .attr('fill', fillColor)
        .attr('stroke', 'none');
    }

    // ── Horizontal line at f(xPos) ──
    clipped
      .append('line')
      .attr('x1', xScale(X_DOMAIN[0]))
      .attr('y1', yScale(fAtAnchor))
      .attr('x2', xScale(X_DOMAIN[1]))
      .attr('y2', yScale(fAtAnchor))
      .attr('stroke', 'var(--color-muted)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.5);

    // ── Function curve ──
    clipped
      .append('path')
      .datum(samples)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .attr('stroke', '#0F6E56')
      .attr('stroke-width', 2);

    // ── Tangent line (optional) ──
    if (showTangent) {
      const tangentFn = (x: number) => fAtAnchor + gradVal * (x - xPos);
      const tangentSamples = [
        { x: X_DOMAIN[0], y: tangentFn(X_DOMAIN[0]) },
        { x: X_DOMAIN[1], y: tangentFn(X_DOMAIN[1]) },
      ];

      clipped
        .append('line')
        .attr('x1', xScale(tangentSamples[0].x))
        .attr('y1', yScale(tangentSamples[0].y))
        .attr('x2', xScale(tangentSamples[1].x))
        .attr('y2', yScale(tangentSamples[1].y))
        .attr('stroke', '#0F6E56')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '6,3')
        .attr('opacity', 0.6);

      clipped
        .append('text')
        .attr('x', xScale(X_DOMAIN[1]) - 4)
        .attr('y', yScale(tangentFn(X_DOMAIN[1])) - 6)
        .attr('text-anchor', 'end')
        .style('fill', '#0F6E56')
        .style('font-size', '11px')
        .style('opacity', '0.7')
        .text('tangent');
    }

    // ── Quadratic upper bound curve ──
    clipped
      .append('path')
      .datum(qSamples)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .attr('stroke', '#534AB7')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,3');

    // ── Next iterate marker ──
    const nextXClamped = clamp(nextX, X_DOMAIN[0], X_DOMAIN[1]);
    const nextY = fn(nextXClamped);

    clipped
      .append('circle')
      .attr('cx', xScale(nextXClamped))
      .attr('cy', yScale(nextY))
      .attr('r', 5)
      .attr('fill', '#534AB7')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);

    // Label for next iterate
    clipped
      .append('text')
      .attr('x', xScale(nextXClamped))
      .attr('y', yScale(nextY) + 16)
      .attr('text-anchor', 'middle')
      .style('fill', '#534AB7')
      .style('font-size', '11px')
      .text('x\u2096\u208A\u2081');

    // ── Decrease annotation ──
    if (decrease > 0.01) {
      // Vertical arrow showing the decrease
      const arrowX = xScale(nextXClamped) + 20;
      const arrowTop = yScale(fAtAnchor);
      const arrowBot = yScale(fAtAnchor - decrease);

      if (arrowBot > arrowTop + 8) {
        g.append('line')
          .attr('x1', arrowX)
          .attr('y1', arrowTop)
          .attr('x2', arrowX)
          .attr('y2', arrowBot)
          .attr('stroke', boundValid ? '#22c55e' : '#ef4444')
          .attr('stroke-width', 1.5);

        g.append('text')
          .attr('x', arrowX + 6)
          .attr('y', (arrowTop + arrowBot) / 2 + 4)
          .style('fill', boundValid ? '#22c55e' : '#ef4444')
          .style('font-size', '11px')
          .style('font-weight', '600')
          .text(`\u0394 = ${decrease.toFixed(3)}`);
      }
    }

    // ── Draggable anchor point ──
    const anchorCircle = g
      .append('circle')
      .attr('cx', xScale(xPos))
      .attr('cy', yScale(fAtAnchor))
      .attr('r', 7)
      .attr('fill', '#D97706')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .style('cursor', 'ew-resize');

    // Anchor label
    g.append('text')
      .attr('x', xScale(xPos))
      .attr('y', yScale(fAtAnchor) - 12)
      .attr('text-anchor', 'middle')
      .style('fill', '#D97706')
      .style('font-size', '11px')
      .text('x\u2096');

    // Drag behavior
    const dragBehavior = d3
      .drag<SVGCircleElement, unknown>()
      .on('start', function () {
        d3.select(this).style('cursor', 'grabbing');
      })
      .on('drag', function (event) {
        const svgRect = svgRef.current!.getBoundingClientRect();
        const mouseX = event.sourceEvent.clientX - svgRect.left - MARGIN.left;
        const newX = clamp(xScale.invert(mouseX), -2.8, 2.8);
        setXPos(newX);
      })
      .on('end', function () {
        d3.select(this).style('cursor', 'ew-resize');
      });

    anchorCircle.call(dragBehavior as any);

    // ── Legend ──
    const legendX = innerW - 150;
    const legendY = 10;
    const legendItems = [
      { color: '#0F6E56', label: `f(x) = ${activeFn.label}`, dash: '' },
      { color: '#534AB7', label: 'quadratic bound', dash: '6,3' },
    ];

    legendItems.forEach((item, i) => {
      const ly = legendY + i * 16;
      g.append('line')
        .attr('x1', legendX)
        .attr('y1', ly)
        .attr('x2', legendX + 18)
        .attr('y2', ly)
        .attr('stroke', item.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', item.dash);

      g.append('text')
        .attr('x', legendX + 24)
        .attr('y', ly + 4)
        .style('fill', 'var(--color-muted)')
        .style('font-size', '11px')
        .text(item.label);
    });
  }, [
    xPos,
    L,
    fn,
    df,
    fnKey,
    activeFn,
    showTangent,
    samples,
    fAtAnchor,
    gradVal,
    nextX,
    decrease,
    boundValid,
    trueL,
    innerW,
    innerH,
    panelWidth,
    panelHeight,
    xScale,
    yScale,
    lineGen,
    clipId,
  ]);

  // ─── Guard: no width yet ───
  if (!containerWidth) return <div ref={containerRef} style={{ minHeight: 400 }} />;

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} />

      {/* Controls */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '16px',
          alignItems: 'center',
          marginTop: '8px',
          fontSize: '0.85rem',
        }}
      >
        {/* Function selector */}
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 13,
            color: 'var(--color-text)',
          }}
        >
          f(x):
          <select
            value={fnKey}
            onChange={(e) => {
              setFnKey(e.target.value);
              setXPos(1.5);
            }}
            style={{
              padding: '2px 6px',
              borderRadius: 4,
              border: '1px solid var(--color-border)',
              background: 'transparent',
              color: 'var(--color-text)',
              fontSize: 13,
            }}
          >
            {Object.entries(FUNCTIONS).map(([key, { label }]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>

        {/* L slider */}
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 13,
            color: 'var(--color-text)',
          }}
        >
          L: {L.toFixed(1)}
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.1}
            value={L}
            onChange={(e) => setL(Number(e.target.value))}
            style={{ width: 100 }}
          />
        </label>

        {/* Show tangent checkbox */}
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 13,
            color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          <input
            type="checkbox"
            checked={showTangent}
            onChange={(e) => setShowTangent(e.target.checked)}
          />
          Show tangent
        </label>

        {/* Decrease value display */}
        <span
          style={{
            fontSize: 13,
            color: boundValid ? '#22c55e' : '#ef4444',
            fontWeight: 600,
          }}
        >
          {'\u0394'} = {decrease.toFixed(4)}
          {!boundValid && ' (bound violated)'}
        </span>
      </div>

      {/* Status line */}
      <div
        style={{
          fontSize: 12,
          color: 'var(--color-muted)',
          marginTop: 4,
        }}
      >
        True smoothness at x\u2096:{' '}
        <span style={{ fontWeight: 600, color: 'var(--color-text)' }}>
          {trueL.toFixed(2)}
        </span>
        {' \u2014 Chosen L: '}
        <span
          style={{
            fontWeight: 600,
            color: boundValid ? '#22c55e' : '#ef4444',
          }}
        >
          {L.toFixed(1)} {boundValid ? '\u2265' : '<'} {trueL.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 30, left: 35 };
const X_DOMAIN: [number, number] = [-3, 3];
const NUM_SAMPLES = 500;

// ─── Function library ───

const FUNCTIONS: Record<string, { fn: (x: number) => number; label: string; convex: boolean }> = {
  'x2': { fn: (x) => x * x, label: 'x\u00B2', convex: true },
  'abs': { fn: (x) => Math.abs(x), label: '|x|', convex: true },
  'exp': { fn: (x) => Math.exp(x), label: 'e\u02E3', convex: true },
  'nonconvex': { fn: (x) => Math.sin(x) + 0.3 * x * x, label: 'sin(x) + 0.3x\u00B2', convex: false },
};

// ─── Helpers ───

function sampleFunction(fn: (x: number) => number): { x: number; y: number }[] {
  const step = (X_DOMAIN[1] - X_DOMAIN[0]) / (NUM_SAMPLES - 1);
  return Array.from({ length: NUM_SAMPLES }, (_, i) => {
    const x = X_DOMAIN[0] + i * step;
    return { x, y: fn(x) };
  });
}

function numericalDerivative(fn: (x: number) => number, x0: number): number {
  const h = 0.001;
  return (fn(x0 + h) - fn(x0 - h)) / (2 * h);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

// ─── Component ───

export default function ConvexFunctionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const chordRef = useRef<SVGSVGElement>(null);
  const epiRef = useRef<SVGSVGElement>(null);
  const tangentRef = useRef<SVGSVGElement>(null);

  // ─── State ───
  const [fnKey, setFnKey] = useState<string>('x2');
  const [theta, setTheta] = useState(0.5);
  const [showEpigraph, setShowEpigraph] = useState(true);
  const [jensenMode, setJensenMode] = useState(false);

  // Draggable points (x-values; y follows the function)
  const [chordA, setChordA] = useState(-2);
  const [chordB, setChordB] = useState(2);
  const [tangentX0, setTangentX0] = useState(0);

  const activeFn = FUNCTIONS[fnKey];
  const fn = activeFn.fn;
  const isConvex = activeFn.convex;

  // ─── Sampled data ───
  const samples = useMemo(() => sampleFunction(fn), [fn]);

  // ─── Y domain (shared across panels) ───
  const yExtent = useMemo(() => {
    const ys = samples.map((d) => d.y);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const pad = (yMax - yMin) * 0.15 || 1;
    return [yMin - pad, yMax + pad] as [number, number];
  }, [samples]);

  // ─── Panel dimensions ───
  const isWide = containerWidth >= SM_BREAKPOINT;
  const panelWidth = useMemo(() => {
    if (!containerWidth) return 0;
    return isWide ? Math.floor((containerWidth - 24) / 3) : containerWidth;
  }, [containerWidth, isWide]);
  const panelHeight = 280;
  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = panelHeight - MARGIN.top - MARGIN.bottom;

  // ─── Scales (shared) ───
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

  // ─── Draw axes helper ───
  const drawAxes = useCallback(
    (g: d3.Selection<SVGGElement, unknown, null, undefined>, label: string) => {
      // x axis
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-muted)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      // y axis
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
        .text(label);
    },
    [innerW, innerH, xScale, yScale],
  );

  // ─── Draw function curve helper ───
  const drawCurve = useCallback(
    (g: d3.Selection<SVGGElement, unknown, null, undefined>) => {
      g.append('path')
        .datum(samples)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', 'var(--color-accent)')
        .attr('stroke-width', 2);
    },
    [samples, lineGen],
  );

  // ═══════════════════════════════════════════
  // LEFT PANEL — Chord inequality
  // ═══════════════════════════════════════════
  useEffect(() => {
    if (!chordRef.current || innerW <= 0 || innerH <= 0) return;

    const svg = d3.select(chordRef.current);
    svg.selectAll('*').remove();
    svg
      .attr('width', panelWidth)
      .attr('height', panelHeight);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    drawAxes(g, 'Chord Inequality');

    // Shaded region between chord and function
    const aX = Math.min(chordA, chordB);
    const bX = Math.max(chordA, chordB);
    const aY = fn(chordA);
    const bY = fn(chordB);

    // chord value at x: linear interpolation between (chordA, aY) and (chordB, bY)
    const chordVal = (x: number): number => {
      if (chordB === chordA) return aY;
      const t = (x - chordA) / (chordB - chordA);
      return aY + t * (bY - aY);
    };

    // Filter samples in the chord interval
    const intervalSamples = samples.filter((d) => d.x >= aX && d.x <= bX);

    // Green region: chord >= function (convex gap)
    const convexArea = d3
      .area<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y0((d) => yScale(d.y))
      .y1((d) => yScale(chordVal(d.x)))
      .defined((d) => chordVal(d.x) >= d.y);

    g.append('path')
      .datum(intervalSamples)
      .attr('d', convexArea)
      .attr('fill', 'rgba(34, 197, 94, 0.25)')
      .attr('stroke', 'none');

    // Red region: function > chord (violation)
    const violationArea = d3
      .area<{ x: number; y: number }>()
      .x((d) => xScale(d.x))
      .y0((d) => yScale(chordVal(d.x)))
      .y1((d) => yScale(d.y))
      .defined((d) => d.y > chordVal(d.x));

    g.append('path')
      .datum(intervalSamples)
      .attr('d', violationArea)
      .attr('fill', 'rgba(239, 68, 68, 0.25)')
      .attr('stroke', 'none');

    // Function curve
    drawCurve(g);

    // Chord line (secant)
    g.append('line')
      .attr('x1', xScale(chordA))
      .attr('y1', yScale(aY))
      .attr('x2', xScale(chordB))
      .attr('y2', yScale(bY))
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,3');

    // Theta interpolation
    const thetaX = chordA + theta * (chordB - chordA);
    const thetaChordY = chordVal(thetaX);
    const thetaFnY = fn(thetaX);

    // Vertical line connecting chord point and function point
    g.append('line')
      .attr('x1', xScale(thetaX))
      .attr('y1', yScale(thetaChordY))
      .attr('x2', xScale(thetaX))
      .attr('y2', yScale(thetaFnY))
      .attr('stroke', 'var(--color-muted)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,2');

    // Dot on chord
    g.append('circle')
      .attr('cx', xScale(thetaX))
      .attr('cy', yScale(thetaChordY))
      .attr('r', 5)
      .attr('fill', '#f59e0b');

    // Dot on function
    g.append('circle')
      .attr('cx', xScale(thetaX))
      .attr('cy', yScale(thetaFnY))
      .attr('r', 5)
      .attr('fill', 'var(--color-accent)');

    // Draggable endpoint A
    const circleA = g
      .append('circle')
      .attr('cx', xScale(chordA))
      .attr('cy', yScale(aY))
      .attr('r', 7)
      .attr('fill', '#ef4444')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .style('cursor', 'ew-resize');

    circleA.call(
      d3
        .drag<SVGCircleElement, unknown>()
        .on('drag', (event) => {
          const svgRect = chordRef.current!.getBoundingClientRect();
          const mouseX = event.sourceEvent.clientX - svgRect.left - MARGIN.left;
          const newX = clamp(xScale.invert(mouseX), X_DOMAIN[0], X_DOMAIN[1]);
          setChordA(newX);
        }),
    );

    // Draggable endpoint B
    const circleB = g
      .append('circle')
      .attr('cx', xScale(chordB))
      .attr('cy', yScale(bY))
      .attr('r', 7)
      .attr('fill', '#3b82f6')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .style('cursor', 'ew-resize');

    circleB.call(
      d3
        .drag<SVGCircleElement, unknown>()
        .on('drag', (event) => {
          const svgRect = chordRef.current!.getBoundingClientRect();
          const mouseX = event.sourceEvent.clientX - svgRect.left - MARGIN.left;
          const newX = clamp(xScale.invert(mouseX), X_DOMAIN[0], X_DOMAIN[1]);
          setChordB(newX);
        }),
    );
  }, [chordA, chordB, theta, fn, samples, innerW, innerH, panelWidth, panelHeight, xScale, yScale, lineGen, drawAxes, drawCurve]);

  // ═══════════════════════════════════════════
  // CENTER PANEL — Epigraph
  // ═══════════════════════════════════════════
  useEffect(() => {
    if (!epiRef.current || innerW <= 0 || innerH <= 0) return;

    const svg = d3.select(epiRef.current);
    svg.selectAll('*').remove();
    svg
      .attr('width', panelWidth)
      .attr('height', panelHeight);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    drawAxes(g, 'Epigraph');

    // Epigraph shading: region above the graph up to the top of the plot
    if (showEpigraph) {
      const epiArea = d3
        .area<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y0((d) => yScale(d.y))
        .y1(0); // top of inner plot area in pixel coords

      g.append('path')
        .datum(samples)
        .attr('d', epiArea)
        .attr('fill', 'rgba(59, 130, 246, 0.15)')
        .attr('stroke', 'none');

      // Label
      const midX = (X_DOMAIN[0] + X_DOMAIN[1]) / 2;
      const labelY = yScale(yExtent[1]) + 20;
      g.append('text')
        .attr('x', xScale(midX))
        .attr('y', labelY)
        .attr('text-anchor', 'middle')
        .style('fill', 'rgba(59, 130, 246, 0.7)')
        .style('font-size', '12px')
        .style('font-style', 'italic')
        .text(isConvex ? 'epi(f) — convex set' : 'epi(f) — not convex');
    }

    // Function curve
    drawCurve(g);
  }, [fn, samples, showEpigraph, isConvex, innerW, innerH, panelWidth, panelHeight, xScale, yScale, yExtent, lineGen, drawAxes, drawCurve]);

  // ═══════════════════════════════════════════
  // RIGHT PANEL — Tangent / Jensen
  // ═══════════════════════════════════════════
  useEffect(() => {
    if (!tangentRef.current || innerW <= 0 || innerH <= 0) return;

    const svg = d3.select(tangentRef.current);
    svg.selectAll('*').remove();
    svg
      .attr('width', panelWidth)
      .attr('height', panelHeight);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    drawAxes(g, jensenMode ? "Jensen's Inequality" : 'First-Order Condition');

    // Function curve
    drawCurve(g);

    if (!jensenMode) {
      // ── Tangent line ──
      const x0 = tangentX0;
      const y0 = fn(x0);
      const slope = numericalDerivative(fn, x0);

      // Draw tangent line across the full domain
      const tangentY = (x: number) => y0 + slope * (x - x0);
      const tLineData = [
        { x: X_DOMAIN[0], y: tangentY(X_DOMAIN[0]) },
        { x: X_DOMAIN[1], y: tangentY(X_DOMAIN[1]) },
      ];

      g.append('line')
        .attr('x1', xScale(tLineData[0].x))
        .attr('y1', yScale(tLineData[0].y))
        .attr('x2', xScale(tLineData[1].x))
        .attr('y2', yScale(tLineData[1].y))
        .attr('stroke', '#10b981')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '6,3');

      // Shade the gap: function - tangent for convex functions
      const gapArea = d3
        .area<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y0((d) => yScale(tangentY(d.x)))
        .y1((d) => yScale(d.y))
        .defined((d) => d.y >= tangentY(d.x) - 0.01);

      g.append('path')
        .datum(samples)
        .attr('d', gapArea)
        .attr('fill', 'rgba(16, 185, 129, 0.15)')
        .attr('stroke', 'none');

      // Where function falls below tangent (violation for non-convex)
      const violationGap = d3
        .area<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y0((d) => yScale(d.y))
        .y1((d) => yScale(tangentY(d.x)))
        .defined((d) => tangentY(d.x) > d.y + 0.01);

      g.append('path')
        .datum(samples)
        .attr('d', violationGap)
        .attr('fill', 'rgba(239, 68, 68, 0.15)')
        .attr('stroke', 'none');

      // Tangent label
      g.append('text')
        .attr('x', xScale(X_DOMAIN[1]) - 4)
        .attr('y', yScale(tangentY(X_DOMAIN[1])) - 6)
        .attr('text-anchor', 'end')
        .style('fill', '#10b981')
        .style('font-size', '11px')
        .text('tangent');

      // Draggable point on curve
      const pointCircle = g
        .append('circle')
        .attr('cx', xScale(x0))
        .attr('cy', yScale(y0))
        .attr('r', 7)
        .attr('fill', '#8b5cf6')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .style('cursor', 'ew-resize');

      pointCircle.call(
        d3
          .drag<SVGCircleElement, unknown>()
          .on('drag', (event) => {
            const svgRect = tangentRef.current!.getBoundingClientRect();
            const mouseX = event.sourceEvent.clientX - svgRect.left - MARGIN.left;
            const newX = clamp(xScale.invert(mouseX), X_DOMAIN[0] + 0.1, X_DOMAIN[1] - 0.1);
            setTangentX0(newX);
          }),
      );

      // x₀ label
      g.append('text')
        .attr('x', xScale(x0))
        .attr('y', yScale(y0) + 18)
        .attr('text-anchor', 'middle')
        .style('fill', '#8b5cf6')
        .style('font-size', '11px')
        .text('x\u2080');
    } else {
      // ── Jensen mode ──
      const sampleXs = [-2, -0.7, 0.7, 2];
      const sampleYs = sampleXs.map(fn);
      const meanX = sampleXs.reduce((s, v) => s + v, 0) / sampleXs.length;
      const meanY = sampleYs.reduce((s, v) => s + v, 0) / sampleYs.length; // E[f(X)]
      const fMeanX = fn(meanX); // f(E[X])

      // Sample points
      sampleXs.forEach((sx, i) => {
        const sy = sampleYs[i];
        // Vertical line from x-axis to function
        g.append('line')
          .attr('x1', xScale(sx))
          .attr('y1', yScale(0))
          .attr('x2', xScale(sx))
          .attr('y2', yScale(sy))
          .attr('stroke', 'var(--color-muted)')
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '2,2');

        g.append('circle')
          .attr('cx', xScale(sx))
          .attr('cy', yScale(sy))
          .attr('r', 5)
          .attr('fill', d3.schemeTableau10[i % 10]);

        g.append('text')
          .attr('x', xScale(sx))
          .attr('y', yScale(sy) - 8)
          .attr('text-anchor', 'middle')
          .style('fill', d3.schemeTableau10[i % 10])
          .style('font-size', '10px')
          .text(`x${i + 1}`);
      });

      // E[X] vertical line
      g.append('line')
        .attr('x1', xScale(meanX))
        .attr('y1', yScale(fMeanX))
        .attr('x2', xScale(meanX))
        .attr('y2', yScale(meanY))
        .attr('stroke', '#f59e0b')
        .attr('stroke-width', 2);

      // f(E[X]) dot
      g.append('circle')
        .attr('cx', xScale(meanX))
        .attr('cy', yScale(fMeanX))
        .attr('r', 6)
        .attr('fill', '#ef4444')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);

      g.append('text')
        .attr('x', xScale(meanX) + 10)
        .attr('y', yScale(fMeanX) + 4)
        .style('fill', '#ef4444')
        .style('font-size', '11px')
        .text('f(E[X])');

      // E[f(X)] dot
      g.append('circle')
        .attr('cx', xScale(meanX))
        .attr('cy', yScale(meanY))
        .attr('r', 6)
        .attr('fill', '#3b82f6')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);

      g.append('text')
        .attr('x', xScale(meanX) + 10)
        .attr('y', yScale(meanY) + 4)
        .style('fill', '#3b82f6')
        .style('font-size', '11px')
        .text('E[f(X)]');

      // Gap annotation
      const gapVal = meanY - fMeanX;
      const gapLabel = isConvex
        ? `gap = ${gapVal.toFixed(2)} \u2265 0`
        : `gap = ${gapVal.toFixed(2)}`;

      g.append('text')
        .attr('x', xScale(meanX) + 10)
        .attr('y', yScale((fMeanX + meanY) / 2) + 4)
        .style('fill', 'var(--color-text)')
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(gapLabel);
    }
  }, [fn, isConvex, samples, tangentX0, jensenMode, innerW, innerH, panelWidth, panelHeight, xScale, yScale, lineGen, drawAxes, drawCurve]);

  // ─── Guard: no width yet ───
  if (!containerWidth) return <div ref={containerRef} style={{ minHeight: 400 }} />;

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {/* Panels */}
      <div
        style={{
          display: 'flex',
          flexDirection: isWide ? 'row' : 'column',
          gap: isWide ? 8 : 12,
          alignItems: 'flex-start',
          justifyContent: 'center',
        }}
      >
        <svg role="img" aria-label="Convex function explorer visualization (panel 1 of 3)" ref={chordRef} />
        <svg role="img" aria-label="Convex function explorer visualization (panel 2 of 3)" ref={epiRef} />
        <svg role="img" aria-label="Convex function explorer visualization (panel 3 of 3)" ref={tangentRef} />
      </div>

      {/* Controls */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 16,
          alignItems: 'center',
          marginTop: 12,
          padding: '8px 0',
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
              setChordA(-2);
              setChordB(2);
              setTangentX0(0);
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

        {/* Theta slider */}
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 13,
            color: 'var(--color-text)',
          }}
        >
          {'\u03B8'}: {theta.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={theta}
            onChange={(e) => setTheta(Number(e.target.value))}
            style={{ width: 100 }}
          />
        </label>

        {/* Epigraph toggle */}
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
            checked={showEpigraph}
            onChange={(e) => setShowEpigraph(e.target.checked)}
          />
          Show epigraph
        </label>

        {/* Jensen mode toggle */}
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
            checked={jensenMode}
            onChange={(e) => setJensenMode(e.target.checked)}
          />
          Jensen mode
        </label>
      </div>

      {/* Convexity indicator */}
      <div
        style={{
          fontSize: 12,
          color: 'var(--color-muted)',
          marginTop: 4,
        }}
      >
        {activeFn.label} is{' '}
        <span
          style={{
            fontWeight: 600,
            color: isConvex ? '#22c55e' : '#ef4444',
          }}
        >
          {isConvex ? 'convex' : 'non-convex'}
        </span>
      </div>
    </div>
  );
}

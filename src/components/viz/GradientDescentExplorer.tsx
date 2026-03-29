import { useState, useEffect, useRef, useMemo, useId, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { makeQuadratic, runGDTrajectory } from './shared/gradientDescentUtils';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 30, left: 45 };
const PANEL_HEIGHT = 320;
const GRID_RES = 100;
const DOMAIN: [number, number] = [-4, 4];

// ─── Colors ───

const COLOR_TRAJECTORY = '#0F6E56';
const COLOR_GRADIENT = '#534AB7';
const COLOR_START = '#D97706';
const COLOR_OPTIMUM = '#ef4444';

// ─── Component ───

export default function GradientDescentExplorer() {
  const instanceId = useId();
  const clipId = `gd-clip-${instanceId.replace(/:/g, '')}`;
  const arrowId = `gd-arrow-${instanceId.replace(/:/g, '')}`;

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───
  const [kappa, setKappa] = useState(5);
  const [stepSizeMode, setStepSizeMode] = useState<string>('1/L');
  const [x0, setX0] = useState<[number, number]>([2.5, 2.5]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // ─── Derived values ───
  const eta = useMemo(() => {
    const problem = makeQuadratic(kappa);
    switch (stepSizeMode) {
      case '1/L': return 1 / problem.L;
      case '2/(L+mu)': return 2 / (problem.L + problem.mu);
      case 'too-large': return 2 / problem.L + 0.1;
      default: return 1 / problem.L;
    }
  }, [kappa, stepSizeMode]);

  const trajectory = useMemo(() => {
    const problem = makeQuadratic(kappa);
    const maxIter = stepSizeMode === 'too-large' ? 50 : 200;
    return runGDTrajectory(problem, x0, eta, maxIter);
  }, [kappa, x0, eta, stepSizeMode]);

  // Reset animation when trajectory changes
  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, [trajectory]);

  // ─── Animation ───
  const animRef = useRef<number>(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!isPlaying) return;
    const advance = () => {
      setCurrentStep(prev => {
        if (prev >= trajectory.path.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
      timeoutRef.current = setTimeout(() => {
        animRef.current = requestAnimationFrame(advance);
      }, 100);
    };
    animRef.current = requestAnimationFrame(advance);
    return () => {
      cancelAnimationFrame(animRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [isPlaying, trajectory.path.length]);

  // ─── Panel dimensions ───
  const isWide = containerWidth >= SM_BREAKPOINT;
  const panelWidth = useMemo(() => {
    if (!containerWidth) return 0;
    return isWide ? Math.floor((containerWidth - 16) / 2) : containerWidth;
  }, [containerWidth, isWide]);

  // ─── Contour data ───
  const contourData = useMemo(() => {
    const problem = makeQuadratic(kappa);
    const step = (DOMAIN[1] - DOMAIN[0]) / (GRID_RES - 1);
    const values: number[] = [];
    for (let j = 0; j < GRID_RES; j++) {
      for (let i = 0; i < GRID_RES; i++) {
        const px = DOMAIN[0] + i * step;
        const py = DOMAIN[0] + j * step;
        values.push(problem.f([px, py]));
      }
    }
    return values;
  }, [kappa]);

  // ─── Left panel: contour + trajectory ───
  useEffect(() => {
    const svg = d3.select(leftSvgRef.current);
    if (!panelWidth || !svg.node()) return;
    svg.selectAll('*').remove();

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain(DOMAIN).range([0, w]);
    const yScale = d3.scaleLinear().domain(DOMAIN).range([h, 0]);

    // Defs: clipPath + arrowhead
    const defs = g.append('defs');
    defs.append('clipPath')
      .attr('id', clipId)
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', w)
      .attr('height', h);

    defs.append('marker')
      .attr('id', arrowId)
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 8)
      .attr('refY', 5)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 Z')
      .attr('fill', COLOR_GRADIENT);

    const plotArea = g.append('g').attr('clip-path', `url(#${clipId})`);

    // Contours
    const contours = d3.contours()
      .size([GRID_RES, GRID_RES])
      .thresholds(20);

    const contourGeo = contours(contourData);

    const contourXScale = d3.scaleLinear().domain([0, GRID_RES]).range(DOMAIN);
    const contourYScale = d3.scaleLinear().domain([0, GRID_RES]).range(DOMAIN);

    const transformPoint = (coords: number[]): [number, number] => {
      return [xScale(contourXScale(coords[0])), yScale(contourYScale(coords[1]))];
    };

    const pathGen = d3.geoPath().projection(
      d3.geoTransform({
        point: function (x: number, y: number) {
          const [px, py] = transformPoint([x, y]);
          this.stream.point(px, py);
        },
      })
    );

    plotArea.selectAll('path.contour')
      .data(contourGeo)
      .join('path')
      .attr('class', 'contour')
      .attr('d', (d) => pathGen(d) ?? '')
      .attr('fill', 'none')
      .attr('stroke', '#64748b')
      .attr('stroke-opacity', 0.3)
      .attr('stroke-width', 1);

    // Optimum crosshair
    const ox = xScale(0);
    const oy = yScale(0);
    plotArea.append('line')
      .attr('x1', ox - 6).attr('y1', oy)
      .attr('x2', ox + 6).attr('y2', oy)
      .attr('stroke', COLOR_OPTIMUM).attr('stroke-width', 2);
    plotArea.append('line')
      .attr('x1', ox).attr('y1', oy - 6)
      .attr('x2', ox).attr('y2', oy + 6)
      .attr('stroke', COLOR_OPTIMUM).attr('stroke-width', 2);

    // Trajectory polyline up to currentStep
    const visiblePath = trajectory.path.slice(0, currentStep + 1);
    if (visiblePath.length > 1) {
      const lineGen = d3.line<number[]>()
        .x(d => xScale(d[0]))
        .y(d => yScale(d[1]));

      plotArea.append('path')
        .datum(visiblePath)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', COLOR_TRAJECTORY)
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.8);
    }

    // Small circles at each iterate
    plotArea.selectAll('circle.iterate')
      .data(visiblePath)
      .join('circle')
      .attr('class', 'iterate')
      .attr('cx', d => xScale(d[0]))
      .attr('cy', d => yScale(d[1]))
      .attr('r', 2.5)
      .attr('fill', COLOR_TRAJECTORY)
      .attr('fill-opacity', 0.6);

    // Gradient arrow at currentStep
    if (currentStep < trajectory.path.length) {
      const pt = trajectory.path[currentStep];
      const problem = makeQuadratic(kappa);
      const grad = problem.grad(pt);
      const gradNorm = Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

      if (gradNorm > 1e-8) {
        // Scale arrow for visibility: fixed length in data coordinates
        const arrowLen = 0.8;
        const dx = (-grad[0] / gradNorm) * arrowLen;
        const dy = (-grad[1] / gradNorm) * arrowLen;

        plotArea.append('line')
          .attr('x1', xScale(pt[0]))
          .attr('y1', yScale(pt[1]))
          .attr('x2', xScale(pt[0] + dx))
          .attr('y2', yScale(pt[1] + dy))
          .attr('stroke', COLOR_GRADIENT)
          .attr('stroke-width', 2.5)
          .attr('marker-end', `url(#${arrowId})`);
      }
    }

    // Draggable start point
    const startCircle = plotArea.append('circle')
      .attr('cx', xScale(x0[0]))
      .attr('cy', yScale(x0[1]))
      .attr('r', 8)
      .attr('fill', COLOR_START)
      .attr('fill-opacity', 0.85)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'grab');

    const dragBehavior = d3.drag<SVGCircleElement, unknown>()
      .on('start', function () {
        d3.select(this).style('cursor', 'grabbing');
      })
      .on('drag', (event) => {
        const newX = Math.max(-3.5, Math.min(3.5, xScale.invert(event.x)));
        const newY = Math.max(-3.5, Math.min(3.5, yScale.invert(event.y)));
        setX0([newX, newY]);
      })
      .on('end', function () {
        d3.select(this).style('cursor', 'grab');
      });

    startCircle.call(dragBehavior);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('font-size', '10px');

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .style('fill', 'currentColor')
      .text('Contour plot + GD trajectory');
  }, [panelWidth, contourData, trajectory, currentStep, kappa, x0, clipId, arrowId]);

  // ─── Right panel: convergence ───
  useEffect(() => {
    const svg = d3.select(rightSvgRef.current);
    if (!panelWidth || !svg.node()) return;
    svg.selectAll('*').remove();

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const visibleObj = trajectory.objectives.slice(0, currentStep + 1);
    const totalSteps = trajectory.objectives.length;

    // X scale: iteration index
    const xScale = d3.scaleLinear()
      .domain([0, Math.max(totalSteps - 1, 1)])
      .range([0, w]);

    // Y scale: log of objective
    const f0 = trajectory.objectives[0] || 1;
    const fMin = Math.max(1e-16, d3.min(trajectory.objectives) ?? 1e-16);
    const yScale = d3.scaleLog()
      .domain([fMin * 0.5, f0 * 2])
      .range([h, 0])
      .clamp(true);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('d')))
      .selectAll('text')
      .style('font-size', '10px');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5, '.0e'))
      .selectAll('text')
      .style('font-size', '10px');

    // Axis labels
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('Iteration k');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2)
      .attr('y', -38)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('f(xₖ)');

    // Reference envelopes (draw for full range)
    const envData = d3.range(0, totalSteps).map(k => k);

    // O(1/k) envelope
    const oOneOverK = envData.map(k => ({
      k,
      v: Math.max(1e-16, f0 * 2 / (k + 1)),
    }));

    const envLine = d3.line<{ k: number; v: number }>()
      .x(d => xScale(d.k))
      .y(d => yScale(d.v));

    g.append('path')
      .datum(oOneOverK)
      .attr('d', envLine)
      .attr('fill', 'none')
      .attr('stroke', COLOR_TRAJECTORY)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,4')
      .attr('stroke-opacity', 0.6);

    // Linear rate envelope
    const linearRate = envData.map(k => ({
      k,
      v: Math.max(1e-16, f0 * Math.pow(1 - 1 / kappa, k)),
    }));

    g.append('path')
      .datum(linearRate)
      .attr('d', envLine)
      .attr('fill', 'none')
      .attr('stroke', COLOR_GRADIENT)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,4')
      .attr('stroke-opacity', 0.6);

    // Objective curve up to currentStep
    if (visibleObj.length > 1) {
      const objLine = d3.line<number>()
        .x((_, i) => xScale(i))
        .y(d => yScale(Math.max(1e-16, d)));

      g.append('path')
        .datum(visibleObj)
        .attr('d', objLine)
        .attr('fill', 'none')
        .attr('stroke', COLOR_TRAJECTORY)
        .attr('stroke-width', 2);
    }

    // Dots on objective
    g.selectAll('circle.obj')
      .data(visibleObj)
      .join('circle')
      .attr('class', 'obj')
      .attr('cx', (_, i) => xScale(i))
      .attr('cy', d => yScale(Math.max(1e-16, d)))
      .attr('r', 2)
      .attr('fill', COLOR_TRAJECTORY);

    // Vertical sync line at currentStep
    g.append('line')
      .attr('x1', xScale(currentStep))
      .attr('y1', 0)
      .attr('x2', xScale(currentStep))
      .attr('y2', h)
      .attr('stroke', '#94a3b8')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3');

    // Legend
    const legend = g.append('g').attr('transform', `translate(${w - 130}, 5)`);

    legend.append('line')
      .attr('x1', 0).attr('y1', 0).attr('x2', 20).attr('y2', 0)
      .attr('stroke', COLOR_TRAJECTORY).attr('stroke-dasharray', '5,4').attr('stroke-width', 1.5);
    legend.append('text')
      .attr('x', 24).attr('y', 4)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('O(1/k)');

    legend.append('line')
      .attr('x1', 0).attr('y1', 16).attr('x2', 20).attr('y2', 16)
      .attr('stroke', COLOR_GRADIENT).attr('stroke-dasharray', '5,4').attr('stroke-width', 1.5);
    legend.append('text')
      .attr('x', 24).attr('y', 20)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('(1 − 1/κ)ᵏ');

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .style('fill', 'currentColor')
      .text('Convergence (log scale)');
  }, [panelWidth, trajectory, currentStep, kappa]);

  // ─── Handlers ───
  const handlePlay = useCallback(() => {
    if (currentStep >= trajectory.path.length - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(prev => !prev);
  }, [currentStep, trajectory.path.length]);

  const handleReset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  // ─── Render ───
  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: isWide ? 'row' : 'column',
          gap: isWide ? 16 : 8,
        }}
      >
        <svg role="img" aria-label="Gradient descent explorer visualization (panel 1 of 2)"
          ref={leftSvgRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          style={{ overflow: 'visible' }}
        />
        <svg role="img" aria-label="Gradient descent explorer visualization (panel 2 of 2)"
          ref={rightSvgRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          style={{ overflow: 'visible' }}
        />
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
        {/* Kappa slider */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 14 }}>
          <span style={{ fontWeight: 500 }}>
            {'κ = '}
            <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: 24, display: 'inline-block' }}>
              {kappa}
            </span>
          </span>
          <input
            type="range"
            min={1}
            max={100}
            step={1}
            value={kappa}
            onChange={e => setKappa(Number(e.target.value))}
            style={{ width: 120 }}
          />
        </label>

        {/* Step size dropdown */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 14 }}>
          <span style={{ fontWeight: 500 }}>Step size:</span>
          <select
            value={stepSizeMode}
            onChange={e => setStepSizeMode(e.target.value)}
            style={{
              padding: '2px 6px',
              borderRadius: 4,
              border: '1px solid #cbd5e1',
              fontSize: 14,
              background: 'inherit',
              color: 'inherit',
            }}
          >
            <option value="1/L">1/L</option>
            <option value="2/(L+mu)">{'2/(L+\u03BC)'}</option>
            <option value="too-large">Too large</option>
          </select>
        </label>

        {/* Play/Pause */}
        <button
          onClick={handlePlay}
          style={{
            padding: '4px 14px',
            borderRadius: 6,
            border: '1px solid #cbd5e1',
            background: isPlaying ? '#fef2f2' : '#f0fdf4',
            color: isPlaying ? '#b91c1c' : '#15803d',
            fontWeight: 600,
            fontSize: 14,
            cursor: 'pointer',
          }}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        {/* Reset */}
        <button
          onClick={handleReset}
          style={{
            padding: '4px 14px',
            borderRadius: 6,
            border: '1px solid #cbd5e1',
            background: 'inherit',
            color: 'inherit',
            fontSize: 14,
            cursor: 'pointer',
          }}
        >
          Reset
        </button>

        {/* Step info */}
        <span style={{ fontSize: 13, color: '#64748b' }}>
          Step {currentStep} / {trajectory.path.length - 1}
          {' | '}
          f(x) = {trajectory.objectives[currentStep]?.toExponential(2) ?? '—'}
          {' | '}
          {'\u03B7'} = {eta.toFixed(4)}
        </span>
      </div>
    </div>
  );
}

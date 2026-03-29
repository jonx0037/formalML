import { useState, useEffect, useRef, useMemo, useId, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { makeQuadratic, runGDTrajectory, nesterovStep } from './shared/gradientDescentUtils';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 30, left: 45 };
const PANEL_HEIGHT = 320;
const GRID_RES = 100;
const DOMAIN: [number, number] = [-4, 4];
const START_POINT: [number, number] = [3, 3];

// ─── Colors ───

const COLOR_GD = '#0F6E56';
const COLOR_NESTEROV = '#534AB7';
const COLOR_MOMENTUM = '#D97706';
const COLOR_OPTIMUM = '#ef4444';

// ─── Component ───

export default function AccelerationExplorer() {
  const instanceId = useId();
  const clipId = `accel-clip-${instanceId.replace(/:/g, '')}`;
  const arrowMomentumId = `accel-arrow-mom-${instanceId.replace(/:/g, '')}`;
  const arrowGradId = `accel-arrow-grad-${instanceId.replace(/:/g, '')}`;

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───
  const [kappa, setKappa] = useState(20);
  const [showMomentum, setShowMomentum] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // ─── Vanilla GD trajectory ───
  const gdTrajectory = useMemo(() => {
    const problem = makeQuadratic(kappa);
    return runGDTrajectory(problem, START_POINT, 1 / problem.L, 150);
  }, [kappa]);

  // ─── Nesterov trajectory ───
  const nesterovTrajectory = useMemo(() => {
    const problem = makeQuadratic(kappa);
    const eta = 1 / problem.L;
    let x = [...START_POINT];
    let xPrev = [...START_POINT];
    let t = 1;
    const xPath: number[][] = [[...START_POINT]];
    const yPath: number[][] = [];
    const objectives: number[] = [problem.f(START_POINT)];
    for (let k = 0; k < 150; k++) {
      const { xNew, yNew, tNew } = nesterovStep(problem.grad, x, xPrev, t, eta);
      yPath.push(yNew);
      xPrev = x;
      x = xNew;
      t = tNew;
      xPath.push(x.slice());
      objectives.push(problem.f(x));
      if (Math.abs(x[0]) > 200 || Math.abs(x[1]) > 200) break;
      if (objectives[objectives.length - 1] < 1e-20) break;
    }
    return { xPath, yPath, objectives };
  }, [kappa]);

  const maxSteps = Math.max(gdTrajectory.path.length, nesterovTrajectory.xPath.length) - 1;

  // Reset animation when kappa changes
  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, [kappa]);

  // ─── Animation ───
  const animRef = useRef<number>(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!isPlaying) return;
    const advance = () => {
      setCurrentStep(prev => {
        if (prev >= maxSteps) {
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
  }, [isPlaying, maxSteps]);

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

  // ─── Left panel: contour + trajectories ───
  useEffect(() => {
    const svg = d3.select(leftSvgRef.current);
    if (!panelWidth || !svg.node()) return;
    svg.selectAll('*').remove();

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain(DOMAIN).range([0, w]);
    const yScale = d3.scaleLinear().domain(DOMAIN).range([h, 0]);

    // Defs: clipPath + arrowheads
    const defs = g.append('defs');
    defs.append('clipPath')
      .attr('id', clipId)
      .append('rect')
      .attr('x', 0).attr('y', 0).attr('width', w).attr('height', h);

    defs.append('marker')
      .attr('id', arrowMomentumId)
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 8).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 Z')
      .attr('fill', COLOR_MOMENTUM);

    defs.append('marker')
      .attr('id', arrowGradId)
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 8).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 Z')
      .attr('fill', COLOR_NESTEROV);

    const plotArea = g.append('g').attr('clip-path', `url(#${clipId})`);

    // Contours
    const contours = d3.contours().size([GRID_RES, GRID_RES]).thresholds(20);
    const contourGeo = contours(contourData);
    const contourXScale = d3.scaleLinear().domain([0, GRID_RES]).range(DOMAIN);
    const contourYScale = d3.scaleLinear().domain([0, GRID_RES]).range(DOMAIN);

    const pathGen = d3.geoPath().projection(
      d3.geoTransform({
        point: function (x: number, y: number) {
          this.stream.point(xScale(contourXScale(x)), yScale(contourYScale(y)));
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

    // Optimum star
    const ox = xScale(0);
    const oy = yScale(0);
    const starSize = 60;
    const starGen = d3.symbol().type(d3.symbolStar).size(starSize);
    plotArea.append('path')
      .attr('d', starGen() as string)
      .attr('transform', `translate(${ox},${oy})`)
      .attr('fill', COLOR_OPTIMUM)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);

    // Line generator
    const lineGen = d3.line<number[]>()
      .x(d => xScale(d[0]))
      .y(d => yScale(d[1]));

    // Vanilla GD trajectory
    const visibleGD = gdTrajectory.path.slice(0, currentStep + 1);
    if (visibleGD.length > 1) {
      plotArea.append('path')
        .datum(visibleGD)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', COLOR_GD)
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.8);
    }
    plotArea.selectAll('circle.gd-iter')
      .data(visibleGD)
      .join('circle')
      .attr('class', 'gd-iter')
      .attr('cx', d => xScale(d[0]))
      .attr('cy', d => yScale(d[1]))
      .attr('r', 2.5)
      .attr('fill', COLOR_GD)
      .attr('fill-opacity', 0.6);

    // Nesterov trajectory
    const visibleNest = nesterovTrajectory.xPath.slice(0, currentStep + 1);
    if (visibleNest.length > 1) {
      plotArea.append('path')
        .datum(visibleNest)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', COLOR_NESTEROV)
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.8);
    }
    plotArea.selectAll('circle.nest-iter')
      .data(visibleNest)
      .join('circle')
      .attr('class', 'nest-iter')
      .attr('cx', d => xScale(d[0]))
      .attr('cy', d => yScale(d[1]))
      .attr('r', 2.5)
      .attr('fill', COLOR_NESTEROV)
      .attr('fill-opacity', 0.6);

    // Momentum arrows
    if (showMomentum && currentStep > 0) {
      const startIdx = Math.max(0, currentStep - 3);
      for (let k = startIdx; k < currentStep && k < nesterovTrajectory.yPath.length; k++) {
        const xk = nesterovTrajectory.xPath[k];
        const yk = nesterovTrajectory.yPath[k];
        const xk1 = nesterovTrajectory.xPath[k + 1];

        // Extrapolation arrow: x_k → y_k (amber dashed)
        plotArea.append('line')
          .attr('x1', xScale(xk[0])).attr('y1', yScale(xk[1]))
          .attr('x2', xScale(yk[0])).attr('y2', yScale(yk[1]))
          .attr('stroke', COLOR_MOMENTUM)
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '4,3')
          .attr('marker-end', `url(#${arrowMomentumId})`);

        // Gradient step arrow: y_k → x_{k+1} (purple solid)
        plotArea.append('line')
          .attr('x1', xScale(yk[0])).attr('y1', yScale(yk[1]))
          .attr('x2', xScale(xk1[0])).attr('y2', yScale(xk1[1]))
          .attr('stroke', COLOR_NESTEROV)
          .attr('stroke-width', 1.5)
          .attr('marker-end', `url(#${arrowGradId})`);
      }
    }

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
      .text('GD vs Nesterov — trajectories');
  }, [panelWidth, contourData, gdTrajectory, nesterovTrajectory, currentStep, showMomentum, clipId, arrowMomentumId, arrowGradId]);

  // ─── Right panel: convergence ───
  useEffect(() => {
    const svg = d3.select(rightSvgRef.current);
    if (!panelWidth || !svg.node()) return;
    svg.selectAll('*').remove();

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const totalSteps = Math.max(gdTrajectory.objectives.length, nesterovTrajectory.objectives.length);
    const visibleGDObj = gdTrajectory.objectives.slice(0, currentStep + 1);
    const visibleNestObj = nesterovTrajectory.objectives.slice(0, currentStep + 1);

    // X scale
    const xScale = d3.scaleLinear()
      .domain([0, Math.max(totalSteps - 1, 1)])
      .range([0, w]);

    // Y scale (log)
    const f0 = gdTrajectory.objectives[0] || 1;
    const allObj = [...gdTrajectory.objectives, ...nesterovTrajectory.objectives];
    const fMin = Math.max(1e-16, d3.min(allObj) ?? 1e-16);
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
      .attr('x', w / 2).attr('y', h + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('Iteration k');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2).attr('y', -38)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('f(x\u2096)');

    // Reference envelopes
    const envData = d3.range(0, totalSteps);
    const envLine = d3.line<{ k: number; v: number }>()
      .x(d => xScale(d.k))
      .y(d => yScale(d.v));

    // O(1/k) envelope — teal dashed
    const oOneOverK = envData.map(k => ({
      k,
      v: Math.max(1e-16, f0 * 2 / (k + 1)),
    }));
    g.append('path')
      .datum(oOneOverK)
      .attr('d', envLine)
      .attr('fill', 'none')
      .attr('stroke', COLOR_GD)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,4')
      .attr('stroke-opacity', 0.6);

    // O(1/k^2) envelope — purple dashed
    const oOneOverK2 = envData.map(k => ({
      k,
      v: Math.max(1e-16, f0 * 2 / ((k + 1) * (k + 1))),
    }));
    g.append('path')
      .datum(oOneOverK2)
      .attr('d', envLine)
      .attr('fill', 'none')
      .attr('stroke', COLOR_NESTEROV)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,4')
      .attr('stroke-opacity', 0.6);

    // GD objective curve
    const objLine = d3.line<number>()
      .x((_, i) => xScale(i))
      .y(d => yScale(Math.max(1e-16, d)));

    if (visibleGDObj.length > 1) {
      g.append('path')
        .datum(visibleGDObj)
        .attr('d', objLine)
        .attr('fill', 'none')
        .attr('stroke', COLOR_GD)
        .attr('stroke-width', 2);
    }

    // Nesterov objective curve
    if (visibleNestObj.length > 1) {
      g.append('path')
        .datum(visibleNestObj)
        .attr('d', objLine)
        .attr('fill', 'none')
        .attr('stroke', COLOR_NESTEROV)
        .attr('stroke-width', 2);
    }

    // Vertical sync line
    g.append('line')
      .attr('x1', xScale(currentStep)).attr('y1', 0)
      .attr('x2', xScale(currentStep)).attr('y2', h)
      .attr('stroke', '#94a3b8')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3');

    // Legend
    const legend = g.append('g').attr('transform', `translate(${w - 140}, 5)`);

    legend.append('line')
      .attr('x1', 0).attr('y1', 0).attr('x2', 20).attr('y2', 0)
      .attr('stroke', COLOR_GD).attr('stroke-width', 2);
    legend.append('text')
      .attr('x', 24).attr('y', 4)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('Vanilla GD');

    legend.append('line')
      .attr('x1', 0).attr('y1', 16).attr('x2', 20).attr('y2', 16)
      .attr('stroke', COLOR_NESTEROV).attr('stroke-width', 2);
    legend.append('text')
      .attr('x', 24).attr('y', 20)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('Nesterov AGD');

    legend.append('line')
      .attr('x1', 0).attr('y1', 32).attr('x2', 20).attr('y2', 32)
      .attr('stroke', COLOR_GD).attr('stroke-dasharray', '5,4').attr('stroke-width', 1.5);
    legend.append('text')
      .attr('x', 24).attr('y', 36)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('O(1/k)');

    legend.append('line')
      .attr('x1', 0).attr('y1', 48).attr('x2', 20).attr('y2', 48)
      .attr('stroke', COLOR_NESTEROV).attr('stroke-dasharray', '5,4').attr('stroke-width', 1.5);
    legend.append('text')
      .attr('x', 24).attr('y', 52)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('O(1/k\u00B2)');

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .style('fill', 'currentColor')
      .text('Convergence (log scale)');
  }, [panelWidth, gdTrajectory, nesterovTrajectory, currentStep]);

  // ─── Handlers ───
  const handlePlay = useCallback(() => {
    if (currentStep >= maxSteps) {
      setCurrentStep(0);
    }
    setIsPlaying(prev => !prev);
  }, [currentStep, maxSteps]);

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
        <svg role="img" aria-label="Acceleration explorer visualization (panel 1 of 2)"
          ref={leftSvgRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          style={{ overflow: 'visible' }}
        />
        <svg role="img" aria-label="Acceleration explorer visualization (panel 2 of 2)"
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
            {'\u03BA = '}
            <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: 24, display: 'inline-block' }}>
              {kappa}
            </span>
          </span>
          <input
            type="range"
            min={2}
            max={100}
            step={1}
            value={kappa}
            onChange={e => setKappa(Number(e.target.value))}
            style={{ width: 120 }}
          />
        </label>

        {/* Show momentum checkbox */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 14 }}>
          <input
            type="checkbox"
            checked={showMomentum}
            onChange={e => setShowMomentum(e.target.checked)}
          />
          <span style={{ fontWeight: 500 }}>Show momentum arrows</span>
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
          Step {currentStep} / {maxSteps}
        </span>
      </div>
    </div>
  );
}

import { useState, useMemo, useId, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  softThreshold,
  softThresholdScalar,
  generateLassoProblem,
  lassoGradient,
  lassoObjective,
  estimateLipschitz,
} from './shared/proximalUtils';

// ─── Constants ───
const SM_BREAKPOINT = 640;
const HEIGHT = 320;
const MARGIN = { top: 20, right: 15, bottom: 40, left: 45 };

const COLOR_TRAJ = '#0F6E56';
const COLOR_PROX = '#534AB7';
const COLOR_GRAD = '#2563EB';
const COLOR_X0 = '#D97706';
const COLOR_OPT = '#DC2626';
const COLOR_TRUTH = '#16A34A';
const COLOR_RECOVER = COLOR_TRAJ;

// ─── 2D problem data ───
const A2 = [
  [1.5, 0.3],
  [-0.2, 1.8],
  [0.7, -0.5],
  [-0.3, 1.1],
  [1.0, 0.6],
];
const bTrue2 = [1.2, -0.8];
// b = A * bTrue (matrix-vector product)
const b2 = A2.map((row) =>
  row.reduce((sum, aij, j) => sum + aij * bTrue2[j], 0),
);

// Lipschitz constant for the 2D problem (largest eigenvalue of AᵀA)
const L2 = estimateLipschitz(A2, A2.length, A2[0].length);

const GRID_SIZE = 80;
const X_RANGE: [number, number] = [-3, 3];
const Y_RANGE: [number, number] = [-3, 3];
const MAX_ITER_2D = 60;
const MAX_ITER_BAR = 100;

type Speed = 'slow' | 'medium' | 'fast';
const SPEED_DELAYS: Record<Speed, number> = { slow: 400, medium: 150, fast: 50 };

// ─── 2D proximal gradient solver ───
function runProxGrad2D(
  x0: number[],
  lambda: number,
  maxIter: number,
): { traj: number[][]; gradSteps: number[][]; objectives: number[] } {
  const eta = 1 / L2;
  const traj = [x0.slice()];
  const gradSteps: number[][] = [];
  const objectives: number[] = [lassoObjective(A2, x0, b2, lambda)];
  let x = x0.slice();

  for (let k = 0; k < maxIter; k++) {
    const grad = lassoGradient(A2, x, b2);
    const xHalf = [x[0] - eta * grad[0], x[1] - eta * grad[1]];
    gradSteps.push(xHalf.slice());
    x = [
      softThresholdScalar(xHalf[0], eta * lambda),
      softThresholdScalar(xHalf[1], eta * lambda),
    ];
    traj.push(x.slice());
    objectives.push(lassoObjective(A2, x, b2, lambda));
  }
  return { traj, gradSteps, objectives };
}

// ─── Component ───
export default function ForwardBackwardExplorer() {
  const uid = useId();
  const id = `fb-${uid.replace(/:/g, '')}`;
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  // Controls
  const [x0, setX0] = useState<[number, number]>([2.0, 2.5]);
  const [lambda, setLambda] = useState(0.3);
  const [showSteps, setShowSteps] = useState(true);
  const [speed, setSpeed] = useState<Speed>('medium');
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? Math.max(containerWidth, 280)
    : Math.max(Math.floor((containerWidth - 24) / 3), 240);
  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

  // ─── 2D contour data ───
  const contourData = useMemo(() => {
    const values = new Array(GRID_SIZE * GRID_SIZE);
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        const x1 = X_RANGE[0] + (j / (GRID_SIZE - 1)) * (X_RANGE[1] - X_RANGE[0]);
        const x2 = Y_RANGE[0] + (i / (GRID_SIZE - 1)) * (Y_RANGE[1] - Y_RANGE[0]);
        let val = 0;
        for (let r = 0; r < A2.length; r++) {
          const res = A2[r][0] * x1 + A2[r][1] * x2 - b2[r];
          val += res * res;
        }
        values[i * GRID_SIZE + j] = val * 0.5;
      }
    }
    return values;
  }, []);

  // ─── 2D trajectory ───
  const trajResult = useMemo(
    () => runProxGrad2D([x0[0], x0[1]], lambda, MAX_ITER_2D),
    [x0, lambda],
  );

  const optimalVal = useMemo(() => {
    const objs = trajResult.objectives;
    return objs[objs.length - 1];
  }, [trajResult]);

  // ─── Larger problem for bar chart ───
  const barProblem = useMemo(() => generateLassoProblem(30, 20, 4, 42), []);

  const barResult = useMemo(() => {
    const { A, b, L } = barProblem;
    const p = A[0].length;
    const eta = 1 / L;
    const x = new Array(p).fill(0);
    for (let k = 0; k < MAX_ITER_BAR; k++) {
      const grad = lassoGradient(A, x, b);
      const xHalf = x.map((xi: number, i: number) => xi - eta * grad[i]);
      const xNew = softThreshold(xHalf, eta * lambda);
      for (let i = 0; i < p; i++) x[i] = xNew[i];
    }
    return x;
  }, [barProblem, lambda]);

  // ─── Animation ───
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(true);
  }, [x0, lambda]);

  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (!isPlaying) return;

    timerRef.current = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= trajResult.traj.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, SPEED_DELAYS[speed]);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, speed, trajResult.traj.length]);

  // ─── Dragging x0 ───
  const isDragging = useRef(false);
  const contourSvgRef = useRef<SVGSVGElement>(null);

  const handleContourPointerDown = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      const svg = contourSvgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const xScale = d3.scaleLinear().domain(X_RANGE).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(Y_RANGE).range([innerH, 0]);
      const mx = e.clientX - rect.left - MARGIN.left;
      const my = e.clientY - rect.top - MARGIN.top;
      const dataX = xScale.invert(mx);
      const dataY = yScale.invert(my);

      // Only start drag if near x0
      const dist = Math.hypot(dataX - x0[0], dataY - x0[1]);
      if (dist < 0.5) {
        isDragging.current = true;
        svg.setPointerCapture(e.pointerId);
      }
    },
    [innerW, innerH, x0],
  );

  const handleContourPointerMove = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (!isDragging.current) return;
      const svg = contourSvgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const xScale = d3.scaleLinear().domain(X_RANGE).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(Y_RANGE).range([innerH, 0]);
      const mx = e.clientX - rect.left - MARGIN.left;
      const my = e.clientY - rect.top - MARGIN.top;
      const nx = Math.max(X_RANGE[0], Math.min(X_RANGE[1], xScale.invert(mx)));
      const ny = Math.max(Y_RANGE[0], Math.min(Y_RANGE[1], yScale.invert(my)));
      setX0([nx, ny]);
    },
    [innerW, innerH],
  );

  const handleContourPointerUp = useCallback(() => {
    isDragging.current = false;
  }, []);

  // ─── Scales ───
  const xScale = useMemo(
    () => d3.scaleLinear().domain(X_RANGE).range([0, innerW]),
    [innerW],
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain(Y_RANGE).range([innerH, 0]),
    [innerH],
  );

  // ─── Contour paths ───
  const contourPaths = useMemo(() => {
    const contourGen = d3
      .contours()
      .size([GRID_SIZE, GRID_SIZE])
      .thresholds(d3.range(0.5, 20, 1.5));

    const geos = contourGen(contourData);

    // Map contour coordinates from grid space to data space, then to pixel space
    const scaleX = d3
      .scaleLinear()
      .domain([0, GRID_SIZE - 1])
      .range([xScale(X_RANGE[0]), xScale(X_RANGE[1])]);
    const scaleY = d3
      .scaleLinear()
      .domain([0, GRID_SIZE - 1])
      .range([yScale(Y_RANGE[0]), yScale(Y_RANGE[1])]);

    const pathGen = d3.geoPath(
      d3.geoTransform({
        point(x: number, y: number) {
          (this as unknown as d3.GeoStream).stream.point(scaleX(x), scaleY(y));
        },
      }),
    );

    return geos.map((g) => ({
      d: pathGen(g) || '',
      value: g.value,
    }));
  }, [contourData, xScale, yScale]);

  const contourColorScale = useMemo(
    () =>
      d3
        .scaleSequential(d3.interpolateBlues)
        .domain([0, 20]),
    [],
  );

  // ─── L1 diamond ───
  const diamondRadius = useMemo(() => {
    // The diamond represents the L1 ball that roughly corresponds
    // to the regularization strength. Larger lambda = smaller feasible region.
    return Math.max(0.2, 2.5 - lambda * 2.0);
  }, [lambda]);

  const diamondPoints = useMemo(() => {
    const pts = [
      [diamondRadius, 0],
      [0, diamondRadius],
      [-diamondRadius, 0],
      [0, -diamondRadius],
    ];
    return pts.map((p) => `${xScale(p[0])},${yScale(p[1])}`).join(' ');
  }, [diamondRadius, xScale, yScale]);

  // ─── Visible trajectory (animated) ───
  const visibleTraj = useMemo(
    () => trajResult.traj.slice(0, currentStep + 1),
    [trajResult, currentStep],
  );

  // ─── Convergence plot scales ───
  const convXScale = useMemo(
    () => d3.scaleLinear().domain([0, MAX_ITER_2D]).range([0, innerW]),
    [innerW],
  );
  const convYScale = useMemo(() => {
    const gaps = trajResult.objectives.map((o) => Math.max(o - optimalVal, 1e-12));
    const maxGap = d3.max(gaps) || 1;
    return d3.scaleLog().domain([1e-8, maxGap * 2]).range([innerH, 0]).clamp(true);
  }, [innerH, trajResult, optimalVal]);

  // ─── Bar chart scales ───
  const barXScale = useMemo(
    () =>
      d3
        .scaleBand<number>()
        .domain(d3.range(barProblem.p))
        .range([0, innerW])
        .padding(0.15),
    [innerW, barProblem.p],
  );
  const barYScale = useMemo(() => {
    const allVals = [...barProblem.xTrue, ...barResult];
    const ext = d3.extent(allVals) as [number, number];
    const pad = (ext[1] - ext[0]) * 0.15 || 1;
    return d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([innerH, 0]);
  }, [innerH, barProblem.xTrue, barResult]);

  // ─── Replay handler ───
  const handleReplay = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(true);
  }, []);

  // ─── Contour panel D3 axes ───
  const contourAxesRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('.axis').remove();
      const g = svg.select<SVGGElement>('.axes-g');
      if (g.empty()) return;

      g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('font-size', '10px');

      g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('font-size', '10px');
    },
    [xScale, yScale, innerW, innerH],
  );

  // ─── Convergence panel D3 axes ───
  const convAxesRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('.axis').remove();
      svg
        .append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top + innerH})`)
        .call(d3.axisBottom(convXScale).ticks(5))
        .selectAll('text')
        .style('font-size', '10px');

      svg
        .append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`)
        .call(
          d3.axisLeft(convYScale).ticks(5, '.0e'),
        )
        .selectAll('text')
        .style('font-size', '10px');
    },
    [convXScale, convYScale, innerW, innerH],
  );

  // ─── Bar chart D3 axes ───
  const barAxesRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('.axis').remove();
      svg
        .append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top + innerH})`)
        .call(
          d3
            .axisBottom(barXScale)
            .tickValues(d3.range(0, barProblem.p, 2))
            .tickFormat((d) => String(d)),
        )
        .selectAll('text')
        .style('font-size', '9px');

      svg
        .append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`)
        .call(d3.axisLeft(barYScale).ticks(5))
        .selectAll('text')
        .style('font-size', '10px');
    },
    [barXScale, barYScale, innerW, innerH],
  );

  // ─── Convergence line ───
  const convLinePath = useMemo(() => {
    const gaps = trajResult.objectives.map((o) => Math.max(o - optimalVal, 1e-12));
    const pts = gaps
      .slice(0, currentStep + 1)
      .map((g, i) => `${convXScale(i)},${convYScale(g)}`);
    return pts.length > 1 ? `M${pts.join('L')}` : '';
  }, [trajResult, optimalVal, currentStep, convXScale, convYScale]);

  if (containerWidth === 0) {
    return (
      <div ref={containerRef} className="w-full min-h-[340px]" aria-label="Loading visualization" />
    );
  }

  const bandW = barXScale.bandwidth();

  return (
    <div ref={containerRef} className="w-full">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-3 text-sm">
        <label className="flex items-center gap-2">
          <span className="font-medium">&lambda;</span>
          <input
            type="range"
            min={0.01}
            max={1.0}
            step={0.01}
            value={lambda}
            onChange={(e) => setLambda(parseFloat(e.target.value))}
            className="w-28"
          />
          <span className="tabular-nums w-10">{lambda.toFixed(2)}</span>
        </label>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showSteps}
            onChange={(e) => setShowSteps(e.target.checked)}
          />
          <span>Show steps</span>
        </label>

        <label className="flex items-center gap-2">
          <span>Speed</span>
          <select
            value={speed}
            onChange={(e) => setSpeed(e.target.value as Speed)}
            className="border rounded px-1 py-0.5 text-sm bg-white dark:bg-gray-800"
          >
            <option value="slow">Slow</option>
            <option value="medium">Medium</option>
            <option value="fast">Fast</option>
          </select>
        </label>

        <button
          onClick={handleReplay}
          className="px-2 py-0.5 border rounded text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          Replay
        </button>
      </div>

      {/* Panels */}
      <div
        className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}
        role="figure"
        aria-label="Forward-backward splitting explorer with contour plot, convergence chart, and recovered vector"
      >
        {/* ──── Left panel: contour plot ──── */}
        <div>
          <p className="text-xs font-medium text-center mb-1 text-gray-600 dark:text-gray-400">
            Contour of g(x) + L1 trajectory
          </p>
          <svg role="img" aria-label="Forward backward explorer visualization (panel 1 of 3)"
            ref={(node) => {
              (contourSvgRef as React.MutableRefObject<SVGSVGElement | null>).current = node;
              (contourAxesRef as React.MutableRefObject<SVGSVGElement | null>).current = node;
            }}
            width={panelWidth}
            height={HEIGHT}
            onPointerDown={handleContourPointerDown}
            onPointerMove={handleContourPointerMove}
            onPointerUp={handleContourPointerUp}
            style={{ touchAction: 'none', cursor: isDragging.current ? 'grabbing' : 'default' }}
          >
            <defs>
              <clipPath id={`${id}-contour-clip`}>
                <rect width={innerW} height={innerH} />
              </clipPath>
              <marker
                id={`${id}-arrow-grad`}
                viewBox="0 0 10 6"
                refX="10"
                refY="3"
                markerWidth="8"
                markerHeight="6"
                orient="auto"
              >
                <path d="M0,0 L10,3 L0,6" fill={COLOR_GRAD} />
              </marker>
              <marker
                id={`${id}-arrow-prox`}
                viewBox="0 0 10 6"
                refX="10"
                refY="3"
                markerWidth="8"
                markerHeight="6"
                orient="auto"
              >
                <path d="M0,0 L10,3 L0,6" fill={COLOR_PROX} />
              </marker>
            </defs>

            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <g className="axes-g" />

              {/* Contour fills */}
              <g clipPath={`url(#${id}-contour-clip)`}>
                {contourPaths.map((c, i) => (
                  <path
                    key={i}
                    d={c.d}
                    fill={contourColorScale(c.value)}
                    fillOpacity={0.35}
                    stroke={contourColorScale(c.value)}
                    strokeOpacity={0.5}
                    strokeWidth={0.5}
                  />
                ))}

                {/* L1 diamond */}
                <polygon
                  points={diamondPoints}
                  fill="none"
                  stroke={COLOR_PROX}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  opacity={0.8}
                />

                {/* Trajectory line */}
                {visibleTraj.length > 1 && (
                  <polyline
                    points={visibleTraj
                      .map((p) => `${xScale(p[0])},${yScale(p[1])}`)
                      .join(' ')}
                    fill="none"
                    stroke={COLOR_TRAJ}
                    strokeWidth={1.5}
                    strokeOpacity={0.7}
                  />
                )}

                {/* Step decomposition arrows */}
                {showSteps &&
                  visibleTraj.slice(0, Math.min(5, visibleTraj.length)).map((pt, i) => {
                    if (i >= trajResult.gradSteps.length || i >= visibleTraj.length - 1) return null;
                    const from = pt;
                    const mid = trajResult.gradSteps[i];
                    const to = visibleTraj[i + 1];
                    return (
                      <g key={`step-${i}`}>
                        {/* Gradient arrow (forward) */}
                        <line
                          x1={xScale(from[0])}
                          y1={yScale(from[1])}
                          x2={xScale(mid[0])}
                          y2={yScale(mid[1])}
                          stroke={COLOR_GRAD}
                          strokeWidth={1.5}
                          markerEnd={`url(#${id}-arrow-grad)`}
                          opacity={0.8}
                        />
                        {/* Prox arrow (backward) */}
                        <line
                          x1={xScale(mid[0])}
                          y1={yScale(mid[1])}
                          x2={xScale(to[0])}
                          y2={yScale(to[1])}
                          stroke={COLOR_PROX}
                          strokeWidth={1.5}
                          markerEnd={`url(#${id}-arrow-prox)`}
                          opacity={0.8}
                        />
                      </g>
                    );
                  })}

                {/* Trajectory dots */}
                {visibleTraj.map((pt, i) => (
                  <circle
                    key={i}
                    cx={xScale(pt[0])}
                    cy={yScale(pt[1])}
                    r={i === 0 ? 5 : 2.5}
                    fill={i === 0 ? COLOR_X0 : COLOR_TRAJ}
                    stroke="white"
                    strokeWidth={i === 0 ? 1.5 : 0.5}
                    style={i === 0 ? { cursor: 'grab' } : undefined}
                  />
                ))}

                {/* Optimal point marker (ground truth) */}
                <circle
                  cx={xScale(bTrue2[0])}
                  cy={yScale(bTrue2[1])}
                  r={4}
                  fill="none"
                  stroke={COLOR_OPT}
                  strokeWidth={2}
                />
                <line
                  x1={xScale(bTrue2[0]) - 4}
                  y1={yScale(bTrue2[1])}
                  x2={xScale(bTrue2[0]) + 4}
                  y2={yScale(bTrue2[1])}
                  stroke={COLOR_OPT}
                  strokeWidth={2}
                />
                <line
                  x1={xScale(bTrue2[0])}
                  y1={yScale(bTrue2[1]) - 4}
                  x2={xScale(bTrue2[0])}
                  y2={yScale(bTrue2[1]) + 4}
                  stroke={COLOR_OPT}
                  strokeWidth={2}
                />
              </g>

              {/* Axis labels */}
              <text
                x={innerW / 2}
                y={innerH + 34}
                textAnchor="middle"
                className="fill-current text-gray-600 dark:text-gray-400"
                style={{ fontSize: '11px' }}
              >
                x₁
              </text>
              <text
                x={-innerH / 2}
                y={-32}
                textAnchor="middle"
                transform="rotate(-90)"
                className="fill-current text-gray-600 dark:text-gray-400"
                style={{ fontSize: '11px' }}
              >
                x₂
              </text>
            </g>
          </svg>
        </div>

        {/* ──── Center panel: convergence ──── */}
        <div>
          <p className="text-xs font-medium text-center mb-1 text-gray-600 dark:text-gray-400">
            F(x<sub>k</sub>) &minus; F* vs iteration (log scale)
          </p>
          <svg role="img" aria-label="Forward backward explorer visualization (panel 2 of 3)" ref={convAxesRef} width={panelWidth} height={HEIGHT}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {/* Convergence line */}
              {convLinePath && (
                <path
                  d={convLinePath}
                  fill="none"
                  stroke={COLOR_TRAJ}
                  strokeWidth={2}
                />
              )}

              {/* Current point highlight */}
              {currentStep > 0 && (() => {
                const gap = Math.max(
                  trajResult.objectives[currentStep] - optimalVal,
                  1e-12,
                );
                return (
                  <circle
                    cx={convXScale(currentStep)}
                    cy={convYScale(gap)}
                    r={4}
                    fill={COLOR_TRAJ}
                    stroke="white"
                    strokeWidth={1}
                  />
                );
              })()}

              {/* Axis labels */}
              <text
                x={innerW / 2}
                y={innerH + 34}
                textAnchor="middle"
                className="fill-current text-gray-600 dark:text-gray-400"
                style={{ fontSize: '11px' }}
              >
                Iteration k
              </text>
              <text
                x={-innerH / 2}
                y={-32}
                textAnchor="middle"
                transform="rotate(-90)"
                className="fill-current text-gray-600 dark:text-gray-400"
                style={{ fontSize: '11px' }}
              >
                F(x_k) − F*
              </text>
            </g>
          </svg>
        </div>

        {/* ──── Right panel: bar chart ──── */}
        <div>
          <p className="text-xs font-medium text-center mb-1 text-gray-600 dark:text-gray-400">
            Recovered sparse vector vs ground truth
          </p>
          <svg role="img" aria-label="Forward backward explorer visualization (panel 3 of 3)" ref={barAxesRef} width={panelWidth} height={HEIGHT}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {/* Zero line */}
              <line
                x1={0}
                y1={barYScale(0)}
                x2={innerW}
                y2={barYScale(0)}
                stroke="#9CA3AF"
                strokeWidth={0.5}
                strokeDasharray="2 2"
              />

              {/* Ground truth bars (background) */}
              {barProblem.xTrue.map((val, i) => {
                if (Math.abs(val) < 1e-10) return null;
                const xPos = barXScale(i);
                if (xPos === undefined) return null;
                const y0 = barYScale(0);
                const y1 = barYScale(val);
                return (
                  <rect
                    key={`truth-${i}`}
                    x={xPos}
                    y={Math.min(y0, y1)}
                    width={bandW}
                    height={Math.abs(y1 - y0)}
                    fill={COLOR_TRUTH}
                    opacity={0.3}
                  />
                );
              })}

              {/* Recovered bars (foreground, narrower) */}
              {barResult.map((val, i) => {
                if (Math.abs(val) < 1e-10) return null;
                const xPos = barXScale(i);
                if (xPos === undefined) return null;
                const y0 = barYScale(0);
                const y1 = barYScale(val);
                const offset = bandW * 0.2;
                return (
                  <rect
                    key={`rec-${i}`}
                    x={xPos + offset}
                    y={Math.min(y0, y1)}
                    width={bandW * 0.6}
                    height={Math.abs(y1 - y0)}
                    fill={COLOR_RECOVER}
                    opacity={0.85}
                  />
                );
              })}

              {/* Axis labels */}
              <text
                x={innerW / 2}
                y={innerH + 34}
                textAnchor="middle"
                className="fill-current text-gray-600 dark:text-gray-400"
                style={{ fontSize: '11px' }}
              >
                Coordinate index
              </text>
              <text
                x={-innerH / 2}
                y={-32}
                textAnchor="middle"
                transform="rotate(-90)"
                className="fill-current text-gray-600 dark:text-gray-400"
                style={{ fontSize: '11px' }}
              >
                Value
              </text>
            </g>
          </svg>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 mt-2 text-xs text-gray-600 dark:text-gray-400 justify-center">
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-3 h-3 rounded-full"
            style={{ backgroundColor: COLOR_X0 }}
          />
          x₀ (drag to move)
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-3 h-3 rounded-full"
            style={{ backgroundColor: COLOR_TRAJ }}
          />
          Trajectory
        </span>
        {showSteps && (
          <>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5" style={{ backgroundColor: COLOR_GRAD }} />
              Gradient (forward)
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5" style={{ backgroundColor: COLOR_PROX }} />
              Prox (backward)
            </span>
          </>
        )}
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-3 h-3 rounded-full border-2"
            style={{ borderColor: COLOR_OPT }}
          />
          Optimum
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-3 h-3"
            style={{ backgroundColor: COLOR_TRUTH, opacity: 0.3 }}
          />
          Ground truth
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block w-3 h-3"
            style={{ backgroundColor: COLOR_RECOVER, opacity: 0.85 }}
          />
          Recovered
        </span>
      </div>
    </div>
  );
}

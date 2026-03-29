import { useState, useEffect, useRef, useMemo, useId, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { projectSimplex, expGradientStep } from './shared/gradientDescentUtils';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const MARGIN = { top: 20, right: 15, bottom: 20, left: 15 };
const LEFT_HEIGHT = 340;
const RIGHT_HEIGHT = 280;
const RIGHT_MARGIN = { top: 30, right: 15, bottom: 30, left: 45 };

// ─── Colors ───

const COLOR_MIRROR = '#534AB7';
const COLOR_PGD = '#0F6E56';
const COLOR_OPTIMUM = '#ef4444';
const COLOR_START = '#D97706';

// ─── Objectives ───

const OBJECTIVES = [
  { c: [0.1, 0.8, 0.3], label: 'Optimum at e\u2081 (c = [0.1, 0.8, 0.3])' },
  { c: [0.3, 0.1, 0.8], label: 'Optimum at e\u2082 (c = [0.3, 0.1, 0.8])' },
  { c: [0.8, 0.3, 0.1], label: 'Optimum at e\u2083 (c = [0.8, 0.3, 0.1])' },
  { c: [0.3, 0.3, 0.4], label: 'Near center (c = [0.3, 0.3, 0.4])' },
];

// ─── Helpers ───

function baryToCart(b: number[], w: number, h: number): [number, number] {
  const v1 = [w / 2, MARGIN.top + 10];
  const v2 = [MARGIN.left + 10, h - MARGIN.bottom - 10];
  const v3 = [w - MARGIN.right - 10, h - MARGIN.bottom - 10];
  return [
    b[0] * v1[0] + b[1] * v2[0] + b[2] * v3[0],
    b[0] * v1[1] + b[1] * v2[1] + b[2] * v3[1],
  ];
}

// ─── Component ───

export default function MirrorDescentExplorer() {
  const instanceId = useId();
  const clipId = `md-clip-${instanceId.replace(/:/g, '')}`;

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───
  const [objectiveIdx, setObjectiveIdx] = useState(0);
  const [eta, setEta] = useState(2.0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // ─── Trajectories ───
  const { mirrorPath, projPath } = useMemo(() => {
    const c = OBJECTIVES[objectiveIdx].c;
    const gradF = () => c;
    const x0 = [1 / 3, 1 / 3, 1 / 3];

    // Mirror descent (exponentiated gradient)
    let xm = [...x0];
    const mirrorPath = [xm.slice()];
    for (let k = 0; k < 40; k++) {
      xm = expGradientStep(xm, gradF(), eta);
      mirrorPath.push(xm.slice());
    }

    // Projected GD — scale step to make PGD visually comparable to mirror descent
    const PGD_STEP_SCALE = 0.1;
    let xp = [...x0];
    const projPath = [xp.slice()];
    for (let k = 0; k < 40; k++) {
      const g = gradF();
      const xRaw = xp.map((xi, i) => xi - eta * PGD_STEP_SCALE * g[i]);
      xp = projectSimplex(xRaw);
      projPath.push(xp.slice());
    }

    return { mirrorPath, projPath };
  }, [objectiveIdx, eta]);

  // Reset animation when parameters change
  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, [objectiveIdx, eta]);

  // ─── Animation ───
  const animRef = useRef<number>(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!isPlaying) return;
    const advance = () => {
      setCurrentStep(prev => {
        if (prev >= mirrorPath.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
      timeoutRef.current = setTimeout(() => {
        animRef.current = requestAnimationFrame(advance);
      }, 120);
    };
    animRef.current = requestAnimationFrame(advance);
    return () => {
      cancelAnimationFrame(animRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [isPlaying, mirrorPath.length]);

  // ─── Panel dimensions ───
  const isWide = containerWidth >= SM_BREAKPOINT;
  const panelWidth = useMemo(() => {
    if (!containerWidth) return 0;
    return isWide ? Math.floor((containerWidth - 16) / 2) : containerWidth;
  }, [containerWidth, isWide]);

  // ─── Left panel: simplex ───
  useEffect(() => {
    const svg = d3.select(leftSvgRef.current);
    if (!panelWidth || !svg.node()) return;
    svg.selectAll('*').remove();

    const w = panelWidth;
    const h = LEFT_HEIGHT;

    // Triangle vertices
    const v1: [number, number] = [w / 2, MARGIN.top + 10];
    const v2: [number, number] = [MARGIN.left + 10, h - MARGIN.bottom - 10];
    const v3: [number, number] = [w - MARGIN.right - 10, h - MARGIN.bottom - 10];

    const g = svg.append('g');

    // Triangle outline
    g.append('polygon')
      .attr('points', `${v1[0]},${v1[1]} ${v2[0]},${v2[1]} ${v3[0]},${v3[1]}`)
      .attr('fill', 'none')
      .attr('stroke', '#64748b')
      .attr('stroke-width', 1.5);

    // Vertex labels
    g.append('text')
      .attr('x', v1[0]).attr('y', v1[1] - 8)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .style('fill', 'currentColor')
      .text('e\u2081');

    g.append('text')
      .attr('x', v2[0] - 8).attr('y', v2[1] + 14)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .style('fill', 'currentColor')
      .text('e\u2082');

    g.append('text')
      .attr('x', v3[0] + 8).attr('y', v3[1] + 14)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .style('fill', 'currentColor')
      .text('e\u2083');

    // Mirror descent trajectory
    const mirrorVisible = mirrorPath.slice(0, currentStep + 1);
    if (mirrorVisible.length > 1) {
      const lineGen = d3.line<number[]>()
        .x(d => baryToCart(d, w, h)[0])
        .y(d => baryToCart(d, w, h)[1]);

      g.append('path')
        .datum(mirrorVisible)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', COLOR_MIRROR)
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.8);
    }

    g.selectAll('circle.mirror')
      .data(mirrorVisible)
      .join('circle')
      .attr('class', 'mirror')
      .attr('cx', d => baryToCart(d, w, h)[0])
      .attr('cy', d => baryToCart(d, w, h)[1])
      .attr('r', 2.5)
      .attr('fill', COLOR_MIRROR)
      .attr('fill-opacity', 0.7);

    // Projected GD trajectory
    const projVisible = projPath.slice(0, currentStep + 1);
    if (projVisible.length > 1) {
      const lineGen = d3.line<number[]>()
        .x(d => baryToCart(d, w, h)[0])
        .y(d => baryToCart(d, w, h)[1]);

      g.append('path')
        .datum(projVisible)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', COLOR_PGD)
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.8);
    }

    g.selectAll('circle.pgd')
      .data(projVisible)
      .join('circle')
      .attr('class', 'pgd')
      .attr('cx', d => baryToCart(d, w, h)[0])
      .attr('cy', d => baryToCart(d, w, h)[1])
      .attr('r', 2.5)
      .attr('fill', COLOR_PGD)
      .attr('fill-opacity', 0.7);

    // Optimum star — argmin c_i gives the vertex the objective pushes toward
    const c = OBJECTIVES[objectiveIdx].c;
    const minIdx = c.indexOf(Math.min(...c));
    const optBary = [0, 0, 0];
    optBary[minIdx] = 1;
    const [ox, oy] = baryToCart(optBary, w, h);

    // Draw a star shape
    const starPath = d3.symbol().type(d3.symbolStar).size(120);
    g.append('path')
      .attr('d', starPath)
      .attr('transform', `translate(${ox},${oy})`)
      .attr('fill', COLOR_OPTIMUM)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);

    // Start point at centroid
    const [sx, sy] = baryToCart([1 / 3, 1 / 3, 1 / 3], w, h);
    g.append('circle')
      .attr('cx', sx)
      .attr('cy', sy)
      .attr('r', 6)
      .attr('fill', COLOR_START)
      .attr('fill-opacity', 0.85)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    // Title
    svg.append('text')
      .attr('x', w / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .style('fill', 'currentColor')
      .text('Probability simplex');

    // Legend
    const legend = g.append('g').attr('transform', `translate(${w - 150}, ${MARGIN.top + 14})`);

    legend.append('line')
      .attr('x1', 0).attr('y1', 0).attr('x2', 16).attr('y2', 0)
      .attr('stroke', COLOR_MIRROR).attr('stroke-width', 2);
    legend.append('text')
      .attr('x', 20).attr('y', 4)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('Mirror descent');

    legend.append('line')
      .attr('x1', 0).attr('y1', 16).attr('x2', 16).attr('y2', 16)
      .attr('stroke', COLOR_PGD).attr('stroke-width', 2);
    legend.append('text')
      .attr('x', 20).attr('y', 20)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('Projected GD');
  }, [panelWidth, mirrorPath, projPath, currentStep, objectiveIdx]);

  // ─── Right panel: Bregman divergence comparison ───
  useEffect(() => {
    const svg = d3.select(rightSvgRef.current);
    if (!panelWidth || !svg.node()) return;
    svg.selectAll('*').remove();

    const w = panelWidth - RIGHT_MARGIN.left - RIGHT_MARGIN.right;
    const h = RIGHT_HEIGHT - RIGHT_MARGIN.top - RIGHT_MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${RIGHT_MARGIN.left},${RIGHT_MARGIN.top})`);

    // Defs: clipPath
    const defs = g.append('defs');
    defs.append('clipPath')
      .attr('id', clipId)
      .append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', w).attr('height', h);

    const plotArea = g.append('g').attr('clip-path', `url(#${clipId})`);

    const p0 = 0.5;
    const nSamples = 300;
    const pMin = 0.01;
    const pMax = 2.0;
    const pValues = d3.range(nSamples).map(i => pMin + (pMax - pMin) * i / (nSamples - 1));

    const klData = pValues.map(p => ({
      p,
      v: p * Math.log(p / p0) - p + p0,
    }));

    const eucData = pValues.map(p => ({
      p,
      v: 0.5 * (p - p0) * (p - p0),
    }));

    const yMax = Math.max(
      d3.max(klData, d => d.v) ?? 1,
      d3.max(eucData, d => d.v) ?? 1
    );

    const xScale = d3.scaleLinear().domain([pMin, pMax]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, yMax * 1.1]).range([h, 0]);

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

    // Axis labels
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 26)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('p');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2)
      .attr('y', -35)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', 'currentColor')
      .text('Divergence');

    // Vertical dashed line at p0
    plotArea.append('line')
      .attr('x1', xScale(p0)).attr('y1', 0)
      .attr('x2', xScale(p0)).attr('y2', h)
      .attr('stroke', '#94a3b8')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3');

    plotArea.append('text')
      .attr('x', xScale(p0) + 4)
      .attr('y', 12)
      .style('font-size', '10px')
      .style('fill', '#94a3b8')
      .text(`p\u2080 = ${p0}`);

    // KL divergence curve
    const klLine = d3.line<{ p: number; v: number }>()
      .x(d => xScale(d.p))
      .y(d => yScale(d.v));

    plotArea.append('path')
      .datum(klData)
      .attr('d', klLine)
      .attr('fill', 'none')
      .attr('stroke', COLOR_MIRROR)
      .attr('stroke-width', 2);

    // Euclidean curve
    const eucLine = d3.line<{ p: number; v: number }>()
      .x(d => xScale(d.p))
      .y(d => yScale(d.v));

    plotArea.append('path')
      .datum(eucData)
      .attr('d', eucLine)
      .attr('fill', 'none')
      .attr('stroke', COLOR_PGD)
      .attr('stroke-width', 2);

    // Legend
    const legend = g.append('g').attr('transform', `translate(${w - 160}, 5)`);

    legend.append('line')
      .attr('x1', 0).attr('y1', 0).attr('x2', 16).attr('y2', 0)
      .attr('stroke', COLOR_MIRROR).attr('stroke-width', 2);
    legend.append('text')
      .attr('x', 20).attr('y', 4)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('KL divergence');

    legend.append('line')
      .attr('x1', 0).attr('y1', 16).attr('x2', 16).attr('y2', 16)
      .attr('stroke', COLOR_PGD).attr('stroke-width', 2);
    legend.append('text')
      .attr('x', 20).attr('y', 20)
      .style('font-size', '10px').style('fill', 'currentColor')
      .text('\u00BD\u2016p \u2212 p\u2080\u2016\u00B2');

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .style('fill', 'currentColor')
      .text('Bregman divergence comparison');
  }, [panelWidth, clipId]);

  // ─── Handlers ───
  const handlePlay = useCallback(() => {
    if (currentStep >= mirrorPath.length - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(prev => !prev);
  }, [currentStep, mirrorPath.length]);

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
        <svg role="img" aria-label="Mirror descent explorer visualization (panel 1 of 2)"
          ref={leftSvgRef}
          width={panelWidth}
          height={LEFT_HEIGHT}
          style={{ overflow: 'visible' }}
        />
        <svg role="img" aria-label="Mirror descent explorer visualization (panel 2 of 2)"
          ref={rightSvgRef}
          width={panelWidth}
          height={RIGHT_HEIGHT}
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
        {/* Objective dropdown */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 14 }}>
          <span style={{ fontWeight: 500 }}>Objective:</span>
          <select
            value={objectiveIdx}
            onChange={e => setObjectiveIdx(Number(e.target.value))}
            style={{
              padding: '2px 6px',
              borderRadius: 4,
              border: '1px solid #cbd5e1',
              fontSize: 14,
              background: 'inherit',
              color: 'inherit',
            }}
          >
            {OBJECTIVES.map((obj, i) => (
              <option key={i} value={i}>{obj.label}</option>
            ))}
          </select>
        </label>

        {/* Step size slider */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 14 }}>
          <span style={{ fontWeight: 500 }}>
            {'\u03B7 = '}
            <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: 32, display: 'inline-block' }}>
              {eta.toFixed(1)}
            </span>
          </span>
          <input
            type="range"
            min={0.1}
            max={5.0}
            step={0.1}
            value={eta}
            onChange={e => setEta(Number(e.target.value))}
            style={{ width: 120 }}
          />
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
          Step {currentStep} / {mirrorPath.length - 1}
        </span>
      </div>
    </div>
  );
}

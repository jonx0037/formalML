import { useState, useEffect, useRef, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { makeQuadratic, armijoBacktracking, runGDTrajectory } from './shared/gradientDescentUtils';

const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 30, left: 40 };

const COLORS = {
  fixed: '#0F6E56',
  exact: '#534AB7',
  armijo: '#D97706',
};

// ─── Component ───

export default function StepSizeExplorer() {
  const instanceId = useId();
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  // ─── State ───
  const [kappa, setKappa] = useState(10);
  const [armijoC, setArmijoC] = useState(0.1);
  const [armijoBeta, setArmijoBeta] = useState(0.5);
  const [showFixed, setShowFixed] = useState(true);
  const [showExact, setShowExact] = useState(true);
  const [showArmijo, setShowArmijo] = useState(true);
  const [x0, setX0] = useState<[number, number]>([2.5, 2.5]);

  // ─── Trajectories ───

  const problem = useMemo(() => makeQuadratic(kappa), [kappa]);

  const fixedTrajectory = useMemo(
    () => runGDTrajectory(problem, x0, 1 / problem.L, 100),
    [problem, x0],
  );

  const exactTrajectory = useMemo(() => {
    let x = [...x0];
    const path = [x.slice()];
    const objectives = [problem.f(x)];
    for (let k = 0; k < 100; k++) {
      const g = problem.grad(x);
      const Ag = [
        problem.A[0][0] * g[0] + problem.A[0][1] * g[1],
        problem.A[1][0] * g[0] + problem.A[1][1] * g[1],
      ];
      const gNorm2 = g[0] * g[0] + g[1] * g[1];
      const gAg = g[0] * Ag[0] + g[1] * Ag[1];
      if (gNorm2 < 1e-20) break;
      const eta = gNorm2 / gAg;
      x = [x[0] - eta * g[0], x[1] - eta * g[1]];
      path.push(x.slice());
      objectives.push(problem.f(x));
    }
    return { path, objectives };
  }, [problem, x0]);

  const armijoTrajectory = useMemo(() => {
    let x = [...x0];
    const path = [x.slice()];
    const objectives = [problem.f(x)];
    for (let k = 0; k < 100; k++) {
      const g = problem.grad(x);
      const gNorm2 = g[0] * g[0] + g[1] * g[1];
      if (gNorm2 < 1e-20) break;
      const d = [-g[0], -g[1]];
      const alpha = armijoBacktracking(problem.f, g, x, d, armijoC, armijoBeta);
      x = [x[0] + alpha * d[0], x[1] + alpha * d[1]];
      path.push(x.slice());
      objectives.push(problem.f(x));
      if (Math.abs(x[0]) > 200 || Math.abs(x[1]) > 200) break;
    }
    return { path, objectives };
  }, [problem, x0, armijoC, armijoBeta]);

  // ─── Layout ───
  const isWide = containerWidth >= SM_BREAKPOINT;
  const panelWidth = isWide ? Math.floor((containerWidth - 16) / 2) : containerWidth;
  const panelHeight = 300;
  const domain: [number, number] = [-4, 4];

  // ─── Left panel: contour plot + trajectories ───
  useEffect(() => {
    const svg = d3.select(leftSvgRef.current);
    svg.selectAll('*').remove();
    if (!panelWidth || panelWidth < 50) return;

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = panelHeight - MARGIN.top - MARGIN.bottom;
    const clipId = `sse-left-${instanceId.replace(/:/g, '')}`;

    const g = svg
      .attr('width', panelWidth)
      .attr('height', panelHeight)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain(domain).range([0, w]);
    const yScale = d3.scaleLinear().domain(domain).range([h, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5))
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
      .text('Contour + Trajectories');

    // Clip path
    g.append('defs')
      .append('clipPath')
      .attr('id', clipId)
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', w)
      .attr('height', h);

    const plotArea = g.append('g').attr('clip-path', `url(#${clipId})`);

    // ── Contours ──
    const gridSize = 100;
    const values = new Float64Array(gridSize * gridSize);
    for (let j = 0; j < gridSize; j++) {
      for (let i = 0; i < gridSize; i++) {
        const cx = domain[0] + (domain[1] - domain[0]) * i / (gridSize - 1);
        const cy = domain[0] + (domain[1] - domain[0]) * j / (gridSize - 1);
        values[j * gridSize + i] = problem.f([cx, cy]);
      }
    }

    const contourGenerator = d3.contours().size([gridSize, gridSize]).thresholds(15);
    const contourData = contourGenerator(values);

    const geoTransform = d3.geoTransform({
      point(px: number, py: number) {
        const sx = xScale(domain[0] + (domain[1] - domain[0]) * px / (gridSize - 1));
        const sy = yScale(domain[0] + (domain[1] - domain[0]) * py / (gridSize - 1));
        this.stream.point(sx, sy);
      },
    });
    const pathGen = d3.geoPath().projection(geoTransform);

    plotArea
      .selectAll('path.contour')
      .data(contourData)
      .enter()
      .append('path')
      .attr('class', 'contour')
      .attr('d', pathGen as any)
      .attr('fill', 'none')
      .attr('stroke', '#64748b')
      .attr('stroke-opacity', 0.3)
      .attr('stroke-width', 1);

    // ── Trajectories ──
    const strategies = [
      { key: 'fixed' as const, data: fixedTrajectory, show: showFixed },
      { key: 'exact' as const, data: exactTrajectory, show: showExact },
      { key: 'armijo' as const, data: armijoTrajectory, show: showArmijo },
    ];

    for (const { key, data, show } of strategies) {
      if (!show) continue;
      const color = COLORS[key];

      // Polyline
      const lineGen = d3
        .line<number[]>()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]));

      plotArea
        .append('path')
        .datum(data.path)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 1.8)
        .attr('stroke-opacity', 0.85);

      // Iterate dots
      plotArea
        .selectAll(`circle.iter-${key}`)
        .data(data.path)
        .enter()
        .append('circle')
        .attr('class', `iter-${key}`)
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('r', 2.5)
        .attr('fill', color)
        .attr('fill-opacity', 0.7);
    }

    // ── Optimum star ──
    const starOuter = 7;
    const starInner = 3;
    let starPath = '';
    for (let i = 0; i < 10; i++) {
      const angle = Math.PI / 2 + (i * Math.PI) / 5;
      const r = i % 2 === 0 ? starOuter : starInner;
      const px = Math.cos(angle) * r;
      const py = -Math.sin(angle) * r;
      starPath += (i === 0 ? 'M' : 'L') + `${px},${py}`;
    }
    starPath += 'Z';

    plotArea
      .append('path')
      .attr('d', starPath)
      .attr('transform', `translate(${xScale(0)},${yScale(0)})`)
      .attr('fill', '#ef4444')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);

    // ── Draggable start point ──
    const dragCircle = plotArea
      .append('circle')
      .attr('cx', xScale(x0[0]))
      .attr('cy', yScale(x0[1]))
      .attr('r', 8)
      .attr('fill', COLORS.armijo)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'grab');

    const drag = d3.drag<SVGCircleElement, unknown>()
      .on('start', function () {
        d3.select(this).style('cursor', 'grabbing');
      })
      .on('drag', (event) => {
        const nx = Math.max(domain[0], Math.min(domain[1], xScale.invert(event.x)));
        const ny = Math.max(domain[0], Math.min(domain[1], yScale.invert(event.y)));
        setX0([nx, ny]);
      })
      .on('end', function () {
        d3.select(this).style('cursor', 'grab');
      });

    dragCircle.call(drag);
  }, [
    containerWidth, panelWidth, panelHeight, problem,
    fixedTrajectory, exactTrajectory, armijoTrajectory,
    showFixed, showExact, showArmijo, x0, instanceId,
  ]);

  // ─── Right panel: convergence (log scale) ───
  useEffect(() => {
    const svg = d3.select(rightSvgRef.current);
    svg.selectAll('*').remove();
    if (!panelWidth || panelWidth < 50) return;

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = panelHeight - MARGIN.top - MARGIN.bottom;
    const clipId = `sse-right-${instanceId.replace(/:/g, '')}`;

    const g = svg
      .attr('width', panelWidth)
      .attr('height', panelHeight)
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Gather all enabled objective sequences
    const strategies = [
      { key: 'fixed' as const, data: fixedTrajectory, show: showFixed },
      { key: 'exact' as const, data: exactTrajectory, show: showExact },
      { key: 'armijo' as const, data: armijoTrajectory, show: showArmijo },
    ];

    const fStar = 0; // optimum of our quadratic
    let maxIter = 0;
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const { data, show } of strategies) {
      if (!show) continue;
      maxIter = Math.max(maxIter, data.objectives.length - 1);
      for (const v of data.objectives) {
        const gap = v - fStar;
        if (gap > 1e-16) {
          minVal = Math.min(minVal, gap);
          maxVal = Math.max(maxVal, gap);
        }
      }
    }

    if (maxIter === 0) maxIter = 1;
    if (minVal === Infinity) { minVal = 1e-10; maxVal = 1; }

    const xScale = d3.scaleLinear().domain([0, maxIter]).range([0, w]);
    const yScale = d3.scaleLog().domain([Math.max(minVal * 0.5, 1e-16), maxVal * 2]).range([h, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('d')))
      .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick line').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').style('font-size', '11px'));

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5, '.0e'))
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
      .text('f(xₖ) − f* (log scale)');

    // Clip path
    g.append('defs')
      .append('clipPath')
      .attr('id', clipId)
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', w)
      .attr('height', h);

    const plotArea = g.append('g').attr('clip-path', `url(#${clipId})`);

    // ── Convergence curves ──
    for (const { key, data, show } of strategies) {
      if (!show) continue;
      const color = COLORS[key];

      const pts = data.objectives
        .map((v, i) => ({ k: i, gap: v - fStar }))
        .filter((d) => d.gap > 1e-16);

      const lineGen = d3
        .line<{ k: number; gap: number }>()
        .x((d) => xScale(d.k))
        .y((d) => yScale(d.gap));

      plotArea
        .append('path')
        .datum(pts)
        .attr('d', lineGen)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 2);
    }

    // Axis labels
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 28)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-size', '11px')
      .text('Iteration k');
  }, [
    containerWidth, panelWidth, panelHeight,
    fixedTrajectory, exactTrajectory, armijoTrajectory,
    showFixed, showExact, showArmijo, instanceId,
  ]);

  // ─── Guard for initial render ───
  if (!containerWidth) {
    return <div ref={containerRef} style={{ minHeight: 400 }} />;
  }

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: isWide ? 'row' : 'column',
          width: '100%',
          gap: isWide ? 16 : 0,
        }}
      >
        <svg ref={leftSvgRef} />
        <svg ref={rightSvgRef} />
      </div>

      {/* ── Controls ── */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: 16,
          padding: '10px 4px 0',
        }}
      >
        {/* κ slider */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '13px', color: 'var(--color-muted)' }}>
          κ = {kappa}
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={kappa}
            onChange={(e) => setKappa(Number(e.target.value))}
            style={{ width: 90 }}
          />
        </label>

        {/* Armijo c slider */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '13px', color: 'var(--color-muted)' }}>
          c = {armijoC.toFixed(2)}
          <input
            type="range"
            min={0.01}
            max={0.5}
            step={0.01}
            value={armijoC}
            onChange={(e) => setArmijoC(Number(e.target.value))}
            style={{ width: 80 }}
          />
        </label>

        {/* Armijo β slider */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '13px', color: 'var(--color-muted)' }}>
          β = {armijoBeta.toFixed(2)}
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={armijoBeta}
            onChange={(e) => setArmijoBeta(Number(e.target.value))}
            style={{ width: 80 }}
          />
        </label>

        {/* Toggles */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '13px', color: COLORS.fixed }}>
          <input type="checkbox" checked={showFixed} onChange={(e) => setShowFixed(e.target.checked)} />
          Fixed
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '13px', color: COLORS.exact }}>
          <input type="checkbox" checked={showExact} onChange={(e) => setShowExact(e.target.checked)} />
          Exact
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '13px', color: COLORS.armijo }}>
          <input type="checkbox" checked={showArmijo} onChange={(e) => setShowArmijo(e.target.checked)} />
          Armijo
        </label>
      </div>
    </div>
  );
}

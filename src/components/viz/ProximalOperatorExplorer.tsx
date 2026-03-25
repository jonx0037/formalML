import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { linspace } from './shared/proximalUtils';

const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 50 };
const HEIGHT = 320;
const X_DOMAIN: [number, number] = [-4, 4];
const NUM_SAMPLES = 300;
const Y_AXIS_PADDING = 1.15;   // 15% headroom above tallest value
const Y_AXIS_DEFAULT_MAX = 6;  // fallback when all finite values are tiny
const Y_AXIS_HARD_CAP = 20;    // prevent extreme y range for indicator funcs

// ─── Colors ───
const COLOR_F = '#0F6E56';       // teal — f(x)
const COLOR_SUM = '#534AB7';     // purple — sum / prox curve
const COLOR_ANCHOR = '#D97706';  // amber — anchor v
const COLOR_QUAD = '#6B6E6B';    // slate — quadratic penalty
const COLOR_STAR = '#DC2626';    // red — minimizer star
const COLOR_IDENTITY = '#3f3f46'; // zinc-700 — identity reference line

// ─── Function definitions with closed-form proximal operators ───

interface FuncDef {
  key: string;
  label: string;
  /** Short name for the formula readout (no parenthetical explanation). */
  shortLabel: string;
  fn: (x: number) => number;
  prox: (v: number, lam: number) => number;
  /** domain of f; returns Infinity outside */
  domain?: (x: number) => boolean;
}

const FUNCTIONS: FuncDef[] = [
  {
    key: 'abs',
    label: '|x|  (soft-thresholding)',
    shortLabel: '|x|',
    fn: (x) => Math.abs(x),
    prox: (v, lam) => Math.sign(v) * Math.max(Math.abs(v) - lam, 0),
  },
  {
    key: 'indicator',
    label: 'ι_{[-1,1]}  (projection)',
    shortLabel: 'ι_{[-1,1]}',
    fn: (x) => (Math.abs(x) <= 1 + 1e-9 ? 0 : Infinity),
    prox: (v, _lam) => Math.max(-1, Math.min(1, v)),
    domain: (x) => Math.abs(x) <= 1 + 1e-9,
  },
  {
    key: 'quad',
    label: 'x²  (shrinkage)',
    shortLabel: 'x²',
    fn: (x) => x * x,
    prox: (v, lam) => v / (1 + 2 * lam),
  },
  {
    key: 'nn_soft',
    label: '|x| + ι_{[0,∞)}  (non-neg soft-thresh)',
    shortLabel: '|x| + ι_{[0,∞)}',
    fn: (x) => (x >= -1e-9 ? Math.abs(x) : Infinity),
    prox: (v, lam) => Math.max(v - lam, 0),
    domain: (x) => x >= -1e-9,
  },
];

export default function ProximalOperatorExplorer() {
  const uid = useId();
  const clipId = `prox-clip-${uid.replace(/:/g, '')}`;
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [v, setV] = useState(1.5);
  const [lambda, setLambda] = useState(1.0);
  const [funcIdx, setFuncIdx] = useState(0);

  const selectedFunc = FUNCTIONS[funcIdx];
  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? Math.max(containerWidth, 280)
    : Math.max(Math.floor((containerWidth - 16) / 2), 280);
  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

  // ─── Computed data ───

  const xs = useMemo(() => linspace(X_DOMAIN[0], X_DOMAIN[1], NUM_SAMPLES), []);

  const proxValue = useMemo(
    () => selectedFunc.prox(v, lambda),
    [v, lambda, funcIdx],
  );

  const leftData = useMemo(() => {
    const quad = (x: number) => (1 / (2 * lambda)) * (x - v) ** 2;
    return xs.map((x) => {
      const fx = selectedFunc.fn(x);
      const qx = quad(x);
      return { x, fx, qx, sum: fx + qx };
    });
  }, [xs, v, lambda, funcIdx]);

  const rightData = useMemo(() => {
    return xs.map((vi) => ({ vi, prox: selectedFunc.prox(vi, lambda) }));
  }, [xs, lambda, funcIdx]);

  // ─── Left panel: proximal landscape ───

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('width', panelWidth).attr('height', HEIGHT);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      // Clip path
      g.append('defs')
        .append('clipPath')
        .attr('id', `${clipId}-left`)
        .append('rect')
        .attr('width', innerW)
        .attr('height', innerH);

      const xScale = d3.scaleLinear().domain(X_DOMAIN).range([0, innerW]);
      const finiteVals = leftData.filter((d) => isFinite(d.fx) && isFinite(d.sum));
      const yMax = d3.max(finiteVals, (d) => Math.max(d.fx, d.qx, d.sum)) ?? Y_AXIS_DEFAULT_MAX;
      const yScale = d3.scaleLinear().domain([0, Math.min(yMax * Y_AXIS_PADDING, Y_AXIS_HARD_CAP)]).range([innerH, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(8))
        .selectAll('text')
        .style('fill', '#a1a1aa');
      g.selectAll('.domain, line').style('stroke', '#52525b');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', '#a1a1aa');

      const plotG = g.append('g').attr('clip-path', `url(#${clipId}-left)`);

      // Line generators
      const makeLine = (yAccessor: (d: (typeof leftData)[0]) => number) =>
        d3.line<(typeof leftData)[0]>()
          .defined((d) => isFinite(yAccessor(d)) && isFinite(d.x))
          .x((d) => xScale(d.x))
          .y((d) => yScale(yAccessor(d)));

      // f(x)
      plotG.append('path')
        .datum(leftData)
        .attr('d', makeLine((d) => d.fx))
        .attr('fill', 'none')
        .style('stroke', COLOR_F)
        .style('stroke-width', '2');

      // Quadratic penalty
      plotG.append('path')
        .datum(leftData)
        .attr('d', makeLine((d) => d.qx))
        .attr('fill', 'none')
        .style('stroke', COLOR_QUAD)
        .style('stroke-width', '1.5')
        .style('stroke-dasharray', '6,3');

      // Sum
      plotG.append('path')
        .datum(leftData)
        .attr('d', makeLine((d) => d.sum))
        .attr('fill', 'none')
        .style('stroke', COLOR_SUM)
        .style('stroke-width', '2.5');

      // Anchor v vertical line
      plotG.append('line')
        .attr('x1', xScale(v))
        .attr('x2', xScale(v))
        .attr('y1', 0)
        .attr('y2', innerH)
        .style('stroke', COLOR_ANCHOR)
        .style('stroke-width', '1.5')
        .style('stroke-dasharray', '4,4');

      // Draggable v circle
      const vCircle = plotG.append('circle')
        .attr('cx', xScale(v))
        .attr('cy', innerH)
        .attr('r', 8)
        .style('fill', COLOR_ANCHOR)
        .style('cursor', 'ew-resize')
        .style('opacity', '0.85');

      // v label
      plotG.append('text')
        .attr('x', xScale(v))
        .attr('y', innerH - 12)
        .attr('text-anchor', 'middle')
        .style('fill', COLOR_ANCHOR)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`v=${v.toFixed(1)}`);

      // Minimizer star
      const starX = xScale(proxValue);
      const sumAtProx = selectedFunc.fn(proxValue) + (1 / (2 * lambda)) * (proxValue - v) ** 2;
      const starY = yScale(sumAtProx);

      if (isFinite(sumAtProx)) {
        const star = d3.symbol().type(d3.symbolStar).size(120);
        plotG.append('path')
          .attr('d', star()!)
          .attr('transform', `translate(${starX},${starY})`)
          .style('fill', COLOR_STAR)
          .style('stroke', '#fff')
          .style('stroke-width', '0.8');

        plotG.append('text')
          .attr('x', starX + 10)
          .attr('y', starY - 8)
          .style('fill', COLOR_STAR)
          .style('font-size', '11px')
          .style('font-weight', '600')
          .text(`prox=${proxValue.toFixed(2)}`);
      }

      // Panel title
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', -12)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Proximal landscape');

      // Legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 130}, 8)`);
      const items = [
        { color: COLOR_F, label: 'f(x)', dash: '' },
        { color: COLOR_QUAD, label: '(1/2λ)‖x−v‖²', dash: '6,3' },
        { color: COLOR_SUM, label: 'sum (objective)', dash: '' },
      ];
      items.forEach(({ color, label, dash }, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`);
        row.append('line')
          .attr('x1', 0).attr('x2', 16).attr('y1', 0).attr('y2', 0)
          .style('stroke', color)
          .style('stroke-width', '2')
          .style('stroke-dasharray', dash);
        row.append('text')
          .attr('x', 20).attr('y', 4)
          .style('fill', '#a1a1aa')
          .style('font-size', '10px')
          .text(label);
      });

      // Drag behavior
      const drag = d3.drag<SVGCircleElement, unknown>()
        .on('drag', (event) => {
          const newV = xScale.invert(event.x - MARGIN.left);
          setV(Math.max(X_DOMAIN[0], Math.min(X_DOMAIN[1], newV)));
        });

      vCircle.call(drag);
    },
    [leftData, v, lambda, proxValue, panelWidth, innerW, innerH],
  );

  // ─── Right panel: prox as function of v ───

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('width', panelWidth).attr('height', HEIGHT);

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      g.append('defs')
        .append('clipPath')
        .attr('id', `${clipId}-right`)
        .append('rect')
        .attr('width', innerW)
        .attr('height', innerH);

      const xScale = d3.scaleLinear().domain(X_DOMAIN).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(X_DOMAIN).range([innerH, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(8))
        .selectAll('text')
        .style('fill', '#a1a1aa');
      g.selectAll('.domain, line').style('stroke', '#52525b');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(8))
        .selectAll('text')
        .style('fill', '#a1a1aa');

      // Axis labels
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 34)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '12px')
        .text('v');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerH / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .style('fill', '#a1a1aa')
        .style('font-size', '12px')
        .text('prox_λf(v)');

      const plotG = g.append('g').attr('clip-path', `url(#${clipId}-right)`);

      // Identity line for reference
      plotG.append('line')
        .attr('x1', xScale(X_DOMAIN[0]))
        .attr('y1', yScale(X_DOMAIN[0]))
        .attr('x2', xScale(X_DOMAIN[1]))
        .attr('y2', yScale(X_DOMAIN[1]))
        .style('stroke', COLOR_IDENTITY)
        .style('stroke-width', '1')
        .style('stroke-dasharray', '4,4');

      // Prox curve
      const line = d3.line<(typeof rightData)[0]>()
        .x((d) => xScale(d.vi))
        .y((d) => yScale(d.prox));

      plotG.append('path')
        .datum(rightData)
        .attr('d', line)
        .attr('fill', 'none')
        .style('stroke', COLOR_SUM)
        .style('stroke-width', '2.5');

      // Current (v, prox) dot
      plotG.append('circle')
        .attr('cx', xScale(v))
        .attr('cy', yScale(proxValue))
        .attr('r', 6)
        .style('fill', COLOR_ANCHOR)
        .style('stroke', '#fff')
        .style('stroke-width', '1.5');

      // Label
      plotG.append('text')
        .attr('x', xScale(v) + 10)
        .attr('y', yScale(proxValue) - 10)
        .style('fill', COLOR_ANCHOR)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(`(${v.toFixed(1)}, ${proxValue.toFixed(2)})`);

      // Panel title
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', -12)
        .attr('text-anchor', 'middle')
        .style('fill', '#d4d4d8')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Proximal operator map');
    },
    [rightData, v, proxValue, panelWidth, innerW, innerH],
  );

  // ─── Render ───

  return (
    <div ref={containerRef} className="w-full">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-4 text-sm">
        <label className="flex items-center gap-2">
          <span className="text-zinc-400">λ</span>
          <input
            type="range"
            min={0.1}
            max={5}
            step={0.1}
            value={lambda}
            onChange={(e) => setLambda(Number(e.target.value))}
            className="w-32"
          />
          <span className="text-zinc-300 font-mono w-10">{lambda.toFixed(1)}</span>
        </label>
        <select
          value={funcIdx}
          onChange={(e) => setFuncIdx(Number(e.target.value))}
          className="bg-zinc-800 text-zinc-200 border border-zinc-600 rounded px-2 py-1"
        >
          {FUNCTIONS.map((f, i) => (
            <option key={f.key} value={i}>
              {f.label}
            </option>
          ))}
        </select>
      </div>

      {/* Panels */}
      <div
        className="flex gap-4"
        style={{ flexDirection: isStacked ? 'column' : 'row' }}
      >
        <svg ref={leftRef} className="overflow-visible" />
        <svg ref={rightRef} className="overflow-visible" />
      </div>

      {/* Formula readout */}
      <p className="mt-3 text-xs text-zinc-400 font-mono">
        prox<sub>λf</sub>(v) = prox<sub>{lambda.toFixed(1)}·{selectedFunc.shortLabel}</sub>
        ({v.toFixed(1)}) = {proxValue.toFixed(3)}
      </p>
    </div>
  );
}

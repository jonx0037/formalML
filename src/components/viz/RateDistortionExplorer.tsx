import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { rateDistortionBinary, rateDistortionGaussian, binaryEntropy } from './shared/informationTheory';

const HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 50, left: 55 };

interface SourceOption {
  label: string;
  type: 'binary' | 'gaussian';
  param: number; // p for binary, sigma^2 for gaussian
}

const SOURCES: SourceOption[] = [
  { label: 'Binary uniform (p = 0.5)', type: 'binary', param: 0.5 },
  { label: 'Binary (p = 0.3)', type: 'binary', param: 0.3 },
  { label: 'Binary (p = 0.1)', type: 'binary', param: 0.1 },
  { label: 'Gaussian (σ² = 1)', type: 'gaussian', param: 1 },
  { label: 'Gaussian (σ² = 2)', type: 'gaussian', param: 2 },
];

function generateCurve(source: SourceOption, nPoints = 300) {
  const points: { D: number; R: number }[] = [];
  if (source.type === 'binary') {
    const p = source.param;
    const Dmax = Math.min(p, 1 - p);
    for (let i = 0; i <= nPoints; i++) {
      const D = (i / nPoints) * Dmax;
      const R = rateDistortionBinary(p, D);
      points.push({ D, R });
    }
  } else {
    const sigma2 = source.param;
    for (let i = 1; i <= nPoints; i++) {
      const D = (i / nPoints) * sigma2;
      const R = rateDistortionGaussian(sigma2, D);
      points.push({ D, R });
    }
  }
  return points;
}

function getSlope(source: SourceOption, D: number): number {
  if (source.type === 'binary') {
    const clampedD = Math.max(0.001, Math.min(D, Math.min(source.param, 1 - source.param) - 0.001));
    return Math.log2(clampedD / (1 - clampedD));
  } else {
    return -1 / (2 * Math.max(D, 0.001) * Math.LN2);
  }
}

function getR0(source: SourceOption): number {
  if (source.type === 'binary') return binaryEntropy(source.param);
  // Gaussian: R(D) → ∞ as D → 0, but we cap for display
  return rateDistortionGaussian(source.param, 0.01);
}

function getDmax(source: SourceOption): number {
  if (source.type === 'binary') return Math.min(source.param, 1 - source.param);
  return source.param;
}

export default function RateDistortionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [sourceIdx, setSourceIdx] = useState(0);
  const [operatingD, setOperatingD] = useState(0.15);

  const source = SOURCES[sourceIdx];
  const curve = useMemo(() => generateCurve(source), [source]);
  const Dmax = getDmax(source);

  // Clamp operating point when source changes
  useEffect(() => {
    setOperatingD(Dmax * 0.3);
  }, [sourceIdx, Dmax]);

  const isSmall = containerWidth < SM_BREAKPOINT;
  const totalWidth = Math.max(containerWidth, 300);
  const leftW = isSmall ? totalWidth : Math.floor(totalWidth * 0.55);
  const rightW = isSmall ? totalWidth : totalWidth - leftW;
  const svgHeight = isSmall ? HEIGHT * 2 + 40 : HEIGHT;

  const drawChart = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (totalWidth <= 0) return;

    // --- Left panel: R(D) curve ---
    const lW = leftW - MARGIN.left - MARGIN.right;
    const lH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const leftG = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const maxR = source.type === 'binary' ? binaryEntropy(source.param) * 1.1 : getR0(source) * 1.05;
    const xScale = d3.scaleLinear().domain([0, Dmax * 1.05]).range([0, lW]);
    const yScale = d3.scaleLinear().domain([0, maxR]).range([lH, 0]);

    // Achievable region (above curve) - green
    const areaAbove = d3.area<{ D: number; R: number }>()
      .x(d => xScale(d.D))
      .y0(d => yScale(d.R))
      .y1(0);
    leftG.append('path')
      .datum(curve)
      .attr('d', areaAbove)
      .style('fill', '#10b981')
      .style('fill-opacity', 0.08);

    // Unachievable region (below curve) - red
    const areaBelow = d3.area<{ D: number; R: number }>()
      .x(d => xScale(d.D))
      .y0(d => yScale(d.R))
      .y1(lH);
    leftG.append('path')
      .datum(curve)
      .attr('d', areaBelow)
      .style('fill', '#ef4444')
      .style('fill-opacity', 0.06);

    // R(D) curve
    const line = d3.line<{ D: number; R: number }>()
      .x(d => xScale(d.D))
      .y(d => yScale(d.R));
    leftG.append('path')
      .datum(curve)
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', '#0F6E56')
      .style('stroke-width', 2.5);

    // Axes
    leftG.append('g')
      .attr('transform', `translate(0,${lH})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    leftG.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    leftG.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    // Axis labels
    leftG.append('text')
      .attr('x', lW / 2).attr('y', lH + 40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-size', '12px')
      .text('Distortion D');
    leftG.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -lH / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-size', '12px')
      .text('Rate R (bits)');

    // Boundary annotations
    if (source.type === 'binary') {
      const R0 = binaryEntropy(source.param);
      leftG.append('circle')
        .attr('cx', xScale(0)).attr('cy', yScale(R0))
        .attr('r', 4).style('fill', '#0F6E56');
      leftG.append('text')
        .attr('x', xScale(0) + 8).attr('y', yScale(R0) - 8)
        .style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)')
        .text(`R(0) = H(X) = ${R0.toFixed(3)}`);
    }
    leftG.append('circle')
      .attr('cx', xScale(Dmax)).attr('cy', yScale(0))
      .attr('r', 4).style('fill', '#0F6E56');
    leftG.append('text')
      .attr('x', xScale(Dmax) - 8).attr('y', yScale(0) - 10)
      .attr('text-anchor', 'end')
      .style('font-size', '10px').style('fill', 'var(--color-text-secondary, #666)')
      .text(`R(D_max) = 0`);

    // Region labels
    leftG.append('text')
      .attr('x', xScale(Dmax * 0.25)).attr('y', yScale(maxR * 0.75))
      .attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', '#10b981').style('font-weight', '600')
      .text('Achievable');
    leftG.append('text')
      .attr('x', xScale(Dmax * 0.7)).attr('y', yScale(maxR * 0.15))
      .attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', '#ef4444').style('font-weight', '600')
      .text('Unachievable');

    // Operating point on curve
    const opR = source.type === 'binary'
      ? rateDistortionBinary(source.param, operatingD)
      : rateDistortionGaussian(source.param, operatingD);

    leftG.append('circle')
      .attr('cx', xScale(operatingD))
      .attr('cy', yScale(opR))
      .attr('r', 7)
      .style('fill', '#534AB7')
      .style('stroke', '#fff')
      .style('stroke-width', 2)
      .style('cursor', 'ew-resize');

    // --- Right panel: Tangent line detail ---
    const rOffset = isSmall ? 0 : leftW;
    const rYOffset = isSmall ? HEIGHT + 20 : 0;
    const rW = (isSmall ? totalWidth : rightW) - MARGIN.left - MARGIN.right;
    const rH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const rightG = svg.append('g')
      .attr('transform', `translate(${rOffset + MARGIN.left},${rYOffset + MARGIN.top})`);

    // Zoomed view around operating point
    const zoomD = Dmax * 0.3;
    const dLo = Math.max(0, operatingD - zoomD);
    const dHi = Math.min(Dmax * 1.05, operatingD + zoomD);
    const rLo = source.type === 'binary'
      ? rateDistortionBinary(source.param, dHi) : rateDistortionGaussian(source.param, dHi);
    const rHi = source.type === 'binary'
      ? rateDistortionBinary(source.param, Math.max(dLo, 0.001)) : rateDistortionGaussian(source.param, Math.max(dLo, 0.001));
    const rPad = (rHi - rLo) * 0.15;

    const xZ = d3.scaleLinear().domain([dLo, dHi]).range([0, rW]);
    const yZ = d3.scaleLinear().domain([Math.max(0, rLo - rPad), rHi + rPad]).range([rH, 0]);

    // Curve in zoom
    const zoomCurve = curve.filter(p => p.D >= dLo && p.D <= dHi);
    const lineZ = d3.line<{ D: number; R: number }>()
      .x(d => xZ(d.D))
      .y(d => yZ(d.R));
    rightG.append('path')
      .datum(zoomCurve)
      .attr('d', lineZ)
      .style('fill', 'none')
      .style('stroke', '#0F6E56')
      .style('stroke-width', 2);

    // Tangent line
    const slope = getSlope(source, operatingD);
    const tangentLen = zoomD * 0.8;
    const tD1 = operatingD - tangentLen;
    const tD2 = operatingD + tangentLen;
    const tR1 = opR + slope * (tD1 - operatingD);
    const tR2 = opR + slope * (tD2 - operatingD);

    rightG.append('line')
      .attr('x1', xZ(Math.max(dLo, tD1))).attr('y1', yZ(tR1))
      .attr('x2', xZ(Math.min(dHi, tD2))).attr('y2', yZ(tR2))
      .style('stroke', '#D97706')
      .style('stroke-width', 2)
      .style('stroke-dasharray', '6,3');

    // Operating point
    rightG.append('circle')
      .attr('cx', xZ(operatingD)).attr('cy', yZ(opR))
      .attr('r', 7)
      .style('fill', '#534AB7')
      .style('stroke', '#fff')
      .style('stroke-width', 2);

    // Axes
    rightG.append('g')
      .attr('transform', `translate(0,${rH})`)
      .call(d3.axisBottom(xZ).ticks(4))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    rightG.append('g')
      .call(d3.axisLeft(yZ).ticks(4))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    rightG.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    rightG.append('text')
      .attr('x', rW / 2).attr('y', rH + 40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-size', '12px')
      .text('Distortion D');

    // Readouts
    const lagrangian = opR + slope * operatingD;
    const readouts = [
      `D = ${operatingD.toFixed(3)}`,
      `R = ${opR.toFixed(3)} bits`,
      `slope s = ${slope.toFixed(3)}`,
      `L = R + sD = ${lagrangian.toFixed(3)}`,
    ];
    readouts.forEach((text, i) => {
      rightG.append('text')
        .attr('x', rW - 5).attr('y', 15 + i * 16)
        .attr('text-anchor', 'end')
        .style('font-size', '11px')
        .style('fill', i < 2 ? '#534AB7' : '#D97706')
        .style('font-weight', '500')
        .text(text);
    });

    // Title
    rightG.append('text')
      .attr('x', rW / 2).attr('y', -12)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-weight', '600')
      .text('Tangent at Operating Point');

    // Left panel title
    leftG.append('text')
      .attr('x', lW / 2).attr('y', -12)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', 'var(--color-text-secondary, #666)')
      .style('font-weight', '600')
      .text('Rate-Distortion Function R(D)');

  }, [containerWidth, source, curve, operatingD, Dmax, isSmall, totalWidth, leftW, rightW]);

  useEffect(() => {
    drawChart();
  }, [drawChart]);

  const handleSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setOperatingD(val);
  }, []);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Source:
          <select
            value={sourceIdx}
            onChange={e => setSourceIdx(Number(e.target.value))}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {SOURCES.map((s, i) => (
              <option key={i} value={i}>{s.label}</option>
            ))}
          </select>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Operating point D:
          <input
            type="range"
            min={0.001}
            max={Dmax - 0.001}
            step={0.001}
            value={operatingD}
            onChange={handleSlider}
            className="ml-2 w-32 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{operatingD.toFixed(3)}</span>
        </label>
      </div>
      <svg ref={svgRef} width={totalWidth} height={svgHeight} />
    </div>
  );
}

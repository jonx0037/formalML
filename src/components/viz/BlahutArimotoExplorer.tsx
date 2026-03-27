import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  entropy,
  blahutArimotoStep,
  blahutArimoto,
  hammingDistortionMatrix,
} from './shared/informationTheory';

const PANEL_HEIGHT = 260;
const BAR_HEIGHT = 200;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 15, bottom: 45, left: 50 };

interface SourcePreset {
  label: string;
  px: number[];
  symbols: string[];
}

const PRESETS: SourcePreset[] = [
  { label: 'Uniform ternary', px: [1 / 3, 1 / 3, 1 / 3], symbols: ['0', '1', '2'] },
  { label: 'Non-uniform [0.5, 0.3, 0.2]', px: [0.5, 0.3, 0.2], symbols: ['0', '1', '2'] },
  { label: 'Quaternary [0.4, 0.3, 0.2, 0.1]', px: [0.4, 0.3, 0.2, 0.1], symbols: ['0', '1', '2', '3'] },
];

function precomputeRDCurve(px: number[], nSlopes = 50) {
  const k = px.length;
  const dMatrix = hammingDistortionMatrix(k);
  const points: { rate: number; distortion: number }[] = [];
  for (let i = 0; i < nSlopes; i++) {
    const s = -20 + (i / (nSlopes - 1)) * 19.5; // -20 to -0.5
    const result = blahutArimoto(px, dMatrix, s);
    points.push({ rate: result.rate, distortion: result.distortion });
  }
  // Sort by distortion
  points.sort((a, b) => a.distortion - b.distortion);
  return points;
}

export default function BlahutArimotoExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const topSvgRef = useRef<SVGSVGElement>(null);
  const barSvgRef = useRef<SVGSVGElement>(null);

  const [presetIdx, setPresetIdx] = useState(1);
  const [slope, setSlope] = useState(-5);
  const [history, setHistory] = useState<{ rate: number; distortion: number }[]>([]);
  const [qXhat, setQXhat] = useState<number[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const runRef = useRef(false);
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const preset = PRESETS[presetIdx];
  const rdCurve = useMemo(() => precomputeRDCurve(preset.px), [preset.px]);

  // True R(D) for current slope
  const truePoint = useMemo(() => {
    const dMatrix = hammingDistortionMatrix(preset.px.length);
    return blahutArimoto(preset.px, dMatrix, slope);
  }, [preset.px, slope]);

  // Reset state when preset or slope changes
  useEffect(() => {
    const k = preset.px.length;
    setQXhat(new Array(k).fill(1 / k));
    setHistory([]);
    setIsRunning(false);
    runRef.current = false;
    if (animRef.current) clearTimeout(animRef.current);
  }, [presetIdx, slope, preset.px.length]);

  const doStep = useCallback(() => {
    setQXhat(prev => {
      const dMatrix = hammingDistortionMatrix(preset.px.length);
      const result = blahutArimotoStep(preset.px, dMatrix, prev, slope);
      setHistory(h => [...h, { rate: result.rate, distortion: result.distortion }]);
      return result.qXhat;
    });
  }, [preset.px, slope]);

  const handleStep = useCallback(() => {
    doStep();
  }, [doStep]);

  const handleRun = useCallback(() => {
    if (isRunning) {
      runRef.current = false;
      setIsRunning(false);
      if (animRef.current) clearTimeout(animRef.current);
      return;
    }
    setIsRunning(true);
    runRef.current = true;

    function tick() {
      if (!runRef.current) return;
      setQXhat(prev => {
        const dMatrix = hammingDistortionMatrix(preset.px.length);
        const result = blahutArimotoStep(preset.px, dMatrix, prev, slope);
        setHistory(h => {
          const newH = [...h, { rate: result.rate, distortion: result.distortion }];
          // Stop if converged
          if (newH.length > 1 && Math.abs(newH[newH.length - 1].rate - newH[newH.length - 2].rate) < 1e-8) {
            runRef.current = false;
            setIsRunning(false);
            return newH;
          }
          if (newH.length >= 100) {
            runRef.current = false;
            setIsRunning(false);
            return newH;
          }
          animRef.current = setTimeout(tick, 100);
          return newH;
        });
        return result.qXhat;
      });
    }
    tick();
  }, [isRunning, preset.px, slope]);

  const handleReset = useCallback(() => {
    const k = preset.px.length;
    setQXhat(new Array(k).fill(1 / k));
    setHistory([]);
    setIsRunning(false);
    runRef.current = false;
    if (animRef.current) clearTimeout(animRef.current);
  }, [preset.px.length]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      runRef.current = false;
      if (animRef.current) clearTimeout(animRef.current);
    };
  }, []);

  // Draw top panels (R(D) curve + convergence)
  useEffect(() => {
    const svg = d3.select(topSvgRef.current);
    svg.selectAll('*').remove();
    if (containerWidth <= 0) return;

    const isSmall = containerWidth < SM_BREAKPOINT;
    const panelW = isSmall ? containerWidth : Math.floor(containerWidth / 2);
    const w = panelW - MARGIN.left - MARGIN.right;
    const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;

    // --- Left: R(D) curve with current point ---
    const g1 = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const maxD = Math.max(...rdCurve.map(p => p.distortion), 0.1);
    const maxR = Math.max(...rdCurve.map(p => p.rate), entropy(preset.px)) * 1.1;

    const xS1 = d3.scaleLinear().domain([0, maxD * 1.05]).range([0, w]);
    const yS1 = d3.scaleLinear().domain([0, maxR]).range([h, 0]);

    // R(D) curve
    const line1 = d3.line<{ rate: number; distortion: number }>()
      .x(d => xS1(d.distortion)).y(d => yS1(d.rate));
    g1.append('path')
      .datum(rdCurve)
      .attr('d', line1)
      .style('fill', 'none')
      .style('stroke', '#0F6E56')
      .style('stroke-width', 2);

    // True target
    g1.append('circle')
      .attr('cx', xS1(truePoint.distortion)).attr('cy', yS1(truePoint.rate))
      .attr('r', 5).style('fill', 'none').style('stroke', '#D97706').style('stroke-width', 2);

    // Current BA estimate
    if (history.length > 0) {
      const last = history[history.length - 1];
      g1.append('circle')
        .attr('cx', xS1(last.distortion)).attr('cy', yS1(last.rate))
        .attr('r', 6).style('fill', '#534AB7').style('stroke', '#fff').style('stroke-width', 2);
    }

    g1.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xS1).ticks(4))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g1.append('g').call(d3.axisLeft(yS1).ticks(4))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g1.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g1.append('text').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', 'var(--color-text-secondary, #666)').text('Distortion D');
    g1.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -38)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', 'var(--color-text-secondary, #666)').text('Rate R (bits)');
    g1.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text('R(D) Curve');

    // --- Right: Convergence plot ---
    const rOffset = isSmall ? 0 : panelW;
    const rYOffset = isSmall ? PANEL_HEIGHT + 10 : 0;
    const g2 = svg.append('g')
      .attr('transform', `translate(${rOffset + MARGIN.left},${rYOffset + MARGIN.top})`);

    const maxIter = Math.max(history.length, 10);
    const xS2 = d3.scaleLinear().domain([0, maxIter]).range([0, w]);
    const rateRange = history.length > 0
      ? [Math.min(...history.map(h => h.rate), truePoint.rate) * 0.95,
         Math.max(...history.map(h => h.rate), truePoint.rate) * 1.05]
      : [0, maxR];
    const yS2 = d3.scaleLinear().domain(rateRange).range([h, 0]);

    // True rate dashed line
    g2.append('line')
      .attr('x1', 0).attr('y1', yS2(truePoint.rate))
      .attr('x2', w).attr('y2', yS2(truePoint.rate))
      .style('stroke', '#D97706').style('stroke-dasharray', '6,3').style('stroke-width', 1.5);
    g2.append('text')
      .attr('x', w - 5).attr('y', yS2(truePoint.rate) - 6)
      .attr('text-anchor', 'end')
      .style('font-size', '10px').style('fill', '#D97706')
      .text(`R* = ${truePoint.rate.toFixed(4)}`);

    // History line
    if (history.length > 1) {
      const convLine = d3.line<{ rate: number; distortion: number }>()
        .x((_, i) => xS2(i + 1)).y(d => yS2(d.rate));
      g2.append('path')
        .datum(history)
        .attr('d', convLine)
        .style('fill', 'none')
        .style('stroke', '#534AB7')
        .style('stroke-width', 2);
    }
    history.forEach((pt, i) => {
      g2.append('circle')
        .attr('cx', xS2(i + 1)).attr('cy', yS2(pt.rate))
        .attr('r', 3).style('fill', '#534AB7');
    });

    g2.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xS2).ticks(5).tickFormat(d3.format('d')))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g2.append('g').call(d3.axisLeft(yS2).ticks(4))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g2.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g2.append('text').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', 'var(--color-text-secondary, #666)').text('Iteration');
    g2.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -38)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', 'var(--color-text-secondary, #666)').text('Rate R (bits)');
    g2.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text('BA Convergence');

  }, [containerWidth, rdCurve, history, truePoint, preset.px]);

  // Draw bar chart
  useEffect(() => {
    const svg = d3.select(barSvgRef.current);
    svg.selectAll('*').remove();
    if (containerWidth <= 0 || qXhat.length === 0) return;

    const w = containerWidth - MARGIN.left - MARGIN.right;
    const h = BAR_HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleBand<number>()
      .domain(qXhat.map((_, i) => i))
      .range([0, w])
      .padding(0.3);
    const yScale = d3.scaleLinear()
      .domain([0, Math.max(...qXhat) * 1.2])
      .range([h, 0]);

    const colors = ['#0F6E56', '#534AB7', '#D97706', '#C62828'];
    g.selectAll('.bar')
      .data(qXhat)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', (_, i) => xScale(i)!)
      .attr('y', d => yScale(d))
      .attr('width', xScale.bandwidth())
      .attr('height', d => h - yScale(d))
      .style('fill', (_, i) => colors[i % colors.length])
      .style('rx', 3);

    // Value labels
    g.selectAll('.label')
      .data(qXhat)
      .join('text')
      .attr('class', 'label')
      .attr('x', (_, i) => xScale(i)! + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d) - 5)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', 'var(--color-text-secondary, #666)')
      .text(d => d.toFixed(3));

    g.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).tickFormat(i => `x̂=${i}`))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g.append('g').call(d3.axisLeft(yScale).ticks(4))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text(`Output Distribution q(x̂) — Iteration ${history.length}`);

  }, [containerWidth, qXhat, history.length]);

  const isSmall = containerWidth < SM_BREAKPOINT;
  const topHeight = isSmall ? PANEL_HEIGHT * 2 + 20 : PANEL_HEIGHT;

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Source:
          <select
            value={presetIdx}
            onChange={e => setPresetIdx(Number(e.target.value))}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {PRESETS.map((p, i) => (
              <option key={i} value={i}>{p.label}</option>
            ))}
          </select>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Slope s:
          <input
            type="range"
            min={-20}
            max={-0.5}
            step={0.1}
            value={slope}
            onChange={e => setSlope(parseFloat(e.target.value))}
            className="ml-2 w-28 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{slope.toFixed(1)}</span>
        </label>
      </div>
      <div className="mb-3 flex gap-2">
        <button
          onClick={handleStep}
          disabled={isRunning}
          className="rounded bg-purple-600 px-3 py-1 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50"
        >
          Step
        </button>
        <button
          onClick={handleRun}
          className="rounded bg-teal-600 px-3 py-1 text-sm font-medium text-white hover:bg-teal-700"
        >
          {isRunning ? 'Pause' : 'Run'}
        </button>
        <button
          onClick={handleReset}
          className="rounded bg-gray-500 px-3 py-1 text-sm font-medium text-white hover:bg-gray-600"
        >
          Reset
        </button>
        <span className="flex items-center text-xs text-gray-500 dark:text-gray-400">
          Iteration: {history.length}
        </span>
      </div>
      <svg ref={topSvgRef} width={containerWidth} height={topHeight} />
      <svg ref={barSvgRef} width={containerWidth} height={BAR_HEIGHT} />
    </div>
  );
}

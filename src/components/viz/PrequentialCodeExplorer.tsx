import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { logStar } from './shared/informationTheory';

const HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 50, left: 55 };
const N_OBS = 200;

type Scenario = 'change-point' | 'iid' | 'gradual-drift';

const SCENARIO_OPTIONS: { value: Scenario; label: string }[] = [
  { value: 'change-point', label: 'Change point detection' },
  { value: 'iid', label: 'i.i.d. Bernoulli' },
  { value: 'gradual-drift', label: 'Gradual drift' },
];

// Seeded LCG PRNG
function lcg(index: number, seed: number): number {
  let s = (seed * 2147483647 + index * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  return (s & 0x7fffffff) / 0x7fffffff;
}

function generateSequence(
  scenario: Scenario,
  pBefore: number,
  pAfter: number,
  changePointFrac: number,
  seed: number
): number[] {
  const seq: number[] = [];
  const cp = Math.floor(N_OBS * changePointFrac);
  for (let i = 0; i < N_OBS; i++) {
    let p: number;
    if (scenario === 'iid') {
      p = pBefore;
    } else if (scenario === 'change-point') {
      p = i < cp ? pBefore : pAfter;
    } else {
      // Gradual drift
      p = pBefore + (pAfter - pBefore) * (i / (N_OBS - 1));
    }
    seq.push(lcg(i, seed) < p ? 1 : 0);
  }
  return seq;
}

// Prequential plug-in code length for single Bernoulli
// Uses Laplace smoothing: theta_hat = (k+1)/(n+2)
function singleModelPrequential(seq: number[], upTo: number): number {
  let codeLen = 0;
  let ones = 0;
  for (let i = 0; i < upTo; i++) {
    const theta = (ones + 1) / (i + 2); // Laplace smoothing
    const p = seq[i] === 1 ? theta : 1 - theta;
    codeLen -= Math.log2(Math.max(p, 1e-15));
    ones += seq[i];
  }
  return codeLen;
}

// Prequential code for two-piece Bernoulli with known change point
function twoModelPrequential(seq: number[], upTo: number, changePoint: number): number {
  // Model overhead: encode that there are 2 segments + the change point location
  const overhead = logStar(2) + Math.log2(Math.max(seq.length, 1));
  let codeLen = overhead;
  let ones1 = 0, ones2 = 0;
  for (let i = 0; i < upTo; i++) {
    if (i < changePoint) {
      const theta = (ones1 + 1) / (i + 2);
      const p = seq[i] === 1 ? theta : 1 - theta;
      codeLen -= Math.log2(Math.max(p, 1e-15));
      ones1 += seq[i];
    } else {
      const count = i - changePoint;
      const theta = (ones2 + 1) / (count + 2);
      const p = seq[i] === 1 ? theta : 1 - theta;
      codeLen -= Math.log2(Math.max(p, 1e-15));
      ones2 += seq[i];
    }
  }
  return codeLen;
}

export default function PrequentialCodeExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [scenario, setScenario] = useState<Scenario>('change-point');
  const [changePointFrac, setChangePointFrac] = useState(0.5);
  const [pBefore, setPBefore] = useState(0.3);
  const [pAfter, setPAfter] = useState(0.7);
  const [currentStep, setCurrentStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [seed, setSeed] = useState(42);
  const runRef = useRef(false);
  const animRef = useRef<number | null>(null);
  const lastTickRef = useRef(0);

  const sequence = useMemo(
    () => generateSequence(scenario, pBefore, pAfter, changePointFrac, seed),
    [scenario, pBefore, pAfter, changePointFrac, seed]
  );

  const changePoint = Math.floor(N_OBS * changePointFrac);

  // Precompute all accumulated code lengths
  const codeLengths = useMemo(() => {
    const single: number[] = [0];
    const two: number[] = [0];
    for (let i = 1; i <= N_OBS; i++) {
      single.push(singleModelPrequential(sequence, i));
      two.push(twoModelPrequential(sequence, i, changePoint));
    }
    return { single, two };
  }, [sequence, changePoint]);

  // Running MLE
  const runningMLE = useMemo(() => {
    const mle: number[] = [];
    let ones = 0;
    for (let i = 0; i < N_OBS; i++) {
      ones += sequence[i];
      mle.push(ones / (i + 1));
    }
    return mle;
  }, [sequence]);

  // Find crossing point
  const crossingPoint = useMemo(() => {
    for (let i = 1; i <= N_OBS; i++) {
      if (codeLengths.two[i] < codeLengths.single[i]) return i;
    }
    return -1;
  }, [codeLengths]);

  // Reset when params change
  useEffect(() => {
    setCurrentStep(0);
    setIsRunning(false);
    runRef.current = false;
    if (animRef.current) cancelAnimationFrame(animRef.current);
  }, [scenario, pBefore, pAfter, changePointFrac, seed]);

  const handleStep = useCallback(() => {
    setCurrentStep(s => Math.min(s + 1, N_OBS));
  }, []);

  const handleRun = useCallback(() => {
    if (isRunning) {
      runRef.current = false;
      setIsRunning(false);
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    setIsRunning(true);
    runRef.current = true;
    lastTickRef.current = performance.now();
    function tick(now: number) {
      if (!runRef.current) return;
      if (now - lastTickRef.current >= 50) {
        lastTickRef.current = now;
        setCurrentStep(s => {
          if (s >= N_OBS) {
            runRef.current = false;
            setIsRunning(false);
            return s;
          }
          return s + 1;
        });
      }
      animRef.current = requestAnimationFrame(tick);
    }
    animRef.current = requestAnimationFrame(tick);
  }, [isRunning]);

  const handleReset = useCallback(() => {
    setSeed(s => s + 1);
    setCurrentStep(0);
    setIsRunning(false);
    runRef.current = false;
    if (animRef.current) cancelAnimationFrame(animRef.current);
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      runRef.current = false;
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  const isSmall = containerWidth < SM_BREAKPOINT;
  const totalWidth = Math.max(containerWidth, 300);
  const leftW = isSmall ? totalWidth : Math.floor(totalWidth * 0.55);
  const rightW = isSmall ? totalWidth : totalWidth - leftW;
  const svgHeight = isSmall ? HEIGHT * 2 + 40 : HEIGHT;

  const drawChart = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (totalWidth <= 0) return;

    const lW = leftW - MARGIN.left - MARGIN.right;
    const lH = HEIGHT - MARGIN.top - MARGIN.bottom;

    // --- Left panel: Accumulated code lengths ---
    const g1 = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain([0, N_OBS]).range([0, lW]);
    const maxLen = Math.max(
      codeLengths.single[N_OBS],
      codeLengths.two[N_OBS]
    ) * 1.05;
    const yScale = d3.scaleLinear().domain([0, maxLen]).range([lH, 0]);

    // Single model line
    const singleLine = d3.line<number>()
      .x((_, i) => xScale(i))
      .y(d => yScale(d));
    g1.append('path')
      .datum(codeLengths.single.slice(0, currentStep + 1))
      .attr('d', singleLine)
      .style('fill', 'none')
      .style('stroke', '#3b82f6')
      .style('stroke-width', 2);

    // Two-piece model line
    const twoLine = d3.line<number>()
      .x((_, i) => xScale(i))
      .y(d => yScale(d));
    g1.append('path')
      .datum(codeLengths.two.slice(0, currentStep + 1))
      .attr('d', twoLine)
      .style('fill', 'none')
      .style('stroke', '#7c3aed')
      .style('stroke-width', 2);

    // Full curves (faint, as guide)
    if (currentStep < N_OBS) {
      g1.append('path')
        .datum(codeLengths.single)
        .attr('d', singleLine)
        .style('fill', 'none')
        .style('stroke', '#3b82f6')
        .style('stroke-width', 1)
        .style('opacity', 0.15);
      g1.append('path')
        .datum(codeLengths.two)
        .attr('d', twoLine)
        .style('fill', 'none')
        .style('stroke', '#7c3aed')
        .style('stroke-width', 1)
        .style('opacity', 0.15);
    }

    // Current step marker
    if (currentStep > 0) {
      g1.append('circle')
        .attr('cx', xScale(currentStep))
        .attr('cy', yScale(codeLengths.single[currentStep]))
        .attr('r', 4).style('fill', '#3b82f6');
      g1.append('circle')
        .attr('cx', xScale(currentStep))
        .attr('cy', yScale(codeLengths.two[currentStep]))
        .attr('r', 4).style('fill', '#7c3aed');
    }

    // Change point line
    if (scenario !== 'iid') {
      g1.append('line')
        .attr('x1', xScale(changePoint)).attr('y1', 0)
        .attr('x2', xScale(changePoint)).attr('y2', lH)
        .style('stroke', '#ef4444').style('stroke-width', 1.5).style('stroke-dasharray', '4,3');
      g1.append('text')
        .attr('x', xScale(changePoint) + 4).attr('y', 12)
        .style('font-size', '9px').style('fill', '#ef4444')
        .text('change point');
    }

    // Crossing point annotation
    if (crossingPoint > 0 && crossingPoint <= currentStep) {
      g1.append('circle')
        .attr('cx', xScale(crossingPoint))
        .attr('cy', yScale(codeLengths.single[crossingPoint]))
        .attr('r', 6)
        .style('fill', 'none').style('stroke', '#10b981').style('stroke-width', 2);
      g1.append('text')
        .attr('x', xScale(crossingPoint) + 8)
        .attr('y', yScale(codeLengths.single[crossingPoint]) - 8)
        .style('font-size', '9px').style('fill', '#10b981').style('font-weight', '600')
        .text(`crossing at i = ${crossingPoint}`);
    }

    g1.append('g').attr('transform', `translate(0,${lH})`).call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g1.append('g').call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    g1.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    g1.append('text').attr('x', lW / 2).attr('y', lH + 40).attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Observation index i');
    g1.append('text').attr('transform', 'rotate(-90)').attr('x', -lH / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Accumulated code length (bits)');
    g1.append('text').attr('x', lW / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text('Prequential Code Length');

    // Legend
    g1.append('line').attr('x1', 4).attr('y1', 8).attr('x2', 24).attr('y2', 8)
      .style('stroke', '#3b82f6').style('stroke-width', 2);
    g1.append('text').attr('x', 28).attr('y', 12).style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)').text('Single Bernoulli');
    g1.append('line').attr('x1', 4).attr('y1', 22).attr('x2', 24).attr('y2', 22)
      .style('stroke', '#7c3aed').style('stroke-width', 2);
    g1.append('text').attr('x', 28).attr('y', 26).style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)').text('Two-piece Bernoulli');

    // --- Right panel: Observation strip + MLE ---
    const rOffset = isSmall ? 0 : leftW;
    const rYOffset = isSmall ? HEIGHT + 20 : 0;
    const rW = (isSmall ? totalWidth : rightW) - MARGIN.left - MARGIN.right;
    const rH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const g2 = svg.append('g')
      .attr('transform', `translate(${rOffset + MARGIN.left},${rYOffset + MARGIN.top})`);

    // Observation strip (top half)
    const stripH = 30;
    const cellW = Math.max(1, rW / N_OBS);
    for (let i = 0; i < Math.min(currentStep, N_OBS); i++) {
      g2.append('rect')
        .attr('x', i * cellW).attr('y', 0)
        .attr('width', cellW).attr('height', stripH)
        .style('fill', sequence[i] === 1 ? '#1e293b' : '#e2e8f0')
        .style('stroke', 'var(--color-text-secondary, #ccc)')
        .style('stroke-width', cellW > 3 ? 0.5 : 0);
    }
    g2.append('text').attr('x', -4).attr('y', stripH / 2 + 4)
      .attr('text-anchor', 'end')
      .style('font-size', '9px').style('fill', 'var(--color-text-secondary, #666)').text('x');

    // Change point marker on strip
    if (scenario !== 'iid') {
      g2.append('line')
        .attr('x1', changePoint * cellW).attr('y1', 0)
        .attr('x2', changePoint * cellW).attr('y2', stripH)
        .style('stroke', '#ef4444').style('stroke-width', 2);
    }

    // Running MLE (bottom portion)
    const mleTop = stripH + 20;
    const mleH = rH - mleTop - 10;
    const xMLE = d3.scaleLinear().domain([0, N_OBS]).range([0, rW]);
    const yMLE = d3.scaleLinear().domain([0, 1]).range([mleH, 0]);

    const mleLine = d3.line<number>()
      .x((_, i) => xMLE(i))
      .y(d => yMLE(d));
    g2.append('path')
      .datum(runningMLE.slice(0, currentStep))
      .attr('d', mleLine)
      .attr('transform', `translate(0,${mleTop})`)
      .style('fill', 'none')
      .style('stroke', '#0F6E56')
      .style('stroke-width', 1.5);

    // True parameters as horizontal lines
    if (scenario === 'iid') {
      g2.append('line')
        .attr('x1', 0).attr('y1', mleTop + yMLE(pBefore))
        .attr('x2', rW).attr('y2', mleTop + yMLE(pBefore))
        .style('stroke', '#f59e0b').style('stroke-width', 1).style('stroke-dasharray', '4,3');
    } else if (scenario === 'change-point') {
      g2.append('line')
        .attr('x1', 0).attr('y1', mleTop + yMLE(pBefore))
        .attr('x2', xMLE(changePoint)).attr('y2', mleTop + yMLE(pBefore))
        .style('stroke', '#f59e0b').style('stroke-width', 1).style('stroke-dasharray', '4,3');
      g2.append('line')
        .attr('x1', xMLE(changePoint)).attr('y1', mleTop + yMLE(pAfter))
        .attr('x2', rW).attr('y2', mleTop + yMLE(pAfter))
        .style('stroke', '#f59e0b').style('stroke-width', 1).style('stroke-dasharray', '4,3');
    }

    const mleG = g2.append('g').attr('transform', `translate(0,${mleTop})`);
    mleG.append('g').attr('transform', `translate(0,${mleH})`).call(d3.axisBottom(xMLE).ticks(5))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    mleG.append('g').call(d3.axisLeft(yMLE).ticks(4))
      .selectAll('text').style('fill', 'var(--color-text-secondary, #666)');
    mleG.selectAll('.domain, .tick line').style('stroke', 'var(--color-text-secondary, #999)');

    mleG.append('text').attr('x', rW / 2).attr('y', mleH + 40).attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Observation index');
    mleG.append('text').attr('transform', 'rotate(-90)').attr('x', -mleH / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary, #666)').style('font-size', '12px').text('Running θ̂');

    g2.append('text').attr('x', rW / 2).attr('y', -12).attr('text-anchor', 'middle')
      .style('font-size', '12px').style('fill', 'var(--color-text-secondary, #666)').style('font-weight', '600')
      .text('Observation Sequence & MLE');

  }, [containerWidth, currentStep, sequence, codeLengths, runningMLE, changePoint, crossingPoint, scenario, pBefore, pAfter, isSmall, totalWidth, leftW, rightW]);

  useEffect(() => {
    drawChart();
  }, [drawChart]);

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Scenario:
          <select
            value={scenario}
            onChange={e => setScenario(e.target.value as Scenario)}
            className="ml-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm text-gray-900 dark:text-gray-100"
          >
            {SCENARIO_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
        {scenario !== 'iid' && (
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Change at:
            <input
              type="range" min={0.2} max={0.8} step={0.05} value={changePointFrac}
              onChange={e => setChangePointFrac(Number(e.target.value))}
              className="ml-2 w-20 align-middle"
            />
            <span className="ml-1 text-xs font-mono">{Math.round(changePointFrac * 100)}%</span>
          </label>
        )}
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          p₁:
          <input
            type="range" min={0.05} max={0.95} step={0.05} value={pBefore}
            onChange={e => setPBefore(Number(e.target.value))}
            className="ml-1 w-20 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{pBefore.toFixed(2)}</span>
        </label>
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          p₂:
          <input
            type="range" min={0.05} max={0.95} step={0.05} value={pAfter}
            onChange={e => setPAfter(Number(e.target.value))}
            className="ml-1 w-20 align-middle"
          />
          <span className="ml-1 text-xs font-mono">{pAfter.toFixed(2)}</span>
        </label>
      </div>
      <div className="mb-3 flex items-center gap-3">
        <button
          onClick={handleStep}
          disabled={currentStep >= N_OBS}
          className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-1 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-40"
        >
          Step
        </button>
        <button
          onClick={handleRun}
          className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-1 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          {isRunning ? 'Pause' : 'Run'}
        </button>
        <button
          onClick={handleReset}
          className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-1 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          Reset
        </button>
        <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
          i = {currentStep} / {N_OBS}
        </span>
      </div>
      <svg ref={svgRef} width={totalWidth} height={svgHeight} />
    </div>
  );
}

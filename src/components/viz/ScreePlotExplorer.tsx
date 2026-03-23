import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { screeDatasets } from '../../data/pca-scree-data';

const MARGIN = { top: 20, right: 16, bottom: 32, left: 40 };
const SVG_HEIGHT = 300;
const MOBILE_BREAKPOINT = 640;

/** Find the elbow index: maximize second discrete derivative of eigenvalues. */
function detectElbow(eigenvalues: number[]): number {
  if (eigenvalues.length < 3) return 0;
  let bestIdx = 1;
  let bestScore = -Infinity;
  for (let i = 1; i < eigenvalues.length - 1; i++) {
    const score =
      (eigenvalues[i - 1] - eigenvalues[i]) -
      (eigenvalues[i] - eigenvalues[i + 1]);
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/** Count eigenvalues exceeding their parallel analysis threshold. */
function parallelK(eigenvalues: number[], pa95: number[]): number {
  let k = 0;
  for (let i = 0; i < eigenvalues.length; i++) {
    if (eigenvalues[i] > pa95[i]) k = i + 1;
    else break;
  }
  return k;
}

export default function ScreePlotExplorer() {
  const { ref, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetIdx, setDatasetIdx] = useState(0);
  const [selectedK, setSelectedK] = useState<number | null>(null);

  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const dataset = screeDatasets[datasetIdx];
  const { eigenvalues, cumulativeEVR, parallelAnalysis95th, kaiserK, threshold95K, nFeatures } = dataset;

  const elbowIdx = useMemo(() => detectElbow(eigenvalues), [eigenvalues]);
  const paK = useMemo(() => parallelK(eigenvalues, parallelAnalysis95th), [eigenvalues, parallelAnalysis95th]);

  // Initialize selectedK when dataset changes
  useEffect(() => {
    setSelectedK(null);
  }, [datasetIdx]);

  const k = selectedK ?? elbowIdx + 1;

  const isMobile = (containerWidth || 800) < MOBILE_BREAKPOINT;
  const totalWidth = containerWidth || 800;
  const panelWidth = isMobile ? totalWidth : Math.floor((totalWidth - 16) / 2);
  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  const handleDatasetChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setDatasetIdx(Number(e.target.value));
  }, []);

  const handleBarClick = useCallback((i: number) => {
    setSelectedK(i + 1);
  }, []);

  // ─── Left panel: Eigenvalue bar chart ───
  useEffect(() => {
    const svg = d3.select(leftSvgRef.current);
    if (!leftSvgRef.current || innerW <= 0) return;
    svg.selectAll('*').remove();

    const xScale = d3
      .scaleBand<number>()
      .domain(d3.range(nFeatures))
      .range([MARGIN.left, MARGIN.left + innerW])
      .padding(0.15);

    const yMax = Math.max(d3.max(eigenvalues) ?? 1, d3.max(parallelAnalysis95th) ?? 1) * 1.1;
    const yScale = d3
      .scaleLinear()
      .domain([0, yMax])
      .range([MARGIN.top + innerH, MARGIN.top]);

    const g = svg.append('g');

    // Grid lines
    const yTicks = yScale.ticks(5);
    yTicks.forEach((t) => {
      g.append('line')
        .attr('x1', MARGIN.left)
        .attr('x2', MARGIN.left + innerW)
        .attr('y1', yScale(t))
        .attr('y2', yScale(t))
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', 0.3);
    });

    // Bars
    g.selectAll('.bar')
      .data(eigenvalues)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', (_, i) => xScale(i)!)
      .attr('y', (d) => yScale(d))
      .attr('width', xScale.bandwidth())
      .attr('height', (d) => MARGIN.top + innerH - yScale(d))
      .style('fill', (_, i) => (i < k ? 'steelblue' : '#94a3b8'))
      .style('fill-opacity', (_, i) => (i < k ? 0.85 : 0.4))
      .style('cursor', 'pointer')
      .each(function (_, i) {
        d3.select(this).on('click', () => handleBarClick(i));
      });

    // Kaiser criterion: horizontal line at lambda = 1
    if (yMax >= 1) {
      g.append('line')
        .attr('x1', MARGIN.left)
        .attr('x2', MARGIN.left + innerW)
        .attr('y1', yScale(1))
        .attr('y2', yScale(1))
        .style('stroke', '#22c55e')
        .style('stroke-width', 1.5)
        .style('stroke-dasharray', '6,3');

      g.append('text')
        .attr('x', MARGIN.left + innerW - 2)
        .attr('y', yScale(1) - 4)
        .attr('text-anchor', 'end')
        .style('font-size', '8px')
        .style('fill', '#22c55e')
        .style('font-family', 'var(--font-sans)')
        .text('Kaiser (λ=1)');
    }

    // Parallel analysis: 95th percentile line
    const paLine = d3
      .line<number>()
      .x((_, i) => xScale(i)! + xScale.bandwidth() / 2)
      .y((d) => yScale(d));

    g.append('path')
      .datum(parallelAnalysis95th)
      .attr('d', paLine)
      .style('fill', 'none')
      .style('stroke', '#ef4444')
      .style('stroke-width', 1.5)
      .style('stroke-dasharray', '4,3');

    g.selectAll('.pa-dot')
      .data(parallelAnalysis95th)
      .join('circle')
      .attr('cx', (_, i) => xScale(i)! + xScale.bandwidth() / 2)
      .attr('cy', (d) => yScale(d))
      .attr('r', 2.5)
      .style('fill', '#ef4444');

    g.append('text')
      .attr('x', MARGIN.left + innerW - 2)
      .attr('y', yScale(parallelAnalysis95th[0]) - 4)
      .attr('text-anchor', 'end')
      .style('font-size', '8px')
      .style('fill', '#ef4444')
      .style('font-family', 'var(--font-sans)')
      .text('Parallel analysis 95th');

    // Elbow marker
    g.append('circle')
      .attr('cx', xScale(elbowIdx)! + xScale.bandwidth() / 2)
      .attr('cy', yScale(eigenvalues[elbowIdx]))
      .attr('r', 8)
      .style('fill', 'none')
      .style('stroke', '#f59e0b')
      .style('stroke-width', 2);

    g.append('text')
      .attr('x', xScale(elbowIdx)! + xScale.bandwidth() / 2 + 10)
      .attr('y', yScale(eigenvalues[elbowIdx]) - 2)
      .style('font-size', '8px')
      .style('fill', '#f59e0b')
      .style('font-family', 'var(--font-sans)')
      .text('Elbow');

    // X-axis tick labels
    const maxLabels = innerW > 300 ? nFeatures : Math.min(nFeatures, 10);
    const step = Math.ceil(nFeatures / maxLabels);
    d3.range(0, nFeatures, step).forEach((i) => {
      g.append('text')
        .attr('x', xScale(i)! + xScale.bandwidth() / 2)
        .attr('y', MARGIN.top + innerH + 14)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px')
        .style('fill', 'var(--color-text)')
        .style('opacity', 0.5)
        .style('font-family', 'var(--font-sans)')
        .text(i + 1);
    });

    // Y-axis tick labels
    yTicks.forEach((t) => {
      g.append('text')
        .attr('x', MARGIN.left - 4)
        .attr('y', yScale(t) + 3)
        .attr('text-anchor', 'end')
        .style('font-size', '9px')
        .style('fill', 'var(--color-text)')
        .style('opacity', 0.5)
        .style('font-family', 'var(--font-sans)')
        .text(t);
    });

    // Axis labels
    g.append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', MARGIN.top + innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.7)
      .style('font-family', 'var(--font-sans)')
      .text('Component');

    g.append('text')
      .attr('x', 12)
      .attr('y', MARGIN.top + innerH / 2)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, 12, ${MARGIN.top + innerH / 2})`)
      .style('font-size', '10px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.7)
      .style('font-family', 'var(--font-sans)')
      .text('Eigenvalue');
  }, [eigenvalues, parallelAnalysis95th, nFeatures, innerW, innerH, k, elbowIdx, handleBarClick]);

  // ─── Right panel: Cumulative EVR curve ───
  useEffect(() => {
    const svg = d3.select(rightSvgRef.current);
    if (!rightSvgRef.current || innerW <= 0) return;
    svg.selectAll('*').remove();

    const xScale = d3
      .scaleLinear()
      .domain([1, nFeatures])
      .range([MARGIN.left, MARGIN.left + innerW]);

    const yScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([MARGIN.top + innerH, MARGIN.top]);

    const g = svg.append('g');

    // Grid lines
    const yTicks = [0, 0.25, 0.5, 0.75, 0.95, 1.0];
    yTicks.forEach((t) => {
      g.append('line')
        .attr('x1', MARGIN.left)
        .attr('x2', MARGIN.left + innerW)
        .attr('y1', yScale(t))
        .attr('y2', yScale(t))
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', 0.3);
    });

    // Shaded area under curve up to selected k
    const areaData = cumulativeEVR.slice(0, k).map((d, i) => ({ i: i + 1, d }));
    const areaGen = d3
      .area<{ i: number; d: number }>()
      .x((d) => xScale(d.i))
      .y0(MARGIN.top + innerH)
      .y1((d) => yScale(d.d));

    g.append('path')
      .datum(areaData)
      .attr('d', areaGen)
      .style('fill', 'steelblue')
      .style('fill-opacity', 0.15);

    // Cumulative EVR line
    const lineGen = d3
      .line<number>()
      .x((_, i) => xScale(i + 1))
      .y((d) => yScale(d));

    g.append('path')
      .datum(cumulativeEVR)
      .attr('d', lineGen)
      .style('fill', 'none')
      .style('stroke', 'steelblue')
      .style('stroke-width', 2);

    // Dots
    g.selectAll('.evr-dot')
      .data(cumulativeEVR)
      .join('circle')
      .attr('cx', (_, i) => xScale(i + 1))
      .attr('cy', (d) => yScale(d))
      .attr('r', 3)
      .style('fill', (_, i) => (i < k ? 'steelblue' : '#94a3b8'))
      .style('fill-opacity', (_, i) => (i < k ? 1 : 0.5));

    // 95% threshold line
    g.append('line')
      .attr('x1', MARGIN.left)
      .attr('x2', MARGIN.left + innerW)
      .attr('y1', yScale(0.95))
      .attr('y2', yScale(0.95))
      .style('stroke', '#ef4444')
      .style('stroke-width', 1.5)
      .style('stroke-dasharray', '6,3');

    g.append('text')
      .attr('x', MARGIN.left + innerW - 2)
      .attr('y', yScale(0.95) - 4)
      .attr('text-anchor', 'end')
      .style('font-size', '8px')
      .style('fill', '#ef4444')
      .style('font-family', 'var(--font-sans)')
      .text('95% variance');

    // Vertical line at selected k
    g.append('line')
      .attr('x1', xScale(k))
      .attr('x2', xScale(k))
      .attr('y1', MARGIN.top)
      .attr('y2', MARGIN.top + innerH)
      .style('stroke', 'steelblue')
      .style('stroke-width', 1.5)
      .style('stroke-dasharray', '4,3')
      .style('stroke-opacity', 0.6);

    // X-axis tick labels
    const maxLabels = innerW > 300 ? nFeatures : Math.min(nFeatures, 10);
    const step = Math.ceil(nFeatures / maxLabels);
    d3.range(1, nFeatures + 1, step).forEach((i) => {
      g.append('text')
        .attr('x', xScale(i))
        .attr('y', MARGIN.top + innerH + 14)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px')
        .style('fill', 'var(--color-text)')
        .style('opacity', 0.5)
        .style('font-family', 'var(--font-sans)')
        .text(i);
    });

    // Y-axis tick labels
    yTicks.forEach((t) => {
      g.append('text')
        .attr('x', MARGIN.left - 4)
        .attr('y', yScale(t) + 3)
        .attr('text-anchor', 'end')
        .style('font-size', '9px')
        .style('fill', 'var(--color-text)')
        .style('opacity', 0.5)
        .style('font-family', 'var(--font-sans)')
        .text(`${Math.round(t * 100)}%`);
    });

    // Axis labels
    g.append('text')
      .attr('x', MARGIN.left + innerW / 2)
      .attr('y', MARGIN.top + innerH + 28)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.7)
      .style('font-family', 'var(--font-sans)')
      .text('Components (k)');

    g.append('text')
      .attr('x', 12)
      .attr('y', MARGIN.top + innerH / 2)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, 12, ${MARGIN.top + innerH / 2})`)
      .style('font-size', '10px')
      .style('fill', 'var(--color-text)')
      .style('opacity', 0.7)
      .style('font-family', 'var(--font-sans)')
      .text('Cumulative EVR');
  }, [cumulativeEVR, nFeatures, innerW, innerH, k]);

  const evrAtK = k <= cumulativeEVR.length ? cumulativeEVR[k - 1] : cumulativeEVR[cumulativeEVR.length - 1];

  return (
    <div ref={ref} className="w-full space-y-3">
      {/* Dataset selector */}
      <div className="flex items-center gap-3">
        <label
          className="text-xs font-medium whitespace-nowrap"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Dataset
        </label>
        <select
          value={datasetIdx}
          onChange={handleDatasetChange}
          className="text-xs rounded border px-2 py-1"
          style={{
            fontFamily: 'var(--font-sans)',
            borderColor: 'var(--color-border)',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
          }}
        >
          {screeDatasets.map((ds, i) => (
            <option key={ds.name} value={i}>
              {ds.name}
            </option>
          ))}
        </select>
      </div>

      {/* Two-panel chart area */}
      <div
        className="flex gap-4"
        style={{ flexDirection: isMobile ? 'column' : 'row' }}
      >
        {/* Left: Eigenvalue bar chart */}
        <div className="flex-1 min-w-0">
          <p
            className="text-xs font-medium mb-1 opacity-70"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Eigenvalue spectrum
          </p>
          <svg
            ref={leftSvgRef}
            width={panelWidth}
            height={SVG_HEIGHT}
            className="rounded-lg border"
            style={{ borderColor: 'var(--color-border)' }}
          />
        </div>

        {/* Right: Cumulative EVR curve */}
        <div className="flex-1 min-w-0">
          <p
            className="text-xs font-medium mb-1 opacity-70"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            Cumulative explained variance
          </p>
          <svg
            ref={rightSvgRef}
            width={panelWidth}
            height={SVG_HEIGHT}
            className="rounded-lg border"
            style={{ borderColor: 'var(--color-border)' }}
          />
        </div>
      </div>

      {/* Readout */}
      <div
        className="text-xs opacity-80 space-y-1"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <p>
          <strong>Selected k = {k}</strong> &mdash; explains{' '}
          <strong>{(evrAtK * 100).toFixed(1)}%</strong> of total variance.
          <span className="opacity-60 ml-1">(Click a bar to change k.)</span>
        </p>
        <p className="opacity-60">
          Criteria comparison: Kaiser = {kaiserK} &nbsp;|&nbsp; Parallel analysis
          = {paK} &nbsp;|&nbsp; Elbow = {elbowIdx + 1} &nbsp;|&nbsp; 95%
          threshold = {threshold95K}
        </p>
      </div>
    </div>
  );
}

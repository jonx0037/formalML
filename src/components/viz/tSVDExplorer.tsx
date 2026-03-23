import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { getTSVDResults, type tSVDResult } from '../../data/tensor-decompositions-data';

// ─── Layout ───

const SM_BREAKPOINT = 640;
const CHART_HEIGHT = 240;
const MARGIN = { top: 28, right: 16, bottom: 36, left: 44 };

// ─── Component ───

export default function TSVDExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const curveRef = useRef<SVGSVGElement>(null);
  const barRef = useRef<SVGSVGElement>(null);

  const [tubalRank, setTubalRank] = useState(3);

  const results = useMemo(() => getTSVDResults(), []);
  const maxRank = results.length; // 10

  const currentResult = useMemo(
    () => results.find((r) => r.tubalRank === tubalRank) ?? results[0],
    [results, tubalRank],
  );

  // Singular values for the first Fourier slice at maximum tubal rank
  const fullSpectrumSlice0 = useMemo(() => {
    const lastResult = results[results.length - 1];
    return lastResult.fourierSingularValues[0] ?? [];
  }, [results]);

  // ─── Panel widths ───

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 280;
    if (containerWidth >= SM_BREAKPOINT) {
      return Math.floor((containerWidth - 16) / 2);
    }
    return containerWidth;
  }, [containerWidth]);

  // ─── Left panel: Eckart-Young curve ───

  useEffect(() => {
    if (!curveRef.current || panelWidth === 0) return;

    const svg = d3.select(curveRef.current);
    svg.selectAll('*').remove();

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear().domain([1, maxRank]).range([0, w]);
    const yExtent = d3.extent(results, (d) => d.relativeError) as [number, number];
    const yScale = d3.scaleLinear()
      .domain([0, yExtent[1] * 1.1])
      .range([h, 0]);

    // Grid lines
    const yAxis = d3.axisLeft(yScale).ticks(5).tickSize(-w);
    g.append('g')
      .call(yAxis)
      .call((sel) => sel.select('.domain').remove())
      .call((sel) => sel.selectAll('.tick line').style('stroke', 'var(--color-muted)').attr('opacity', 0.2))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').attr('font-size', 9));

    const xAxis = d3.axisBottom(xScale).ticks(maxRank).tickFormat(d3.format('d'));
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(xAxis)
      .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').attr('font-size', 9));

    // Line
    const line = d3.line<tSVDResult>()
      .x((d) => xScale(d.tubalRank))
      .y((d) => yScale(d.relativeError));

    g.append('path')
      .datum(results)
      .attr('fill', 'none')
      .style('stroke', 'var(--color-accent)')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Circles (all points)
    g.selectAll('.dot')
      .data(results)
      .enter()
      .append('circle')
      .attr('cx', (d) => xScale(d.tubalRank))
      .attr('cy', (d) => yScale(d.relativeError))
      .attr('r', (d) => (d.tubalRank === tubalRank ? 6 : 4))
      .style('fill', (d) =>
        d.tubalRank === tubalRank ? 'var(--color-accent)' : 'var(--color-surface)',
      )
      .style('stroke', 'var(--color-accent)')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (_, d) => setTubalRank(d.tubalRank));

    // Axis labels
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 30)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .text('Tubal rank k');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2)
      .attr('y', -32)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .text('Relative error');

    // Title
    g.append('text')
      .attr('x', w / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text('Eckart–Young optimality curve');
  }, [tubalRank, panelWidth, results, maxRank]);

  // ─── Right panel: Singular value spectrum ───

  useEffect(() => {
    if (!barRef.current || panelWidth === 0) return;

    const svg = d3.select(barRef.current);
    svg.selectAll('*').remove();

    const w = panelWidth - MARGIN.left - MARGIN.right;
    const h = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const svs = fullSpectrumSlice0;
    if (svs.length === 0) return;

    const xScale = d3.scaleBand<number>()
      .domain(svs.map((_, i) => i))
      .range([0, w])
      .padding(0.15);

    const yMax = d3.max(svs) ?? 1;
    const yScale = d3.scaleLinear()
      .domain([0, yMax * 1.05])
      .range([h, 0]);

    // Grid lines
    const yAxis = d3.axisLeft(yScale).ticks(5).tickSize(-w);
    g.append('g')
      .call(yAxis)
      .call((sel) => sel.select('.domain').remove())
      .call((sel) => sel.selectAll('.tick line').style('stroke', 'var(--color-muted)').attr('opacity', 0.2))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').attr('font-size', 9));

    const xAxis = d3.axisBottom(xScale).tickFormat((d) => `${Number(d) + 1}`);
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(xAxis)
      .call((sel) => sel.select('.domain').style('stroke', 'var(--color-border)'))
      .call((sel) => sel.selectAll('.tick text').style('fill', 'var(--color-muted)').attr('font-size', 9));

    // Bars
    svs.forEach((s, i) => {
      g.append('rect')
        .attr('x', xScale(i)!)
        .attr('y', yScale(s))
        .attr('width', xScale.bandwidth())
        .attr('height', h - yScale(s))
        .attr('rx', 1)
        .style('fill', i < tubalRank ? 'var(--color-accent)' : 'var(--color-muted)')
        .attr('opacity', i < tubalRank ? 0.9 : 0.3);
    });

    // Rank divider
    if (tubalRank < svs.length) {
      const xPos = xScale(tubalRank)! - xScale.step() * xScale.paddingInner() / 2;
      g.append('line')
        .attr('x1', xPos).attr('y1', 0)
        .attr('x2', xPos).attr('y2', h)
        .attr('stroke', '#ef4444')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3');
    }

    // Axis labels
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 30)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .text('Singular value index');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2)
      .attr('y', -32)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .text('Magnitude');

    // Title
    g.append('text')
      .attr('x', w / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text(`Fourier slice 0 — singular values (k = ${tubalRank})`);
  }, [tubalRank, panelWidth, fullSpectrumSlice0]);

  // ─── Slider handler ───

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => setTubalRank(parseInt(e.target.value)),
    [],
  );

  // ─── Render ───

  return (
    <div ref={containerRef} className="w-full space-y-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        {/* Eckart-Young curve */}
        <svg
          ref={curveRef}
          width={panelWidth}
          height={CHART_HEIGHT}
          className="shrink-0 rounded-lg border border-[var(--color-border)]"
        />

        {/* Singular value spectrum */}
        <svg
          ref={barRef}
          width={panelWidth}
          height={CHART_HEIGHT}
          className="shrink-0 rounded-lg border border-[var(--color-border)]"
        />
      </div>

      {/* Relative error readout */}
      <div
        className="flex flex-wrap gap-x-6 gap-y-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-xs"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <span>
          Tubal rank k = <strong>{tubalRank}</strong>
        </span>
        <span>
          Relative error = <strong>{currentResult.relativeError.toFixed(4)}</strong>
        </span>
      </div>

      {/* Tubal rank slider */}
      <div className="flex items-center gap-3">
        <label
          className="min-w-[120px] whitespace-nowrap text-xs font-medium"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Tubal rank k = {tubalRank}
        </label>
        <input
          type="range"
          min={1}
          max={maxRank}
          step={1}
          value={tubalRank}
          onChange={handleSliderChange}
          className="w-full accent-[var(--color-accent)]"
        />
      </div>
    </div>
  );
}

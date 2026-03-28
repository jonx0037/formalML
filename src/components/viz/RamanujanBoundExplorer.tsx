import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  randomRegularGraph, adjacencySpectrum, alonBoppanaBound, createRng,
  type Graph,
} from './shared/graphTheory';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HISTOGRAM_HEIGHT = 340;
const BOXPLOT_HEIGHT = 200;
const LINEPLOT_HEIGHT = 180;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 20, bottom: 40, left: 50 };

const DEGREES = [3, 4, 5, 6] as const;
const ENSEMBLE_SIZES = [50, 100, 200] as const;
const BOXPLOT_NS = [10, 20, 30, 40, 50];

const TEAL = '#14b8a6';
const AMBER = '#f59e0b';
const RED = '#ef4444';
const TEAL_LIGHT = 'rgba(20, 184, 166, 0.25)';

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

interface BoxStats {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
}

function boxplotStats(values: number[]): BoxStats {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    min: sorted[0],
    q1: quantile(sorted, 0.25),
    median: quantile(sorted, 0.5),
    q3: quantile(sorted, 0.75),
    max: sorted[sorted.length - 1],
  };
}

// ---------------------------------------------------------------------------
// Ensemble generation
// ---------------------------------------------------------------------------

function generateEnsemble(d: number, n: number, count: number): number[] {
  const lambda2s: number[] = [];
  for (let i = 0; i < count; i++) {
    const g = randomRegularGraph(n, d, 1000 + i);
    const spectrum = adjacencySpectrum(g);
    // spectrum is sorted descending — lambda2 is index 1
    lambda2s.push(spectrum[1]);
  }
  return lambda2s;
}

// Pre-computed data for all n values at a given (d, ensembleSize)
interface PrecomputedData {
  d: number;
  ensembleSize: number;
  // Map from n -> array of lambda2 values
  byN: Map<number, number[]>;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function RamanujanBoundExplorer() {
  // Responsive container
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isNarrow = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // SVG refs
  const histSvgRef = useRef<SVGSVGElement>(null);
  const boxSvgRef = useRef<SVGSVGElement>(null);
  const lineSvgRef = useRef<SVGSVGElement>(null);

  // State
  const [degree, setDegree] = useState<number>(3);
  const [nSize, setNSize] = useState<number>(30);
  const [ensembleSize, setEnsembleSize] = useState<number>(100);
  const [computing, setComputing] = useState(true);

  // Pre-computed data ref
  const precomputedRef = useRef<PrecomputedData | null>(null);
  const [precomputed, setPrecomputed] = useState<PrecomputedData | null>(null);

  // =========================================================================
  // Pre-compute all data for the right panel (all n values)
  // =========================================================================

  useEffect(() => {
    setComputing(true);

    // Use requestAnimationFrame to let the UI update with "Computing..." first
    const rafId = requestAnimationFrame(() => {
      const byN = new Map<number, number[]>();

      for (const n of BOXPLOT_NS) {
        byN.set(n, generateEnsemble(degree, n, ensembleSize));
      }

      // Also generate for the current nSize if not in BOXPLOT_NS
      if (!BOXPLOT_NS.includes(nSize)) {
        byN.set(nSize, generateEnsemble(degree, nSize, ensembleSize));
      }

      const data: PrecomputedData = { d: degree, ensembleSize, byN };
      precomputedRef.current = data;
      setPrecomputed(data);
      setComputing(false);
    });

    return () => cancelAnimationFrame(rafId);
  }, [degree, ensembleSize]);

  // Generate data for the histogram n if not already in pre-computed set
  const histogramData = useMemo(() => {
    if (!precomputed) return [];
    const existing = precomputed.byN.get(nSize);
    if (existing) return existing;
    // Generate fresh for this n
    return generateEnsemble(degree, nSize, ensembleSize);
  }, [precomputed, nSize, degree, ensembleSize]);

  const threshold = alonBoppanaBound(degree);

  const ramanujanCount = useMemo(
    () => histogramData.filter((v) => v <= threshold + 1e-9).length,
    [histogramData, threshold],
  );

  const ramanujanPct = histogramData.length > 0
    ? Math.round((ramanujanCount / histogramData.length) * 100)
    : 0;

  // =========================================================================
  // Panel 1: Histogram of lambda2
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(histSvgRef.current);
    if (!histSvgRef.current || containerWidth === 0 || histogramData.length === 0) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.48);
    const panelHeight = HISTOGRAM_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    const plotW = panelWidth - MARGIN.left - MARGIN.right;
    const plotH = panelHeight - MARGIN.top - MARGIN.bottom;

    const xMin = Math.min(...histogramData) - 0.2;
    const xMax = Math.max(...histogramData, threshold) + 0.2;

    const xScale = d3.scaleLinear()
      .domain([xMin, xMax])
      .range([MARGIN.left, panelWidth - MARGIN.right]);

    // Create histogram bins
    const histogram = d3.bin()
      .domain([xMin, xMax] as [number, number])
      .thresholds(20);

    const bins = histogram(histogramData);
    const yMax = d3.max(bins, (b) => b.length) || 1;

    const yScale = d3.scaleLinear()
      .domain([0, yMax])
      .nice()
      .range([panelHeight - MARGIN.bottom, MARGIN.top]);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text(`Distribution of \u03BB\u2082 for random ${degree}-regular graphs (n=${nSize})`);

    // Bars
    svg.selectAll('rect.hist-bar')
      .data(bins)
      .join('rect')
      .attr('class', 'hist-bar')
      .attr('x', (d) => xScale(d.x0!) + 1)
      .attr('y', (d) => yScale(d.length))
      .attr('width', (d) => Math.max(0, xScale(d.x1!) - xScale(d.x0!) - 2))
      .attr('height', (d) => Math.max(0, yScale(0) - yScale(d.length)))
      .style('fill', (d) => {
        // Color based on whether the bin midpoint is <= threshold
        const mid = ((d.x0 || 0) + (d.x1 || 0)) / 2;
        return mid <= threshold + 1e-9 ? TEAL : AMBER;
      })
      .style('fill-opacity', '0.85')
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', '1');

    // Threshold line
    svg.append('line')
      .attr('x1', xScale(threshold))
      .attr('y1', MARGIN.top)
      .attr('x2', xScale(threshold))
      .attr('y2', panelHeight - MARGIN.bottom)
      .style('stroke', RED)
      .style('stroke-width', '2')
      .style('stroke-dasharray', '6,4');

    // Threshold label
    svg.append('text')
      .attr('x', xScale(threshold) + 5)
      .attr('y', MARGIN.top + 12)
      .style('fill', RED)
      .style('font-size', '11px')
      .style('font-weight', '600')
      .text(`2\u221A(d\u22121) = ${threshold.toFixed(2)}`);

    // Annotation: X% Ramanujan
    svg.append('text')
      .attr('x', panelWidth - MARGIN.right - 4)
      .attr('y', MARGIN.top + 12)
      .style('text-anchor', 'end')
      .style('fill', TEAL)
      .style('font-size', '13px')
      .style('font-weight', '700')
      .text(`${ramanujanPct}% Ramanujan`);

    // X axis
    const xAxis = d3.axisBottom(xScale).ticks(8);
    svg.append('g')
      .attr('transform', `translate(0,${panelHeight - MARGIN.bottom})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    // Y axis
    const yAxis = d3.axisLeft(yScale).ticks(6).tickFormat(d3.format('d'));
    svg.append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(yAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    svg.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Axis labels
    svg.append('text')
      .attr('x', (MARGIN.left + panelWidth - MARGIN.right) / 2)
      .attr('y', panelHeight - 4)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('\u03BB\u2082');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(MARGIN.top + panelHeight - MARGIN.bottom) / 2)
      .attr('y', 14)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('Count');
  }, [histogramData, containerWidth, isNarrow, degree, nSize, threshold, ramanujanPct]);

  // =========================================================================
  // Panel 2: Box plot of lambda2 vs n
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(boxSvgRef.current);
    if (!boxSvgRef.current || containerWidth === 0 || !precomputed) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.48);
    const panelHeight = BOXPLOT_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    // Compute box stats for each n
    const stats: { n: number; box: BoxStats }[] = [];
    for (const n of BOXPLOT_NS) {
      const data = precomputed.byN.get(n);
      if (data && data.length > 0) {
        stats.push({ n, box: boxplotStats(data) });
      }
    }

    if (stats.length === 0) return;

    const xScale = d3.scaleBand()
      .domain(BOXPLOT_NS.map(String))
      .range([MARGIN.left, panelWidth - MARGIN.right])
      .padding(0.3);

    const allValues = stats.flatMap((s) => [s.box.min, s.box.max]);
    const yMin = Math.min(...allValues, threshold) - 0.3;
    const yMax = Math.max(...allValues, threshold) + 0.3;

    const yScale = d3.scaleLinear()
      .domain([yMin, yMax])
      .nice()
      .range([panelHeight - MARGIN.bottom, MARGIN.top]);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text(`\u03BB\u2082 vs n (d=${degree})`);

    // Threshold line
    svg.append('line')
      .attr('x1', MARGIN.left)
      .attr('y1', yScale(threshold))
      .attr('x2', panelWidth - MARGIN.right)
      .attr('y2', yScale(threshold))
      .style('stroke', RED)
      .style('stroke-width', '1.5')
      .style('stroke-dasharray', '6,4');

    svg.append('text')
      .attr('x', panelWidth - MARGIN.right - 4)
      .attr('y', yScale(threshold) - 5)
      .style('text-anchor', 'end')
      .style('fill', RED)
      .style('font-size', '10px')
      .text(`2\u221A(d\u22121)`);

    const boxWidth = xScale.bandwidth();

    for (const { n, box } of stats) {
      const cx = (xScale(String(n)) || 0) + boxWidth / 2;
      const x0 = xScale(String(n)) || 0;

      // Whisker: min to Q1
      svg.append('line')
        .attr('x1', cx)
        .attr('y1', yScale(box.min))
        .attr('x2', cx)
        .attr('y2', yScale(box.q1))
        .style('stroke', TEAL)
        .style('stroke-width', '1.5');

      // Whisker: Q3 to max
      svg.append('line')
        .attr('x1', cx)
        .attr('y1', yScale(box.q3))
        .attr('x2', cx)
        .attr('y2', yScale(box.max))
        .style('stroke', TEAL)
        .style('stroke-width', '1.5');

      // Min cap
      svg.append('line')
        .attr('x1', cx - boxWidth * 0.25)
        .attr('y1', yScale(box.min))
        .attr('x2', cx + boxWidth * 0.25)
        .attr('y2', yScale(box.min))
        .style('stroke', TEAL)
        .style('stroke-width', '1.5');

      // Max cap
      svg.append('line')
        .attr('x1', cx - boxWidth * 0.25)
        .attr('y1', yScale(box.max))
        .attr('x2', cx + boxWidth * 0.25)
        .attr('y2', yScale(box.max))
        .style('stroke', TEAL)
        .style('stroke-width', '1.5');

      // Box: Q1 to Q3
      svg.append('rect')
        .attr('x', x0)
        .attr('y', yScale(box.q3))
        .attr('width', boxWidth)
        .attr('height', Math.max(0, yScale(box.q1) - yScale(box.q3)))
        .style('fill', TEAL_LIGHT)
        .style('stroke', TEAL)
        .style('stroke-width', '1.5');

      // Median line
      svg.append('line')
        .attr('x1', x0)
        .attr('y1', yScale(box.median))
        .attr('x2', x0 + boxWidth)
        .attr('y2', yScale(box.median))
        .style('stroke', TEAL)
        .style('stroke-width', '2.5');
    }

    // X axis
    const xAxis = d3.axisBottom(xScale);
    svg.append('g')
      .attr('transform', `translate(0,${panelHeight - MARGIN.bottom})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    // Y axis
    const yAxis = d3.axisLeft(yScale).ticks(6);
    svg.append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(yAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    svg.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Axis labels
    svg.append('text')
      .attr('x', (MARGIN.left + panelWidth - MARGIN.right) / 2)
      .attr('y', panelHeight - 4)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('n (vertices)');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(MARGIN.top + panelHeight - MARGIN.bottom) / 2)
      .attr('y', 14)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('\u03BB\u2082');
  }, [precomputed, containerWidth, isNarrow, degree, threshold]);

  // =========================================================================
  // Panel 3: Ramanujan fraction line plot
  // =========================================================================

  useEffect(() => {
    const svg = d3.select(lineSvgRef.current);
    if (!lineSvgRef.current || containerWidth === 0 || !precomputed) return;
    svg.selectAll('*').remove();

    const panelWidth = isNarrow
      ? containerWidth
      : Math.floor(containerWidth * 0.48);
    const panelHeight = LINEPLOT_HEIGHT;

    svg.attr('width', panelWidth).attr('height', panelHeight);

    // Compute Ramanujan fraction for each n
    const points: { n: number; fraction: number }[] = [];
    for (const n of BOXPLOT_NS) {
      const data = precomputed.byN.get(n);
      if (data && data.length > 0) {
        const ram = data.filter((v) => v <= threshold + 1e-9).length;
        points.push({ n, fraction: ram / data.length });
      }
    }

    if (points.length === 0) return;

    const xScale = d3.scaleLinear()
      .domain([BOXPLOT_NS[0], BOXPLOT_NS[BOXPLOT_NS.length - 1]])
      .range([MARGIN.left, panelWidth - MARGIN.right]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([panelHeight - MARGIN.bottom, MARGIN.top]);

    // Title
    svg.append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text(`Ramanujan fraction vs n (d=${degree})`);

    // Grid lines
    for (const tick of [0, 0.25, 0.5, 0.75, 1.0]) {
      svg.append('line')
        .attr('x1', MARGIN.left)
        .attr('y1', yScale(tick))
        .attr('x2', panelWidth - MARGIN.right)
        .attr('y2', yScale(tick))
        .style('stroke', 'var(--color-border)')
        .style('stroke-opacity', '0.4')
        .style('stroke-width', '1');
    }

    // Line
    const line = d3.line<{ n: number; fraction: number }>()
      .x((d) => xScale(d.n))
      .y((d) => yScale(d.fraction));

    svg.append('path')
      .datum(points)
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', TEAL)
      .style('stroke-width', '2');

    // Dots
    svg.selectAll('circle.frac-dot')
      .data(points)
      .join('circle')
      .attr('class', 'frac-dot')
      .attr('cx', (d) => xScale(d.n))
      .attr('cy', (d) => yScale(d.fraction))
      .attr('r', 5)
      .style('fill', TEAL)
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', '2');

    // Value labels on dots
    svg.selectAll('text.frac-label')
      .data(points)
      .join('text')
      .attr('class', 'frac-label')
      .attr('x', (d) => xScale(d.n))
      .attr('y', (d) => yScale(d.fraction) - 10)
      .style('text-anchor', 'middle')
      .style('fill', TEAL)
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text((d) => `${Math.round(d.fraction * 100)}%`);

    // X axis
    const xAxis = d3.axisBottom(xScale)
      .tickValues(BOXPLOT_NS)
      .tickFormat(d3.format('d'));
    svg.append('g')
      .attr('transform', `translate(0,${panelHeight - MARGIN.bottom})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    // Y axis
    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d3.format('.0%'));
    svg.append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(yAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');

    svg.selectAll('.domain, .tick line')
      .style('stroke', 'var(--color-border)');

    // Axis labels
    svg.append('text')
      .attr('x', (MARGIN.left + panelWidth - MARGIN.right) / 2)
      .attr('y', panelHeight - 4)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('n (vertices)');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(MARGIN.top + panelHeight - MARGIN.bottom) / 2)
      .attr('y', 14)
      .style('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('Fraction Ramanujan');
  }, [precomputed, containerWidth, isNarrow, degree, threshold]);

  // =========================================================================
  // Render
  // =========================================================================

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '12px',
          alignItems: 'center',
          marginBottom: '12px',
          padding: '10px 12px',
          borderRadius: '8px',
          border: '1px solid var(--color-border)',
          background: 'var(--color-surface)',
        }}
      >
        {/* Degree selector */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
          <span style={{ fontWeight: 600 }}>Degree d:</span>
          <div style={{ display: 'flex', gap: '4px' }}>
            {DEGREES.map((d) => (
              <button
                key={d}
                onClick={() => setDegree(d)}
                style={{
                  padding: '3px 10px',
                  borderRadius: '4px',
                  border: '1px solid var(--color-border)',
                  background: degree === d ? TEAL : 'var(--color-bg)',
                  color: degree === d ? '#fff' : 'var(--color-text)',
                  fontSize: '12px',
                  fontWeight: degree === d ? 700 : 400,
                  cursor: 'pointer',
                }}
              >
                {d}
              </button>
            ))}
          </div>
        </label>

        {/* Size n slider */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
          <span style={{ fontWeight: 600 }}>n:</span>
          <input
            type="range"
            min={8}
            max={50}
            step={1}
            value={nSize}
            onChange={(e) => setNSize(parseInt(e.target.value, 10))}
            style={{ width: '100px' }}
          />
          <span style={{ minWidth: '28px', textAlign: 'right', color: 'var(--color-text-secondary)', fontSize: '12px' }}>
            {nSize}
          </span>
        </label>

        {/* Ensemble size selector */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text)' }}>
          <span style={{ fontWeight: 600 }}>Ensemble:</span>
          <select
            value={ensembleSize}
            onChange={(e) => setEnsembleSize(parseInt(e.target.value, 10))}
            style={{
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg)',
              color: 'var(--color-text)',
              fontSize: '13px',
            }}
          >
            {ENSEMBLE_SIZES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>

      {/* Computing state */}
      {computing && (
        <div
          style={{
            padding: '20px',
            textAlign: 'center',
            color: 'var(--color-text-secondary)',
            fontSize: '14px',
          }}
        >
          Computing ensemble spectra...
        </div>
      )}

      {/* Panels */}
      {!computing && (
        <div
          style={{
            display: 'flex',
            flexDirection: isNarrow ? 'column' : 'row',
            gap: '12px',
            justifyContent: 'space-between',
          }}
        >
          {/* Left: histogram */}
          <div
            style={{
              flex: isNarrow ? undefined : '0 0 48%',
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              overflow: 'hidden',
              background: 'var(--color-surface)',
            }}
          >
            <svg ref={histSvgRef} />
          </div>

          {/* Right: box plot + line plot stacked */}
          <div
            style={{
              flex: isNarrow ? undefined : '0 0 48%',
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
            }}
          >
            <div
              style={{
                border: '1px solid var(--color-border)',
                borderRadius: '8px',
                overflow: 'hidden',
                background: 'var(--color-surface)',
              }}
            >
              <svg ref={boxSvgRef} />
            </div>
            <div
              style={{
                border: '1px solid var(--color-border)',
                borderRadius: '8px',
                overflow: 'hidden',
                background: 'var(--color-surface)',
              }}
            >
              <svg ref={lineSvgRef} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

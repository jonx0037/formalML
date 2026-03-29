import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 24, right: 12, bottom: 36, left: 44 };

/**
 * Seeded linear congruential generator (LCG) for deterministic random numbers.
 * Uses the Park-Miller multiplier 16807 (a = 7^5) with modulus 2^31 - 1.
 * Three mixing rounds reduce correlation between adjacent index values.
 */
function lcg(index: number, seed: number): number {
  let s = (seed * 2147483647 + index * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  return (s & 0x7fffffff) / 0x7fffffff;
}

// ─── CRP sequence generation ───

interface CRPResult {
  assignments: number[];
  tableHistory: number[][];
}

function generateCRP(alpha: number, totalN: number, seed: number): CRPResult {
  const assignments: number[] = [];
  const tableSizes: number[] = [];
  const tableHistory: number[][] = [[]];

  for (let i = 0; i < totalN; i++) {
    const u = lcg(i * 3 + 1, seed + i * 7);
    const total = alpha + i;
    let cumulative = 0;
    let assigned = -1;

    // Try each existing table
    for (let k = 0; k < tableSizes.length; k++) {
      cumulative += tableSizes[k] / total;
      if (u < cumulative) {
        assigned = k;
        break;
      }
    }

    // New table
    if (assigned === -1) {
      assigned = tableSizes.length;
      tableSizes.push(0);
    }

    tableSizes[assigned]++;
    assignments.push(assigned);
    tableHistory.push([...tableSizes]);
  }

  return { assignments, tableHistory };
}

// ─── Table packing layout ───

interface TableCircle {
  index: number;
  size: number;
  x: number;
  y: number;
  r: number;
}

function packTables(sizes: number[], width: number, height: number): TableCircle[] {
  if (sizes.length === 0) return [];

  const maxSize = Math.max(...sizes);
  const minR = 10;
  const maxR = Math.min(40, Math.min(width, height) * 0.15);

  const circles: TableCircle[] = sizes.map((size, index) => {
    const r = Math.max(minR, Math.sqrt(size / Math.max(maxSize, 1)) * maxR);
    return { index, size, x: width / 2, y: height / 2, r };
  });

  // Use d3.forceSimulation for packing
  const simulation = d3.forceSimulation(circles as d3.SimulationNodeDatum[] as Array<TableCircle & d3.SimulationNodeDatum>)
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('charge', d3.forceManyBody().strength(1))
    .force('collide', d3.forceCollide<TableCircle & d3.SimulationNodeDatum>().radius(d => d.r + 3).iterations(3))
    .force('x', d3.forceX(width / 2).strength(0.08))
    .force('y', d3.forceY(height / 2).strength(0.08))
    .stop();

  // Run a few iterations synchronously
  for (let i = 0; i < 120; i++) simulation.tick();

  // Clamp to bounds
  for (const c of circles) {
    c.x = Math.max(c.r + 4, Math.min(width - c.r - 4, c.x));
    c.y = Math.max(c.r + 4, Math.min(height - c.r - 4, c.y));
  }

  return circles;
}

// ─── Component ───

export default function CRPSimulator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const restaurantSvgRef = useRef<SVGSVGElement>(null);
  const barSvgRef = useRef<SVGSVGElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [alpha, setAlpha] = useState(2.0);
  const [totalN, setTotalN] = useState(50);
  const [speed, setSpeed] = useState(5);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // Pre-compute the full CRP sequence
  const crp = useMemo(() => generateCRP(alpha, totalN, 42), [alpha, totalN]);

  // Reset when alpha or totalN change
  const prevAlpha = useRef(alpha);
  const prevN = useRef(totalN);
  useEffect(() => {
    if (prevAlpha.current !== alpha || prevN.current !== totalN) {
      setCurrentStep(0);
      setIsPlaying(false);
      prevAlpha.current = alpha;
      prevN.current = totalN;
    }
  }, [alpha, totalN]);

  // Animation
  useEffect(() => {
    if (isPlaying && currentStep < totalN) {
      intervalRef.current = setInterval(() => {
        setCurrentStep(prev => {
          const next = prev + 1;
          if (next >= totalN) {
            setIsPlaying(false);
          }
          return Math.min(next, totalN);
        });
      }, 1000 / speed);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isPlaying, totalN, speed]);

  // Current snapshot
  const currentSizes = useMemo(() => {
    return crp.tableHistory[currentStep] || [];
  }, [crp, currentStep]);

  const numTables = currentSizes.length;
  const tableColors = d3.schemeTableau10;

  // Layout
  const isDesktop = (containerWidth || 0) > 600;
  const panelW = isDesktop ? Math.floor((containerWidth - 28) / 2) : containerWidth;
  const panelH = Math.min(320, Math.max(200, panelW * 0.7));

  // Restaurant panel (circles)
  useEffect(() => {
    const svg = restaurantSvgRef.current;
    if (!svg || panelW <= 0) return;

    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    if (currentSizes.length === 0) {
      sel.append('text')
        .attr('x', panelW / 2)
        .attr('y', panelH / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '13px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text-secondary)')
        .text('Press Play to seat customers');
      return;
    }

    const circles = packTables(currentSizes, panelW, panelH);

    // Draw table circles
    const groups = sel.selectAll<SVGGElement, TableCircle>('g.table')
      .data(circles, d => String(d.index))
      .join('g')
      .attr('class', 'table')
      .attr('transform', d => `translate(${d.x},${d.y})`);

    groups.append('circle')
      .attr('r', d => d.r)
      .style('fill', d => tableColors[d.index % tableColors.length])
      .style('fill-opacity', 0.75)
      .style('stroke', d => tableColors[d.index % tableColors.length])
      .style('stroke-width', 2)
      .style('stroke-opacity', 0.9);

    groups.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .style('font-size', d => `${Math.max(9, Math.min(14, d.r * 0.7))}px`)
      .style('font-family', 'var(--font-mono)')
      .style('fill', '#fff')
      .style('font-weight', '600')
      .style('pointer-events', 'none')
      .text(d => d.size);
  }, [currentSizes, panelW, panelH, tableColors]);

  // Bar chart panel
  useEffect(() => {
    const svg = barSvgRef.current;
    if (!svg || panelW <= 0) return;

    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    if (currentSizes.length === 0) {
      sel.append('text')
        .attr('x', panelW / 2)
        .attr('y', panelH / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '13px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text-secondary)')
        .text('No tables yet');
      return;
    }

    const innerW = panelW - MARGIN.left - MARGIN.right;
    const innerH = panelH - MARGIN.top - MARGIN.bottom;

    const xScale = d3.scaleBand<number>()
      .domain(currentSizes.map((_, i) => i))
      .range([0, innerW])
      .padding(0.2);

    const yMax = Math.max(1, ...currentSizes);
    const yScale = d3.scaleLinear()
      .domain([0, yMax])
      .nice()
      .range([innerH, 0]);

    const g = sel.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(Math.min(yMax, 6)).tickFormat(d3.format('d')))
      .call(ax => ax.selectAll('.tick line')
        .clone()
        .attr('x2', innerW)
        .style('stroke', 'var(--color-border)')
        .style('stroke-dasharray', '3 3'))
      .call(ax => ax.selectAll('.tick text')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text-secondary)'))
      .call(ax => ax.select('.domain').remove());

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).tickFormat(i => `T${Number(i) + 1}`))
      .call(ax => ax.selectAll('.tick text')
        .style('font-size', currentSizes.length > 15 ? '8px' : '10px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text-secondary)'))
      .call(ax => ax.select('.domain').style('stroke', 'var(--color-border)'));

    // Axis labels
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + MARGIN.bottom - 4)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-sans)')
      .style('fill', 'var(--color-text-secondary)')
      .text('Table');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -MARGIN.left + 14)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-sans)')
      .style('fill', 'var(--color-text-secondary)')
      .text('Customers');

    // Bars
    g.selectAll('rect.bar')
      .data(currentSizes)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', (_, i) => xScale(i)!)
      .attr('y', d => yScale(d))
      .attr('width', xScale.bandwidth())
      .attr('height', d => innerH - yScale(d))
      .style('fill', (_, i) => tableColors[i % tableColors.length])
      .style('fill-opacity', 0.8)
      .style('rx', '2');

    // Bar value labels
    g.selectAll('text.bar-label')
      .data(currentSizes)
      .join('text')
      .attr('class', 'bar-label')
      .attr('x', (_, i) => xScale(i)! + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d) - 4)
      .attr('text-anchor', 'middle')
      .style('font-size', '9px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', 'var(--color-text-secondary)')
      .text(d => d);
  }, [currentSizes, panelW, panelH, tableColors]);

  // Handlers
  const handlePlayPause = useCallback(() => {
    if (currentStep >= totalN) {
      setCurrentStep(0);
      setIsPlaying(true);
    } else {
      setIsPlaying(p => !p);
    }
  }, [currentStep, totalN]);

  const handleReset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  const handleAlpha = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setAlpha(parseFloat(e.target.value));
  }, []);

  const handleTotalN = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setTotalN(parseInt(e.target.value, 10));
  }, []);

  const handleSpeed = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSpeed(parseInt(e.target.value, 10));
  }, []);

  if (!containerWidth) {
    return <div ref={containerRef} style={{ minHeight: 400 }} />;
  }

  const buttonStyle: React.CSSProperties = {
    padding: '4px 14px',
    fontSize: '12px',
    fontFamily: 'var(--font-mono)',
    border: '1px solid var(--color-border)',
    borderRadius: '4px',
    background: 'transparent',
    color: 'var(--color-text-secondary)',
    cursor: 'pointer',
  };

  const labelStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
    fontFamily: 'var(--font-mono)',
    color: 'var(--color-text-secondary)',
  };

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      {/* Title */}
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, marginBottom: '12px', color: 'var(--color-text)' }}>
        Chinese Restaurant Process
      </div>

      {/* Two-panel layout */}
      <div style={{ display: 'flex', flexDirection: isDesktop ? 'row' : 'column', gap: '12px' }}>
        {/* Left panel: Restaurant */}
        <div style={{ flex: 1 }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
            marginBottom: '6px', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)',
          }}>
            <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '11px', color: 'var(--color-text)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Restaurant
            </span>
            <span>Customer {currentStep}/{totalN}</span>
          </div>
          <svg role="img" aria-label="CRPSimulator visualization (panel 1 of 2)"
            ref={restaurantSvgRef}
            width={panelW}
            height={panelH}
            style={{ border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)' }}
          />
        </div>

        {/* Right panel: Bar chart */}
        <div style={{ flex: 1 }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
            marginBottom: '6px', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)',
          }}>
            <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '11px', color: 'var(--color-text)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Table Sizes
            </span>
            <span>{numTables} table{numTables !== 1 ? 's' : ''}</span>
          </div>
          <svg role="img" aria-label="CRPSimulator visualization (panel 2 of 2)"
            ref={barSvgRef}
            width={panelW}
            height={panelH}
            style={{ border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)' }}
          />
        </div>
      </div>

      {/* Controls */}
      <div style={{ marginTop: '14px', display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center' }}>
        {/* Play / Pause */}
        <button onClick={handlePlayPause} style={{ ...buttonStyle, minWidth: '70px' }}>
          {isPlaying ? 'Pause' : currentStep >= totalN ? 'Replay' : 'Play'}
        </button>

        {/* Reset */}
        <button onClick={handleReset} style={buttonStyle}>
          Reset
        </button>

        {/* Alpha slider */}
        <label style={labelStyle}>
          <span style={{ whiteSpace: 'nowrap' }}>
            {'\u03B1'}: <strong style={{ color: 'var(--color-text)', minWidth: '32px', display: 'inline-block' }}>{alpha.toFixed(1)}</strong>
          </span>
          <input
            type="range" min={0.5} max={20} step={0.5} value={alpha}
            onChange={handleAlpha}
            style={{ width: '100px', accentColor: 'var(--color-accent)' }}
          />
        </label>

        {/* N slider */}
        <label style={labelStyle}>
          <span style={{ whiteSpace: 'nowrap' }}>
            N: <strong style={{ color: 'var(--color-text)', minWidth: '28px', display: 'inline-block' }}>{totalN}</strong>
          </span>
          <input
            type="range" min={10} max={200} step={10} value={totalN}
            onChange={handleTotalN}
            style={{ width: '100px', accentColor: 'var(--color-accent)' }}
          />
        </label>

        {/* Speed slider */}
        <label style={labelStyle}>
          <span style={{ whiteSpace: 'nowrap' }}>
            Speed: <strong style={{ color: 'var(--color-text)', minWidth: '20px', display: 'inline-block' }}>{speed}</strong>/s
          </span>
          <input
            type="range" min={1} max={20} step={1} value={speed}
            onChange={handleSpeed}
            style={{ width: '80px', accentColor: 'var(--color-accent)' }}
          />
        </label>
      </div>

      {/* Info line */}
      <div style={{
        marginTop: '10px', fontSize: '11px', lineHeight: 1.5,
        fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)',
      }}>
        With {'\u03B1'} = {alpha.toFixed(1)}, the expected number of tables after {totalN} customers
        is {'\u2248'} {(alpha * Array.from({ length: totalN }, (_, i) => 1 / (alpha + i)).reduce((a, b) => a + b, 0)).toFixed(1)}.
        {currentStep > 0 && ` Currently: ${numTables} table${numTables !== 1 ? 's' : ''} occupied.`}
      </div>
    </div>
  );
}

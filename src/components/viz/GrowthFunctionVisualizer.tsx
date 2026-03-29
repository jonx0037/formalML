import { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 56 };

const COLORS = {
  sauerShelah: '#0F6E56',
  simplified: '#D97706',
  ratio: '#534AB7',
  phaseTransition: '#DC2626',
} as const;

/** Compute binomial coefficient C(n,k) using multiplicative formula */
function choose(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  if (k > n - k) k = n - k;
  let result = 1;
  for (let i = 0; i < k; i++) {
    result = result * (n - i) / (i + 1);
  }
  return Math.round(result);
}

/** Growth function: sum_{i=0}^{d} C(m, i) */
function growthFunction(m: number, d: number): number {
  // Use (em/d)^d approximation for large m to avoid overflow
  if (m > 60 && d > 0) {
    const exact = growthFunctionExact(m, d);
    if (!isFinite(exact)) return Math.pow(Math.E * m / d, d);
    return exact;
  }
  return growthFunctionExact(m, d);
}

function growthFunctionExact(m: number, d: number): number {
  let sum = 0;
  for (let i = 0; i <= d; i++) {
    sum += choose(m, i);
  }
  return sum;
}

export default function GrowthFunctionVisualizer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const [vcDim, setVcDim] = useState(5);
  const [showTransition, setShowTransition] = useState(true);

  const isDesktop = (containerWidth || 0) > 640;
  const panelW = isDesktop ? Math.floor((containerWidth - 24) / 2) : containerWidth;
  const panelH = Math.min(280, Math.max(200, panelW * 0.65));

  const data = useMemo(() => {
    const mMax = Math.max(vcDim * 4, 30);
    const mRange = Array.from({ length: mMax }, (_, i) => i + 1);
    const exponential = mRange.map(m => Math.pow(2, m));
    const sauerShelah = mRange.map(m => growthFunction(m, vcDim));
    const simplified = mRange.map(m => m >= vcDim ? Math.pow(Math.E * m / vcDim, vcDim) : Math.pow(2, m));
    const ratio = mRange.map((m, i) => sauerShelah[i] / exponential[i]);
    return { mRange, exponential, sauerShelah, simplified, ratio, mMax };
  }, [vcDim]);

  // Left panel: growth function (log scale)
  useEffect(() => {
    const svg = leftSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { mRange, exponential, sauerShelah, simplified, mMax } = data;

    const xScale = d3.scaleLinear().domain([1, mMax]).range([MARGIN.left, panelW - MARGIN.right]);
    const yMax = Math.pow(2, Math.min(mMax, 30));
    const yScale = d3.scaleLog().domain([1, yMax]).range([panelH - MARGIN.bottom, MARGIN.top]).clamp(true);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(6).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5, '.0e').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    // Axis labels
    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('m (number of points)');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('Π(m)');

    const clip = (v: number) => Math.max(1, Math.min(v, yMax));

    // 2^m (gray dashed)
    const expLine = d3.line<number>().x((_, i) => xScale(mRange[i])).y(d => yScale(clip(d)));
    sel.append('path').datum(exponential.map(v => clip(v))).attr('d', expLine)
      .style('fill', 'none').style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1.5).style('stroke-dasharray', '6 3').style('opacity', 0.5);

    // Sauer-Shelah exact (teal)
    sel.append('path').datum(sauerShelah.map(v => clip(v))).attr('d', expLine)
      .style('fill', 'none').style('stroke', COLORS.sauerShelah).style('stroke-width', 2.5);

    // Simplified (em/d)^d (amber dashed)
    sel.append('path').datum(simplified.map(v => clip(v))).attr('d', expLine)
      .style('fill', 'none').style('stroke', COLORS.simplified).style('stroke-width', 1.5).style('stroke-dasharray', '4 3');

    // Phase transition annotation
    if (showTransition && vcDim <= mMax) {
      sel.append('line')
        .attr('x1', xScale(vcDim)).attr('x2', xScale(vcDim))
        .attr('y1', yScale(1)).attr('y2', yScale(yMax))
        .style('stroke', COLORS.phaseTransition).style('stroke-dasharray', '3 2').style('opacity', 0.6);
      sel.append('text').attr('x', xScale(vcDim) + 4).attr('y', MARGIN.top + 14)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', COLORS.phaseTransition)
        .text(`m = d = ${vcDim}`);
    }

    // Legend
    const ly = panelH - MARGIN.bottom - 50;
    const lx = panelW - MARGIN.right - 100;
    [
      { label: '2^m', color: 'var(--color-text-secondary)', dash: true },
      { label: 'Sauer–Shelah', color: COLORS.sauerShelah, dash: false },
      { label: '(em/d)^d', color: COLORS.simplified, dash: true },
    ].forEach(({ label, color, dash }, i) => {
      const line = sel.append('line').attr('x1', lx).attr('x2', lx + 14).attr('y1', ly + i * 12).attr('y2', ly + i * 12)
        .style('stroke', color).style('stroke-width', 1.5);
      if (dash) line.style('stroke-dasharray', '4 3');
      sel.append('text').attr('x', lx + 18).attr('y', ly + i * 12 + 3)
        .style('font-size', '8px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text(label);
    });

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('Growth Function Π(m)');
  }, [data, panelW, panelH, showTransition, vcDim]);

  // Right panel: shattering ratio
  useEffect(() => {
    const svg = rightSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { mRange, ratio, mMax } = data;

    const xScale = d3.scaleLinear().domain([1, mMax]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([0, 1.05]).range([panelH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(6).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    // Axis labels
    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('m (number of points)');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('Π(m) / 2^m');

    // Ratio curve
    const lineGen = d3.line<number>().x((_, i) => xScale(mRange[i])).y(d => yScale(Math.max(0, Math.min(d, 1.05))));
    sel.append('path').datum(ratio).attr('d', lineGen)
      .style('fill', 'none').style('stroke', COLORS.ratio).style('stroke-width', 2.5);

    // Phase transition
    if (showTransition && vcDim <= mMax) {
      sel.append('line')
        .attr('x1', xScale(vcDim)).attr('x2', xScale(vcDim))
        .attr('y1', yScale(0)).attr('y2', yScale(1.05))
        .style('stroke', COLORS.phaseTransition).style('stroke-dasharray', '3 2').style('opacity', 0.6);
      sel.append('text').attr('x', xScale(vcDim) + 4).attr('y', MARGIN.top + 14)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', COLORS.phaseTransition)
        .text(`m = d = ${vcDim}`);
    }

    // Ratio = 1 line
    sel.append('line')
      .attr('x1', MARGIN.left).attr('x2', panelW - MARGIN.right)
      .attr('y1', yScale(1)).attr('y2', yScale(1))
      .style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '4 3').style('opacity', 0.3);

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('Shattering Ratio Π(m) / 2^m');
  }, [data, panelW, panelH, showTransition, vcDim]);

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
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, marginBottom: '12px', color: 'var(--color-text)' }}>
        Growth Function & Sauer–Shelah Bound
      </div>

      <div style={{ display: 'flex', flexDirection: isDesktop ? 'row' : 'column', gap: '8px' }}>
        <svg role="img" aria-label="Growth function visualizer visualization (panel 1 of 2)" ref={leftSvgRef} width={panelW} height={panelH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
        <svg role="img" aria-label="Growth function visualizer visualization (panel 2 of 2)" ref={rightSvgRef} width={panelW} height={panelH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center', marginTop: '12px' }}>
        <div style={{ flex: '1 1 200px' }}>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            VC dimension d: <strong style={{ color: 'var(--color-text)' }}>{vcDim}</strong>
          </div>
          <input type="range" min={1} max={15} step={1} value={vcDim}
            onChange={e => setVcDim(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', cursor: 'pointer' }}>
          <input type="checkbox" checked={showTransition} onChange={e => setShowTransition(e.target.checked)}
            style={{ accentColor: COLORS.phaseTransition }} />
          Show phase transition at m = d
        </label>
      </div>
    </div>
  );
}

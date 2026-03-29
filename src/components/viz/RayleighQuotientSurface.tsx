import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const N_SAMPLES = 200;
const PANEL_GAP = 24;
const MAX_PANEL_SIZE = 360;

export default function RayleighQuotientSurface() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const polarRef = useRef<SVGSVGElement>(null);
  const cartRef = useRef<SVGSVGElement>(null);

  const [a, setA] = useState(3);
  const [b, setB] = useState(1);
  const [c, setC] = useState(2);

  const panelSize = useMemo(() => {
    if (!containerWidth) return 300;
    return Math.min(Math.floor((containerWidth - PANEL_GAP) / 2), MAX_PANEL_SIZE);
  }, [containerWidth]);
  const panelH = panelSize;

  // Eigenvalues and eigenvector angles
  const eigen = useMemo(() => {
    const disc = Math.sqrt((a - c) ** 2 + 4 * b ** 2);
    const lambda1 = (a + c - disc) / 2;
    const lambda2 = (a + c + disc) / 2;
    let theta1: number, theta2: number;
    if (Math.abs(b) < 1e-12) {
      theta1 = a <= c ? 0 : Math.PI / 2;
      theta2 = a <= c ? Math.PI / 2 : 0;
    } else {
      theta1 = Math.atan2(b, lambda1 - c);
      theta2 = Math.atan2(b, lambda2 - c);
    }
    return { lambda1, lambda2, theta1, theta2 };
  }, [a, b, c]);

  // R(theta) samples
  const samples = useMemo(() => {
    const pts: { theta: number; R: number }[] = [];
    for (let i = 0; i <= N_SAMPLES; i++) {
      const theta = (i / N_SAMPLES) * 2 * Math.PI;
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      const R = a * cos * cos + 2 * b * cos * sin + c * sin * sin;
      pts.push({ theta, R });
    }
    return pts;
  }, [a, b, c]);

  const rExtent = useMemo(() => {
    const vals = samples.map((s) => s.R);
    return [Math.min(...vals), Math.max(...vals)] as [number, number];
  }, [samples]);

  // ─── Polar plot ───
  useEffect(() => {
    if (!polarRef.current || panelSize === 0) return;
    const svg = d3.select(polarRef.current);
    svg.selectAll('*').remove();

    const cx = panelSize / 2;
    const cy = panelSize / 2;
    const margin = 32;
    const maxR = Math.max(Math.abs(rExtent[0]), Math.abs(rExtent[1]), 0.01);
    const rScale = (panelSize / 2 - margin) / maxR;

    const g = svg.append('g');

    // Radial grid
    const gridSteps = 4;
    for (let i = 1; i <= gridSteps; i++) {
      const r = (i / gridSteps) * (panelSize / 2 - margin);
      g.append('circle')
        .attr('cx', cx).attr('cy', cy).attr('r', r)
        .attr('fill', 'none')
        .style('stroke', 'var(--color-border)')
        .attr('stroke-opacity', 0.3);
    }
    // Radial lines
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * 2 * Math.PI;
      g.append('line')
        .attr('x1', cx).attr('y1', cy)
        .attr('x2', cx + Math.cos(angle) * (panelSize / 2 - margin))
        .attr('y2', cy - Math.sin(angle) * (panelSize / 2 - margin))
        .style('stroke', 'var(--color-border)')
        .attr('stroke-opacity', 0.15);
    }

    // Unit circle reference (R=1 maps to rScale distance)
    g.append('circle')
      .attr('cx', cx).attr('cy', cy).attr('r', rScale)
      .attr('fill', 'none')
      .style('stroke', 'var(--color-text)')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-dasharray', '4,3');

    // Polar curve — convert to Cartesian
    const polarPts: [number, number][] = samples.map((s) => [
      cx + s.R * rScale * Math.cos(s.theta),
      cy - s.R * rScale * Math.sin(s.theta),
    ]);

    // Filled region
    const line = d3.line<[number, number]>().x((d) => d[0]).y((d) => d[1]).curve(d3.curveCatmullRomClosed);
    const allPositive = rExtent[0] >= 0;
    const allNegative = rExtent[1] <= 0;
    const fillColor = allNegative ? '#ef4444' : allPositive ? '#3b82f6' : '#8b5cf6';

    g.append('path')
      .datum(polarPts)
      .attr('d', line)
      .attr('fill', fillColor)
      .attr('fill-opacity', 0.12)
      .attr('stroke', fillColor)
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.8);

    // Color segments: overlay positive/negative coloring if indefinite
    if (!allPositive && !allNegative) {
      const positivePts = samples.filter((s) => s.R >= 0);
      const negativePts = samples.filter((s) => s.R < 0);
      for (const { pts, color } of [
        { pts: positivePts, color: '#3b82f6' },
        { pts: negativePts, color: '#ef4444' },
      ]) {
        if (pts.length < 2) continue;
        const coords: [number, number][] = pts.map((s) => [
          cx + s.R * rScale * Math.cos(s.theta),
          cy - s.R * rScale * Math.sin(s.theta),
        ]);
        g.selectAll(null)
          .data(coords)
          .join('circle')
          .attr('cx', (d) => d[0])
          .attr('cy', (d) => d[1])
          .attr('r', 2)
          .attr('fill', color)
          .attr('fill-opacity', 0.6);
      }
    }

    // Eigenvector markers
    const { lambda1, lambda2, theta1, theta2 } = eigen;
    for (const { lam, th, label } of [
      { lam: lambda1, th: theta1, label: '\u03BB\u2081' },
      { lam: lambda2, th: theta2, label: '\u03BB\u2082' },
    ]) {
      const px = cx + lam * rScale * Math.cos(th);
      const py = cy - lam * rScale * Math.sin(th);
      g.append('circle')
        .attr('cx', px).attr('cy', py).attr('r', 5)
        .attr('fill', lam === lambda2 ? '#3b82f6' : '#ef4444')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);
      g.append('text')
        .attr('x', px + 8).attr('y', py + 4)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .text(`${label} = ${lam.toFixed(2)}`);
    }
  }, [samples, eigen, panelSize]);

  // ─── Cartesian plot ───
  useEffect(() => {
    if (!cartRef.current || panelSize === 0) return;
    const svg = d3.select(cartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 16, right: 16, bottom: 32, left: 44 };
    const innerW = panelSize - margin.left - margin.right;
    const innerH = panelH - margin.top - margin.bottom;

    const { lambda1, lambda2, theta1, theta2 } = eigen;
    const pad = Math.max(0.2, (rExtent[1] - rExtent[0]) * 0.1);

    const xScale = d3.scaleLinear().domain([0, 2 * Math.PI]).range([margin.left, margin.left + innerW]);
    const yScale = d3.scaleLinear().domain([rExtent[0] - pad, rExtent[1] + pad]).range([margin.top + innerH, margin.top]);

    const g = svg.append('g');

    // Axes
    const piTicks = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2, 2 * Math.PI];
    const piLabels = ['0', '\u03C0/2', '\u03C0', '3\u03C0/2', '2\u03C0'];
    const xAxis = d3.axisBottom(xScale).tickValues(piTicks).tickFormat((_, i) => piLabels[i]);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    g.append('g')
      .attr('transform', `translate(0,${margin.top + innerH})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10);
    g.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(yAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10);

    // Style axis lines
    g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

    // Horizontal dashed lines at eigenvalues
    for (const { lam, label, color } of [
      { lam: lambda1, label: '\u03BB\u2081', color: '#ef4444' },
      { lam: lambda2, label: '\u03BB\u2082', color: '#3b82f6' },
    ]) {
      const y = yScale(lam);
      g.append('line')
        .attr('x1', margin.left).attr('x2', margin.left + innerW)
        .attr('y1', y).attr('y2', y)
        .attr('stroke', color)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '6,3')
        .attr('stroke-opacity', 0.7);
      g.append('text')
        .attr('x', margin.left + innerW + 2).attr('y', y + 4)
        .attr('font-size', 10)
        .attr('font-weight', 600)
        .attr('fill', color)
        .style('font-family', 'var(--font-sans)')
        .text(label);
    }

    // Vertical dashed lines at eigenvector angles (all occurrences in [0, 2pi])
    const eigAngles = [theta1, theta2].flatMap((th) => {
      const normalized = ((th % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      const opposite = (normalized + Math.PI) % (2 * Math.PI);
      return [normalized, opposite];
    });
    for (const th of eigAngles) {
      g.append('line')
        .attr('x1', xScale(th)).attr('x2', xScale(th))
        .attr('y1', margin.top).attr('y2', margin.top + innerH)
        .style('stroke', 'var(--color-text)')
        .attr('stroke-opacity', 0.2)
        .attr('stroke-dasharray', '3,3');
    }

    // R(theta) curve
    const line = d3.line<{ theta: number; R: number }>()
      .x((d) => xScale(d.theta))
      .y((d) => yScale(d.R))
      .curve(d3.curveCatmullRom);

    g.append('path')
      .datum(samples)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2);

    // Mark extrema (where R = lambda at eigenvector angles)
    for (const { lam, th, color } of [
      { lam: lambda1, th: theta1, color: '#ef4444' },
      { lam: lambda2, th: theta2, color: '#3b82f6' },
    ]) {
      const norm = ((th % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      const opp = (norm + Math.PI) % (2 * Math.PI);
      for (const angle of [norm, opp]) {
        g.append('circle')
          .attr('cx', xScale(angle))
          .attr('cy', yScale(lam))
          .attr('r', 4)
          .attr('fill', color)
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);
      }
    }

    // Y-axis label
    g.append('text')
      .attr('x', -(margin.top + innerH / 2))
      .attr('y', 12)
      .attr('transform', 'rotate(-90)')
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('opacity', 0.6)
      .text('R(\u03B8)');
  }, [samples, eigen, panelSize]);

  const handleA = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setA(parseFloat(e.target.value)), []);
  const handleB = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setB(parseFloat(e.target.value)), []);
  const handleC = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setC(parseFloat(e.target.value)), []);

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Panels */}
      <div className="flex flex-col items-center gap-3 sm:flex-row sm:items-start sm:justify-center">
        <svg role="img" aria-label="Rayleigh quotient surface visualization (panel 1 of 2)"
          ref={polarRef}
          width={panelSize}
          height={panelH}
          className="rounded-lg border border-[var(--color-border)]"
        />
        <svg role="img" aria-label="Rayleigh quotient surface visualization (panel 2 of 2)"
          ref={cartRef}
          width={panelSize}
          height={panelH}
          className="rounded-lg border border-[var(--color-border)]"
        />
      </div>

      {/* Matrix display */}
      <p
        className="text-center text-sm opacity-70"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        A = [[{a.toFixed(1)}, {b.toFixed(1)}], [{b.toFixed(1)}, {c.toFixed(1)}]]
        {' \u2014 '}
        {eigen.lambda1.toFixed(2)} ≤ R(x) ≤ {eigen.lambda2.toFixed(2)}
      </p>

      {/* Sliders */}
      <div className="flex flex-col gap-3 sm:flex-row sm:gap-6">
        {([
          { label: 'a', value: a, onChange: handleA, min: -3, max: 5 },
          { label: 'b', value: b, onChange: handleB, min: -3, max: 3 },
          { label: 'c', value: c, onChange: handleC, min: -3, max: 5 },
        ] as const).map(({ label, value, onChange, min, max }) => (
          <div key={label} className="flex items-center gap-2 flex-1">
            <label
              className="text-xs font-medium whitespace-nowrap min-w-[60px]"
              style={{ fontFamily: 'var(--font-sans)' }}
            >
              {label} = {value.toFixed(1)}
            </label>
            <input
              type="range"
              min={min}
              max={max}
              step={0.1}
              value={value}
              onChange={onChange}
              className="w-full accent-[var(--color-accent)]"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

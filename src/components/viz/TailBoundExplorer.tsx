import { useState, useMemo, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 50 };

// ─── Distribution definitions ───

type DistributionKey = 'exponential' | 'uniform' | 'normal' | 'beta';

interface DistDef {
  label: string;
  mean: number;
  variance: number;
  /** True tail P(X >= t) */
  tail: (t: number) => number;
  /** MGF E[e^{λX}] — needed for Chernoff */
  mgf: (lambda: number) => number;
  tRange: [number, number];
}

const DISTRIBUTIONS: Record<DistributionKey, DistDef> = {
  exponential: {
    label: 'Exponential(1)',
    mean: 1,
    variance: 1,
    tail: (t) => (t <= 0 ? 1 : Math.exp(-t)),
    mgf: (lambda) => (lambda < 1 ? 1 / (1 - lambda) : 1e6),
    tRange: [0.5, 8],
  },
  uniform: {
    label: 'Uniform[0,1]',
    mean: 0.5,
    variance: 1 / 12,
    tail: (t) => (t <= 0 ? 1 : t >= 1 ? 0 : 1 - t),
    mgf: (lambda) => (Math.abs(lambda) < 1e-8 ? 1 : (Math.exp(lambda) - 1) / lambda),
    tRange: [0.1, 3],
  },
  normal: {
    label: 'N(0,1)',
    mean: 0,
    variance: 1,
    // Two-sided: P(|X| >= t)
    tail: (t) => 2 * normalSF(t),
    mgf: (lambda) => Math.exp(lambda * lambda / 2),
    tRange: [0.1, 5],
  },
  beta: {
    label: 'Beta(2,5)',
    mean: 2 / 7,
    variance: (2 * 5) / (49 * 8), // a*b / ((a+b)^2 (a+b+1))
    tail: (t) => (t <= 0 ? 1 : t >= 1 ? 0 : 1 - betaCDF(t, 2, 5)),
    mgf: (_lambda) => 1, // Not used directly; Chernoff uses numerical grid
    tRange: [0.1, 2],
  },
};

// Normal survival function approximation (Abramowitz & Stegun 7.1.26)
function normalSF(x: number): number {
  if (x < 0) return 1 - normalSF(-x);
  const t = 1 / (1 + 0.2316419 * x);
  const d = 0.3989422804014327; // 1/sqrt(2*pi)
  const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + 1.330274429 * t))));
  return d * Math.exp(-x * x / 2) * poly;
}

// Incomplete beta approximation via simple numerical integration
function betaCDF(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const steps = 200;
  const dx = x / steps;
  let sum = 0;
  for (let i = 0; i < steps; i++) {
    const t = (i + 0.5) * dx;
    sum += Math.pow(t, a - 1) * Math.pow(1 - t, b - 1) * dx;
  }
  // Beta function B(a,b)
  const fullSteps = 200;
  const fullDx = 1 / fullSteps;
  let fullSum = 0;
  for (let i = 0; i < fullSteps; i++) {
    const t = (i + 0.5) * fullDx;
    fullSum += Math.pow(t, a - 1) * Math.pow(1 - t, b - 1) * fullDx;
  }
  return sum / fullSum;
}

const BOUND_COLORS = {
  true: '#1E293B',
  markov: '#D97706',
  chebyshev: '#534AB7',
  hoeffding: '#0F6E56',
};

export default function TailBoundExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const [dist, setDist] = useState<DistributionKey>('exponential');
  const [delta, setDelta] = useState(0.05);
  const [showMarkov, setShowMarkov] = useState(true);
  const [showChebyshev, setShowChebyshev] = useState(true);
  const [showHoeffding, setShowHoeffding] = useState(true);

  const isDesktop = (containerWidth || 0) > 640;
  const panelW = isDesktop ? Math.floor((containerWidth - 12) / 2) : containerWidth;
  const panelH = Math.min(280, Math.max(200, panelW * 0.65));

  const cfg = DISTRIBUTIONS[dist];

  // ─── Left panel: Tail probability bounds ───

  const tailData = useMemo(() => {
    const nPts = 150;
    const [tMin, tMax] = cfg.tRange;
    const ts = Array.from({ length: nPts }, (_, i) => tMin + (tMax - tMin) * i / (nPts - 1));

    const trueTail = ts.map(t => Math.max(cfg.tail(t), 1e-15));
    const markov = ts.map(t => Math.min(Math.max(cfg.mean, 0.001) / t, 1));
    const chebyshev = ts.map(t => {
      const dev = t - cfg.mean;
      return dev > 0 ? Math.min(cfg.variance / (dev * dev), 1) : 1;
    });

    // Chernoff: min over λ of e^{-λt} M_X(λ)
    const chernoff = ts.map(t => {
      let best = 1;
      for (let j = 1; j <= 100; j++) {
        const lam = j * 0.05;
        const val = Math.exp(-lam * t) * cfg.mgf(lam);
        if (val < best && isFinite(val)) best = val;
      }
      return Math.min(best, 1);
    });

    return { ts, trueTail, markov, chebyshev, chernoff };
  }, [cfg]);

  useEffect(() => {
    const svg = leftSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { ts, trueTail, markov, chebyshev, chernoff } = tailData;

    const xScale = d3.scaleLinear().domain([ts[0], ts[ts.length - 1]]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLog().domain([1e-8, 2]).range([panelH - MARGIN.bottom, MARGIN.top]).clamp(true);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5, '.0e').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    // Axis labels
    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('t');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 12)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text(dist === 'normal' ? 'P(|X| ≥ t)' : 'P(X ≥ t)');

    const line = (vals: number[], color: string, dashed: boolean) => {
      const lineGen = d3.line<number>()
        .defined((d) => d > 1e-15 && isFinite(d))
        .x((_, i) => xScale(ts[i]))
        .y(d => yScale(d));
      sel.append('path').datum(vals).attr('d', lineGen)
        .style('fill', 'none').style('stroke', color).style('stroke-width', dashed ? 1.8 : 2.5)
        .style('stroke-dasharray', dashed ? '6 3' : 'none');
    };

    // True tail (always shown)
    line(trueTail, BOUND_COLORS.true, false);
    if (showMarkov) line(markov, BOUND_COLORS.markov, true);
    if (showChebyshev) line(chebyshev, BOUND_COLORS.chebyshev, true);
    if (showHoeffding) line(chernoff, BOUND_COLORS.hoeffding, true);

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('Tail Probability Bounds');
  }, [tailData, panelW, panelH, showMarkov, showChebyshev, showHoeffding, dist]);

  // ─── Right panel: Sample complexity ───

  useEffect(() => {
    const svg = rightSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const nPts = 150;
    const eps = Array.from({ length: nPts }, (_, i) => 0.01 + 0.99 * i / (nPts - 1));

    // Chebyshev: n >= σ² / (ε² δ)
    const nCheb = eps.map(e => cfg.variance / (e * e * delta));
    // Hoeffding: n >= (b-a)² / (2ε²) * log(2/δ)   — assume [0,1]-bounded
    const range = dist === 'normal' ? 6 : dist === 'exponential' ? 10 : 1; // approximate range
    const nHoeff = eps.map(e => (range * range) / (2 * e * e) * Math.log(2 / delta));

    const maxN = Math.min(d3.max([...nCheb.slice(10), ...nHoeff.slice(10)]) || 10000, 100000);

    const xScale = d3.scaleLinear().domain([0.01, 1]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLog().domain([1, maxN]).range([panelH - MARGIN.bottom, MARGIN.top]).clamp(true);

    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5, '.0e').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('ε');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 12)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('n required');

    const lineGen = d3.line<number>()
      .defined(d => d > 0 && d <= maxN && isFinite(d))
      .x((_, i) => xScale(eps[i]))
      .y(d => yScale(Math.max(d, 1)));

    sel.append('path').datum(nCheb).attr('d', lineGen)
      .style('fill', 'none').style('stroke', BOUND_COLORS.chebyshev).style('stroke-width', 2).style('stroke-dasharray', '6 3');
    sel.append('path').datum(nHoeff).attr('d', lineGen)
      .style('fill', 'none').style('stroke', BOUND_COLORS.hoeffding).style('stroke-width', 2);

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text(`Sample Complexity (δ = ${delta})`);

    // Legend
    const legendY = MARGIN.top + 10;
    const legendX = panelW - MARGIN.right - 100;
    [{ label: 'Chebyshev', color: BOUND_COLORS.chebyshev, dash: true },
     { label: 'Hoeffding', color: BOUND_COLORS.hoeffding, dash: false }].forEach((item, i) => {
      const y = legendY + i * 16;
      sel.append('line').attr('x1', legendX).attr('x2', legendX + 18).attr('y1', y).attr('y2', y)
        .style('stroke', item.color).style('stroke-width', 2).style('stroke-dasharray', item.dash ? '6 3' : 'none');
      sel.append('text').attr('x', legendX + 22).attr('y', y + 4)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text(item.label);
    });
  }, [cfg, delta, panelW, panelH, dist]);

  // ─── Render ───

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
        Tail Bound Explorer
      </div>

      {/* Two-panel layout */}
      <div style={{ display: 'grid', gridTemplateColumns: isDesktop ? '1fr 1fr' : '1fr', gap: '8px' }}>
        <svg ref={leftSvgRef} width={panelW} height={panelH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
        <svg ref={rightSvgRef} width={panelW} height={panelH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
      </div>

      {/* Controls */}
      <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '14px', alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          Distribution:
          <select
            value={dist}
            onChange={e => setDist(e.target.value as DistributionKey)}
            style={{
              padding: '3px 8px', fontSize: '12px', fontFamily: 'var(--font-mono)',
              border: '1px solid var(--color-border)', borderRadius: '4px',
              background: 'var(--color-surface)', color: 'var(--color-text)',
            }}
          >
            {(Object.keys(DISTRIBUTIONS) as DistributionKey[]).map(k => (
              <option key={k} value={k}>{DISTRIBUTIONS[k].label}</option>
            ))}
          </select>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          δ = {delta.toFixed(2)}
          <input type="range" min={0.01} max={0.5} step={0.01} value={delta}
            onChange={e => setDelta(parseFloat(e.target.value))}
            style={{ width: '80px', accentColor: 'var(--color-accent)' }} />
        </label>

        {/* Bound toggles */}
        {[
          { key: 'markov', label: 'Markov', color: BOUND_COLORS.markov, checked: showMarkov, set: setShowMarkov },
          { key: 'chebyshev', label: 'Chebyshev', color: BOUND_COLORS.chebyshev, checked: showChebyshev, set: setShowChebyshev },
          { key: 'hoeffding', label: 'Chernoff/Hoeffding', color: BOUND_COLORS.hoeffding, checked: showHoeffding, set: setShowHoeffding },
        ].map(({ key, label, color, checked, set }) => (
          <label key={key} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', fontFamily: 'var(--font-mono)', color }}>
            <input type="checkbox" checked={checked} onChange={() => set(!checked)} style={{ accentColor: color }} />
            {label}
          </label>
        ))}
      </div>

      {/* Legend for left panel */}
      <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '16px', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
        <span><span style={{ display: 'inline-block', width: 16, height: 2, background: BOUND_COLORS.true, verticalAlign: 'middle', marginRight: 4 }} /> True tail</span>
        <span><span style={{ display: 'inline-block', width: 16, height: 2, background: BOUND_COLORS.markov, verticalAlign: 'middle', marginRight: 4, borderTop: '2px dashed #D97706' }} /> Markov</span>
        <span><span style={{ display: 'inline-block', width: 16, height: 2, background: BOUND_COLORS.chebyshev, verticalAlign: 'middle', marginRight: 4, borderTop: '2px dashed #534AB7' }} /> Chebyshev</span>
        <span><span style={{ display: 'inline-block', width: 16, height: 2, background: BOUND_COLORS.hoeffding, verticalAlign: 'middle', marginRight: 4, borderTop: '2px dashed #0F6E56' }} /> Chernoff</span>
      </div>
    </div>
  );
}

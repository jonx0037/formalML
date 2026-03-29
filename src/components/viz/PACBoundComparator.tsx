import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 56 };

const BOUND_COLORS = ['#0F6E56', '#534AB7', '#D97706', '#2563EB'] as const;
const BOUND_LABELS = [
  'Realizable finite',
  'Agnostic finite',
  'VC bound',
  'Rademacher',
] as const;

/** Realizable finite: n = ceil(log(|H|/delta) / epsilon) */
function realizableSampleComplexity(eps: number, H: number, delta: number): number {
  return Math.ceil(Math.log(H / delta) / eps);
}

/** Agnostic finite: n = ceil(2 log(2|H|/delta) / epsilon^2) */
function agnosticSampleComplexity(eps: number, H: number, delta: number): number {
  return Math.ceil(2 * Math.log(2 * H / delta) / (eps * eps));
}

/** VC bound epsilon given n: sqrt((8d log(2en/d) + 8 log(4/delta)) / n) */
function vcBoundEpsilon(n: number, d: number, delta: number): number {
  if (n <= 0 || d <= 0 || Math.E * n <= d) return Infinity;
  const val = (8 * d * Math.log(2 * Math.E * n / d) + 8 * Math.log(4 / delta)) / n;
  return val > 0 ? Math.sqrt(val) : Infinity;
}

/** VC bound: invert for sample complexity via binary search */
function vcSampleComplexity(eps: number, d: number, delta: number): number {
  let lo = 1, hi = 1e7;
  for (let i = 0; i < 50; i++) {
    const mid = Math.floor((lo + hi) / 2);
    if (vcBoundEpsilon(mid, d, delta) > eps) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Rademacher bound epsilon given n: 4 sqrt(2d log(en/d)/n) + 3 sqrt(log(2/delta)/(2n)) */
function rademacherBoundEpsilon(n: number, d: number, delta: number): number {
  if (n <= 0 || d <= 0 || Math.E * n <= d) return Infinity;
  const logArg = Math.E * n / d;
  const term1 = 4 * Math.sqrt(2 * d * Math.log(logArg) / n);
  const term2 = 3 * Math.sqrt(Math.log(2 / delta) / (2 * n));
  return term1 + term2;
}

/** Rademacher: invert for sample complexity via binary search */
function rademacherSampleComplexity(eps: number, d: number, delta: number): number {
  let lo = 1, hi = 1e7;
  for (let i = 0; i < 50; i++) {
    const mid = Math.floor((lo + hi) / 2);
    if (rademacherBoundEpsilon(mid, d, delta) > eps) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export default function PACBoundComparator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const [logH, setLogH] = useState(2);
  const [vcDim, setVcDim] = useState(10);
  const [logDelta, setLogDelta] = useState(-1.3);
  const [showBounds, setShowBounds] = useState([true, true, true, true]);

  const H = Math.pow(10, logH);
  const delta = Math.pow(10, logDelta);

  const isDesktop = (containerWidth || 0) > 640;
  const panelW = isDesktop ? Math.floor((containerWidth - 24) / 2) : containerWidth;
  const panelH = Math.min(300, Math.max(220, panelW * 0.65));

  const toggleBound = useCallback((idx: number) => {
    setShowBounds(prev => prev.map((v, i) => i === idx ? !v : v));
  }, []);

  // Precompute curves
  const curves = useMemo(() => {
    // Left panel: sample complexity n vs epsilon
    const epsRange = Array.from({ length: 80 }, (_, i) => 0.01 + (0.49) * i / 79);
    const nCurves = [
      epsRange.map(eps => realizableSampleComplexity(eps, H, delta)),
      epsRange.map(eps => agnosticSampleComplexity(eps, H, delta)),
      epsRange.map(eps => vcSampleComplexity(eps, vcDim, delta)),
      epsRange.map(eps => rademacherSampleComplexity(eps, vcDim, delta)),
    ];

    // Right panel: bound epsilon vs n
    const nRange = Array.from({ length: 100 }, (_, i) => 50 + (9950) * i / 99);
    const epsCurves = [
      nRange.map(n => Math.log(H / delta) / n), // realizable: eps = log(|H|/delta)/n
      nRange.map(n => Math.sqrt(2 * Math.log(2 * H / delta) / n)),
      nRange.map(n => vcBoundEpsilon(n, vcDim, delta)),
      nRange.map(n => rademacherBoundEpsilon(n, vcDim, delta)),
    ];

    return { epsRange, nCurves, nRange, epsCurves };
  }, [H, delta, vcDim]);

  // Left panel: n vs epsilon
  useEffect(() => {
    const svg = leftSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { epsRange, nCurves } = curves;

    const xScale = d3.scaleLinear().domain([0.01, 0.5]).range([MARGIN.left, panelW - MARGIN.right]);
    const nMax = 1e6;
    const yScale = d3.scaleLog().domain([10, nMax]).range([panelH - MARGIN.bottom, MARGIN.top]).clamp(true);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5, '.0e').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('ε (accuracy)');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('n (samples needed)');

    nCurves.forEach((nVals, idx) => {
      if (!showBounds[idx]) return;
      const lineGen = d3.line<number>()
        .defined(d => d >= 10 && d <= nMax)
        .x((_, i) => xScale(epsRange[i]))
        .y(d => yScale(Math.max(10, Math.min(d, nMax))));
      sel.append('path').datum(nVals).attr('d', lineGen)
        .style('fill', 'none').style('stroke', BOUND_COLORS[idx]).style('stroke-width', 2);
    });

    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('Sample Complexity n(ε)');
  }, [curves, showBounds, panelW, panelH]);

  // Right panel: epsilon vs n
  useEffect(() => {
    const svg = rightSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { nRange, epsCurves } = curves;

    const xScale = d3.scaleLog().domain([50, 10000]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([0, 1.5]).range([panelH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(4, ',.0f').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('n (sample size)');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('ε (bound)');

    epsCurves.forEach((epsVals, idx) => {
      if (!showBounds[idx]) return;
      const lineGen = d3.line<number>()
        .defined(d => isFinite(d) && d >= 0 && d <= 1.5)
        .x((_, i) => xScale(nRange[i]))
        .y(d => yScale(Math.min(d, 1.5)));
      sel.append('path').datum(epsVals).attr('d', lineGen)
        .style('fill', 'none').style('stroke', BOUND_COLORS[idx]).style('stroke-width', 2);
    });

    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('Bound ε(n)');
  }, [curves, showBounds, panelW, panelH]);

  const hStr = H >= 1000 ? H.toExponential(0) : H.toFixed(0);

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
        PAC Bound Comparator
      </div>

      {/* Panels */}
      <div style={{ display: 'flex', flexDirection: isDesktop ? 'row' : 'column', gap: '8px' }}>
        <svg role="img" aria-label="PACBound comparator visualization (panel 1 of 2)" ref={leftSvgRef} width={panelW} height={panelH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
        <svg role="img" aria-label="PACBound comparator visualization (panel 2 of 2)" ref={rightSvgRef} width={panelW} height={panelH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
      </div>

      {/* Bound toggles */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', marginTop: '12px' }}>
        {BOUND_LABELS.map((label, idx) => (
          <label key={label} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', cursor: 'pointer' }}>
            <input type="checkbox" checked={showBounds[idx]} onChange={() => toggleBound(idx)}
              style={{ accentColor: BOUND_COLORS[idx] }} />
            <span style={{ width: 12, height: 3, background: BOUND_COLORS[idx], borderRadius: 1, display: 'inline-block' }} />
            {label}
          </label>
        ))}
      </div>

      {/* Sliders */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginTop: '12px' }}>
        <div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            |H| (hypotheses): <strong style={{ color: 'var(--color-text)' }}>{hStr}</strong>
          </div>
          <input type="range" min={1} max={5} step={0.1} value={logH}
            onChange={e => setLogH(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
        </div>
        <div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            VC dim d: <strong style={{ color: 'var(--color-text)' }}>{vcDim}</strong>
          </div>
          <input type="range" min={1} max={50} step={1} value={vcDim}
            onChange={e => setVcDim(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
        </div>
        <div>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', marginBottom: 2 }}>
            δ (failure prob): <strong style={{ color: 'var(--color-text)' }}>{delta.toFixed(4)}</strong>
          </div>
          <input type="range" min={-3} max={-0.3} step={0.05} value={logDelta}
            onChange={e => setLogDelta(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-accent)' }} />
        </div>
      </div>
    </div>
  );
}

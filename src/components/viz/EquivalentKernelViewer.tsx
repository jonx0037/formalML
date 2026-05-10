import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { equivalentKernel, kGaussian, paletteKR } from './shared/kernel-regression';

// =============================================================================
// EquivalentKernelViewer — qualitative shape display of K^*_p^(c)(u) over
// u ∈ [-3.5, 3.5] for p ∈ {0, 1, 2, 3} at user-selected boundary parameter c.
// At c = ∞ (interior), K*_0 = K*_1 = K and K*_2 = K*_3 = ½(3-u²)K; at c = 0
// (strict boundary) all four diverge. Companion to §3.4 / §6.1.
//
// Static fallback: public/images/topics/local-regression/03_equivalent_kernel.png
// =============================================================================

const HEIGHT = 380;
const SM_BREAKPOINT = 640;
const NPTS = 281; // u-grid resolution
const U_MIN = -3.5;
const U_MAX = 3.5;
const C_OPTIONS: Array<{ label: string; value: number }> = [
  { label: 'c = 0  (strict boundary)', value: 0 },
  { label: 'c = 0.5', value: 0.5 },
  { label: 'c = 1.0', value: 1.0 },
  { label: 'c = 2.0', value: 2.0 },
  { label: 'c = ∞  (interior)', value: Infinity },
];
const DEGREE_COLORS: Record<number, string> = {
  0: paletteKR.posterior,
  1: paletteKR.band,
  2: paletteKR.alt,
  3: paletteKR.truth,
};

export default function EquivalentKernelViewer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [cIdx, setCIdx] = useState(4); // default interior
  const [enabled, setEnabled] = useState<Record<number, boolean>>({ 0: true, 1: true, 2: true, 3: true });

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;
  const c = C_OPTIONS[cIdx].value;

  // Precompute K^*_p^(c)(u-grid) for every (p, c) combination once (5 × 4 = 20
  // entries, ~720k Simpson-panel kernel evaluations total at startup, ~36ms).
  // Switching the c-slider then becomes an O(1) lookup, no main-thread jank.
  // Catches PR #80 review feedback (Copilot) on per-c-change recomputation.
  const { uGrid, kStarValuesByCIdx } = useMemo(() => {
    const grid = new Float64Array(NPTS);
    for (let i = 0; i < NPTS; i++) grid[i] = U_MIN + ((U_MAX - U_MIN) * i) / (NPTS - 1);
    const cache: Record<number, Record<number, Float64Array>> = {};
    C_OPTIONS.forEach((opt, ci) => {
      const ksByP: Record<number, Float64Array> = {};
      for (const p of [0, 1, 2, 3]) {
        const Kstar = equivalentKernel(kGaussian, p, opt.value);
        const arr = new Float64Array(NPTS);
        for (let i = 0; i < NPTS; i++) {
          const u = grid[i];
          // Outside [-c, ∞), the kernel is meaningless — set to NaN to break the line.
          arr[i] = isFinite(opt.value) && u < -opt.value ? NaN : Kstar(u);
        }
        ksByP[p] = arr;
      }
      cache[ci] = ksByP;
    });
    return { uGrid: grid, kStarValuesByCIdx: cache };
  }, []);
  const kStarValues = kStarValuesByCIdx[cIdx];

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 24, right: 16, bottom: 40, left: 50 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([U_MIN, U_MAX]).range([0, innerW]);
      // Y-domain: encompass all enabled curves with some padding.
      let yMin = 0, yMax = 0;
      for (const p of [0, 1, 2, 3]) {
        if (!enabled[p]) continue;
        for (const v of kStarValues[p]) {
          if (!isFinite(v)) continue;
          if (v < yMin) yMin = v;
          if (v > yMax) yMax = v;
        }
      }
      const yPad = (yMax - yMin) * 0.08 || 0.05;
      const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).nice().range([innerH, 0]);

      // Zero baseline.
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(0)).attr('y2', yScale(0))
        .style('stroke', 'var(--color-border)').style('stroke-width', 1).style('stroke-dasharray', '3 3');

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)')
        .style('font-size', '12px').text('u  (scaled coordinate)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -36)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '12px').text('K*_p(u)');

      const lineGen = d3
        .line<number>()
        .defined((v) => isFinite(v))
        .x((_, i) => xScale(uGrid[i]))
        .y((v) => yScale(v));

      for (const p of [0, 1, 2, 3]) {
        if (!enabled[p]) continue;
        g.append('path').datum(Array.from(kStarValues[p])).attr('d', lineGen)
          .style('fill', 'none').style('stroke', DEGREE_COLORS[p]).style('stroke-width', 2.2);
      }

      // Boundary marker at u = -c (only when c < ∞).
      if (isFinite(c)) {
        const x0 = xScale(-c);
        if (x0 >= 0 && x0 <= innerW) {
          g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', 0).attr('y2', innerH)
            .style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1)
            .style('stroke-dasharray', '4 4').style('opacity', 0.7);
          g.append('text').attr('x', x0 + 4).attr('y', 12).style('fill', 'var(--color-text-secondary)')
            .style('font-family', 'var(--font-sans)').style('font-size', '10px').text(`u = -c`);
        }
      }
    },
    [w, kStarValues, uGrid, enabled, c],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 10, display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)' }}>boundary parameter:</span>
          {C_OPTIONS.map((opt, i) => (
            <button key={i} type="button" onClick={() => setCIdx(i)}
              style={{
                padding: '4px 10px', border: '1px solid var(--color-border)', borderRadius: 4,
                background: cIdx === i ? paletteKR.posterior : 'var(--color-surface)',
                color: cIdx === i ? 'white' : 'var(--color-text)', cursor: 'pointer',
                fontFamily: 'var(--font-sans)', fontSize: 12,
              }}>{opt.label}</button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)' }}>show p:</span>
          {[0, 1, 2, 3].map((pi) => (
            <label key={pi} style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
              <input type="checkbox" checked={!!enabled[pi]} onChange={(e) => setEnabled({ ...enabled, [pi]: e.target.checked })} />
              <span style={{ color: DEGREE_COLORS[pi], fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 600 }}>p={pi}</span>
            </label>
          ))}
        </div>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} />
    </div>
  );
}

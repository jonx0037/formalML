import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  kGaussian,
  localPolynomialGcv,
  localPolynomialLooCv,
  mulberry32,
  paletteKR,
  sampleToyUni,
} from './shared/kernel-regression';

// =============================================================================
// BandwidthSelectorAtP — LOO-CV and GCV objective curves at p ∈ {0, 1, 3} on
// the §1 toy. Same data, single sample (seed 42), bandwidth grid log-spaced.
// Vertical markers at the LOO-CV minimizer per p.
// =============================================================================

const HEIGHT = 420;
const SM_BREAKPOINT = 640;
const N = 200;
const SIGMA = 0.2;
const H_GRID_PTS = 18;
const DEGREE_COLORS: Record<number, string> = {
  0: paletteKR.posterior,
  1: paletteKR.band,
  3: paletteKR.truth,
};
const DEGREE_LABELS: Record<number, string> = {
  0: 'p = 0  (NW)',
  1: 'p = 1  (local-linear)',
  3: 'p = 3  (local-cubic)',
};

export default function BandwidthSelectorAtP() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [enabled, setEnabled] = useState<Record<number, boolean>>({ 0: true, 1: true, 3: true });
  const [showGcv, setShowGcv] = useState(true);

  const w = containerWidth;
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { hGrid, scores, looMinimizers } = useMemo(() => {
    const grid = new Float64Array(H_GRID_PTS);
    const lo = Math.log(0.04);
    const hi = Math.log(0.40);
    for (let i = 0; i < H_GRID_PTS; i++) grid[i] = Math.exp(lo + ((hi - lo) * i) / (H_GRID_PTS - 1));

    const rng = mulberry32(42);
    const { X, Y } = sampleToyUni(N, SIGMA, rng);

    const sc: Record<number, { loo: Float64Array; gcv: Float64Array }> = {};
    const mins: Record<number, number> = {};
    for (const p of [0, 1, 3]) {
      const loo = new Float64Array(H_GRID_PTS);
      const gcv = new Float64Array(H_GRID_PTS);
      for (let i = 0; i < H_GRID_PTS; i++) {
        loo[i] = localPolynomialLooCv(X, Y, grid[i], p, kGaussian);
        gcv[i] = localPolynomialGcv(X, Y, grid[i], p, kGaussian);
      }
      let minIdx = 0;
      for (let i = 1; i < H_GRID_PTS; i++) if (loo[i] < loo[minIdx]) minIdx = i;
      mins[p] = grid[minIdx];
      sc[p] = { loo, gcv };
    }
    return { hGrid: grid, scores: sc, looMinimizers: mins };
  }, []);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 30, right: 16, bottom: 50, left: 60 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([hGrid[0], hGrid[hGrid.length - 1]]).range([0, innerW]);
      let yMin = Infinity, yMax = -Infinity;
      for (const p of [0, 1, 3]) {
        if (!enabled[p]) continue;
        for (const v of scores[p].loo) {
          if (v < yMin) yMin = v;
          if (v > yMax) yMax = v;
        }
      }
      const yScale = d3.scaleLinear().domain([yMin * 0.95, yMax * 1.05]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6).tickFormat((d) => (d as number).toFixed(2)))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 36).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)')
        .style('font-size', '12px').text('bandwidth h  (log scale)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -46)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '12px').text('CV objective');

      const lineGen = d3.line<number>().x((_, i) => xScale(hGrid[i])).y((v) => yScale(v));

      for (const p of [0, 1, 3]) {
        if (!enabled[p]) continue;
        // LOO solid
        g.append('path').datum(Array.from(scores[p].loo)).attr('d', lineGen)
          .style('fill', 'none').style('stroke', DEGREE_COLORS[p]).style('stroke-width', 2);
        // GCV dashed
        if (showGcv) {
          g.append('path').datum(Array.from(scores[p].gcv)).attr('d', lineGen)
            .style('fill', 'none').style('stroke', DEGREE_COLORS[p]).style('stroke-width', 1.2)
            .style('stroke-dasharray', '4 3').style('opacity', 0.7);
        }
        // LOO minimizer marker
        const xMin = xScale(looMinimizers[p]);
        g.append('line').attr('x1', xMin).attr('x2', xMin).attr('y1', 0).attr('y2', innerH)
          .style('stroke', DEGREE_COLORS[p]).style('stroke-width', 1).style('stroke-dasharray', '2 3')
          .style('opacity', 0.5);
      }

      // Minimizer values legend
      let yCursor = 8;
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 0)`);
      legend.append('text').attr('x', 0).attr('y', yCursor).style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', '11px').style('font-weight', '600')
        .text('LOO-CV minimizers:');
      yCursor += 16;
      for (const p of [0, 1, 3]) {
        if (!enabled[p]) continue;
        legend.append('text').attr('x', 0).attr('y', yCursor)
          .style('fill', DEGREE_COLORS[p])
          .style('font-family', 'var(--font-mono)').style('font-size', '11px')
          .text(`${DEGREE_LABELS[p]}: ĥ ≈ ${looMinimizers[p].toFixed(3)}`);
        yCursor += 14;
      }
    },
    [w, hGrid, scores, looMinimizers, enabled, showGcv],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 10, display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)' }}>show:</span>
          {[0, 1, 3].map((pi) => (
            <label key={pi} style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
              <input type="checkbox" checked={!!enabled[pi]} onChange={(e) => setEnabled({ ...enabled, [pi]: e.target.checked })} />
              <span style={{ color: DEGREE_COLORS[pi], fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 600 }}>p={pi}</span>
            </label>
          ))}
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
          <input type="checkbox" checked={showGcv} onChange={(e) => setShowGcv(e.target.checked)} />
          <span style={{ fontFamily: 'var(--font-sans)', fontSize: 13, color: 'var(--color-text)' }}>show GCV (dashed)</span>
        </label>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} />
    </div>
  );
}

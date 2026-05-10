import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  kGaussian,
  localPolynomial,
  mTrueUni,
  mulberry32,
  paletteKR,
  sampleToyUni,
} from './shared/kernel-regression';

// =============================================================================
// BoundaryBiasUniform — empirical bias profile across x ∈ [0, 1] at fixed h.
// Plots |E[m̂_h(x)] - m(x)| vs x for p ∈ {0, 1, 2, 3} from B replicates.
// MC is small (B = 30, commit-on-release for h) to keep the viz responsive;
// the larger-B static figure (B = 500) lives at 06_bias_profile.png.
// =============================================================================

const HEIGHT = 400;
const SM_BREAKPOINT = 640;
const N = 200;
const SIGMA = 0.2;
const B_MC = 30;
const X_GRID_PTS = 41;
const DEGREE_COLORS: Record<number, string> = {
  0: paletteKR.posterior,
  1: paletteKR.band,
  2: paletteKR.alt,
  3: paletteKR.truth,
};

export default function BoundaryBiasUniform() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [hDisplay, setHDisplay] = useState(0.08);
  const [hCommitted, setHCommitted] = useState(0.08);
  const [enabled, setEnabled] = useState<Record<number, boolean>>({ 0: true, 1: true, 3: true });

  const w = containerWidth;
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const xGrid = useMemo(() => {
    const g = new Float64Array(X_GRID_PTS);
    for (let i = 0; i < X_GRID_PTS; i++) g[i] = i / (X_GRID_PTS - 1);
    return g;
  }, []);

  // For each p, compute empirical bias |E[hat m] - m| by averaging B replicate fits.
  const biasProfiles = useMemo(() => {
    const truths = new Float64Array(X_GRID_PTS);
    for (let i = 0; i < X_GRID_PTS; i++) truths[i] = mTrueUni(xGrid[i]);

    const result: Record<number, Float64Array> = {};
    for (const p of [0, 1, 2, 3]) {
      if (!enabled[p]) continue;
      const meanFit = new Float64Array(X_GRID_PTS);
      const rng = mulberry32(42 + p);
      for (let b = 0; b < B_MC; b++) {
        const { X, Y } = sampleToyUni(N, SIGMA, rng);
        const fit = localPolynomial(X, Y, xGrid, hCommitted, p, kGaussian);
        for (let i = 0; i < X_GRID_PTS; i++) meanFit[i] += fit[i];
      }
      for (let i = 0; i < X_GRID_PTS; i++) meanFit[i] /= B_MC;
      const bias = new Float64Array(X_GRID_PTS);
      for (let i = 0; i < X_GRID_PTS; i++) bias[i] = Math.abs(meanFit[i] - truths[i]);
      result[p] = bias;
    }
    return result;
  }, [hCommitted, xGrid, enabled]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 24, right: 16, bottom: 50, left: 56 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      let yMax = 0;
      for (const p of [0, 1, 2, 3]) if (biasProfiles[p]) for (const v of biasProfiles[p]) if (v > yMax) yMax = v;
      yMax = Math.max(yMax * 1.15, 0.05);
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 36).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)')
        .style('font-size', '12px').text('x  (evaluation point)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -42)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '12px').text('|empirical bias|');

      const lineGen = d3.line<number>().x((_, i) => xScale(xGrid[i])).y((v) => yScale(v));
      for (const p of [0, 1, 2, 3]) {
        if (!biasProfiles[p]) continue;
        g.append('path').datum(Array.from(biasProfiles[p])).attr('d', lineGen)
          .style('fill', 'none').style('stroke', DEGREE_COLORS[p]).style('stroke-width', 2);
      }

      // Boundary shading
      g.append('rect').attr('x', 0).attr('y', 0).attr('width', xScale(hCommitted)).attr('height', innerH)
        .style('fill', 'var(--color-border)').style('opacity', 0.2);
      g.append('rect').attr('x', xScale(1 - hCommitted)).attr('y', 0)
        .attr('width', innerW - xScale(1 - hCommitted)).attr('height', innerH)
        .style('fill', 'var(--color-border)').style('opacity', 0.2);
    },
    [w, biasProfiles, xGrid, hCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 10, display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
          <label style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)', whiteSpace: 'nowrap' }}>
            h = <strong>{hDisplay.toFixed(3)}</strong>
          </label>
          <input type="range" min={0.04} max={0.20} step={0.005} value={hDisplay}
            onChange={(e) => setHDisplay(parseFloat(e.target.value))}
            onMouseUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
            aria-label="bandwidth h" style={{ flex: 1, minWidth: 180 }} />
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          {[0, 1, 2, 3].map((pi) => (
            <label key={pi} style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
              <input type="checkbox" checked={!!enabled[pi]} onChange={(e) => setEnabled({ ...enabled, [pi]: e.target.checked })} />
              <span style={{ color: DEGREE_COLORS[pi], fontFamily: 'var(--font-sans)', fontSize: 13, fontWeight: 600 }}>p={pi}</span>
            </label>
          ))}
        </div>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} />
      <p style={{ fontFamily: 'var(--font-sans)', fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 8 }}>
        Shaded regions show one bandwidth h on each side of the support boundary (where the kernel mass starts to truncate).
        B = {B_MC} MC replicates per p (live); the 06_bias_profile.png static figure uses B = 500 for cleaner curves.
      </p>
    </div>
  );
}

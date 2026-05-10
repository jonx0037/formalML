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
  silvermanKernel,
} from './shared/kernel-regression';

// =============================================================================
// SmoothingSplineBridge — overlay local-cubic (p = 3) and Silverman-kernel
// smoother on the §1 toy. Per Silverman's (1984) equivalent-variable-kernel
// theorem, the smoothing spline is asymptotically equivalent to the K_S
// kernel smoother — and at moderate n, the two fits visually coincide for
// well-matched bandwidths.
//
// Slider: bandwidth h shared between the two estimators (Silverman uses
// h_silv = 1.4 * h as a rule-of-thumb scaling that approximately matches
// the local-cubic effective df at this n).
//
// Note: A genuine cubic smoothing spline (Reinsch algorithm) is non-trivial
// to implement client-side and outside the scope of this PR; the K_S kernel
// smoother is the asymptotic-equivalent stand-in promised by Silverman 1984.
// =============================================================================

const HEIGHT = 420;
const SM_BREAKPOINT = 640;
const N = 200;
const SIGMA = 0.2;

// Silverman kernel smoother — Nadaraya–Watson with K_S as the kernel.
function silvermanSmoother(
  X: Float64Array, Y: Float64Array, xEval: Float64Array, h: number,
): Float64Array {
  const n = X.length;
  const out = new Float64Array(xEval.length);
  for (let g = 0; g < xEval.length; g++) {
    const xg = xEval[g];
    let num = 0; let den = 0;
    for (let i = 0; i < n; i++) {
      const w = silvermanKernel((X[i] - xg) / h) / h;
      num += w * Y[i];
      den += w;
    }
    out[g] = den !== 0 ? num / den : NaN;
  }
  return out;
}

export default function SmoothingSplineBridge() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [hDisplay, setHDisplay] = useState(0.10);
  const [hCommitted, setHCommitted] = useState(0.10);

  const w = containerWidth;
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { X, Y, xGrid, mTrue } = useMemo(() => {
    const rng = mulberry32(42);
    const sample = sampleToyUni(N, SIGMA, rng);
    const grid = new Float64Array(201);
    for (let g = 0; g < 201; g++) grid[g] = g / 200;
    const truth = new Float64Array(201);
    for (let g = 0; g < 201; g++) truth[g] = mTrueUni(grid[g]);
    return { X: sample.X, Y: sample.Y, xGrid: grid, mTrue: truth };
  }, []);

  const { lcFit, splFit, gap } = useMemo(() => {
    const lc = localPolynomial(X, Y, xGrid, hCommitted, 3, kGaussian);
    // Silverman scaling: K_S has different effective bandwidth than Gaussian. Empirical match: h_silv ≈ 1.4 h.
    const spl = silvermanSmoother(X, Y, xGrid, 1.4 * hCommitted);
    let maxGap = 0;
    let meanGap = 0;
    let countValid = 0;
    for (let g = 0; g < xGrid.length; g++) {
      if (!isFinite(spl[g])) continue;
      const d = Math.abs(lc[g] - spl[g]);
      if (d > maxGap) maxGap = d;
      meanGap += d;
      countValid++;
    }
    return { lcFit: lc, splFit: spl, gap: { max: maxGap, mean: meanGap / countValid } };
  }, [X, Y, xGrid, hCommitted]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 30, right: 16, bottom: 40, left: 50 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-1.4, 1.6]).nice().range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');

      g.append('g').selectAll('circle').data(Array.from(X).map((xi, i) => ({ x: xi, y: Y[i] })))
        .enter().append('circle')
        .attr('cx', (d) => xScale(d.x)).attr('cy', (d) => yScale(d.y))
        .attr('r', 1.8).style('fill', paletteKR.data).style('opacity', 0.5);

      const lineGen = d3.line<number>()
        .defined((v) => isFinite(v))
        .x((_, i) => xScale(xGrid[i]))
        .y((v) => yScale(v));
      g.append('path').datum(Array.from(mTrue)).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteKR.truth)
        .style('stroke-width', 1.4).style('stroke-dasharray', '5 3').style('opacity', 0.85);
      g.append('path').datum(Array.from(lcFit)).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteKR.posterior).style('stroke-width', 2.4);
      g.append('path').datum(Array.from(splFit)).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteKR.alt).style('stroke-width', 1.6).style('stroke-dasharray', '3 2');

      // Title
      svg.append('text').attr('x', margin.left).attr('y', 16)
        .style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)')
        .style('font-size', '13px').text(`Local-cubic vs Silverman-kernel — mean |gap| ≈ ${gap.mean.toFixed(3)},  max ≈ ${gap.max.toFixed(3)}`);

      // Legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 240}, 10)`);
      const items = [
        { color: paletteKR.truth, dash: '5 3', label: 'true m(x)' },
        { color: paletteKR.posterior, dash: 'none', label: 'local-cubic (p = 3)' },
        { color: paletteKR.alt, dash: '3 2', label: 'Silverman kernel K_S' },
      ];
      items.forEach((it, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 22).attr('y1', 16 * i + 4).attr('y2', 16 * i + 4)
          .style('stroke', it.color).style('stroke-width', 2).style('stroke-dasharray', it.dash);
        legend.append('text').attr('x', 28).attr('y', 16 * i + 8)
          .style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)')
          .style('font-size', '11px').text(it.label);
      });
    },
    [w, X, Y, xGrid, mTrue, lcFit, splFit, gap],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <label style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)' }}>
          local-cubic bandwidth h = <strong>{hDisplay.toFixed(3)}</strong>
          <span style={{ color: 'var(--color-text-secondary)', fontSize: 12, marginLeft: 8 }}>
            (Silverman h_S ≈ {(1.4 * hDisplay).toFixed(3)}, asymptotic-match scaling)
          </span>
        </label>
        <input type="range" min={0.04} max={0.20} step={0.005} value={hDisplay}
          onChange={(e) => setHDisplay(parseFloat(e.target.value))}
          onMouseUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          aria-label="bandwidth h" style={{ flex: 1, minWidth: 200 }} />
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} />
      <p style={{ fontFamily: 'var(--font-sans)', fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 8 }}>
        The Silverman kernel K_S is asymptotically equivalent to the cubic smoothing spline (Silverman 1984).
        We use it as a stand-in here — a true Reinsch-algorithm spline implementation is outside this PR's scope;
        the static figure 10_local_cubic_vs_smoothing_spline.png shows the comparison against scipy's `make_smoothing_spline`.
      </p>
    </div>
  );
}

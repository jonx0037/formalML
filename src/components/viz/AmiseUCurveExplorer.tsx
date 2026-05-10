import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { hStarAmiseUni, KERNEL_CONSTANTS, paletteKR } from './shared/kernel-regression';

// =============================================================================
// AmiseUCurveExplorer — interactive companion to §4.4's AMISE U-curve figure.
// Plots three curves on log-log: bias²(h) ∝ h⁴, variance(h) ∝ 1/(nh), and
// AMISE(h) = bias² + variance, with vertical marker at the analytical h^* and
// horizontal marker at AMISE(h^*).
//
// Closed-form analytical expressions for the §1 toy (X ~ Uniform[0,1],
// m(x) = sin(2πx) + x/2):
//   bias²(h)    = (h⁴ / 4) · μ_2(K)² · θ_{m,f}      with θ_{m,f} = 8π⁴
//   variance(h) = R(K) · ν_σ / (n · h)              with ν_σ = σ²
//   h^*         = (R(K) · ν_σ / (μ_2² · θ_{m,f} · n))^(1/5)
//
// Numerical anchor: at n = 200, σ = 0.2, h^* ≈ 0.0373 (notebook §4.4).
//
// Static fallback: public/images/topics/kernel-regression/09_amise_ucurve.png
// =============================================================================

const HEIGHT = 420;
const SM_BREAKPOINT = 640;
const SIGMA = 0.2;
const N_OPTIONS = [50, 100, 200, 500, 1000, 2000, 5000];
const THETA_MF = 8 * Math.pow(Math.PI, 4);

export default function AmiseUCurveExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(2); // default n = 200

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;
  const N = N_OPTIONS[nIdx];

  const { mu2, R } = KERNEL_CONSTANTS.gaussian;

  const { hGrid, biasSq, variance, amise, hStar, amiseAtHStar } = useMemo(() => {
    const NPTS = 200;
    const lo = Math.log(0.005);
    const hi = Math.log(0.5);
    const grid = new Float64Array(NPTS);
    const bs = new Float64Array(NPTS);
    const vr = new Float64Array(NPTS);
    const am = new Float64Array(NPTS);
    const nuSigma = SIGMA * SIGMA;
    // Hoist coefficients out of the loop — they're invariant in h.
    const bsCoeff = (mu2 * mu2 * THETA_MF) / 4;
    const vrCoeff = (R * nuSigma) / N;
    for (let i = 0; i < NPTS; i++) {
      const h = Math.exp(lo + ((hi - lo) * i) / (NPTS - 1));
      grid[i] = h;
      bs[i] = Math.pow(h, 4) * bsCoeff;
      vr[i] = vrCoeff / h;
      am[i] = bs[i] + vr[i];
    }
    const hS = hStarAmiseUni(N, SIGMA);
    const amiseAt = Math.pow(hS, 4) * bsCoeff + vrCoeff / hS;
    return { hGrid: grid, biasSq: bs, variance: vr, amise: am, hStar: hS, amiseAtHStar: amiseAt };
  }, [N, mu2, R]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 28, right: 16, bottom: 50, left: 60 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLog().domain([hGrid[0], hGrid[hGrid.length - 1]]).range([0, innerW]);
      // Y range: cover all three curves, with a floor to keep the log scale finite.
      const yMin = Math.max(1e-6, Math.min(amiseAtHStar / 4, biasSq[0], variance[variance.length - 1]) / 4);
      const yMax = Math.max(amise[0], amise[amise.length - 1]) * 1.2;
      const yScale = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);

      const draw = (data: Float64Array, color: string, dash: string | null, width: number) => {
        const line = d3
          .line<number>()
          .defined((d) => d > 0 && Number.isFinite(d))
          .x((_, i) => xScale(hGrid[i]))
          .y((d) => yScale(d));
        g.append('path')
          .datum(Array.from(data))
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', color)
          .style('stroke-width', width)
          .style('stroke-dasharray', dash ?? 'none');
      };

      draw(biasSq, paletteKR.band, '5 3', 1.6); // bias²
      draw(variance, paletteKR.alt, '5 3', 1.6); // variance
      draw(amise, paletteKR.posterior, null, 2.2); // AMISE

      // h* marker.
      g.append('line')
        .attr('x1', xScale(hStar))
        .attr('y1', 0)
        .attr('x2', xScale(hStar))
        .attr('y2', innerH)
        .style('stroke', paletteKR.truth)
        .style('stroke-dasharray', '3 3')
        .style('stroke-width', 1.4);
      // AMISE(h*) marker.
      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(amiseAtHStar))
        .attr('x2', innerW)
        .attr('y2', yScale(amiseAtHStar))
        .style('stroke', paletteKR.truth)
        .style('stroke-dasharray', '3 3')
        .style('stroke-width', 1.0)
        .style('opacity', 0.5);
      g.append('circle')
        .attr('cx', xScale(hStar))
        .attr('cy', yScale(amiseAtHStar))
        .attr('r', 5)
        .style('fill', paletteKR.truth)
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', 1.2);

      // Axes.
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6, '.2g'))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6, '.1e'))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .text('bandwidth h (log scale)');
      g.append('text')
        .attr('x', -42)
        .attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle')
        .style('font-size', '11px')
        .attr('transform', `rotate(-90,-42,${innerH / 2})`)
        .text('error (log scale)');

      svg
        .append('text')
        .attr('x', w / 2)
        .attr('y', 18)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text(`AMISE U-curve, n = ${N}:  h^* = ${hStar.toFixed(4)},  AMISE(h^*) = ${amiseAtHStar.toExponential(2)}`);

      // Legend.
      const legendX = 8;
      const legendY = 8;
      const legendG = g.append('g').attr('transform', `translate(${legendX}, ${legendY})`);
      const items = [
        { color: paletteKR.posterior, label: 'AMISE = bias² + var', dash: null, width: 2.2 },
        { color: paletteKR.band, label: 'bias² ∝ h⁴', dash: '5 3', width: 1.6 },
        { color: paletteKR.alt, label: 'variance ∝ 1/(nh)', dash: '5 3', width: 1.6 },
        { color: paletteKR.truth, label: `h^* = ${hStar.toFixed(4)}`, dash: '3 3', width: 1.4 },
      ];
      items.forEach((it, i) => {
        const yOff = i * 14;
        legendG
          .append('line')
          .attr('x1', 0)
          .attr('y1', yOff)
          .attr('x2', 16)
          .attr('y2', yOff)
          .style('stroke', it.color)
          .style('stroke-width', it.width)
          .style('stroke-dasharray', it.dash ?? 'none');
        legendG
          .append('text')
          .attr('x', 22)
          .attr('y', yOff + 3)
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '10px')
          .text(it.label);
      });
    },
    [w, hGrid, biasSq, variance, amise, hStar, amiseAtHStar, N],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
    >
      <div
        className="flex flex-wrap items-center gap-4 mb-4 text-sm"
        style={{ flexDirection: isMobile ? 'column' : 'row', alignItems: isMobile ? 'stretch' : 'center' }}
      >
        <label className="flex items-center gap-2 flex-1 min-w-[220px]">
          <span className="text-[var(--color-text-secondary)] whitespace-nowrap">n: {N}</span>
          <input
            type="range"
            min={0}
            max={N_OPTIONS.length - 1}
            step={1}
            value={nIdx}
            onChange={(e) => setNIdx(Number(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
          />
        </label>
        <span className="ml-auto text-xs text-[var(--color-text-secondary)] font-mono">
          h^* ∝ n^(-1/5),  AMISE(h^*) ∝ n^(-4/5)
        </span>
      </div>

      <svg ref={renderRef} width={w} height={HEIGHT} />

      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        Two competing rates set the optimal bandwidth: bias² grows as h⁴ (slope +4 on log-log),
        variance falls as 1/(nh) (slope −1). Their sum bottoms out at h^* ∝ n^(-1/5). As n grows,
        the variance curve drops, h^* shifts left, and AMISE(h^*) shrinks at the slow
        nonparametric rate n^(-4/5) — much slower than the parametric n^(-1).
      </p>
    </div>
  );
}

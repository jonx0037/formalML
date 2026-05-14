import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  fillIsotropicGaussian,
  gaussianRng,
  mpAtomZero,
  mpDensity,
  mpDensityGrid,
  mpSupport,
  mulberry32,
  outerGramMatrix,
  symEigJacobi,
} from './shared/double-descent';

// =============================================================================
// MarchenkoPasturAnimator — §5.4 (Figure 5)
//
// Single panel: empirical eigenvalue histogram with analytic MP density overlay.
// γ slider live (closed-form density, no recompute); n slider commit-on-release.
// Static fallback: 05_marchenko_pastur_densities.png
// =============================================================================

const HEIGHT = 420;

export default function MarchenkoPasturAnimator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [gamma, setGamma] = useState(1.0);
  const [nDisplay, setNDisplay] = useState(500);
  const [nCommitted, setNCommitted] = useState(500);
  const [showSupport, setShowSupport] = useState(true);
  const [showAtom, setShowAtom] = useState(true);

  // Sample eigenvalues from a finite-n realization at the committed n and the live γ.
  const empirical = useMemo(() => {
    const n = nCommitted;
    const p = Math.max(1, Math.round(gamma * n));
    const rng = mulberry32(12345);
    const gauss = gaussianRng(rng);
    // For γ > 1, the n×n matrix XX^T / n has the nonzero eigenvalues; for γ ≤ 1,
    // we use X^T X / n on p×p. Choose whichever is smaller for efficiency.
    if (p <= n) {
      const X = new Float64Array(n * p);
      fillIsotropicGaussian(X, n, p, gauss);
      const G = new Float64Array(p * p);
      for (let i = 0; i < p; i++) {
        for (let j = i; j < p; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += X[k * p + i] * X[k * p + j];
          G[i * p + j] = s / n;
          G[j * p + i] = s / n;
        }
      }
      const { values } = symEigJacobi(G, p);
      return Array.from(values).filter((v) => v >= 0);
    } else {
      // γ > 1: use X X^T / n (n×n)
      const X = new Float64Array(n * p);
      fillIsotropicGaussian(X, n, p, gauss);
      const G = outerGramMatrix(X, n, p);
      for (let i = 0; i < n * n; i++) G[i] /= n;
      const { values } = symEigJacobi(G, n);
      return Array.from(values).filter((v) => v >= 0);
    }
  }, [gamma, nCommitted]);

  // Analytic MP density on a fine grid.
  const analytic = useMemo(() => {
    const grid = mpDensityGrid(gamma, 200);
    return Array.from({ length: grid.lambda.length }, (_, i) => ({
      lambda: grid.lambda[i],
      f: grid.f[i],
    }));
  }, [gamma]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 32, bottom: 52, left: 60 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const { lamMinus, lamPlus } = mpSupport(gamma);
      const xMax = Math.max(lamPlus + 0.5, 6);
      const xScale = d3.scaleLinear().domain([0, xMax]).range([0, W]);

      // Histogram of empirical eigenvalues over the same range.
      // For γ > 1, the sample contains only the n nonzero eigenvalues of XXᵀ/n
      // (the dual matrix), so a histogram normalized by sample size would
      // integrate to 1 — but the MP bulk density only integrates to 1/γ in
      // that regime (the remaining 1 - 1/γ mass is the atom at zero). Scale
      // the histogram densities by 1/γ in the γ > 1 case so the empirical
      // bars overlay the analytic bulk consistently.
      const nBins = 60;
      const bins = d3.bin<number, number>().domain([0, xMax]).thresholds(nBins)(empirical);
      const binW = xMax / nBins;
      const total = empirical.length || 1;
      const bulkMass = gamma > 1 ? 1 / gamma : 1;
      const densities = bins.map((b) => (b.length / (total * binW)) * bulkMass);
      const yMax = Math.max(d3.max(densities) ?? 1, d3.max(analytic, (d) => d.f) ?? 1) * 1.15;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([H, 0]);

      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xScale).ticks(8))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').text('eigenvalue λ');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -44)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px')
        .text('density');

      // Histogram bars
      g.selectAll('rect.bin').data(bins).enter().append('rect')
        .attr('x', (d) => xScale(d.x0!))
        .attr('y', (d, i) => yScale(densities[i]))
        .attr('width', (d) => Math.max(0, xScale(d.x1!) - xScale(d.x0!) - 1))
        .attr('height', (d, i) => H - yScale(densities[i]))
        .style('fill', 'var(--color-accent)').attr('opacity', 0.35);

      // MP analytic curve
      const lineGen = d3.line<{ lambda: number; f: number }>()
        .x((d) => xScale(d.lambda)).y((d) => yScale(d.f));
      g.append('path').datum(analytic).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#9B1D20').attr('stroke-width', 2.2);

      // Support endpoints
      if (showSupport) {
        [lamMinus, lamPlus].forEach((x, idx) => {
          g.append('line').attr('x1', xScale(x)).attr('x2', xScale(x)).attr('y1', 0).attr('y2', H)
            .style('stroke', '#534AB7').attr('stroke-dasharray', '4 4').attr('opacity', 0.6);
          g.append('text').attr('x', xScale(x) + 4).attr('y', 14)
            .style('fill', '#534AB7').style('font-size', '10px')
            .text(idx === 0 ? `λ₋ = ${lamMinus.toFixed(3)}` : `λ₊ = ${lamPlus.toFixed(3)}`);
        });
      }

      // Atom at λ = 0 for γ > 1
      const atom = mpAtomZero(gamma);
      if (showAtom && atom > 0) {
        const atomH = atom * (H * 0.85);
        g.append('rect').attr('x', xScale(0) - 4).attr('y', H - atomH)
          .attr('width', 8).attr('height', atomH).style('fill', '#D97706').attr('opacity', 0.8);
        g.append('text').attr('x', xScale(0) + 8).attr('y', H - atomH - 4)
          .style('fill', '#D97706').style('font-size', '11px')
          .text(`atom @ 0: ${atom.toFixed(2)}`);
      }
    },
    [containerWidth, empirical, analytic, gamma, showSupport, showAtom],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Marchenko-Pastur empirical histogram with analytic density overlay." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 220 }}>
          <span>γ = p/n: <strong>{gamma.toFixed(2)}</strong></span>
          <input type="range" min={0.1} max={4.0} step={0.05} value={gamma}
            onChange={(e) => setGamma(parseFloat(e.target.value))} aria-label="Aspect ratio gamma" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>n: <strong>{nDisplay}</strong></span>
          <input type="range" min={50} max={400} step={25} value={nDisplay}
            onChange={(e) => setNDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setNCommitted(nDisplay)} onTouchEnd={() => setNCommitted(nDisplay)}
            onKeyUp={() => setNCommitted(nDisplay)} aria-label="Sample size n" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showSupport} onChange={(e) => setShowSupport(e.target.checked)} />
          <span>show support endpoints</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showAtom} onChange={(e) => setShowAtom(e.target.checked)} />
          <span>show λ = 0 atom (when γ &gt; 1)</span>
        </label>
      </div>
    </div>
  );
}

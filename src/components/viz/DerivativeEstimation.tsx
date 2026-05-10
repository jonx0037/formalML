import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  kGaussian,
  localPolynomialCoefs,
  mTrueUni,
  mulberry32,
  paletteKR,
  sampleToyUni,
} from './shared/kernel-regression';

// =============================================================================
// DerivativeEstimation — read off m̂, m̂', m̂'', m̂''' from a single local-cubic
// (p = 3) fit at user-selected x₀. Compares the WLS-coefficient estimates to
// the analytical derivatives of m(x) = sin(2πx) + x/2.
//
// Truth at general x:
//   m(x)    = sin(2πx) + x/2
//   m'(x)   = 2π cos(2πx) + 1/2
//   m''(x)  = -4π² sin(2πx)
//   m'''(x) = -8π³ cos(2πx)
// =============================================================================

const HEIGHT = 360;
const SM_BREAKPOINT = 720;
const N = 200;
const SIGMA = 0.2;

function truthAt(x: number, j: number): number {
  switch (j) {
    case 0: return Math.sin(2 * Math.PI * x) + x / 2;
    case 1: return 2 * Math.PI * Math.cos(2 * Math.PI * x) + 0.5;
    case 2: return -4 * Math.PI * Math.PI * Math.sin(2 * Math.PI * x);
    case 3: return -8 * Math.PI * Math.PI * Math.PI * Math.cos(2 * Math.PI * x);
    default: return NaN;
  }
}

export default function DerivativeEstimation() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [x0, setX0] = useState(0.125);
  const [h, setH] = useState(0.15);

  const w = containerWidth;
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { X, Y, fitCurve, mTrueCurve, xGrid } = useMemo(() => {
    const rng = mulberry32(42);
    const sample = sampleToyUni(N, SIGMA, rng);
    const grid = new Float64Array(201);
    for (let g = 0; g < 201; g++) grid[g] = g / 200;
    const truth = new Float64Array(201);
    for (let g = 0; g < 201; g++) truth[g] = mTrueUni(grid[g]);
    return { X: sample.X, Y: sample.Y, fitCurve: null, mTrueCurve: truth, xGrid: grid };
  }, []);
  void fitCurve;

  // Coefficients at x0 (for derivative readout) and full curve (for plot).
  const { coefs, fitGrid } = useMemo(() => {
    const single = new Float64Array([x0]);
    const cf = localPolynomialCoefs(X, Y, single, h, 3, kGaussian);
    const fg = (() => {
      const fit = new Float64Array(201);
      const fullCoefs = localPolynomialCoefs(X, Y, xGrid, h, 3, kGaussian);
      for (let g = 0; g < 201; g++) fit[g] = fullCoefs[g * 4 + 0];
      return fit;
    })();
    return { coefs: cf, fitGrid: fg };
  }, [X, Y, xGrid, x0, h]);

  // Derivative table: m^(j) = j! * beta_j.
  const factorials = [1, 1, 2, 6];
  const tableRows = [0, 1, 2, 3].map((j) => ({
    j,
    estimate: factorials[j] * coefs[j],
    truth: truthAt(x0, j),
  }));

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const margin = { top: 24, right: 16, bottom: 40, left: 50 };
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
        .attr('r', 1.6).style('fill', paletteKR.data).style('opacity', 0.45);

      const lineGen = d3.line<number>().x((_, i) => xScale(xGrid[i])).y((v) => yScale(v));
      g.append('path').datum(Array.from(mTrueCurve)).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteKR.truth)
        .style('stroke-width', 1.4).style('stroke-dasharray', '5 3').style('opacity', 0.85);
      g.append('path').datum(Array.from(fitGrid)).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteKR.posterior).style('stroke-width', 2);

      // x_0 marker
      g.append('line').attr('x1', xScale(x0)).attr('x2', xScale(x0))
        .attr('y1', 0).attr('y2', innerH).style('stroke', paletteKR.accent)
        .style('stroke-width', 1.5).style('stroke-dasharray', '4 3').style('opacity', 0.7);
      g.append('circle').attr('cx', xScale(x0)).attr('cy', yScale(coefs[0]))
        .attr('r', 5).style('fill', paletteKR.posterior).style('stroke', 'white').style('stroke-width', 1.5);
    },
    [w, X, Y, xGrid, mTrueCurve, fitGrid, x0, coefs],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 10, display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
          <label style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)', whiteSpace: 'nowrap' }}>
            x₀ = <strong>{x0.toFixed(3)}</strong>
          </label>
          <input type="range" min={0.05} max={0.95} step={0.005} value={x0}
            onChange={(e) => setX0(parseFloat(e.target.value))}
            aria-label="evaluation point x_0" style={{ flex: 1, minWidth: 160 }} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
          <label style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)', whiteSpace: 'nowrap' }}>
            h = <strong>{h.toFixed(3)}</strong>
          </label>
          <input type="range" min={0.06} max={0.30} step={0.005} value={h}
            onChange={(e) => setH(parseFloat(e.target.value))}
            aria-label="bandwidth h" style={{ flex: 1, minWidth: 160 }} />
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <div style={{ flex: 2 }}>
          <svg ref={renderRef} width={w * (isMobile ? 1 : 0.65)} height={HEIGHT} />
        </div>
        <div style={{ flex: 1, paddingTop: 10 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: 'var(--color-text)' }}>j</th>
                <th style={{ textAlign: 'right', padding: '6px 4px', color: 'var(--color-text)' }}>m̂^(j)(x₀)</th>
                <th style={{ textAlign: 'right', padding: '6px 4px', color: 'var(--color-text)' }}>m^(j)(x₀)</th>
                <th style={{ textAlign: 'right', padding: '6px 4px', color: 'var(--color-text)' }}>err</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row) => {
                const err = Math.abs(row.estimate - row.truth);
                return (
                  <tr key={row.j} style={{ borderBottom: '1px solid var(--color-border)' }}>
                    <td style={{ padding: '4px', color: 'var(--color-text)' }}>{row.j}</td>
                    <td style={{ padding: '4px', textAlign: 'right', color: paletteKR.posterior }}>{row.estimate.toFixed(3)}</td>
                    <td style={{ padding: '4px', textAlign: 'right', color: paletteKR.truth }}>{row.truth.toFixed(3)}</td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--color-text-secondary)' }}>{err.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p style={{ fontFamily: 'var(--font-sans)', fontSize: 11, color: 'var(--color-text-secondary)', marginTop: 8 }}>
            Variance grows by ≈1/h² per derivative order — m̂'' is much noisier than m̂, m̂''' worse still.
          </p>
        </div>
      </div>
    </div>
  );
}

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
// LocalPolynomialFit — main interactive estimator for §2. Single panel showing
// degree-p local polynomial fit on the §1 toy. Bandwidth slider with
// commit-on-release; degree toggle for p ∈ {0, 1, 2, 3}.
//
// Same data, same kernel as DegreeLadderExplorer (seed 42, n = 200, σ = 0.2).
// =============================================================================

const HEIGHT = 420;
const SM_BREAKPOINT = 640;
const N_SAMPLE = 200;
const SIGMA = 0.2;

export default function LocalPolynomialFit() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [p, setP] = useState<0 | 1 | 2 | 3>(1);
  const [hDisplay, setHDisplay] = useState(0.08);
  const [hCommitted, setHCommitted] = useState(0.08);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const { X, Y, xGrid, mTrue } = useMemo(() => {
    const rng = mulberry32(42);
    const sample = sampleToyUni(N_SAMPLE, SIGMA, rng);
    const grid = new Float64Array(201);
    for (let g = 0; g < 201; g++) grid[g] = g / 200;
    const truth = new Float64Array(201);
    for (let g = 0; g < 201; g++) truth[g] = mTrueUni(grid[g]);
    return { X: sample.X, Y: sample.Y, xGrid: grid, mTrue: truth };
  }, []);

  const fit = useMemo(
    () => localPolynomial(X, Y, xGrid, hCommitted, p, kGaussian),
    [X, Y, xGrid, hCommitted, p],
  );

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

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '12px').text('x');
      g.append('text').attr('transform', `rotate(-90)`).attr('x', -innerH / 2).attr('y', -36)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '12px').text('Y, m̂(x)');

      // Scatter.
      g.append('g').selectAll('circle').data(Array.from(X).map((xi, i) => ({ x: xi, y: Y[i] })))
        .enter().append('circle')
        .attr('cx', (d) => xScale(d.x)).attr('cy', (d) => yScale(d.y))
        .attr('r', 2).style('fill', paletteKR.data).style('opacity', 0.55);

      const lineGen = d3.line<number>().x((_, i) => xScale(xGrid[i])).y((d) => yScale(d));
      g.append('path').datum(Array.from(mTrue)).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteKR.truth)
        .style('stroke-width', 1.6).style('stroke-dasharray', '5 3').style('opacity', 0.85);
      g.append('path').datum(Array.from(fit)).attr('d', lineGen)
        .style('fill', 'none').style('stroke', paletteKR.posterior).style('stroke-width', 2.4);

      // Legend.
      const legendY = 8;
      const swatchW = 22;
      const items = [
        { color: paletteKR.truth, dash: '5 3', label: 'true m(x)' },
        { color: paletteKR.posterior, dash: 'none', label: `m̂_h(x) at p = ${p}` },
        { color: paletteKR.data, dash: 'none', label: 'data', dot: true },
      ];
      let xCursor = 0;
      const lg = g.append('g').attr('transform', `translate(0, ${-legendY})`);
      items.forEach((it) => {
        if (it.dot) {
          lg.append('circle').attr('cx', xCursor + 6).attr('cy', 4).attr('r', 3)
            .style('fill', it.color).style('opacity', 0.6);
        } else {
          lg.append('line').attr('x1', xCursor).attr('x2', xCursor + swatchW)
            .attr('y1', 4).attr('y2', 4).style('stroke', it.color)
            .style('stroke-width', 2).style('stroke-dasharray', it.dash);
        }
        lg.append('text').attr('x', xCursor + swatchW + 6).attr('y', 8)
          .style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)')
          .style('font-size', '11px').text(it.label);
        xCursor += swatchW + 6 + it.label.length * 6.5 + 16;
      });
    },
    [w, fit, X, Y, xGrid, mTrue, p],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 10, display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: isMobile ? 'stretch' : 'center' }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)' }}>degree p:</span>
          {([0, 1, 2, 3] as const).map((pi) => (
            <button key={pi} type="button" onClick={() => setP(pi)}
              style={{
                padding: '4px 12px', border: '1px solid var(--color-border)', borderRadius: 4,
                background: p === pi ? paletteKR.posterior : 'var(--color-surface)',
                color: p === pi ? 'white' : 'var(--color-text)', cursor: 'pointer',
                fontFamily: 'var(--font-sans)', fontSize: 13,
              }}>{pi}</button>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
          <label style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)', whiteSpace: 'nowrap' }}>
            h = <strong>{hDisplay.toFixed(3)}</strong>
          </label>
          <input type="range" min={0.02} max={0.30} step={0.005} value={hDisplay}
            onChange={(e) => setHDisplay(parseFloat(e.target.value))}
            onMouseUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
            aria-label="bandwidth h" style={{ flex: 1, minWidth: 180 }} />
        </div>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} />
    </div>
  );
}

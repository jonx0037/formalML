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
// DegreeLadderExplorer — 4-panel comparison of NW (p=0), local-linear (p=1),
// local-quadratic (p=2), and local-cubic (p=3) fits at the same bandwidth on
// the §1 toy. Drag the bandwidth slider to see all four fits update.
//
// Static fallback: public/images/topics/local-regression/01_section1_teaser.png
// =============================================================================

const HEIGHT_PANEL = 220;
const SM_BREAKPOINT = 640;
const N_SAMPLE = 200;
const SIGMA = 0.2;
const DEGREES: Array<0 | 1 | 2 | 3> = [0, 1, 2, 3];
const DEGREE_LABEL: Record<number, string> = {
  0: 'p = 0  (NW)',
  1: 'p = 1  (local-linear)',
  2: 'p = 2  (local-quadratic)',
  3: 'p = 3  (local-cubic)',
};

export default function DegreeLadderExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [hDisplay, setHDisplay] = useState(0.08);
  const [hCommitted, setHCommitted] = useState(0.08);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // Sample once: §1 toy at seed 42.
  const { X, Y, xGrid, mTrue } = useMemo(() => {
    const rng = mulberry32(42);
    const sample = sampleToyUni(N_SAMPLE, SIGMA, rng);
    const grid = new Float64Array(101);
    for (let g = 0; g < 101; g++) grid[g] = g / 100;
    const truth = new Float64Array(101);
    for (let g = 0; g < 101; g++) truth[g] = mTrueUni(grid[g]);
    return { X: sample.X, Y: sample.Y, xGrid: grid, mTrue: truth };
  }, []);

  // Recompute the four fits whenever the committed bandwidth changes.
  const fits = useMemo(() => {
    return DEGREES.map((p) => localPolynomial(X, Y, xGrid, hCommitted, p, kGaussian));
  }, [X, Y, xGrid, hCommitted]);

  const panelW = containerWidth > 0 ? (isMobile ? containerWidth : containerWidth / 2 - 8) : 0;

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <label style={{ fontFamily: 'var(--font-sans)', fontSize: 14, color: 'var(--color-text)' }}>
          bandwidth h = <strong>{hDisplay.toFixed(3)}</strong>
        </label>
        <input
          type="range"
          min={0.02}
          max={0.30}
          step={0.005}
          value={hDisplay}
          onChange={(e) => setHDisplay(parseFloat(e.target.value))}
          onMouseUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => setHCommitted(parseFloat((e.target as HTMLInputElement).value))}
          aria-label="bandwidth h"
          style={{ flex: 1, minWidth: 200 }}
        />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: isMobile ? 0 : 16 }}>
        {DEGREES.map((p, i) => (
          <Panel key={p} p={p} fit={fits[i]} X={X} Y={Y} xGrid={xGrid} mTrue={mTrue} width={panelW} />
        ))}
      </div>
    </div>
  );
}

function Panel({
  p,
  fit,
  X,
  Y,
  xGrid,
  mTrue,
  width,
}: {
  p: 0 | 1 | 2 | 3;
  fit: Float64Array;
  X: Float64Array;
  Y: Float64Array;
  xGrid: Float64Array;
  mTrue: Float64Array;
  width: number;
}) {
  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const margin = { top: 24, right: 8, bottom: 26, left: 32 };
      const innerW = width - margin.left - margin.right;
      const innerH = HEIGHT_PANEL - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-1.4, 1.6]).nice().range([innerH, 0]);

      // Axes.
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '11px');
      g.append('g').call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text').style('fill', 'var(--color-text-secondary)').style('font-size', '11px');

      // Data scatter.
      g.append('g')
        .selectAll('circle')
        .data(Array.from(X).map((xi, i) => ({ x: xi, y: Y[i] })))
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', 1.6)
        .style('fill', paletteKR.data)
        .style('opacity', 0.5);

      // Truth.
      const lineGen = d3.line<number>().x((_, i) => xScale(xGrid[i])).y((d) => yScale(d));
      g.append('path')
        .datum(Array.from(mTrue))
        .attr('d', lineGen)
        .style('fill', 'none')
        .style('stroke', paletteKR.truth)
        .style('stroke-width', 1.4)
        .style('stroke-dasharray', '4 3')
        .style('opacity', 0.85);

      // Fit.
      g.append('path')
        .datum(Array.from(fit))
        .attr('d', lineGen)
        .style('fill', 'none')
        .style('stroke', paletteKR.posterior)
        .style('stroke-width', 2);

      // Title (degree label).
      g.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(DEGREE_LABEL[p]);
    },
    [width, fit, X, Y, xGrid, mTrue, p],
  );

  return (
    <div style={{ width: width > 0 ? width : '100%' }}>
      <svg ref={renderRef} width={width} height={HEIGHT_PANEL} />
    </div>
  );
}

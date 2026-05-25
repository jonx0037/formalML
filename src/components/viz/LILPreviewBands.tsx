import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, gaussianFrom, zHalf } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 1.2 — LIL preview bands.
//
// Same 200 trajectories from Viz 1.1, overlaid with two bands:
//   (a) shrinking ±z_{δ/2}/√n fixed-n Gaussian band — trajectories exit it
//   (b) ±σ√(2 log log n / n) LIL band — envelopes the fluctuations as n grows
//
// One interactive slider: δ.
// -----------------------------------------------------------------------------

const MARGIN = { top: 30, right: 20, bottom: 42, left: 56 };
const PANEL_H = 380;
const N_PATHS = 200;
const N_MAX = 10000;

const COLOR_PATH = '#9aa3ad';
const COLOR_FIXED = 'var(--color-text-secondary)';
const COLOR_LIL = '#b45309';

function generatePaths(): Float64Array[] {
  const rng = mulberry32(20260525);
  const gauss = gaussianFrom(rng);
  const paths: Float64Array[] = [];
  for (let b = 0; b < N_PATHS; b++) {
    const p = new Float64Array(N_MAX);
    let s = 0;
    for (let n = 1; n <= N_MAX; n++) {
      s += gauss();
      p[n - 1] = s / n;
    }
    paths.push(p);
  }
  return paths;
}

function buildBandData(delta: number) {
  const z = zHalf(delta);
  const out: { n: number; fixed: number; lil: number }[] = [];
  for (let n = 10; n <= N_MAX; n = Math.max(n + 1, Math.floor(n * 1.04))) {
    const fixed = z / Math.sqrt(n);
    // LIL band — only defined for n > e (so log log n is real and positive)
    const lglg = n > 3 ? Math.log(Math.log(n)) : 0;
    const lil = lglg > 0 ? Math.sqrt((2 * lglg) / n) : 0;
    out.push({ n, fixed, lil });
  }
  return out;
}

export default function LILPreviewBands() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [delta, setDelta] = useState(0.05);

  const paths = useMemo(() => generatePaths(), []);
  const bandData = useMemo(() => buildBandData(delta), [delta]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const z = zHalf(delta);
      const yMax = Math.max(0.7, z / Math.sqrt(10));
      const xScale = d3.scaleLog().domain([10, N_MAX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([innerH, 0]);

      // axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)');

      // zero line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-width', 0.8)
        .attr('opacity', 0.4);

      // trajectories
      const pathGen = d3
        .line<number>()
        .x((_, i) => xScale(i + 1))
        .y((v) => yScale(v))
        .defined((_, i) => i + 1 >= 10);
      for (const p of paths) {
        g.append('path')
          .datum(Array.from(p) as number[])
          .attr('d', pathGen as never)
          .style('fill', 'none')
          .style('stroke', COLOR_PATH)
          .attr('stroke-width', 0.4)
          .attr('opacity', 0.4);
      }

      // fixed-n band — upper and lower
      const fixedLine = d3
        .line<{ n: number; fixed: number }>()
        .x((d) => xScale(d.n))
        .y((d) => yScale(d.fixed));
      const fixedLineNeg = d3
        .line<{ n: number; fixed: number }>()
        .x((d) => xScale(d.n))
        .y((d) => yScale(-d.fixed));
      g.append('path')
        .datum(bandData)
        .attr('d', fixedLine as never)
        .style('fill', 'none')
        .style('stroke', COLOR_FIXED)
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 3');
      g.append('path')
        .datum(bandData)
        .attr('d', fixedLineNeg as never)
        .style('fill', 'none')
        .style('stroke', COLOR_FIXED)
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 3');

      // LIL band
      const lilLine = d3
        .line<{ n: number; lil: number }>()
        .x((d) => xScale(d.n))
        .y((d) => yScale(d.lil))
        .defined((d) => d.lil > 0);
      const lilLineNeg = d3
        .line<{ n: number; lil: number }>()
        .x((d) => xScale(d.n))
        .y((d) => yScale(-d.lil))
        .defined((d) => d.lil > 0);
      g.append('path')
        .datum(bandData)
        .attr('d', lilLine as never)
        .style('fill', 'none')
        .style('stroke', COLOR_LIL)
        .attr('stroke-width', 1.6);
      g.append('path')
        .datum(bandData)
        .attr('d', lilLineNeg as never)
        .style('fill', 'none')
        .style('stroke', COLOR_LIL)
        .attr('stroke-width', 1.6);

      // legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 4)`);
      legend
        .append('line')
        .attr('x1', 0)
        .attr('x2', 22)
        .attr('y1', 6)
        .attr('y2', 6)
        .style('stroke', COLOR_FIXED)
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 3');
      legend
        .append('text')
        .attr('x', 28)
        .attr('y', 10)
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('fill', 'var(--color-text)')
        .text(`±z_{δ/2}/√n (δ=${delta.toFixed(2)})`);
      legend
        .append('line')
        .attr('x1', 0)
        .attr('x2', 22)
        .attr('y1', 26)
        .attr('y2', 26)
        .style('stroke', COLOR_LIL)
        .attr('stroke-width', 1.6);
      legend
        .append('text')
        .attr('x', 28)
        .attr('y', 30)
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('fill', 'var(--color-text)')
        .text('±σ√(2 log log n / n)  — LIL');

      // axis label
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 32)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '11px')
        .text('n (log scale)');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '11px')
        .text('running mean X̄_n');
    },
    [paths, bandData, delta, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div
        style={{
          padding: '10px 0',
          fontSize: '13px',
          display: 'flex',
          gap: '14px',
          alignItems: 'center',
          flexWrap: 'wrap',
        }}
      >
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '220px' }}>
          <span>
            Nominal δ = <strong>{delta.toFixed(3)}</strong> (fixed-n band only)
          </span>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.005}
            value={delta}
            aria-label="Nominal false-rejection level"
            onChange={(e) => setDelta(Number(e.target.value))}
          />
        </label>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            color: 'var(--color-text-secondary)',
          }}
        >
          200 trajectories, n ∈ [10, 10000], log-x.
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

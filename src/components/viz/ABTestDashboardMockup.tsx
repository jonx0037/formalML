import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, gaussianFrom, gaussianMixtureHW, zHalf } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 11.1 — Stylized A/B-test dashboard mockup.
//
// Central panel: running lift estimate with anytime-valid CS (PrPl-EB-like
// shading) over time, plus fixed-n band overlay for contrast.
// Right column: current n, decision indicator, look counter.
// -----------------------------------------------------------------------------

const MARGIN = { top: 28, right: 18, bottom: 38, left: 56 };
const PANEL_H = 360;
const N_MAX = 8000;
const TRUE_LIFT = 0.01;
const SIGMA = 0.1;

const COLOR_AVI = 'var(--color-accent)';
const COLOR_FIXED = '#c44545';

function simulate(seed: number, lookEvery: number): {
  ng: number[];
  means: number[];
  aviLo: number[];
  aviHi: number[];
  fixedLo: number[];
  fixedHi: number[];
  lookPoints: number[];
} {
  const rng = mulberry32(seed);
  const gauss = gaussianFrom(rng);
  const ng: number[] = [];
  const means: number[] = [];
  const aviLo: number[] = [];
  const aviHi: number[] = [];
  const fixedLo: number[] = [];
  const fixedHi: number[] = [];
  const lookPoints: number[] = [];

  const delta = 0.05;
  const z = zHalf(delta);
  let cumsum = 0;
  for (let n = 1; n <= N_MAX; n++) {
    const x = TRUE_LIFT + SIGMA * gauss();
    cumsum += x;
    const mean = cumsum / n;
    ng.push(n);
    means.push(mean);
    const hwAVI = gaussianMixtureHW(n, 1, delta, SIGMA);
    const hwFixed = (z * SIGMA) / Math.sqrt(n);
    aviLo.push(mean - hwAVI);
    aviHi.push(mean + hwAVI);
    fixedLo.push(mean - hwFixed);
    fixedHi.push(mean + hwFixed);
    if (n % lookEvery === 0) lookPoints.push(n);
  }
  return { ng, means, aviLo, aviHi, fixedLo, fixedHi, lookPoints };
}

export default function ABTestDashboardMockup() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [lookCadence, setLookCadence] = useState(500);
  const sim = useMemo(() => simulate(20261701, lookCadence), [lookCadence]);
  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  // Decision logic: stop-and-ship when AVI CS excludes 0 favorably
  const decisionPoint = sim.aviLo.findIndex((v) => v > 0);
  const currentN = decisionPoint === -1 ? N_MAX : decisionPoint + 1;
  const status: 'continue' | 'ship' = decisionPoint === -1 ? 'continue' : 'ship';
  const lookCount = sim.lookPoints.filter((p) => p <= currentN).length;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([100, N_MAX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-0.04, 0.06]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '.2%')).selectAll('text').style('fill', 'var(--color-text)');

      // zero ref
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(0)).attr('y2', yScale(0)).style('stroke', 'var(--color-text-secondary)').attr('stroke-width', 1).attr('stroke-dasharray', '4 3');

      // AVI band
      const areaAVI = d3.area<number>().x((_, i) => xScale(sim.ng[i])).y0((_, i) => yScale(sim.aviLo[i])).y1((_, i) => yScale(sim.aviHi[i]));
      g.append('path').datum(sim.means).attr('d', areaAVI as never).style('fill', COLOR_AVI).style('opacity', 0.18);

      // Fixed-n band (dashed lines)
      const lineFixedLo = d3.line<number>().x((_, i) => xScale(sim.ng[i])).y((_, i) => yScale(sim.fixedLo[i]));
      const lineFixedHi = d3.line<number>().x((_, i) => xScale(sim.ng[i])).y((_, i) => yScale(sim.fixedHi[i]));
      g.append('path').datum(sim.means).attr('d', lineFixedLo as never).style('fill', 'none').style('stroke', COLOR_FIXED).attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3');
      g.append('path').datum(sim.means).attr('d', lineFixedHi as never).style('fill', 'none').style('stroke', COLOR_FIXED).attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3');

      // mean
      const lineMean = d3.line<number>().x((_, i) => xScale(sim.ng[i])).y((v) => yScale(v));
      g.append('path').datum(sim.means).attr('d', lineMean as never).style('fill', 'none').style('stroke', 'var(--color-text)').attr('stroke-width', 1.5);

      // look markers
      sim.lookPoints.forEach((n) => {
        if (n <= currentN) {
          g.append('circle').attr('cx', xScale(n)).attr('cy', yScale(sim.means[n - 1])).attr('r', 2).style('fill', 'var(--color-text)').attr('opacity', 0.6);
        }
      });

      // decision marker
      if (status === 'ship') {
        g.append('line').attr('x1', xScale(currentN)).attr('x2', xScale(currentN)).attr('y1', 0).attr('y2', innerH).style('stroke', COLOR_AVI).attr('stroke-width', 2);
        g.append('text').attr('x', xScale(currentN) + 6).attr('y', 14).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_AVI).text(`SHIP at n=${currentN}`);
      }

      g.append('text').attr('x', innerW - 4).attr('y', 14).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_AVI).text('PrPl-EB-like CS (anytime-valid)');
      g.append('text').attr('x', innerW - 4).attr('y', 28).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_FIXED).text('fixed-n band (DASHED — NOT valid under peeking)');

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`users (n) — true lift = ${(TRUE_LIFT * 100).toFixed(1)}%`);
    },
    [sim, w, currentN, status],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <span>Look cadence (every N users) = <strong>{lookCadence}</strong></span>
          <input type="range" min={100} max={2000} step={100} value={lookCadence} aria-label="Look cadence" onChange={(e) => setLookCadence(Number(e.target.value))} />
        </label>
        <div style={{ display: 'flex', gap: '8px', flex: 1, flexWrap: 'wrap' }}>
          <div style={{ padding: '8px 14px', border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-surface)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
            current n = <strong>{currentN.toLocaleString()}</strong>
          </div>
          <div style={{ padding: '8px 14px', border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-surface)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
            decision: <strong style={{ color: status === 'ship' ? COLOR_AVI : 'var(--color-text-secondary)' }}>{status === 'ship' ? 'SHIP' : 'continue'}</strong>
          </div>
          <div style={{ padding: '8px 14px', border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-surface)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
            looks taken: <strong>{lookCount}</strong>
          </div>
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}

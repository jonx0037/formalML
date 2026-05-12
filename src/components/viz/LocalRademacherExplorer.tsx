import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  localRademacherFixedPoint,
  localRademacherThresholdClass,
  mulberry32,
} from './shared/generalization-bounds';

// =============================================================================
// LocalRademacherExplorer — §9.
//
//   Left  — R̂_S(F_r) as a function of r at fixed n, with the fixed-point
//           line  r = R̂_S(F_r) + log(1/δ)/n  overlaid; intersection is r*.
//   Right — r* vs. n on log-log axes, with the 1/√n and 1/n envelopes.
//
// Commit-on-release.
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 1000;
const N_GRID = [60, 100, 200, 500, 1000];

export default function LocalRademacherExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayN, setDisplayN] = useState(200);
  const [committedN, setCommittedN] = useState(200);
  const [delta, setDelta] = useState(0.05);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor(containerWidth / 2);

  const leftData = useMemo(() => {
    const rng = mulberry32(20260511);
    const rGrid = d3.range(0.005, 0.50, 0.02);
    const radCurve = rGrid.map((r) => ({
      r,
      rad: localRademacherThresholdClass(committedN, r, 80, rng),
    }));
    const logTerm = Math.log(1 / delta) / committedN;
    const fpRng = mulberry32(20260512);
    const rStar = localRademacherFixedPoint(committedN, delta, fpRng, 80, 14);
    return { rGrid, radCurve, logTerm, rStar };
  }, [committedN, delta]);

  const rightData = useMemo(() => {
    return N_GRID.map((n) => {
      const rng = mulberry32(20260513 + n);
      const r = localRademacherFixedPoint(n, delta, rng, 70, 14);
      return { n, rStar: r, slow: 1 / Math.sqrt(n), fast: 1 / n };
    });
  }, [delta]);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 32, right: 12, bottom: 40, left: 50 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([0, 0.5]).range([0, w]);
      const yMax = Math.max(0.4, d3.max(leftData.radCurve, (p) => p.rad) ?? 0.4);
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('r (P_n second-moment radius)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -36).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('R̂_S(F_r)');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Local Rademacher + fixed-point line (intersection = r*)');

      const radLine = d3.line<{ r: number; rad: number }>().x((d) => x(d.r)).y((d) => y(d.rad));
      g.append('path').datum(leftData.radCurve)
        .attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', radLine);

      // Fixed-point line: rhs = r - log(1/δ)/n.  Plot y = r - logTerm vs r.
      const fpPts = leftData.rGrid.map((r) => ({ r, val: Math.max(0, r - leftData.logTerm) }));
      const fpLine = d3.line<{ r: number; val: number }>().x((d) => x(d.r)).y((d) => y(d.val));
      g.append('path').datum(fpPts).attr('fill', 'none').style('stroke', '#ef4444').style('stroke-width', 1.5)
        .style('stroke-dasharray', '4 3').attr('d', fpLine);

      // r* marker
      g.append('line').attr('x1', x(leftData.rStar)).attr('x2', x(leftData.rStar))
        .attr('y1', 0).attr('y2', h).style('stroke', '#16a34a').style('stroke-dasharray', '2 2');
      g.append('text').attr('x', x(leftData.rStar) + 6).attr('y', 16)
        .style('font-size', '10px').style('fill', '#15803d').text(`r* = ${leftData.rStar.toFixed(4)}`);
    },
    [leftData, panelWidth],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 40, left: 50 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([50, 1100]).range([0, w]);
      const y = d3.scaleLog().domain([0.0005, 0.5]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(4, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -36).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('rate / r*');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Fast-rate fixed point vs slow-rate envelope');

      type Pt = { n: number; rStar: number; slow: number; fast: number };
      const rLine = d3.line<Pt>().x((p) => x(p.n)).y((p) => y(Math.max(0.0005, p.rStar)));
      const slow = d3.line<Pt>().x((p) => x(p.n)).y((p) => y(p.slow));
      const fast = d3.line<Pt>().x((p) => x(p.n)).y((p) => y(p.fast));
      g.append('path').datum(rightData).attr('fill', 'none').style('stroke', '#16a34a').style('stroke-width', 2).attr('d', rLine);
      g.append('path').datum(rightData).attr('fill', 'none').style('stroke', '#7c3aed').style('stroke-width', 1.5).style('stroke-dasharray', '4 3').attr('d', slow);
      g.append('path').datum(rightData).attr('fill', 'none').style('stroke', '#ef4444').style('stroke-width', 1.5).style('stroke-dasharray', '4 3').attr('d', fast);

      rightData.forEach((p) => {
        g.append('circle').attr('cx', x(p.n)).attr('cy', y(Math.max(0.0005, p.rStar))).attr('r', 3.5).style('fill', '#16a34a');
      });
      const lg = g.append('g').attr('transform', `translate(${w - 165},${10})`);
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0).style('stroke', '#16a34a').style('stroke-width', 2);
      lg.append('text').attr('x', 22).attr('y', 4).style('font-size', '11px').style('fill', 'var(--color-text)').text('local r*');
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 16).attr('y2', 16).style('stroke', '#7c3aed').style('stroke-dasharray', '4 3');
      lg.append('text').attr('x', 22).attr('y', 20).style('font-size', '11px').style('fill', 'var(--color-text)').text('1/√n (slow)');
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 32).attr('y2', 32).style('stroke', '#ef4444').style('stroke-dasharray', '4 3');
      lg.append('text').attr('x', 22).attr('y', 36).style('font-size', '11px').style('fill', 'var(--color-text)').text('1/n (fast)');
    },
    [rightData, panelWidth],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <div style={{ display: 'grid', gridTemplateColumns: isStacked ? '1fr' : '1fr 1fr', gap: 8 }}>
        <svg ref={leftRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Local Rademacher vs r with fixed-point intersection." />
        <svg ref={rightRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Fixed-point r* vs n with rate envelopes." />
      </div>
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          n (left panel):
          <input
            type="range" min={60} max={1000} step={20} value={displayN}
            onChange={(e) => setDisplayN(parseInt(e.target.value, 10))}
            onMouseUp={() => setCommittedN(displayN)}
            onTouchEnd={() => setCommittedN(displayN)}
            onKeyUp={() => setCommittedN(displayN)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }} aria-label="sample size n"
          />
          <span style={{ marginLeft: 6 }}>{displayN}</span>
        </label>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          confidence δ:
          <input
            type="range" min={0.005} max={0.3} step={0.005} value={delta}
            onChange={(e) => setDelta(parseFloat(e.target.value))}
            style={{ marginLeft: 8, verticalAlign: 'middle' }} aria-label="confidence delta"
          />
          <span style={{ marginLeft: 6 }}>{delta.toFixed(3)}</span>
        </label>
      </figcaption>
    </figure>
  );
}

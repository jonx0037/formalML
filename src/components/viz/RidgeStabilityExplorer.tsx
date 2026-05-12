import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bousquetElisseeffDeviation,
  gaussianFrom,
  mulberry32,
  ridgeStabilityBeta,
} from './shared/generalization-bounds';

// =============================================================================
// RidgeStabilityExplorer — §11.  Ridge β-stability and Bousquet–Elisseeff bound.
//
//   Left  — β(λ) on log-log axes with the theoretical envelope C/(λn).
//   Right — BE deviation bound vs λ.
//
// (Named `RidgeStabilityExplorer` to disambiguate from the statistical-TDA
// topic's StabilityExplorer.tsx.)
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 1000;
// 7 logarithmically-spaced λ values across 3.5 decades — enough to render the
// β-vs-λ curve smoothly without paying the cost of the original 11-point grid.
const LAMBDA_GRID = [0.005, 0.025, 0.1, 0.25, 1.0, 2.5, 10.0];
const N_SWAPS = 4;  // per-lambda; the max-diff is dominated by a few large swaps anyway

export default function RidgeStabilityExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayN, setDisplayN] = useState(80);
  const [committedN, setCommittedN] = useState(80);
  const [displayD, setDisplayD] = useState(30);
  const [committedD, setCommittedD] = useState(30);
  const [delta, setDelta] = useState(0.05);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor(containerWidth / 2);

  const data = useMemo(() => {
    return LAMBDA_GRID.map((lam) => {
      const unif = mulberry32(20260511);
      const gauss = gaussianFrom(mulberry32(20260512));
      const beta = ridgeStabilityBeta(committedN, committedD, lam, N_SWAPS, unif, gauss);
      const dev = bousquetElisseeffDeviation(beta, committedN, delta);
      return { lam, beta, dev, theoretical: 4.0 / (lam * committedN) };
    });
  }, [committedN, committedD, delta]);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 40, left: 56 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([0.003, 12]).range([0, w]);
      const yMax = Math.max((d3.max(data, (d) => Math.max(d.beta, d.theoretical)) ?? 1) * 1.2, 0.01);
      const yMin = Math.max(1e-4, (d3.min(data, (d) => Math.min(d.beta, d.theoretical)) ?? 1e-3) * 0.5);
      const y = d3.scaleLog().domain([yMin, yMax]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5, '~g'));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('regularization λ');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('β(λ) stability');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Ridge β-stability vs theoretical O(1/(λn))');

      type Pt = { lam: number; beta: number; dev: number; theoretical: number };
      const empLine = d3.line<Pt>().x((p) => x(p.lam)).y((p) => y(Math.max(yMin, p.beta)));
      const thLine = d3.line<Pt>().x((p) => x(p.lam)).y((p) => y(Math.max(yMin, p.theoretical)));
      g.append('path').datum(data).attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', empLine);
      g.append('path').datum(data).attr('fill', 'none').style('stroke', '#ef4444').style('stroke-width', 1.5).style('stroke-dasharray', '4 3').attr('d', thLine);
      data.forEach((p) => g.append('circle').attr('cx', x(p.lam)).attr('cy', y(Math.max(yMin, p.beta))).attr('r', 3.5).style('fill', '#3b82f6'));
    },
    [data, panelWidth],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 40, left: 56 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([0.003, 12]).range([0, w]);
      const y = d3.scaleLog().domain([0.1, Math.max(10, (d3.max(data, (p) => p.dev) ?? 1) * 1.2)]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5, '~g'));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('regularization λ');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('BE deviation');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Bousquet–Elisseeff deviation β + (2nβ+M)√(log(1/δ)/(2n))');

      type Pt = { lam: number; dev: number };
      const line = d3.line<Pt>().x((p) => x(p.lam)).y((p) => y(Math.max(0.1, p.dev)));
      g.append('path').datum(data).attr('fill', 'none').style('stroke', '#10b981').style('stroke-width', 2).attr('d', line);
      data.forEach((p) => g.append('circle').attr('cx', x(p.lam)).attr('cy', y(Math.max(0.1, p.dev))).attr('r', 3.5).style('fill', '#10b981'));
    },
    [data, panelWidth],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <div style={{ display: 'grid', gridTemplateColumns: isStacked ? '1fr' : '1fr 1fr', gap: 8 }}>
        <svg ref={leftRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Ridge β-stability vs lambda." />
        <svg ref={rightRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Bousquet-Elisseeff deviation bound vs lambda." />
      </div>
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          n:
          <input
            type="range" min={50} max={200} step={10} value={displayN}
            onChange={(e) => setDisplayN(parseInt(e.target.value, 10))}
            onMouseUp={() => setCommittedN(displayN)}
            onTouchEnd={() => setCommittedN(displayN)}
            onKeyUp={() => setCommittedN(displayN)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }} aria-label="sample size n"
          />
          <span style={{ marginLeft: 6 }}>{displayN}</span>
        </label>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          d:
          <input
            type="range" min={10} max={100} step={5} value={displayD}
            onChange={(e) => setDisplayD(parseInt(e.target.value, 10))}
            onMouseUp={() => setCommittedD(displayD)}
            onTouchEnd={() => setCommittedD(displayD)}
            onKeyUp={() => setCommittedD(displayD)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }} aria-label="dimension d"
          />
          <span style={{ marginLeft: 6 }}>{displayD}</span>
        </label>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          δ:
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

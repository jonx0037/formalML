import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  mulberry32,
  robinsonDGP,
  workedExamplePipeline,
} from './shared/causal-inference-methods';

// =============================================================================
// CausalEstimatorForestPlot — §13 worked example, two panels combined.
//
//  (A) Single-sample forest plot at n = 1500 for IPW, OR, AIPW, DML-lasso,
//      DML-RF-proxy. Each row shows point estimate + 95% Wald CI.
//  (B) Monte Carlo distributions across B replicates (slider).
// =============================================================================

const HEIGHT_FOREST = 240;
const HEIGHT_MC = 320;
const N = 1500;

const NAMES = ['IPW (Hájek)', 'OR (g-comp)', 'AIPW', 'DML-lasso', 'DML-RF-proxy'] as const;
const COLORS = ['#c0504d', '#7b3c10', '#3a6e3a', '#1f4e79', '#2e7baa'];

export default function CausalEstimatorForestPlot() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [bDisplay, setBDisplay] = useState(40);
  const [B, setB] = useState(40);

  const single = useMemo(() => {
    const rng = mulberry32(20260512 + 137);
    const s = robinsonDGP(N, 10, 1.0, rng);
    const r = workedExamplePipeline(s, mulberry32(20260512 + 139));
    return [r.ipw, r.or, r.aipw, r.dmlLasso, r.dmlRF];
  }, []);

  const mc = useMemo(() => {
    const out: Float64Array[] = NAMES.map(() => new Float64Array(B));
    for (let b = 0; b < B; b++) {
      const rng = mulberry32(20260512 + 200 + b * 61);
      const s = robinsonDGP(800, 10, 1.0, rng);
      const r = workedExamplePipeline(s, mulberry32(20260512 + 300 + b * 67));
      out[0][b] = r.ipw.tau;
      out[1][b] = r.or.tau;
      out[2][b] = r.aipw.tau;
      out[3][b] = r.dmlLasso.tau;
      out[4][b] = r.dmlRF.tau;
    }
    return out;
  }, [B]);

  const forestRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 20, right: 24, bottom: 40, left: 110 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT_FOREST - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT_FOREST}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLinear().domain([0.5, 1.5]).range([0, W]);
      const yScale = d3.scaleBand().domain(NAMES.slice()).range([0, H]).padding(0.2);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale));
      g.append('line').attr('x1', xScale(1)).attr('x2', xScale(1)).attr('y1', 0).attr('y2', H)
        .style('stroke', '#3a6e3a').style('stroke-width', 1.5);
      single.forEach((r, i) => {
        const y = yScale(NAMES[i])! + yScale.bandwidth() / 2;
        g.append('line').attr('x1', xScale(r.ciLow)).attr('x2', xScale(r.ciHigh))
          .attr('y1', y).attr('y2', y)
          .style('stroke', COLORS[i]).style('stroke-width', 2);
        g.append('circle').attr('cx', xScale(r.tau)).attr('cy', y).attr('r', 5)
          .style('fill', COLORS[i]);
        g.append('text').attr('x', xScale(r.ciHigh) + 6).attr('y', y + 4)
          .style('fill', 'var(--color-text)').style('font-size', '10px')
          .text(`${r.tau.toFixed(3)} ± ${(1.96 * r.se).toFixed(3)}`);
      });
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('τ̂ with 95% Wald CI');
      g.append('text').attr('x', 4).attr('y', 12).style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text(`(A) Single sample, n = ${N}`);
    },
    [containerWidth, single],
  );

  const mcRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 20, right: 24, bottom: 40, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT_MC - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT_MC}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLinear().domain([0.5, 1.5]).range([0, W]);
      const rowH = H / NAMES.length;
      const yBase = d3.scaleBand().domain(NAMES.slice()).range([0, H]).padding(0.15);
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('line').attr('x1', xScale(1)).attr('x2', xScale(1)).attr('y1', 0).attr('y2', H)
        .style('stroke', '#3a6e3a').style('stroke-width', 1.5);
      mc.forEach((arr, i) => {
        const y0 = yBase(NAMES[i])!;
        const bins = d3.bin().domain([0.5, 1.5]).thresholds(28)(Array.from(arr));
        const maxC = d3.max(bins, (d) => d.length)! || 1;
        bins.forEach((d) => {
          const h = (d.length / maxC) * (yBase.bandwidth());
          g.append('rect').attr('x', xScale(d.x0!)).attr('y', y0 + yBase.bandwidth() - h)
            .attr('width', Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0)).attr('height', h)
            .style('fill', COLORS[i]).style('opacity', 0.6);
        });
        const m = d3.mean(arr)!;
        g.append('line').attr('x1', xScale(m)).attr('x2', xScale(m))
          .attr('y1', y0).attr('y2', y0 + yBase.bandwidth())
          .style('stroke', COLORS[i]).style('stroke-width', 1.5).style('stroke-dasharray', '3 3');
        g.append('text').attr('x', 4).attr('y', y0 + 14)
          .style('fill', COLORS[i]).style('font-size', '11px').text(`${NAMES[i]}: mean = ${m.toFixed(3)}`);
      });
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text(`τ̂ distribution across B = ${B} replicates (n = 800)`);
    },
    [containerWidth, mc, B],
  );

  const commitB = () => setB(bDisplay);

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <svg ref={forestRef} style={{ width: '100%', height: HEIGHT_FOREST }} />
      <div style={{ marginTop: '0.5rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          B replicates =
          <input type="range" min={20} max={80} step={10} value={bDisplay}
            onChange={(e) => setBDisplay(parseInt(e.target.value))}
            onMouseUp={commitB} onTouchEnd={commitB} onKeyUp={commitB}
            aria-label="Number of MC replicates" style={{ width: 140 }} />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{bDisplay}</span>
        </label>
        <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>(B) MC distribution across B replicates</span>
      </div>
      <svg ref={mcRef} style={{ width: '100%', height: HEIGHT_MC }} />
    </div>
  );
}

import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  correctPropensityFeatures,
  ipwEstimate,
  logisticFitPredict,
  mulberry32,
  robinsonDGP,
} from './shared/causal-inference-methods';

// =============================================================================
// IPWEstimatorExplorer — §4 (IPW with sandwich SE).
//
// Three controls:
//   · n (200, 500, 1000, 2000)
//   · IPW form: HT vs Hájek
//   · Trim threshold δ ∈ [0, 0.2]
//
// One panel: a histogram of B = 200 Monte Carlo IPW point estimates with the
// true τ = 1 marker overlaid and an empirical 95% Wald CI band shaded from
// the mean ± 1.96 · mean(SE).
//
// Static fallback: public/images/topics/causal-inference-methods/04_ipw_variance_inflation.png
// =============================================================================

const HEIGHT = 340;
const B_REP = 200;
const SM_BREAKPOINT = 640;

export default function IPWEstimatorExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState<200 | 500 | 1000 | 2000>(500);
  const [form, setForm] = useState<'ht' | 'hajek'>('hajek');
  const [trim, setTrim] = useState(0);
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { taus, meanTau, sdTau, meanSE, coverage, maxWeight } = useMemo(() => {
    const taus = new Float64Array(B_REP);
    const ses = new Float64Array(B_REP);
    const maxWs = new Float64Array(B_REP);
    let covered = 0;
    for (let b = 0; b < B_REP; b++) {
      const rng = mulberry32(20260512 + b * 7 + n);
      const sample = robinsonDGP(n, 10, 1.0, rng);
      const propFeat = correctPropensityFeatures(sample.X, n, sample.p);
      const { ePred } = logisticFitPredict(propFeat, sample.D, n, 3);
      const res = ipwEstimate(sample.D, sample.Y, ePred, { form, trim });
      taus[b] = res.tau;
      ses[b] = res.se;
      maxWs[b] = res.maxWeight;
      if (res.ciLow <= 1 && res.ciHigh >= 1) covered++;
    }
    let s = 0;
    for (let b = 0; b < B_REP; b++) s += taus[b];
    const mt = s / B_REP;
    let v = 0, mse = 0;
    for (let b = 0; b < B_REP; b++) {
      v += (taus[b] - mt) ** 2;
      mse += ses[b];
    }
    return {
      taus,
      meanTau: mt,
      sdTau: Math.sqrt(v / (B_REP - 1)),
      meanSE: mse / B_REP,
      coverage: covered / B_REP,
      maxWeight: d3.max(maxWs)!,
    };
  }, [n, form, trim]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 24, bottom: 40, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const tauArr = Array.from(taus);
      const lo = Math.min(0.5, d3.min(tauArr)!);
      const hi = Math.max(1.5, d3.max(tauArr)!);
      const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
      // Histogram via d3.bin.
      const binGen = d3.bin().domain(xScale.domain() as [number, number]).thresholds(30);
      const bins = binGen(tauArr);
      const yScale = d3.scaleLinear().domain([0, d3.max(bins, (d) => d.length)! * 1.1]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(4));
      // CI band.
      g.append('rect')
        .attr('x', xScale(meanTau - 1.96 * meanSE)).attr('y', 0)
        .attr('width', xScale(meanTau + 1.96 * meanSE) - xScale(meanTau - 1.96 * meanSE))
        .attr('height', H)
        .style('fill', '#1f4e79').style('opacity', 0.10);
      // Bars.
      g.selectAll('rect.bar')
        .data(bins)
        .join('rect')
        .attr('class', 'bar')
        .attr('x', (d) => xScale(d.x0!) + 1)
        .attr('y', (d) => yScale(d.length))
        .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
        .attr('height', (d) => H - yScale(d.length))
        .style('fill', '#1f4e79').style('opacity', 0.7);
      // True τ = 1.
      g.append('line')
        .attr('x1', xScale(1)).attr('x2', xScale(1))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', '#3a6e3a').style('stroke-width', 2);
      // Mean.
      g.append('line')
        .attr('x1', xScale(meanTau)).attr('x2', xScale(meanTau))
        .attr('y1', 0).attr('y2', H)
        .style('stroke', '#c0504d').style('stroke-width', 1.5)
        .style('stroke-dasharray', '4 4');
      // Labels.
      g.append('text').attr('x', W / 2).attr('y', H + 32).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('IPW point estimate τ̂');
      g.append('text').attr('transform', `translate(-40, ${H / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px')
        .text('Replicate count');
    },
    [containerWidth, taus, meanTau, meanSE],
  );

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '1.5rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          n =
          <select value={n} onChange={(e) => setN(parseInt(e.target.value) as 200 | 500 | 1000 | 2000)}
            style={{ padding: '0.2rem 0.4rem', fontSize: '0.9rem' }}>
            <option value={200}>200</option>
            <option value={500}>500</option>
            <option value={1000}>1000</option>
            <option value={2000}>2000</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          Form =
          <select value={form} onChange={(e) => setForm(e.target.value as 'ht' | 'hajek')}
            style={{ padding: '0.2rem 0.4rem', fontSize: '0.9rem' }}>
            <option value="ht">Horvitz–Thompson</option>
            <option value="hajek">Hájek (stabilized)</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          Trim δ =
          <input type="range" min={0} max={0.2} step={0.01} value={trim}
            onChange={(e) => setTrim(parseFloat(e.target.value))}
            aria-label="Trim threshold" style={{ width: 140 }} />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{trim.toFixed(2)}</span>
        </label>
      </div>
      <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '0.4rem', flexWrap: 'wrap' }}>
        <span>mean τ̂ = <span style={{ fontFamily: 'var(--font-mono)' }}>{meanTau.toFixed(3)}</span></span>
        <span>SD = <span style={{ fontFamily: 'var(--font-mono)' }}>{sdTau.toFixed(3)}</span></span>
        <span>mean SE = <span style={{ fontFamily: 'var(--font-mono)' }}>{meanSE.toFixed(3)}</span></span>
        <span>95% coverage = <span style={{ fontFamily: 'var(--font-mono)' }}>{(coverage * 100).toFixed(1)}%</span></span>
        <span>max IPW weight = <span style={{ fontFamily: 'var(--font-mono)' }}>{maxWeight.toFixed(1)}</span></span>
      </div>
      <svg ref={svgRef} style={{ width: '100%', height: HEIGHT }} />
    </div>
  );
}

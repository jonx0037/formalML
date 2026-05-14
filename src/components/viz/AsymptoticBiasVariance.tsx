import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  excessRiskSweep,
  gaussianRng,
  hastieRiskDecomp,
  mulberry32,
  sampleSphereBetaStar,
} from './shared/double-descent';

// =============================================================================
// AsymptoticBiasVariance — §6.4 (Figures 6 and 7)
//
// Analytic Hastie 2022 curve R(γ) with optional empirical MC overlay and
// optional bias²/variance decomposition. x-axis is γ = p/n.
// Static fallback: 06_hastie_analytic_vs_mc.png, 07_hastie_decomposition.png
// =============================================================================

const HEIGHT = 460;
const GAMMA_GRID = (() => {
  const out: number[] = [];
  for (let g = 0.05; g <= 4.0; g += 0.05) out.push(g);
  return out;
})();

export default function AsymptoticBiasVariance() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [snrLog, setSnrLog] = useState(0);
  const [snrLogCommitted, setSnrLogCommitted] = useState(0);
  const [showMC, setShowMC] = useState(true);
  const [showDecomp, setShowDecomp] = useState(false);
  const [nMC, setNMC] = useState(50);
  const [nMCCommitted, setNMCCommitted] = useState(50);

  const analytic = useMemo(() => {
    const snr = Math.pow(10, snrLog);
    const sigma2 = 1;
    const r2 = snr * sigma2;
    return GAMMA_GRID.map((g) => {
      const d = hastieRiskDecomp(g, r2, sigma2);
      return { gamma: g, bias2: d.bias2, variance: d.variance, total: d.total };
    });
  }, [snrLog]);

  const mcCurve = useMemo(() => {
    if (!showMC) return null;
    const snr = Math.pow(10, snrLogCommitted);
    const sigma = 1;
    const r = Math.sqrt(snr);
    const n = nMCCommitted;
    // For each γ pick p = round(γ · n) and sweep over distinct p values.
    const pSet = new Set<number>();
    for (const g of GAMMA_GRID) pSet.add(Math.max(1, Math.round(g * n)));
    const pArr = Array.from(pSet).sort((a, b) => a - b);
    const pGrid = new Int32Array(pArr);
    const pMax = pArr[pArr.length - 1];
    const rng = mulberry32(2024);
    const gauss = gaussianRng(rng);
    const betaStar = sampleSphereBetaStar(pMax, r, gauss);
    const out = excessRiskSweep({
      n,
      pMax,
      pGrid,
      betaStarFull: betaStar,
      sigma,
      B: 15,
      rng: mulberry32(7919),
    });
    return pArr.map((p, i) => ({ gamma: p / n, mean: out.mean[i] }));
  }, [showMC, snrLogCommitted, nMCCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 32, bottom: 56, left: 64 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const allVals = analytic.map((d) => d.total).filter((v) => Number.isFinite(v) && v > 0);
      const mcVals = (mcCurve ?? []).map((d) => d.mean).filter((v) => v > 0);
      const yMin = Math.max(1e-3, d3.min([...allVals, ...mcVals]) ?? 1e-2);
      const yMax = Math.min(1e4, d3.max([...allVals, ...mcVals]) ?? 100);

      const xScale = d3.scaleLinear().domain([0, 4]).range([0, W]);
      const yScale = d3.scaleLog().domain([yMin * 0.7, yMax * 1.4]).range([H, 0]);

      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xScale).ticks(8))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6, '~g'))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 38).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').text('γ = p / n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -48)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px')
        .text('excess risk (log scale)');

      // Threshold marker
      g.append('line').attr('x1', xScale(1)).attr('x2', xScale(1)).attr('y1', 0).attr('y2', H)
        .style('stroke', 'var(--color-accent)').attr('stroke-dasharray', '4 4').attr('opacity', 0.5);

      const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0]))
        .y((d) => yScale(Math.max(yMin * 0.7, d[1])))
        .defined((d) => Number.isFinite(d[1]) && d[1] > 0);

      // Analytic total (red dashed)
      const totalData: [number, number][] = analytic.map((d) => [d.gamma, d.total]);
      g.append('path').datum(totalData).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#9B1D20').attr('stroke-width', 2.5)
        .attr('stroke-dasharray', showMC ? '5 3' : '');

      if (showDecomp) {
        const biasData: [number, number][] = analytic.map((d) => [d.gamma, d.bias2]);
        const varData: [number, number][] = analytic.map((d) => [d.gamma, d.variance]);
        g.append('path').datum(biasData).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', '#D97706').attr('stroke-width', 1.8).attr('stroke-dasharray', '2 4');
        g.append('path').datum(varData).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', '#534AB7').attr('stroke-width', 1.8).attr('stroke-dasharray', '6 2');
      }

      // Empirical MC overlay
      if (mcCurve) {
        const mcData: [number, number][] = mcCurve.map((d) => [d.gamma, d.mean]);
        g.append('path').datum(mcData).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', 'var(--color-accent)').attr('stroke-width', 1.6).attr('opacity', 0.85);
        g.selectAll('circle.mc').data(mcCurve).enter().append('circle')
          .attr('cx', (d) => xScale(d.gamma)).attr('cy', (d) => yScale(Math.max(yMin * 0.7, d.mean)))
          .attr('r', 2.5).style('fill', 'var(--color-accent)').filter((d) => d.mean > 0);
      }

      // Legend
      let yLeg = 14;
      g.append('line').attr('x1', W - 130).attr('x2', W - 110).attr('y1', yLeg).attr('y2', yLeg)
        .style('stroke', '#9B1D20').attr('stroke-width', 2.5).attr('stroke-dasharray', showMC ? '5 3' : '');
      g.append('text').attr('x', W - 104).attr('y', yLeg + 4)
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('Hastie 2022 (analytic)');
      if (showMC) {
        yLeg += 16;
        g.append('line').attr('x1', W - 130).attr('x2', W - 110).attr('y1', yLeg).attr('y2', yLeg)
          .style('stroke', 'var(--color-accent)').attr('stroke-width', 1.6);
        g.append('text').attr('x', W - 104).attr('y', yLeg + 4)
          .style('fill', 'var(--color-text)').style('font-size', '11px').text(`empirical (n = ${nMCCommitted})`);
      }
      if (showDecomp) {
        yLeg += 16;
        g.append('line').attr('x1', W - 130).attr('x2', W - 110).attr('y1', yLeg).attr('y2', yLeg)
          .style('stroke', '#D97706').attr('stroke-width', 1.8).attr('stroke-dasharray', '2 4');
        g.append('text').attr('x', W - 104).attr('y', yLeg + 4)
          .style('fill', 'var(--color-text)').style('font-size', '11px').text('bias²');
        yLeg += 16;
        g.append('line').attr('x1', W - 130).attr('x2', W - 110).attr('y1', yLeg).attr('y2', yLeg)
          .style('stroke', '#534AB7').attr('stroke-width', 1.8).attr('stroke-dasharray', '6 2');
        g.append('text').attr('x', W - 104).attr('y', yLeg + 4)
          .style('fill', 'var(--color-text)').style('font-size', '11px').text('variance');
      }
    },
    [containerWidth, analytic, mcCurve, showDecomp, showMC, nMCCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Hastie 2022 analytic curve and empirical Monte Carlo at varying γ." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
          <span>SNR: <strong>{Math.pow(10, snrLog).toFixed(2)}</strong></span>
          <input type="range" min={-2} max={2} step={0.05} value={snrLog}
            onChange={(e) => setSnrLog(parseFloat(e.target.value))}
            onMouseUp={() => setSnrLogCommitted(snrLog)} onTouchEnd={() => setSnrLogCommitted(snrLog)}
            onKeyUp={() => setSnrLogCommitted(snrLog)} aria-label="Signal-to-noise ratio" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showMC} onChange={(e) => setShowMC(e.target.checked)} />
          <span>show empirical MC overlay</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showDecomp} onChange={(e) => setShowDecomp(e.target.checked)} />
          <span>show bias²/variance decomposition</span>
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
          <span>n (MC only): <strong>{nMC}</strong></span>
          <input type="range" min={20} max={300} step={10} value={nMC}
            onChange={(e) => setNMC(parseInt(e.target.value, 10))}
            onMouseUp={() => setNMCCommitted(nMC)} onTouchEnd={() => setNMCCommitted(nMC)}
            onKeyUp={() => setNMCCommitted(nMC)} aria-label="Monte Carlo sample size" />
        </label>
      </div>
    </div>
  );
}
